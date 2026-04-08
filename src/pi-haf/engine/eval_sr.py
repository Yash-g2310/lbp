from __future__ import annotations

import gc
import logging
import time
from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from .metrics import SuperResolutionMetrics


@torch.no_grad()
def evaluate_super_resolution(
    model: nn.Module,
    sampler,
    test_loader,
    device: Optional[Union[str, torch.device]] = None,
    flush_every_n_batches: int = 0,
    max_images: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    log_every_n_batches: int = 0,
    label: str = "eval",
) -> Dict[str, float]:
    """Evaluate SR model over a test loader and return dataset-average PSNR/SSIM.

    Contract:
    - lr_imgs are expected in normalized [-1,1] for sampler/model inference.
    - sampler output is expected in [0,1] after post-processing.
    - hr_imgs from loader are converted to [0,1] before metric computation.
    """
    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_obj = torch.device(device)
    if max_images is not None and max_images <= 0:
        raise ValueError(f"max_images must be > 0 when provided, got {max_images}")

    model = model.to(device_obj)
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    start_time = time.perf_counter()

    for batch_idx, (lr_imgs, hr_imgs) in enumerate(test_loader):
        if max_images is not None and total_samples >= max_images:
            break

        if lr_imgs.ndim != 4 or hr_imgs.ndim != 4:
            raise ValueError(
                "test_loader must yield rank-4 tensors [B,C,H,W], "
                f"got lr={tuple(lr_imgs.shape)} hr={tuple(hr_imgs.shape)} at batch {batch_idx}"
            )

        if max_images is not None:
            remaining = max_images - total_samples
            if remaining <= 0:
                break
            if lr_imgs.shape[0] > remaining:
                lr_imgs = lr_imgs[:remaining]
                hr_imgs = hr_imgs[:remaining]

        lr_imgs = lr_imgs.to(device_obj, non_blocking=True)
        hr_imgs = hr_imgs.to(device_obj, non_blocking=True)

        sr_imgs = sampler.sample(lr_imgs)

        if sr_imgs.ndim != 4:
            raise ValueError(f"sampler output must be rank-4 [B,C,H,W], got {tuple(sr_imgs.shape)}")
        if sr_imgs.shape != hr_imgs.shape:
            raise ValueError(
                "sampler output and hr batch shape mismatch: "
                f"pred={tuple(sr_imgs.shape)} hr={tuple(hr_imgs.shape)}"
            )

        # Ground-truth batch is normalized to [-1,1] in Task6 transforms.
        hr_imgs_01 = torch.clamp((hr_imgs * 0.5) + 0.5, 0.0, 1.0)
        sr_imgs_01 = torch.clamp(sr_imgs, 0.0, 1.0)

        psnr_batch = SuperResolutionMetrics.compute_psnr_batch(hr_imgs_01, sr_imgs_01, data_range=1.0)
        ssim_batch = SuperResolutionMetrics.compute_ssim_batch(hr_imgs_01, sr_imgs_01, data_range=1.0)

        total_psnr += float(psnr_batch.sum().item())
        total_ssim += float(ssim_batch.sum().item())
        total_samples += int(psnr_batch.shape[0])

        if logger is not None and log_every_n_batches > 0 and (batch_idx + 1) % log_every_n_batches == 0:
            elapsed = max(1e-6, time.perf_counter() - start_time)
            throughput = float(total_samples) / elapsed
            logger.info(
                "%s progress: batch=%d samples=%d throughput=%.2f img/s",
                label,
                batch_idx + 1,
                total_samples,
                throughput,
            )

        # Release per-batch tensors to avoid long-loop memory growth.
        del lr_imgs, hr_imgs, sr_imgs, hr_imgs_01, sr_imgs_01, psnr_batch, ssim_batch
        if flush_every_n_batches > 0 and (batch_idx + 1) % flush_every_n_batches == 0:
            gc.collect()
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

    if total_samples == 0:
        raise ValueError("test_loader produced zero samples; cannot compute metrics")

    if logger is not None:
        elapsed = max(1e-6, time.perf_counter() - start_time)
        logger.info(
            "%s done: samples=%d time=%.2fs avg_psnr=%.4f avg_ssim=%.6f",
            label,
            total_samples,
            elapsed,
            total_psnr / total_samples,
            total_ssim / total_samples,
        )

    return {
        "psnr": total_psnr / total_samples,
        "ssim": total_ssim / total_samples,
        "num_samples": float(total_samples),
    }


__all__ = ["evaluate_super_resolution"]
