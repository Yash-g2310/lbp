from __future__ import annotations

import gc
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from .metrics import SuperResolutionMetrics
from .train_sr import ReflowSampler


def _resolve_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _parse_shape3(name: str, value: Any) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-item list/tuple [C,H,W], got {value}")
    shape = (int(value[0]), int(value[1]), int(value[2]))
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"{name} dimensions must be > 0, got {shape}")
    return shape


def parse_task6b_shape_contract(
    dataset_cfg: Dict[str, Any],
    model_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, tuple[int, int, int]]:
    """Parse and validate explicit Task6B LR/HR shape contract.

    This helper enforces explicit `dataset.lr_target_shape` and `dataset.hr_target_shape`
    usage for notebook paths to prevent silent fallback drift.
    """
    if not isinstance(dataset_cfg, dict):
        raise TypeError(f"dataset_cfg must be a dictionary, got {type(dataset_cfg)}")

    if "lr_target_shape" not in dataset_cfg or "hr_target_shape" not in dataset_cfg:
        raise ValueError(
            "Task6B requires explicit dataset.lr_target_shape and dataset.hr_target_shape in config"
        )

    lr_shape = _parse_shape3("dataset.lr_target_shape", dataset_cfg["lr_target_shape"])
    hr_shape = _parse_shape3("dataset.hr_target_shape", dataset_cfg["hr_target_shape"])

    if hr_shape[0] != lr_shape[0]:
        raise ValueError(
            "Task6B LR/HR channel mismatch: "
            f"lr_target_shape={lr_shape}, hr_target_shape={hr_shape}"
        )
    if hr_shape[1] != lr_shape[1] * 2 or hr_shape[2] != lr_shape[2] * 2:
        raise ValueError(
            "Task6B HR shape must be exactly 2x LR shape: "
            f"lr_target_shape={lr_shape}, hr_target_shape={hr_shape}"
        )

    if model_cfg is not None:
        if not isinstance(model_cfg, dict):
            raise TypeError(f"model_cfg must be a dictionary, got {type(model_cfg)}")
        input_size = model_cfg.get("input_size", [75, 75])
        if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
            raise ValueError(f"model.input_size must be [H,W], got {input_size}")
        input_h, input_w = int(input_size[0]), int(input_size[1])
        model_lr_shape = (int(model_cfg.get("lr_channels", 1)), input_h, input_w)
        model_hr_shape = (int(model_cfg.get("in_channels", 1)), input_h * 2, input_w * 2)
        if lr_shape != model_lr_shape or hr_shape != model_hr_shape:
            raise ValueError(
                "Task6B dataset/model shape contract mismatch. "
                f"Expected lr_target_shape={model_lr_shape}, hr_target_shape={model_hr_shape}; "
                f"got lr_target_shape={lr_shape}, hr_target_shape={hr_shape}"
            )

    return {
        "lr_target_shape": lr_shape,
        "hr_target_shape": hr_shape,
    }


def resolve_task6b_history_epochs(
    checkpoint: Dict[str, Any],
    in_memory_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve history epoch count with checkpoint fallback and mismatch warnings."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dictionary, got {type(checkpoint)}")

    warnings: list[str] = []

    in_memory_epochs: Optional[int] = None
    if isinstance(in_memory_history, dict):
        train_values = in_memory_history.get("train_total_loss", [])
        if isinstance(train_values, list):
            in_memory_epochs = int(len(train_values))

    checkpoint_history_epochs: Optional[int] = None
    checkpoint_history_obj = checkpoint.get("history")
    if isinstance(checkpoint_history_obj, dict):
        ckpt_train_values = checkpoint_history_obj.get("train_total_loss", [])
        if isinstance(ckpt_train_values, list):
            checkpoint_history_epochs = int(len(ckpt_train_values))

    completed_epochs: Optional[int] = None
    completed_epochs_obj = checkpoint.get("completed_epochs")
    if completed_epochs_obj is not None:
        completed_epochs = int(completed_epochs_obj)

    resolved_epochs = 0
    for candidate in (in_memory_epochs, completed_epochs, checkpoint_history_epochs):
        if candidate is not None and candidate > 0:
            resolved_epochs = int(candidate)
            break

    if completed_epochs is not None and checkpoint_history_epochs is not None and completed_epochs != checkpoint_history_epochs:
        warnings.append(
            "Checkpoint metadata mismatch: completed_epochs "
            f"({completed_epochs}) != len(checkpoint.history.train_total_loss) ({checkpoint_history_epochs})"
        )
    if in_memory_epochs is not None and completed_epochs is not None and in_memory_epochs != completed_epochs:
        warnings.append(
            "In-memory history mismatch with checkpoint completed_epochs: "
            f"in_memory={in_memory_epochs}, checkpoint={completed_epochs}"
        )
    if in_memory_epochs is not None and checkpoint_history_epochs is not None and in_memory_epochs != checkpoint_history_epochs:
        warnings.append(
            "In-memory history mismatch with checkpoint history: "
            f"in_memory={in_memory_epochs}, checkpoint_history={checkpoint_history_epochs}"
        )

    return {
        "resolved_epochs": int(resolved_epochs),
        "in_memory_epochs": in_memory_epochs,
        "checkpoint_completed_epochs": completed_epochs,
        "checkpoint_history_epochs": checkpoint_history_epochs,
        "warnings": warnings,
    }


def build_task6b_checkpoint_provenance(
    checkpoint_path: Path,
    checkpoint: Dict[str, Any],
    history_resolution: Dict[str, Any],
    trained_in_session: bool,
) -> Dict[str, Any]:
    """Build standardized provenance block for Task6B reports."""
    if not isinstance(checkpoint, dict):
        raise TypeError(f"checkpoint must be a dictionary, got {type(checkpoint)}")
    if not isinstance(history_resolution, dict):
        raise TypeError(
            f"history_resolution must be a dictionary, got {type(history_resolution)}"
        )

    mtime_utc = None
    if checkpoint_path.exists() and checkpoint_path.is_file():
        mtime_utc = datetime.fromtimestamp(checkpoint_path.stat().st_mtime, tz=timezone.utc).isoformat()

    return {
        "trained_in_session": bool(trained_in_session),
        "checkpoint_schema_version": checkpoint.get("checkpoint_schema_version"),
        "checkpoint_completed_epochs": history_resolution.get("checkpoint_completed_epochs"),
        "checkpoint_history_epochs": history_resolution.get("checkpoint_history_epochs"),
        "history_in_memory_epochs": history_resolution.get("in_memory_epochs"),
        "history_resolution_warnings": list(history_resolution.get("warnings", [])),
        "checkpoint_training_end_timestamp_utc": checkpoint.get("training_end_timestamp_utc"),
        "checkpoint_config_fingerprint_sha256": checkpoint.get("config_fingerprint_sha256"),
        "checkpoint_file_mtime_utc": mtime_utc,
    }


def _batch_correlation(lr_img: torch.Tensor, hr_img: torch.Tensor) -> float:
    """Compute Pearson correlation between upscaled LR and HR in [0,1] space."""
    lr_01 = torch.clamp((lr_img * 0.5) + 0.5, 0.0, 1.0)
    hr_01 = torch.clamp((hr_img * 0.5) + 0.5, 0.0, 1.0)

    lr_up = F.interpolate(
        lr_01.unsqueeze(0),
        size=tuple(hr_01.shape[-2:]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)

    a = lr_up.flatten()
    b = hr_01.flatten()
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = torch.sqrt((a_centered * a_centered).sum() * (b_centered * b_centered).sum())
    if float(denom.item()) <= 1e-8:
        return 0.0
    corr = (a_centered * b_centered).sum() / denom
    return float(corr.item())


@torch.no_grad()
def select_best_aligned_preview(
    loader,
    max_probe_batches: int = 2,
) -> Dict[str, Any]:
    """Pick the best LR/HR-aligned sample from the first few loader batches."""
    if max_probe_batches <= 0:
        raise ValueError(f"max_probe_batches must be > 0, got {max_probe_batches}")

    best: Optional[Dict[str, Any]] = None
    global_offset = 0

    for batch_idx, (lr_batch, hr_batch) in enumerate(loader):
        if batch_idx >= max_probe_batches:
            break

        if lr_batch.ndim != 4 or hr_batch.ndim != 4:
            raise ValueError(
                "Expected rank-4 batches [B,C,H,W], "
                f"got lr={tuple(lr_batch.shape)} hr={tuple(hr_batch.shape)}"
            )

        batch_size = int(lr_batch.shape[0])
        for sample_idx in range(batch_size):
            lr_img = lr_batch[sample_idx]
            hr_img = hr_batch[sample_idx]
            corr = _batch_correlation(lr_img, hr_img)
            candidate = {
                "lr": lr_img.detach().cpu(),
                "hr": hr_img.detach().cpu(),
                "correlation": corr,
                "batch_index": int(batch_idx),
                "sample_index": int(sample_idx),
                "global_index": int(global_offset + sample_idx),
            }
            if best is None or corr > float(best["correlation"]):
                best = candidate

        global_offset += batch_size

    if best is None:
        raise RuntimeError("Could not extract preview sample from loader")

    return best


class TTASamplerWrapper:
    """OOM-safe sequential TTA sampler wrapper with inverse alignment."""

    def __init__(self, base_sampler: ReflowSampler, micro_batch_size: int = 1):
        if micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be > 0, got {micro_batch_size}")
        self.base_sampler = base_sampler
        self.micro_batch_size = int(micro_batch_size)

    @staticmethod
    def _hflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(-1,))

    @staticmethod
    def _vflip(x: torch.Tensor) -> torch.Tensor:
        return torch.flip(x, dims=(-2,))

    @staticmethod
    def _rot90_k1(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=1, dims=(-2, -1))

    @staticmethod
    def _rot90_k3(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, k=3, dims=(-2, -1))

    def _run_variant(self, lr_batch: torch.Tensor, transform, inverse) -> torch.Tensor:
        transformed = transform(lr_batch)
        outputs = []
        for start in range(0, transformed.shape[0], self.micro_batch_size):
            lr_chunk = transformed[start : start + self.micro_batch_size]
            sr_chunk = self.base_sampler.sample(lr_chunk)
            outputs.append(inverse(sr_chunk))
            del lr_chunk, sr_chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pred = torch.cat(outputs, dim=0)
        del transformed, outputs
        return pred

    def sample(self, lr_batch: torch.Tensor) -> torch.Tensor:
        variants = [
            (lambda x: x, lambda x: x),
            (self._hflip, self._hflip),
            (self._vflip, self._vflip),
            (self._rot90_k1, self._rot90_k3),
        ]
        acc = None
        for transform, inverse in variants:
            pred = self._run_variant(lr_batch, transform, inverse)
            acc = pred if acc is None else (acc + pred)
            del pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if acc is None:
            raise RuntimeError("TTA accumulator unexpectedly empty")
        return torch.clamp(acc / 4.0, 0.0, 1.0)


@torch.no_grad()
def collect_tta_records(
    eval_loader,
    tta_sampler,
    device: Optional[Union[str, torch.device]],
    max_records: int = 8,
    amp_enabled: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> List[Dict[str, Any]]:
    """Collect a bounded set of per-image tensors and metrics for reporting."""
    if max_records <= 0:
        raise ValueError(f"max_records must be > 0, got {max_records}")

    device_obj = _resolve_device(device)
    records: List[Dict[str, Any]] = []

    for lr_imgs, hr_imgs in eval_loader:
        lr_imgs = lr_imgs.to(device_obj, non_blocking=True)
        hr_imgs = hr_imgs.to(device_obj, non_blocking=True)

        if amp_enabled and device_obj.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                sr_imgs = tta_sampler.sample(lr_imgs)
        else:
            sr_imgs = tta_sampler.sample(lr_imgs)

        hr_01 = torch.clamp((hr_imgs * 0.5) + 0.5, 0.0, 1.0)
        psnr_batch = SuperResolutionMetrics.compute_psnr_batch(hr_01, sr_imgs, data_range=1.0)
        ssim_batch = SuperResolutionMetrics.compute_ssim_batch(hr_01, sr_imgs, data_range=1.0)

        for idx in range(sr_imgs.shape[0]):
            if len(records) >= max_records:
                break
            records.append(
                {
                    "lr": lr_imgs[idx].detach().cpu(),
                    "sr": sr_imgs[idx].detach().cpu(),
                    "hr": hr_01[idx].detach().cpu(),
                    "psnr": float(psnr_batch[idx].item()),
                    "ssim": float(ssim_batch[idx].item()),
                }
            )

        del lr_imgs, hr_imgs, sr_imgs, hr_01, psnr_batch, ssim_batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(records) >= max_records:
            break

    if not records:
        raise RuntimeError("No per-image TTA records collected")
    return records


@torch.no_grad()
def evaluate_sampler_with_records(
    model,
    sampler,
    test_loader,
    device: Optional[Union[str, torch.device]] = None,
    max_images: Optional[int] = None,
    max_records: int = 8,
    flush_every_n_batches: int = 0,
    logger: Optional[logging.Logger] = None,
    log_every_n_batches: int = 0,
    label: str = "eval",
) -> Dict[str, Any]:
    """Evaluate sampler once and collect first max_records outputs in the same pass."""
    if max_images is not None and max_images <= 0:
        raise ValueError(f"max_images must be > 0 when provided, got {max_images}")
    if max_records <= 0:
        raise ValueError(f"max_records must be > 0, got {max_records}")

    device_obj = _resolve_device(device)
    model = model.to(device_obj)
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    total_samples = 0
    records: List[Dict[str, Any]] = []
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
        if sr_imgs.shape != hr_imgs.shape:
            raise ValueError(
                "sampler output and hr batch shape mismatch: "
                f"pred={tuple(sr_imgs.shape)} hr={tuple(hr_imgs.shape)}"
            )

        hr_imgs_01 = torch.clamp((hr_imgs * 0.5) + 0.5, 0.0, 1.0)
        sr_imgs_01 = torch.clamp(sr_imgs, 0.0, 1.0)

        psnr_batch = SuperResolutionMetrics.compute_psnr_batch(hr_imgs_01, sr_imgs_01, data_range=1.0)
        ssim_batch = SuperResolutionMetrics.compute_ssim_batch(hr_imgs_01, sr_imgs_01, data_range=1.0)

        total_psnr += float(psnr_batch.sum().item())
        total_ssim += float(ssim_batch.sum().item())
        total_samples += int(psnr_batch.shape[0])

        for idx in range(sr_imgs.shape[0]):
            if len(records) >= max_records:
                break
            records.append(
                {
                    "lr": lr_imgs[idx].detach().cpu(),
                    "sr": sr_imgs_01[idx].detach().cpu(),
                    "hr": hr_imgs_01[idx].detach().cpu(),
                    "psnr": float(psnr_batch[idx].item()),
                    "ssim": float(ssim_batch[idx].item()),
                }
            )

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

        del lr_imgs, hr_imgs, sr_imgs, hr_imgs_01, sr_imgs_01, psnr_batch, ssim_batch
        if flush_every_n_batches > 0 and (batch_idx + 1) % flush_every_n_batches == 0:
            gc.collect()
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

    if total_samples == 0:
        raise ValueError("test_loader produced zero samples; cannot compute metrics")
    if not records:
        raise RuntimeError("No per-image records collected during evaluation")

    metrics = {
        "psnr": total_psnr / total_samples,
        "ssim": total_ssim / total_samples,
        "num_samples": float(total_samples),
    }

    if logger is not None:
        elapsed = max(1e-6, time.perf_counter() - start_time)
        logger.info(
            "%s done: samples=%d time=%.2fs avg_psnr=%.4f avg_ssim=%.6f",
            label,
            total_samples,
            elapsed,
            metrics["psnr"],
            metrics["ssim"],
        )

    return {
        "metrics": metrics,
        "records": records,
    }


__all__ = [
    "TTASamplerWrapper",
    "build_task6b_checkpoint_provenance",
    "collect_tta_records",
    "evaluate_sampler_with_records",
    "parse_task6b_shape_contract",
    "resolve_task6b_history_epochs",
    "select_best_aligned_preview",
]
