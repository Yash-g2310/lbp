from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import numpy as np
import torch
import torch.optim as optim

from lbp_project.config.io import load_yaml
from lbp_project.config.stage_policy import infer_stage_mode, validate_stage_policy
from lbp_project.data.dataset import get_dataloaders
from lbp_project.data.preflight import build_download_matrix, enforce_startup_preflight, format_download_matrix
from lbp_project.models.factory import build_depth_model
from lbp_project.training.ablation import ablation_plan_payload, format_ablation_plan, resolve_ablation_plan
from lbp_project.training.stages import (
    compute_stage_boundaries,
    compute_curriculum_weights,
    compute_staged_aux_weights,
    summarize_stage_schedule,
)
from lbp_project.utils.logger import setup_wandb
from lbp_project.utils.losses import (
    EdgeAwareSmoothnessLoss,
    OrdinalDepthLoss,
    RectifiedFlowLoss,
    SILogLoss,
    ScaleShiftInvariantLoss,
    WaveletEdgeLoss,
    depth_to_inverse_normalized,
    inverse_normalized_to_metric_depth,
    reconstruct_depth_from_velocity,
)
from lbp_project.utils.run_manifest import config_sha256


STAGE_A = "stage_a"
STAGE_B = "stage_b"
STAGE_B_FINAL_REAL_SPLITS = ("validation", "test")
STAGE_B_FINAL_REAL_LAYER_KEYS = ("layer_all", "layer_first")


@dataclass
class StageBRuntimeContract:
    enabled: bool
    max_epochs: int
    max_runtime_hours: float
    job_start_ts: float
    job_start_source: str
    hard_deadline_ts: float
    require_terminal_full_real_eval: bool
    hard_fail_on_terminal_eval_failure: bool


def resolve_amp_dtype(hardware_cfg: Dict[str, Any], device: torch.device) -> torch.dtype:
    raw = str(hardware_cfg.get("amp_dtype", "float16")).strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype = mapping.get(raw)
    if dtype is None:
        raise ValueError(f"Unsupported hardware.amp_dtype='{raw}'. Use one of: float16, bfloat16")

    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return dtype


def build_grad_scaler(device: torch.device, enabled: bool) -> Any:
    grad_scaler_ctor = getattr(torch.amp, "GradScaler", None)
    if grad_scaler_ctor is not None:
        try:
            return grad_scaler_ctor(device=device.type, enabled=enabled)
        except TypeError:
            return grad_scaler_ctor(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled and device.type == "cuda")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Layered Depth Estimation")
    parser.add_argument("--config", type=str, default="configs/local/dev.yaml", help="Path to config YAML")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    return load_yaml(config_path)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def log_terminal(enabled: bool, message: str) -> None:
    if enabled and is_main_process():
        print(message, flush=True)


def tensor_stats(x: torch.Tensor | None, name: str) -> str:
    if x is None:
        return f"{name}=None"
    with torch.no_grad():
        finite = torch.isfinite(x)
        finite_count = int(finite.sum().item())
        total = int(x.numel())
        if finite_count == 0:
            return f"{name}: shape={tuple(x.shape)} finite=0/{total}"

        x_f = x[finite]
        mean_val = x_f.float().mean() if not (torch.is_floating_point(x_f) or torch.is_complex(x_f)) else x_f.mean()
        return (
            f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
            f"finite={finite_count}/{total} min={float(x_f.min().item()):.6g} "
            f"max={float(x_f.max().item()):.6g} mean={float(mean_val.item()):.6g}"
        )


def validate_stage_outputs(
    outputs: Dict[str, torch.Tensor],
    stage_name: str,
    epoch: int | None = None,
    step: int | None = None,
) -> None:
    for key in ("bottleneck", "decoder", "final"):
        out = outputs.get(key)
        if out is None:
            raise RuntimeError(f"Missing model output '{key}' in stage '{stage_name}'")
        if not torch.isfinite(out).all():
            ctx = ""
            if epoch is not None and step is not None:
                ctx = f" epoch={epoch+1} step={step+1}"
            raise RuntimeError(
                f"Non-finite model output detected at {stage_name}.{key}{ctx}; {tensor_stats(out, f'{stage_name}.{key}') }"
            )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    sched_cfg = cfg["training"]["scheduler"]
    name = sched_cfg["name"].lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg["t_max_epochs"]),
            eta_min=float(sched_cfg["eta_min"]),
        )
    raise ValueError(f"Unsupported scheduler: {sched_cfg['name']}")


def curriculum_weights(epoch: int, total_epochs: int, cfg: Dict[str, Any]) -> Tuple[float, float]:
    return compute_curriculum_weights(epoch, total_epochs, cfg["training"]["curriculum"])


def staged_aux_weights(epoch: int, total_epochs: int, cfg: Dict[str, Any]) -> Tuple[float, float]:
    return compute_staged_aux_weights(epoch, total_epochs, cfg.get("training", {}).get("staged_losses", {}))


def resolve_loss_mode(cfg: Dict[str, Any]) -> str:
    staged_cfg = cfg.get("training", {}).get("staged_losses", {})
    raw_mode = str(staged_cfg.get("mode", cfg.get("training", {}).get("loss_mode", "legacy"))).strip().lower()
    aliases = {
        "legacy": "legacy",
        "silog": "legacy",
        "silog_legacy": "legacy",
        "flow": "flow_staged",
        "flow_staged": "flow_staged",
        "rectified_flow": "flow_staged",
    }
    if raw_mode not in aliases:
        raise ValueError(
            "Unsupported training loss mode '{}'. Use one of: {}".format(
                raw_mode,
                sorted(set(aliases.keys())),
            )
        )
    return aliases[raw_mode]


def sample_rectified_flow_state(
    target_depth: torch.Tensor,
    t_low: float,
    t_high: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if target_depth.ndim != 4:
        raise ValueError(f"target_depth must be [B,1,H,W], got {tuple(target_depth.shape)}")
    if target_depth.shape[1] != 1:
        raise ValueError(f"target_depth channel must be 1, got {target_depth.shape[1]}")
    if t_low < 0.0 or t_high > 1.0 or t_high < t_low:
        raise ValueError(f"flow timestep range must satisfy 0 <= t_low <= t_high <= 1, got [{t_low}, {t_high}]")

    b = target_depth.shape[0]
    t = torch.empty(b, device=target_depth.device, dtype=target_depth.dtype).uniform_(t_low, t_high)
    noise = torch.randn_like(target_depth)
    t_view = t.view(-1, 1, 1, 1)
    noisy_depth = (1.0 - t_view) * noise + t_view * target_depth
    velocity_target = target_depth - noise
    return noisy_depth, t, velocity_target


def flow_component_weights_for_epoch(
    epoch: int,
    total_epochs: int,
    staged_cfg: Dict[str, Any],
) -> Dict[str, float | str]:
    stage_a_fraction = float(staged_cfg.get("stage_a_fraction", 0.3))
    stage_b_fraction = float(staged_cfg.get("stage_b_fraction", 0.7))
    boundaries = compute_stage_boundaries(total_epochs, stage_a_fraction, stage_b_fraction)
    stage2_active = epoch >= boundaries.stage_a_end_epoch

    flow_weight = float(staged_cfg.get("flow_weight", 1.0))
    ssi_weight = float(staged_cfg.get("ssi_weight", 1.0))
    wavelet_weight = float(staged_cfg.get("wavelet_weight", 0.0)) if stage2_active else 0.0
    ordinal_weight = float(staged_cfg.get("ordinal_weight", 0.0)) if stage2_active else 0.0

    return {
        "flow_weight": flow_weight,
        "ssi_weight": ssi_weight,
        "wavelet_weight": wavelet_weight,
        "ordinal_weight": ordinal_weight,
        "stage_label": "stage2_flow_ssi_wavelet_ordinal" if stage2_active else "stage1_flow_ssi",
    }


def compute_dynamic_multistage_loss(
    model: torch.nn.Module,
    criterion: SILogLoss,
    ordinal_criterion: OrdinalDepthLoss,
    smoothness_criterion: EdgeAwareSmoothnessLoss,
    images: torch.Tensor,
    depth_targets: torch.Tensor,
    depth_layer_mask: torch.Tensor,
    decoder_w: float,
    bottleneck_w: float,
    use_ckpt: bool,
    precomputed_dino: torch.Tensor | None,
    ordinal_weight: float,
    smoothness_weight: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if depth_targets.ndim != 5:
        raise ValueError(f"depth_targets must be [B,L,1,H,W], got {tuple(depth_targets.shape)}")
    if depth_layer_mask.ndim != 2:
        raise ValueError(f"depth_layer_mask must be [B,L], got {tuple(depth_layer_mask.shape)}")
    if depth_targets.shape[0] != images.shape[0] or depth_targets.shape[1] != depth_layer_mask.shape[1]:
        raise ValueError(
            "Batch/layer mismatch across inputs: "
            f"images={tuple(images.shape)} depth_targets={tuple(depth_targets.shape)} "
            f"depth_layer_mask={tuple(depth_layer_mask.shape)}"
        )

    batch_size = images.shape[0]
    num_layers = depth_targets.shape[1]
    depth_layer_mask = depth_layer_mask.to(dtype=torch.bool)

    total = images.new_tensor(0.0)
    components: Dict[str, float] = {}
    active_layers = 0
    smooth_losses: list[torch.Tensor] = []
    pred_by_layer: Dict[int, torch.Tensor] = {}
    pred_valid_by_layer: Dict[int, torch.Tensor] = {}

    for layer_idx in range(num_layers):
        sample_mask = depth_layer_mask[:, layer_idx]
        if not bool(sample_mask.any()):
            continue

        layer_id = layer_idx + 1
        selected_idx = torch.nonzero(sample_mask, as_tuple=False).squeeze(1)

        images_sub = images.index_select(0, selected_idx)
        depth_sub = depth_targets.index_select(0, selected_idx)[:, layer_idx]
        precomputed_sub = (
            precomputed_dino.index_select(0, selected_idx) if precomputed_dino is not None else None
        )
        target_layer = torch.full(
            (images_sub.shape[0],),
            layer_id,
            device=images.device,
            dtype=torch.long,
        )

        out = model(
            images_sub,
            target_layer=target_layer,
            return_intermediate=True,
            use_checkpointing=use_ckpt,
            precomputed_dino=precomputed_sub,
        )
        validate_stage_outputs(out, f"l{layer_id}")

        l_b = criterion(out["bottleneck"], depth_sub)
        l_d = criterion(out["decoder"], depth_sub)
        l_f = criterion(out["final"], depth_sub)

        total = total + l_f + decoder_w * l_d + bottleneck_w * l_b
        active_layers += 1

        components[f"l{layer_id}_b"] = float(l_b.detach().item())
        components[f"l{layer_id}_d"] = float(l_d.detach().item())
        components[f"l{layer_id}_f"] = float(l_f.detach().item())

        pred_full = images.new_zeros((batch_size, 1, images.shape[2], images.shape[3]))
        pred_full[selected_idx] = out["final"]
        pred_by_layer[layer_id] = pred_full
        pred_valid_by_layer[layer_id] = sample_mask

        if smoothness_weight > 0.0:
            valid_mask = torch.isfinite(depth_sub) & (depth_sub > 0)
            smooth_losses.append(smoothness_criterion(out["final"], images_sub, valid_mask=valid_mask))

    if active_layers == 0:
        raise RuntimeError("No active depth layers found in batch; cannot compute dynamic loss")

    total = total / float(active_layers)
    components["active_layers"] = float(active_layers)

    if ordinal_weight > 0.0 and len(pred_by_layer) > 1:
        ordinal_terms: list[torch.Tensor] = []
        for near_id in range(1, num_layers):
            far_id = near_id + 1
            if near_id not in pred_by_layer or far_id not in pred_by_layer:
                continue
            pair_mask = pred_valid_by_layer[near_id] & pred_valid_by_layer[far_id]
            if not bool(pair_mask.any()):
                continue

            near_pred = pred_by_layer[near_id][pair_mask]
            far_pred = pred_by_layer[far_id][pair_mask]
            near_target = depth_targets[pair_mask, near_id - 1]
            far_target = depth_targets[pair_mask, far_id - 1]
            ordinal_terms.append(ordinal_criterion(near_pred, far_pred, near_target, far_target))

        if ordinal_terms:
            ordinal_loss = torch.stack(ordinal_terms).mean()
            total = total + ordinal_weight * ordinal_loss
            components["ordinal"] = float(ordinal_loss.detach().item())

    if smoothness_weight > 0.0 and smooth_losses:
        smoothness_loss = torch.stack(smooth_losses).mean()
        total = total + smoothness_weight * smoothness_loss
        components["smoothness"] = float(smoothness_loss.detach().item())

    return total, components


def compute_dynamic_flow_loss(
    model: torch.nn.Module,
    flow_criterion: RectifiedFlowLoss,
    ssi_criterion: ScaleShiftInvariantLoss,
    wavelet_criterion: WaveletEdgeLoss,
    ordinal_criterion: OrdinalDepthLoss,
    images: torch.Tensor,
    depth_targets: torch.Tensor,
    depth_layer_mask: torch.Tensor,
    use_ckpt: bool,
    precomputed_dino: torch.Tensor | None,
    flow_weight: float,
    ssi_weight: float,
    wavelet_weight: float,
    ordinal_weight: float,
    flow_t_low: float,
    flow_t_high: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if depth_targets.ndim != 5:
        raise ValueError(f"depth_targets must be [B,L,1,H,W], got {tuple(depth_targets.shape)}")
    if depth_layer_mask.ndim != 2:
        raise ValueError(f"depth_layer_mask must be [B,L], got {tuple(depth_layer_mask.shape)}")
    if depth_targets.shape[0] != images.shape[0] or depth_targets.shape[1] != depth_layer_mask.shape[1]:
        raise ValueError(
            "Batch/layer mismatch across inputs: "
            f"images={tuple(images.shape)} depth_targets={tuple(depth_targets.shape)} "
            f"depth_layer_mask={tuple(depth_layer_mask.shape)}"
        )

    batch_size = images.shape[0]
    num_layers = depth_targets.shape[1]
    depth_layer_mask = depth_layer_mask.to(dtype=torch.bool)

    total = images.new_tensor(0.0)
    components: Dict[str, float] = {}
    active_layers = 0
    flow_terms: list[torch.Tensor] = []
    ssi_terms: list[torch.Tensor] = []
    wavelet_terms: list[torch.Tensor] = []
    pred_by_layer: Dict[int, torch.Tensor] = {}
    pred_valid_by_layer: Dict[int, torch.Tensor] = {}

    for layer_idx in range(num_layers):
        sample_mask = depth_layer_mask[:, layer_idx]
        if not bool(sample_mask.any()):
            continue

        layer_id = layer_idx + 1
        selected_idx = torch.nonzero(sample_mask, as_tuple=False).squeeze(1)

        images_sub = images.index_select(0, selected_idx)
        depth_sub_metric = depth_targets.index_select(0, selected_idx)[:, layer_idx]
        precomputed_sub = (
            precomputed_dino.index_select(0, selected_idx) if precomputed_dino is not None else None
        )
        target_layer = torch.full(
            (images_sub.shape[0],),
            layer_id,
            device=images.device,
            dtype=torch.long,
        )

        depth_sub_flow, flow_valid_mask, inv_max = depth_to_inverse_normalized(depth_sub_metric)

        noisy_sub, t_sub, velocity_target = sample_rectified_flow_state(
            depth_sub_flow,
            t_low=flow_t_low,
            t_high=flow_t_high,
        )

        out = model(
            images_sub,
            target_layer=target_layer,
            return_intermediate=True,
            use_checkpointing=use_ckpt,
            precomputed_dino=precomputed_sub,
            flow_noisy_depth=noisy_sub,
            flow_t=t_sub,
            return_velocity=True,
        )
        validate_stage_outputs(out, f"l{layer_id}")
        if "velocity" not in out:
            raise RuntimeError("Flow objective expected model output key 'velocity' but it was missing")

        velocity_pred = out["velocity"]
        if not torch.isfinite(velocity_pred).all():
            raise RuntimeError(f"Non-finite velocity prediction detected at flow layer {layer_id}")

        reconstructed_flow = reconstruct_depth_from_velocity(noisy_sub, velocity_pred, t_sub)
        reconstructed_metric = inverse_normalized_to_metric_depth(
            reconstructed_flow,
            inv_max,
            valid_mask=flow_valid_mask,
        )

        flow_loss = flow_criterion(velocity_pred, velocity_target, valid_mask=flow_valid_mask)
        ssi_loss = ssi_criterion(reconstructed_flow, depth_sub_flow, valid_mask=flow_valid_mask)
        layer_total = flow_weight * flow_loss + ssi_weight * ssi_loss
        flow_terms.append(flow_loss)
        ssi_terms.append(ssi_loss)

        if wavelet_weight > 0.0:
            reconstructed_wavelet = torch.where(flow_valid_mask, reconstructed_flow, depth_sub_flow)
            wavelet_loss = wavelet_criterion(reconstructed_wavelet, depth_sub_flow)
            layer_total = layer_total + wavelet_weight * wavelet_loss
            wavelet_terms.append(wavelet_loss)

        total = total + layer_total
        active_layers += 1

        pred_full = images.new_zeros((batch_size, 1, images.shape[2], images.shape[3]))
        pred_full[selected_idx] = reconstructed_metric
        pred_by_layer[layer_id] = pred_full
        pred_valid_by_layer[layer_id] = sample_mask

    if active_layers == 0:
        raise RuntimeError("No active depth layers found in batch; cannot compute flow loss")

    total = total / float(active_layers)
    components["active_layers"] = float(active_layers)
    components["flow"] = float(torch.stack(flow_terms).mean().detach().item()) if flow_terms else 0.0
    components["ssi"] = float(torch.stack(ssi_terms).mean().detach().item()) if ssi_terms else 0.0
    if wavelet_terms:
        components["wavelet"] = float(torch.stack(wavelet_terms).mean().detach().item())

    if ordinal_weight > 0.0 and len(pred_by_layer) > 1:
        ordinal_terms: list[torch.Tensor] = []
        for near_id in range(1, num_layers):
            far_id = near_id + 1
            if near_id not in pred_by_layer or far_id not in pred_by_layer:
                continue
            pair_mask = pred_valid_by_layer[near_id] & pred_valid_by_layer[far_id]
            if not bool(pair_mask.any()):
                continue

            near_pred = pred_by_layer[near_id][pair_mask]
            far_pred = pred_by_layer[far_id][pair_mask]
            near_target = depth_targets[pair_mask, near_id - 1]
            far_target = depth_targets[pair_mask, far_id - 1]
            ordinal_terms.append(ordinal_criterion(near_pred, far_pred, near_target, far_target))

        if ordinal_terms:
            ordinal_loss = torch.stack(ordinal_terms).mean()
            total = total + ordinal_weight * ordinal_loss
            components["ordinal"] = float(ordinal_loss.detach().item())

    return total, components


def validate_component_losses(components: Dict[str, float], epoch: int, step: int, eps: float = 1e-10) -> None:
    bad: list[str] = []
    all_tiny = True
    for key, val in components.items():
        if not np.isfinite(val):
            bad.append(f"{key}=non-finite")
            continue
        if val > eps:
            all_tiny = False
        if val < 0:
            bad.append(f"{key}={val}")
    if bad:
        raise RuntimeError(
            "Invalid component losses detected: "
            + ", ".join(bad)
            + f" (epoch={epoch+1} step={step+1})"
        )
    if all_tiny:
        raise RuntimeError(
            "All component losses are near-zero; training likely collapsed. "
            f"epoch={epoch+1} step={step+1} components={components}"
        )


def compute_multistage_loss(
    model: torch.nn.Module,
    criterion: SILogLoss,
    images: torch.Tensor,
    depth_1: torch.Tensor,
    depth_2: torch.Tensor,
    decoder_w: float,
    bottleneck_w: float,
    use_ckpt: bool,
    precomputed_dino: torch.Tensor | None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    out1 = model(
        images,
        target_layer=1,
        return_intermediate=True,
        use_checkpointing=use_ckpt,
        precomputed_dino=precomputed_dino,
    )
    out2 = model(
        images,
        target_layer=2,
        return_intermediate=True,
        use_checkpointing=use_ckpt,
        precomputed_dino=precomputed_dino,
    )

    validate_stage_outputs(out1, "l1")
    validate_stage_outputs(out2, "l2")

    l1_b = criterion(out1["bottleneck"], depth_1)
    l1_d = criterion(out1["decoder"], depth_1)
    l1_f = criterion(out1["final"], depth_1)
    l2_b = criterion(out2["bottleneck"], depth_2)
    l2_d = criterion(out2["decoder"], depth_2)
    l2_f = criterion(out2["final"], depth_2)

    total = (
        l1_f
        + decoder_w * l1_d
        + bottleneck_w * l1_b
        + l2_f
        + decoder_w * l2_d
        + bottleneck_w * l2_b
    )
    stats = {
        "l1_b": float(l1_b.detach().item()),
        "l1_d": float(l1_d.detach().item()),
        "l1_f": float(l1_f.detach().item()),
        "l2_b": float(l2_b.detach().item()),
        "l2_d": float(l2_d.detach().item()),
        "l2_f": float(l2_f.detach().item()),
    }
    return total, stats


def compute_single_stage_loss(
    model: torch.nn.Module,
    criterion: SILogLoss,
    images: torch.Tensor,
    depth: torch.Tensor,
    decoder_w: float,
    bottleneck_w: float,
    use_ckpt: bool,
    precomputed_dino: torch.Tensor | None,
    target_layer: int,
    prefix: str,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Quick runtime checks for NaNs/Infs in inputs to help trace upstream data issues.
    img_stats = tensor_stats(images, "images")
    depth_stats = tensor_stats(depth, "depth")
    dino_stats = tensor_stats(precomputed_dino, "precomputed_dino")
    if "finite=0/" in img_stats or "finite=0/" in depth_stats or "finite=0/" in dino_stats:
        raise RuntimeError(
            "Detected non-finite inputs before model forward: "
            f"{img_stats}; {depth_stats}; {dino_stats}"
        )

    # Log a brief summary to terminal for immediate visibility when verbose logging is enabled.
    # Disabled detailed per-batch input stats to reduce log spam.
    log_terminal(False, f"[debug] input stats: {img_stats}; {depth_stats}; {dino_stats}")
    out = model(
        images,
        target_layer=target_layer,
        return_intermediate=True,
        use_checkpointing=use_ckpt,
        precomputed_dino=precomputed_dino,
    )

    validate_stage_outputs(out, prefix)

    l_b = criterion(out["bottleneck"], depth)
    l_d = criterion(out["decoder"], depth)
    l_f = criterion(out["final"], depth)

    total = l_f + decoder_w * l_d + bottleneck_w * l_b
    stats = {
        f"{prefix}_b": float(l_b.detach().item()),
        f"{prefix}_d": float(l_d.detach().item()),
        f"{prefix}_f": float(l_f.detach().item()),
    }
    return total, stats


def compute_single_stage_loss_with_recovery(
    model: torch.nn.Module,
    criterion: SILogLoss,
    images: torch.Tensor,
    depth: torch.Tensor,
    decoder_w: float,
    bottleneck_w: float,
    use_ckpt: bool,
    precomputed_dino: torch.Tensor | None,
    target_layer: int,
    prefix: str,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    retry_in_fp32: bool,
    log_to_terminal: bool,
) -> Tuple[torch.Tensor, Dict[str, float], bool]:
    amp_ctx = (
        torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with amp_ctx:
        loss, comps = compute_single_stage_loss(
            model,
            criterion,
            images,
            depth,
            decoder_w,
            bottleneck_w,
            use_ckpt,
            precomputed_dino,
            target_layer=target_layer,
            prefix=prefix,
        )

    used_fp32_retry = False
    if torch.isfinite(loss):
        return loss, comps, used_fp32_retry

    if not retry_in_fp32:
        return loss, comps, used_fp32_retry

    if device.type == "cuda":
        torch.cuda.empty_cache()

    fp32_ctx = torch.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext()
    with fp32_ctx:
        loss, comps = compute_single_stage_loss(
            model,
            criterion,
            images.float(),
            depth.float(),
            decoder_w,
            bottleneck_w,
            use_ckpt,
            precomputed_dino.float() if precomputed_dino is not None else None,
            target_layer=target_layer,
            prefix=prefix,
        )
    used_fp32_retry = True
    if torch.isfinite(loss):
        log_terminal(log_to_terminal, f"[warn] recovered non-finite {prefix} loss via fp32 retry")
    return loss, comps, used_fp32_retry


def save_checkpoint(
    path: Path,
    epoch: int,
    best_val_loss: float,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    *,
    config_sha256_value: str,
    source_config_path: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": 2,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config_sha256": str(config_sha256_value),
            "source_config_path": str(source_config_path),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
        },
        path,
    )


def run_periodic_real_eval(
    config_path: str,
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    epoch: int,
    log_to_terminal: bool,
) -> Dict[str, float] | None:
    eval_cfg = cfg.get("evaluation", {})
    report_dir = Path(str(eval_cfg.get("report_dir", "./runs/current/reports")))
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / f"real_tuple_eval_epoch_{epoch+1}.json"

    splits = eval_cfg.get("real_splits", [eval_cfg.get("real_split", "validation")])
    layer_keys = eval_cfg.get("real_layer_keys", [eval_cfg.get("real_layer_key", "layer_all")])
    target_layer = int(eval_cfg.get("target_layer", 1))
    max_samples = int(eval_cfg.get("periodic_real_max_samples", 100))

    cmd = [
        sys.executable,
        "scripts/eval/eval_real_tuples.py",
        "--config",
        config_path,
        "--checkpoint",
        str(checkpoint_path),
        "--splits",
        ",".join(str(s) for s in splits),
        "--layer-keys",
        ",".join(str(k) for k in layer_keys),
        "--target-layer",
        str(target_layer),
        "--max-samples",
        str(max_samples),
        "--output",
        str(output_path),
    ]

    if bool(eval_cfg.get("real_use_precomputed_dino", False)):
        cmd.append("--use-precomputed-dino")
        idx_override = str(eval_cfg.get("real_precomputed_index_path", "")).strip()
        if idx_override:
            cmd.extend(["--precomputed-index-path", idx_override])

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        log_terminal(
            log_to_terminal,
            (
                f"[warn] periodic real eval failed at epoch {epoch+1}; "
                f"returncode={exc.returncode}"
            ),
        )
        if exc.stderr:
            tail = "\n".join(exc.stderr.strip().splitlines()[-5:])
            log_terminal(log_to_terminal, f"[warn] periodic eval stderr tail:\n{tail}")
        return None

    try:
        report = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log_terminal(log_to_terminal, f"[warn] periodic real eval report parse failed: {exc}")
        return None

    agg = report.get("aggregate", {})
    return {
        "pairs_acc": float(agg.get("pairs_acc", 0.0)),
        "trips_acc": float(agg.get("trips_acc", 0.0)),
        "quads_acc": float(agg.get("quads_acc", 0.0)),
        "all_acc": float(agg.get("all_acc", 0.0)),
    }


def _resolve_job_start_ts(fallback_ts: float) -> Tuple[float, str]:
    fallback = fallback_ts if fallback_ts > 0.0 else time.time()
    for key in ("LBP_JOB_START_TS", "SLURM_JOB_START_TS"):
        raw = str(os.environ.get(key, "")).strip()
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value > 0.0:
            return value, key
    return fallback, "process_start_fallback"


def resolve_stage_b_runtime_contract(
    cfg: Dict[str, Any],
    stage_mode: str,
    process_start_ts: float,
) -> StageBRuntimeContract:
    eval_cfg = cfg.get("evaluation", {})
    runtime_cfg = eval_cfg.get("stage_b_runtime", {})
    if runtime_cfg and not isinstance(runtime_cfg, dict):
        raise ValueError("evaluation.stage_b_runtime must be a mapping when provided")

    runtime_cfg = runtime_cfg if isinstance(runtime_cfg, dict) else {}
    enabled = stage_mode == STAGE_B and bool(runtime_cfg.get("enabled", True))

    max_epochs = int(runtime_cfg.get("max_epochs", 30))
    if max_epochs < 1:
        raise ValueError(f"evaluation.stage_b_runtime.max_epochs must be >= 1, got {max_epochs}")

    max_runtime_hours = float(runtime_cfg.get("max_runtime_hours", 24.0))
    if max_runtime_hours <= 0.0:
        raise ValueError(
            "evaluation.stage_b_runtime.max_runtime_hours must be > 0, "
            f"got {max_runtime_hours}"
        )

    require_terminal_full_real_eval = bool(runtime_cfg.get("require_terminal_full_real_eval", True))
    hard_fail_on_terminal_eval_failure = bool(
        runtime_cfg.get("hard_fail_on_terminal_eval_failure", True)
    )
    job_start_ts, source = _resolve_job_start_ts(process_start_ts)
    deadline_ts = job_start_ts + max_runtime_hours * 3600.0

    return StageBRuntimeContract(
        enabled=enabled,
        max_epochs=max_epochs,
        max_runtime_hours=max_runtime_hours,
        job_start_ts=job_start_ts,
        job_start_source=source,
        hard_deadline_ts=deadline_ts,
        require_terminal_full_real_eval=require_terminal_full_real_eval,
        hard_fail_on_terminal_eval_failure=hard_fail_on_terminal_eval_failure,
    )


def write_stage_b_runtime_state(report_dir: Path, payload: Dict[str, Any]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / "stage_b_runtime_state.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def resolve_terminal_eval_checkpoint(
    eval_cfg: Dict[str, Any],
    latest_ckpt: Path,
    best_ckpt: Path,
) -> Path:
    runtime_cfg = eval_cfg.get("stage_b_runtime", {})
    prefer_latest = True
    if isinstance(runtime_cfg, dict):
        prefer_latest = bool(runtime_cfg.get("terminal_eval_use_latest", True))

    candidates = [latest_ckpt, best_ckpt] if prefer_latest else [best_ckpt, latest_ckpt]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No checkpoint available for Stage B terminal evaluation. "
        f"Checked: {[str(c) for c in candidates]}"
    )


def run_terminal_full_real_eval(
    config_path: str,
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    stop_reason: str,
    log_to_terminal: bool,
) -> Dict[str, Any]:
    eval_cfg = cfg.get("evaluation", {})
    report_dir = Path(str(eval_cfg.get("report_dir", "./runs/current/reports")))
    report_dir.mkdir(parents=True, exist_ok=True)
    output_path = report_dir / "real_tuple_eval_terminal_full.json"

    split_arg = ",".join(STAGE_B_FINAL_REAL_SPLITS)
    layer_key_arg = ",".join(STAGE_B_FINAL_REAL_LAYER_KEYS)
    target_layer = int(eval_cfg.get("target_layer", 1))

    cmd = [
        sys.executable,
        "scripts/eval/eval_real_tuples.py",
        "--config",
        config_path,
        "--checkpoint",
        str(checkpoint_path),
        "--splits",
        split_arg,
        "--layer-keys",
        layer_key_arg,
        "--target-layer",
        str(target_layer),
        "--max-samples",
        "0",
        "--output",
        str(output_path),
    ]

    if bool(eval_cfg.get("real_use_precomputed_dino", False)):
        cmd.append("--use-precomputed-dino")
        idx_override = str(eval_cfg.get("real_precomputed_index_path", "")).strip()
        if idx_override:
            cmd.extend(["--precomputed-index-path", idx_override])

    log_terminal(
        log_to_terminal,
        (
            "[stage-b][terminal-eval] running strict final full real eval "
            f"stop_reason={stop_reason} checkpoint={checkpoint_path}"
        ),
    )
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr_tail = ""
        if exc.stderr:
            stderr_tail = "\n".join(exc.stderr.strip().splitlines()[-10:])
        raise RuntimeError(
            "Stage B terminal full real eval failed with non-zero exit code. "
            f"returncode={exc.returncode}\n"
            f"stderr_tail={stderr_tail}"
        ) from exc

    try:
        report = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(
            "Stage B terminal full real eval produced an unreadable report at "
            f"{output_path}: {exc}"
        ) from exc

    agg = report.get("aggregate", {})
    return {
        "report_path": str(output_path),
        "checkpoint": str(checkpoint_path),
        "stop_reason": stop_reason,
        "splits": list(STAGE_B_FINAL_REAL_SPLITS),
        "layer_keys": list(STAGE_B_FINAL_REAL_LAYER_KEYS),
        "max_samples": 0,
        "aggregate": {
            "pairs_acc": float(agg.get("pairs_acc", 0.0)),
            "trips_acc": float(agg.get("trips_acc", 0.0)),
            "quads_acc": float(agg.get("quads_acc", 0.0)),
            "all_acc": float(agg.get("all_acc", 0.0)),
        },
    }


def maybe_resume(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: Any,
    device: torch.device,
    *,
    resume_enabled: bool,
    expected_config_sha256: str,
    require_config_sha256_match: bool,
    fail_on_config_mismatch: bool,
    log_to_terminal: bool,
) -> Tuple[int, float]:
    if not bool(resume_enabled):
        log_terminal(log_to_terminal, "[resume] disabled by config; starting from epoch 0")
        return 0, float("inf")

    if not path.exists():
        return 0, float("inf")

    checkpoint = torch.load(path, map_location=device)

    observed_config_sha = str(checkpoint.get("config_sha256", "")).strip()
    if require_config_sha256_match:
        mismatch = (not observed_config_sha) or (observed_config_sha != str(expected_config_sha256).strip())
        if mismatch:
            msg = (
                "Resume checkpoint config hash mismatch. "
                f"checkpoint={path} observed={observed_config_sha or '<missing>'} "
                f"expected={expected_config_sha256}"
            )
            if fail_on_config_mismatch:
                raise RuntimeError(msg)
            log_terminal(
                log_to_terminal,
                f"[resume][warn] {msg}; ignoring checkpoint and starting from epoch 0",
            )
            return 0, float("inf")

    model_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    opt_key = "optimizer_state" if "optimizer_state" in checkpoint else "optimizer_state_dict"
    sch_key = "scheduler_state" if "scheduler_state" in checkpoint else "scheduler_state_dict"
    sca_key = "scaler_state" if "scaler_state" in checkpoint else "scaler_state_dict"

    model.load_state_dict(checkpoint[model_key])
    optimizer.load_state_dict(checkpoint[opt_key])
    scheduler.load_state_dict(checkpoint[sch_key])
    scaler.load_state_dict(checkpoint[sca_key])

    # Immediate finiteness checks to detect corrupted checkpoints early.
    for n, p in model.named_parameters():
        if not torch.isfinite(p).all():
            torch.save({
                'checkpoint_path': str(path),
                'bad_param': n,
                'model_state': checkpoint.get(model_key),
                'optimizer_state': checkpoint.get(opt_key),
            }, 'nan_dbg_loaded_ckpt.pt')
            raise RuntimeError(
                f"Loaded checkpoint {path} contains non-finite parameter {n}; saved nan_dbg_loaded_ckpt.pt"
            )
    try:
        for p in optimizer.state.keys():
            st = optimizer.state[p]
            for k, v in st.items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                    torch.save({
                        'checkpoint_path': str(path),
                        'bad_opt_buffer': k,
                        'model_state': checkpoint.get(model_key),
                        'optimizer_state': checkpoint.get(opt_key),
                    }, 'nan_dbg_loaded_ckpt_optbuf.pt')
                    raise RuntimeError(
                        f"Loaded checkpoint {path} contains non-finite optimizer buffer {k}; saved nan_dbg_loaded_ckpt_optbuf.pt"
                    )
    except Exception:
        # Non-fatal if optimizer state structure is unexpected; we only want to avoid masking real errors.
        pass

    best_val = checkpoint.get("best_val_loss", checkpoint.get("best_loss", float("inf")))
    log_terminal(
        log_to_terminal,
        (
            f"[resume] loaded checkpoint={path} epoch={int(checkpoint['epoch'])} "
            f"config_sha256={observed_config_sha or '<missing>'}"
        ),
    )
    return int(checkpoint["epoch"]) + 1, float(best_val)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_config_sha = config_sha256(config)
    process_start_ts = time.time()
    ablation_plan = resolve_ablation_plan(config)
    ablation_payload = ablation_plan_payload(ablation_plan)

    startup_matrix = build_download_matrix(config)
    print(format_download_matrix(startup_matrix, prefix="[startup]"), flush=True)
    print(format_ablation_plan(ablation_plan, prefix="[startup][ablation]"), flush=True)
    print(f"[startup][ablation-payload] {json.dumps(ablation_payload, sort_keys=True)}", flush=True)
    preflight_warnings = enforce_startup_preflight(
        config,
        strict_server_policy=bool(config.get("data", {}).get("require_local_staging", False)),
    )
    for warning in preflight_warnings:
        print(f"[startup][warn] {warning}", flush=True)
    inferred_stage_mode = infer_stage_mode(config, requested_mode="auto")
    print(f"[startup] inferred_stage_mode={inferred_stage_mode}", flush=True)
    for warning in validate_stage_policy(config, stage_mode=inferred_stage_mode, strict=False):
        print(f"[startup][warn] {warning}", flush=True)

    set_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and config["hardware"]["device"] == "cuda" else "cpu")
    force_fp32_impl = bool(config["hardware"].get("force_fp32_impl", False))
    amp_enabled = (not force_fp32_impl) and bool(config["hardware"].get("amp", True)) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(config["hardware"], device) if amp_enabled else torch.float32
    amp_scaler_enabled = amp_enabled and amp_dtype == torch.float16
    main_process = is_main_process()
    log_cfg = config.get("logging", {})
    eval_cfg = config.get("evaluation", {})
    log_to_terminal = bool(log_cfg.get("log_to_terminal", True))
    verbose_components = bool(log_cfg.get("verbose_components", True))
    resume_cfg = config.get("training", {}).get("resume", {})
    if not isinstance(resume_cfg, dict):
        raise ValueError("training.resume must be a mapping when provided")
    resume_enabled = bool(resume_cfg.get("enabled", True))
    resume_require_config_sha256_match = bool(
        resume_cfg.get("require_config_sha256_match", True)
    )
    resume_fail_on_config_mismatch = bool(
        resume_cfg.get("fail_on_config_mismatch", True)
    )
    periodic_eval_every = int(eval_cfg.get("periodic_real_eval_every_epochs", 0))
    staged_cfg = config.get("training", {}).get("staged_losses", {})
    loss_mode = resolve_loss_mode(config)
    flow_mode_enabled = loss_mode == "flow_staged"
    stage_b_contract = resolve_stage_b_runtime_contract(
        config,
        stage_mode=inferred_stage_mode,
        process_start_ts=process_start_ts,
    )

    fft_mode = "fp32" if force_fp32_impl else config["architecture"]["fft_mode"]
    model_cfg = dict(config)
    model_cfg["architecture"] = dict(config["architecture"])
    model_cfg["architecture"]["fft_mode"] = fft_mode
    model_cfg["architecture"]["max_layer_id"] = int(
        config["architecture"].get("max_layer_id", config["data"].get("max_layers", 8))
    )
    if flow_mode_enabled:
        model_cfg["architecture"]["enable_velocity_head"] = bool(
            model_cfg["architecture"].get("enable_velocity_head", True)
        )
        if not bool(model_cfg["architecture"]["enable_velocity_head"]):
            raise ValueError(
                "Flow-staged loss mode requires architecture.enable_velocity_head=true"
            )
    model: torch.nn.Module = build_depth_model(
        model_cfg,
        device,
        use_precomputed_dino=bool(config["data"]["use_precomputed_dino"]),
    )

    compile_requested = bool(config["hardware"].get("compile_model", False))
    if compile_requested and hasattr(torch, "compile"):
        compile_mode = str(config["hardware"].get("compile_mode", "default"))
        compile_dynamic = bool(config["hardware"].get("compile_dynamic", False))
        compile_options = dict(config["hardware"].get("compile_options", {}) or {})
        if bool(config["hardware"].get("compile_no_cudagraphs", True)):
            compile_options.setdefault("triton.cudagraphs", False)
        compile_options.setdefault("max_autotune_gemm", False)
        # Torch 2.10 disallows passing both mode and options together.
        if compile_options:
            compiled_model = torch.compile(
                model,
                dynamic=compile_dynamic,
                options=compile_options,
            )
        else:
            compiled_model = torch.compile(
                model,
                mode=compile_mode,
                dynamic=compile_dynamic,
            )
        model = cast(torch.nn.Module, compiled_model)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, config)
    criterion = SILogLoss(strict_empty_target=bool(config["training"].get("strict_empty_target", False)))
    ordinal_criterion = OrdinalDepthLoss(margin=float(staged_cfg.get("ordinal_margin", 1.0e-2)))
    smoothness_criterion = EdgeAwareSmoothnessLoss()
    flow_criterion = RectifiedFlowLoss() if flow_mode_enabled else None
    ssi_criterion = ScaleShiftInvariantLoss() if flow_mode_enabled else None
    wavelet_criterion = (
        WaveletEdgeLoss(
            family=str(staged_cfg.get("wavelet_family", "sym4")),
            levels=int(staged_cfg.get("wavelet_levels", 2)),
            fallback_family=str(staged_cfg.get("wavelet_fallback_family", "bior3.5")),
        )
        if flow_mode_enabled
        else None
    )
    flow_t_low = float(staged_cfg.get("flow_t_low", 0.0))
    flow_t_high = float(staged_cfg.get("flow_t_high", 1.0))
    if flow_mode_enabled and (flow_t_low < 0.0 or flow_t_high > 1.0 or flow_t_high < flow_t_low):
        raise ValueError(
            f"training.staged_losses flow timestep range must satisfy 0<=low<=high<=1; got [{flow_t_low}, {flow_t_high}]"
        )
    scaler = build_grad_scaler(device, enabled=amp_scaler_enabled)

    train_loader, val_loader = get_dataloaders(config)

    ckpt_cfg = config["training"]["checkpoint"]
    ckpt_dir = Path(ckpt_cfg["dir"])
    latest_ckpt = ckpt_dir / ckpt_cfg["latest_name"]
    best_ckpt = ckpt_dir / ckpt_cfg["best_name"]

    start_epoch, best_val_loss = maybe_resume(
        latest_ckpt,
        model,
        optimizer,
        scheduler,
        scaler,
        device,
        resume_enabled=resume_enabled,
        expected_config_sha256=run_config_sha,
        require_config_sha256_match=resume_require_config_sha256_match,
        fail_on_config_mismatch=resume_fail_on_config_mismatch,
        log_to_terminal=log_to_terminal,
    )
    run = setup_wandb(config, model=model if main_process else None)
    report_dir = Path(str(eval_cfg.get("report_dir", "./runs/current/reports")))
    stage_b_state_path = report_dir / "stage_b_runtime_state.json"
    stage_b_stop_reason: str | None = None
    stage_b_stop_epoch: int | None = None
    stage_b_partial_epoch = False
    stage_b_terminal_eval_result: Dict[str, Any] | None = None
    last_completed_epoch = start_epoch - 1

    accum_steps = int(config["training"]["accum_steps"])
    max_epochs = int(config["training"]["epochs"])
    log_terminal(log_to_terminal, summarize_stage_schedule(max_epochs, config["training"]))
    log_every = int(log_cfg.get("train_log_every_steps", log_cfg.get("log_every_steps", 20)))
    use_ckpt = bool(config["architecture"].get("use_gradient_checkpointing", False))
    dynamic_layers_enabled = bool(config["training"].get("dynamic_layers_enabled", False))
    memory_efficient_multistage = bool(config["training"].get("memory_efficient_multistage", True))
    retry_nonfinite_with_fp32 = bool(config["training"].get("retry_nonfinite_with_fp32", False))
    auto_disable_amp_on_nonfinite = bool(config["training"].get("auto_disable_amp_on_nonfinite", False))
    auto_disable_amp_on_overflow = bool(config["training"].get("auto_disable_amp_on_overflow", True))
    amp_overflow_patience = int(config["training"].get("amp_overflow_patience", 8))
    amp_disable_scale_threshold = float(config["training"].get("amp_disable_scale_threshold", 128.0))
    skip_nonfinite_grad_step = bool(config["training"].get("skip_nonfinite_grad_step", True))
    max_consecutive_nonfinite_steps = int(config["training"].get("max_consecutive_nonfinite_steps", 4))
    grad_clip_norm = float(config["training"].get("grad_clip_norm", 1.0))
    global_step = start_epoch * max(1, len(train_loader))
    amp_runtime_enabled = amp_enabled
    amp_runtime_dtype = amp_dtype
    consecutive_amp_overflows = 0
    consecutive_nonfinite_steps = 0
    prev_flow_stage_label: str | None = None

    log_terminal(
        log_to_terminal,
        (
            f"[train] device={device.type} amp={amp_enabled} compile={bool(config['hardware'].get('compile_model', False))} "
            f"amp_dtype={str(amp_dtype).replace('torch.', '')} "
            f"force_fp32_impl={force_fp32_impl} fft_mode={fft_mode} "
            f"precomputed_dino={bool(config['data'].get('use_precomputed_dino', False))} "
            f"loss_mode={loss_mode} start_epoch={start_epoch} config_sha256={run_config_sha}"
        ),
    )
    if run is not None:
        log_terminal(log_to_terminal, f"[train] W&B enabled: project={log_cfg.get('project')} run={run.name}")
    log_terminal(log_to_terminal, f"[train] Logging cadence: every {log_every} train steps")

    if stage_b_contract.enabled:
        log_terminal(
            log_to_terminal,
            (
                "[stage-b][runtime-contract] "
                f"max_epochs={stage_b_contract.max_epochs} "
                f"max_runtime_hours={stage_b_contract.max_runtime_hours:.2f} "
                f"job_start_source={stage_b_contract.job_start_source} "
                f"job_start_ts={stage_b_contract.job_start_ts:.3f} "
                f"deadline_ts={stage_b_contract.hard_deadline_ts:.3f}"
            ),
        )
        if main_process:
            write_stage_b_runtime_state(
                report_dir,
                {
                    "status": "running",
                    "stage_mode": inferred_stage_mode,
                    "max_epochs": stage_b_contract.max_epochs,
                    "max_runtime_hours": stage_b_contract.max_runtime_hours,
                    "job_start_ts": stage_b_contract.job_start_ts,
                    "job_start_source": stage_b_contract.job_start_source,
                    "deadline_ts": stage_b_contract.hard_deadline_ts,
                    "require_terminal_full_real_eval": stage_b_contract.require_terminal_full_real_eval,
                    "hard_fail_on_terminal_eval_failure": stage_b_contract.hard_fail_on_terminal_eval_failure,
                },
            )

    sched_cfg = config["training"]["scheduler"]
    if sched_cfg["name"].lower() == "cosine":
        t_max_epochs = int(sched_cfg["t_max_epochs"])
        if t_max_epochs != max_epochs:
            log_terminal(
                log_to_terminal,
                f"[warn] scheduler.t_max_epochs ({t_max_epochs}) != training.epochs ({max_epochs}); verify annealing intent",
            )

    for epoch in range(start_epoch, max_epochs):
        if stage_b_contract.enabled:
            if epoch >= stage_b_contract.max_epochs:
                stage_b_stop_reason = f"epoch_cap_reached_{stage_b_contract.max_epochs}"
                stage_b_stop_epoch = epoch
                log_terminal(
                    log_to_terminal,
                    (
                        "[stage-b][stop] stopping before epoch start due to epoch cap "
                        f"(max_epochs={stage_b_contract.max_epochs})"
                    ),
                )
                break

            now_ts = time.time()
            if now_ts >= stage_b_contract.hard_deadline_ts:
                elapsed_h = (now_ts - stage_b_contract.job_start_ts) / 3600.0
                stage_b_stop_reason = f"time_cap_reached_{stage_b_contract.max_runtime_hours:.2f}h"
                stage_b_stop_epoch = epoch
                log_terminal(
                    log_to_terminal,
                    (
                        "[stage-b][stop] stopping before epoch start due to wall-clock cap "
                        f"elapsed_hours={elapsed_h:.3f} max_hours={stage_b_contract.max_runtime_hours:.3f}"
                    ),
                )
                break

        model.train()
        optimizer.zero_grad(set_to_none=True)
        stop_current_epoch_early = False

        epoch_train_loss = 0.0
        step_counter = 0
        if flow_mode_enabled:
            decoder_w, bottleneck_w = 0.0, 0.0
            flow_weights = flow_component_weights_for_epoch(epoch, max_epochs, staged_cfg)
            flow_w = float(flow_weights["flow_weight"])
            ssi_w = float(flow_weights["ssi_weight"])
            wavelet_w = float(flow_weights["wavelet_weight"])
            ordinal_w = float(flow_weights["ordinal_weight"])
            smoothness_w = 0.0
            flow_stage_label = str(flow_weights["stage_label"])
            if flow_stage_label != prev_flow_stage_label:
                log_terminal(
                    log_to_terminal,
                    f"[train][stage-transition] epoch={epoch+1} -> {flow_stage_label}",
                )
                prev_flow_stage_label = flow_stage_label
        else:
            decoder_w, bottleneck_w = curriculum_weights(epoch, max_epochs, config)
            flow_w, ssi_w, wavelet_w = 0.0, 0.0, 0.0
            ordinal_w, smoothness_w = staged_aux_weights(epoch, max_epochs, config)

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            precomputed_dino = batch.get("dino_features")
            if precomputed_dino is not None:
                precomputed_dino = precomputed_dino.to(device, non_blocking=True)

            depth_1 = batch["depth_1"].to(device, non_blocking=True).float()
            depth_2 = batch["depth_2"].to(device, non_blocking=True).float()
            depth_targets = batch.get("depth_targets")
            depth_layer_mask = batch.get("depth_layer_mask")
            if depth_targets is not None:
                depth_targets = depth_targets.to(device, non_blocking=True).float()
            if depth_layer_mask is not None:
                depth_layer_mask = depth_layer_mask.to(device, non_blocking=True)

            if flow_mode_enabled:
                if depth_targets is None or depth_layer_mask is None:
                    depth_targets = torch.stack([depth_1, depth_2], dim=1)
                    depth_layer_mask = torch.ones(
                        (images.shape[0], depth_targets.shape[1]),
                        dtype=torch.bool,
                        device=device,
                    )
                amp_ctx = (
                    torch.autocast(device_type="cuda", enabled=amp_runtime_enabled, dtype=amp_runtime_dtype)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp_ctx:
                    if flow_criterion is None or wavelet_criterion is None or ssi_criterion is None:
                        raise RuntimeError(
                            "Flow mode is enabled but flow/wavelet/ssi criteria were not initialized"
                        )
                    loss, components = compute_dynamic_flow_loss(
                        model,
                        flow_criterion,
                        ssi_criterion,
                        wavelet_criterion,
                        ordinal_criterion,
                        images,
                        depth_targets,
                        depth_layer_mask,
                        use_ckpt,
                        precomputed_dino,
                        flow_weight=flow_w,
                        ssi_weight=ssi_w,
                        wavelet_weight=wavelet_w,
                        ordinal_weight=ordinal_w,
                        flow_t_low=flow_t_low,
                        flow_t_high=flow_t_high,
                    )
                    scaled_loss = loss / accum_steps
            elif dynamic_layers_enabled and depth_targets is not None and depth_layer_mask is not None:
                amp_ctx = (
                    torch.autocast(device_type="cuda", enabled=amp_runtime_enabled, dtype=amp_runtime_dtype)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp_ctx:
                    loss, components = compute_dynamic_multistage_loss(
                        model,
                        criterion,
                        ordinal_criterion,
                        smoothness_criterion,
                        images,
                        depth_targets,
                        depth_layer_mask,
                        decoder_w,
                        bottleneck_w,
                        use_ckpt,
                        precomputed_dino,
                        ordinal_weight=ordinal_w,
                        smoothness_weight=smoothness_w,
                    )
                    scaled_loss = loss / accum_steps
            else:
                amp_ctx = (
                    torch.autocast(device_type="cuda", enabled=amp_runtime_enabled, dtype=amp_runtime_dtype)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp_ctx:
                    loss, components = compute_multistage_loss(
                        model,
                        criterion,
                        images,
                        depth_1,
                        depth_2,
                        decoder_w,
                        bottleneck_w,
                        use_ckpt,
                        precomputed_dino,
                    )
                    scaled_loss = loss / accum_steps

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss detected. "
                    f"epoch={epoch+1} step={step+1} components={components}; "
                    f"{tensor_stats(images, 'images')}; {tensor_stats(depth_1, 'depth_1')}; "
                    f"{tensor_stats(depth_2, 'depth_2')}; {tensor_stats(precomputed_dino, 'precomputed_dino')}"
                )

            scaler.scale(scaled_loss).backward()
            validate_component_losses(components, epoch, step)

            should_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                skip_step = False
                nonfinite_grad_param = ""
                for n, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        nonfinite_grad_param = n
                        break

                if nonfinite_grad_param:
                    torch.save({
                        'epoch': epoch,
                        'step': step,
                        'global_step': global_step,
                        'bad_param': nonfinite_grad_param,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                    }, 'nan_dbg_grad.pt')

                    if amp_scaler_enabled and amp_runtime_enabled:
                        log_terminal(
                            log_to_terminal,
                            (
                                f"[warn] non-finite gradient detected for param {nonfinite_grad_param} "
                                f"at epoch={epoch+1} step={step+1}; skipping optimizer step"
                            ),
                        )
                        skip_step = True
                        consecutive_nonfinite_steps += 1
                        if auto_disable_amp_on_nonfinite:
                            amp_runtime_enabled = False
                            scaler = build_grad_scaler(device, enabled=False)
                            log_terminal(
                                log_to_terminal,
                                (
                                    "[warn] disabling AMP after non-finite gradient event; "
                                    f"bad_param={nonfinite_grad_param}"
                                ),
                            )
                    elif skip_nonfinite_grad_step:
                        log_terminal(
                            log_to_terminal,
                            (
                                f"[warn] non-finite gradient detected for param {nonfinite_grad_param} "
                                f"at epoch={epoch+1} step={step+1}; skipping optimizer step"
                            ),
                        )
                        skip_step = True
                        consecutive_nonfinite_steps += 1
                    else:
                        raise RuntimeError(
                            f"NaN gradient detected for param {nonfinite_grad_param} at epoch={epoch+1} step={step+1}"
                        )
                    grad_norm = torch.tensor(float("nan"), device=device)
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                    if not torch.isfinite(grad_norm):
                        if amp_scaler_enabled and amp_runtime_enabled:
                            log_terminal(
                                log_to_terminal,
                                f"[warn] non-finite grad_norm at epoch={epoch+1} step={step+1}; "
                                "allowing GradScaler to handle overflow/skip",
                            )
                            consecutive_nonfinite_steps += 1
                        elif skip_nonfinite_grad_step:
                            log_terminal(
                                log_to_terminal,
                                f"[warn] non-finite grad_norm at epoch={epoch+1} step={step+1}; skipping optimizer step",
                            )
                            skip_step = True
                            consecutive_nonfinite_steps += 1
                        else:
                            raise RuntimeError(
                                f"Non-finite gradient norm detected at epoch={epoch+1} step={step+1}: grad_norm={float(grad_norm)}"
                            )
                    else:
                        consecutive_nonfinite_steps = 0

                if consecutive_nonfinite_steps >= max(1, max_consecutive_nonfinite_steps):
                    raise RuntimeError(
                        "Repeated non-finite gradient events detected; aborting to avoid silent training collapse. "
                        f"count={consecutive_nonfinite_steps} epoch={epoch+1} step={step+1}"
                    )

                if not skip_step:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    # Debug probe: detect non-finite parameters or optimizer buffers right after the update.
                    for n, p in model.named_parameters():
                        if not torch.isfinite(p).all():
                            torch.save({
                                'epoch': epoch,
                                'step': step,
                                'global_step': global_step,
                                'bad_param': n,
                                'model_state': model.state_dict(),
                                'optimizer_state': optimizer.state_dict(),
                            }, 'nan_dbg_param.pt')
                            raise RuntimeError(f"NaN parameter {n} after optimizer.step() at epoch={epoch+1} step={step+1}")
                    # Check common optimizer buffers (e.g., Adam exp_avg/exp_avg_sq)
                    try:
                        for p in list(optimizer.state.keys()):
                            st = optimizer.state[p]
                            for k, v in st.items():
                                if isinstance(v, torch.Tensor):
                                    if not torch.isfinite(v).all():
                                        torch.save({
                                            'epoch': epoch,
                                            'step': step,
                                            'global_step': global_step,
                                            'bad_opt_buffer': k,
                                            'model_state': model.state_dict(),
                                            'optimizer_state': optimizer.state_dict(),
                                        }, 'nan_dbg_optbuf.pt')
                                        raise RuntimeError(f"NaN optimizer buffer {k} after optimizer.step() at epoch={epoch+1} step={step+1}")
                    except Exception:
                        # Avoid crashing the debugger probe itself in unusual optimizer implementations.
                        pass
                    scale_after = scaler.get_scale()
                    if amp_scaler_enabled and amp_runtime_enabled and scale_after < scale_before:
                        consecutive_amp_overflows += 1
                        log_terminal(
                            log_to_terminal,
                            f"[warn] AMP overflow/skip at epoch={epoch+1} step={step+1}; scaler {scale_before} -> {scale_after}",
                        )
                        if auto_disable_amp_on_overflow and (
                            consecutive_amp_overflows >= max(1, amp_overflow_patience)
                            or scale_after <= amp_disable_scale_threshold
                        ):
                            amp_runtime_enabled = False
                            scaler = build_grad_scaler(device, enabled=False)
                            log_terminal(
                                log_to_terminal,
                                (
                                    "[warn] disabling AMP due to repeated overflow; "
                                    f"consecutive_overflows={consecutive_amp_overflows} "
                                    f"scale={scale_before}->{scale_after}"
                                ),
                            )
                    else:
                        consecutive_amp_overflows = 0
                optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = None

            global_step += 1

            loss_value = float(loss.detach().item())
            epoch_train_loss += loss_value
            step_counter += 1

            if ((step + 1) % log_every == 0):
                log_payload = {
                    "train/loss": loss_value,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/decoder_weight": decoder_w,
                    "train/bottleneck_weight": bottleneck_w,
                    "train/ordinal_weight": ordinal_w,
                    "train/smoothness_weight": smoothness_w,
                    "train/epoch": epoch,
                    "train/step_in_epoch": step,
                    **{f"train/{k}": v for k, v in components.items()},
                }
                if flow_mode_enabled:
                    log_payload.update(
                        {
                            "train/flow_weight": flow_w,
                            "train/ssi_weight": ssi_w,
                            "train/wavelet_weight": wavelet_w,
                        }
                    )
                if grad_norm is not None:
                    log_payload["train/grad_norm"] = float(grad_norm.item())

                if run is not None and main_process:
                    run.log(log_payload, step=global_step)

                if main_process and log_to_terminal:
                    base_msg = (
                        f"[train][epoch={epoch+1}/{max_epochs} step={step+1}/{len(train_loader)} global_step={global_step}] "
                        f"loss={loss_value:.5f} lr={scheduler.get_last_lr()[0]:.3e} "
                        f"grad_norm={float(grad_norm.item()):.4f}" if grad_norm is not None else
                        f"[train][epoch={epoch+1}/{max_epochs} step={step+1}/{len(train_loader)} global_step={global_step}] "
                        f"loss={loss_value:.5f} lr={scheduler.get_last_lr()[0]:.3e}"
                    )
                    if verbose_components:
                        comp_msg = " ".join(
                            [f"{k}={v:.5f}" for k, v in components.items()]
                        )
                        log_terminal(True, f"{base_msg} {comp_msg}")
                    else:
                        log_terminal(True, base_msg)

            if stage_b_contract.enabled and time.time() >= stage_b_contract.hard_deadline_ts:
                elapsed_h = (time.time() - stage_b_contract.job_start_ts) / 3600.0
                stage_b_stop_reason = f"time_cap_reached_{stage_b_contract.max_runtime_hours:.2f}h"
                stage_b_stop_epoch = epoch + 1
                stage_b_partial_epoch = True
                stop_current_epoch_early = True
                log_terminal(
                    log_to_terminal,
                    (
                        "[stage-b][stop] wall-clock cap reached during epoch; "
                        f"epoch={epoch+1} step={step+1} elapsed_hours={elapsed_h:.3f}"
                    ),
                )
                break

        if stop_current_epoch_early:
            if step_counter > 0:
                save_checkpoint(
                    latest_ckpt,
                    epoch,
                    best_val_loss,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config_sha256_value=run_config_sha,
                    source_config_path=args.config,
                )
                log_terminal(
                    log_to_terminal,
                    f"[stage-b][stop] saved latest checkpoint before terminal eval: {latest_ckpt}",
                )
                last_completed_epoch = epoch
            else:
                log_terminal(
                    log_to_terminal,
                    "[stage-b][stop] wall-clock cap triggered before any train step in epoch; no checkpoint update",
                )
            break

        train_loss = epoch_train_loss / max(1, step_counter)

        model.eval()
        val_running = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                depth_1 = batch["depth_1"].to(device, non_blocking=True).float()
                depth_2 = batch["depth_2"].to(device, non_blocking=True).float()
                depth_targets = batch.get("depth_targets")
                depth_layer_mask = batch.get("depth_layer_mask")
                if depth_targets is not None:
                    depth_targets = depth_targets.to(device, non_blocking=True).float()
                if depth_layer_mask is not None:
                    depth_layer_mask = depth_layer_mask.to(device, non_blocking=True)
                precomputed_dino = batch.get("dino_features")
                if precomputed_dino is not None:
                    precomputed_dino = precomputed_dino.to(device, non_blocking=True)

                amp_ctx = (
                    torch.autocast(device_type="cuda", enabled=amp_runtime_enabled, dtype=amp_runtime_dtype)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with amp_ctx:
                    if flow_mode_enabled:
                        if depth_targets is None or depth_layer_mask is None:
                            depth_targets = torch.stack([depth_1, depth_2], dim=1)
                            depth_layer_mask = torch.ones(
                                (images.shape[0], depth_targets.shape[1]),
                                dtype=torch.bool,
                                device=device,
                            )
                        if flow_criterion is None or wavelet_criterion is None or ssi_criterion is None:
                            raise RuntimeError(
                                "Flow mode is enabled but flow/wavelet/ssi criteria were not initialized"
                            )
                        val_loss, _ = compute_dynamic_flow_loss(
                            model,
                            flow_criterion,
                            ssi_criterion,
                            wavelet_criterion,
                            ordinal_criterion,
                            images,
                            depth_targets,
                            depth_layer_mask,
                            use_ckpt,
                            precomputed_dino,
                            flow_weight=flow_w,
                            ssi_weight=ssi_w,
                            wavelet_weight=wavelet_w,
                            ordinal_weight=ordinal_w,
                            flow_t_low=flow_t_low,
                            flow_t_high=flow_t_high,
                        )
                    elif dynamic_layers_enabled and depth_targets is not None and depth_layer_mask is not None:
                        val_loss, _ = compute_dynamic_multistage_loss(
                            model,
                            criterion,
                            ordinal_criterion,
                            smoothness_criterion,
                            images,
                            depth_targets,
                            depth_layer_mask,
                            decoder_w,
                            bottleneck_w,
                            use_ckpt,
                            precomputed_dino,
                            ordinal_weight=ordinal_w,
                            smoothness_weight=smoothness_w,
                        )
                    else:
                        val_loss, _ = compute_multistage_loss(
                            model,
                            criterion,
                            images,
                            depth_1,
                            depth_2,
                            decoder_w,
                            bottleneck_w,
                            use_ckpt,
                            precomputed_dino,
                        )
                if not torch.isfinite(val_loss):
                    raise RuntimeError(
                        "Non-finite validation loss detected. "
                        f"epoch={epoch+1} val_step={val_steps+1} val_loss={float(val_loss.detach().item())}; "
                        f"{tensor_stats(images, 'images')}; {tensor_stats(depth_1, 'depth_1')}; "
                        f"{tensor_stats(depth_2, 'depth_2')}; {tensor_stats(precomputed_dino, 'precomputed_dino')}"
                    )
                val_running += float(val_loss.detach().item())
                val_steps += 1

        val_loss = val_running / max(1, val_steps)
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError(
                f"Epoch aggregate loss became non-finite at epoch={epoch+1}: train_loss={train_loss}, val_loss={val_loss}"
            )
        scheduler.step()

        if run is not None and main_process:
            run.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                },
                step=global_step,
            )

        if flow_mode_enabled:
            epoch_msg = (
                f"[epoch {epoch+1}/{max_epochs}] train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} flow_w={flow_w:.4f} ssi_w={ssi_w:.4f} "
                f"wavelet_w={wavelet_w:.4f} ordinal_w={ordinal_w:.4f}"
            )
        else:
            epoch_msg = (
                f"[epoch {epoch+1}/{max_epochs}] train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} decoder_w={decoder_w:.4f} bottleneck_w={bottleneck_w:.4f}"
            )
        log_terminal(log_to_terminal, epoch_msg)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                best_ckpt,
                epoch,
                best_val_loss,
                model,
                optimizer,
                scheduler,
                scaler,
                config_sha256_value=run_config_sha,
                source_config_path=args.config,
            )
            log_terminal(log_to_terminal, f"[ckpt] Saved new best checkpoint: {best_ckpt} (val_loss={val_loss:.5f})")

        if (epoch + 1) % int(ckpt_cfg.get("save_every_epochs", 1)) == 0:
            save_checkpoint(
                latest_ckpt,
                epoch,
                best_val_loss,
                model,
                optimizer,
                scheduler,
                scaler,
                config_sha256_value=run_config_sha,
                source_config_path=args.config,
            )
            log_terminal(log_to_terminal, f"[ckpt] Updated latest checkpoint: {latest_ckpt}")

        if main_process and periodic_eval_every > 0 and ((epoch + 1) % periodic_eval_every == 0):
            metrics = run_periodic_real_eval(
                config_path=args.config,
                cfg=config,
                checkpoint_path=latest_ckpt,
                epoch=epoch,
                log_to_terminal=log_to_terminal,
            )
            if metrics is not None:
                log_terminal(
                    log_to_terminal,
                    (
                        f"[periodic-real-eval][epoch {epoch+1}] "
                        f"pairs_acc={metrics['pairs_acc']:.4f} "
                        f"trips_acc={metrics['trips_acc']:.4f} "
                        f"quads_acc={metrics['quads_acc']:.4f} "
                        f"all_acc={metrics['all_acc']:.4f}"
                    ),
                )
                if run is not None:
                    run.log(
                        {
                            "periodic_real/pairs_acc": metrics["pairs_acc"],
                            "periodic_real/trips_acc": metrics["trips_acc"],
                            "periodic_real/quads_acc": metrics["quads_acc"],
                            "periodic_real/all_acc": metrics["all_acc"],
                            "periodic_real/epoch": epoch + 1,
                        },
                        step=global_step,
                    )

        last_completed_epoch = epoch
        if stage_b_contract.enabled and (epoch + 1) >= stage_b_contract.max_epochs:
            stage_b_stop_reason = f"epoch_cap_reached_{stage_b_contract.max_epochs}"
            stage_b_stop_epoch = epoch + 1
            log_terminal(
                log_to_terminal,
                (
                    "[stage-b][stop] epoch cap reached after checkpoint/eval sequence "
                    f"(epoch={epoch+1})"
                ),
            )
            break

    if stage_b_contract.enabled:
        if stage_b_stop_reason is None:
            stage_b_stop_reason = "train_loop_completed"
            stage_b_stop_epoch = last_completed_epoch + 1 if last_completed_epoch >= 0 else 0

        if not latest_ckpt.exists() and last_completed_epoch >= 0:
            save_checkpoint(
                latest_ckpt,
                last_completed_epoch,
                best_val_loss,
                model,
                optimizer,
                scheduler,
                scaler,
                config_sha256_value=run_config_sha,
                source_config_path=args.config,
            )
            log_terminal(
                log_to_terminal,
                f"[stage-b] latest checkpoint missing; saved terminal checkpoint to {latest_ckpt}",
            )

        terminal_eval_error: str | None = None
        if stage_b_contract.require_terminal_full_real_eval and main_process:
            terminal_checkpoint = resolve_terminal_eval_checkpoint(eval_cfg, latest_ckpt, best_ckpt)
            try:
                stage_b_terminal_eval_result = run_terminal_full_real_eval(
                    config_path=args.config,
                    cfg=config,
                    checkpoint_path=terminal_checkpoint,
                    stop_reason=stage_b_stop_reason,
                    log_to_terminal=log_to_terminal,
                )
                agg = stage_b_terminal_eval_result.get("aggregate", {})
                log_terminal(
                    log_to_terminal,
                    (
                        "[stage-b][terminal-eval] completed "
                        f"pairs_acc={float(agg.get('pairs_acc', 0.0)):.4f} "
                        f"trips_acc={float(agg.get('trips_acc', 0.0)):.4f} "
                        f"quads_acc={float(agg.get('quads_acc', 0.0)):.4f} "
                        f"all_acc={float(agg.get('all_acc', 0.0)):.4f}"
                    ),
                )
                if run is not None:
                    run.log(
                        {
                            "stage_b_terminal_eval/pairs_acc": float(agg.get("pairs_acc", 0.0)),
                            "stage_b_terminal_eval/trips_acc": float(agg.get("trips_acc", 0.0)),
                            "stage_b_terminal_eval/quads_acc": float(agg.get("quads_acc", 0.0)),
                            "stage_b_terminal_eval/all_acc": float(agg.get("all_acc", 0.0)),
                            "stage_b_terminal_eval/max_samples": 0,
                        },
                        step=global_step,
                    )
            except Exception as exc:
                terminal_eval_error = str(exc)
                log_terminal(
                    log_to_terminal,
                    f"[stage-b][terminal-eval][error] {terminal_eval_error}",
                )
                if stage_b_contract.hard_fail_on_terminal_eval_failure:
                    if main_process:
                        write_stage_b_runtime_state(
                            report_dir,
                            {
                                "status": "failed_terminal_eval",
                                "stop_reason": stage_b_stop_reason,
                                "stop_epoch": stage_b_stop_epoch,
                                "partial_epoch": stage_b_partial_epoch,
                                "last_completed_epoch": last_completed_epoch,
                                "global_step": global_step,
                                "max_epochs": stage_b_contract.max_epochs,
                                "max_runtime_hours": stage_b_contract.max_runtime_hours,
                                "job_start_ts": stage_b_contract.job_start_ts,
                                "job_start_source": stage_b_contract.job_start_source,
                                "deadline_ts": stage_b_contract.hard_deadline_ts,
                                "terminal_eval_error": terminal_eval_error,
                            },
                        )
                    if run is not None:
                        run.finish()
                        log_terminal(log_to_terminal, "[wandb] Run finished and synced.")
                    raise

        if main_process:
            write_stage_b_runtime_state(
                report_dir,
                {
                    "status": "stopped",
                    "stop_reason": stage_b_stop_reason,
                    "stop_epoch": stage_b_stop_epoch,
                    "partial_epoch": stage_b_partial_epoch,
                    "last_completed_epoch": last_completed_epoch,
                    "global_step": global_step,
                    "max_epochs": stage_b_contract.max_epochs,
                    "max_runtime_hours": stage_b_contract.max_runtime_hours,
                    "job_start_ts": stage_b_contract.job_start_ts,
                    "job_start_source": stage_b_contract.job_start_source,
                    "deadline_ts": stage_b_contract.hard_deadline_ts,
                    "terminal_eval_required": stage_b_contract.require_terminal_full_real_eval,
                    "terminal_eval_hard_fail": stage_b_contract.hard_fail_on_terminal_eval_failure,
                    "terminal_eval_error": terminal_eval_error,
                    "terminal_eval_result": stage_b_terminal_eval_result,
                },
            )
            log_terminal(log_to_terminal, f"[stage-b] runtime state written to {stage_b_state_path}")

    if run is not None:
        run.finish()
        log_terminal(log_to_terminal, "[wandb] Run finished and synced.")


if __name__ == "__main__":
    main()