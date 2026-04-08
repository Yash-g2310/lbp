"""Shared inference helpers for evaluation scripts."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch


def _autocast_context(device: torch.device, amp_enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=amp_enabled)
    return nullcontext()


def _predict_direct_depth(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_layer: int | torch.Tensor,
    precomputed_dino: torch.Tensor | None,
    device: torch.device,
    amp_enabled: bool,
) -> torch.Tensor:
    amp_ctx = _autocast_context(device=device, amp_enabled=amp_enabled)
    with amp_ctx:
        return model(
            images,
            target_layer=target_layer,
            return_intermediate=False,
            precomputed_dino=precomputed_dino,
        )


def _predict_flow_reconstruction(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_layer: int | torch.Tensor,
    precomputed_dino: torch.Tensor | None,
    device: torch.device,
    amp_enabled: bool,
    flow_steps: int,
    flow_t_low: float,
    flow_t_high: float,
    flow_init: str,
    depth_clip_min: float,
    depth_clip_max: float,
) -> torch.Tensor:
    if flow_steps < 1:
        raise ValueError(f"flow_steps must be >= 1, got {flow_steps}")
    if flow_t_low < 0.0 or flow_t_high > 1.0 or flow_t_high < flow_t_low:
        raise ValueError(
            "flow timestep range must satisfy 0 <= flow_t_low <= flow_t_high <= 1, "
            f"got [{flow_t_low}, {flow_t_high}]"
        )

    bsz, _, height, width = images.shape
    flow_init_norm = str(flow_init).strip().lower()
    if flow_init_norm == "zeros":
        noisy_depth = torch.zeros((bsz, 1, height, width), device=device, dtype=images.dtype)
    elif flow_init_norm in {"gaussian", "normal", "noise"}:
        noisy_depth = torch.randn((bsz, 1, height, width), device=device, dtype=images.dtype)
    else:
        raise ValueError(
            "evaluation.flow_inference_init must be one of: zeros, gaussian. "
            f"Got '{flow_init}'"
        )

    dt = (flow_t_high - flow_t_low) / float(flow_steps)
    if dt <= 0.0:
        raise ValueError(f"flow integration step must be > 0, got dt={dt}")

    for step in range(flow_steps):
        t_scalar = flow_t_low + step * dt
        t_batch = torch.full((bsz,), float(t_scalar), device=device, dtype=images.dtype)

        amp_ctx = _autocast_context(device=device, amp_enabled=amp_enabled)
        with amp_ctx:
            velocity = model(
                images,
                target_layer=target_layer,
                return_intermediate=False,
                precomputed_dino=precomputed_dino,
                flow_noisy_depth=noisy_depth,
                flow_t=t_batch,
                return_velocity=True,
            )

        noisy_depth = noisy_depth + velocity * dt

    # Convert normalized inverse-depth proxy to metric-like depth for existing evaluators.
    inverse_depth = ((noisy_depth + 1.0) * 0.5).clamp_min(1.0e-6)
    depth = torch.reciprocal(inverse_depth)
    if depth_clip_max > 0.0:
        depth = torch.clamp(depth, min=depth_clip_min, max=depth_clip_max)
    else:
        depth = torch.clamp(depth, min=depth_clip_min)
    return depth


def predict_depth_for_eval(
    model: torch.nn.Module,
    images: torch.Tensor,
    target_layer: int | torch.Tensor,
    precomputed_dino: torch.Tensor | None,
    device: torch.device,
    amp_enabled: bool,
    *,
    use_flow_inference: bool,
    flow_steps: int,
    flow_t_low: float,
    flow_t_high: float,
    flow_init: str,
    depth_clip_min: float,
    depth_clip_max: float,
) -> torch.Tensor:
    if not use_flow_inference:
        return _predict_direct_depth(
            model,
            images,
            target_layer=target_layer,
            precomputed_dino=precomputed_dino,
            device=device,
            amp_enabled=amp_enabled,
        )

    return _predict_flow_reconstruction(
        model,
        images,
        target_layer=target_layer,
        precomputed_dino=precomputed_dino,
        device=device,
        amp_enabled=amp_enabled,
        flow_steps=int(flow_steps),
        flow_t_low=float(flow_t_low),
        flow_t_high=float(flow_t_high),
        flow_init=str(flow_init),
        depth_clip_min=float(depth_clip_min),
        depth_clip_max=float(depth_clip_max),
    )


def resolve_flow_eval_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    training_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
    staged_cfg = training_cfg.get("staged_losses", {}) if isinstance(training_cfg, dict) else {}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}

    mode = str(staged_cfg.get("mode", training_cfg.get("loss_mode", "legacy"))).strip().lower()
    flow_mode = mode in {"flow", "flow_staged", "rectified_flow"}

    return {
        "flow_mode": flow_mode,
        "use_flow_inference": bool(eval_cfg.get("flow_inference_enabled", flow_mode)),
        "flow_steps": int(eval_cfg.get("flow_inference_steps", 16)),
        "flow_init": str(eval_cfg.get("flow_inference_init", "zeros")),
        "flow_t_low": float(staged_cfg.get("flow_t_low", 0.0)),
        "flow_t_high": float(staged_cfg.get("flow_t_high", 1.0)),
        "depth_clip_min": float(data_cfg.get("depth_clip_min", 1.0e-3)),
        "depth_clip_max": float(data_cfg.get("depth_clip_max", 30.0)),
    }
