"""Centralized stage-boundary and weight scheduling helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass(frozen=True)
class StageBoundaries:
    stage_a_end_epoch: int
    stage_b_end_epoch: int


def compute_stage_boundaries(total_epochs: int, stage_a_fraction: float, stage_b_fraction: float) -> StageBoundaries:
    if total_epochs < 1:
        raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
    if not (0.0 <= stage_a_fraction <= 1.0):
        raise ValueError(f"stage_a_fraction must be in [0,1], got {stage_a_fraction}")
    if not (0.0 <= stage_b_fraction <= 1.0):
        raise ValueError(f"stage_b_fraction must be in [0,1], got {stage_b_fraction}")
    if stage_b_fraction < stage_a_fraction:
        raise ValueError("stage_b_fraction must be >= stage_a_fraction")

    stage_a_end = max(1, int(total_epochs * stage_a_fraction))
    stage_b_end = max(stage_a_end + 1, int(total_epochs * stage_b_fraction))
    return StageBoundaries(stage_a_end_epoch=stage_a_end, stage_b_end_epoch=stage_b_end)


def compute_curriculum_weights(epoch: int, total_epochs: int, curriculum_cfg: Dict[str, Any]) -> Tuple[float, float]:
    enabled = bool(curriculum_cfg.get("enabled", True))
    decoder_weight = float(curriculum_cfg.get("decoder_weight", 0.5))
    bottleneck_weight = float(curriculum_cfg.get("bottleneck_weight", 0.25))

    if not enabled:
        return decoder_weight, bottleneck_weight

    midpoint_fraction = float(curriculum_cfg.get("midpoint_fraction", 0.5))
    midpoint = int(total_epochs * midpoint_fraction)
    if epoch < midpoint:
        return decoder_weight, bottleneck_weight

    tail_len = max(1, total_epochs - midpoint - 1)
    progress = (epoch - midpoint) / tail_len
    min_decay = float(curriculum_cfg.get("min_decay", 0.1))
    if min_decay <= 0:
        raise ValueError(f"training.curriculum.min_decay must be > 0, got {min_decay}")

    decay = max(min_decay, 1.0 - progress)
    return decoder_weight * decay, bottleneck_weight * decay


def compute_staged_aux_weights(epoch: int, total_epochs: int, staged_cfg: Dict[str, Any]) -> Tuple[float, float]:
    if not bool(staged_cfg.get("enabled", False)):
        return 0.0, 0.0

    stage_a_fraction = float(staged_cfg.get("stage_a_fraction", 0.3))
    stage_b_fraction = float(staged_cfg.get("stage_b_fraction", 0.7))
    boundaries = compute_stage_boundaries(total_epochs, stage_a_fraction, stage_b_fraction)

    ordinal_weight = float(staged_cfg.get("ordinal_weight", 0.0))
    smoothness_weight = float(staged_cfg.get("smoothness_weight", 0.0))

    if epoch < boundaries.stage_a_end_epoch:
        return 0.0, 0.0
    if epoch < boundaries.stage_b_end_epoch:
        return ordinal_weight, 0.0
    return ordinal_weight, smoothness_weight


def summarize_stage_schedule(total_epochs: int, training_cfg: Dict[str, Any]) -> str:
    curriculum_cfg = training_cfg.get("curriculum", {})
    staged_cfg = training_cfg.get("staged_losses", {})
    mode = str(staged_cfg.get("mode", training_cfg.get("loss_mode", "legacy"))).strip().lower()

    stage_a_fraction = float(staged_cfg.get("stage_a_fraction", 0.3))
    stage_b_fraction = float(staged_cfg.get("stage_b_fraction", 0.7))
    boundaries = compute_stage_boundaries(total_epochs, stage_a_fraction, stage_b_fraction)

    if mode in {"flow", "flow_staged", "rectified_flow"}:
        return (
            "[train] stage schedule: "
            f"mode=flow_staged epochs={total_epochs} "
            f"stage1_end={boundaries.stage_a_end_epoch} "
            f"flow_weight={float(staged_cfg.get('flow_weight', 1.0)):.4f} "
            f"ssi_weight={float(staged_cfg.get('ssi_weight', 1.0)):.4f} "
            f"wavelet_weight={float(staged_cfg.get('wavelet_weight', 0.0)):.4f} "
            f"ordinal_weight={float(staged_cfg.get('ordinal_weight', 0.0)):.4f}"
        )

    return (
        "[train] stage schedule: "
        f"epochs={total_epochs} "
        f"stageA_end={boundaries.stage_a_end_epoch} "
        f"stageB_end={boundaries.stage_b_end_epoch} "
        f"curriculum_midpoint={float(curriculum_cfg.get('midpoint_fraction', 0.5)):.3f} "
        f"ordinal_weight={float(staged_cfg.get('ordinal_weight', 0.0)):.4f} "
        f"smoothness_weight={float(staged_cfg.get('smoothness_weight', 0.0)):.4f}"
    )
