"""Phase-9A ablation scaffold helpers.

This module intentionally keeps ablation handling thin and explicit:
- Parse and normalize ablation config.
- Enforce scaffold-only policy for Phase 9A.
- Provide a structured payload for logging/manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


PHASE_9A = "phase_9a"
SUPPORTED_ABLATION_VARIANTS = {
    "baseline",
    "frequency_only",
    "wavelet_only",
    "hybrid_wavelet_frequency",
}


@dataclass(frozen=True)
class AblationPlan:
    enabled: bool
    variant: str
    scaffold_only: bool
    allow_scaffold_only: bool
    notes: str


def resolve_ablation_plan(cfg: Dict[str, Any], phase: str = PHASE_9A) -> AblationPlan:
    train_cfg = cfg.get("training", {})
    raw_ablation = train_cfg.get("ablation", {})

    if raw_ablation is None:
        raw_ablation = {}
    if not isinstance(raw_ablation, dict):
        raise ValueError("training.ablation must be a mapping when provided")

    enabled = bool(raw_ablation.get("enabled", False))
    variant = str(raw_ablation.get("variant", "baseline")).strip().lower() or "baseline"
    allow_scaffold_only = bool(raw_ablation.get("allow_scaffold_only", False))
    notes = str(raw_ablation.get("notes", "")).strip()

    if variant not in SUPPORTED_ABLATION_VARIANTS:
        raise ValueError(
            "training.ablation.variant must be one of {} (got '{}')".format(
                sorted(SUPPORTED_ABLATION_VARIANTS),
                variant,
            )
        )

    scaffold_only = phase == PHASE_9A
    if enabled and scaffold_only and not allow_scaffold_only:
        raise ValueError(
            "Ablation execution is scaffold-only in Phase 9A. "
            "Set training.ablation.allow_scaffold_only=true to acknowledge scaffold mode, "
            "or set training.ablation.enabled=false."
        )

    return AblationPlan(
        enabled=enabled,
        variant=variant,
        scaffold_only=scaffold_only,
        allow_scaffold_only=allow_scaffold_only,
        notes=notes,
    )


def ablation_plan_payload(plan: AblationPlan) -> Dict[str, Any]:
    return {
        "enabled": bool(plan.enabled),
        "variant": str(plan.variant),
        "scaffold_only": bool(plan.scaffold_only),
        "allow_scaffold_only": bool(plan.allow_scaffold_only),
        "notes": str(plan.notes),
    }


def format_ablation_plan(plan: AblationPlan, prefix: str = "[ablation]") -> str:
    if not plan.enabled:
        return f"{prefix} enabled=false"

    lines = [
        f"{prefix} enabled=true variant={plan.variant}",
        f"{prefix} scaffold_only={plan.scaffold_only}",
        f"{prefix} allow_scaffold_only={plan.allow_scaffold_only}",
    ]
    if plan.notes:
        lines.append(f"{prefix} notes={plan.notes}")
    return "\n".join(lines)
