"""Stage-B promotion gate evaluation helpers."""

from __future__ import annotations

from typing import Any, Dict

from lbp_project.metric_gate import (
    PairsTrendGateResult,
    evaluate_pairs_trend_gate,
    format_pairs_trend_gate,
    require_pairs_trend_gate,
)


StageBGateResult = PairsTrendGateResult


def evaluate_stage_b_gate(cfg: Dict[str, Any]) -> StageBGateResult:
    return evaluate_pairs_trend_gate(
        cfg,
        gate_key="stage_b_gate",
        default_enabled=False,
        default_min_pairs_acc=5.0,
    )


def format_stage_b_gate(result: StageBGateResult, prefix: str = "[stage-b-gate]") -> str:
    return format_pairs_trend_gate(result, prefix=prefix)


def require_stage_b_gate(cfg: Dict[str, Any], prefix: str = "[stage-b-gate]") -> StageBGateResult:
    result = evaluate_stage_b_gate(cfg)
    return require_pairs_trend_gate(result, prefix=prefix)
