"""Stage-A local promotion gate helpers."""

from __future__ import annotations

from typing import Any, Dict

from lbp_project.metric_gate import (
    PairsTrendGateResult,
    evaluate_pairs_trend_gate,
    format_pairs_trend_gate,
    require_pairs_trend_gate,
)


StageAGateResult = PairsTrendGateResult


def evaluate_stage_a_gate(cfg: Dict[str, Any]) -> StageAGateResult:
    return evaluate_pairs_trend_gate(
        cfg,
        gate_key="stage_a_gate",
        default_enabled=True,
        default_min_pairs_acc=0.05,
    )


def format_stage_a_gate(result: StageAGateResult, prefix: str = "[stage-a-gate]") -> str:
    return format_pairs_trend_gate(result, prefix=prefix)


def require_stage_a_gate(cfg: Dict[str, Any], prefix: str = "[stage-a-gate]") -> StageAGateResult:
    result = evaluate_stage_a_gate(cfg)
    return require_pairs_trend_gate(result, prefix=prefix)
