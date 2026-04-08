"""Generic tuple-metric gate evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple


_EPOCH_REPORT_RE = re.compile(r"real_tuple_eval_epoch_(\d+)\.json$")


@dataclass
class PairsTrendGateResult:
    gate_key: str
    enabled: bool
    passed: bool
    min_pairs_acc: float
    latest_pairs_acc: float | None
    pairs_history: List[Tuple[int, float]] = field(default_factory=list)
    evidence_files: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


def _extract_pairs_acc(report_path: Path) -> float:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    aggregate = payload.get("aggregate", {})
    return float(aggregate.get("pairs_acc", 0.0))


def _collect_periodic_pairs(report_dir: Path) -> tuple[List[Tuple[int, float]], List[str]]:
    history: List[Tuple[int, float]] = []
    evidence: List[str] = []

    for path in sorted(report_dir.glob("real_tuple_eval_epoch_*.json"), key=lambda p: p.name):
        match = _EPOCH_REPORT_RE.search(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        try:
            pairs_acc = _extract_pairs_acc(path)
        except Exception:
            continue
        history.append((epoch, pairs_acc))
        evidence.append(str(path))

    history.sort(key=lambda x: x[0])
    return history, evidence


def _collect_fallback_report(report_path: Path) -> tuple[List[Tuple[int, float]], List[str]]:
    if not report_path.exists():
        return [], []

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    pairs_acc = None

    # final_report.json -> real_tuple_eval.aggregate.pairs_acc
    real_eval = payload.get("real_tuple_eval")
    if isinstance(real_eval, dict):
        pairs_acc = float(real_eval.get("aggregate", {}).get("pairs_acc", 0.0))

    # direct eval report -> aggregate.pairs_acc
    if pairs_acc is None:
        pairs_acc = float(payload.get("aggregate", {}).get("pairs_acc", 0.0))

    return [(1, pairs_acc)], [str(report_path)]


def evaluate_pairs_trend_gate(
    cfg: Dict[str, Any],
    gate_key: str,
    default_enabled: bool,
    default_min_pairs_acc: float = 0.05,
) -> PairsTrendGateResult:
    eval_cfg = cfg.get("evaluation", {})
    gate_cfg = eval_cfg.get(gate_key, {})

    if not isinstance(gate_cfg, dict):
        gate_cfg = {}

    enabled = bool(gate_cfg.get("enabled", default_enabled))
    min_pairs = float(gate_cfg.get("min_pairs_acc", default_min_pairs_acc))

    if not enabled:
        return PairsTrendGateResult(
            gate_key=gate_key,
            enabled=False,
            passed=True,
            min_pairs_acc=min_pairs,
            latest_pairs_acc=None,
        )

    report_dir = Path(str(gate_cfg.get("report_dir", eval_cfg.get("report_dir", "./runs/current/reports"))))
    min_points = int(gate_cfg.get("min_points", 2))
    require_trend = bool(gate_cfg.get("require_non_decreasing_pairs_trend", True))
    trend_eps = float(gate_cfg.get("trend_eps", 1.0e-8))

    history, evidence_files = _collect_periodic_pairs(report_dir)

    source_report = str(gate_cfg.get("source_report", "")).strip()
    if not history:
        fallback_path = Path(source_report) if source_report else report_dir / "final_report.json"
        fallback_history, fallback_evidence = _collect_fallback_report(fallback_path)
        history.extend(fallback_history)
        evidence_files.extend(fallback_evidence)

    issues: List[str] = []
    if not history:
        issues.append(
            "No stage evidence reports found. Expected periodic reports like "
            "real_tuple_eval_epoch_*.json or a valid final_report.json."
        )

    latest_pairs = history[-1][1] if history else None
    if latest_pairs is not None and latest_pairs < min_pairs:
        issues.append(
            f"Latest pairs_acc={latest_pairs:.4f} is below threshold min_pairs_acc={min_pairs:.4f}."
        )

    if require_trend:
        required_points = max(1, min_points)
        if len(history) < required_points:
            issues.append(
                f"Non-decreasing trend check requires at least {required_points} points, got {len(history)}."
            )
        else:
            for idx in range(1, len(history)):
                prev_epoch, prev_val = history[idx - 1]
                cur_epoch, cur_val = history[idx]
                if cur_val + trend_eps < prev_val:
                    issues.append(
                        "pairs_acc trend is decreasing: "
                        f"epoch {prev_epoch} ({prev_val:.4f}) -> epoch {cur_epoch} ({cur_val:.4f})."
                    )
                    break

    return PairsTrendGateResult(
        gate_key=gate_key,
        enabled=True,
        passed=not issues,
        min_pairs_acc=min_pairs,
        latest_pairs_acc=latest_pairs,
        pairs_history=history,
        evidence_files=evidence_files,
        issues=issues,
    )


def format_pairs_trend_gate(result: PairsTrendGateResult, prefix: str) -> str:
    if not result.enabled:
        return f"{prefix} disabled"

    lines = [
        f"{prefix} enabled=true passed={result.passed}",
        f"{prefix} min_pairs_acc={result.min_pairs_acc:.4f}",
        f"{prefix} latest_pairs_acc={result.latest_pairs_acc if result.latest_pairs_acc is not None else 'n/a'}",
    ]

    if result.pairs_history:
        history_str = ", ".join([f"e{ep}:{val:.4f}" for ep, val in result.pairs_history])
        lines.append(f"{prefix} pairs_history={history_str}")
    else:
        lines.append(f"{prefix} pairs_history=<none>")

    if result.evidence_files:
        lines.append(f"{prefix} evidence_files={len(result.evidence_files)}")
        for path in result.evidence_files:
            lines.append(f"{prefix}   - {path}")
    else:
        lines.append(f"{prefix} evidence_files=<none>")

    if result.issues:
        lines.append(f"{prefix} issues:")
        for issue in result.issues:
            lines.append(f"{prefix}   - {issue}")
    else:
        lines.append(f"{prefix} issues: <none>")

    return "\n".join(lines)


def require_pairs_trend_gate(result: PairsTrendGateResult, prefix: str) -> PairsTrendGateResult:
    if result.enabled and not result.passed:
        raise RuntimeError(format_pairs_trend_gate(result, prefix=prefix))
    return result
