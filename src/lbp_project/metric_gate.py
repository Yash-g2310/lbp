"""Generic tuple-metric gate evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Tuple


_EPOCH_REPORT_RE = re.compile(r"real_tuple_eval_epoch_(\d+)\.json$")
_SUPPORTED_METRIC_KEYS = {"pairs_acc", "trips_acc", "quads_acc", "all_acc"}


@dataclass
class PairsTrendGateResult:
    gate_key: str
    enabled: bool
    passed: bool
    metric_key: str
    min_pairs_acc: float
    latest_pairs_acc: float | None
    pairs_history: List[Tuple[int, float]] = field(default_factory=list)
    evidence_files: List[str] = field(default_factory=list)
    rejected_evidence: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


def _parse_iso8601(value: str) -> datetime | None:
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _extract_eval_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    real_eval = payload.get("real_tuple_eval")
    if isinstance(real_eval, dict):
        return real_eval
    return payload


def _extract_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    eval_payload = _extract_eval_payload(payload)
    metadata = eval_payload.get("metadata")
    if isinstance(metadata, dict):
        return metadata

    root_metadata = payload.get("metadata")
    if isinstance(root_metadata, dict):
        return root_metadata
    return {}


def _extract_metric(payload: Dict[str, Any], metric_key: str) -> float:
    eval_payload = _extract_eval_payload(payload)
    aggregate = eval_payload.get("aggregate", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    return float(aggregate.get(metric_key, 0.0))


def _extract_metric_total(payload: Dict[str, Any], metric_key: str) -> float | None:
    if not metric_key.endswith("_acc"):
        return None

    total_key = metric_key[:-4] + "_total"
    eval_payload = _extract_eval_payload(payload)
    aggregate = eval_payload.get("aggregate", {})
    if not isinstance(aggregate, dict) or total_key not in aggregate:
        return None
    return float(aggregate.get(total_key, 0.0))


def _extract_missing_layer_tuples(payload: Dict[str, Any]) -> float | None:
    eval_payload = _extract_eval_payload(payload)
    per_eval = eval_payload.get("per_eval")
    if not isinstance(per_eval, dict):
        return None

    found = False
    total = 0.0
    for entry in per_eval.values():
        if not isinstance(entry, dict) or "missing_layer_tuples" not in entry:
            continue
        total += float(entry.get("missing_layer_tuples", 0.0))
        found = True

    if not found:
        return None
    return total


def _validate_report_provenance(
    payload: Dict[str, Any],
    expected_config_sha256: str,
    require_config_sha256_match: bool,
    min_report_timestamp: datetime | None,
    require_report_timestamp: bool,
) -> List[str]:
    issues: List[str] = []
    metadata = _extract_metadata(payload)

    if require_config_sha256_match:
        observed_sha = str(metadata.get("config_sha256", "")).strip()
        if not observed_sha:
            issues.append("missing metadata.config_sha256")
        elif observed_sha != expected_config_sha256:
            issues.append(
                f"metadata.config_sha256 mismatch: observed={observed_sha} expected={expected_config_sha256}"
            )

    if min_report_timestamp is not None:
        observed_raw = str(metadata.get("timestamp_utc", "")).strip()
        observed_ts = _parse_iso8601(observed_raw) if observed_raw else None
        if observed_ts is None:
            if require_report_timestamp:
                issues.append("missing/invalid metadata.timestamp_utc")
        elif observed_ts < min_report_timestamp:
            issues.append(
                "report timestamp precedes current run start: "
                f"observed={observed_raw} required_min={min_report_timestamp.isoformat()}"
            )

    return issues


def _collect_periodic_pairs(
    report_dir: Path,
    metric_key: str,
    expected_config_sha256: str,
    require_config_sha256_match: bool,
    min_report_timestamp: datetime | None,
    require_report_timestamp: bool,
) -> tuple[List[Tuple[int, float]], List[str], List[str]]:
    history: List[Tuple[int, float]] = []
    evidence: List[str] = []
    rejected: List[str] = []

    for path in sorted(report_dir.glob("real_tuple_eval_epoch_*.json"), key=lambda p: p.name):
        match = _EPOCH_REPORT_RE.search(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rejected.append(f"{path}: unreadable JSON ({exc})")
            continue

        provenance_issues = _validate_report_provenance(
            payload,
            expected_config_sha256=expected_config_sha256,
            require_config_sha256_match=require_config_sha256_match,
            min_report_timestamp=min_report_timestamp,
            require_report_timestamp=require_report_timestamp,
        )
        if provenance_issues:
            rejected.append(f"{path}: " + "; ".join(provenance_issues))
            continue

        try:
            metric_value = _extract_metric(payload, metric_key)
        except Exception as exc:
            rejected.append(f"{path}: metric extraction failed for {metric_key} ({exc})")
            continue

        history.append((epoch, metric_value))
        evidence.append(str(path))

    history.sort(key=lambda x: x[0])
    return history, evidence, rejected


def _collect_fallback_report(
    report_path: Path,
    metric_key: str,
    expected_config_sha256: str,
    require_config_sha256_match: bool,
    min_report_timestamp: datetime | None,
    require_report_timestamp: bool,
) -> tuple[List[Tuple[int, float]], List[str], List[str]]:
    if not report_path.exists():
        return [], [], []

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [], [f"{report_path}: unreadable JSON ({exc})"]

    provenance_issues = _validate_report_provenance(
        payload,
        expected_config_sha256=expected_config_sha256,
        require_config_sha256_match=require_config_sha256_match,
        min_report_timestamp=min_report_timestamp,
        require_report_timestamp=require_report_timestamp,
    )
    if provenance_issues:
        return [], [], [f"{report_path}: " + "; ".join(provenance_issues)]

    try:
        metric_value = _extract_metric(payload, metric_key)
    except Exception as exc:
        return [], [], [f"{report_path}: metric extraction failed for {metric_key} ({exc})"]

    return [(1, metric_value)], [str(report_path)], []


def evaluate_pairs_trend_gate(
    cfg: Dict[str, Any],
    gate_key: str,
    default_enabled: bool,
    default_min_pairs_acc: float = 5.0,
) -> PairsTrendGateResult:
    eval_cfg = cfg.get("evaluation", {})
    gate_cfg = eval_cfg.get(gate_key, {})

    if not isinstance(gate_cfg, dict):
        gate_cfg = {}

    enabled = bool(gate_cfg.get("enabled", default_enabled))
    default_metric_key = "quads_acc" if gate_key in {"stage_a_gate", "stage_b_gate"} else "pairs_acc"
    metric_key = str(gate_cfg.get("metric_key", default_metric_key)).strip().lower()
    if metric_key not in _SUPPORTED_METRIC_KEYS:
        raise ValueError(
            f"Unsupported evaluation.{gate_key}.metric_key='{metric_key}'. "
            f"Expected one of {sorted(_SUPPORTED_METRIC_KEYS)}"
        )
    min_pairs = float(gate_cfg.get("min_pairs_acc", default_min_pairs_acc))

    if not enabled:
        return PairsTrendGateResult(
            gate_key=gate_key,
            enabled=False,
            passed=True,
            metric_key=metric_key,
            min_pairs_acc=min_pairs,
            latest_pairs_acc=None,
        )

    report_dir = Path(str(gate_cfg.get("report_dir", eval_cfg.get("report_dir", "./runs/current/reports"))))
    min_points = int(gate_cfg.get("min_points", 2))
    require_trend = bool(gate_cfg.get("require_non_decreasing_pairs_trend", True))
    trend_eps = float(gate_cfg.get("trend_eps", 1.0e-8))

    expected_config_sha256 = str(gate_cfg.get("expected_config_sha256", "")).strip()
    require_config_sha256_match = bool(
        gate_cfg.get("require_config_sha256_match", bool(expected_config_sha256))
    )
    min_report_timestamp_raw = str(gate_cfg.get("min_report_timestamp_utc", "")).strip()
    min_report_timestamp = _parse_iso8601(min_report_timestamp_raw) if min_report_timestamp_raw else None
    require_report_timestamp = bool(
        gate_cfg.get("require_report_timestamp", bool(min_report_timestamp_raw))
    )
    allow_fallback_report = bool(gate_cfg.get("allow_fallback_report", True))

    history, evidence_files, rejected_evidence = _collect_periodic_pairs(
        report_dir,
        metric_key=metric_key,
        expected_config_sha256=expected_config_sha256,
        require_config_sha256_match=require_config_sha256_match,
        min_report_timestamp=min_report_timestamp,
        require_report_timestamp=require_report_timestamp,
    )

    source_report = str(gate_cfg.get("source_report", "")).strip()
    if (not history) and allow_fallback_report:
        fallback_path = Path(source_report) if source_report else report_dir / "final_report.json"
        fallback_history, fallback_evidence, fallback_rejected = _collect_fallback_report(
            fallback_path,
            metric_key=metric_key,
            expected_config_sha256=expected_config_sha256,
            require_config_sha256_match=require_config_sha256_match,
            min_report_timestamp=min_report_timestamp,
            require_report_timestamp=require_report_timestamp,
        )
        history.extend(fallback_history)
        evidence_files.extend(fallback_evidence)
        rejected_evidence.extend(fallback_rejected)

    issues: List[str] = []
    if not history:
        issues.append(
            "No stage evidence reports found. Expected periodic reports like "
            "real_tuple_eval_epoch_*.json or a valid final_report.json."
        )
        if rejected_evidence:
            issues.append(
                "All candidate evidence reports were rejected by provenance checks. "
                f"Rejected={len(rejected_evidence)}"
            )

    latest_pairs = history[-1][1] if history else None
    if latest_pairs is not None and latest_pairs < min_pairs:
        issues.append(
            f"Latest {metric_key}={latest_pairs:.4f} is below threshold min_pairs_acc={min_pairs:.4f}."
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
                        f"{metric_key} trend is decreasing: "
                        f"epoch {prev_epoch} ({prev_val:.4f}) -> epoch {cur_epoch} ({cur_val:.4f})."
                    )
                    break

    min_metric_total = float(gate_cfg.get("min_metric_total", 0.0))
    max_missing_layer_tuples = float(gate_cfg.get("max_missing_layer_tuples", -1.0))
    max_missing_layer_ratio = float(gate_cfg.get("max_missing_layer_ratio", -1.0))
    if history and evidence_files and (
        min_metric_total > 0.0 or max_missing_layer_tuples >= 0.0 or max_missing_layer_ratio >= 0.0
    ):
        latest_report_path = Path(evidence_files[-1])
        try:
            latest_payload = json.loads(latest_report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(
                f"Failed to parse latest evidence report {latest_report_path} for coverage checks: {exc}"
            )
        else:
            metric_total = _extract_metric_total(latest_payload, metric_key)
            if min_metric_total > 0.0:
                if metric_total is None:
                    issues.append(
                        f"Coverage check requested but aggregate total for {metric_key} was not found in latest report."
                    )
                elif metric_total < min_metric_total:
                    issues.append(
                        f"Latest {metric_key[:-4]}_total={metric_total:.1f} is below min_metric_total={min_metric_total:.1f}."
                    )

            missing_layer_tuples = _extract_missing_layer_tuples(latest_payload)
            if max_missing_layer_tuples >= 0.0:
                if missing_layer_tuples is None:
                    issues.append(
                        "max_missing_layer_tuples is configured but latest report does not expose per_eval.missing_layer_tuples"
                    )
                elif missing_layer_tuples > max_missing_layer_tuples:
                    issues.append(
                        "Latest missing_layer_tuples exceeds threshold: "
                        f"{missing_layer_tuples:.1f} > {max_missing_layer_tuples:.1f}"
                    )

            if max_missing_layer_ratio >= 0.0:
                if missing_layer_tuples is None:
                    issues.append(
                        "max_missing_layer_ratio is configured but latest report does not expose per_eval.missing_layer_tuples"
                    )
                elif metric_total is None or metric_total <= 0.0:
                    issues.append(
                        f"Cannot enforce max_missing_layer_ratio without positive {metric_key[:-4]}_total in latest report"
                    )
                else:
                    ratio = missing_layer_tuples / metric_total
                    if ratio > max_missing_layer_ratio:
                        issues.append(
                            "Latest missing_layer_tuples ratio exceeds threshold: "
                            f"{ratio:.4f} > {max_missing_layer_ratio:.4f}"
                        )

    return PairsTrendGateResult(
        gate_key=gate_key,
        enabled=True,
        passed=not issues,
        metric_key=metric_key,
        min_pairs_acc=min_pairs,
        latest_pairs_acc=latest_pairs,
        pairs_history=history,
        evidence_files=evidence_files,
        rejected_evidence=rejected_evidence,
        issues=issues,
    )


def format_pairs_trend_gate(result: PairsTrendGateResult, prefix: str) -> str:
    if not result.enabled:
        return f"{prefix} disabled"

    lines = [
        f"{prefix} enabled=true passed={result.passed}",
        f"{prefix} metric_key={result.metric_key}",
        f"{prefix} min_pairs_acc={result.min_pairs_acc:.4f}",
        f"{prefix} latest_pairs_acc={result.latest_pairs_acc if result.latest_pairs_acc is not None else 'n/a'}",
    ]

    if result.pairs_history:
        history_str = ", ".join([f"e{ep}:{val:.4f}" for ep, val in result.pairs_history])
        lines.append(f"{prefix} metric_history={history_str}")
    else:
        lines.append(f"{prefix} metric_history=<none>")

    if result.evidence_files:
        lines.append(f"{prefix} evidence_files={len(result.evidence_files)}")
        for path in result.evidence_files:
            lines.append(f"{prefix}   - {path}")
    else:
        lines.append(f"{prefix} evidence_files=<none>")

    if result.rejected_evidence:
        lines.append(f"{prefix} rejected_evidence={len(result.rejected_evidence)}")
        for note in result.rejected_evidence:
            lines.append(f"{prefix}   - {note}")
    else:
        lines.append(f"{prefix} rejected_evidence=<none>")

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
