"""Stage-aware runtime policy helpers for Stage A and Stage B workflows."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

STAGE_A = "stage_a"
STAGE_B = "stage_b"
STAGE_AUTO = "auto"
SUPPORTED_STAGE_MODES = {STAGE_A, STAGE_B, STAGE_AUTO}


@dataclass
class StagePolicyResult:
    stage_mode: str
    warnings: List[str] = field(default_factory=list)
    changes: List[str] = field(default_factory=list)


def _normalize_stage_mode(raw_mode: str) -> str:
    mode = str(raw_mode).strip().lower()
    aliases = {
        "a": STAGE_A,
        "stagea": STAGE_A,
        "local": STAGE_A,
        "b": STAGE_B,
        "stageb": STAGE_B,
        "server": STAGE_B,
    }
    mode = aliases.get(mode, mode)
    if mode not in SUPPORTED_STAGE_MODES:
        raise ValueError(
            "Unsupported stage mode '{}'; expected one of {}".format(
                raw_mode,
                sorted(SUPPORTED_STAGE_MODES),
            )
        )
    return mode


def infer_stage_mode(cfg: Dict[str, Any], requested_mode: str = STAGE_AUTO) -> str:
    mode = _normalize_stage_mode(requested_mode)
    if mode != STAGE_AUTO:
        return mode

    experiment_mode = str(cfg.get("experiment", {}).get("stage_mode", STAGE_AUTO))
    mode = _normalize_stage_mode(experiment_mode)
    if mode != STAGE_AUTO:
        return mode

    # Server-like configs use local staging and stricter cluster checks.
    if bool(cfg.get("data", {}).get("require_local_staging", False)):
        return STAGE_B
    return STAGE_A


def _require(condition: bool, message: str, strict: bool, warnings: List[str]) -> None:
    if condition:
        return
    if strict:
        raise ValueError(message)
    warnings.append(message)


def _set(cfg: Dict[str, Any], dotted_key: str, value: Any, changes: List[str]) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for part in parts[:-1]:
        child = cur.get(part)
        if not isinstance(child, dict):
            child = {}
            cur[part] = child
        cur = child
    leaf = parts[-1]
    old = cur.get(leaf)
    if old != value:
        cur[leaf] = value
        changes.append(f"{dotted_key}: {old!r} -> {value!r}")


def apply_stage_policy(
    cfg: Dict[str, Any],
    requested_mode: str = STAGE_AUTO,
    stage_a_epochs: int | None = None,
    stage_b_periodic_eval_every: int = 10,
    strict: bool = False,
) -> Tuple[Dict[str, Any], StagePolicyResult]:
    runtime_cfg = deepcopy(cfg)
    stage_mode = infer_stage_mode(runtime_cfg, requested_mode=requested_mode)
    result = StagePolicyResult(stage_mode=stage_mode)

    train_cfg = runtime_cfg.setdefault("training", {})
    eval_cfg = runtime_cfg.setdefault("evaluation", {})
    data_cfg = runtime_cfg.setdefault("data", {})

    if stage_mode == STAGE_A:
        if stage_a_epochs is not None:
            if int(stage_a_epochs) != 5:
                raise ValueError(f"Stage A requires exactly 5 epochs, got {stage_a_epochs}")

        # Stage-A runtime contract is fixed to exactly 5 epochs.
        _set(runtime_cfg, "training.epochs", 5, result.changes)

        epochs = int(train_cfg.get("epochs", 0))
        _require(
            epochs == 5,
            (
                "Stage A policy expects training.epochs=5. "
                f"Current value: {epochs}."
            ),
            strict,
            result.warnings,
        )

        _set(runtime_cfg, "evaluation.periodic_real_eval_every_epochs", 1, result.changes)
        _set(runtime_cfg, "evaluation.run_after_train", True, result.changes)

        if "periodic_real_max_samples" not in eval_cfg:
            base_max = int(eval_cfg.get("real_max_samples", 100))
            if base_max <= 0:
                base_max = 100
            _set(runtime_cfg, "evaluation.periodic_real_max_samples", min(base_max, 100), result.changes)

        stage_a_gate_cfg = eval_cfg.get("stage_a_gate")
        if not isinstance(stage_a_gate_cfg, dict):
            eval_cfg["stage_a_gate"] = {}
            stage_a_gate_cfg = eval_cfg["stage_a_gate"]
        _set(runtime_cfg, "evaluation.stage_a_gate.enabled", True, result.changes)
        if "min_pairs_acc" not in stage_a_gate_cfg:
            _set(runtime_cfg, "evaluation.stage_a_gate.min_pairs_acc", 0.05, result.changes)
        if "require_non_decreasing_pairs_trend" not in stage_a_gate_cfg:
            _set(
                runtime_cfg,
                "evaluation.stage_a_gate.require_non_decreasing_pairs_trend",
                True,
                result.changes,
            )
        if "min_points" not in stage_a_gate_cfg:
            _set(runtime_cfg, "evaluation.stage_a_gate.min_points", 2, result.changes)
        if "trend_eps" not in stage_a_gate_cfg:
            _set(runtime_cfg, "evaluation.stage_a_gate.trend_eps", 1.0e-8, result.changes)
        if "report_dir" not in stage_a_gate_cfg:
            _set(
                runtime_cfg,
                "evaluation.stage_a_gate.report_dir",
                str(eval_cfg.get("report_dir", "./runs/current/reports")),
                result.changes,
            )

        # Stage-A runs should not require Stage-B promotion evidence.
        _set(runtime_cfg, "evaluation.stage_b_gate.enabled", False, result.changes)

    elif stage_mode == STAGE_B:
        if stage_b_periodic_eval_every < 1:
            raise ValueError(
                "stage_b_periodic_eval_every must be >= 1, got {}".format(stage_b_periodic_eval_every)
            )

        _set(
            runtime_cfg,
            "evaluation.periodic_real_eval_every_epochs",
            int(stage_b_periodic_eval_every),
            result.changes,
        )
        _set(runtime_cfg, "evaluation.run_after_train", True, result.changes)

        # Hard promotion gate defaults: Stage-A tuple evidence must be strong enough.
        gate_cfg = eval_cfg.get("stage_b_gate")
        if not isinstance(gate_cfg, dict):
            eval_cfg["stage_b_gate"] = {}
            gate_cfg = eval_cfg["stage_b_gate"]

        _set(runtime_cfg, "evaluation.stage_b_gate.enabled", True, result.changes)
        if "min_pairs_acc" not in gate_cfg:
            _set(runtime_cfg, "evaluation.stage_b_gate.min_pairs_acc", 0.05, result.changes)
        if "require_non_decreasing_pairs_trend" not in gate_cfg:
            _set(
                runtime_cfg,
                "evaluation.stage_b_gate.require_non_decreasing_pairs_trend",
                True,
                result.changes,
            )
        if "min_points" not in gate_cfg:
            _set(runtime_cfg, "evaluation.stage_b_gate.min_points", 2, result.changes)
        if "trend_eps" not in gate_cfg:
            _set(runtime_cfg, "evaluation.stage_b_gate.trend_eps", 1.0e-8, result.changes)
        if "report_dir" not in gate_cfg:
            _set(
                runtime_cfg,
                "evaluation.stage_b_gate.report_dir",
                str(eval_cfg.get("report_dir", "./runs/current/reports")),
                result.changes,
            )

        _require(
            bool(data_cfg.get("require_local_staging", False)),
            "Stage B policy expects data.require_local_staging=true for server-like runs.",
            strict,
            result.warnings,
        )
        _require(
            bool(data_cfg.get("use_precomputed_dino", False)),
            "Stage B policy expects data.use_precomputed_dino=true.",
            strict,
            result.warnings,
        )

    runtime_cfg.setdefault("experiment", {})["stage_mode"] = stage_mode
    return runtime_cfg, result


def validate_stage_policy(cfg: Dict[str, Any], stage_mode: str, strict: bool = False) -> List[str]:
    mode = infer_stage_mode(cfg, requested_mode=stage_mode)
    warnings: List[str] = []

    train_cfg = cfg.get("training", {})
    eval_cfg = cfg.get("evaluation", {})
    data_cfg = cfg.get("data", {})

    if mode == STAGE_A:
        epochs = int(train_cfg.get("epochs", 0))
        _require(
            epochs == 5,
            f"Stage A policy expects training.epochs=5, got {epochs}",
            strict,
            warnings,
        )
        periodic = int(eval_cfg.get("periodic_real_eval_every_epochs", 0))
        _require(
            periodic == 1,
            "Stage A policy expects evaluation.periodic_real_eval_every_epochs=1",
            strict,
            warnings,
        )
        stage_a_gate_cfg = eval_cfg.get("stage_a_gate", {})
        _require(
            bool(stage_a_gate_cfg.get("enabled", False)),
            "Stage A policy expects evaluation.stage_a_gate.enabled=true",
            strict,
            warnings,
        )

    elif mode == STAGE_B:
        periodic = int(eval_cfg.get("periodic_real_eval_every_epochs", 0))
        _require(
            periodic == 10,
            "Stage B policy expects evaluation.periodic_real_eval_every_epochs=10",
            strict,
            warnings,
        )
        gate_cfg = eval_cfg.get("stage_b_gate", {})
        _require(
            bool(gate_cfg.get("enabled", False)),
            "Stage B policy expects evaluation.stage_b_gate.enabled=true",
            strict,
            warnings,
        )
        _require(
            bool(data_cfg.get("require_local_staging", False)),
            "Stage B policy expects data.require_local_staging=true",
            strict,
            warnings,
        )

    return warnings


def format_stage_policy_summary(result: StagePolicyResult, prefix: str = "[stage-policy]") -> str:
    lines = [f"{prefix} mode={result.stage_mode}"]
    if result.changes:
        lines.append(f"{prefix} applied changes:")
        for change in result.changes:
            lines.append(f"{prefix}   - {change}")
    else:
        lines.append(f"{prefix} applied changes: <none>")

    if result.warnings:
        lines.append(f"{prefix} warnings:")
        for warning in result.warnings:
            lines.append(f"{prefix}   - {warning}")
    else:
        lines.append(f"{prefix} warnings: <none>")

    return "\n".join(lines)
