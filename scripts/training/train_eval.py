#!/usr/bin/env python3
"""Run training followed by optional synthetic and real benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml
from lbp_project.config.stage_policy import apply_stage_policy, format_stage_policy_summary
from lbp_project.metric_gate import PairsTrendGateResult
from lbp_project.stage_a_gate import evaluate_stage_a_gate, format_stage_a_gate
from lbp_project.stage_gate import evaluate_stage_b_gate, format_stage_b_gate
from lbp_project.training.ablation import ablation_plan_payload, format_ablation_plan, resolve_ablation_plan
from lbp_project.utils.run_manifest import build_run_manifest, config_sha256, write_manifest


DEFAULT_QUICKCHECK_FIXTURE_CHECKPOINT = (
    PROJECT_ROOT / "runs" / "current" / "quickcheck" / "checkpoints" / "quickcheck" / "sanity_roundtrip.pth"
)
DEFAULT_QUICKCHECK_FIXTURE_REPORT = (
    PROJECT_ROOT / "runs" / "current" / "quickcheck" / "reports" / "real_tuple_eval_quickcheck.json"
)
STAGE_B_FINAL_REAL_SPLITS = ["validation", "test"]
STAGE_B_FINAL_REAL_LAYER_KEYS = ["layer_all", "layer_first"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and run post-training evaluations")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--python", default="python", help="Python executable")
    p.add_argument("--skip-train", action="store_true", help="Skip training and only evaluate")
    p.add_argument(
        "--stage-mode",
        default="auto",
        help="Stage policy mode: auto, stage_a/local, or stage_b/server",
    )
    p.add_argument(
        "--stage-a-epochs",
        type=int,
        default=5,
        help="Epoch count override for Stage A policy workflows (must be 5)",
    )
    p.add_argument(
        "--stage-b-periodic-eval-every",
        type=int,
        default=10,
        help="Periodic real-eval cadence for Stage B policy workflows",
    )
    p.add_argument(
        "--strict-stage-policy",
        action="store_true",
        help="Fail if stage-policy requirements are violated instead of warning",
    )
    p.add_argument(
        "--runtime-config-path",
        default="",
        help="Optional output path for generated stage-policy runtime config",
    )
    p.add_argument(
        "--allow-stage-skip-train",
        action="store_true",
        help="Allow --skip-train for stage_a/stage_b workflows (disabled by default for evidence integrity)",
    )
    return p.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _write_runtime_config(cfg: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def _is_stage_a_request(raw_mode: str) -> bool:
    mode = str(raw_mode).strip().lower()
    return mode in {"a", "stagea", "stage_a", "local"}


def _gate_payload(result: PairsTrendGateResult | None) -> Dict[str, Any] | None:
    if result is None:
        return None
    return {
        "enabled": bool(result.enabled),
        "passed": bool(result.passed),
        "metric_key": str(result.metric_key),
        "min_pairs_acc": float(result.min_pairs_acc),
        "latest_pairs_acc": result.latest_pairs_acc,
        "min_metric_threshold": float(result.min_pairs_acc),
        "latest_metric": result.latest_pairs_acc,
        "pairs_history": [[int(epoch), float(value)] for epoch, value in result.pairs_history],
        "metric_history": [[int(epoch), float(value)] for epoch, value in result.pairs_history],
        "evidence_files": list(result.evidence_files),
        "rejected_evidence": list(result.rejected_evidence),
        "issues": list(result.issues),
    }


def _read_json_if_exists(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _format_stage_b_gate_failure(result: PairsTrendGateResult, runtime_cfg: Dict[str, Any]) -> str:
    base = format_stage_b_gate(result)
    eval_cfg = runtime_cfg.get("evaluation", {})
    gate_cfg = eval_cfg.get("stage_b_gate", {})
    if not isinstance(gate_cfg, dict):
        gate_cfg = {}
    report_dir = Path(str(gate_cfg.get("report_dir", eval_cfg.get("report_dir", "./runs/current/reports"))))
    source_report = str(gate_cfg.get("source_report", report_dir / "final_report.json"))

    hints = [
        "[stage-b-gate][hint] strict mode is active; promotion evidence is mandatory before Stage-B orchestration.",
        f"[stage-b-gate][hint] expected evidence directory: {report_dir}",
        f"[stage-b-gate][hint] expected fallback source report: {source_report}",
    ]
    if any("No stage evidence reports found" in issue for issue in result.issues):
        hints.append(
            "[stage-b-gate][hint] to emit fixture evidence using existing quickcheck pipeline, run: "
            "EMIT_STAGE_GATE_FIXTURE=1 python cli.py quickcheck --config configs/server/quickcheck.yaml"
        )
        hints.append(
            f"[stage-b-gate][hint] quickcheck fixture source report: {DEFAULT_QUICKCHECK_FIXTURE_REPORT}"
        )
    return base + "\n" + "\n".join(hints)


def _resolve_checkpoint(
    runtime_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    skip_train: bool,
) -> tuple[Path, str]:
    ckpt_cfg = runtime_cfg["training"]["checkpoint"]
    ckpt_dir = Path(str(ckpt_cfg["dir"]))
    ckpt_name = str(eval_cfg.get("checkpoint_name", ckpt_cfg["best_name"]))

    candidates: list[tuple[str, Path]] = [
        ("configured_checkpoint", ckpt_dir / ckpt_name),
        ("configured_latest", ckpt_dir / str(ckpt_cfg["latest_name"])),
    ]

    if skip_train:
        fixture_cfg = str(eval_cfg.get("fixture_checkpoint", "")).strip()
        if fixture_cfg:
            candidates.append(("fixture_checkpoint", Path(fixture_cfg)))
        candidates.append(("quickcheck_fixture", DEFAULT_QUICKCHECK_FIXTURE_CHECKPOINT))

    for source, path in candidates:
        if path.exists():
            return path, source

    candidate_lines = "\n".join([f"  - {source}: {path}" for source, path in candidates])
    quickcheck_hint = (
        "Generate a local fixture checkpoint via existing quickcheck pipeline: "
        "python cli.py quickcheck --config configs/local/quickcheck.yaml"
    )
    raise FileNotFoundError(
        "No usable checkpoint found for evaluation.\n"
        f"Candidates checked:\n{candidate_lines}\n"
        f"{quickcheck_hint}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    runtime_cfg, stage_result = apply_stage_policy(
        cfg,
        requested_mode=args.stage_mode,
        stage_a_epochs=args.stage_a_epochs if _is_stage_a_request(args.stage_mode) else None,
        stage_b_periodic_eval_every=int(args.stage_b_periodic_eval_every),
        strict=bool(args.strict_stage_policy),
    )
    ablation_plan = resolve_ablation_plan(runtime_cfg)
    ablation_payload = ablation_plan_payload(ablation_plan)

    if args.runtime_config_path.strip():
        runtime_cfg_path = Path(args.runtime_config_path)
    else:
        runtime_cfg_path = (
            PROJECT_ROOT
            / "runs"
            / "current"
            / "generated"
            / f"{stage_result.stage_mode}_{Path(args.config).stem}_runtime.yaml"
        )
    _write_runtime_config(runtime_cfg, runtime_cfg_path)

    print(format_stage_policy_summary(stage_result, prefix="[stage]"), flush=True)
    print(format_ablation_plan(ablation_plan, prefix="[stage-ablation]"), flush=True)
    print(f"[stage] runtime_config={runtime_cfg_path}", flush=True)

    if args.skip_train and stage_result.stage_mode in {"stage_a", "stage_b"} and not bool(args.allow_stage_skip_train):
        raise RuntimeError(
            "--skip-train is disabled for stage_a/stage_b workflows by default to prevent stale-evidence promotion. "
            "If you explicitly need eval-only stage checks, pass --allow-stage-skip-train."
        )

    stage_b_gate_result: PairsTrendGateResult | None = None
    if stage_result.stage_mode == "stage_b":
        stage_b_gate_result = evaluate_stage_b_gate(runtime_cfg)
        print(format_stage_b_gate(stage_b_gate_result), flush=True)
        if stage_b_gate_result is not None and stage_b_gate_result.enabled and not stage_b_gate_result.passed:
            raise RuntimeError(_format_stage_b_gate_failure(stage_b_gate_result, runtime_cfg))

    eval_cfg = runtime_cfg.get("evaluation", {})
    report_dir = Path(str(eval_cfg.get("report_dir", "./runs/current/reports")))
    stage_b_gate_payload = _gate_payload(stage_b_gate_result)
    start_manifest = build_run_manifest(
        runtime_cfg,
        config_path=str(runtime_cfg_path),
        stage_mode=stage_result.stage_mode,
        project_root=PROJECT_ROOT,
        extra={
            "skip_train": bool(args.skip_train),
            "ablation": ablation_payload,
            "stage_b_gate_pre": stage_b_gate_payload,
        },
    )
    start_manifest_path = write_manifest(
        report_dir,
        f"{stage_result.stage_mode}_run_manifest_start.json",
        start_manifest,
    )
    print(f"[stage] start_manifest={start_manifest_path}", flush=True)

    run_after_train = bool(eval_cfg.get("run_after_train", False))
    run_synth = bool(eval_cfg.get("run_synth_validation", True))
    run_real = bool(eval_cfg.get("run_real_tuple_eval", True))

    if not args.skip_train:
        run_cmd([args.python, "train.py", "--config", str(runtime_cfg_path)])

    if not run_after_train:
        print("[info] evaluation.run_after_train=false; skipping post-train evaluation")
        return

    checkpoint, checkpoint_source = _resolve_checkpoint(runtime_cfg, eval_cfg, skip_train=bool(args.skip_train))
    print(f"[stage] evaluation_checkpoint={checkpoint} source={checkpoint_source}", flush=True)

    report_dir.mkdir(parents=True, exist_ok=True)

    synth_report = report_dir / "synth_eval.json"
    real_report = report_dir / "real_tuple_eval.json"
    stage_b_state_path = report_dir / "stage_b_runtime_state.json"
    stage_b_terminal_real_report = report_dir / "real_tuple_eval_terminal_full.json"
    stage_b_state_payload = _read_json_if_exists(stage_b_state_path)
    stage_b_terminal_real_payload = _read_json_if_exists(stage_b_terminal_real_report)

    if stage_result.stage_mode == "stage_b" and stage_b_terminal_real_payload is not None:
        print(
            "[stage-b] found mandatory terminal full real-eval artifact from training runtime; "
            f"reusing {stage_b_terminal_real_report}",
            flush=True,
        )

    if run_synth:
        run_cmd(
            [
                args.python,
                "scripts/eval/eval_synth_depth.py",
                "--config",
                str(runtime_cfg_path),
                "--checkpoint",
                str(checkpoint),
                "--max-batches",
                str(int(eval_cfg.get("synth_max_batches", 0))),
                "--output",
                str(synth_report),
            ]
        )

    should_run_real_eval = run_real and not (
        stage_result.stage_mode == "stage_b" and stage_b_terminal_real_payload is not None
    )

    if should_run_real_eval:
        if stage_result.stage_mode == "stage_b":
            splits = STAGE_B_FINAL_REAL_SPLITS
            layer_keys = STAGE_B_FINAL_REAL_LAYER_KEYS
            real_max_samples = 0
            print(
                "[stage-b] enforcing strict terminal full real-eval contract in train-eval fallback path",
                flush=True,
            )
        else:
            splits = eval_cfg.get("real_splits", [eval_cfg.get("real_split", "validation")])
            layer_keys = eval_cfg.get("real_layer_keys", [eval_cfg.get("real_layer_key", "layer_all")])
            real_max_samples = int(eval_cfg.get("real_max_samples", 0))

        split_arg = ",".join(str(s) for s in splits)
        layer_key_arg = ",".join(str(k) for k in layer_keys)
        run_cmd(
            [
                args.python,
                "scripts/eval/eval_real_tuples.py",
                "--config",
                str(runtime_cfg_path),
                "--checkpoint",
                str(checkpoint),
                "--splits",
                split_arg,
                "--layer-keys",
                layer_key_arg,
                "--target-layer",
                str(int(eval_cfg.get("target_layer", 1))),
                "--max-samples",
                str(real_max_samples),
                "--output",
                str(real_report),
            ]
        )

    if stage_result.stage_mode == "stage_b":
        # Re-read after potential fallback run.
        stage_b_state_payload = _read_json_if_exists(stage_b_state_path)
        stage_b_terminal_real_payload = _read_json_if_exists(stage_b_terminal_real_report)
        if stage_b_terminal_real_payload is None and real_report.exists():
            stage_b_terminal_real_payload = _read_json_if_exists(real_report)
        if stage_b_terminal_real_payload is None:
            raise RuntimeError(
                "Stage B requires terminal full real evaluation artifact, but none was found. "
                f"Expected one of: {stage_b_terminal_real_report} or {real_report}"
            )

    stage_a_gate_result: PairsTrendGateResult | None = None
    if stage_result.stage_mode == "stage_a":
        stage_a_gate_cfg = eval_cfg.get("stage_a_gate")
        if not isinstance(stage_a_gate_cfg, dict):
            eval_cfg["stage_a_gate"] = {}
            stage_a_gate_cfg = eval_cfg["stage_a_gate"]

        stage_a_gate_cfg.setdefault("metric_key", "quads_acc")
        stage_a_gate_cfg["expected_config_sha256"] = config_sha256(runtime_cfg)
        stage_a_gate_cfg["require_config_sha256_match"] = True
        stage_a_gate_cfg["min_report_timestamp_utc"] = str(start_manifest.get("timestamp_utc", ""))
        stage_a_gate_cfg["require_report_timestamp"] = True
        stage_a_gate_cfg.setdefault("allow_fallback_report", False)

        stage_a_gate_result = evaluate_stage_a_gate(runtime_cfg)
        print(format_stage_a_gate(stage_a_gate_result), flush=True)
    stage_a_gate_payload = _gate_payload(stage_a_gate_result)

    final_summary: Dict[str, Any] = {
        "checkpoint_used": str(checkpoint),
        "checkpoint_source": checkpoint_source,
        "stage_mode": stage_result.stage_mode,
        "runtime_config": str(runtime_cfg_path),
        "stage_policy_warnings": stage_result.warnings,
        "stage_policy_changes": stage_result.changes,
        "ablation": ablation_payload,
        "stage_a_gate": stage_a_gate_payload,
        "stage_b_gate": stage_b_gate_payload,
        "start_manifest": str(start_manifest_path),
    }
    if synth_report.exists():
        final_summary["synth_eval"] = json.loads(synth_report.read_text(encoding="utf-8"))
    if stage_result.stage_mode == "stage_b" and stage_b_terminal_real_payload is not None:
        final_summary["real_tuple_eval"] = stage_b_terminal_real_payload
    elif real_report.exists():
        final_summary["real_tuple_eval"] = json.loads(real_report.read_text(encoding="utf-8"))
    if stage_b_state_payload is not None:
        final_summary["stage_b_runtime_state"] = stage_b_state_payload

    final_path = report_dir / "final_report.json"
    final_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    if stage_a_gate_result is not None and stage_a_gate_result.enabled and not stage_a_gate_result.passed:
        raise RuntimeError(format_stage_a_gate(stage_a_gate_result))

    done_manifest = build_run_manifest(
        runtime_cfg,
        config_path=str(runtime_cfg_path),
        stage_mode=stage_result.stage_mode,
        project_root=PROJECT_ROOT,
        extra={
            "final_report": str(final_path),
            "checkpoint_used": str(checkpoint),
        },
    )
    done_manifest_path = write_manifest(
        report_dir,
        f"{stage_result.stage_mode}_run_manifest_done.json",
        done_manifest,
    )
    final_summary["done_manifest"] = str(done_manifest_path)
    final_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    print("[final-summary]")
    print(json.dumps(final_summary, indent=2))
    print(f"[done] wrote {final_path}")


if __name__ == "__main__":
    main()
