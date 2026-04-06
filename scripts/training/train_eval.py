#!/usr/bin/env python3
"""Run training followed by optional synthetic and real benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and run post-training evaluations")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument("--python", default="python", help="Python executable")
    p.add_argument("--skip-train", action="store_true", help="Skip training and only evaluate")
    return p.parse_args()


def run_cmd(cmd: list[str]) -> None:
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    eval_cfg = cfg.get("evaluation", {})

    run_after_train = bool(eval_cfg.get("run_after_train", False))
    run_synth = bool(eval_cfg.get("run_synth_validation", True))
    run_real = bool(eval_cfg.get("run_real_tuple_eval", True))

    if not args.skip_train:
        run_cmd([args.python, "train.py", "--config", args.config])

    if not run_after_train:
        print("[info] evaluation.run_after_train=false; skipping post-train evaluation")
        return

    ckpt_cfg = cfg["training"]["checkpoint"]
    ckpt_dir = Path(ckpt_cfg["dir"])
    ckpt_name = str(eval_cfg.get("checkpoint_name", ckpt_cfg["best_name"]))
    checkpoint = ckpt_dir / ckpt_name
    if not checkpoint.exists():
        fallback = ckpt_dir / ckpt_cfg["latest_name"]
        if fallback.exists():
            checkpoint = fallback
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint} or fallback {fallback}")

    report_dir = Path(str(eval_cfg.get("report_dir", "./runs/current/reports")))
    report_dir.mkdir(parents=True, exist_ok=True)

    synth_report = report_dir / "synth_eval.json"
    real_report = report_dir / "real_tuple_eval.json"

    if run_synth:
        run_cmd(
            [
                args.python,
                "scripts/eval/eval_synth_depth.py",
                "--config",
                args.config,
                "--checkpoint",
                str(checkpoint),
                "--max-batches",
                str(int(eval_cfg.get("synth_max_batches", 0))),
                "--output",
                str(synth_report),
            ]
        )

    if run_real:
        splits = eval_cfg.get("real_splits", [eval_cfg.get("real_split", "validation")])
        layer_keys = eval_cfg.get("real_layer_keys", [eval_cfg.get("real_layer_key", "layer_all")])
        split_arg = ",".join(str(s) for s in splits)
        layer_key_arg = ",".join(str(k) for k in layer_keys)
        run_cmd(
            [
                args.python,
                "scripts/eval/eval_real_tuples.py",
                "--config",
                args.config,
                "--checkpoint",
                str(checkpoint),
                "--splits",
                split_arg,
                "--layer-keys",
                layer_key_arg,
                "--target-layer",
                str(int(eval_cfg.get("target_layer", 1))),
                "--max-samples",
                str(int(eval_cfg.get("real_max_samples", 0))),
                "--output",
                str(real_report),
            ]
        )

    final_summary: Dict[str, Any] = {"checkpoint_used": str(checkpoint)}
    if synth_report.exists():
        final_summary["synth_eval"] = json.loads(synth_report.read_text(encoding="utf-8"))
    if real_report.exists():
        final_summary["real_tuple_eval"] = json.loads(real_report.read_text(encoding="utf-8"))

    final_path = report_dir / "final_report.json"
    final_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")

    print("[final-summary]")
    print(json.dumps(final_summary, indent=2))
    print(f"[done] wrote {final_path}")


if __name__ == "__main__":
    main()
