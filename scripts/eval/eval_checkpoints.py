#!/usr/bin/env python3
"""Evaluate one or more checkpoints on synthetic and real benchmarks.

This wrapper runs:
- scripts/eval/eval_synth_depth.py (validation synthetic depth)
- scripts/eval/eval_real_tuples.py (real tuple benchmark on requested splits)

It stores per-checkpoint JSON reports and writes an aggregate summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate multiple checkpoints")
    p.add_argument("--config", required=True, help="Path to YAML config")
    p.add_argument(
        "--checkpoints-dir",
        default="",
        help="Directory containing .pth checkpoints (default: training.checkpoint.dir from config)",
    )
    p.add_argument(
        "--pattern",
        default="*.pth",
        help="Glob pattern inside checkpoints-dir (default: *.pth)",
    )
    p.add_argument(
        "--checkpoints",
        default="",
        help="Comma-separated checkpoint paths; if set, overrides --checkpoints-dir/--pattern",
    )
    p.add_argument("--python", default="python", help="Python executable")
    p.add_argument(
        "--real-splits",
        default="",
        help="Comma-separated splits for real tuple evaluation (default from config)",
    )
    p.add_argument(
        "--real-layer-keys",
        default="",
        help="Comma-separated tuple keys; default from config evaluation.real_layer_keys",
    )
    p.add_argument(
        "--target-layer",
        type=int,
        default=-1,
        help="Override model target layer for real eval (default uses config)",
    )
    p.add_argument(
        "--synth-max-batches",
        type=int,
        default=-1,
        help="Override synthetic max batches (default uses config)",
    )
    p.add_argument(
        "--real-max-samples",
        type=int,
        default=-1,
        help="Override real max samples (default uses config)",
    )
    p.add_argument("--skip-synth", action="store_true", help="Skip synthetic depth evaluation")
    p.add_argument("--skip-real", action="store_true", help="Skip real tuple evaluation")
    p.add_argument(
        "--output-dir",
        default="",
        help="Output directory (default: <evaluation.report_dir>/checkpoints_scan)",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if a checkpoint evaluation fails",
    )
    return p.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def resolve_checkpoints(args: argparse.Namespace, cfg: Dict[str, Any]) -> List[Path]:
    if args.checkpoints.strip():
        ckpts = [Path(x.strip()) for x in args.checkpoints.split(",") if x.strip()]
    else:
        ckpt_dir = Path(args.checkpoints_dir.strip()) if args.checkpoints_dir.strip() else Path(
            str(cfg["training"]["checkpoint"]["dir"])
        )
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
        ckpts = sorted(ckpt_dir.glob(args.pattern))

    if not ckpts:
        raise FileNotFoundError("No checkpoints found with the given selection")

    missing = [str(p) for p in ckpts if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing[:5]}")

    return ckpts


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    eval_cfg = cfg.get("evaluation", {})

    checkpoints = resolve_checkpoints(args, cfg)

    base_report_dir = (
        Path(args.output_dir)
        if args.output_dir.strip()
        else Path(str(eval_cfg.get("report_dir", "./runs/current/reports"))) / "checkpoints_scan"
    )
    base_report_dir.mkdir(parents=True, exist_ok=True)

    if args.real_layer_keys.strip():
        real_layer_keys = args.real_layer_keys.strip()
    else:
        keys = eval_cfg.get("real_layer_keys", [eval_cfg.get("real_layer_key", "layer_all")])
        real_layer_keys = ",".join(str(k) for k in keys)

    if args.real_splits.strip():
        real_splits = args.real_splits.strip()
    else:
        splits = eval_cfg.get("real_splits", [eval_cfg.get("real_split", "validation")])
        real_splits = ",".join(str(s) for s in splits)
    target_layer = int(args.target_layer) if args.target_layer > 0 else int(eval_cfg.get("target_layer", 1))
    synth_max_batches = (
        int(args.synth_max_batches)
        if args.synth_max_batches >= 0
        else int(eval_cfg.get("synth_max_batches", 0))
    )
    real_max_samples = (
        int(args.real_max_samples)
        if args.real_max_samples >= 0
        else int(eval_cfg.get("real_max_samples", 0))
    )

    summary: Dict[str, Any] = {
        "config": args.config,
        "num_checkpoints": len(checkpoints),
        "checkpoints": [str(p) for p in checkpoints],
        "real_splits": [s.strip() for s in real_splits.split(",") if s.strip()],
        "real_layer_keys": [k.strip() for k in real_layer_keys.split(",") if k.strip()],
        "results": [],
        "failures": [],
    }

    for ckpt in checkpoints:
        ckpt_name = ckpt.stem
        out_dir = base_report_dir / ckpt_name
        out_dir.mkdir(parents=True, exist_ok=True)

        synth_out = out_dir / "synth_eval.json"
        real_out = out_dir / "real_tuple_eval.json"

        result_entry: Dict[str, Any] = {
            "checkpoint": str(ckpt),
            "status": "ok",
            "outputs": {},
            "metrics": {},
        }

        try:
            if not args.skip_synth:
                run_cmd(
                    [
                        args.python,
                        "scripts/eval/eval_synth_depth.py",
                        "--config",
                        args.config,
                        "--checkpoint",
                        str(ckpt),
                        "--max-batches",
                        str(synth_max_batches),
                        "--output",
                        str(synth_out),
                    ]
                )
                result_entry["outputs"]["synth"] = str(synth_out)
                synth_json = json.loads(synth_out.read_text(encoding="utf-8"))
                result_entry["metrics"]["synth_silog_mean"] = synth_json.get("silog_mean")
                result_entry["metrics"]["synth_abs_rel_layer1"] = synth_json.get("abs_rel_layer1")
                result_entry["metrics"]["synth_abs_rel_layer2"] = synth_json.get("abs_rel_layer2")

            if not args.skip_real:
                run_cmd(
                    [
                        args.python,
                        "scripts/eval/eval_real_tuples.py",
                        "--config",
                        args.config,
                        "--checkpoint",
                        str(ckpt),
                        "--splits",
                        real_splits,
                        "--layer-keys",
                        real_layer_keys,
                        "--target-layer",
                        str(target_layer),
                        "--max-samples",
                        str(real_max_samples),
                        "--output",
                        str(real_out),
                    ]
                )
                result_entry["outputs"]["real"] = str(real_out)
                real_json = json.loads(real_out.read_text(encoding="utf-8"))
                agg = real_json.get("aggregate", {})
                result_entry["metrics"]["real_all_acc"] = agg.get("all_acc")
                result_entry["metrics"]["real_quads_acc"] = agg.get("quads_acc")
                result_entry["metrics"]["official_quadruplet_accuracy"] = real_json.get(
                    "official_quadruplet_accuracy"
                )

        except Exception as exc:
            result_entry["status"] = "failed"
            result_entry["error"] = str(exc)
            summary["failures"].append(
                {
                    "checkpoint": str(ckpt),
                    "error": str(exc),
                }
            )
            if args.stop_on_error:
                summary["results"].append(result_entry)
                summary_path = base_report_dir / "summary.json"
                summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                raise

        summary["results"].append(result_entry)

    summary_path = base_report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[checkpoint-eval] done")
    print(json.dumps(summary, indent=2))
    print(f"[checkpoint-eval] summary: {summary_path}")


if __name__ == "__main__":
    main()
