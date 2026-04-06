"""Unified CLI for local and Slurm workflows."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(cmd: Sequence[str], env: dict[str, str] | None = None) -> None:
    print(f"[cmd] {' '.join(cmd)}", flush=True)
    subprocess.run(list(cmd), check=True, cwd=_project_root(), env=env)


def _default_python() -> str:
    return sys.executable


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LBP project unified CLI")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run training")
    train.add_argument("--config", default="configs/local/dev.yaml", help="Path to config")
    train.add_argument("--python", default=_default_python(), help="Python executable")

    train_eval = sub.add_parser("train-eval", help="Run train then configured post-eval")
    train_eval.add_argument("--config", default="configs/local/dev.yaml", help="Path to config")
    train_eval.add_argument("--python", default=_default_python(), help="Python executable")
    train_eval.add_argument("--skip-train", action="store_true", help="Skip training stage")

    quickcheck = sub.add_parser("quickcheck", help="Run full quickcheck pipeline")
    quickcheck.add_argument("--config", default="configs/local/quickcheck.yaml", help="Path to config")
    quickcheck.add_argument("--python", default=_default_python(), help="Python executable")

    eval_real = sub.add_parser("eval-real", help="Run real tuple evaluation")
    eval_real.add_argument("--config", required=True, help="Path to config")
    eval_real.add_argument("--checkpoint", required=True, help="Checkpoint path")
    eval_real.add_argument("--python", default=_default_python(), help="Python executable")

    eval_checkpoints = sub.add_parser("eval-checkpoints", help="Evaluate many checkpoints")
    eval_checkpoints.add_argument("--config", required=True, help="Path to config")
    eval_checkpoints.add_argument("--python", default=_default_python(), help="Python executable")

    return p


def main() -> None:
    parser = _build_parser()
    args, extra = parser.parse_known_args()

    if args.command == "train":
        _run([args.python, "train.py", "--config", args.config, *extra])
        return

    if args.command == "train-eval":
        cmd = [args.python, "scripts/training/train_eval.py", "--config", args.config, "--python", args.python]
        if args.skip_train:
            cmd.append("--skip-train")
        cmd.extend(extra)
        _run(cmd)
        return

    if args.command == "quickcheck":
        cmd = ["bash", "scripts/quickcheck/run.sh", args.config]
        env = dict(os.environ)
        env["PYTHON_BIN"] = args.python
        _run(cmd, env=env)
        return

    if args.command == "eval-real":
        cmd = [
            args.python,
            "scripts/eval/eval_real_tuples.py",
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            *extra,
        ]
        _run(cmd)
        return

    if args.command == "eval-checkpoints":
        cmd = [
            args.python,
            "scripts/eval/eval_checkpoints.py",
            "--config",
            args.config,
            "--python",
            args.python,
            *extra,
        ]
        _run(cmd)
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
