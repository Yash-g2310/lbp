#!/usr/bin/env python3
"""Validate and compare training YAML configs for required keys and basic types."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml

REQUIRED_PATHS = [
    ("experiment.name", str),
    ("experiment.seed", int),
    ("hardware.device", str),
    ("hardware.num_workers", int),
    ("data.train_dataset_name", str),
    ("data.train_split", str),
    ("data.val_dataset_name", str),
    ("data.val_split", str),
    ("data.cache_dir", str),
    ("data.batch_size", int),
    ("data.val_batch_size", int),
    ("data.use_precomputed_dino", bool),
    ("architecture.base_channels", int),
    ("architecture.num_sfin", int),
    ("architecture.num_rhag", int),
    ("training.epochs", int),
    ("training.accum_steps", int),
    ("training.learning_rate", (int, float)),
    ("training.grad_clip_norm", (int, float)),
    ("training.checkpoint.dir", str),
    ("logging.use_wandb", bool),
    ("logging.mode", str),
    ("logging.log_to_terminal", bool),
    ("logging.train_log_every_steps", int),
]


def deep_get(payload: Dict[str, Any], dotted_key: str) -> Any:
    cur: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted_key)
        cur = cur[part]
    return cur


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit YAML configs for required schema")
    p.add_argument("--configs", nargs="+", required=True, help="Config paths to validate")
    return p.parse_args()


def audit_one(path: Path) -> bool:
    ok = True
    cfg = load_yaml(path)
    print(f"\n== Auditing {path} ==")
    for key, expected in REQUIRED_PATHS:
        try:
            value = deep_get(cfg, key)
        except KeyError:
            ok = False
            print(f"[FAIL] Missing key: {key}")
            continue

        if isinstance(expected, tuple):
            good = isinstance(value, expected)
            exp_name = "/".join(t.__name__ for t in expected)
        else:
            good = isinstance(value, expected)
            exp_name = expected.__name__

        if not good:
            ok = False
            print(f"[FAIL] {key}: expected {exp_name}, got {type(value).__name__}")

    # Domain sanity checks
    if deep_get(cfg, "training.epochs") <= 0:
        ok = False
        print("[FAIL] training.epochs must be > 0")
    if deep_get(cfg, "data.batch_size") <= 0:
        ok = False
        print("[FAIL] data.batch_size must be > 0")
    if deep_get(cfg, "data.val_batch_size") <= 0:
        ok = False
        print("[FAIL] data.val_batch_size must be > 0")
    if deep_get(cfg, "training.accum_steps") <= 0:
        ok = False
        print("[FAIL] training.accum_steps must be > 0")
    if float(deep_get(cfg, "training.learning_rate")) <= 0:
        ok = False
        print("[FAIL] training.learning_rate must be > 0")
    if float(deep_get(cfg, "training.grad_clip_norm")) <= 0:
        ok = False
        print("[FAIL] training.grad_clip_norm must be > 0")
    if deep_get(cfg, "logging.train_log_every_steps") <= 0:
        ok = False
        print("[FAIL] logging.train_log_every_steps must be > 0")

    if deep_get(cfg, "architecture.base_channels") <= 0:
        ok = False
        print("[FAIL] architecture.base_channels must be > 0")
    if deep_get(cfg, "architecture.num_sfin") <= 0:
        ok = False
        print("[FAIL] architecture.num_sfin must be > 0")
    if deep_get(cfg, "architecture.num_rhag") <= 0:
        ok = False
        print("[FAIL] architecture.num_rhag must be > 0")

    midpoint = float(deep_get(cfg, "training.curriculum.midpoint_fraction"))
    if midpoint < 0.0 or midpoint > 1.0:
        ok = False
        print("[FAIL] training.curriculum.midpoint_fraction must be in [0, 1]")

    if ok:
        print("[OK] Schema and basic sanity checks passed")
    return ok


def main() -> None:
    args = parse_args()
    all_ok = True
    for cfg_path in args.configs:
        ok = audit_one(Path(cfg_path))
        all_ok = all_ok and ok

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
