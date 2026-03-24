#!/usr/bin/env python3
"""Audit dataset splits and optional precomputed feature shard index consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit dataset and feature shard consistency")
    p.add_argument("--config", action="append", required=True, help="Path to one or more YAML configs")
    p.add_argument("--sample-check", type=int, default=3, help="How many index entries to path-check")
    p.add_argument(
        "--fallback-cache-dir",
        type=str,
        default="",
        help="Optional cache dir used if configured cache_dir is inaccessible on this machine",
    )
    return p.parse_args()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def print_split_info(ds_name: str, split: str, cache_dir: str) -> int:
    ds = load_dataset(ds_name, split=split, cache_dir=cache_dir, streaming=False)
    first = ds[0]
    print(f"  - {ds_name}:{split} -> samples={len(ds)} keys={sorted(first.keys())}")
    return len(ds)


def check_index(index_path: Path, split_names: Iterable[str], sample_check: int) -> bool:
    if not index_path.exists():
        print(f"  [FAIL] Precomputed index not found: {index_path}")
        return False

    with index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("samples", {})
    if not isinstance(samples, dict):
        print("  [FAIL] index.json missing 'samples' object")
        return False

    ok = True
    for split in split_names:
        split_map = samples.get(split)
        if not isinstance(split_map, dict):
            print(f"  [FAIL] index.json missing split map for '{split}'")
            ok = False
            continue

        print(f"  - index split '{split}' entries={len(split_map)}")

        checked = 0
        for sample_id, meta in split_map.items():
            shard_path = Path(meta.get("shard_path", ""))
            if not shard_path.exists():
                print(f"  [FAIL] Missing shard path for sample {sample_id}: {shard_path}")
                ok = False
                break
            checked += 1
            if checked >= sample_check:
                break

    if ok:
        print("  [OK] Feature shard index basic checks passed")
    return ok


def main() -> None:
    args = parse_args()
    all_ok = True

    for cfg_path in args.config:
        cfg_file = Path(cfg_path)
        cfg = load_yaml(cfg_file)
        data_cfg = cfg["data"]

        print(f"\n== Shard audit for {cfg_file} ==")
        cache_dir = data_cfg["cache_dir"]
        try:
            train_count = print_split_info(data_cfg["train_dataset_name"], data_cfg["train_split"], cache_dir)
            val_count = print_split_info(data_cfg["val_dataset_name"], data_cfg["val_split"], cache_dir)
        except PermissionError as exc:
            if not args.fallback_cache_dir:
                raise
            print(f"  [warn] cache_dir '{cache_dir}' inaccessible ({exc}). Retrying with fallback cache dir.")
            train_count = print_split_info(
                data_cfg["train_dataset_name"], data_cfg["train_split"], args.fallback_cache_dir
            )
            val_count = print_split_info(
                data_cfg["val_dataset_name"], data_cfg["val_split"], args.fallback_cache_dir
            )

        print(f"  - expected HF shard families can differ by dataset/split; sample counts are canonical")
        print(f"  - observed counts: train={train_count}, val={val_count}")

        if data_cfg.get("use_precomputed_dino", False):
            idx_ok = check_index(
                Path(data_cfg["precomputed_index_path"]),
                (data_cfg["train_split"], data_cfg["val_split"]),
                args.sample_check,
            )
            all_ok = all_ok and idx_ok

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
