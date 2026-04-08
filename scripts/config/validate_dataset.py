#!/usr/bin/env python3
"""Audit dataset splits and optional precomputed feature shard index consistency."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml
from lbp_project.data.hf_loading import load_dataset_split_with_policy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit dataset and feature shard consistency")
    p.add_argument("--config", action="append", required=True, help="Path to one or more YAML configs")
    p.add_argument("--sample-check", type=int, default=3, help="How many index entries to path-check")
    p.add_argument(
        "--verify-index-coverage",
        action="store_true",
        help="Validate every expected sample id exists in index split and every index shard path exists",
    )
    p.add_argument(
        "--max-missing-report",
        type=int,
        default=20,
        help="Max missing/extra sample IDs to print per split during strict coverage checks",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=2000,
        help="Emit progress every N samples while collecting/checking strict index coverage",
    )
    p.add_argument(
        "--fallback-cache-dir",
        type=str,
        default="",
        help="Optional cache dir used if configured cache_dir is inaccessible on this machine",
    )
    return p.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def get_sample_id(sample: Dict[str, Any], index: int) -> str:
    key = sample.get("__key__")
    if key is not None:
        return str(key)
    return str(index)


def _to_int_set(values: Set[str]) -> Tuple[bool, Set[int]]:
    ints: Set[int] = set()
    try:
        for v in values:
            ints.add(int(v))
    except ValueError:
        return False, set()
    return True, ints


def _is_dense_range(values: Set[int]) -> bool:
    if not values:
        return False
    lo = min(values)
    hi = max(values)
    return len(values) == (hi - lo + 1)


def _is_equivalent_numeric_namespace(expected_keys: Set[str], index_keys: Set[str]) -> bool:
    """
    Accept equivalent coverage when two split key spaces differ only by namespace.

    Example accepted case:
    - expected keys: 14800..15299
    - index keys:    0..499
    """
    if len(expected_keys) != len(index_keys):
        return False

    expected_ok, expected_ints = _to_int_set(expected_keys)
    index_ok, index_ints = _to_int_set(index_keys)
    if not expected_ok or not index_ok:
        return False

    if not _is_dense_range(expected_ints) or not _is_dense_range(index_ints):
        return False

    n = len(expected_ints)
    return index_ints == set(range(n))


def print_split_info(
    ds_name: str,
    split: str,
    cache_dir: str,
    allow_downloads: bool,
    allow_cache_repair: bool,
    allow_partial_local_shards: bool,
    partial_local_min_shards: int,
    collect_ids: bool = False,
    progress_every: int = 2000,
) -> Tuple[int, Set[str]]:
    ds = load_dataset_split_with_policy(
        ds_name,
        split,
        cache_dir=cache_dir,
        allow_downloads=allow_downloads,
        allow_cache_repair=allow_cache_repair,
        allow_partial_local_shards=allow_partial_local_shards,
        partial_local_min_shards=partial_local_min_shards,
        log_prefix="[validate-dataset]",
    )
    first = ds[0]
    log(f"  - {ds_name}:{split} -> samples={len(ds)} keys={sorted(first.keys())}")
    if not collect_ids:
        return len(ds), set()

    ids: Set[str] = set()
    total = len(ds)
    log(f"  - collecting sample ids for split '{split}' (total={total})")
    for i, sample in enumerate(ds):
        ids.add(get_sample_id(sample, i))
        if progress_every > 0 and (i + 1) % progress_every == 0:
            log(f"    [progress] split '{split}' collected {i + 1}/{total} ids")
    log(f"  - collected ids for split '{split}': {len(ids)}")
    return len(ds), ids


def check_index(
    index_path: Path,
    split_names: Iterable[str],
    sample_check: int,
    expected_counts: Dict[str, int],
    expected_ids: Dict[str, Set[str]],
    verify_index_coverage: bool,
    max_missing_report: int,
    progress_every: int,
) -> bool:
    if not index_path.exists():
        log(f"  [FAIL] Precomputed index not found: {index_path}")
        return False

    with index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    samples = payload.get("samples", {})
    if not isinstance(samples, dict):
        log("  [FAIL] index.json missing 'samples' object")
        return False

    ok = True
    for split in split_names:
        split_map = samples.get(split)
        if not isinstance(split_map, dict):
            log(f"  [FAIL] index.json missing split map for '{split}'")
            ok = False
            continue

        log(f"  - index split '{split}' entries={len(split_map)}")

        expected_count = expected_counts.get(split)
        if expected_count is not None and len(split_map) != expected_count:
            log(
                f"  [FAIL] index split '{split}' entries ({len(split_map)}) != dataset sample count ({expected_count})"
            )
            ok = False

        index_keys = set(split_map.keys())
        if verify_index_coverage and split in expected_ids:
            missing = expected_ids[split] - index_keys
            extra = index_keys - expected_ids[split]
            if missing or extra:
                if _is_equivalent_numeric_namespace(expected_ids[split], index_keys):
                    log(
                        f"  [warn] split '{split}' sample-id namespace differs (dataset keys vs index-local keys), "
                        "but coverage counts are equivalent"
                    )
                else:
                    if missing:
                        sample_missing = sorted(missing)[:max_missing_report]
                        log(
                            f"  [FAIL] split '{split}' missing {len(missing)} sample ids in index. "
                            f"examples={sample_missing}"
                        )
                        ok = False
                    if extra:
                        sample_extra = sorted(extra)[:max_missing_report]
                        log(
                            f"  [FAIL] split '{split}' has {len(extra)} unexpected sample ids in index. "
                            f"examples={sample_extra}"
                        )
                        ok = False

        checked = 0
        if verify_index_coverage:
            log(f"  - verifying shard paths for split '{split}' (entries={len(split_map)})")
        for sample_id, meta in split_map.items():
            shard_path = Path(meta.get("shard_path", ""))
            if not shard_path.exists():
                log(f"  [FAIL] Missing shard path for sample {sample_id}: {shard_path}")
                ok = False
                if not verify_index_coverage:
                    break
            checked += 1
            if verify_index_coverage and progress_every > 0 and checked % progress_every == 0:
                log(f"    [progress] split '{split}' verified {checked}/{len(split_map)} shard paths")
            if not verify_index_coverage and checked >= sample_check:
                break

    if ok:
        log("  [OK] Feature shard index basic checks passed")
    return ok


def main() -> None:
    args = parse_args()
    all_ok = True

    for cfg_path in args.config:
        cfg_file = Path(cfg_path)
        cfg = load_yaml(cfg_file)
        data_cfg = cfg["data"]
        allow_downloads = bool(data_cfg.get("allow_hf_downloads", True))
        allow_partial_local_shards = bool(data_cfg.get("allow_partial_local_shards", False))
        partial_local_min_shards = int(data_cfg.get("partial_local_shards_min_per_split", 1))
        allow_cache_repair = bool(
            data_cfg.get(
                "repair_hf_cache_once",
                allow_downloads and not allow_partial_local_shards,
            )
        )

        log(f"\n== Shard audit for {cfg_file} ==")
        cache_dir = data_cfg["cache_dir"]
        try:
            train_count, train_ids = print_split_info(
                data_cfg["train_dataset_name"],
                data_cfg["train_split"],
                cache_dir,
                allow_downloads,
                allow_cache_repair,
                allow_partial_local_shards,
                partial_local_min_shards,
                collect_ids=args.verify_index_coverage,
                progress_every=args.progress_every,
            )
            val_count, val_ids = print_split_info(
                data_cfg["val_dataset_name"],
                data_cfg["val_split"],
                cache_dir,
                allow_downloads,
                allow_cache_repair,
                allow_partial_local_shards,
                partial_local_min_shards,
                collect_ids=args.verify_index_coverage,
                progress_every=args.progress_every,
            )
        except PermissionError as exc:
            if not args.fallback_cache_dir:
                raise
            log(f"  [warn] cache_dir '{cache_dir}' inaccessible ({exc}). Retrying with fallback cache dir.")
            train_count, train_ids = print_split_info(
                data_cfg["train_dataset_name"],
                data_cfg["train_split"],
                args.fallback_cache_dir,
                allow_downloads,
                allow_cache_repair,
                allow_partial_local_shards,
                partial_local_min_shards,
                collect_ids=args.verify_index_coverage,
                progress_every=args.progress_every,
            )
            val_count, val_ids = print_split_info(
                data_cfg["val_dataset_name"],
                data_cfg["val_split"],
                args.fallback_cache_dir,
                allow_downloads,
                allow_cache_repair,
                allow_partial_local_shards,
                partial_local_min_shards,
                collect_ids=args.verify_index_coverage,
                progress_every=args.progress_every,
            )

        log("  - expected HF shard families can differ by dataset/split; sample counts are canonical")
        log(f"  - observed counts: train={train_count}, val={val_count}")

        if data_cfg.get("use_precomputed_dino", False):
            idx_ok = check_index(
                Path(data_cfg["precomputed_index_path"]),
                (data_cfg["train_split"], data_cfg["val_split"]),
                args.sample_check,
                {
                    data_cfg["train_split"]: train_count,
                    data_cfg["val_split"]: val_count,
                },
                {
                    data_cfg["train_split"]: train_ids,
                    data_cfg["val_split"]: val_ids,
                },
                args.verify_index_coverage,
                args.max_missing_report,
                args.progress_every,
            )
            all_ok = all_ok and idx_ok

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
