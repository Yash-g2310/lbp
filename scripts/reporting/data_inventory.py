#!/usr/bin/env python3
"""Build a data inventory report from project configs and reachable storage roots."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict YAML payload in {path}, got {type(data).__name__}")
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a data inventory report from one or more configs")
    p.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/local/dev.yaml",
            "configs/local/quickcheck.yaml",
            "configs/server/default.yaml",
            "configs/server/quickcheck.yaml",
        ],
        help="Config files to inspect",
    )
    p.add_argument(
        "--output",
        default="docs/generated/data_inventory_report.json",
        help="Output JSON report path",
    )
    p.add_argument(
        "--max-files-per-root",
        type=int,
        default=200000,
        help="Maximum files to scan per root before truncating extension stats",
    )
    return p.parse_args()


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _safe_iter_files(root: Path) -> Iterable[Path]:
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            for entry in current.iterdir():
                if entry.is_dir():
                    stack.append(entry)
                elif entry.is_file():
                    yield entry
        except (PermissionError, FileNotFoundError, OSError):
            continue


def inspect_root(path: Path, max_files: int) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "kind": "missing",
        "total_bytes": 0,
        "file_count": 0,
        "scanned_files": 0,
        "scan_truncated": False,
        "top_extensions": {},
        "errors": [],
    }

    if not path.exists():
        return info

    if path.is_file():
        ext = path.suffix.lower().lstrip(".") or "<no_ext>"
        try:
            size = path.stat().st_size
        except OSError as exc:
            info["errors"].append(str(exc))
            size = 0
        info.update(
            {
                "kind": "file",
                "total_bytes": int(size),
                "file_count": 1,
                "scanned_files": 1,
                "top_extensions": {ext: 1},
            }
        )
        return info

    info["kind"] = "dir"
    exts = Counter()

    scanned = 0
    total_bytes = 0
    file_count = 0

    for fp in _safe_iter_files(path):
        file_count += 1
        try:
            total_bytes += fp.stat().st_size
        except OSError as exc:
            info["errors"].append(str(exc))
            continue

        if scanned < max_files:
            ext = fp.suffix.lower().lstrip(".") or "<no_ext>"
            exts[ext] += 1
            scanned += 1
        else:
            info["scan_truncated"] = True

    info["total_bytes"] = int(total_bytes)
    info["file_count"] = int(file_count)
    info["scanned_files"] = int(scanned)
    info["top_extensions"] = dict(exts.most_common(15))
    return info


def bytes_human(n: int) -> str:
    unit = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    idx = 0
    while x >= 1024.0 and idx < len(unit) - 1:
        x /= 1024.0
        idx += 1
    return f"{x:.2f} {unit[idx]}"


def config_summary(config_path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})

    real_splits = eval_cfg.get("real_splits")
    if not real_splits:
        real_splits = [eval_cfg.get("real_split", "validation")]

    real_layer_keys = eval_cfg.get("real_layer_keys")
    if not real_layer_keys:
        real_layer_keys = [eval_cfg.get("real_layer_key", "layer_all")]

    return {
        "config": str(config_path),
        "train": {
            "dataset": data.get("train_dataset_name"),
            "split": data.get("train_split"),
        },
        "validation": {
            "dataset": data.get("val_dataset_name"),
            "split": data.get("val_split"),
        },
        "real_eval": {
            "dataset": eval_cfg.get("real_dataset_name"),
            "splits": [str(s) for s in real_splits],
            "layer_keys": [str(k) for k in real_layer_keys],
        },
        "storage": {
            "cache_dir": str(data.get("cache_dir", "")),
            "staged_root": str(data.get("staged_root", "")),
            "precomputed_index_path": str(data.get("precomputed_index_path", "")),
            "checkpoint_dir": str(cfg.get("training", {}).get("checkpoint", {}).get("dir", "")),
            "report_dir": str(eval_cfg.get("report_dir", "")),
        },
    }


def main() -> None:
    args = parse_args()

    configs: list[Path] = []
    summaries: list[Dict[str, Any]] = []
    roots: dict[str, Path] = {}

    for cfg_raw in args.configs:
        cfg_path = _resolve_path(cfg_raw)
        cfg = load_yaml_file(cfg_path)
        configs.append(cfg_path)

        summary = config_summary(cfg_path, cfg)
        summaries.append(summary)

        storage = summary["storage"]
        try:
            cfg_key_prefix = cfg_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            cfg_key_prefix = str(cfg_path)
        for key, raw in storage.items():
            if not raw:
                continue
            resolved = _resolve_path(raw)
            roots[f"{cfg_key_prefix}:{key}"] = resolved

    root_reports: Dict[str, Any] = {}
    for key, root_path in sorted(roots.items()):
        root_reports[key] = inspect_root(root_path, max_files=int(args.max_files_per_root))

    report = {
        "project_root": str(PROJECT_ROOT),
        "configs": [str(c) for c in configs],
        "split_mapping": summaries,
        "roots": root_reports,
        "notes": [
            "Paths outside this machine or without permissions will appear as missing/inaccessible.",
            "File-format stats are based on extensions from scanned files.",
        ],
    }

    out_path = _resolve_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[inventory] data split mapping")
    for item in summaries:
        train = item["train"]
        val = item["validation"]
        real = item["real_eval"]
        print(
            f"- {item['config']}: train={train['dataset']}:{train['split']} "
            f"val={val['dataset']}:{val['split']} real={real['dataset']}:{','.join(real['splits'])}"
        )

    print("[inventory] storage roots")
    for key, info in root_reports.items():
        size_h = bytes_human(int(info.get("total_bytes", 0)))
        print(
            f"- {key} -> exists={info['exists']} kind={info['kind']} "
            f"files={info['file_count']} size={size_h}"
        )

    print(f"[inventory] wrote {out_path}")


if __name__ == "__main__":
    main()
