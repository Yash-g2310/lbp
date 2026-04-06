#!/usr/bin/env python3
"""Build a consolidated data-truth report for dataset/cache/notebook state.

This script is read-only with respect to datasets and notebooks; it only writes
report files under docs/generated/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
PARQUET_FILE_RE = re.compile(r"^data/(?P<split>[a-zA-Z0-9_]+)-\d{5}-of-\d{5}\.parquet$")
ARROW_FILE_RE = re.compile(r"-(?P<split>[a-zA-Z0-9_]+)-\d{5}-of-(?P<of>\d{5})\.arrow$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a consolidated data-truth report")
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
        "--notebooks",
        nargs="+",
        default=[
            "notebooks/experiment1.ipynb",
            "notebooks/unet_connect.ipynb",
        ],
        help="Notebook files to audit",
    )
    p.add_argument(
        "--output-json",
        default="docs/generated/data_truth_report.json",
        help="Output JSON report path",
    )
    p.add_argument(
        "--output-md",
        default="docs/generated/data_truth_report.md",
        help="Output Markdown report path",
    )
    p.add_argument(
        "--http-timeout",
        type=float,
        default=20.0,
        help="HTTP timeout seconds for live HF metadata",
    )
    return p.parse_args()


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _canonical_token(text: str) -> str:
    return NON_ALNUM_RE.sub("", text.lower())


def load_yaml_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict YAML payload in {path}, got {type(payload).__name__}")
    return payload


def fetch_json(url: str, timeout_s: float) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    req = Request(url, headers={"User-Agent": "lbp-data-truth-report/1.0"})
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8")
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload, None
        return None, f"Non-object JSON payload from {url}"
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        return None, str(exc)


def _safe_rel(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _normalize_real_splits(eval_cfg: Dict[str, Any]) -> List[str]:
    if isinstance(eval_cfg.get("real_splits"), list):
        return [str(x) for x in eval_cfg.get("real_splits", []) if str(x)]
    if eval_cfg.get("real_split"):
        return [str(eval_cfg["real_split"])]
    return ["validation"]


def _normalize_real_layer_keys(eval_cfg: Dict[str, Any]) -> List[str]:
    if isinstance(eval_cfg.get("real_layer_keys"), list):
        return [str(x) for x in eval_cfg.get("real_layer_keys", []) if str(x)]
    if eval_cfg.get("real_layer_key"):
        return [str(eval_cfg["real_layer_key"])]
    return ["layer_all"]


def build_config_records(config_paths: Iterable[Path]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for cfg_path in config_paths:
        cfg = load_yaml_file(cfg_path)
        data = cfg.get("data", {})
        ev = cfg.get("evaluation", {})
        rec = {
            "config_path": str(cfg_path),
            "config_path_rel": _safe_rel(cfg_path),
            "train": {
                "dataset": str(data.get("train_dataset_name", "")),
                "split": str(data.get("train_split", "")),
            },
            "validation": {
                "dataset": str(data.get("val_dataset_name", "")),
                "split": str(data.get("val_split", "")),
            },
            "real_eval": {
                "dataset": str(ev.get("real_dataset_name", "")),
                "splits": _normalize_real_splits(ev),
                "layer_keys": _normalize_real_layer_keys(ev),
            },
            "storage": {
                "cache_dir": str(data.get("cache_dir", "")),
                "staged_root": str(data.get("staged_root", "")),
                "precomputed_index_path": str(data.get("precomputed_index_path", "")),
                "checkpoint_dir": str(cfg.get("training", {}).get("checkpoint", {}).get("dir", "")),
                "report_dir": str(ev.get("report_dir", "")),
            },
            "precomputed": {
                "use_precomputed_dino": bool(data.get("use_precomputed_dino", False)),
                "precomputed_index_path": str(data.get("precomputed_index_path", "")),
                "real_use_precomputed_dino": bool(ev.get("real_use_precomputed_dino", False)),
                "real_precomputed_index_path": str(
                    ev.get("real_precomputed_index_path", data.get("precomputed_index_path", ""))
                ),
            },
        }
        records.append(rec)
    return records


def collect_dataset_usage(config_records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    usage: Dict[str, Dict[str, Any]] = {}

    for rec in config_records:
        train = rec["train"]
        val = rec["validation"]
        real = rec["real_eval"]

        ds_train = str(train.get("dataset", "")).strip()
        sp_train = str(train.get("split", "")).strip()
        if ds_train and sp_train:
            item = usage.setdefault(ds_train, {"splits": set(), "sources": set()})
            item["splits"].add(sp_train)
            item["sources"].add(rec["config_path_rel"])

        ds_val = str(val.get("dataset", "")).strip()
        sp_val = str(val.get("split", "")).strip()
        if ds_val and sp_val:
            item = usage.setdefault(ds_val, {"splits": set(), "sources": set()})
            item["splits"].add(sp_val)
            item["sources"].add(rec["config_path_rel"])

        ds_real = str(real.get("dataset", "")).strip()
        real_splits = real.get("splits", [])
        if ds_real:
            item = usage.setdefault(ds_real, {"splits": set(), "sources": set()})
            for split_name in real_splits:
                item["splits"].add(str(split_name))
            item["sources"].add(rec["config_path_rel"])

    normalized: Dict[str, Dict[str, Any]] = {}
    for dataset_id, item in usage.items():
        normalized[dataset_id] = {
            "splits": sorted(item["splits"]),
            "sources": sorted(item["sources"]),
        }
    return normalized


def _extract_top_features(features_obj: Any) -> List[str]:
    names: List[str] = []
    if isinstance(features_obj, list):
        for item in features_obj:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
    elif isinstance(features_obj, dict):
        names.extend([str(k) for k in features_obj.keys()])
    return names


def _count_parquet_by_split_from_siblings(siblings: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not isinstance(siblings, list):
        return out
    for item in siblings:
        if not isinstance(item, dict):
            continue
        name = str(item.get("rfilename", ""))
        m = PARQUET_FILE_RE.match(name)
        if not m:
            continue
        split = m.group("split")
        out[split] = out.get(split, 0) + 1
    return out


def _parse_hf_api_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    card = payload.get("cardData", {}) if isinstance(payload.get("cardData"), dict) else {}
    ds_info = card.get("dataset_info", {}) if isinstance(card.get("dataset_info"), dict) else {}

    split_stats: Dict[str, Dict[str, Any]] = {}
    for split_entry in ds_info.get("splits", []) if isinstance(ds_info.get("splits"), list) else []:
        if not isinstance(split_entry, dict):
            continue
        split_name = str(split_entry.get("name", "")).strip()
        if not split_name:
            continue
        split_stats[split_name] = {
            "num_examples": split_entry.get("num_examples"),
            "num_bytes": split_entry.get("num_bytes"),
        }

    return {
        "id": payload.get("id"),
        "sha": payload.get("sha"),
        "last_modified": payload.get("lastModified"),
        "split_stats": split_stats,
        "top_features": _extract_top_features(ds_info.get("features", {})),
        "parquet_counts_by_split": _count_parquet_by_split_from_siblings(payload.get("siblings", [])),
    }


def _parse_ds_server_splits(payload: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for item in payload.get("splits", []) if isinstance(payload.get("splits"), list) else []:
        if isinstance(item, dict) and item.get("split"):
            out.append(str(item["split"]))
    return sorted(set(out))


def _parse_ds_server_parquet(payload: Dict[str, Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in payload.get("parquet_files", []) if isinstance(payload.get("parquet_files"), list) else []:
        if not isinstance(item, dict):
            continue
        split = str(item.get("split", "")).strip()
        if not split:
            continue
        counts[split] = counts.get(split, 0) + 1
    return counts


def fetch_remote_dataset_metadata(dataset_id: str, timeout_s: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"dataset_id": dataset_id, "errors": []}

    api_url = f"https://huggingface.co/api/datasets/{quote(dataset_id, safe='/')}"
    api_payload, api_err = fetch_json(api_url, timeout_s)
    if api_payload is None:
        out["errors"].append({"source": "hf_api", "error": api_err})
        out["hf_api"] = {}
    else:
        out["hf_api"] = _parse_hf_api_metadata(api_payload)

    splits_url = "https://datasets-server.huggingface.co/splits?" + urlencode({"dataset": dataset_id})
    splits_payload, splits_err = fetch_json(splits_url, timeout_s)
    if splits_payload is None:
        out["errors"].append({"source": "datasets_server_splits", "error": splits_err})
        out["datasets_server_splits"] = []
    else:
        out["datasets_server_splits"] = _parse_ds_server_splits(splits_payload)

    parquet_url = "https://datasets-server.huggingface.co/parquet?" + urlencode({"dataset": dataset_id})
    parquet_payload, parquet_err = fetch_json(parquet_url, timeout_s)
    if parquet_payload is None:
        out["errors"].append({"source": "datasets_server_parquet", "error": parquet_err})
        out["datasets_server_parquet_counts"] = {}
    else:
        out["datasets_server_parquet_counts"] = _parse_ds_server_parquet(parquet_payload)

    return out


def discover_local_dataset_entries(cache_roots: Iterable[Path]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    seen_info_paths: set[str] = set()

    for root in cache_roots:
        if not root.exists() or not root.is_dir():
            continue

        for info_file in root.glob("*___*/default/*/*/dataset_info.json"):
            info_file_str = str(info_file)
            if info_file_str in seen_info_paths:
                continue
            seen_info_paths.add(info_file_str)

            try:
                payload = json.loads(info_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue

            try:
                rel = info_file.relative_to(root)
                namespace_dir = rel.parts[0]
            except (ValueError, IndexError):
                continue

            owner, _, cache_dataset_dir = namespace_dir.partition("___")
            dataset_name = str(payload.get("dataset_name", ""))
            splits_payload = payload.get("splits", {})

            family_by_split: Dict[str, int] = {}
            if isinstance(splits_payload, dict):
                for split_name, split_info in splits_payload.items():
                    if isinstance(split_info, dict) and isinstance(split_info.get("shard_lengths"), list):
                        family_by_split[str(split_name)] = len(split_info.get("shard_lengths", []))
                    else:
                        family_by_split[str(split_name)] = 0

            present_by_split: Dict[str, int] = {}
            of_total_by_split: Dict[str, int] = {}
            for arrow_file in info_file.parent.glob("*.arrow"):
                m = ARROW_FILE_RE.search(arrow_file.name)
                if not m:
                    continue
                split_name = m.group("split")
                present_by_split[split_name] = present_by_split.get(split_name, 0) + 1
                of_total = int(m.group("of"))
                of_total_by_split[split_name] = max(of_total_by_split.get(split_name, 0), of_total)

            try:
                mtime = info_file.stat().st_mtime
            except OSError:
                mtime = 0.0

            entries.append(
                {
                    "cache_root": str(root),
                    "owner": owner,
                    "cache_dataset_dir": cache_dataset_dir,
                    "dataset_name": dataset_name,
                    "dataset_token": _canonical_token(f"{owner}/{dataset_name}"),
                    "cache_token": _canonical_token(f"{owner}/{cache_dataset_dir}"),
                    "dataset_info_path": str(info_file),
                    "cache_leaf_dir": str(info_file.parent),
                    "local_arrow_family_by_split": family_by_split,
                    "local_arrow_files_present_by_split": present_by_split,
                    "local_arrow_of_total_by_split": of_total_by_split,
                    "download_size": payload.get("download_size"),
                    "dataset_size": payload.get("dataset_size"),
                    "mtime": mtime,
                }
            )

    return entries


def select_local_entry(dataset_id: str, entries: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    token = _canonical_token(dataset_id)
    candidates = [
        e
        for e in entries
        if token == e.get("dataset_token")
        or token == e.get("cache_token")
        or token.endswith(str(e.get("dataset_token", "")))
    ]

    if not candidates:
        owner = dataset_id.split("/", 1)[0] if "/" in dataset_id else ""
        dataset_part = dataset_id.split("/", 1)[1] if "/" in dataset_id else dataset_id
        owner_tok = _canonical_token(owner)
        dataset_tok = _canonical_token(dataset_part)
        candidates = [
            e
            for e in entries
            if _canonical_token(str(e.get("owner", ""))) == owner_tok
            and _canonical_token(str(e.get("dataset_name", ""))) == dataset_tok
        ]

    if not candidates:
        return None

    candidates.sort(key=lambda e: float(e.get("mtime", 0.0)), reverse=True)
    return candidates[0]


def classify_split_status(
    remote_parquet: Optional[int],
    local_family: Optional[int],
    local_present: Optional[int],
) -> Tuple[str, str]:
    note_parts: List[str] = []

    if local_family is None and local_present is None:
        status = "no_local_cache"
        note_parts.append("No local cache metadata for this dataset split.")
    elif local_family is None:
        status = "local_split_unknown"
        note_parts.append("Split not described in local dataset_info metadata.")
    elif local_present is None:
        status = "local_files_unknown"
        note_parts.append("Local arrow file count could not be computed.")
    elif local_family == 0 and local_present == 0:
        status = "local_not_materialized"
        note_parts.append("Local split has no materialized arrow shards.")
    elif local_present == local_family:
        status = "local_materialized_complete"
        note_parts.append("All local arrow shards listed in local family are present.")
    elif local_present < local_family:
        status = "local_materialized_partial"
        note_parts.append("Only a subset of local arrow shards are present in cache.")
    else:
        status = "local_materialized_inconsistent"
        note_parts.append("Local present arrow shards exceed local family count metadata.")

    if remote_parquet is not None and local_family is not None:
        if local_family < remote_parquet:
            note_parts.append(
                "Local arrow shard family is smaller than remote parquet family; cache appears partially materialized."
            )
        elif local_family == remote_parquet:
            note_parts.append("Local arrow shard family size matches remote parquet family size.")
        else:
            note_parts.append(
                "Local arrow shard family exceeds remote parquet count; verify dataset revision or cache drift."
            )

    note_parts.append(
        "Missing local shards are treated as not present locally; deletion cannot be proven without filesystem history."
    )
    return status, " ".join(note_parts)


def build_reconciliation_rows(
    dataset_id: str,
    usage_splits: List[str],
    remote_meta: Dict[str, Any],
    local_entry: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    hf_api_counts = remote_meta.get("hf_api", {}).get("parquet_counts_by_split", {})
    ds_server_counts = remote_meta.get("datasets_server_parquet_counts", {})

    remote_counts: Dict[str, int] = {}
    if isinstance(hf_api_counts, dict):
        for split_name, count in hf_api_counts.items():
            if isinstance(count, int):
                remote_counts[str(split_name)] = count

    if isinstance(ds_server_counts, dict):
        # Keep canonical HF API sibling counts as primary; datasets-server can be
        # partial for large datasets (e.g., "partial-test"). Only use it to fill
        # missing split keys, and never downscale an existing split count.
        for split_name, count in ds_server_counts.items():
            if not isinstance(count, int):
                continue
            split_key = str(split_name)
            if split_key not in remote_counts:
                remote_counts[split_key] = count
            else:
                remote_counts[split_key] = max(remote_counts[split_key], count)

    hf_split_stats = remote_meta.get("hf_api", {}).get("split_stats", {})
    remote_split_names = set()
    if isinstance(hf_split_stats, dict):
        remote_split_names.update(hf_split_stats.keys())

    local_family_by_split = local_entry.get("local_arrow_family_by_split", {}) if local_entry else {}
    local_present_by_split = local_entry.get("local_arrow_files_present_by_split", {}) if local_entry else {}

    split_names = set(usage_splits)
    split_names.update(remote_split_names)
    split_names.update(local_family_by_split.keys())
    split_names.update(local_present_by_split.keys())

    rows: List[Dict[str, Any]] = []
    for split in sorted(s for s in split_names if s):
        remote_parquet = remote_counts.get(split)
        local_family = local_family_by_split.get(split)
        local_present = local_present_by_split.get(split)
        status, note = classify_split_status(remote_parquet, local_family, local_present)

        rows.append(
            {
                "dataset": dataset_id,
                "split": split,
                "remote_parquet_shards": remote_parquet,
                "remote_examples": (hf_split_stats.get(split) or {}).get("num_examples") if isinstance(hf_split_stats, dict) else None,
                "local_arrow_family_shards": local_family,
                "local_arrow_files_present": local_present,
                "status": status,
                "note": note,
            }
        )

    return rows


def _path_kind(path: Path) -> str:
    if path.exists():
        if path.is_file():
            return "file"
        if path.is_dir():
            return "dir"
    return "missing"


def _path_scope(path_str: str) -> str:
    if path_str.startswith("/mnt/"):
        return "server_mount"
    if path_str.startswith("/"):
        return "local_absolute"
    return "project_relative"


def collect_precomputed_status(config_records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rec in config_records:
        pre = rec.get("precomputed", {})
        for label, use_key, path_key in (
            ("train_val", "use_precomputed_dino", "precomputed_index_path"),
            ("real_eval", "real_use_precomputed_dino", "real_precomputed_index_path"),
        ):
            raw_path = str(pre.get(path_key, "")).strip()
            resolved = _resolve_path(raw_path) if raw_path else None
            exists = bool(resolved.exists()) if resolved is not None else False
            rows.append(
                {
                    "config": rec.get("config_path_rel"),
                    "mode": label,
                    "enabled": bool(pre.get(use_key, False)),
                    "index_path": raw_path,
                    "resolved_path": str(resolved) if resolved is not None else "",
                    "path_scope": _path_scope(raw_path) if raw_path else "unset",
                    "exists": exists,
                    "kind": _path_kind(resolved) if resolved is not None else "missing",
                }
            )
    return rows


def _join_source(cell: Dict[str, Any]) -> str:
    src = cell.get("source", [])
    if isinstance(src, list):
        return "".join(str(x) for x in src)
    return str(src)


def analyze_notebook(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "path_rel": _safe_rel(path),
            "exists": False,
            "error": "Notebook file not found",
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "path": str(path),
            "path_rel": _safe_rel(path),
            "exists": True,
            "error": str(exc),
        }

    cells = payload.get("cells", []) if isinstance(payload, dict) else []
    code_cells = [c for c in cells if isinstance(c, dict) and c.get("cell_type") == "code"]

    output_cells = 0
    executed_cells = 0
    error_output_cells = 0
    dataset_calls: List[str] = []
    tuple_refs = 0
    checkpoint_refs: List[str] = []

    for cell in code_cells:
        outputs = cell.get("outputs", [])
        if isinstance(outputs, list) and outputs:
            output_cells += 1
            for out in outputs:
                if isinstance(out, dict) and (out.get("output_type") == "error" or out.get("ename")):
                    error_output_cells += 1
                    break
        if cell.get("execution_count") is not None:
            executed_cells += 1

        source_text = _join_source(cell)
        for line in source_text.splitlines():
            line_stripped = line.strip()
            if "load_dataset(" in line_stripped:
                dataset_calls.append(line_stripped)
            if "tuples.json" in line_stripped:
                tuple_refs += 1
            if ".pth" in line_stripped and ("torch.save(" in line_stripped or "torch.load(" in line_stripped):
                checkpoint_refs.append(line_stripped)

    status = "fresh_or_unrun"
    if executed_cells == 0 and output_cells > 0:
        status = "stale_outputs_not_executed"
    elif error_output_cells > 0:
        status = "contains_error_outputs"

    minimal_steps: List[str]
    name = path.name.lower()
    if "experiment1" in name:
        minimal_steps = [
            "Load one local sample and print keys: dataset[0].keys()",
            "Verify tuple schema presence: sample['tuples.json']['layer_all']['pairs']",
            "Run one forward pass and one tuple-eval call on a single sample",
        ]
    else:
        minimal_steps = [
            "Run streaming load and fetch one sample only (no long train loop)",
            "Verify tuple schema and image key availability on that sample",
            "If checkpoint path is referenced, validate file existence before torch.load",
        ]

    return {
        "path": str(path),
        "path_rel": _safe_rel(path),
        "exists": True,
        "cell_count": len(cells),
        "code_cell_count": len(code_cells),
        "code_cells_with_outputs": output_cells,
        "code_cells_with_execution_count": executed_cells,
        "code_cells_with_error_outputs": error_output_cells,
        "status": status,
        "dataset_calls": dataset_calls[:12],
        "tuple_reference_lines": tuple_refs,
        "checkpoint_reference_lines": checkpoint_refs[:12],
        "minimal_validation_steps": minimal_steps,
    }


def render_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Data Truth Report")
    lines.append("")
    lines.append(f"Generated at: {report.get('generated_at_utc', '')}")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Authority used: live HuggingFace metadata + current local cache state.")
    lines.append("Interpretation rule: missing local shards are treated as not present locally; deletion is not asserted.")
    lines.append("")

    lines.append("## Dataset Reconciliation")
    lines.append("")
    for dataset_id, ds in sorted((report.get("datasets") or {}).items()):
        lines.append(f"### {dataset_id}")
        lines.append("")

        hf_api = ds.get("remote", {}).get("hf_api", {})
        top_features = hf_api.get("top_features", []) if isinstance(hf_api, dict) else []
        if top_features:
            lines.append("Top-level fields: " + ", ".join(str(x) for x in top_features))
            lines.append("")

        rows = ds.get("reconciliation", [])
        lines.append("| split | remote parquet | remote examples | local arrow family | local arrow present | status |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for row in rows:
            lines.append(
                "| {split} | {remote} | {examples} | {family} | {present} | {status} |".format(
                    split=row.get("split", ""),
                    remote=("-" if row.get("remote_parquet_shards") is None else row.get("remote_parquet_shards")),
                    examples=("-" if row.get("remote_examples") is None else row.get("remote_examples")),
                    family=("-" if row.get("local_arrow_family_shards") is None else row.get("local_arrow_family_shards")),
                    present=("-" if row.get("local_arrow_files_present") is None else row.get("local_arrow_files_present")),
                    status=row.get("status", ""),
                )
            )
        lines.append("")

    lines.append("## Precomputed DINO Index Paths")
    lines.append("")
    lines.append("| config | mode | enabled | index path | scope | exists | kind |")
    lines.append("|---|---|---|---|---|---|---|")
    for row in report.get("precomputed_dino", []):
        lines.append(
            "| {config} | {mode} | {enabled} | {path} | {scope} | {exists} | {kind} |".format(
                config=row.get("config", ""),
                mode=row.get("mode", ""),
                enabled=str(bool(row.get("enabled", False))).lower(),
                path=row.get("index_path", ""),
                scope=row.get("path_scope", ""),
                exists=str(bool(row.get("exists", False))).lower(),
                kind=row.get("kind", ""),
            )
        )
    lines.append("")

    lines.append("## Notebook State")
    lines.append("")
    lines.append("| notebook | code cells | outputs in code cells | execution_count present | error outputs | status |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for nb in report.get("notebooks", []):
        lines.append(
            "| {path} | {code} | {out} | {exec_count} | {err} | {status} |".format(
                path=nb.get("path_rel", nb.get("path", "")),
                code=nb.get("code_cell_count", "-"),
                out=nb.get("code_cells_with_outputs", "-"),
                exec_count=nb.get("code_cells_with_execution_count", "-"),
                err=nb.get("code_cells_with_error_outputs", "-"),
                status=nb.get("status", nb.get("error", "")),
            )
        )
    lines.append("")

    lines.append("## Server Verification Checklist")
    lines.append("")
    lines.append("1. Verify server index path exists: test -f /mnt/home2/home/yash_g/layered_depth/features/index.json")
    lines.append("2. Count shard files: ls /mnt/home2/home/yash_g/layered_depth/features/*_shard_*.pt | wc -l")
    lines.append("3. Validate index payload keys and split coverage before training/eval jobs.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    config_paths = [_resolve_path(x) for x in args.configs]
    notebook_paths = [_resolve_path(x) for x in args.notebooks]

    config_records = build_config_records(config_paths)
    dataset_usage = collect_dataset_usage(config_records)

    remote_by_dataset: Dict[str, Dict[str, Any]] = {}
    for dataset_id in sorted(dataset_usage.keys()):
        remote_by_dataset[dataset_id] = fetch_remote_dataset_metadata(dataset_id, timeout_s=float(args.http_timeout))

    cache_roots = set()
    cache_roots.add(Path.home() / ".cache" / "huggingface" / "datasets")
    for rec in config_records:
        cache_dir = str(rec.get("storage", {}).get("cache_dir", "")).strip()
        if cache_dir:
            cache_roots.add(_resolve_path(cache_dir))

    local_entries = discover_local_dataset_entries(sorted(cache_roots, key=str))

    dataset_reports: Dict[str, Dict[str, Any]] = {}
    for dataset_id, usage in sorted(dataset_usage.items()):
        local_entry = select_local_entry(dataset_id, local_entries)
        reconciliation = build_reconciliation_rows(
            dataset_id,
            usage_splits=list(usage.get("splits", [])),
            remote_meta=remote_by_dataset.get(dataset_id, {}),
            local_entry=local_entry,
        )
        dataset_reports[dataset_id] = {
            "usage": usage,
            "remote": remote_by_dataset.get(dataset_id, {}),
            "local_cache": local_entry,
            "reconciliation": reconciliation,
        }

    notebooks = [analyze_notebook(p) for p in notebook_paths]
    precomputed_rows = collect_precomputed_status(config_records)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "configs": [_safe_rel(p) for p in config_paths],
        "dataset_usage": dataset_usage,
        "datasets": dataset_reports,
        "precomputed_dino": precomputed_rows,
        "notebooks": notebooks,
        "notes": [
            "Remote parquet counts are from live endpoints when reachable.",
            "Local arrow counts describe current materialized cache on this machine.",
            "Deletion cannot be proven from cache state alone.",
        ],
    }

    out_json = _resolve_path(args.output_json)
    out_md = _resolve_path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")

    print(f"[data-truth] wrote JSON: {out_json}")
    print(f"[data-truth] wrote Markdown: {out_md}")
    print(f"[data-truth] datasets: {len(dataset_reports)} | notebooks: {len(notebooks)}")


if __name__ == "__main__":
    main()
