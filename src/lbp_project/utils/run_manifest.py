"""Run/report manifest helpers for reproducible stage workflows."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from lbp_project.models.backbone_policy import resolve_backbone_spec


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def config_sha256(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(_json_safe(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_output(project_root: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def git_metadata(project_root: Path) -> Dict[str, Any]:
    commit = _git_output(project_root, ["rev-parse", "HEAD"])
    branch = _git_output(project_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    dirty = bool(_git_output(project_root, ["status", "--porcelain"]))
    return {
        "commit": commit or None,
        "branch": branch or None,
        "dirty": dirty,
    }


def build_run_manifest(
    cfg: Dict[str, Any],
    config_path: str,
    stage_mode: str,
    project_root: Path,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    backbone = resolve_backbone_spec(cfg.get("architecture", {}))

    manifest: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage_mode": stage_mode,
        "config_path": str(config_path),
        "config_sha256": config_sha256(cfg),
        "git": git_metadata(project_root),
        "experiment": {
            "name": cfg.get("experiment", {}).get("name"),
            "seed": cfg.get("experiment", {}).get("seed"),
        },
        "backbone": {
            "descriptor": backbone.descriptor,
            "backend": backbone.backend,
            "repo": backbone.repo,
            "model": backbone.model,
            "expected_embed_dim": backbone.expected_embed_dim,
        },
        "data_contract": {
            "train_dataset_name": data_cfg.get("train_dataset_name"),
            "train_split": data_cfg.get("train_split"),
            "val_dataset_name": data_cfg.get("val_dataset_name"),
            "val_split": data_cfg.get("val_split"),
            "require_local_staging": data_cfg.get("require_local_staging"),
            "use_precomputed_dino": data_cfg.get("use_precomputed_dino"),
        },
        "evaluation_contract": {
            "run_after_train": eval_cfg.get("run_after_train"),
            "periodic_real_eval_every_epochs": eval_cfg.get("periodic_real_eval_every_epochs"),
            "real_dataset_name": eval_cfg.get("real_dataset_name"),
            "real_splits": eval_cfg.get("real_splits", [eval_cfg.get("real_split")]),
            "real_layer_keys": eval_cfg.get("real_layer_keys", [eval_cfg.get("real_layer_key")]),
        },
    }

    if extra:
        manifest["extra"] = _json_safe(extra)
    return manifest


def write_manifest(report_dir: str | Path, name: str, payload: Dict[str, Any]) -> Path:
    root = Path(report_dir)
    out_dir = root / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    out_path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return out_path
