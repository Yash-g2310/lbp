"""Startup preflight checks and download policy helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from lbp_project.models.backbone_policy import resolve_backbone_spec


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def build_download_matrix(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    data_cfg = cfg.get("data", {})
    eval_cfg = cfg.get("evaluation", {})
    auth_cfg = cfg.get("auth", {})

    mandatory = [
        "{}:{} (train)".format(data_cfg.get("train_dataset_name", "<missing>"), data_cfg.get("train_split", "<missing>")),
        "{}:{} (val)".format(data_cfg.get("val_dataset_name", "<missing>"), data_cfg.get("val_split", "<missing>")),
    ]

    if bool(data_cfg.get("use_precomputed_dino", False)):
        mandatory.append(f"precomputed index: {data_cfg.get('precomputed_index_path', '<missing>')}")

    runtime_interactive = [
        f"backbone weights: {resolve_backbone_spec(cfg.get('architecture', {})).descriptor}",
    ]

    if bool(auth_cfg.get("require_wandb_login", False)):
        runtime_interactive.append("Weights & Biases authentication")
    if bool(auth_cfg.get("require_hf_login", False)):
        runtime_interactive.append("Hugging Face authentication")

    optional = []
    if bool(eval_cfg.get("run_real_tuple_eval", False)):
        optional.append(
            "{}:{}".format(
                eval_cfg.get("real_dataset_name", "princeton-vl/LayeredDepth"),
                ",".join(str(s) for s in eval_cfg.get("real_splits", [eval_cfg.get("real_split", "validation")]))
                if eval_cfg.get("real_splits") is not None
                else str(eval_cfg.get("real_split", "validation")),
            )
        )

    return {
        "mandatory_pre_downloads": mandatory,
        "runtime_interactive_downloads": runtime_interactive,
        "optional_downloads": optional,
    }


def format_download_matrix(matrix: Dict[str, List[str]], prefix: str = "[preflight]") -> str:
    lines: List[str] = [f"{prefix} download policy matrix"]
    for key in ("mandatory_pre_downloads", "runtime_interactive_downloads", "optional_downloads"):
        lines.append(f"{prefix} {key}:")
        values = matrix.get(key, [])
        if not values:
            lines.append(f"{prefix}   - <none>")
            continue
        for value in values:
            lines.append(f"{prefix}   - {value}")
    return "\n".join(lines)


def collect_hardware_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    hw_cfg = cfg.get("hardware", {})
    requested_device = str(hw_cfg.get("device", "cpu")).strip().lower()
    target_gpu_class = str(hw_cfg.get("target_gpu_class", "")).strip()
    min_vram_gb = _as_float(hw_cfg.get("min_vram_gb", 0.0), 0.0)

    profile: Dict[str, Any] = {
        "requested_device": requested_device,
        "target_gpu_class": target_gpu_class,
        "min_vram_gb": min_vram_gb,
        "cuda_available": False,
        "cuda_device_count": 0,
        "selected_cuda_index": 0,
        "gpu_name": "",
        "gpu_total_vram_gb": 0.0,
        "issues": [],
    }

    if requested_device != "cuda":
        return profile

    try:
        import torch
    except Exception as exc:
        profile["issues"].append(
            "CUDA device requested but torch import failed: {}".format(exc)
        )
        return profile

    cuda_available = bool(torch.cuda.is_available())
    profile["cuda_available"] = cuda_available
    profile["cuda_device_count"] = int(torch.cuda.device_count()) if cuda_available else 0
    if not cuda_available:
        profile["issues"].append("hardware.device=cuda but no CUDA device is available")
        return profile

    selected_index = int(hw_cfg.get("cuda_device_index", 0))
    if selected_index < 0 or selected_index >= profile["cuda_device_count"]:
        profile["issues"].append(
            "cuda_device_index={} is out of range for device_count={}".format(
                selected_index,
                profile["cuda_device_count"],
            )
        )
        return profile

    profile["selected_cuda_index"] = selected_index
    props = torch.cuda.get_device_properties(selected_index)
    total_vram_gb = float(props.total_memory) / (1024.0 ** 3)
    profile["gpu_name"] = str(props.name)
    profile["gpu_total_vram_gb"] = total_vram_gb

    if min_vram_gb > 0.0 and total_vram_gb + 1.0e-6 < min_vram_gb:
        profile["issues"].append(
            (
                "GPU VRAM below policy: detected={:.2f}GB required_min={:.2f}GB "
                "(target_gpu_class={})"
            ).format(
                total_vram_gb,
                min_vram_gb,
                target_gpu_class or "unspecified",
            )
        )

    return profile


def format_hardware_profile(profile: Dict[str, Any], prefix: str = "[preflight][hardware]") -> str:
    lines = [
        "{} requested_device={} target_gpu_class={} min_vram_gb={:.2f}".format(
            prefix,
            profile.get("requested_device", "unknown"),
            profile.get("target_gpu_class", "") or "unspecified",
            float(profile.get("min_vram_gb", 0.0)),
        ),
        "{} cuda_available={} cuda_device_count={} selected_cuda_index={}".format(
            prefix,
            bool(profile.get("cuda_available", False)),
            int(profile.get("cuda_device_count", 0)),
            int(profile.get("selected_cuda_index", 0)),
        ),
    ]

    if str(profile.get("gpu_name", "")).strip():
        lines.append(
            "{} gpu_name={} gpu_total_vram_gb={:.2f}".format(
                prefix,
                profile.get("gpu_name", ""),
                float(profile.get("gpu_total_vram_gb", 0.0)),
            )
        )

    issues = profile.get("issues", [])
    if issues:
        lines.append(f"{prefix} issues:")
        for issue in issues:
            lines.append(f"{prefix}   - {issue}")
    else:
        lines.append(f"{prefix} issues: <none>")
    return "\n".join(lines)


def enforce_hardware_profile(cfg: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    profile = collect_hardware_profile(cfg)
    issues = list(profile.get("issues", []))
    if issues and strict:
        raise RuntimeError("; ".join(str(issue) for issue in issues))
    return profile


def _load_index_payload(index_path: Path) -> Dict[str, Any]:
    if not index_path.exists():
        raise FileNotFoundError(
            f"Precomputed feature index not found: {index_path}. "
            "Generate it with scripts/data/precompute_dino.py."
        )
    with index_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid precomputed index format: {index_path}")
    return payload


def validate_precomputed_index_compatibility(
    cfg: Dict[str, Any],
    strict: bool,
    index_path: str | Path | None = None,
) -> List[str]:
    data_cfg = cfg["data"]
    index_file = Path(index_path) if index_path is not None else Path(str(data_cfg["precomputed_index_path"]))
    payload = _load_index_payload(index_file)

    samples = payload.get("samples")
    if not isinstance(samples, dict):
        raise RuntimeError(
            f"Invalid precomputed index '{index_file}': missing 'samples' mapping"
        )

    required_splits = [str(data_cfg["train_split"]), str(data_cfg["val_split"])]
    missing_splits = [split for split in required_splits if split not in samples]
    if missing_splits:
        raise RuntimeError(
            f"Precomputed index '{index_file}' missing split(s): {missing_splits}. "
            f"Available splits: {sorted(samples.keys())}"
        )

    expected = resolve_backbone_spec(cfg["architecture"])
    metadata = payload.get("metadata")
    warnings: List[str] = []

    if not isinstance(metadata, dict):
        msg = (
            f"Precomputed index '{index_file}' has no metadata block for compatibility checks. "
            "Regenerate via scripts/data/precompute_dino.py to stamp backbone metadata."
        )
        if strict:
            raise RuntimeError(msg)
        warnings.append(msg)
        return warnings

    index_desc = str(metadata.get("backbone_descriptor", "")).strip()
    index_dim = metadata.get("dino_embed_dim")

    if index_desc:
        if index_desc != expected.descriptor:
            raise RuntimeError(
                "Precomputed index backbone mismatch: index='{}' config='{}'. "
                "Regenerate index or align architecture.backbone_* settings.".format(
                    index_desc,
                    expected.descriptor,
                )
            )
    elif strict:
        raise RuntimeError(
            f"Precomputed index '{index_file}' missing metadata.backbone_descriptor in strict mode"
        )

    if index_dim is not None:
        try:
            index_dim_int = int(index_dim)
        except Exception as exc:
            raise RuntimeError(
                f"Invalid metadata.dino_embed_dim in precomputed index '{index_file}': {index_dim}"
            ) from exc
        if index_dim_int != expected.expected_embed_dim:
            raise RuntimeError(
                "Precomputed index feature dim mismatch: index={} config={}. "
                "Regenerate features or align architecture.dino_embed_dim.".format(
                    index_dim_int,
                    expected.expected_embed_dim,
                )
            )
    elif strict:
        raise RuntimeError(
            f"Precomputed index '{index_file}' missing metadata.dino_embed_dim in strict mode"
        )

    return warnings


def enforce_startup_preflight(cfg: Dict[str, Any], strict_server_policy: bool = False) -> List[str]:
    data_cfg = cfg["data"]
    warnings: List[str] = []

    cache_dir = Path(str(data_cfg["cache_dir"]))
    fallback_cache_dir_raw = str(data_cfg.get("fallback_cache_dir", "")).strip()
    fallback_cache_dir = Path(fallback_cache_dir_raw) if fallback_cache_dir_raw else None
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if fallback_cache_dir is None:
            raise RuntimeError(
                (
                    "Unable to create/access data.cache_dir='{}'. "
                    "Provide a writable data.fallback_cache_dir for this host or update data.cache_dir. "
                    "Original error: {}"
                ).format(cache_dir, exc)
            ) from exc

        fallback_cache_dir.mkdir(parents=True, exist_ok=True)
        warnings.append(
            "cache_dir '{}' is not writable on this host; using fallback_cache_dir '{}'".format(
                cache_dir,
                fallback_cache_dir,
            )
        )
        data_cfg["cache_dir"] = str(fallback_cache_dir)
        cache_dir = fallback_cache_dir

    require_staging = bool(data_cfg.get("require_local_staging", False))
    staged_root = Path(str(data_cfg.get("staged_root", "")))
    if require_staging and not staged_root.exists():
        raise FileNotFoundError(
            f"Required staged_root does not exist: {staged_root}. "
            "Set data.staged_root to a valid path or disable data.require_local_staging."
        )

    if strict_server_policy and not bool(data_cfg.get("use_precomputed_dino", False)):
        raise RuntimeError("Server preflight policy requires data.use_precomputed_dino=true")

    if bool(data_cfg.get("use_precomputed_dino", False)):
        strict_compat = bool(
            data_cfg.get("precomputed_strict_compatibility", strict_server_policy or require_staging)
        )
        warnings.extend(validate_precomputed_index_compatibility(cfg, strict=strict_compat))

    return warnings
