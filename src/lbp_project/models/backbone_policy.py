"""Backbone policy resolution and candidate selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


SUPPORTED_BACKENDS = {"torch_hub", "timm_hf"}


@dataclass(frozen=True)
class BackboneSpec:
    backend: str
    repo: str
    model: str
    expected_embed_dim: int
    descriptor: str


def _clean_string(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"architecture.{field_name} must be a non-empty string")
    return text


def _coerce_backend(repo: str, model: str, raw_backend: str) -> str:
    backend = raw_backend.strip().lower()
    if backend:
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                "Unsupported architecture.backbone_backend='{}'; supported: {}".format(
                    backend,
                    ", ".join(sorted(SUPPORTED_BACKENDS)),
                )
            )
        return backend

    # Auto-detect backend for backwards compatibility.
    if model.startswith("timm/") or repo.lower() in {"timm", "huggingface_timm", "hf_timm"}:
        return "timm_hf"
    return "torch_hub"


def _normalize_timm_model(model: str) -> str:
    if model.startswith("hf-hub:"):
        model = model[len("hf-hub:") :]
    return model


def _dedupe_specs(specs: List[BackboneSpec]) -> List[BackboneSpec]:
    deduped: List[BackboneSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.descriptor in seen:
            continue
        deduped.append(spec)
        seen.add(spec.descriptor)
    return deduped


def resolve_backbone_spec(architecture_cfg: Dict[str, Any]) -> BackboneSpec:
    repo = _clean_string(architecture_cfg.get("backbone_repo", "timm"), "backbone_repo")
    model = _clean_string(
        architecture_cfg.get("backbone_model", "timm/convnext_small.dinov3_lvd1689m"),
        "backbone_model",
    )
    backend = _coerce_backend(repo, model, str(architecture_cfg.get("backbone_backend", "")))

    try:
        expected_embed_dim = int(architecture_cfg.get("dino_embed_dim", 0))
    except Exception as exc:
        raise ValueError("architecture.dino_embed_dim must be an integer") from exc
    if expected_embed_dim < 1:
        raise ValueError(f"architecture.dino_embed_dim must be >= 1, got {expected_embed_dim}")

    if backend == "timm_hf":
        model = _normalize_timm_model(model)
        repo = "timm"
        descriptor = f"timm_hf:{model}"
    else:
        descriptor = f"torch_hub:{repo}:{model}"

    return BackboneSpec(
        backend=backend,
        repo=repo,
        model=model,
        expected_embed_dim=expected_embed_dim,
        descriptor=descriptor,
    )


def collect_backbone_candidates(architecture_cfg: Dict[str, Any]) -> List[BackboneSpec]:
    primary = resolve_backbone_spec(architecture_cfg)
    raw_fallbacks = architecture_cfg.get("backbone_fallback_models", [])
    if raw_fallbacks is None:
        raw_fallbacks = []
    if not isinstance(raw_fallbacks, (list, tuple)):
        raise ValueError("architecture.backbone_fallback_models must be a list when provided")

    candidates: List[BackboneSpec] = [primary]
    for i, fallback in enumerate(raw_fallbacks):
        fallback_cfg = dict(architecture_cfg)
        if isinstance(fallback, str):
            fallback_cfg["backbone_model"] = fallback
            fallback_cfg.pop("backbone_backend", None)
        elif isinstance(fallback, dict):
            fallback_cfg.update(fallback)
        else:
            raise ValueError(
                "architecture.backbone_fallback_models[{}] must be a string or mapping".format(i)
            )
        candidates.append(resolve_backbone_spec(fallback_cfg))

    return _dedupe_specs(candidates)


def should_stop_on_primary_backbone_failure(architecture_cfg: Dict[str, Any]) -> bool:
    value = architecture_cfg.get("backbone_stop_on_failure", True)
    if isinstance(value, bool):
        return value
    raise ValueError(
        "architecture.backbone_stop_on_failure must be a boolean; "
        f"got {type(value).__name__}"
    )


def is_backbone_fallback_approved(architecture_cfg: Dict[str, Any]) -> bool:
    value = architecture_cfg.get("backbone_fallback_approved", False)
    if isinstance(value, bool):
        return value
    raise ValueError(
        "architecture.backbone_fallback_approved must be a boolean; "
        f"got {type(value).__name__}"
    )
