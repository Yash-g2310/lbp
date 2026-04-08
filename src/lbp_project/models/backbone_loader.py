"""Backbone loading and feature extraction adapters."""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn as nn

from .backbone_policy import BackboneSpec


def _extract_tensor(raw_features: Any, spec: BackboneSpec) -> torch.Tensor:
    features = raw_features

    if isinstance(features, dict):
        for key in (
            "x_norm_patchtokens",
            "x_prenorm",
            "x",
            "features",
            "last_hidden_state",
        ):
            if key in features:
                features = features[key]
                break

    if isinstance(features, (list, tuple)):
        if not features:
            raise RuntimeError(f"Backbone {spec.descriptor} returned an empty feature sequence")
        features = features[-1]

    if not torch.is_tensor(features):
        raise RuntimeError(
            f"Backbone {spec.descriptor} produced unsupported feature type: {type(features).__name__}"
        )

    if features.ndim == 3:
        batch, tokens, channels = features.shape
        side = int(math.isqrt(tokens))
        if side * side != tokens:
            raise RuntimeError(
                "Backbone {} returned non-square token grid: tokens={} shape={}".format(
                    spec.descriptor,
                    tokens,
                    tuple(features.shape),
                )
            )
        return features.transpose(1, 2).reshape(batch, channels, side, side)

    if features.ndim == 4:
        return features

    raise RuntimeError(
        f"Backbone {spec.descriptor} returned unsupported feature rank {features.ndim}"
    )


class FrozenBackbone(nn.Module):
    """Uniform wrapper that always returns [B, C, H, W] features."""

    def __init__(self, module: nn.Module, spec: BackboneSpec) -> None:
        super().__init__()
        self.module = module
        self.spec = spec
        self.module.eval()
        for param in self.module.parameters():
            param.requires_grad = False

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            forward_features = getattr(self.module, "forward_features", None)
            if callable(forward_features):
                raw = forward_features(x)
            else:
                raw = self.module(x)
            features = _extract_tensor(raw, self.spec)
        if features.ndim != 4:
            raise RuntimeError(
                f"Backbone {self.spec.descriptor} adapter produced invalid shape {tuple(features.shape)}"
            )
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


def _load_torch_hub_backbone(spec: BackboneSpec) -> nn.Module:
    try:
        loaded = torch.hub.load(spec.repo, spec.model)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load torch.hub backbone '{}'. Ensure repo/model is valid and internet/auth is available."
            .format(spec.descriptor)
        ) from exc

    if not isinstance(loaded, nn.Module):
        raise RuntimeError(
            "torch.hub.load returned unsupported type for '{}': {}".format(
                spec.descriptor,
                type(loaded).__name__,
            )
        )
    return cast(nn.Module, loaded)


def _load_timm_hf_backbone(spec: BackboneSpec) -> nn.Module:
    try:
        import timm
    except Exception as exc:
        raise RuntimeError(
            "Backbone '{}' requires the timm package. Install it in the active environment."
            .format(spec.descriptor)
        ) from exc

    model_name = spec.model
    # timm supports Hugging Face Hub models via hf-hub:<model-id>.
    if "/" in model_name and not model_name.startswith("hf-hub:"):
        model_name = f"hf-hub:{model_name}"

    try:
        loaded = timm.create_model(model_name, pretrained=True)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load timm/HF backbone '{}'. Check model id and HF access."
            .format(spec.descriptor)
        ) from exc

    if not isinstance(loaded, nn.Module):
        raise RuntimeError(
            "timm.create_model returned unsupported type for '{}': {}".format(
                spec.descriptor,
                type(loaded).__name__,
            )
        )
    return cast(nn.Module, loaded)


def load_frozen_backbone(spec: BackboneSpec, device: torch.device | None = None) -> FrozenBackbone:
    if spec.backend == "torch_hub":
        module = _load_torch_hub_backbone(spec)
    elif spec.backend == "timm_hf":
        module = _load_timm_hf_backbone(spec)
    else:
        raise ValueError(f"Unsupported backbone backend: {spec.backend}")

    wrapped = FrozenBackbone(module, spec)
    if device is not None:
        wrapped = wrapped.to(device)
    return wrapped
