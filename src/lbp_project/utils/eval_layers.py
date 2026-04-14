"""Helpers for tuple-driven layer expansion and deterministic multi-pass inference."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Iterable, List

import torch

from lbp_project.utils.metrics import extract_required_layer_ids


def resolve_required_layers(
    sample: dict,
    layer_key: str,
    target_layer_override: int | None = None,
    max_layer_id: int | None = None,
) -> List[int]:
    if target_layer_override is not None and int(target_layer_override) > 0:
        target = int(target_layer_override)
        if max_layer_id is not None and target > int(max_layer_id):
            return []
        return [target]

    required = extract_required_layer_ids(sample, layer_key=layer_key)
    required = sorted({int(layer_id) for layer_id in required if int(layer_id) > 0})
    if max_layer_id is not None:
        max_layer_id = int(max_layer_id)
        required = [layer_id for layer_id in required if layer_id <= max_layer_id]
    return required


def predict_depth_by_layer(
    model: torch.nn.Module,
    image: torch.Tensor,
    layer_ids: Iterable[int],
    amp_enabled: bool,
    device: torch.device,
    precomputed_dino: torch.Tensor | None = None,
) -> Dict[int, torch.Tensor]:
    predictions: Dict[int, torch.Tensor] = {}
    ordered_layers = sorted({int(layer_id) for layer_id in layer_ids if int(layer_id) > 0})

    for layer_id in ordered_layers:
        amp_ctx = (
            torch.autocast(device_type="cuda", enabled=amp_enabled)
            if device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            predictions[layer_id] = model(
                image,
                target_layer=layer_id,
                return_intermediate=False,
                precomputed_dino=precomputed_dino,
            )

    return predictions
