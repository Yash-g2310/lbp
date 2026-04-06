"""Checkpoint loading helpers shared across evaluation and orchestration scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> Dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=device)


def model_state_dict_from_checkpoint(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    raise KeyError("Checkpoint missing model state keys: expected model_state or model_state_dict")


def load_checkpoint_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    strict: bool = True,
    allow_missing_prefixes: Iterable[str] | None = None,
) -> Tuple[list[str], list[str]]:
    checkpoint = load_checkpoint(checkpoint_path, device)
    state_dict = model_state_dict_from_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    allow_prefixes = tuple(allow_missing_prefixes or ())
    if allow_prefixes and missing:
        missing = [k for k in missing if not k.startswith(allow_prefixes)]

    return list(missing), list(unexpected)
