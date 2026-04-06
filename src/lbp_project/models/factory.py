"""Factory helpers for constructing configured model instances."""

from __future__ import annotations

from typing import Any, Dict

import torch

from lbp_project.models.wrapper import DINOSFIN_Architecture_NEW


def build_depth_model(
    cfg: Dict[str, Any],
    device: torch.device,
    use_precomputed_dino: bool | None = None,
) -> torch.nn.Module:
    if use_precomputed_dino is None:
        use_precomputed_dino = bool(cfg["data"].get("use_precomputed_dino", False))

    model = DINOSFIN_Architecture_NEW(
        strategy=cfg["architecture"]["strategy"],
        base_channels=int(cfg["architecture"]["base_channels"]),
        num_sfin=int(cfg["architecture"]["num_sfin"]),
        num_rhag=int(cfg["architecture"]["num_rhag"]),
        window_size=int(cfg["architecture"]["window_size"]),
        dino_embed_dim=int(cfg["architecture"]["dino_embed_dim"]),
        fft_mode=cfg["architecture"]["fft_mode"],
        fft_pad_size=int(cfg["architecture"]["fft_pad_size"]),
        use_precomputed_dino=bool(use_precomputed_dino),
    ).to(device)
    return model
