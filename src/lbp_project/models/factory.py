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

    adaln_cfg = cfg["architecture"].get("adaln_zero", {}) or {}

    model = DINOSFIN_Architecture_NEW(
        strategy=cfg["architecture"]["strategy"],
        base_channels=int(cfg["architecture"]["base_channels"]),
        num_sfin=int(cfg["architecture"]["num_sfin"]),
        num_rhag=int(cfg["architecture"]["num_rhag"]),
        window_size=int(cfg["architecture"]["window_size"]),
        dino_embed_dim=int(cfg["architecture"]["dino_embed_dim"]),
        backbone_repo=str(cfg["architecture"].get("backbone_repo", "timm")),
        backbone_model=str(
            cfg["architecture"].get("backbone_model", "timm/convnext_small.dinov3_lvd1689m")
        ),
        backbone_backend=str(cfg["architecture"].get("backbone_backend", "")),
        backbone_fallback_models=cfg["architecture"].get("backbone_fallback_models"),
        backbone_stop_on_failure=bool(cfg["architecture"].get("backbone_stop_on_failure", True)),
        backbone_fallback_approved=bool(cfg["architecture"].get("backbone_fallback_approved", False)),
        max_layer_id=int(cfg["architecture"].get("max_layer_id", 8)),
        enable_velocity_head=bool(cfg["architecture"].get("enable_velocity_head", False)),
        velocity_hidden_channels=int(cfg["architecture"].get("velocity_hidden_channels", 64)),
        fft_mode=cfg["architecture"]["fft_mode"],
        fft_pad_size=int(cfg["architecture"]["fft_pad_size"]),
        adaln_zero_enabled=bool(adaln_cfg.get("enabled", False)),
        adaln_layer_embed_dim=int(adaln_cfg.get("layer_embed_dim", 64)),
        adaln_time_embed_dim=int(adaln_cfg.get("time_embed_dim", 64)),
        adaln_condition_dim=int(adaln_cfg.get("condition_dim", 128)),
        adaln_timestep_default=float(adaln_cfg.get("timestep_default", 1.0)),
        use_precomputed_dino=bool(use_precomputed_dino),
    ).to(device)
    return model
