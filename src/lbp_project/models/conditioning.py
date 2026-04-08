"""Conditioning primitives used by AdaLN-Zero in the active lbp_project runtime."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_num_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for normalized timesteps in [0, 1]."""

    def __init__(self, embed_dim: int, max_period: float = 10_000.0):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        if max_period <= 1.0:
            raise ValueError(f"max_period must be > 1.0, got {max_period}")

        self.embed_dim = int(embed_dim)
        self.max_period = float(max_period)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps.squeeze(1)
        if timesteps.ndim != 1:
            raise ValueError(
                f"timesteps must have shape [B] or [B,1], got {tuple(timesteps.shape)}"
            )

        half_dim = self.embed_dim // 2
        if half_dim == 0:
            return timesteps[:, None]

        t = timesteps.to(dtype=torch.float32)
        freq = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=t.device, dtype=t.dtype))
            * torch.arange(0, half_dim, device=t.device, dtype=t.dtype)
            / float(max(1, half_dim - 1))
        )
        angles = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class AdaLNZero2d(nn.Module):
    """AdaLN-Zero modulation block with identity-preserving zero initialization."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if cond_dim <= 0:
            raise ValueError(f"cond_dim must be > 0, got {cond_dim}")

        self.channels = int(channels)
        self.cond_dim = int(cond_dim)
        self.norm = nn.GroupNorm(
            num_groups=_resolve_num_groups(self.channels),
            num_channels=self.channels,
            affine=False,
        )
        self.modulation_act = nn.SiLU()
        self.modulation_fc = nn.Linear(self.cond_dim, self.channels * 3)

        # AdaLN-Zero requirement: modulation starts as strict identity.
        nn.init.zeros_(self.modulation_fc.weight)
        nn.init.zeros_(self.modulation_fc.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be rank-4 [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"AdaLNZero2d expected C={self.channels}, got C={x.shape[1]}"
            )
        if cond.ndim != 2:
            raise ValueError(f"cond must be rank-2 [B,D], got {tuple(cond.shape)}")
        if cond.shape[0] != x.shape[0]:
            raise ValueError(f"Batch mismatch x={x.shape[0]} cond={cond.shape[0]}")
        if cond.shape[1] != self.cond_dim:
            raise ValueError(
                f"AdaLNZero2d expected cond_dim={self.cond_dim}, got {cond.shape[1]}"
            )

        params = self.modulation_fc(self.modulation_act(cond))
        shift, scale, gate = torch.chunk(params, chunks=3, dim=1)
        h = self.norm(x)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + gate[:, :, None, None] * h
