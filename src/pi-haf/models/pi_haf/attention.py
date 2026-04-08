from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .embeddings import TimeConditionedAffine
from .utils import _resolve_num_groups


class ChannelGate(nn.Module):
    """Simple channel gate (SE-like) for local channel recalibration."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if reduction <= 0:
            raise ValueError(f"reduction must be > 0, got {reduction}")

        hidden = max(1, channels // reduction)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class FlowHAB(nn.Module):
    """RHAG-inspired hybrid attention block operating on windowed self-attention."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        time_embed_dim: int,
        shift_size: int = 0,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if channels % num_heads != 0:
            raise ValueError(
                f"channels must be divisible by num_heads. got channels={channels}, heads={num_heads}"
            )
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")
        if shift_size < 0:
            raise ValueError(f"shift_size must be >= 0, got {shift_size}")
        if shift_size >= window_size:
            raise ValueError(
                f"shift_size must be < window_size. got shift_size={shift_size}, window_size={window_size}"
            )
        if mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be > 0, got {mlp_ratio}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        hidden = int(round(channels * mlp_ratio))
        groups = _resolve_num_groups(channels)

        self.channels = int(channels)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.time_affine1 = TimeConditionedAffine(channels=channels, time_embed_dim=time_embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.channel_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            ChannelGate(channels=channels),
        )

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.time_affine2 = TimeConditionedAffine(channels=channels, time_embed_dim=time_embed_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.dropout = nn.Dropout(p=dropout)

    def _partition_windows(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        b, c, h, w = x.shape
        ws = self.window_size
        if h % ws != 0 or w % ws != 0:
            raise ValueError(
                f"FlowHAB requires H and W divisible by {ws}. Got HxW={h}x{w}."
            )

        x = x.view(b, c, h // ws, ws, w // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = x.view(-1, ws * ws, c)
        return windows, h, w

    def _reverse_windows(self, windows: torch.Tensor, b: int, h: int, w: int) -> torch.Tensor:
        ws = self.window_size
        x = windows.view(b, h // ws, w // ws, ws, ws, self.channels)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(b, self.channels, h, w)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise ValueError(f"FlowHAB expected C={self.channels}, got C={x.shape[1]}")

        b = x.shape[0]
        residual = x

        h = self.norm1(x)
        h = self.time_affine1(h, t_emb)
        if self.shift_size > 0:
            h = torch.roll(
                h,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(-2, -1),
            )
        windows, padded_h, padded_w = self._partition_windows(h)
        attn_out, _ = self.attn(windows, windows, windows, need_weights=False)
        h = self._reverse_windows(attn_out, b=b, h=padded_h, w=padded_w)
        if self.shift_size > 0:
            h = torch.roll(
                h,
                shifts=(self.shift_size, self.shift_size),
                dims=(-2, -1),
            )
        h = self.proj(h)
        x = residual + self.dropout(h)

        x = x + self.channel_conv(x)

        residual = x
        h = self.norm2(x)
        h = self.time_affine2(h, t_emb)
        h = self.ffn(h)
        return residual + self.dropout(h)
