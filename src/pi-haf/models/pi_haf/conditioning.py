from __future__ import annotations

from typing import List, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .utils import _resolve_num_groups, _validate_nchw


class LRConditionEncoder(nn.Module):
    """Lightweight LR feature extractor for cross-attention key/value features."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_feature_levels: int,
    ):
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be > 0, got {in_channels}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be > 0, got {hidden_channels}")
        if num_feature_levels <= 0:
            raise ValueError(f"num_feature_levels must be > 0, got {num_feature_levels}")

        groups = _resolve_num_groups(hidden_channels)
        self.stem = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(num_groups=groups, num_channels=hidden_channels),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                    nn.SiLU(),
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                )
                for _ in range(num_feature_levels)
            ]
        )

    def forward(self, lr_img: torch.Tensor) -> List[torch.Tensor]:
        _validate_nchw("lr_img", lr_img)
        x = self.stem(lr_img)
        outputs: List[torch.Tensor] = []
        for block in self.blocks:
            x = x + block(x)
            outputs.append(x)
        return outputs


class CrossAttention2D(nn.Module):
    """2D cross-attention: query from backbone features, key/value from LR features."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
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
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        self.channels = int(channels)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.norm_q = nn.GroupNorm(num_groups=_resolve_num_groups(channels), num_channels=channels)
        self.norm_kv = nn.GroupNorm(num_groups=_resolve_num_groups(channels), num_channels=channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def _forward_impl(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        if q_feat.ndim != 4 or kv_feat.ndim != 4:
            raise ValueError(
                f"q_feat and kv_feat must be rank-4. got {tuple(q_feat.shape)} and {tuple(kv_feat.shape)}"
            )
        if q_feat.shape[1] != self.channels or kv_feat.shape[1] != self.channels:
            raise ValueError(
                "CrossAttention2D channel mismatch. "
                f"expected C={self.channels}, got q={q_feat.shape[1]}, kv={kv_feat.shape[1]}"
            )

        if q_feat.shape[-2:] != kv_feat.shape[-2:]:
            kv_feat = F.interpolate(
                kv_feat,
                size=q_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        b, c, h, w = q_feat.shape
        q_tokens = self.norm_q(q_feat).flatten(2).transpose(1, 2)
        kv_tokens = self.norm_kv(kv_feat).flatten(2).transpose(1, 2)
        attn_tokens, _ = self.attn(q_tokens, kv_tokens, kv_tokens, need_weights=False)
        attn_map = attn_tokens.transpose(1, 2).reshape(b, c, h, w)
        attn_map = self.out_proj(attn_map)
        return q_feat + torch.tanh(self.gate) * attn_map

    def forward(self, q_feat: torch.Tensor, kv_feat: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training and q_feat.requires_grad:
            return cast(
                torch.Tensor,
                torch_checkpoint(self._forward_impl, q_feat, kv_feat, use_reentrant=False),
            )
        return self._forward_impl(q_feat, kv_feat)
