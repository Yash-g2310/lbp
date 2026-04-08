from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .attention import FlowHAB
from .embeddings import FrequencyMixing2D, TimeConditionedAffine
from .utils import _crop_to_original, _pad_to_even, _pad_to_window_multiple, _resolve_num_groups


class FlowSFINBlock(nn.Module):
    """SFIN-style block with local/global paths, timestep conditioning, and checkpointing."""

    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        ffn_expansion: float = 2.0,
        dropout: float = 0.0,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if channels % 2 != 0:
            raise ValueError(
                f"FlowSFINBlock requires even channels for split local/global paths. Got {channels}."
            )
        if ffn_expansion <= 0.0:
            raise ValueError(f"ffn_expansion must be > 0, got {ffn_expansion}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0,1), got {dropout}")

        hidden = int(round(channels * ffn_expansion))
        half = channels // 2
        groups = _resolve_num_groups(channels)

        self.channels = int(channels)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.time_affine1 = TimeConditionedAffine(channels=channels, time_embed_dim=time_embed_dim)
        self.local_path = nn.Sequential(
            nn.Conv2d(half, half, kernel_size=3, padding=1, groups=max(1, half // 4), bias=False),
            nn.SiLU(),
            nn.Conv2d(half, half, kernel_size=1, bias=False),
        )
        self.global_path = nn.Sequential(
            FrequencyMixing2D(channels=half),
            nn.Conv2d(half, half, kernel_size=1, bias=False),
            nn.SiLU(),
        )
        self.mix = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.time_affine2 = TimeConditionedAffine(channels=channels, time_embed_dim=time_embed_dim)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, kernel_size=1),
        )
        self.dropout = nn.Dropout(p=dropout)

    def _forward_impl(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm1(x)
        h = self.time_affine1(h, t_emb)
        h_local, h_global = torch.chunk(h, chunks=2, dim=1)
        h_local = self.local_path(h_local)
        h_global = self.global_path(h_global)
        h = torch.cat([h_local, h_global], dim=1)
        h = self.mix(h)
        x = residual + self.dropout(h)

        residual = x
        h = self.norm2(x)
        h = self.time_affine2(h, t_emb)
        h = self.ffn(h)
        return residual + self.dropout(h)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training and x.requires_grad:
            return cast(
                torch.Tensor,
                torch_checkpoint(self._forward_impl, x, t_emb, use_reentrant=False),
            )
        return self._forward_impl(x, t_emb)


class FlowRHAG(nn.Module):
    """RHAG-style group with explicit odd-size and window-divisibility padding logic."""

    def __init__(
        self,
        channels: int,
        time_embed_dim: int,
        num_heads: int,
        num_blocks: int,
        window_size: int,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        pad_mode: str = "reflect",
        strict_76_mode: bool = True,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0, got {window_size}")

        self.channels = int(channels)
        self.window_size = int(window_size)
        self.pad_mode = pad_mode
        self.strict_76_mode = bool(strict_76_mode)
        self.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        if self.strict_76_mode and (76 % self.window_size) != 0:
            raise ValueError(
                "strict_76_mode=True requires window_size to divide 76 exactly. "
                f"Got window_size={self.window_size}."
            )

        shift_size = self.window_size // 2
        blocks = []
        for block_idx in range(num_blocks):
            blocks.append(
                FlowHAB(
                    channels=channels,
                    num_heads=num_heads,
                    window_size=window_size,
                    time_embed_dim=time_embed_dim,
                    shift_size=shift_size if (block_idx % 2 == 1) else 0,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def _run_hab(self, block: nn.Module, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if self.use_gradient_checkpointing and self.training and x.requires_grad:
            return cast(
                torch.Tensor,
                torch_checkpoint(block, x, t_emb, use_reentrant=False),
            )
        return cast(torch.Tensor, block(x, t_emb))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be rank-4 [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(f"FlowRHAG expected C={self.channels}, got C={x.shape[1]}")

        in_h, in_w = x.shape[-2:]

        x, even_info = _pad_to_even(x, pad_mode=self.pad_mode)
        if self.strict_76_mode and x.shape[-2:] != (76, 76):
            raise ValueError(
                f"strict_76_mode expected intermediate shape (76,76), got {tuple(x.shape[-2:])}"
            )

        x, window_info = _pad_to_window_multiple(
            x,
            window_size=self.window_size,
            pad_mode=self.pad_mode,
        )
        if x.shape[-2] % self.window_size != 0 or x.shape[-1] % self.window_size != 0:
            raise RuntimeError(
                "Window padding failed. "
                f"window_size={self.window_size}, HxW={tuple(x.shape[-2:])}"
            )

        shortcut = x
        for block in self.blocks:
            x = self._run_hab(block, x, t_emb)
        x = self.out_conv(x) + shortcut

        x = _crop_to_original(x, window_info)
        x = _crop_to_original(x, even_info)

        if x.shape[-2:] != (in_h, in_w):
            raise RuntimeError(
                "Padding/cropping invariance failed in FlowRHAG. "
                f"expected={(in_h, in_w)}, got={tuple(x.shape[-2:])}"
            )
        return x
