from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _resolve_num_groups(channels: int, preferred_max: int = 8) -> int:
    """Select a valid GroupNorm group count for the given channel width."""
    if channels <= 0:
        raise ValueError(f"channels must be > 0, got {channels}")
    for groups in range(min(preferred_max, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


@dataclass(frozen=True)
class SpatialPadInfo:
    """Metadata describing right/bottom padding for deterministic crop-back."""

    original_h: int
    original_w: int
    pad_right: int
    pad_bottom: int


def _validate_nchw(
    name: str,
    x: torch.Tensor,
    expected_channels: Optional[int] = None,
    expected_hw: Optional[Tuple[int, int]] = None,
) -> None:
    """Fail-fast validation for NCHW tensors."""
    if x.ndim != 4:
        raise ValueError(f"{name} must be rank-4 [B,C,H,W], got shape {tuple(x.shape)}")
    if expected_channels is not None and x.shape[1] != expected_channels:
        raise ValueError(
            f"{name} channel mismatch: expected C={expected_channels}, got C={x.shape[1]}"
        )
    if expected_hw is not None and x.shape[-2:] != expected_hw:
        raise ValueError(
            f"{name} spatial mismatch: expected HxW={expected_hw}, got HxW={tuple(x.shape[-2:])}"
        )


def _pad_right_bottom(
    x: torch.Tensor,
    pad_right: int,
    pad_bottom: int,
    pad_mode: str,
) -> Tuple[torch.Tensor, SpatialPadInfo]:
    """Pad right and bottom edges and return crop metadata."""
    if pad_right < 0 or pad_bottom < 0:
        raise ValueError(
            f"pad_right and pad_bottom must be >= 0, got ({pad_right}, {pad_bottom})"
        )
    if pad_mode not in {"constant", "reflect", "replicate", "circular"}:
        raise ValueError(
            f"Unsupported pad_mode '{pad_mode}'. Use constant|reflect|replicate|circular."
        )

    h, w = x.shape[-2:]
    info = SpatialPadInfo(
        original_h=h,
        original_w=w,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )
    if pad_right == 0 and pad_bottom == 0:
        return x, info
    return F.pad(x, (0, pad_right, 0, pad_bottom), mode=pad_mode), info


def _crop_to_original(x: torch.Tensor, info: SpatialPadInfo) -> torch.Tensor:
    """Crop tensor back to its original size before right/bottom padding."""
    return x[..., : info.original_h, : info.original_w]


def _pad_to_even(x: torch.Tensor, pad_mode: str) -> Tuple[torch.Tensor, SpatialPadInfo]:
    """Pad odd spatial dimensions to even dimensions by adding one pixel at most."""
    h, w = x.shape[-2:]
    pad_bottom = h % 2
    pad_right = w % 2
    return _pad_right_bottom(
        x,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        pad_mode=pad_mode,
    )


def _pad_to_window_multiple(
    x: torch.Tensor,
    window_size: int,
    pad_mode: str,
) -> Tuple[torch.Tensor, SpatialPadInfo]:
    """Pad H/W to nearest multiple of window_size using right/bottom padding."""
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")

    h, w = x.shape[-2:]
    pad_bottom = (window_size - (h % window_size)) % window_size
    pad_right = (window_size - (w % window_size)) % window_size
    return _pad_right_bottom(
        x,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
        pad_mode=pad_mode,
    )
