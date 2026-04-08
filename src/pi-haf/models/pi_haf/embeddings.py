from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for continuous timesteps."""

    def __init__(self, embed_dim: int, max_period: float = 10_000.0):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}")
        if max_period <= 1.0:
            raise ValueError(f"max_period must be > 1.0, got {max_period}")
        self.embed_dim = int(embed_dim)
        self.max_period = float(max_period)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[1] == 1:
            t = t.squeeze(1)
        if t.ndim != 1:
            raise ValueError(f"t must have shape [B] or [B,1], got {tuple(t.shape)}")

        half_dim = self.embed_dim // 2
        if half_dim == 0:
            return t[:, None]

        t = t.to(dtype=torch.float32)
        device = t.device
        freq = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=device, dtype=t.dtype))
            * torch.arange(0, half_dim, device=device, dtype=t.dtype)
            / float(max(1, half_dim - 1))
        )
        angles = t[:, None] * freq[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if self.embed_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeConditionedAffine(nn.Module):
    """Per-channel scale/shift modulation from timestep embeddings."""

    def __init__(self, channels: int, time_embed_dim: int):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if time_embed_dim <= 0:
            raise ValueError(f"time_embed_dim must be > 0, got {time_embed_dim}")

        self.channels = int(channels)
        self.to_scale_shift = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, channels * 2),
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be rank-4 [B,C,H,W], got {tuple(x.shape)}")
        if t_emb.ndim != 2:
            raise ValueError(f"t_emb must be rank-2 [B,D], got {tuple(t_emb.shape)}")
        if x.shape[0] != t_emb.shape[0]:
            raise ValueError(f"Batch mismatch x={x.shape[0]} t_emb={t_emb.shape[0]}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"TimeConditionedAffine expected C={self.channels}, got C={x.shape[1]}"
            )

        scale_shift = self.to_scale_shift(t_emb)
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=1)
        return x * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class FrequencyMixing2D(nn.Module):
    """FFT-based feature mixer inspired by SFIN global-frequency paths."""

    def __init__(self, channels: int):
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        self.channels = int(channels)
        self.mix = nn.Conv2d(channels * 2, channels * 2, kernel_size=1, bias=False)
        # Start near identity in frequency space; learn mixing strength over training.
        self.mix_gain = nn.Parameter(torch.zeros(1, channels * 2, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"x must be rank-4 [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"FrequencyMixing2D expected C={self.channels}, got C={x.shape[1]}"
            )

        compute_dtype = x.dtype
        freq = torch.fft.rfft2(x.float(), norm="ortho")
        freq_cat = torch.cat([freq.real, freq.imag], dim=1)
        freq_cat = freq_cat.to(dtype=compute_dtype)
        delta = self.mix(freq_cat)
        gain = torch.tanh(self.mix_gain).to(dtype=compute_dtype)
        mixed = (freq_cat + gain * delta).float()
        real, imag = torch.chunk(mixed, chunks=2, dim=1)
        freq_out = torch.complex(real, imag)
        restored = torch.fft.irfft2(freq_out, s=x.shape[-2:], norm="ortho")
        return restored.to(dtype=compute_dtype)
