from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_pair(name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor) -> None:
    if a.ndim != 4 or b.ndim != 4:
        raise ValueError(
            f"{name_a} and {name_b} must be rank-4 [B,C,H,W], got {tuple(a.shape)} and {tuple(b.shape)}"
        )
    if a.shape != b.shape:
        raise ValueError(f"{name_a} and {name_b} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if a.device != b.device:
        raise ValueError(f"{name_a} and {name_b} must be on same device")


def _validate_timestep(t: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    if t.ndim == 2 and t.shape[1] == 1:
        t = t.squeeze(1)
    if t.ndim != 1:
        raise ValueError(f"t must have shape [B] or [B,1], got {tuple(t.shape)}")
    if t.shape[0] != batch_size:
        raise ValueError(f"t batch mismatch: expected {batch_size}, got {t.shape[0]}")
    return t.to(device=device, dtype=torch.float32)


def fft_magnitude_weighting(shape_hw: Tuple[int, int], device: torch.device, exponent: float) -> torch.Tensor:
    """Generate radial frequency weights for rFFT2 output shape [H, W//2+1]."""
    h, w = shape_hw
    fy = torch.fft.fftfreq(h, d=1.0, device=device).abs()
    fx = torch.fft.rfftfreq(w, d=1.0, device=device).abs()
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    weights = torch.clamp(radius, min=1e-6).pow(exponent)
    return weights


class RectifiedFlowLoss(nn.Module):
    """L2 loss on velocity fields."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, pred_v: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
        _validate_pair("pred_v", pred_v, "target_v", target_v)
        return F.mse_loss(pred_v, target_v, reduction=self.reduction)


class MassConservationLoss(nn.Module):
    """Photometric mass conservation between pooled reconstructed HR and LR image."""

    def __init__(self, reduction: str = "mean", pool_kernel: int = 2):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        if pool_kernel <= 0:
            raise ValueError("pool_kernel must be > 0")
        self.reduction = reduction
        self.pool_kernel = int(pool_kernel)

    def forward(self, pred_hr: torch.Tensor, lr_imgs: torch.Tensor) -> torch.Tensor:
        if pred_hr.ndim != 4:
            raise ValueError(f"pred_hr must be rank-4 [B,C,H,W], got {tuple(pred_hr.shape)}")
        if lr_imgs.ndim != 4:
            raise ValueError(f"lr_imgs must be rank-4 [B,C,H,W], got {tuple(lr_imgs.shape)}")
        if pred_hr.shape[0] != lr_imgs.shape[0]:
            raise ValueError(
                f"Batch mismatch between pred_hr and lr_imgs: {pred_hr.shape[0]} vs {lr_imgs.shape[0]}"
            )
        if pred_hr.shape[1] != lr_imgs.shape[1]:
            raise ValueError(
                f"Channel mismatch between pred_hr and lr_imgs: {pred_hr.shape[1]} vs {lr_imgs.shape[1]}"
            )

        expected_h = lr_imgs.shape[-2] * self.pool_kernel
        expected_w = lr_imgs.shape[-1] * self.pool_kernel
        if pred_hr.shape[-2:] != (expected_h, expected_w):
            raise ValueError(
                "pred_hr spatial shape must be pool_kernel-times LR shape. "
                f"expected={(expected_h, expected_w)}, got={tuple(pred_hr.shape[-2:])}"
            )

        # Task6 tensors are normalized to [-1,1]; unnormalize before photometric pooling.
        unnorm_pred_hr = pred_hr * 0.5 + 0.5
        unnorm_lr = lr_imgs * 0.5 + 0.5
        pooled_pred = F.avg_pool2d(
            unnorm_pred_hr,
            kernel_size=self.pool_kernel,
            stride=self.pool_kernel,
        )

        if pooled_pred.shape != unnorm_lr.shape:
            raise RuntimeError(
                f"pooled_pred shape mismatch: expected {tuple(unnorm_lr.shape)}, got {tuple(pooled_pred.shape)}"
            )
        return F.mse_loss(pooled_pred, unnorm_lr, reduction=self.reduction)


class FocalFrequencyLoss(nn.Module):
    """Frequency-domain L2 with radial emphasis on higher frequencies."""

    def __init__(self, reduction: str = "mean", freq_exponent: float = 2.0):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        if freq_exponent < 0.0:
            raise ValueError("freq_exponent must be >= 0")
        self.reduction = reduction
        self.freq_exponent = float(freq_exponent)

    def forward(self, pred_hr: torch.Tensor, hr_imgs: torch.Tensor) -> torch.Tensor:
        _validate_pair("pred_hr", pred_hr, "hr_imgs", hr_imgs)

        pred_fft = torch.fft.rfft2(pred_hr.float(), norm="ortho")
        target_fft = torch.fft.rfft2(hr_imgs.float(), norm="ortho")

        weights = fft_magnitude_weighting(
            shape_hw=(pred_hr.shape[-2], pred_hr.shape[-1]),
            device=pred_hr.device,
            exponent=self.freq_exponent,
        ).to(dtype=torch.float32)

        diff = (pred_fft - target_fft).abs().pow(2)
        weighted = diff * weights[None, None, :, :]

        if self.reduction == "mean":
            return weighted.mean()
        return weighted.sum()


class CompositeRectifiedFlowLoss(nn.Module):
    """Weighted aggregate of flow, mass, and focal-frequency losses."""

    def __init__(
        self,
        flow_weight: float = 1.0,
        mass_weight: float = 0.1,
        freq_weight: float = 0.1,
        reduction: str = "mean",
        freq_exponent: float = 2.0,
    ):
        super().__init__()
        if flow_weight < 0 or mass_weight < 0 or freq_weight < 0:
            raise ValueError("All loss weights must be >= 0")

        self.flow_weight = float(flow_weight)
        self.mass_weight = float(mass_weight)
        self.freq_weight = float(freq_weight)

        self.flow_loss = RectifiedFlowLoss(reduction=reduction)
        self.mass_loss = MassConservationLoss(reduction=reduction)
        self.freq_loss = FocalFrequencyLoss(reduction=reduction, freq_exponent=freq_exponent)

    @staticmethod
    def reconstruct_pred_hr(v_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _validate_pair("v_pred", v_pred, "x_t", x_t)
        t_checked = _validate_timestep(t, batch_size=v_pred.shape[0], device=v_pred.device)
        t_view = t_checked.view(-1, 1, 1, 1).to(dtype=v_pred.dtype)
        return x_t + v_pred * (1.0 - t_view)

    def compute_components(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        lr_imgs: torch.Tensor,
        hr_imgs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        _validate_pair("v_pred", v_pred, "v_target", v_target)
        _validate_pair("x_t", x_t, "hr_imgs", hr_imgs)
        if lr_imgs.ndim != 4:
            raise ValueError(f"lr_imgs must be rank-4 [B,C,H,W], got {tuple(lr_imgs.shape)}")
        if lr_imgs.shape[0] != v_pred.shape[0]:
            raise ValueError(f"lr_imgs batch mismatch: expected {v_pred.shape[0]}, got {lr_imgs.shape[0]}")

        pred_hr = self.reconstruct_pred_hr(v_pred=v_pred, x_t=x_t, t=t)

        flow = self.flow_loss(v_pred, v_target)
        mass = self.mass_loss(pred_hr, lr_imgs)
        freq = self.freq_loss(pred_hr, hr_imgs)
        total = self.flow_weight * flow + self.mass_weight * mass + self.freq_weight * freq
        return {
            "total_loss": total,
            "flow_loss": flow,
            "mass_loss": mass,
            "freq_loss": freq,
        }

    def forward(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        lr_imgs: torch.Tensor,
        hr_imgs: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_components(v_pred, v_target, x_t, t, lr_imgs, hr_imgs)["total_loss"]


__all__ = [
    "RectifiedFlowLoss",
    "MassConservationLoss",
    "FocalFrequencyLoss",
    "CompositeRectifiedFlowLoss",
    "fft_magnitude_weighting",
]
