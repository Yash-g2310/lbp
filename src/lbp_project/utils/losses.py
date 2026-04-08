import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


def _validate_rank4(name: str, x: torch.Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"{name} must be rank-4 [B,C,H,W], got {tuple(x.shape)}")


def normalize_flow_timestep(
    t: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if t.ndim == 0:
        t = t.view(1).expand(batch_size)
    if t.ndim == 2 and t.shape[1] == 1:
        t = t.squeeze(1)
    if t.ndim != 1 or t.shape[0] != batch_size:
        raise ValueError(f"flow timestep must have shape [B] or [B,1], got {tuple(t.shape)}")

    t = t.to(device=device, dtype=dtype)
    if not torch.isfinite(t).all():
        raise ValueError("flow timestep contains non-finite values")
    if ((t < 0.0) | (t > 1.0)).any():
        raise ValueError("flow timestep values must be in [0, 1]")
    return t


def reconstruct_depth_from_velocity(
    noisy_depth: torch.Tensor,
    velocity: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    _validate_rank4("noisy_depth", noisy_depth)
    _validate_rank4("velocity", velocity)
    if noisy_depth.shape != velocity.shape:
        raise ValueError(
            "noisy_depth and velocity must have identical shapes, "
            f"got {tuple(noisy_depth.shape)} vs {tuple(velocity.shape)}"
        )

    t_checked = normalize_flow_timestep(
        t,
        batch_size=noisy_depth.shape[0],
        device=noisy_depth.device,
        dtype=noisy_depth.dtype,
    )
    t_view = t_checked.view(-1, 1, 1, 1)
    return noisy_depth + velocity * (1.0 - t_view)


def depth_to_inverse_normalized(
    depth: torch.Tensor,
    eps: float = 1.0e-6,
    invalid_fill: float = -1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map positive metric depth to inverse-depth normalized space in [-1, 1]."""
    _validate_rank4("depth", depth)
    valid = torch.isfinite(depth) & (depth > 0)

    safe_depth = torch.where(valid, torch.clamp(depth, min=eps), torch.ones_like(depth))
    inv = torch.where(valid, torch.reciprocal(safe_depth), torch.zeros_like(depth))

    inv_max = inv.flatten(1).amax(dim=1, keepdim=True).view(-1, 1, 1, 1)
    inv_max = torch.clamp(inv_max, min=eps)

    normalized = (2.0 * (inv / inv_max)) - 1.0
    normalized = torch.where(valid, normalized, depth.new_full(depth.shape, invalid_fill))
    return normalized, valid, inv_max


def inverse_normalized_to_metric_depth(
    depth_norm: torch.Tensor,
    inv_max: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Map inverse-depth normalized space back to positive metric depth."""
    _validate_rank4("depth_norm", depth_norm)
    _validate_rank4("inv_max", inv_max)
    if inv_max.shape[0] != depth_norm.shape[0] or inv_max.shape[1] != 1:
        raise ValueError(
            "inv_max must be [B,1,1,1] and batch-aligned with depth_norm, "
            f"got depth_norm={tuple(depth_norm.shape)} inv_max={tuple(inv_max.shape)}"
        )

    inv = ((depth_norm + 1.0) * 0.5) * torch.clamp(inv_max, min=eps)
    out = depth_norm.new_zeros(depth_norm.shape)

    if valid_mask is None:
        valid = torch.isfinite(inv) & (inv > eps)
    else:
        if valid_mask.shape != depth_norm.shape:
            raise ValueError(
                "valid_mask must match depth_norm shape. "
                f"mask={tuple(valid_mask.shape)} depth_norm={tuple(depth_norm.shape)}"
            )
        valid = valid_mask.to(dtype=torch.bool) & torch.isfinite(inv) & (inv > eps)

    if bool(valid.any()):
        out[valid] = torch.reciprocal(torch.clamp(inv[valid], min=eps))
    return out


class SILogLoss(nn.Module):
    def __init__(self, lambd=0.5, strict_empty_target: bool = False):
        super().__init__()
        self.lambd = lambd
        self.strict_empty_target = bool(strict_empty_target)
        self._warned_empty_target = False

    def forward(self, pred, target):
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

        valid_mask = torch.isfinite(target) & (target > 0)
        if valid_mask.sum() == 0:
            invalid_cnt = int((~torch.isfinite(target)).sum().item())
            non_positive_cnt = int(((target <= 0) & torch.isfinite(target)).sum().item())

            # Some samples (especially layer-2) may legitimately have no valid depth.
            # In non-strict mode, treat this as "no supervision" for this term.
            if not self.strict_empty_target:
                if not self._warned_empty_target:
                    warnings.warn(
                        "SILogLoss: empty valid target mask encountered; returning tiny no-op loss.",
                        RuntimeWarning,
                    )
                    self._warned_empty_target = True
                return pred.new_tensor(1e-6)

            pred_finite = pred[torch.isfinite(pred)]
            if pred_finite.numel() > 0:
                pred_min = float(pred_finite.min().item())
                pred_max = float(pred_finite.max().item())
            else:
                pred_min = float("nan")
                pred_max = float("nan")
            raise RuntimeError(
                "SILogLoss received empty valid target mask. "
                f"invalid_target={invalid_cnt} non_positive_target={non_positive_cnt} "
                f"pred_min={pred_min:.6g} pred_max={pred_max:.6g}"
            )

        pred_valid = torch.clamp(pred[valid_mask], min=1e-6)
        tgt_valid = torch.clamp(target[valid_mask], min=1e-6)

        finite_mask = torch.isfinite(pred_valid) & torch.isfinite(tgt_valid)
        if finite_mask.sum() == 0:
            invalid_pred_cnt = int((~torch.isfinite(pred_valid)).sum().item())
            invalid_tgt_cnt = int((~torch.isfinite(tgt_valid)).sum().item())
            pred_total = int(pred_valid.numel())
            tgt_total = int(tgt_valid.numel())
            raise RuntimeError(
                "SILogLoss finite filtering removed all samples. "
                f"invalid_pred={invalid_pred_cnt}/{pred_total} invalid_target={invalid_tgt_cnt}/{tgt_total} "
                f"pred_finite_before_filter={int(torch.isfinite(pred_valid).sum().item())} "
                f"target_finite_before_filter={int(torch.isfinite(tgt_valid).sum().item())}"
            )
        pred_valid = pred_valid[finite_mask]
        tgt_valid = tgt_valid[finite_mask]

        diff_log = torch.log(pred_valid) - torch.log(tgt_valid)
        silog_term = torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
        loss = torch.sqrt(torch.clamp(silog_term, min=1e-12))
        return loss


class ScaleShiftInvariantLoss(nn.Module):
    """Signed-domain scale-and-shift invariant depth consistency loss."""

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_rank4("pred", pred)
        _validate_rank4("target", target)
        if pred.shape != target.shape:
            raise ValueError(
                "pred/target shape mismatch for SSI loss: "
                f"pred={tuple(pred.shape)} target={tuple(target.shape)}"
            )

        if valid_mask is not None:
            if valid_mask.shape != pred.shape:
                raise ValueError(
                    "valid_mask must match pred shape. "
                    f"mask={tuple(valid_mask.shape)} pred={tuple(pred.shape)}"
                )
            base_valid = valid_mask.to(dtype=torch.bool)
        else:
            base_valid = torch.ones_like(pred, dtype=torch.bool)

        finite_valid = base_valid & torch.isfinite(pred) & torch.isfinite(target)
        bsz = pred.shape[0]
        sample_losses: list[torch.Tensor] = []

        for i in range(bsz):
            valid_i = finite_valid[i].reshape(-1)
            if int(valid_i.sum().item()) < 2:
                continue

            pred_i = pred[i].reshape(-1)[valid_i]
            target_i = target[i].reshape(-1)[valid_i]

            sum_p = pred_i.sum()
            sum_t = target_i.sum()
            sum_p2 = (pred_i * pred_i).sum()
            sum_pt = (pred_i * target_i).sum()
            n = pred_i.new_tensor(float(pred_i.numel()))

            det = n * sum_p2 - sum_p * sum_p
            safe = torch.abs(det) > self.eps
            det_safe = torch.where(safe, det, torch.ones_like(det))

            # Closed-form least-squares fit for target ≈ a*pred + b.
            a = torch.where(
                safe,
                (n * sum_pt - sum_p * sum_t) / det_safe,
                pred_i.new_tensor(1.0),
            )
            b = torch.where(
                safe,
                (sum_t - a * sum_p) / torch.clamp(n, min=1.0),
                pred_i.new_tensor(0.0),
            )

            residual = a * pred_i + b - target_i
            sample_losses.append(residual.abs().mean())

        if not sample_losses:
            return pred.new_tensor(0.0)
        return torch.stack(sample_losses).mean()


class RectifiedFlowLoss(nn.Module):
    """L2 objective for velocity prediction in rectified flow."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.reduction = reduction

    def forward(
        self,
        pred_velocity: torch.Tensor,
        target_velocity: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _validate_rank4("pred_velocity", pred_velocity)
        _validate_rank4("target_velocity", target_velocity)
        if pred_velocity.shape != target_velocity.shape:
            raise ValueError(
                "velocity shape mismatch: "
                f"pred={tuple(pred_velocity.shape)} target={tuple(target_velocity.shape)}"
            )

        if valid_mask is not None:
            if valid_mask.shape != pred_velocity.shape:
                raise ValueError(
                    "valid_mask must match velocity tensors. "
                    f"mask={tuple(valid_mask.shape)} pred={tuple(pred_velocity.shape)}"
                )
            valid = valid_mask.to(dtype=torch.bool) & torch.isfinite(pred_velocity) & torch.isfinite(target_velocity)
            if int(valid.sum().item()) == 0:
                return pred_velocity.new_tensor(0.0)
            diff = pred_velocity[valid] - target_velocity[valid]
            if self.reduction == "sum":
                return (diff * diff).sum()
            return (diff * diff).mean()

        return F.mse_loss(pred_velocity, target_velocity, reduction=self.reduction)


class OrdinalDepthLoss(nn.Module):
    """Hinge loss enforcing near-layer depth < far-layer depth on overlapping valid pixels."""

    def __init__(self, margin: float = 1.0e-2):
        super().__init__()
        self.margin = float(margin)
        if self.margin < 0.0:
            raise ValueError(f"margin must be >= 0, got {self.margin}")

    def forward(
        self,
        pred_near: torch.Tensor,
        pred_far: torch.Tensor,
        target_near: torch.Tensor,
        target_far: torch.Tensor,
    ) -> torch.Tensor:
        if pred_near.shape != pred_far.shape:
            raise ValueError(f"pred shape mismatch: near={pred_near.shape}, far={pred_far.shape}")
        if target_near.shape != target_far.shape:
            raise ValueError(f"target shape mismatch: near={target_near.shape}, far={target_far.shape}")
        if pred_near.shape != target_near.shape:
            raise ValueError(
                f"prediction/target shape mismatch: pred={pred_near.shape}, target={target_near.shape}"
            )

        valid = (
            torch.isfinite(pred_near)
            & torch.isfinite(pred_far)
            & torch.isfinite(target_near)
            & torch.isfinite(target_far)
            & (target_near > 0)
            & (target_far > 0)
        )
        if valid.sum() == 0:
            return pred_near.new_tensor(0.0)

        near_vals = pred_near[valid]
        far_vals = pred_far[valid]
        return F.relu(near_vals - far_vals + self.margin).mean()


class EdgeAwareSmoothnessLoss(nn.Module):
    """Image-aware depth smoothness prior for stable geometry in low-texture regions."""

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(
        self,
        pred_depth: torch.Tensor,
        image: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred_depth.ndim != 4 or pred_depth.shape[1] != 1:
            raise ValueError(f"pred_depth must be [B,1,H,W], got {tuple(pred_depth.shape)}")
        if image.ndim != 4:
            raise ValueError(f"image must be [B,C,H,W], got {tuple(image.shape)}")
        if pred_depth.shape[0] != image.shape[0] or pred_depth.shape[-2:] != image.shape[-2:]:
            raise ValueError(
                "pred_depth and image must share batch/spatial dimensions. "
                f"pred={tuple(pred_depth.shape)}, image={tuple(image.shape)}"
            )

        pred_dx = torch.abs(pred_depth[:, :, :, 1:] - pred_depth[:, :, :, :-1])
        pred_dy = torch.abs(pred_depth[:, :, 1:, :] - pred_depth[:, :, :-1, :])

        img_dx = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]), dim=1, keepdim=True)
        img_dy = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]), dim=1, keepdim=True)

        weight_x = torch.exp(-img_dx)
        weight_y = torch.exp(-img_dy)

        if valid_mask is not None:
            if valid_mask.shape != pred_depth.shape:
                raise ValueError(
                    "valid_mask must match pred_depth shape. "
                    f"mask={tuple(valid_mask.shape)}, pred={tuple(pred_depth.shape)}"
                )
            valid_mask = valid_mask.to(dtype=torch.bool)
            mask_x = valid_mask[:, :, :, 1:] & valid_mask[:, :, :, :-1]
            mask_y = valid_mask[:, :, 1:, :] & valid_mask[:, :, :-1, :]
            denom_x = mask_x.float().sum().clamp_min(self.eps)
            denom_y = mask_y.float().sum().clamp_min(self.eps)
            smooth_x = (pred_dx * weight_x * mask_x.float()).sum() / denom_x
            smooth_y = (pred_dy * weight_y * mask_y.float()).sum() / denom_y
            return smooth_x + smooth_y

        return (pred_dx * weight_x).mean() + (pred_dy * weight_y).mean()


class WaveletEdgeLoss(nn.Module):
    """Pure-Torch multi-level wavelet edge consistency loss."""

    _FILTERS = {
        "sym4": {
            "dec_lo": [
                -0.07576571478927333,
                -0.02963552764599851,
                0.49761866763201545,
                0.8037387518059161,
                0.29785779560527736,
                -0.09921954357684722,
                -0.012603967262037833,
                0.0322231006040427,
            ],
            "dec_hi": [
                -0.0322231006040427,
                -0.012603967262037833,
                0.09921954357684722,
                0.29785779560527736,
                -0.8037387518059161,
                0.49761866763201545,
                0.02963552764599851,
                -0.07576571478927333,
            ],
        },
        "bior3.5": {
            "dec_lo": [
                -0.013810679320049757,
                0.04143203796014927,
                0.052480581416189075,
                -0.26792717880896527,
                -0.07181553246425873,
                0.966747552403483,
                0.966747552403483,
                -0.07181553246425873,
                -0.26792717880896527,
                0.052480581416189075,
                0.04143203796014927,
                -0.013810679320049757,
            ],
            "dec_hi": [
                0.0,
                0.0,
                0.0,
                0.0,
                -0.1767766952966369,
                0.5303300858899106,
                -0.5303300858899106,
                0.1767766952966369,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        },
    }

    def __init__(
        self,
        family: str = "sym4",
        levels: int = 2,
        fallback_family: str = "bior3.5",
    ):
        super().__init__()
        self.family = str(family).strip().lower()
        self.fallback_family = str(fallback_family).strip().lower()
        self.levels = int(levels)
        if self.levels < 1:
            raise ValueError(f"levels must be >= 1, got {self.levels}")
        self._validate_family(self.family)
        self._validate_family(self.fallback_family)

    def _validate_family(self, family: str) -> None:
        if family not in self._FILTERS:
            supported = ", ".join(sorted(self._FILTERS.keys()))
            raise ValueError(f"Unsupported wavelet family '{family}'. Supported: {supported}")

    def _make_analysis_weight(
        self,
        channels: int,
        family: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, int]:
        coeffs = self._FILTERS[family]
        lo = torch.tensor(coeffs["dec_lo"], device=device, dtype=dtype)
        hi = torch.tensor(coeffs["dec_hi"], device=device, dtype=dtype)
        kernel_size = int(lo.numel())
        if int(hi.numel()) != kernel_size:
            raise RuntimeError(f"Invalid filter-bank length mismatch for family '{family}'")

        ll = torch.outer(lo, lo)
        lh = torch.outer(lo, hi)
        hl = torch.outer(hi, lo)
        hh = torch.outer(hi, hi)
        base = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4,1,K,K]
        weight = base.repeat(channels, 1, 1, 1).contiguous()  # [4*C,1,K,K]
        # Keep output close to half-resolution for even input dimensions.
        pad = max(0, (kernel_size - 2) // 2)
        return weight, pad

    def _decompose(
        self,
        x: torch.Tensor,
        family: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _validate_rank4("wavelet_input", x)
        b, c, _, _ = x.shape
        weight, pad = self._make_analysis_weight(c, family, x.device, x.dtype)
        coeffs = F.conv2d(x, weight=weight, stride=2, padding=pad, groups=c)
        coeffs = coeffs.view(b, c, 4, coeffs.shape[-2], coeffs.shape[-1])
        ll = coeffs[:, :, 0]
        lh = coeffs[:, :, 1]
        hl = coeffs[:, :, 2]
        hh = coeffs[:, :, 3]
        return ll, lh, hl, hh

    def _forward_family(self, pred_depth: torch.Tensor, target_depth: torch.Tensor, family: str) -> torch.Tensor:
        _validate_rank4("pred_depth", pred_depth)
        _validate_rank4("target_depth", target_depth)
        if pred_depth.shape != target_depth.shape:
            raise ValueError(
                "pred_depth and target_depth must match for wavelet loss, "
                f"got {tuple(pred_depth.shape)} vs {tuple(target_depth.shape)}"
            )

        pred = pred_depth
        target = target_depth
        losses: list[torch.Tensor] = []

        for _ in range(self.levels):
            if min(pred.shape[-2], pred.shape[-1]) < 2:
                break
            ll_p, lh_p, hl_p, hh_p = self._decompose(pred, family)
            ll_t, lh_t, hl_t, hh_t = self._decompose(target, family)

            band = (lh_p - lh_t).abs().mean() + (hl_p - hl_t).abs().mean() + (hh_p - hh_t).abs().mean()
            if not torch.isfinite(band):
                raise RuntimeError(f"Non-finite wavelet edge loss band encountered for family '{family}'")
            losses.append(band)

            pred = ll_p
            target = ll_t

        if not losses:
            return pred_depth.new_tensor(0.0)
        return torch.stack(losses).mean()

    def forward(self, pred_depth: torch.Tensor, target_depth: torch.Tensor) -> torch.Tensor:
        try:
            return self._forward_family(pred_depth, target_depth, self.family)
        except Exception as exc:
            if self.fallback_family == self.family:
                raise
            warnings.warn(
                (
                    f"WaveletEdgeLoss primary family '{self.family}' failed ({exc}); "
                    f"retrying with fallback '{self.fallback_family}'."
                ),
                RuntimeWarning,
            )
            return self._forward_family(pred_depth, target_depth, self.fallback_family)