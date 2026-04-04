import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings


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