import torch
import torch.nn as nn
import torch.nn.functional as F


class SILogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target):
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[-2:], mode='bilinear', align_corners=False)

        valid_mask = torch.isfinite(target) & (target > 0)
        if valid_mask.sum() == 0:
            invalid_cnt = int((~torch.isfinite(target)).sum().item())
            non_positive_cnt = int(((target <= 0) & torch.isfinite(target)).sum().item())
            raise RuntimeError(
                "SILogLoss received empty valid target mask. "
                f"invalid_target={invalid_cnt} non_positive_target={non_positive_cnt} "
                f"pred_min={float(torch.nanmin(pred).item()):.6g} pred_max={float(torch.nanmax(pred).item()):.6g}"
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