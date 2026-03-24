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
            # Keep graph connectivity so backward remains well-defined.
            return pred.sum() * 0.0

        pred_valid = torch.clamp(pred[valid_mask], min=1e-6)
        tgt_valid = torch.clamp(target[valid_mask], min=1e-6)

        finite_mask = torch.isfinite(pred_valid) & torch.isfinite(tgt_valid)
        if finite_mask.sum() == 0:
            return pred.sum() * 0.0
        pred_valid = pred_valid[finite_mask]
        tgt_valid = tgt_valid[finite_mask]

        diff_log = torch.log(pred_valid) - torch.log(tgt_valid)
        silog_term = torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
        loss = torch.sqrt(torch.clamp(silog_term, min=1e-12))
        return loss