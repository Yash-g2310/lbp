"""Metrics for evaluation - ROC, AUC, SSIM, PSNR, etc."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


class ClassificationMetrics:
    """Classification metrics"""
    
    @staticmethod
    def compute_roc_auc(y_true, y_pred_proba):
        """Compute ROC AUC score"""
        return roc_auc_score(y_true, y_pred_proba)
    
    @staticmethod
    def compute_roc_curve(y_true, y_pred_proba):
        """Compute ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        return fpr, tpr, auc(fpr, tpr)
    
    @staticmethod
    def compute_confusion_matrix(y_true, y_pred):
        """Compute confusion matrix"""
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def compute_accuracy(y_true, y_pred):
        """Compute accuracy"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def compute_f1_score(y_true, y_pred):
        """Compute F1 score"""
        from sklearn.metrics import f1_score
        return f1_score(y_true, y_pred)


class SuperResolutionMetrics:
    """Super-resolution metrics"""

    @staticmethod
    def _to_numpy(image):
        if torch.is_tensor(image):
            return image.detach().cpu().numpy()
        return image

    @staticmethod
    def _validate_batched_tensors(img_true: torch.Tensor, img_pred: torch.Tensor) -> None:
        if not torch.is_tensor(img_true) or not torch.is_tensor(img_pred):
            raise TypeError("img_true and img_pred must be torch.Tensor for batched metrics")
        if img_true.ndim != 4 or img_pred.ndim != 4:
            raise ValueError(
                "Batched SR metrics require rank-4 tensors [B,C,H,W], "
                f"got {tuple(img_true.shape)} and {tuple(img_pred.shape)}"
            )
        if img_true.shape != img_pred.shape:
            raise ValueError(
                f"img_true and img_pred shape mismatch: {tuple(img_true.shape)} vs {tuple(img_pred.shape)}"
            )
        if img_true.device != img_pred.device:
            raise ValueError("img_true and img_pred must be on the same device")

    @staticmethod
    def _validate_data_range(data_range: float) -> float:
        value = float(data_range)
        if value <= 0.0:
            raise ValueError(f"data_range must be > 0, got {data_range}")
        return value

    @staticmethod
    def _validate_unit_interval(
        image: torch.Tensor,
        name: str,
        tolerance: float = 1e-6,
    ) -> None:
        min_val = float(image.min().item())
        max_val = float(image.max().item())
        if min_val < (0.0 - tolerance) or max_val > (1.0 + tolerance):
            raise ValueError(
                f"{name} must be in [0,1] for batched SR metrics. "
                f"got min={min_val:.6f}, max={max_val:.6f}"
            )

    @staticmethod
    def _gaussian_window(
        window_size: int,
        sigma: float,
        channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        coords = torch.arange(window_size, device=device, dtype=dtype)
        coords = coords - (window_size - 1) / 2.0
        kernel_1d = torch.exp(-(coords.pow(2)) / (2.0 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    
    @staticmethod
    def compute_psnr(img_true, img_pred, data_range=1.0):
        """Compute Peak Signal-to-Noise Ratio (PSNR)"""
        img_true = SuperResolutionMetrics._to_numpy(img_true)
        img_pred = SuperResolutionMetrics._to_numpy(img_pred)
        
        return psnr(img_true, img_pred, data_range=data_range)
    
    @staticmethod
    def compute_ssim(img_true, img_pred, data_range=1.0, channel_axis=None):
        """Compute Structural Similarity Index (SSIM)"""
        img_true = SuperResolutionMetrics._to_numpy(img_true)
        img_pred = SuperResolutionMetrics._to_numpy(img_pred)
        
        return ssim(img_true, img_pred, data_range=data_range, channel_axis=channel_axis)
    
    @staticmethod
    def compute_mse(img_true, img_pred):
        """Compute Mean Squared Error"""
        img_true = SuperResolutionMetrics._to_numpy(img_true)
        img_pred = SuperResolutionMetrics._to_numpy(img_pred)
        
        return np.mean((img_true - img_pred) ** 2)
    
    @staticmethod
    def compute_mae(img_true, img_pred):
        """Compute Mean Absolute Error"""
        img_true = SuperResolutionMetrics._to_numpy(img_true)
        img_pred = SuperResolutionMetrics._to_numpy(img_pred)
        
        return np.mean(np.abs(img_true - img_pred))

    @staticmethod
    def compute_psnr_batch(
        img_true: torch.Tensor,
        img_pred: torch.Tensor,
        data_range: float = 1.0,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Compute per-sample PSNR for batched [B,C,H,W] tensors in [0,1]."""
        SuperResolutionMetrics._validate_batched_tensors(img_true, img_pred)
        dr = SuperResolutionMetrics._validate_data_range(data_range)

        true = img_true.to(dtype=torch.float32)
        pred = img_pred.to(dtype=torch.float32)
        SuperResolutionMetrics._validate_unit_interval(true, name="img_true")
        SuperResolutionMetrics._validate_unit_interval(pred, name="img_pred")

        mse = (pred - true).pow(2).mean(dim=(1, 2, 3))
        mse = torch.clamp(mse, min=eps)
        psnr_vals = 10.0 * torch.log10((dr * dr) / mse)
        return psnr_vals

    @staticmethod
    def compute_ssim_batch(
        img_true: torch.Tensor,
        img_pred: torch.Tensor,
        data_range: float = 1.0,
        window_size: int = 11,
        sigma: float = 1.5,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        """Compute per-sample SSIM for batched [B,C,H,W] tensors in [0,1]."""
        SuperResolutionMetrics._validate_batched_tensors(img_true, img_pred)
        dr = SuperResolutionMetrics._validate_data_range(data_range)

        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
        if sigma <= 0.0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        true = img_true.to(dtype=torch.float32)
        pred = img_pred.to(dtype=torch.float32)
        SuperResolutionMetrics._validate_unit_interval(true, name="img_true")
        SuperResolutionMetrics._validate_unit_interval(pred, name="img_pred")

        channels = true.shape[1]
        padding = window_size // 2

        kernel = SuperResolutionMetrics._gaussian_window(
            window_size=window_size,
            sigma=sigma,
            channels=channels,
            device=true.device,
            dtype=true.dtype,
        )

        mu_true = F.conv2d(true, kernel, padding=padding, groups=channels)
        mu_pred = F.conv2d(pred, kernel, padding=padding, groups=channels)

        mu_true_sq = mu_true.pow(2)
        mu_pred_sq = mu_pred.pow(2)
        mu_true_pred = mu_true * mu_pred

        sigma_true_sq = F.conv2d(true * true, kernel, padding=padding, groups=channels) - mu_true_sq
        sigma_pred_sq = F.conv2d(pred * pred, kernel, padding=padding, groups=channels) - mu_pred_sq
        sigma_true_pred = F.conv2d(true * pred, kernel, padding=padding, groups=channels) - mu_true_pred

        c1 = (0.01 * dr) ** 2
        c2 = (0.03 * dr) ** 2

        numerator = (2.0 * mu_true_pred + c1) * (2.0 * sigma_true_pred + c2)
        denominator = (mu_true_sq + mu_pred_sq + c1) * (sigma_true_sq + sigma_pred_sq + c2)
        denominator = torch.clamp(denominator, min=eps)

        ssim_map = numerator / denominator
        return ssim_map.mean(dim=(1, 2, 3))
