"""Task 6B-specific transforms for paired real telescope super-resolution.

This module is intentionally isolated from existing Task1/Task6A transform logic.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_tensor_image(image: torch.Tensor, *, name: str) -> None:
    """Validate that an image tensor is CHW floating-point and finite."""
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(image)}")
    if image.ndim != 3:
        raise ValueError(f"{name} must be CHW tensor with ndim=3, got shape {tuple(image.shape)}")
    if not torch.is_floating_point(image):
        raise TypeError(f"{name} must be floating-point tensor, got dtype {image.dtype}")
    if not torch.isfinite(image).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def _sample_uniform(low: float, high: float, generator: Optional[torch.Generator]) -> float:
    if low > high:
        raise ValueError(f"Invalid range: low={low} > high={high}")
    if low == high:
        return float(low)

    if generator is None:
        sample = torch.rand((), dtype=torch.float32)
    else:
        sample = torch.rand((), generator=generator, dtype=torch.float32)
    return float(low + (high - low) * sample.item())


def _sample_bernoulli(probability: float, generator: Optional[torch.Generator]) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True

    if generator is None:
        sample = torch.rand((), dtype=torch.float32)
    else:
        sample = torch.rand((), generator=generator, dtype=torch.float32)
    return bool(sample.item() < probability)


def _sample_rotation_k(rotation_choices: Tuple[int, ...], generator: Optional[torch.Generator]) -> int:
    if len(rotation_choices) == 1:
        return int(rotation_choices[0])

    if generator is None:
        idx = torch.randint(0, len(rotation_choices), size=(1,), dtype=torch.int64)
    else:
        idx = torch.randint(
            0,
            len(rotation_choices),
            size=(1,),
            generator=generator,
            dtype=torch.int64,
        )
    return int(rotation_choices[int(idx.item())])


class JointD4Transform:
    """Apply one shared D4 geometric augmentation to an (LR, HR) pair.

    The sampled random state consists of horizontal flip, vertical flip, and right-angle
    rotation k in {0,1,2,3}. The exact same sampled parameters are applied to both LR and
    HR tensors, preserving pixel-level spatial alignment.
    """

    def __init__(
        self,
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        rotation_choices: Sequence[int] = (0, 1, 2, 3),
        seed: Optional[int] = None,
    ):
        _validate_probability(horizontal_flip_p, "horizontal_flip_p")
        _validate_probability(vertical_flip_p, "vertical_flip_p")

        valid_rotations = {0, 1, 2, 3}
        choices = tuple(int(k) for k in rotation_choices)
        if not choices:
            raise ValueError("rotation_choices must not be empty")
        if any(k not in valid_rotations for k in choices):
            raise ValueError(f"rotation_choices must be subset of {valid_rotations}, got {choices}")

        self.horizontal_flip_p = float(horizontal_flip_p)
        self.vertical_flip_p = float(vertical_flip_p)
        self.rotation_choices = choices
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def __call__(self, lr: torch.Tensor, hr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _validate_tensor_image(lr, name="lr")
        _validate_tensor_image(hr, name="hr")

        if lr.shape[0] != hr.shape[0]:
            raise ValueError(
                "JointD4Transform requires matching channel counts for LR/HR, "
                f"got lr_channels={lr.shape[0]} hr_channels={hr.shape[0]}"
            )

        do_hflip = _sample_bernoulli(self.horizontal_flip_p, self._generator)
        do_vflip = _sample_bernoulli(self.vertical_flip_p, self._generator)
        rot_k = _sample_rotation_k(self.rotation_choices, self._generator)

        if do_hflip:
            lr = torch.flip(lr, dims=(2,))
            hr = torch.flip(hr, dims=(2,))

        if do_vflip:
            lr = torch.flip(lr, dims=(1,))
            hr = torch.flip(hr, dims=(1,))

        if rot_k != 0:
            lr = torch.rot90(lr, k=rot_k, dims=(1, 2))
            hr = torch.rot90(hr, k=rot_k, dims=(1, 2))

        _validate_tensor_image(lr, name="lr_aug")
        _validate_tensor_image(hr, name="hr_aug")
        return lr.to(dtype=torch.float32), hr.to(dtype=torch.float32)


class TelescopeNoiseTransform:
    """Apply numerically stable LR-only telescope sensor noise.

    Noise model:
    1) Additive Gaussian read noise with sigma sampled from a light range.
    2) Poisson-like shot noise using a Gaussian approximation where variance scales
       with positive signal intensity.
    """

    def __init__(
        self,
        gaussian_sigma_range: Tuple[float, float] = (0.002, 0.01),
        shot_noise_scale_range: Tuple[float, float] = (0.002, 0.02),
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        eps: float = 1e-6,
        seed: Optional[int] = None,
    ):
        g_low, g_high = gaussian_sigma_range
        s_low, s_high = shot_noise_scale_range

        if g_low < 0.0 or g_high < 0.0:
            raise ValueError(f"gaussian_sigma_range must be non-negative, got {gaussian_sigma_range}")
        if s_low < 0.0 or s_high < 0.0:
            raise ValueError(
                f"shot_noise_scale_range must be non-negative, got {shot_noise_scale_range}"
            )
        if g_low > g_high:
            raise ValueError(f"Invalid gaussian_sigma_range {gaussian_sigma_range}")
        if s_low > s_high:
            raise ValueError(f"Invalid shot_noise_scale_range {shot_noise_scale_range}")
        if clamp_min >= clamp_max:
            raise ValueError(f"clamp_min must be < clamp_max, got {clamp_min} >= {clamp_max}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.gaussian_sigma_range = (float(g_low), float(g_high))
        self.shot_noise_scale_range = (float(s_low), float(s_high))
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.eps = float(eps)
        self._generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def _normal_like(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._generator is None:
            return torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device)
        return torch.randn(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=self._generator,
        )

    def __call__(self, lr: torch.Tensor) -> torch.Tensor:
        _validate_tensor_image(lr, name="lr")

        lr_float = lr.to(dtype=torch.float32)

        read_sigma = _sample_uniform(
            self.gaussian_sigma_range[0],
            self.gaussian_sigma_range[1],
            self._generator,
        )
        shot_scale = _sample_uniform(
            self.shot_noise_scale_range[0],
            self.shot_noise_scale_range[1],
            self._generator,
        )

        gaussian_noise = self._normal_like(lr_float) * read_sigma

        # Shot noise approximation: variance grows with signal intensity.
        positive_signal = torch.clamp(lr_float, min=0.0)
        shot_std = torch.sqrt((positive_signal * shot_scale).clamp(min=0.0) + self.eps)
        shot_noise = self._normal_like(lr_float) * shot_std

        noisy_lr = lr_float + gaussian_noise + shot_noise
        if not torch.isfinite(noisy_lr).all():
            raise ValueError("TelescopeNoiseTransform produced NaN/Inf values before clamping")

        noisy_lr = torch.clamp(noisy_lr, min=self.clamp_min, max=self.clamp_max)
        if not torch.isfinite(noisy_lr).all():
            raise ValueError("TelescopeNoiseTransform produced NaN/Inf values after clamping")

        return noisy_lr.to(dtype=torch.float32)


__all__ = ["JointD4Transform", "TelescopeNoiseTransform"]
