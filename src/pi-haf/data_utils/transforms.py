"""Transform utilities for DeepLense image pipelines."""

from typing import Callable, Literal, Sequence

import numpy as np
import torch
from torchvision import transforms

TransformStage = Literal["train", "eval"]


class RandomRightAngleRotation:
    """Randomly rotate a CHW tensor by right angles to respect lensing symmetry."""

    def __init__(self, k_choices: Sequence[int] = (1, 2, 3)):
        self.k_choices = tuple(k_choices)
        valid = {1, 2, 3}
        if not self.k_choices:
            raise ValueError("k_choices must contain at least one value from {1,2,3}.")
        if any(k not in valid for k in self.k_choices):
            raise ValueError(f"k_choices must be a subset of {valid}, got {self.k_choices}")

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        _validate_tensor_image(image)
        k = int(np.random.choice(self.k_choices))
        return torch.rot90(image, k=k, dims=(1, 2))


def _validate_tensor_image(image: torch.Tensor) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor input for transform pipeline, got {type(image)}")
    if image.ndim != 3:
        raise ValueError(f"Expected CHW tensor with ndim=3, got shape {tuple(image.shape)}")


def build_transform(
    stage: TransformStage,
    normalize: bool = True,
    enable_d4: bool = True,
    horizontal_flip_p: float = 0.5,
    vertical_flip_p: float = 0.5,
    rotation_choices: Sequence[int] = (1, 2, 3),
) -> transforms.Compose:
    """Create a configurable transform pipeline for Task 1.

    For `stage="train"`, D4-inspired augmentations can be enabled.
    For `stage="eval"`, only deterministic preprocessing is applied.
    """
    if stage not in {"train", "eval"}:
        raise ValueError(f"stage must be one of ['train', 'eval'], got {stage}")

    transform_list = []

    if stage == "train" and enable_d4:
        transform_list.extend(
            [
                transforms.RandomHorizontalFlip(p=horizontal_flip_p),
                transforms.RandomVerticalFlip(p=vertical_flip_p),
                RandomRightAngleRotation(k_choices=rotation_choices),
            ]
        )

    if normalize:
        transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    return transforms.Compose(transform_list)


def get_augmentation_transforms(normalize: bool = False) -> Callable[[torch.Tensor], torch.Tensor]:
    """Legacy-compatible Task6 transform builder.

    Task6 notebooks currently apply one transform callable to both LR and HR tensors.
    To avoid spatial misalignment between pairs, this helper is deterministic by default.
    """
    transform_list = []
    if normalize:
        # Dynamic per-sample normalization for 1- or 3-channel tensors.
        def _normalize(image: torch.Tensor) -> torch.Tensor:
            _validate_tensor_image(image)
            if image.shape[0] == 1:
                return transforms.Normalize(mean=[0.5], std=[0.5])(image)
            if image.shape[0] == 3:
                return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
            raise ValueError(f"Unsupported channel count for Task6 normalization: {image.shape[0]}")

        transform_list.append(_normalize)

    return transforms.Compose(transform_list)


__all__ = [
    "RandomRightAngleRotation",
    "build_transform",
    "get_augmentation_transforms",
    "TransformStage",
]
