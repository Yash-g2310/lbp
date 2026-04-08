"""Task 6B dataset and deterministic dataloader utilities.

This module is intentionally isolated from existing Task1/Task6A dataset code.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .task6b_transforms import JointD4Transform, TelescopeNoiseTransform


def _seed_worker(worker_id: int) -> None:
    """Seed numpy for deterministic data loading workers."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def _resolve_pin_memory(pin_memory: Optional[bool]) -> bool:
    if pin_memory is None:
        return torch.cuda.is_available()
    return pin_memory


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: Optional[bool],
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Create DataLoader with deterministic generator and worker seeding."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    max_workers = os.cpu_count() or 1
    if num_workers > max_workers:
        raise ValueError(f"num_workers={num_workers} exceeds available CPUs={max_workers}")

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=_resolve_pin_memory(pin_memory),
        generator=generator,
        worker_init_fn=_seed_worker,
    )


def _validate_finite_tensor(tensor: torch.Tensor, *, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or Inf values")


def _validate_tensor_image(tensor: torch.Tensor, *, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be torch.Tensor, got {type(tensor)}")
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be CHW tensor with ndim=3, got shape {tuple(tensor.shape)}")
    if not torch.is_floating_point(tensor):
        raise TypeError(f"{name} must be floating-point tensor, got dtype {tensor.dtype}")
    _validate_finite_tensor(tensor, name=name)


def _ensure_chw(array: np.ndarray) -> np.ndarray:
    """Convert [H,W], [H,W,C], or [C,H,W] into [C,H,W]."""
    if array.ndim == 2:
        return np.expand_dims(array, axis=0)

    if array.ndim == 3:
        if array.shape[0] in {1, 3}:
            return array
        if array.shape[-1] in {1, 3}:
            return np.transpose(array, (2, 0, 1))

    raise ValueError(
        f"Unsupported sample shape {array.shape}. Expected [H,W], [C,H,W], or [H,W,C] with 1 or 3 channels."
    )


def _to_chw_tensor(sample: np.ndarray, *, source: Path) -> torch.Tensor:
    """Convert a numeric numpy sample to CHW float32 tensor."""
    if not np.issubdtype(sample.dtype, np.number):
        raise TypeError(f"Expected numeric array in {source}, got dtype {sample.dtype}")

    chw = _ensure_chw(np.asarray(sample).astype(np.float32, copy=False))
    tensor = torch.from_numpy(chw)
    _validate_tensor_image(tensor, name=f"tensor:{source}")
    return tensor


def _normalize_task6_tensor(image: torch.Tensor) -> torch.Tensor:
    """Normalize CHW tensor to the Task6 contract in [-1, 1].

    The on-disk real-telescope arrays are not guaranteed to be in a fixed domain.
    This function supports three safe cases:
    - Input already in [0,1] -> map to [-1,1].
    - Input already in [-1,1] -> preserve then clamp.
    - Input outside both ranges -> min/max scale to [0,1], then map to [-1,1].
    """
    _validate_tensor_image(image, name="image_before_normalize")

    min_value = float(image.min().item())
    max_value = float(image.max().item())

    if 0.0 <= min_value and max_value <= 1.0:
        image_01 = image
    elif -1.0 <= min_value and max_value <= 1.0:
        image_01 = (image + 1.0) * 0.5
    else:
        dynamic_range = max_value - min_value
        if dynamic_range <= 1e-8:
            image_01 = torch.full_like(image, 0.5)
        else:
            image_01 = (image - min_value) / dynamic_range

    normalized = torch.clamp((image_01 * 2.0) - 1.0, -1.0, 1.0)

    _validate_tensor_image(normalized, name="image_after_normalize")
    return normalized


def _validate_target_shape(shape: Tuple[int, int, int]) -> None:
    if len(shape) != 3:
        raise ValueError(f"target_shape must contain 3 values (C,H,W), got {shape}")
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"target_shape dimensions must be > 0, got {shape}")


def _validate_lr_hr_shape_contract(
    lr_target_shape: Tuple[int, int, int],
    hr_target_shape: Tuple[int, int, int],
) -> None:
    _validate_target_shape(lr_target_shape)
    _validate_target_shape(hr_target_shape)

    if lr_target_shape[0] != hr_target_shape[0]:
        raise ValueError(
            "LR/HR channel mismatch in target shapes. "
            f"Got lr={lr_target_shape}, hr={hr_target_shape}"
        )
    if hr_target_shape[1] != 2 * lr_target_shape[1] or hr_target_shape[2] != 2 * lr_target_shape[2]:
        raise ValueError(
            "Task6B expects exact 2x spatial scaling between LR and HR target shapes. "
            f"Got lr={lr_target_shape}, hr={hr_target_shape}"
        )


def _validate_sample_shape(tensor: torch.Tensor, *, name: str, target_shape: Tuple[int, int, int]) -> None:
    _validate_tensor_image(tensor, name=name)
    if tuple(tensor.shape) != target_shape:
        raise ValueError(f"{name} expected shape {target_shape}, got {tuple(tensor.shape)}")


def _extract_numeric_suffix_id(stem: str) -> int:
    match = re.search(r"(\d+)$", stem)
    if match is None:
        raise ValueError(f"Could not parse trailing numeric id from stem '{stem}'")
    return int(match.group(1))


def _collect_npy_by_stem(directory: Path) -> Dict[str, Path]:
    """Collect .npy files keyed by trailing numeric id with duplicate protection."""
    if not directory.exists():
        raise FileNotFoundError(f"Missing directory: {directory}")

    files = sorted(directory.rglob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in directory: {directory}")

    stem_to_path: Dict[str, Path] = {}
    numeric_id_to_stem: Dict[int, str] = {}
    for file_path in files:
        stem = file_path.stem
        if stem in stem_to_path:
            raise ValueError(
                f"Duplicate filename stem '{stem}' found in {directory}. "
                "Pairing requires unique stems."
            )

        numeric_id = _extract_numeric_suffix_id(stem)
        if numeric_id in numeric_id_to_stem:
            raise ValueError(
                f"Duplicate trailing numeric id '{numeric_id}' found in {directory}. "
                f"Conflicting stems: '{numeric_id_to_stem[numeric_id]}' and '{stem}'."
            )
        numeric_id_to_stem[numeric_id] = stem
        stem_to_path[stem] = file_path

    return stem_to_path


def _assert_aligned_numeric_ids(lr_files: Sequence[Path], hr_files: Sequence[Path]) -> None:
    if len(lr_files) != len(hr_files):
        raise ValueError(
            f"LR/HR file counts must match, got lr={len(lr_files)} hr={len(hr_files)}"
        )

    mismatches: List[Tuple[int, int, int]] = []
    for index, (lr_path, hr_path) in enumerate(zip(lr_files, hr_files)):
        lr_id = _extract_numeric_suffix_id(lr_path.stem)
        hr_id = _extract_numeric_suffix_id(hr_path.stem)
        if lr_id != hr_id:
            mismatches.append((index, lr_id, hr_id))

    if mismatches:
        preview = ", ".join(
            f"idx={idx}:LR_{lr_id}/HR_{hr_id}" for idx, lr_id, hr_id in mismatches[:10]
        )
        raise ValueError(f"LR/HR numeric-id alignment failed: {preview}")


def pair_task6b_files_by_stem(data_root: str) -> Tuple[List[Path], List[Path]]:
    """Pair LR/HR files by trailing numeric id and return aligned sorted file lists."""
    root = Path(data_root)
    lr_map = _collect_npy_by_stem(root / "LR")
    hr_map = _collect_npy_by_stem(root / "HR")

    lr_id_map = {_extract_numeric_suffix_id(stem): path for stem, path in lr_map.items()}
    hr_id_map = {_extract_numeric_suffix_id(stem): path for stem, path in hr_map.items()}

    lr_ids = set(lr_id_map.keys())
    hr_ids = set(hr_id_map.keys())

    missing_hr = sorted(lr_ids - hr_ids)
    missing_lr = sorted(hr_ids - lr_ids)
    if missing_hr or missing_lr:
        raise ValueError(
            "LR/HR trailing-id mismatch detected. "
            f"Missing in HR: {missing_hr[:10]} | Missing in LR: {missing_lr[:10]}"
        )

    ordered_ids = sorted(lr_ids)
    lr_files = [lr_id_map[idx] for idx in ordered_ids]
    hr_files = [hr_id_map[idx] for idx in ordered_ids]
    _assert_aligned_numeric_ids(lr_files=lr_files, hr_files=hr_files)
    return lr_files, hr_files


def split_task6b_pairs(
    lr_files: Sequence[Path],
    hr_files: Sequence[Path],
    seed: int = 42,
    split_counts: Tuple[int, int, int] = (240, 30, 30),
) -> Tuple[List[Path], List[Path], List[Path], List[Path], List[Path], List[Path]]:
    """Deterministically split paired files into train/val/test using a seeded generator."""
    if len(lr_files) != len(hr_files):
        raise ValueError(f"LR/HR file counts must match, got lr={len(lr_files)} hr={len(hr_files)}")

    train_count, val_count, test_count = split_counts
    if train_count <= 0 or val_count <= 0 or test_count <= 0:
        raise ValueError(f"split_counts must be positive, got {split_counts}")

    total = len(lr_files)
    if train_count + val_count + test_count != total:
        raise ValueError(
            "split_counts must sum to total file count, "
            f"got split_sum={train_count + val_count + test_count}, total={total}"
        )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator).tolist()

    train_idx = permutation[:train_count]
    val_idx = permutation[train_count : train_count + val_count]
    test_idx = permutation[train_count + val_count :]

    train_lr = [lr_files[i] for i in train_idx]
    train_hr = [hr_files[i] for i in train_idx]
    val_lr = [lr_files[i] for i in val_idx]
    val_hr = [hr_files[i] for i in val_idx]
    test_lr = [lr_files[i] for i in test_idx]
    test_hr = [hr_files[i] for i in test_idx]

    train_stems = {p.stem for p in train_lr}
    val_stems = {p.stem for p in val_lr}
    test_stems = {p.stem for p in test_lr}

    if train_stems & val_stems or train_stems & test_stems or val_stems & test_stems:
        raise RuntimeError("Deterministic split produced overlapping stems across subsets")

    return train_lr, train_hr, val_lr, val_hr, test_lr, test_hr


class Task6BRealDataset(Dataset):
    """Real telescope paired dataset with joint geometric and LR-only noise transforms.

    Pipeline in __getitem__:
    1) Load numpy files and convert to CHW float32 tensors.
    2) Apply joint_transforms(lr, hr).
    3) Apply lr_transforms(lr) only.
    4) Resize LR/HR to model-compatible spatial targets (75/150 by default).
    5) Normalize both tensors to Task6 range contract.
    6) Validate shape and return.
    """

    def __init__(
        self,
        lr_files: Sequence[Path],
        hr_files: Sequence[Path],
        joint_transforms: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        lr_transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        normalize: bool = True,
        hr_target_shape: Tuple[int, int, int] = (1, 150, 150),
        lr_target_shape: Optional[Tuple[int, int, int]] = None,
        target_shape: Optional[Tuple[int, int, int]] = None,
    ):
        if len(lr_files) != len(hr_files):
            raise ValueError(f"Task6BRealDataset requires equal LR/HR counts, got {len(lr_files)} and {len(hr_files)}")
        if len(lr_files) == 0:
            raise ValueError("Task6BRealDataset requires at least one paired sample")

        if target_shape is not None:
            _validate_target_shape(target_shape)
            hr_target_shape = target_shape

        _validate_target_shape(hr_target_shape)

        if lr_target_shape is None:
            if hr_target_shape[1] % 2 != 0 or hr_target_shape[2] % 2 != 0:
                raise ValueError(
                    "Task6BRealDataset expects even HR spatial size when deriving LR target. "
                    f"Got hr_target_shape={hr_target_shape}"
                )
            lr_target_shape = (
                int(hr_target_shape[0]),
                int(hr_target_shape[1] // 2),
                int(hr_target_shape[2] // 2),
            )

        _validate_lr_hr_shape_contract(lr_target_shape=lr_target_shape, hr_target_shape=hr_target_shape)

        self.lr_files = [Path(path) for path in lr_files]
        self.hr_files = [Path(path) for path in hr_files]
        self.joint_transforms = joint_transforms
        self.lr_transforms = lr_transforms
        self.normalize = bool(normalize)
        self.hr_target_shape: Tuple[int, int, int] = (
            int(hr_target_shape[0]),
            int(hr_target_shape[1]),
            int(hr_target_shape[2]),
        )
        self.lr_target_shape: Tuple[int, int, int] = (
            int(lr_target_shape[0]),
            int(lr_target_shape[1]),
            int(lr_target_shape[2]),
        )
        # Backward-compatible alias for older Task6B notebook/script code paths.
        self.target_shape = self.hr_target_shape

    def __len__(self) -> int:
        return len(self.lr_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr_path = self.lr_files[idx]
        hr_path = self.hr_files[idx]

        try:
            lr_np = np.load(lr_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load LR sample: {lr_path}") from exc

        try:
            hr_np = np.load(hr_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load HR sample: {hr_path}") from exc

        lr = _to_chw_tensor(lr_np, source=lr_path)
        hr = _to_chw_tensor(hr_np, source=hr_path)

        if self.joint_transforms is not None:
            lr, hr = self.joint_transforms(lr, hr)

        if self.lr_transforms is not None:
            lr = self.lr_transforms(lr)

        # Explicitly reconcile on-disk LR/HR spatial sizes (64/128) to model contract (75/150).
        lr = F.interpolate(
            lr.unsqueeze(0),
            size=(self.lr_target_shape[1], self.lr_target_shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        hr = F.interpolate(
            hr.unsqueeze(0),
            size=(self.hr_target_shape[1], self.hr_target_shape[2]),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        _validate_tensor_image(lr, name="lr_after_resize")
        _validate_tensor_image(hr, name="hr_after_resize")

        if self.normalize:
            lr = _normalize_task6_tensor(lr)
            hr = _normalize_task6_tensor(hr)

        _validate_sample_shape(lr, name="lr", target_shape=self.lr_target_shape)
        _validate_sample_shape(hr, name="hr", target_shape=self.hr_target_shape)

        return lr.to(dtype=torch.float32), hr.to(dtype=torch.float32)


def get_task6b_dataloaders(
    data_root: str = "data/dataset_task6B",
    batch_size: int = 16,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    seed: int = 42,
    expected_total_samples: int = 300,
    hr_target_shape: Tuple[int, int, int] = (1, 150, 150),
    lr_target_shape: Optional[Tuple[int, int, int]] = None,
    target_shape: Optional[Tuple[int, int, int]] = None,
    train_joint_transforms: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
    train_lr_transforms: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build deterministic Task6B train/val/test dataloaders (240/30/30).

    Train loader receives joint geometric augmentation and LR-only telescope noise.
    Val and test loaders are deterministic and use no stochastic transforms.
    """
    lr_files, hr_files = pair_task6b_files_by_stem(data_root)

    if expected_total_samples != len(lr_files):
        raise ValueError(
            f"Expected exactly {expected_total_samples} paired samples, discovered {len(lr_files)}"
        )

    split = split_task6b_pairs(lr_files=lr_files, hr_files=hr_files, seed=seed, split_counts=(240, 30, 30))
    train_lr, train_hr, val_lr, val_hr, test_lr, test_hr = split

    joint_tf = train_joint_transforms if train_joint_transforms is not None else JointD4Transform()
    lr_tf = train_lr_transforms if train_lr_transforms is not None else TelescopeNoiseTransform()

    if target_shape is not None:
        _validate_target_shape(target_shape)
        hr_target_shape = target_shape

    if lr_target_shape is None:
        if hr_target_shape[1] % 2 != 0 or hr_target_shape[2] % 2 != 0:
            raise ValueError(
                "hr_target_shape spatial dims must be even when lr_target_shape is omitted. "
                f"Got hr_target_shape={hr_target_shape}"
            )
        lr_target_shape = (
            int(hr_target_shape[0]),
            int(hr_target_shape[1] // 2),
            int(hr_target_shape[2] // 2),
        )

    _validate_lr_hr_shape_contract(lr_target_shape=lr_target_shape, hr_target_shape=hr_target_shape)

    train_dataset = Task6BRealDataset(
        lr_files=train_lr,
        hr_files=train_hr,
        joint_transforms=joint_tf,
        lr_transforms=lr_tf,
        normalize=True,
        hr_target_shape=hr_target_shape,
        lr_target_shape=lr_target_shape,
    )
    val_dataset = Task6BRealDataset(
        lr_files=val_lr,
        hr_files=val_hr,
        joint_transforms=None,
        lr_transforms=None,
        normalize=True,
        hr_target_shape=hr_target_shape,
        lr_target_shape=lr_target_shape,
    )
    test_dataset = Task6BRealDataset(
        lr_files=test_lr,
        hr_files=test_hr,
        joint_transforms=None,
        lr_transforms=None,
        normalize=True,
        hr_target_shape=hr_target_shape,
        lr_target_shape=lr_target_shape,
    )

    train_loader = _build_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
        seed=seed,
    )
    val_loader = _build_loader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        seed=seed + 1,
    )
    test_loader = _build_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
        seed=seed + 2,
    )

    return train_loader, val_loader, test_loader


__all__ = [
    "Task6BRealDataset",
    "get_task6b_dataloaders",
    "pair_task6b_files_by_stem",
    "split_task6b_pairs",
]
