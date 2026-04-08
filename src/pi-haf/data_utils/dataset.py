"""PyTorch dataset and DataLoader utilities for DeepLense Task 1."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from .transforms import build_transform


DEFAULT_CLASS_TO_IDX: Dict[str, int] = {"no": 0, "sphere": 1, "vort": 2}


@dataclass
class DataLoaderConfig:
    """Configuration for Task 1 dataloader construction."""

    data_root: str = "data/dataset"
    split_mode: str = "strict"
    aggregated_train_ratio: float = 0.9
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: Optional[bool] = None
    seed: int = 42
    target_shape: Tuple[int, int, int] = (1, 150, 150)

    def validate(self) -> None:
        if self.split_mode not in {"strict", "aggregated"}:
            raise ValueError(f"split_mode must be 'strict' or 'aggregated', got {self.split_mode}")
        if not (0.0 < self.aggregated_train_ratio < 1.0):
            raise ValueError(
                f"aggregated_train_ratio must be in (0,1), got {self.aggregated_train_ratio}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        if len(self.target_shape) != 3:
            raise ValueError(f"target_shape must have 3 values (C,H,W), got {self.target_shape}")
        if any(dim <= 0 for dim in self.target_shape):
            raise ValueError(f"target_shape values must be > 0, got {self.target_shape}")


def _seed_worker(worker_id: int) -> None:
    """Seed numpy per worker for deterministic data loading behavior."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def _resolve_pin_memory(pin_memory: Optional[bool]) -> bool:
    if pin_memory is None:
        return torch.cuda.is_available()
    return pin_memory


def _ensure_chw(array: np.ndarray, target_shape: Tuple[int, int, int] = (1, 150, 150)) -> np.ndarray:
    """Convert input array into channel-first format and validate target shape."""
    if array.ndim == 2:
        array = np.expand_dims(array, axis=0)
    elif array.ndim == 3 and array.shape[-1] == 1:
        array = np.transpose(array, (2, 0, 1))
    elif array.ndim == 3 and array.shape[0] == 1:
        pass
    else:
        raise ValueError(
            f"Unsupported input shape {array.shape}. Expected [H,W], [1,H,W], or [H,W,1]."
        )

    if array.shape != target_shape:
        raise ValueError(f"Expected shape {target_shape}, got {array.shape}")
    return array


def _validate_class_map(class_to_idx: Dict[str, int]) -> None:
    if not class_to_idx:
        raise ValueError("class_to_idx cannot be empty")
    idx_values = list(class_to_idx.values())
    if len(set(idx_values)) != len(idx_values):
        raise ValueError(f"class_to_idx values must be unique, got {class_to_idx}")


def _collect_samples(split_root: Path, class_to_idx: Dict[str, int]) -> List[Tuple[Path, int]]:
    """Collect (file_path, label_idx) pairs from a split directory."""
    if not split_root.exists():
        raise FileNotFoundError(f"Missing split directory: {split_root}")

    samples: List[Tuple[Path, int]] = []
    for class_name, label_idx in class_to_idx.items():
        class_dir = split_root / class_name
        if not class_dir.exists():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}. Expected classes: {list(class_to_idx.keys())}"
            )

        class_files = sorted(class_dir.rglob("*.npy"))
        if not class_files:
            raise FileNotFoundError(f"No .npy files found under class directory: {class_dir}")
        samples.extend((file_path, label_idx) for file_path in class_files)

    if not samples:
        raise RuntimeError(f"No samples discovered in split directory: {split_root}")

    return samples


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: Optional[bool],
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Construct a deterministic DataLoader with worker seeding."""
    generator = torch.Generator().manual_seed(seed)
    resolved_pin_memory = _resolve_pin_memory(pin_memory)

    max_workers = os.cpu_count() or 1
    if num_workers > max_workers:
        raise ValueError(f"num_workers={num_workers} exceeds available CPUs={max_workers}")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=resolved_pin_memory,
        generator=generator,
        worker_init_fn=_seed_worker,
    )


class DeepLenseDataset(Dataset):
    """Task 1 dataset for 1-channel DeepLense `.npy` images and 3-way labels."""

    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_shape: Tuple[int, int, int] = (1, 150, 150),
    ):
        if not samples:
            raise ValueError("samples cannot be empty")
        self.samples = list(samples)
        self.transform = transform
        self.target_shape = target_shape

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path, label = self.samples[idx]

        try:
            array = np.load(file_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load numpy file: {file_path}") from exc

        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"Expected numeric array in {file_path}, got dtype {array.dtype}")

        array = array.astype(np.float32, copy=False)
        array = _ensure_chw(array, target_shape=self.target_shape)
        image = torch.from_numpy(array).to(dtype=torch.float32)

        if self.transform is not None:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def get_strict_dataloaders(
    data_root: str = "data/dataset",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    seed: int = 42,
    class_to_idx: Optional[Dict[str, int]] = None,
    target_shape: Tuple[int, int, int] = (1, 150, 150),
) -> Tuple[DataLoader, DataLoader]:
    """Create strict dataloaders from existing train/val folders."""
    class_map = class_to_idx or DEFAULT_CLASS_TO_IDX
    _validate_class_map(class_map)

    root = Path(data_root)
    train_samples = _collect_samples(root / "train", class_map)
    val_samples = _collect_samples(root / "val", class_map)

    train_dataset = DeepLenseDataset(
        train_samples,
        transform=build_transform(stage="train", enable_d4=True),
        target_shape=target_shape,
    )
    val_dataset = DeepLenseDataset(
        val_samples,
        transform=build_transform(stage="eval", enable_d4=False),
        target_shape=target_shape,
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
        seed=seed,
    )
    return train_loader, val_loader


def get_aggregated_dataloaders(
    data_root: str = "data/dataset",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: Optional[bool] = None,
    seed: int = 42,
    train_ratio: float = 0.9,
    class_to_idx: Optional[Dict[str, int]] = None,
    target_shape: Tuple[int, int, int] = (1, 150, 150),
) -> Tuple[DataLoader, DataLoader]:
    """Create deterministic 90:10 style dataloaders by pooling train and val."""
    class_map = class_to_idx or DEFAULT_CLASS_TO_IDX
    _validate_class_map(class_map)
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")

    root = Path(data_root)
    all_samples = _collect_samples(root / "train", class_map) + _collect_samples(root / "val", class_map)

    full_dataset = DeepLenseDataset(all_samples, transform=None, target_shape=target_shape)
    train_size = int(train_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError(
            f"Invalid split sizes from train_ratio={train_ratio}. total={len(full_dataset)} "
            f"produced train={train_size}, val={val_size}."
        )

    split_generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        full_dataset,
        lengths=[train_size, val_size],
        generator=split_generator,
    )

    train_samples = [all_samples[i] for i in train_subset.indices]
    val_samples = [all_samples[i] for i in val_subset.indices]

    train_dataset = DeepLenseDataset(
        train_samples,
        transform=build_transform(stage="train", enable_d4=True),
        target_shape=target_shape,
    )
    val_dataset = DeepLenseDataset(
        val_samples,
        transform=build_transform(stage="eval", enable_d4=False),
        target_shape=target_shape,
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
        seed=seed,
    )
    return train_loader, val_loader


def get_task1_dataloaders(
    config: DataLoaderConfig,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build Task 1 dataloaders from a validated dataloader config."""
    config.validate()

    if config.split_mode == "strict":
        return get_strict_dataloaders(
            data_root=config.data_root,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            seed=config.seed,
            class_to_idx=class_to_idx,
            target_shape=config.target_shape,
        )

    return get_aggregated_dataloaders(
        data_root=config.data_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
        train_ratio=config.aggregated_train_ratio,
        class_to_idx=class_to_idx,
        target_shape=config.target_shape,
    )


def dataloader_config_from_dict(config: Dict[str, Any]) -> DataLoaderConfig:
    """Construct a DataLoaderConfig from a task config dictionary."""
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    runtime_cfg = config.get("runtime", {})

    return DataLoaderConfig(
        data_root=dataset_cfg.get("data_root", "data/dataset"),
        split_mode=dataset_cfg.get("split_mode", "strict"),
        aggregated_train_ratio=float(dataset_cfg.get("aggregated_train_ratio", 0.9)),
        batch_size=int(training_cfg.get("batch_size", 32)),
        num_workers=int(training_cfg.get("num_workers", 4)),
        pin_memory=training_cfg.get("pin_memory"),
        seed=int(runtime_cfg.get("seed", 42)),
        target_shape=tuple(dataset_cfg.get("target_shape", [1, 150, 150])),
    )


class Task6Dataset(Dataset):
    """Dataset class for Task 6 - Super-Resolution."""

    def __init__(self, lr_data, hr_data, transforms=None):
        if len(lr_data) != len(hr_data):
            raise ValueError(
                f"Task6Dataset requires equal LR/HR sample counts, got lr={len(lr_data)} hr={len(hr_data)}"
            )
        self.lr_data = lr_data
        self.hr_data = hr_data
        self.transforms = transforms

    def __len__(self):
        return len(self.lr_data)

    @staticmethod
    def _to_chw_tensor(sample) -> torch.Tensor:
        array = np.asarray(sample)
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(f"Task6Dataset expects numeric arrays, got dtype={array.dtype}")

        if array.ndim == 2:
            array = np.expand_dims(array, axis=0)
        elif array.ndim == 3:
            if array.shape[0] in {1, 3}:
                pass
            elif array.shape[-1] in {1, 3}:
                array = np.transpose(array, (2, 0, 1))
            else:
                raise ValueError(
                    f"Unsupported Task6 sample shape {array.shape}. Expected CHW or HWC with 1 or 3 channels."
                )
        else:
            raise ValueError(f"Task6 sample must have 2 or 3 dimensions, got shape {array.shape}")

        return torch.from_numpy(array.astype(np.float32, copy=False))

    def __getitem__(self, idx):
        lr_sample = self._to_chw_tensor(self.lr_data[idx])
        hr_sample = self._to_chw_tensor(self.hr_data[idx])

        if self.transforms:
            lr_sample = self.transforms(lr_sample)
            hr_sample = self.transforms(hr_sample)

        if lr_sample.ndim != 3 or hr_sample.ndim != 3:
            raise ValueError(
                f"Task6 transformed samples must be CHW tensors, got lr={tuple(lr_sample.shape)} hr={tuple(hr_sample.shape)}"
            )

        return lr_sample.to(dtype=torch.float32), hr_sample.to(dtype=torch.float32)


__all__ = [
    "DEFAULT_CLASS_TO_IDX",
    "DataLoaderConfig",
    "DeepLenseDataset",
    "Task6Dataset",
    "dataloader_config_from_dict",
    "get_aggregated_dataloaders",
    "get_strict_dataloaders",
    "get_task1_dataloaders",
]
