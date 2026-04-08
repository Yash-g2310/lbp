"""Typed Task 1 configuration loading and validation utilities."""

from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ..models.registry import list_available_models


@dataclass
class DatasetSection:
    data_root: str = "data/dataset"
    split_mode: str = "strict"
    aggregated_train_ratio: float = 0.9
    target_shape: List[int] = field(default_factory=lambda: [1, 150, 150])
    class_names: List[str] = field(default_factory=lambda: ["no", "sphere", "vort"])


@dataclass
class ModelSection:
    architecture: str = "deeplense_resnet"
    pretrained: bool = True
    num_classes: int = 3
    dropout_rate: float = 0.3
    input_size: List[int] = field(default_factory=lambda: [150, 150])


@dataclass
class TrainingSection:
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    num_epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    compile_model: bool = True
    channels_last: bool = True
    amp: bool = True
    amp_dtype: str = "float16"
    gradient_clip_norm: float = 0.0
    label_smoothing: float = 0.0


@dataclass
class ValidationSection:
    eval_frequency: int = 1
    save_best_only: bool = True
    metric: str = "val_loss"
    early_stopping_patience: int = 7
    early_stopping_min_delta: float = 0.0


@dataclass
class UncertaintySection:
    enabled: bool = True
    mc_passes: int = 50


@dataclass
class TransferLearningSection:
    enabled: bool = True
    lp_epochs: int = 5
    ft_epochs: int = 30
    lp_lr: float = 1e-3
    ft_lr: float = 3e-4
    freeze_strategy: str = "backbone"
    lp_checkpoint_name: str = "task1_lp_best.pth"
    ft_checkpoint_name: str = "task1_ft_best.pth"


@dataclass
class RuntimeSection:
    device: str = "auto"
    seed: int = 42
    checkpoint_dir: str = "weights"
    checkpoint_name: str = "task1_best.pth"


@dataclass
class Task1Config:
    dataset: DatasetSection = field(default_factory=DatasetSection)
    model: ModelSection = field(default_factory=ModelSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    validation: ValidationSection = field(default_factory=ValidationSection)
    uncertainty: UncertaintySection = field(default_factory=UncertaintySection)
    transfer_learning: TransferLearningSection = field(default_factory=TransferLearningSection)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)

    def validate(self) -> None:
        if self.dataset.split_mode not in {"strict", "aggregated"}:
            raise ValueError(
                f"dataset.split_mode must be one of ['strict', 'aggregated'], got {self.dataset.split_mode}"
            )
        if not (0.0 < self.dataset.aggregated_train_ratio < 1.0):
            raise ValueError(
                "dataset.aggregated_train_ratio must be in (0, 1), "
                f"got {self.dataset.aggregated_train_ratio}"
            )

        if len(self.dataset.target_shape) != 3:
            raise ValueError(
                f"dataset.target_shape must be [C,H,W], got {self.dataset.target_shape}"
            )
        if self.dataset.target_shape[0] != 1:
            raise ValueError(
                f"Task1 expects single-channel inputs. Got C={self.dataset.target_shape[0]}"
            )
        if any(v <= 0 for v in self.dataset.target_shape):
            raise ValueError(
                f"dataset.target_shape values must be > 0, got {self.dataset.target_shape}"
            )

        if len(self.model.input_size) != 2:
            raise ValueError(f"model.input_size must be [H,W], got {self.model.input_size}")
        if any(v <= 0 for v in self.model.input_size):
            raise ValueError(f"model.input_size values must be > 0, got {self.model.input_size}")
        if self.model.input_size != self.dataset.target_shape[1:]:
            raise ValueError(
                "model.input_size must match dataset.target_shape spatial dims. "
                f"Got input_size={self.model.input_size}, target_shape={self.dataset.target_shape}"
            )

        available_models = set(list_available_models())
        if self.model.architecture not in available_models:
            raise ValueError(
                f"model.architecture='{self.model.architecture}' is not registered. "
                f"Available: {sorted(available_models)}"
            )

        if self.model.num_classes != len(self.dataset.class_names):
            raise ValueError(
                "model.num_classes must match dataset.class_names length. "
                f"Got num_classes={self.model.num_classes}, class_names={self.dataset.class_names}"
            )

        if self.training.batch_size <= 0:
            raise ValueError(f"training.batch_size must be > 0, got {self.training.batch_size}")
        if self.training.num_workers < 0:
            raise ValueError(f"training.num_workers must be >= 0, got {self.training.num_workers}")
        if self.training.num_epochs <= 0:
            raise ValueError(f"training.num_epochs must be > 0, got {self.training.num_epochs}")
        if self.training.learning_rate <= 0:
            raise ValueError(
                f"training.learning_rate must be > 0, got {self.training.learning_rate}"
            )
        if self.training.gradient_clip_norm < 0:
            raise ValueError(
                "training.gradient_clip_norm must be >= 0, "
                f"got {self.training.gradient_clip_norm}"
            )
        if not (0.0 <= self.training.label_smoothing < 1.0):
            raise ValueError(
                "training.label_smoothing must be in [0, 1), "
                f"got {self.training.label_smoothing}"
            )

        if self.training.optimizer not in {"adamw", "adam"}:
            raise ValueError(
                f"training.optimizer must be one of ['adamw', 'adam'], got {self.training.optimizer}"
            )

        if self.training.scheduler not in {"none", "cosine"}:
            raise ValueError(
                f"training.scheduler must be one of ['none', 'cosine'], got {self.training.scheduler}"
            )

        if self.training.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError(
                f"training.amp_dtype must be one of ['float16', 'bfloat16'], got {self.training.amp_dtype}"
            )

        if self.validation.early_stopping_patience < 0:
            raise ValueError(
                "validation.early_stopping_patience must be >= 0, "
                f"got {self.validation.early_stopping_patience}"
            )
        if self.validation.early_stopping_min_delta < 0:
            raise ValueError(
                "validation.early_stopping_min_delta must be >= 0, "
                f"got {self.validation.early_stopping_min_delta}"
            )

        if self.transfer_learning.lp_epochs <= 0:
            raise ValueError(
                f"transfer_learning.lp_epochs must be > 0, got {self.transfer_learning.lp_epochs}"
            )
        if self.transfer_learning.ft_epochs <= 0:
            raise ValueError(
                f"transfer_learning.ft_epochs must be > 0, got {self.transfer_learning.ft_epochs}"
            )
        if self.transfer_learning.lp_lr <= 0:
            raise ValueError(
                f"transfer_learning.lp_lr must be > 0, got {self.transfer_learning.lp_lr}"
            )
        if self.transfer_learning.ft_lr <= 0:
            raise ValueError(
                f"transfer_learning.ft_lr must be > 0, got {self.transfer_learning.ft_lr}"
            )
        if self.transfer_learning.freeze_strategy not in {"backbone"}:
            raise ValueError(
                "transfer_learning.freeze_strategy must be one of ['backbone'], "
                f"got {self.transfer_learning.freeze_strategy}"
            )

        if self.runtime.device not in {"auto", "cuda", "cpu"}:
            raise ValueError(
                f"runtime.device must be one of ['auto', 'cuda', 'cpu'], got {self.runtime.device}"
            )

        if self.uncertainty.mc_passes <= 0:
            raise ValueError(
                f"uncertainty.mc_passes must be > 0, got {self.uncertainty.mc_passes}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def class_to_idx(self) -> Dict[str, int]:
        """Build deterministic class mapping from configured class names."""
        return {name: idx for idx, name in enumerate(self.dataset.class_names)}



def _merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries while preserving unspecified defaults."""
    merged = dict(base)
    for key, value in updates.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _as_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"Expected boolean-like value for {field_name}, got {value!r}")


def _coerce_task1_types(merged_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce YAML values into strict runtime types for Task 1 config dataclasses."""
    cfg = _merge_dicts({}, merged_cfg)

    cfg["dataset"]["aggregated_train_ratio"] = float(cfg["dataset"]["aggregated_train_ratio"])
    cfg["dataset"]["target_shape"] = [int(v) for v in cfg["dataset"]["target_shape"]]

    cfg["model"]["pretrained"] = _as_bool(cfg["model"]["pretrained"], "model.pretrained")
    cfg["model"]["num_classes"] = int(cfg["model"]["num_classes"])
    cfg["model"]["dropout_rate"] = float(cfg["model"]["dropout_rate"])
    cfg["model"]["input_size"] = [int(v) for v in cfg["model"]["input_size"]]

    cfg["training"]["batch_size"] = int(cfg["training"]["batch_size"])
    cfg["training"]["num_workers"] = int(cfg["training"]["num_workers"])
    cfg["training"]["pin_memory"] = _as_bool(cfg["training"]["pin_memory"], "training.pin_memory")
    cfg["training"]["num_epochs"] = int(cfg["training"]["num_epochs"])
    cfg["training"]["learning_rate"] = float(cfg["training"]["learning_rate"])
    cfg["training"]["weight_decay"] = float(cfg["training"]["weight_decay"])
    cfg["training"]["warmup_epochs"] = int(cfg["training"]["warmup_epochs"])
    cfg["training"]["compile_model"] = _as_bool(
        cfg["training"]["compile_model"], "training.compile_model"
    )
    cfg["training"]["channels_last"] = _as_bool(
        cfg["training"]["channels_last"], "training.channels_last"
    )
    cfg["training"]["amp"] = _as_bool(cfg["training"]["amp"], "training.amp")
    cfg["training"]["gradient_clip_norm"] = float(cfg["training"]["gradient_clip_norm"])
    cfg["training"]["label_smoothing"] = float(cfg["training"]["label_smoothing"])

    cfg["validation"]["eval_frequency"] = int(cfg["validation"]["eval_frequency"])
    cfg["validation"]["save_best_only"] = _as_bool(
        cfg["validation"]["save_best_only"], "validation.save_best_only"
    )
    cfg["validation"]["early_stopping_patience"] = int(
        cfg["validation"]["early_stopping_patience"]
    )
    cfg["validation"]["early_stopping_min_delta"] = float(
        cfg["validation"]["early_stopping_min_delta"]
    )

    cfg["uncertainty"]["enabled"] = _as_bool(cfg["uncertainty"]["enabled"], "uncertainty.enabled")
    cfg["uncertainty"]["mc_passes"] = int(cfg["uncertainty"]["mc_passes"])

    cfg["transfer_learning"]["enabled"] = _as_bool(
        cfg["transfer_learning"]["enabled"], "transfer_learning.enabled"
    )
    cfg["transfer_learning"]["lp_epochs"] = int(cfg["transfer_learning"]["lp_epochs"])
    cfg["transfer_learning"]["ft_epochs"] = int(cfg["transfer_learning"]["ft_epochs"])
    cfg["transfer_learning"]["lp_lr"] = float(cfg["transfer_learning"]["lp_lr"])
    cfg["transfer_learning"]["ft_lr"] = float(cfg["transfer_learning"]["ft_lr"])

    cfg["runtime"]["seed"] = int(cfg["runtime"]["seed"])

    return cfg



def load_task1_config(config_path: str) -> Task1Config:
    """Load Task 1 config from YAML and validate all sections."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw_cfg = yaml.safe_load(handle) or {}

    default_cfg = Task1Config().to_dict()
    merged_cfg = _merge_dicts(default_cfg, raw_cfg)
    merged_cfg = _coerce_task1_types(merged_cfg)

    cfg = Task1Config(
        dataset=DatasetSection(**merged_cfg["dataset"]),
        model=ModelSection(**merged_cfg["model"]),
        training=TrainingSection(**merged_cfg["training"]),
        validation=ValidationSection(**merged_cfg["validation"]),
        uncertainty=UncertaintySection(**merged_cfg["uncertainty"]),
        transfer_learning=TransferLearningSection(**merged_cfg["transfer_learning"]),
        runtime=RuntimeSection(**merged_cfg["runtime"]),
    )
    cfg.validate()
    return cfg


__all__ = [
    "Task1Config",
    "load_task1_config",
]
