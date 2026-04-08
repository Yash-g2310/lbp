from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.data_utils.task6b_dataset import get_task6b_dataloaders
from src.engine.losses import CompositeRectifiedFlowLoss
from src.engine.train_sr import ReflowSRTrainer
from src.models.pi_haf.backbone import PIHAFBackbone
from src.models.task6b_lora import (
    LoRAMultiheadAttention,
    prepare_task6b_model,
    print_trainable_parameters,
    set_lora_trainable,
)


LOGGER = logging.getLogger(__name__)


def _resolve_device(device_name: Optional[str]) -> torch.device:
    if device_name in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_name = str(device_name)
    resolved = torch.device(resolved_name)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device is set to 'cuda' but no CUDA device is available")
    return resolved


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    key = dtype_name.strip().lower()
    if key == "float16":
        return torch.float16
    if key == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype '{dtype_name}'. Use 'float16' or 'bfloat16'.")


def _ramp_weight(target_weight: float, epoch: int, ramp_epochs: int) -> float:
    if target_weight < 0.0:
        raise ValueError(f"target_weight must be >= 0, got {target_weight}")
    if ramp_epochs < 0:
        raise ValueError(f"ramp_epochs must be >= 0, got {ramp_epochs}")
    if target_weight == 0.0 or ramp_epochs == 0:
        return target_weight
    scale = min(1.0, float(epoch) / float(ramp_epochs))
    return target_weight * scale


def _parse_shape3(name: str, value: Any) -> Tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-item list/tuple [C,H,W], got {value}")
    shape = (int(value[0]), int(value[1]), int(value[2]))
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"{name} dimensions must be > 0, got {shape}")
    return shape


def _parse_epoch_window(name: str, value: Any, *, num_epochs: int) -> Tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be [start,end], got {value}")
    start = int(value[0])
    end = int(value[1])
    if start < 1 or end < start:
        raise ValueError(f"{name} must satisfy 1 <= start <= end, got [{start}, {end}]")
    if end > num_epochs:
        raise ValueError(f"{name} end must be <= training.num_epochs ({num_epochs}), got {end}")
    return start, end


def _resolve_swa_window(training_cfg: Dict[str, Any], *, num_epochs: int) -> Tuple[int, int]:
    configured = training_cfg.get("swa_epochs")
    if configured is not None:
        return _parse_epoch_window("training.swa_epochs", configured, num_epochs=num_epochs)
    # Default to final 10 epochs (or fewer when num_epochs < 10).
    return max(1, num_epochs - 9), num_epochs


def _validate_config(config_dict: Dict[str, Any]) -> None:
    required_top = ["model", "training", "losses", "vram", "lora", "dataset"]
    for key in required_top:
        if key not in config_dict:
            raise ValueError(f"Missing required config section: '{key}'")

    training_cfg = config_dict["training"]
    if "stage1_lr" not in training_cfg:
        raise ValueError("Missing training.stage1_lr in config")
    if "stage2_lr" not in training_cfg:
        raise ValueError("Missing training.stage2_lr in config")

    num_epochs = int(training_cfg.get("num_epochs", 0))
    if num_epochs <= 0:
        raise ValueError(f"training.num_epochs must be > 0, got {num_epochs}")

    stage1_lr = float(training_cfg["stage1_lr"])
    stage2_lr = float(training_cfg["stage2_lr"])
    if stage1_lr <= 0.0 or stage2_lr <= 0.0:
        raise ValueError(f"stage1_lr and stage2_lr must be > 0, got {stage1_lr}, {stage2_lr}")

    stage1_start, stage1_end = _parse_epoch_window(
        "training.stage1_epochs",
        training_cfg.get("stage1_epochs", [1, 5]),
        num_epochs=num_epochs,
    )
    stage2_start, stage2_end = _parse_epoch_window(
        "training.stage2_epochs",
        training_cfg.get("stage2_epochs", [stage1_end + 1, num_epochs]),
        num_epochs=num_epochs,
    )
    if stage1_start != 1:
        raise ValueError(f"training.stage1_epochs must start at 1, got {stage1_start}")
    if stage1_end + 1 != stage2_start:
        raise ValueError(
            "training.stage1_epochs and training.stage2_epochs must be contiguous. "
            f"Got stage1={stage1_start, stage1_end}, stage2={stage2_start, stage2_end}"
        )
    if stage2_end != num_epochs:
        raise ValueError(
            f"training.stage2_epochs must end at training.num_epochs ({num_epochs}), got {stage2_end}"
        )

    _resolve_swa_window(training_cfg, num_epochs=num_epochs)

    lora_cfg = config_dict["lora"]
    rank = int(lora_cfg.get("rank", 0))
    alpha = int(lora_cfg.get("alpha", 0))
    dropout = float(lora_cfg.get("dropout", -1.0))
    if rank <= 0:
        raise ValueError(f"lora.rank must be > 0, got {rank}")
    if alpha <= 0:
        raise ValueError(f"lora.alpha must be > 0, got {alpha}")
    if not (0.0 <= dropout < 1.0):
        raise ValueError(f"lora.dropout must be in [0,1), got {dropout}")


def _extract_state_dict(pretrained_obj: Any) -> Dict[str, torch.Tensor]:
    if isinstance(pretrained_obj, dict):
        if "model_state_dict" in pretrained_obj:
            state = pretrained_obj["model_state_dict"]
            if not isinstance(state, dict):
                raise TypeError("'model_state_dict' exists but is not a dictionary")
            return state
        if "state_dict" in pretrained_obj:
            state = pretrained_obj["state_dict"]
            if not isinstance(state, dict):
                raise TypeError("'state_dict' exists but is not a dictionary")
            return state

        if pretrained_obj and all(torch.is_tensor(v) for v in pretrained_obj.values()):
            return pretrained_obj

    raise TypeError(
        "Unsupported checkpoint format. Expected raw state_dict or dict containing 'model_state_dict'."
    )


def _strict_load_pretrained(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model_state = model.state_dict()
    model_keys = set(model_state.keys())
    ckpt_keys = set(state_dict.keys())

    missing_keys = sorted(model_keys - ckpt_keys)
    unexpected_keys = sorted(ckpt_keys - model_keys)

    shape_mismatches: List[str] = []
    for key in sorted(model_keys & ckpt_keys):
        if model_state[key].shape != state_dict[key].shape:
            shape_mismatches.append(
                f"{key}: model={tuple(model_state[key].shape)} ckpt={tuple(state_dict[key].shape)}"
            )

    if missing_keys or unexpected_keys or shape_mismatches:
        raise ValueError(
            "Strict checkpoint load failed. "
            f"Missing keys={missing_keys[:10]}, Unexpected keys={unexpected_keys[:10]}, "
            f"Shape mismatches={shape_mismatches[:10]}"
        )

    model.load_state_dict(state_dict, strict=True)


def _build_trainable_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    eps: float,
) -> torch.optim.Optimizer:
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")

    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters found when building optimizer")

    return torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay, eps=eps)


def _count_trainable_lora_params(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, LoRAMultiheadAttention):
            if module.lora_q_a.requires_grad:
                count += int(module.lora_q_a.numel())
            if module.lora_q_b.requires_grad:
                count += int(module.lora_q_b.numel())
            if module.lora_v_a.requires_grad:
                count += int(module.lora_v_a.numel())
            if module.lora_v_b.requires_grad:
                count += int(module.lora_v_b.numel())
    return count


def _config_fingerprint_sha256(config_dict: Dict[str, Any]) -> str:
    serialized = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def train_task6b(
    config_dict: dict,
    pretrained_6a_path: str,
    save_path: str,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train Task6B with two-stage optimization and final SWA averaging.

    Stage windows and SWA window are config-driven.
    """
    if not isinstance(config_dict, dict):
        raise TypeError(f"config_dict must be dict, got {type(config_dict)}")

    _validate_config(config_dict)

    pretrained_path = Path(pretrained_6a_path)
    if not pretrained_path.exists() or not pretrained_path.is_file():
        raise FileNotFoundError(f"pretrained_6a_path does not exist or is not a file: {pretrained_path}")

    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_cfg = dict(config_dict.get("model", {}))
    model_cfg.pop("architecture", None)
    training_cfg = config_dict.get("training", {})
    losses_cfg = config_dict.get("losses", {})
    vram_cfg = config_dict.get("vram", {})
    lora_cfg = config_dict.get("lora", {})
    dataset_cfg = config_dict.get("dataset", {})

    seed = int(config_dict.get("seed", 42))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    num_epochs = int(training_cfg.get("num_epochs", 30))
    stage1_lr = float(training_cfg["stage1_lr"])
    stage2_lr = float(training_cfg["stage2_lr"])
    weight_decay = float(training_cfg.get("weight_decay", 1e-5))
    stage1_start, stage1_end = _parse_epoch_window(
        "training.stage1_epochs",
        training_cfg.get("stage1_epochs", [1, 5]),
        num_epochs=num_epochs,
    )
    stage2_start, stage2_end = _parse_epoch_window(
        "training.stage2_epochs",
        training_cfg.get("stage2_epochs", [stage1_end + 1, num_epochs]),
        num_epochs=num_epochs,
    )
    swa_start, swa_end = _resolve_swa_window(training_cfg, num_epochs=num_epochs)
    swa_window_size = int(swa_end - swa_start + 1)

    amp_dtype = _resolve_amp_dtype(str(vram_cfg.get("amp_dtype", "bfloat16")))
    default_eps = 1e-6 if amp_dtype == torch.bfloat16 else 1e-8
    adamw_eps = float(training_cfg.get("adamw_eps", default_eps))

    model = PIHAFBackbone(config=model_cfg)

    loaded = torch.load(pretrained_path, map_location="cpu")
    pretrained_state = _extract_state_dict(loaded)
    _strict_load_pretrained(model, pretrained_state)

    device = _resolve_device(config_dict.get("device", "auto"))
    model = model.to(device)

    model = prepare_task6b_model(
        model=model,
        rank=int(lora_cfg.get("rank", 16)),
        alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
    )
    print_trainable_parameters(model)

    hr_target_shape = _parse_shape3(
        "dataset.hr_target_shape",
        dataset_cfg.get("hr_target_shape", dataset_cfg.get("target_shape", [1, 150, 150])),
    )

    lr_target_cfg = dataset_cfg.get("lr_target_shape")
    if lr_target_cfg is None:
        if hr_target_shape[1] % 2 != 0 or hr_target_shape[2] % 2 != 0:
            raise ValueError(
                "dataset.hr_target_shape spatial dims must be even when lr_target_shape is omitted. "
                f"Got hr_target_shape={hr_target_shape}"
            )
        lr_target_shape = (hr_target_shape[0], hr_target_shape[1] // 2, hr_target_shape[2] // 2)
    else:
        lr_target_shape = _parse_shape3("dataset.lr_target_shape", lr_target_cfg)

    model_input_size = model_cfg.get("input_size", [75, 75])
    if not isinstance(model_input_size, (list, tuple)) or len(model_input_size) != 2:
        raise ValueError(f"model.input_size must be [H,W], got {model_input_size}")
    input_h, input_w = int(model_input_size[0]), int(model_input_size[1])

    expected_lr_shape = (int(model_cfg.get("lr_channels", 1)), input_h, input_w)
    expected_hr_shape = (int(model_cfg.get("in_channels", 1)), input_h * 2, input_w * 2)
    if lr_target_shape != expected_lr_shape or hr_target_shape != expected_hr_shape:
        raise ValueError(
            "Task6B dataset/model shape contract mismatch. "
            f"Expected lr_target_shape={expected_lr_shape}, hr_target_shape={expected_hr_shape}; "
            f"got lr_target_shape={lr_target_shape}, hr_target_shape={hr_target_shape}"
        )

    train_loader, val_loader, _ = get_task6b_dataloaders(
        data_root=str(dataset_cfg.get("data_root", "data/dataset_task6B")),
        batch_size=int(training_cfg.get("batch_size", 16)),
        num_workers=int(training_cfg.get("num_workers", 4)),
        pin_memory=training_cfg.get("pin_memory"),
        seed=seed,
        expected_total_samples=int(dataset_cfg.get("expected_total_samples", 300)),
        hr_target_shape=hr_target_shape,
        lr_target_shape=lr_target_shape,
    )

    criterion = CompositeRectifiedFlowLoss(
        flow_weight=float(losses_cfg.get("flow_weight", 1.0)),
        mass_weight=float(losses_cfg.get("mass_weight", 0.1)),
        freq_weight=float(losses_cfg.get("freq_weight", 0.1)),
        reduction=str(losses_cfg.get("reduction", "mean")),
        freq_exponent=float(losses_cfg.get("freq_exponent", 2.0)),
    )

    stage1_optimizer = _build_trainable_optimizer(
        model=model,
        learning_rate=stage1_lr,
        weight_decay=weight_decay,
        eps=adamw_eps,
    )

    trainer = ReflowSRTrainer(
        model=model,
        criterion=criterion,
        optimizer=stage1_optimizer,
        device=str(device),
        amp_enabled=bool(vram_cfg.get("enable_mixed_precision", True)),
        amp_dtype=amp_dtype,
        gradient_accumulation_steps=int(vram_cfg.get("gradient_accumulation_steps", 2)),
        gradient_clip_norm=float(vram_cfg.get("gradient_clip_norm", 1.0)),
    )

    base_flow_weight = float(criterion.flow_weight)
    base_mass_weight = float(criterion.mass_weight)
    base_freq_weight = float(criterion.freq_weight)
    mass_ramp_epochs = int(losses_cfg.get("mass_ramp_epochs", 0))
    freq_ramp_epochs = int(losses_cfg.get("freq_ramp_epochs", 0))

    history: Dict[str, List[float]] = {
        "train_total_loss": [],
        "train_flow_loss": [],
        "train_mass_loss": [],
        "train_freq_loss": [],
        "val_total_loss": [],
        "val_flow_loss": [],
        "val_mass_loss": [],
        "val_freq_loss": [],
    }

    stage2_scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None
    stage2_transition_done = False

    swa_state: Dict[str, torch.Tensor] = {}
    swa_samples = 0

    for epoch in range(1, num_epochs + 1):
        if epoch == stage2_start and not stage2_transition_done:
            LOGGER.info("Transitioning to Stage-2 LoRA fine-tuning at epoch %d", epoch)
            set_lora_trainable(trainer.model, True)

            trainable_lora_params = _count_trainable_lora_params(trainer.model)
            if trainable_lora_params <= 0:
                raise RuntimeError(
                    "Stage-2 transition failed: no LoRA parameters are trainable after set_lora_trainable"
                )

            stage2_optimizer = _build_trainable_optimizer(
                model=trainer.model,
                learning_rate=stage2_lr,
                weight_decay=weight_decay,
                eps=adamw_eps,
            )
            trainer.optimizer = stage2_optimizer
            stage2_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                stage2_optimizer,
                T_max=max(1, stage2_end - stage2_start + 1),
            )
            stage2_transition_done = True

        criterion.flow_weight = base_flow_weight
        criterion.mass_weight = _ramp_weight(base_mass_weight, epoch=epoch, ramp_epochs=mass_ramp_epochs)
        criterion.freq_weight = _ramp_weight(base_freq_weight, epoch=epoch, ramp_epochs=freq_ramp_epochs)

        scheduler = stage2_scheduler if stage2_transition_done and stage2_start <= epoch <= stage2_end else None
        train_metrics = trainer.train_epoch(train_loader, scheduler=scheduler)
        val_metrics = trainer.validate(val_loader)

        history["train_total_loss"].append(train_metrics["total_loss"])
        history["train_flow_loss"].append(train_metrics["flow_loss"])
        history["train_mass_loss"].append(train_metrics["mass_loss"])
        history["train_freq_loss"].append(train_metrics["freq_loss"])
        history["val_total_loss"].append(val_metrics["total_loss"])
        history["val_flow_loss"].append(val_metrics["flow_loss"])
        history["val_mass_loss"].append(val_metrics["mass_loss"])
        history["val_freq_loss"].append(val_metrics["freq_loss"])

        stage_name = "stage2" if epoch >= stage2_start else "stage1"
        LOGGER.info(
            "Epoch %d/%d [%s] | train_total=%.6f val_total=%.6f | loss_w(flow=%.3f,mass=%.3f,freq=%.3f)",
            epoch,
            num_epochs,
            stage_name,
            train_metrics["total_loss"],
            val_metrics["total_loss"],
            criterion.flow_weight,
            criterion.mass_weight,
            criterion.freq_weight,
        )

        if swa_start <= epoch <= swa_end:
            current_state = trainer.model.state_dict()
            swa_samples += 1
            for key, value in current_state.items():
                if torch.is_floating_point(value):
                    if key not in swa_state:
                        swa_state[key] = value.detach().clone().to(
                            device=value.device,
                            dtype=value.dtype,
                        ) / float(swa_window_size)
                    else:
                        swa_state[key] = swa_state[key] + (
                            value.detach().to(device=swa_state[key].device, dtype=swa_state[key].dtype)
                            / float(swa_window_size)
                        )

    if not stage2_transition_done:
        raise RuntimeError(
            f"Stage-2 transition never occurred; expected transition at epoch {stage2_start}"
        )

    if swa_samples != swa_window_size:
        raise RuntimeError(
            "SWA snapshot count mismatch. "
            f"Expected {swa_window_size} snapshots from epochs {swa_start}-{swa_end}, got {swa_samples}"
        )

    final_state = trainer.model.state_dict()
    merged_state: Dict[str, torch.Tensor] = {}
    for key, value in final_state.items():
        if torch.is_floating_point(value):
            if key not in swa_state:
                raise RuntimeError(f"Missing SWA tensor for floating key: {key}")
            merged_state[key] = swa_state[key]
        else:
            merged_state[key] = value

    trainer.model.load_state_dict(merged_state, strict=True)

    completed_epochs = int(len(history.get("train_total_loss", [])))
    checkpoint = {
        "model_state_dict": trainer.model.state_dict(),
        "history": history,
        "config": config_dict,
        "swa_epochs": list(range(swa_start, swa_end + 1)),
        "completed_epochs": completed_epochs,
        "training_end_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_schema_version": 2,
        "config_fingerprint_sha256": _config_fingerprint_sha256(config_dict),
    }
    torch.save(checkpoint, output_path)
    LOGGER.info("Saved Task6B SWA checkpoint to %s", output_path)

    for key, values in history.items():
        if len(values) != num_epochs:
            raise RuntimeError(f"History key '{key}' expected length {num_epochs}, got {len(values)}")

    return trainer.model, history


__all__ = ["train_task6b"]
