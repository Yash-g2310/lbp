"""Training and evaluation utilities for DeepLense classification.

This module is notebook-friendly and provides:
1) High-throughput PyTorch 2.x training (`train_model`)
2) Epistemic uncertainty estimation via MC Dropout (`evaluate_mc_dropout`)
3) Multiclass ROC/AUC plotting (`plot_and_calculate_roc`)
"""

from contextlib import nullcontext
import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from .checkpoints import CheckpointManager


LOGGER = logging.getLogger(__name__)


def _get_device(device: Optional[str] = None) -> torch.device:
    if device == "auto" or device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device is set to 'cuda' but no CUDA device is available.")
    return resolved


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype '{dtype_name}'. Use 'float16' or 'bfloat16'.")


def _has_dropout_layers(model: nn.Module) -> bool:
    dropout_types = (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)
    return any(isinstance(module, dropout_types) for module in model.modules())


def _build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    parameter_groups: Optional[List[Dict[str, Any]]] = None,
) -> torch.optim.Optimizer:
    if parameter_groups is None:
        parameters = [param for param in model.parameters() if param.requires_grad]
        if not parameters:
            raise ValueError("No trainable parameters found for optimizer construction")
    else:
        parameters = parameter_groups

    name = optimizer_name.lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Expected one of ['adam', 'adamw']")


def _get_backbone_head_parameters(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    if not hasattr(model, "backbone"):
        raise ValueError("Model does not expose a 'backbone' attribute required for LP-FT training")

    backbone = model.backbone
    if hasattr(backbone, "fc"):
        head_module = backbone.fc
    elif hasattr(backbone, "head"):
        head_module = backbone.head
    else:
        raise ValueError("Backbone does not expose an 'fc' or 'head' module for LP-FT training")

    head_param_ids = {id(param) for param in head_module.parameters()}
    backbone_params: List[nn.Parameter] = []
    head_params: List[nn.Parameter] = []

    for param in backbone.parameters():
        if id(param) in head_param_ids:
            head_params.append(param)
        else:
            backbone_params.append(param)

    backbone_param_ids = {id(param) for param in backbone.parameters()}
    for param in model.parameters():
        if id(param) not in backbone_param_ids:
            head_params.append(param)

    return backbone_params, head_params


def freeze_backbone_for_linear_probe(model: nn.Module, freeze_strategy: str = "backbone") -> Dict[str, int]:
    if freeze_strategy != "backbone":
        raise ValueError(f"Unsupported freeze strategy '{freeze_strategy}'")

    if not hasattr(model, "backbone"):
        raise ValueError("Model does not expose a 'backbone' attribute required for LP-FT training")

    for param in model.parameters():
        param.requires_grad = True

    backbone = model.backbone
    for param in backbone.parameters():
        param.requires_grad = False

    if hasattr(backbone, "fc"):
        head_module = backbone.fc
    elif hasattr(backbone, "head"):
        head_module = backbone.head
    else:
        raise ValueError("Backbone does not expose an 'fc' or 'head' module for LP-FT training")

    for param in head_module.parameters():
        param.requires_grad = True

    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    return {"trainable_params": trainable, "frozen_params": frozen}


def unfreeze_all_parameters(model: nn.Module) -> int:
    for param in model.parameters():
        param.requires_grad = True
    return sum(param.numel() for param in model.parameters())


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
) -> Optional[Any]:
    name = scheduler_name.lower()
    if name == "none":
        return None

    if name != "cosine":
        raise ValueError(f"Unsupported scheduler '{scheduler_name}'. Expected one of ['none', 'cosine']")

    cosine_tmax = max(1, epochs - max(0, warmup_epochs))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_tmax)

    if warmup_epochs <= 0:
        return cosine_scheduler

    warmup_lambda = lambda epoch: min(1.0, float(epoch + 1) / float(max(1, warmup_epochs)))
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def _validate_non_empty_loader(loader, name: str) -> None:
    try:
        size = len(loader)
    except TypeError:
        size = None

    if size is not None and size == 0:
        raise ValueError(f"{name} is empty; cannot run training/evaluation")


def _autocast_context(
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype = torch.float16,
):
    """Create an AMP autocast context with torch.amp and safe fallback."""
    if device.type != "cuda" or not amp_enabled:
        return nullcontext()

    # Prefer torch.amp API, fallback for older runtime behavior.
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=amp_dtype)
    return torch.cuda.amp.autocast(dtype=amp_dtype)


def _enable_dropout_only(module: nn.Module) -> None:
    """Enable dropout stochastically while keeping other layers (e.g., BatchNorm) in eval mode."""
    if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        module.train()


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    optimizer_name: str = "adamw",
    scheduler_name: str = "none",
    warmup_epochs: int = 0,
    compile_model: bool = True,
    channels_last: bool = True,
    amp_enabled: bool = True,
    device: Optional[str] = None,
    amp_dtype: torch.dtype = torch.float16,
    checkpoint_manager: Optional[CheckpointManager] = None,
    monitor: str = "val_loss",
    parameter_groups: Optional[List[Dict[str, Any]]] = None,
    gradient_clip_norm: float = 0.0,
    label_smoothing: float = 0.0,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    restore_best_weights: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train a classification model with PyTorch 2.x performance best practices.

    Key features:
    - `torch.compile(..., mode="max-autotune")`
    - Channels-last memory layout for 4D image tensors
    - AMP autocast + GradScaler
    - Inner-loop logging avoids `.item()`/`.cpu()` sync stalls

    Returns:
        compiled_model: The (possibly compiled) trained model
        history: Dict with epoch-wise train/val loss and accuracy
    """
    if monitor not in {"val_loss", "val_acc"}:
        raise ValueError(f"monitor must be one of ['val_loss', 'val_acc'], got {monitor}")
    if gradient_clip_norm < 0:
        raise ValueError(f"gradient_clip_norm must be >= 0, got {gradient_clip_norm}")
    if not (0.0 <= label_smoothing < 1.0):
        raise ValueError(f"label_smoothing must be in [0, 1), got {label_smoothing}")
    if early_stopping_patience < 0:
        raise ValueError(
            f"early_stopping_patience must be >= 0, got {early_stopping_patience}"
        )
    if early_stopping_min_delta < 0:
        raise ValueError(
            f"early_stopping_min_delta must be >= 0, got {early_stopping_min_delta}"
        )

    _validate_non_empty_loader(train_loader, name="train_loader")
    _validate_non_empty_loader(val_loader, name="val_loader")

    runtime_device = _get_device(device)
    LOGGER.info("Starting training on device=%s", runtime_device)

    if channels_last:
        model = model.to(device=runtime_device)
    else:
        model = model.to(device=runtime_device)

    if compile_model and hasattr(torch, "compile"):
        compiled_model = torch.compile(model, mode="max-autotune")
        if isinstance(compiled_model, nn.Module):
            model = cast(nn.Module, compiled_model)
            LOGGER.info("torch.compile enabled with mode=max-autotune")
        else:
            LOGGER.warning(
                "torch.compile returned non-Module callable (%s); falling back to eager module",
                type(compiled_model).__name__,
            )

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"train_model expected nn.Module but got {type(model).__name__}. "
            "Set training.compile_model=False if this persists."
        )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = _build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=lr,
        weight_decay=weight_decay,
        parameter_groups=parameter_groups,
    )
    scheduler = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=scheduler_name,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
    )
    LOGGER.info(
        "Optimizer=%s Scheduler=%s WarmupEpochs=%s AMP=%s ChannelsLast=%s",
        optimizer_name,
        scheduler_name,
        warmup_epochs,
        amp_enabled,
        channels_last,
    )
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=(runtime_device.type == "cuda" and amp_enabled))
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(runtime_device.type == "cuda" and amp_enabled))

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lr": [],
    }

    best_metric = float("inf") if monitor == "val_loss" else float("-inf")
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()

        train_loss_tensors: List[torch.Tensor] = []
        train_correct = torch.zeros(1, device=runtime_device)
        train_total = torch.zeros(1, device=runtime_device)

        for inputs, targets in train_loader:
            if channels_last:
                inputs = inputs.to(
                    device=runtime_device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            else:
                inputs = inputs.to(device=runtime_device, non_blocking=True)
            targets = targets.to(device=runtime_device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(runtime_device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
                logits = model(inputs)
                loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                raise RuntimeError("Encountered non-finite loss during training")

            scaler.scale(loss).backward()
            if gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            train_loss_tensors.append(loss.detach())
            preds = torch.argmax(logits.detach(), dim=1)
            train_correct += (preds == targets).sum()
            train_total += targets.new_tensor(targets.size(0), dtype=torch.float32)

        model.eval()
        val_loss_tensors: List[torch.Tensor] = []
        val_correct = torch.zeros(1, device=runtime_device)
        val_total = torch.zeros(1, device=runtime_device)

        with torch.no_grad():
            for inputs, targets in val_loader:
                if channels_last:
                    inputs = inputs.to(
                        device=runtime_device,
                        non_blocking=True,
                        memory_format=torch.channels_last,
                    )
                else:
                    inputs = inputs.to(device=runtime_device, non_blocking=True)
                targets = targets.to(device=runtime_device, non_blocking=True)

                with _autocast_context(runtime_device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
                    logits = model(inputs)
                    loss = criterion(logits, targets)

                if not torch.isfinite(loss):
                    raise RuntimeError("Encountered non-finite loss during validation")

                val_loss_tensors.append(loss.detach())
                preds = torch.argmax(logits.detach(), dim=1)
                val_correct += (preds == targets).sum()
                val_total += targets.new_tensor(targets.size(0), dtype=torch.float32)

        if scheduler is not None:
            scheduler.step()

        # Single sync per metric at epoch end.
        epoch_train_loss = torch.stack(train_loss_tensors).mean().item()
        epoch_val_loss = torch.stack(val_loss_tensors).mean().item()
        epoch_train_acc = (train_correct / train_total).item() * 100.0
        epoch_val_acc = (val_correct / val_total).item() * 100.0

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        metric_value = epoch_val_loss if monitor == "val_loss" else epoch_val_acc
        if monitor == "val_loss":
            improved = metric_value < (best_metric - early_stopping_min_delta)
        else:
            improved = metric_value > (best_metric + early_stopping_min_delta)

        if improved:
            best_metric = metric_value
            epochs_without_improvement = 0
            if restore_best_weights:
                best_state_dict = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
        else:
            epochs_without_improvement += 1

        if checkpoint_manager is not None:
            saved_path = checkpoint_manager.save(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                history=history,
                metric_value=metric_value,
            )
            if saved_path is not None:
                LOGGER.info("Checkpoint saved to %s (monitor=%s value=%.6f)", saved_path, monitor, metric_value)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"train_loss={epoch_train_loss:.4f} val_loss={epoch_val_loss:.4f} "
            f"train_acc={epoch_train_acc:.2f}% val_acc={epoch_val_acc:.2f}% "
            f"lr={optimizer.param_groups[0]['lr']:.3e}"
        )

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            LOGGER.info(
                "Early stopping triggered at epoch %s (monitor=%s, best=%.6f)",
                epoch + 1,
                monitor,
                best_metric,
            )
            break

    if restore_best_weights and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history


def train_model_from_config(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train model from a validated task configuration dictionary."""
    training_cfg = config["training"]
    runtime_cfg = config["runtime"]
    validation_cfg = config["validation"]

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=runtime_cfg["checkpoint_dir"],
        checkpoint_name=runtime_cfg["checkpoint_name"],
        monitor=validation_cfg["metric"],
        mode="min" if validation_cfg["metric"] == "val_loss" else "max",
        save_best_only=validation_cfg["save_best_only"],
    )

    return train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(training_cfg["num_epochs"]),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
        optimizer_name=training_cfg["optimizer"],
        scheduler_name=training_cfg["scheduler"],
        warmup_epochs=int(training_cfg["warmup_epochs"]),
        compile_model=bool(training_cfg["compile_model"]),
        channels_last=bool(training_cfg["channels_last"]),
        amp_enabled=bool(training_cfg["amp"]),
        amp_dtype=_resolve_amp_dtype(training_cfg["amp_dtype"]),
        device=runtime_cfg["device"],
        checkpoint_manager=checkpoint_manager,
        monitor=validation_cfg["metric"],
        gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 0.0)),
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        early_stopping_patience=int(validation_cfg.get("early_stopping_patience", 0)),
        early_stopping_min_delta=float(validation_cfg.get("early_stopping_min_delta", 0.0)),
    )


def _merge_lp_ft_history(
    lp_history: Dict[str, List[float]],
    ft_history: Dict[str, List[float]],
) -> Dict[str, Any]:
    combined = {
        "train_loss": lp_history["train_loss"] + ft_history["train_loss"],
        "val_loss": lp_history["val_loss"] + ft_history["val_loss"],
        "train_acc": lp_history["train_acc"] + ft_history["train_acc"],
        "val_acc": lp_history["val_acc"] + ft_history["val_acc"],
        "lr": lp_history["lr"] + ft_history["lr"],
    }
    return {
        "lp": lp_history,
        "ft": ft_history,
        "combined": combined,
        "lp_epochs": len(lp_history["train_loss"]),
        "ft_epochs": len(ft_history["train_loss"]),
    }


def train_lp_ft_model_from_config(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Run LP-FT transfer learning schedule defined in config.

    LP stage freezes the backbone and trains the classifier head.
    FT stage unfreezes all parameters and trains with lower LR on backbone.
    """
    transfer_cfg = config.get("transfer_learning", {})
    if not transfer_cfg.get("enabled", False):
        model_out, history = train_model_from_config(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
        )
        return model_out, {
            "lp": history,
            "ft": {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []},
            "combined": history,
            "lp_epochs": len(history["train_loss"]),
            "ft_epochs": 0,
        }

    training_cfg = config["training"]
    runtime_cfg = config["runtime"]
    validation_cfg = config["validation"]

    freeze_stats = freeze_backbone_for_linear_probe(
        model=model,
        freeze_strategy=transfer_cfg["freeze_strategy"],
    )
    LOGGER.info(
        "LP phase with freeze strategy '%s': trainable=%s frozen=%s",
        transfer_cfg["freeze_strategy"],
        freeze_stats["trainable_params"],
        freeze_stats["frozen_params"],
    )

    lp_checkpoint_manager = CheckpointManager(
        checkpoint_dir=runtime_cfg["checkpoint_dir"],
        checkpoint_name=transfer_cfg["lp_checkpoint_name"],
        monitor=validation_cfg["metric"],
        mode="min" if validation_cfg["metric"] == "val_loss" else "max",
        save_best_only=validation_cfg["save_best_only"],
    )

    model, lp_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(transfer_cfg["lp_epochs"]),
        lr=float(transfer_cfg["lp_lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
        optimizer_name=training_cfg["optimizer"],
        scheduler_name=training_cfg["scheduler"],
        warmup_epochs=int(training_cfg["warmup_epochs"]),
        compile_model=bool(training_cfg["compile_model"]),
        channels_last=bool(training_cfg["channels_last"]),
        amp_enabled=bool(training_cfg["amp"]),
        amp_dtype=_resolve_amp_dtype(training_cfg["amp_dtype"]),
        device=runtime_cfg["device"],
        checkpoint_manager=lp_checkpoint_manager,
        monitor=validation_cfg["metric"],
        gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 0.0)),
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        early_stopping_patience=int(validation_cfg.get("early_stopping_patience", 0)),
        early_stopping_min_delta=float(validation_cfg.get("early_stopping_min_delta", 0.0)),
    )

    unfreeze_all_parameters(model)
    backbone_params, head_params = _get_backbone_head_parameters(model)
    ft_lr = float(transfer_cfg["ft_lr"])
    ft_parameter_groups = [
        {
            "params": [param for param in backbone_params if param.requires_grad],
            "lr": ft_lr * 0.5,
            "weight_decay": float(training_cfg["weight_decay"]),
        },
        {
            "params": [param for param in head_params if param.requires_grad],
            "lr": ft_lr,
            "weight_decay": float(training_cfg["weight_decay"]),
        },
    ]
    ft_parameter_groups = [group for group in ft_parameter_groups if group["params"]]

    ft_checkpoint_manager = CheckpointManager(
        checkpoint_dir=runtime_cfg["checkpoint_dir"],
        checkpoint_name=transfer_cfg["ft_checkpoint_name"],
        monitor=validation_cfg["metric"],
        mode="min" if validation_cfg["metric"] == "val_loss" else "max",
        save_best_only=validation_cfg["save_best_only"],
    )

    model, ft_history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(transfer_cfg["ft_epochs"]),
        lr=ft_lr,
        weight_decay=float(training_cfg["weight_decay"]),
        optimizer_name=training_cfg["optimizer"],
        scheduler_name=training_cfg["scheduler"],
        warmup_epochs=int(training_cfg["warmup_epochs"]),
        compile_model=bool(training_cfg["compile_model"]),
        channels_last=bool(training_cfg["channels_last"]),
        amp_enabled=bool(training_cfg["amp"]),
        amp_dtype=_resolve_amp_dtype(training_cfg["amp_dtype"]),
        device=runtime_cfg["device"],
        checkpoint_manager=ft_checkpoint_manager,
        monitor=validation_cfg["metric"],
        parameter_groups=ft_parameter_groups,
        gradient_clip_norm=float(training_cfg.get("gradient_clip_norm", 0.0)),
        label_smoothing=float(training_cfg.get("label_smoothing", 0.0)),
        early_stopping_patience=int(validation_cfg.get("early_stopping_patience", 0)),
        early_stopping_min_delta=float(validation_cfg.get("early_stopping_min_delta", 0.0)),
    )

    return model, _merge_lp_ft_history(lp_history=lp_history, ft_history=ft_history)


@torch.no_grad()
def evaluate_mc_dropout(
    model: nn.Module,
    dataloader,
    num_passes: int = 50,
    device: Optional[str] = None,
    channels_last: bool = True,
    amp_enabled: bool = True,
    amp_dtype: torch.dtype = torch.float16,
):
    """Evaluate with Monte Carlo Dropout for epistemic uncertainty estimation.

    Why MC Dropout for physics applications:
    In astrophysical inference, a hard class label alone is often insufficient because
    rare lens morphologies, low-SNR observations, or distribution shift can produce
    overconfident mistakes. MC Dropout performs repeated stochastic forward passes,
    approximating a Bayesian posterior predictive distribution. The predictive mean
    gives the class probability estimate, while predictive variance provides an
    epistemic uncertainty proxy that is useful for anomaly triage and follow-up
    analysis by domain experts.

    Returns:
        y_true: np.ndarray of shape [N]
        mean_probs: np.ndarray of shape [N, C]
        var_probs: np.ndarray of shape [N, C]
        pred_labels: np.ndarray of shape [N]
    """
    if num_passes <= 0:
        raise ValueError(f"num_passes must be > 0, got {num_passes}")

    _validate_non_empty_loader(dataloader, name="dataloader")

    if not _has_dropout_layers(model):
        raise ValueError(
            "MC Dropout requested but no dropout layers were found in the model. "
            "Set a non-zero dropout rate in model config before using uncertainty evaluation."
        )

    runtime_device = _get_device(device)
    if channels_last:
        model = model.to(device=runtime_device)
    else:
        model = model.to(device=runtime_device)

    model.eval()
    model.apply(_enable_dropout_only)
    if not any(
        isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout))
        and module.training
        for module in model.modules()
    ):
        raise RuntimeError("Dropout layers were not activated during MC-dropout evaluation")

    LOGGER.info("Running MC Dropout with passes=%s", num_passes)

    all_true: List[torch.Tensor] = []
    all_mean_probs: List[torch.Tensor] = []
    all_var_probs: List[torch.Tensor] = []

    for inputs, targets in dataloader:
        if channels_last:
            inputs = inputs.to(device=runtime_device, non_blocking=True, memory_format=torch.channels_last)
        else:
            inputs = inputs.to(device=runtime_device, non_blocking=True)
        targets = targets.to(device=runtime_device, non_blocking=True)

        stochastic_probs: List[torch.Tensor] = []
        for _ in range(num_passes):
            with _autocast_context(runtime_device, amp_enabled=amp_enabled, amp_dtype=amp_dtype):
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1)
            stochastic_probs.append(probs)

        stacked = torch.stack(stochastic_probs, dim=0)  # [T, B, C]
        mean_probs = stacked.mean(dim=0)
        var_probs = stacked.var(dim=0, unbiased=False)

        all_true.append(targets)
        all_mean_probs.append(mean_probs)
        all_var_probs.append(var_probs)

    y_true_t = torch.cat(all_true, dim=0)
    mean_probs_t = torch.cat(all_mean_probs, dim=0)
    var_probs_t = torch.cat(all_var_probs, dim=0)
    pred_labels_t = torch.argmax(mean_probs_t, dim=1)

    if (var_probs_t < 0).any():
        raise RuntimeError("Negative predictive variance detected during MC-dropout evaluation")

    LOGGER.info(
        "MC Dropout complete: samples=%s classes=%s mean_variance=%.6e",
        mean_probs_t.shape[0],
        mean_probs_t.shape[1],
        var_probs_t.mean().item(),
    )

    return (
        y_true_t.cpu().numpy(),
        mean_probs_t.cpu().numpy(),
        var_probs_t.cpu().numpy(),
        pred_labels_t.cpu().numpy(),
    )


def plot_and_calculate_roc(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: Tuple[str, str, str] = ("no", "sphere", "vort"),
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
) -> Dict[str, object]:
    """Compute and plot multiclass ROC curves with macro OVR AUC.

    Args:
        y_true: Integer labels with values in {0,1,2}
        y_probs: Predicted class probabilities of shape [N, 3]
        class_names: Human-readable class names in index order

    Returns:
        Dict containing macro AUC and per-class ROC metadata.
    """
    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)

    if y_probs.ndim != 2 or y_probs.shape[1] != 3:
        raise ValueError(f"Expected y_probs shape [N, 3], got {y_probs.shape}")

    macro_auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")

    classes = np.array([0, 1, 2])
    y_true_bin = np.asarray(label_binarize(y_true, classes=classes))

    per_class: Dict[int, Dict[str, object]] = {}
    plt.figure(figsize=(10, 8))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for class_idx, color in zip(range(3), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_probs[:, class_idx])
        class_auc = auc(fpr, tpr)
        per_class[class_idx] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(class_auc),
        }

        plt.plot(
            fpr,
            tpr,
            lw=2.2,
            color=color,
            label=f"Class {class_idx} ({class_names[class_idx]}) AUC = {class_auc:.4f}",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.5, label="Random chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(f"DeepLense Multiclass ROC (Macro OVR AUC = {macro_auc:.4f})", fontsize=14)
    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.8)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")

    if show_plot:
        plt.show()
    else:
        plt.close()

    return {
        "macro_auc_ovr": float(macro_auc),
        "per_class": per_class,
    }


class SuperResolutionTrainer:
    """Lightweight trainer for Task 6 notebook compatibility.

    This class provides a minimal training/validation API used by existing
    Task6 notebooks while keeping logging and error checks production-safe.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer,
        criterion,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        resolved_device = str(device) if isinstance(device, torch.device) else device
        self.device = _get_device(resolved_device)
        self.model = self.model.to(self.device)

    def train_epoch(self, train_loader) -> float:
        _validate_non_empty_loader(train_loader, name="train_loader")

        self.model.train()
        epoch_losses: List[torch.Tensor] = []

        for lr_imgs, hr_imgs in train_loader:
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            sr_imgs = self.model(lr_imgs)
            loss = self.criterion(sr_imgs, hr_imgs)

            if not torch.isfinite(loss):
                raise RuntimeError("Encountered non-finite SR training loss")

            loss.backward()
            self.optimizer.step()
            epoch_losses.append(loss.detach())

        mean_loss = torch.stack(epoch_losses).mean().item()
        LOGGER.info("Task6 train epoch complete: loss=%.6f", mean_loss)
        return mean_loss

    @torch.no_grad()
    def validate(self, val_loader) -> float:
        _validate_non_empty_loader(val_loader, name="val_loader")

        self.model.eval()
        epoch_losses: List[torch.Tensor] = []

        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)

            sr_imgs = self.model(lr_imgs)
            loss = self.criterion(sr_imgs, hr_imgs)

            if not torch.isfinite(loss):
                raise RuntimeError("Encountered non-finite SR validation loss")

            epoch_losses.append(loss.detach())

        mean_loss = torch.stack(epoch_losses).mean().item()
        LOGGER.info("Task6 validation complete: loss=%.6f", mean_loss)
        return mean_loss


__all__ = [
    "CheckpointManager",
    "SuperResolutionTrainer",
    "freeze_backbone_for_linear_probe",
    "train_model",
    "train_model_from_config",
    "train_lp_ft_model_from_config",
    "unfreeze_all_parameters",
    "evaluate_mc_dropout",
    "plot_and_calculate_roc",
]
