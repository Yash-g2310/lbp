from __future__ import annotations

from contextlib import nullcontext
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .losses import CompositeRectifiedFlowLoss
from .train import CheckpointManager


LOGGER = logging.getLogger(__name__)


def _get_device(device: Optional[str] = None) -> torch.device:
    if device in {None, "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device is set to 'cuda' but no CUDA device is available")
    return resolved


def _validate_non_empty_loader(loader, name: str) -> None:
    try:
        size = len(loader)
    except TypeError:
        size = None
    if size is not None and size == 0:
        raise ValueError(f"{name} is empty; cannot run")


def _resolve_amp_dtype(dtype_name: str) -> torch.dtype:
    key = dtype_name.strip().lower()
    if key == "float16":
        return torch.float16
    if key == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported amp dtype '{dtype_name}'. Use 'float16' or 'bfloat16'.")


def _autocast_context(device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype):
    if device.type != "cuda" or not amp_enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    eps: float,
) -> torch.optim.Optimizer:
    name = optimizer_name.strip().lower()
    if learning_rate <= 0.0:
        raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
    if weight_decay < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay}")
    if eps <= 0.0:
        raise ValueError(f"eps must be > 0, got {eps}")

    params_with_decay = []
    params_without_decay = []
    no_decay_names = []

    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_gate_param = ("skip_gates" in param_name) or param_name.endswith(".gate") or param_name == "gate"
        if is_gate_param:
            params_without_decay.append(param)
            no_decay_names.append(param_name)
        else:
            params_with_decay.append(param)

    if not params_with_decay and not params_without_decay:
        raise ValueError("No trainable parameters found for optimizer construction")

    param_groups = []
    if params_with_decay:
        param_groups.append({"params": params_with_decay, "weight_decay": weight_decay})
    if params_without_decay:
        param_groups.append({"params": params_without_decay, "weight_decay": 0.0})

    if name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, eps=eps)
    elif name == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=learning_rate, eps=eps)
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'")

    if no_decay_names:
        LOGGER.info(
            "Optimizer no-decay gate params: count=%d names=%s",
            len(no_decay_names),
            no_decay_names,
        )
    return optimizer


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int,
):
    key = scheduler_name.strip().lower()
    if key == "none":
        return None
    if key != "cosine":
        raise ValueError(f"Unsupported scheduler '{scheduler_name}'")
    if warmup_epochs < 0:
        raise ValueError(f"warmup_epochs must be >= 0, got {warmup_epochs}")

    cosine_tmax = max(1, epochs - max(0, warmup_epochs))
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_tmax)
    if warmup_epochs <= 0:
        return cosine_scheduler

    warmup_lambda = lambda epoch: min(1.0, float(epoch + 1) / float(max(1, warmup_epochs)))
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    if warmup_epochs >= epochs:
        return warmup_scheduler

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )


def _ramp_weight(target_weight: float, epoch: int, ramp_epochs: int) -> float:
    if target_weight < 0.0:
        raise ValueError(f"target_weight must be >= 0, got {target_weight}")
    if ramp_epochs < 0:
        raise ValueError(f"ramp_epochs must be >= 0, got {ramp_epochs}")
    if target_weight == 0.0 or ramp_epochs == 0:
        return target_weight
    scale = min(1.0, float(epoch) / float(ramp_epochs))
    return target_weight * scale


class ReflowSRTrainer:
    """Rectified-flow super-resolution trainer with AMP and accumulation."""

    def __init__(
        self,
        model: nn.Module,
        criterion: CompositeRectifiedFlowLoss,
        optimizer: torch.optim.Optimizer,
        device: Optional[Union[str, torch.device]] = None,
        amp_enabled: bool = True,
        amp_dtype: torch.dtype = torch.bfloat16,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float = 1.0,
    ):
        if gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if gradient_clip_norm <= 0:
            raise ValueError("gradient_clip_norm must be > 0")

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        resolved_device = str(device) if isinstance(device, torch.device) else device
        self.device = _get_device(resolved_device)
        self.model = self.model.to(self.device)
        self.amp_enabled = bool(amp_enabled and self.device.type == "cuda")
        self.amp_dtype = amp_dtype
        self.grad_accum_steps = int(gradient_accumulation_steps)
        self.grad_clip_norm = float(gradient_clip_norm)
        scaler_enabled = bool(self.amp_enabled and self.amp_dtype == torch.float16)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)  # type: ignore[attr-defined]
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    def _sample_flow_state(
        self,
        hr_imgs: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build HR-space state x_t and velocity target v*=x_hr-noise."""
        if hr_imgs.ndim != 4:
            raise ValueError(f"hr_imgs must be rank-4 [B,C,H,W], got {tuple(hr_imgs.shape)}")

        b = hr_imgs.shape[0]
        if t is None:
            t = torch.rand(b, device=hr_imgs.device, dtype=hr_imgs.dtype)
        else:
            t = t.to(device=hr_imgs.device, dtype=hr_imgs.dtype)
            if t.ndim == 2 and t.shape[1] == 1:
                t = t.squeeze(1)
            if t.ndim != 1 or t.shape[0] != b:
                raise ValueError(f"t must have shape [B] or [B,1], got {tuple(t.shape)}")

        noise = torch.randn_like(hr_imgs)
        t_view = t[:, None, None, None]
        x_t = (1.0 - t_view) * noise + t_view * hr_imgs
        v_target = hr_imgs - noise
        return x_t, t, v_target

    def train_epoch(self, train_loader, scheduler=None, log_frequency: int = 10) -> Dict[str, float]:
        _validate_non_empty_loader(train_loader, name="train_loader")

        self.model.train()
        meter: Dict[str, List[torch.Tensor]] = {
            "total_loss": [],
            "flow_loss": [],
            "mass_loss": [],
            "freq_loss": [],
        }

        self.optimizer.zero_grad(set_to_none=True)
        for step_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)

            x_t, t, v_target = self._sample_flow_state(hr_imgs)

            with _autocast_context(self.device, self.amp_enabled, self.amp_dtype):
                v_pred = self.model(x_t, t, lr_imgs)
                comp = self.criterion.compute_components(v_pred, v_target, x_t, t, lr_imgs, hr_imgs)
                total_loss = comp["total_loss"] / self.grad_accum_steps

            if not torch.isfinite(total_loss):
                raise RuntimeError("Encountered non-finite training loss")

            self.scaler.scale(total_loss).backward()

            meter["total_loss"].append(comp["total_loss"].detach())
            meter["flow_loss"].append(comp["flow_loss"].detach())
            meter["mass_loss"].append(comp["mass_loss"].detach())
            meter["freq_loss"].append(comp["freq_loss"].detach())

            should_step = (step_idx + 1) % self.grad_accum_steps == 0
            if should_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            if log_frequency > 0 and (step_idx + 1) % log_frequency == 0:
                LOGGER.info("Task6 reflow train step %d/%d", step_idx + 1, len(train_loader))

        if scheduler is not None:
            scheduler.step()

        return {k: torch.stack(v).mean().item() for k, v in meter.items()}

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        _validate_non_empty_loader(val_loader, name="val_loader")

        self.model.eval()
        meter: Dict[str, List[torch.Tensor]] = {
            "total_loss": [],
            "flow_loss": [],
            "mass_loss": [],
            "freq_loss": [],
        }

        for lr_imgs, hr_imgs in val_loader:
            lr_imgs = lr_imgs.to(self.device, non_blocking=True)
            hr_imgs = hr_imgs.to(self.device, non_blocking=True)

            x_t, t, v_target = self._sample_flow_state(hr_imgs)
            with _autocast_context(self.device, self.amp_enabled, self.amp_dtype):
                v_pred = self.model(x_t, t, lr_imgs)
                comp = self.criterion.compute_components(v_pred, v_target, x_t, t, lr_imgs, hr_imgs)

            for key in meter:
                meter[key].append(comp[key].detach())

        return {k: torch.stack(v).mean().item() for k, v in meter.items()}

    def get_checkpoint_dict(self) -> Dict[str, Any]:
        return {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.amp_enabled else None,
        }

    def load_checkpoint_dict(self, ckpt: Dict[str, Any]) -> None:
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scaler_state = ckpt.get("scaler_state_dict")
        if self.amp_enabled and scaler_state is not None:
            self.scaler.load_state_dict(scaler_state)


class ReflowSampler:
    """Euler ODE sampler for PI-HAF rectified flow."""

    def __init__(
        self,
        model: nn.Module,
        num_steps: int = 50,
        device: Optional[Union[str, torch.device]] = None,
    ):
        if num_steps <= 0:
            raise ValueError("num_steps must be > 0")
        self.model = model
        resolved_device = str(device) if isinstance(device, torch.device) else device
        self.device = _get_device(resolved_device)
        self.model = self.model.to(self.device)
        self.num_steps = int(num_steps)

    @torch.no_grad()
    def sample(self, x_lr: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        if x_lr.ndim != 4:
            raise ValueError(f"x_lr must be rank-4 [B,C,H,W], got {tuple(x_lr.shape)}")
        if num_samples <= 0:
            raise ValueError("num_samples must be > 0")

        self.model.eval()
        x_lr = x_lr.to(self.device)
        if num_samples > 1:
            x_lr = x_lr.repeat_interleave(num_samples, dim=0)

        b, c, h, w = x_lr.shape
        phi = torch.randn(b, c, h * 2, w * 2, device=self.device, dtype=x_lr.dtype)
        ts = torch.linspace(0.0, 1.0, steps=self.num_steps + 1, device=self.device, dtype=x_lr.dtype)

        for idx in range(self.num_steps):
            t = torch.full((b,), float(ts[idx].item()), device=self.device, dtype=x_lr.dtype)
            dt = ts[idx + 1] - ts[idx]
            v = self.model(phi, t, x_lr)
            phi = phi + dt * v

        # Inference continuity contract: convert normalized [-1,1] state to [0,1].
        phi = torch.clamp((phi * 0.5) + 0.5, 0.0, 1.0)
        return phi


def train_sr_from_config(
    model: nn.Module,
    train_loader,
    val_loader,
    config: Dict[str, Any],
    checkpoint_dir: str = "weights",
    checkpoint_name: str = "task6a_reflow_best.pth",
    device: str = "auto",
    num_epochs: Optional[int] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train PI-HAF with config-driven optimizer/loss/trainer wiring."""

    training_cfg = config.get("training", {})
    losses_cfg = config.get("losses", {})
    vram_cfg = config.get("vram", {})

    epochs = int(num_epochs if num_epochs is not None else training_cfg.get("num_epochs", 100))
    lr = float(training_cfg.get("learning_rate", 1e-4))
    wd = float(training_cfg.get("weight_decay", 1e-5))
    optimizer_name = str(training_cfg.get("optimizer", "adamw")).lower()
    warmup_epochs = int(training_cfg.get("warmup_epochs", 0))

    amp_dtype = _resolve_amp_dtype(str(vram_cfg.get("amp_dtype", "bfloat16")))
    default_eps = 1e-6 if amp_dtype == torch.bfloat16 else 1e-8
    optimizer_eps = float(training_cfg.get("adamw_eps", default_eps))
    optimizer = _build_optimizer(
        model=model,
        optimizer_name=optimizer_name,
        learning_rate=lr,
        weight_decay=wd,
        eps=optimizer_eps,
    )

    criterion = CompositeRectifiedFlowLoss(
        flow_weight=float(losses_cfg.get("flow_weight", 1.0)),
        mass_weight=float(losses_cfg.get("mass_weight", 0.1)),
        freq_weight=float(losses_cfg.get("freq_weight", 0.1)),
        reduction=str(losses_cfg.get("reduction", "mean")),
        freq_exponent=float(losses_cfg.get("freq_exponent", 2.0)),
    )

    trainer = ReflowSRTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        amp_enabled=bool(vram_cfg.get("enable_mixed_precision", True)),
        amp_dtype=amp_dtype,
        gradient_accumulation_steps=int(vram_cfg.get("gradient_accumulation_steps", 2)),
        gradient_clip_norm=float(vram_cfg.get("gradient_clip_norm", 1.0)),
    )

    scheduler = _build_scheduler(
        optimizer=optimizer,
        scheduler_name=str(training_cfg.get("scheduler", "none")),
        epochs=epochs,
        warmup_epochs=warmup_epochs,
    )

    base_flow_weight = float(criterion.flow_weight)
    base_mass_weight = float(criterion.mass_weight)
    base_freq_weight = float(criterion.freq_weight)
    mass_ramp_epochs = int(losses_cfg.get("mass_ramp_epochs", warmup_epochs))
    freq_ramp_epochs = int(losses_cfg.get("freq_ramp_epochs", warmup_epochs))

    monitor = str(config.get("validation", {}).get("monitor", "val_loss"))
    monitor = "val_loss" if monitor not in {"val_loss", "val_acc"} else monitor
    mode = str(config.get("validation", {}).get("mode", "min"))
    mode = "min" if mode not in {"min", "max"} else mode

    checkpoint_mgr = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        monitor=monitor,
        mode=mode,
        save_best_only=bool(config.get("validation", {}).get("save_best_only", True)),
    )

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

    for epoch in range(1, epochs + 1):
        criterion.flow_weight = base_flow_weight
        criterion.mass_weight = _ramp_weight(base_mass_weight, epoch=epoch, ramp_epochs=mass_ramp_epochs)
        criterion.freq_weight = _ramp_weight(base_freq_weight, epoch=epoch, ramp_epochs=freq_ramp_epochs)

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

        metric_value = val_metrics["total_loss"] if monitor == "val_loss" else -val_metrics["total_loss"]
        checkpoint_mgr.save(
            epoch=epoch,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=scheduler,  # type: ignore[arg-type]
            history=history,
            metric_value=metric_value,
        )

        LOGGER.info(
            "Epoch %d/%d | train_total=%.6f val_total=%.6f | loss_w(flow=%.3f,mass=%.3f,freq=%.3f)",
            epoch,
            epochs,
            train_metrics["total_loss"],
            val_metrics["total_loss"],
            criterion.flow_weight,
            criterion.mass_weight,
            criterion.freq_weight,
        )

    return trainer.model, history


__all__ = [
    "ReflowSRTrainer",
    "ReflowSampler",
    "train_sr_from_config",
]
