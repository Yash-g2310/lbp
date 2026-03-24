from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
import yaml

from data.dataset import get_dataloaders
from models.wrapper import DINOSFIN_Architecture_NEW
from utils.logger import setup_wandb
from utils.losses import SILogLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Layered Depth Estimation")
    parser.add_argument("--config", type=str, default="configs/local.yaml", help="Path to config YAML")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_main_process() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def log_terminal(enabled: bool, message: str) -> None:
    if enabled and is_main_process():
        print(message, flush=True)


def tensor_stats(x: torch.Tensor | None, name: str) -> str:
    if x is None:
        return f"{name}=None"
    with torch.no_grad():
        finite = torch.isfinite(x)
        finite_count = int(finite.sum().item())
        total = int(x.numel())
        if finite_count == 0:
            return f"{name}: shape={tuple(x.shape)} finite=0/{total}"

        x_f = x[finite]
        return (
            f"{name}: shape={tuple(x.shape)} dtype={x.dtype} "
            f"finite={finite_count}/{total} min={float(x_f.min().item()):.6g} "
            f"max={float(x_f.max().item()):.6g} mean={float(x_f.mean().item()):.6g}"
        )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    sched_cfg = cfg["training"]["scheduler"]
    name = sched_cfg["name"].lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(sched_cfg["t_max_epochs"]),
            eta_min=float(sched_cfg["eta_min"]),
        )
    raise ValueError(f"Unsupported scheduler: {sched_cfg['name']}")


def curriculum_weights(epoch: int, total_epochs: int, cfg: Dict[str, Any]) -> Tuple[float, float]:
    cur_cfg = cfg["training"]["curriculum"]
    if not cur_cfg.get("enabled", True):
        return float(cur_cfg["decoder_weight"]), float(cur_cfg["bottleneck_weight"])

    midpoint = int(total_epochs * float(cur_cfg["midpoint_fraction"]))
    if epoch < midpoint:
        return float(cur_cfg["decoder_weight"]), float(cur_cfg["bottleneck_weight"])

    tail_len = max(1, total_epochs - midpoint - 1)
    progress = (epoch - midpoint) / tail_len
    min_decay = float(cur_cfg.get("min_decay", 0.0))
    decay = max(min_decay, 1.0 - progress)
    return float(cur_cfg["decoder_weight"]) * decay, float(cur_cfg["bottleneck_weight"]) * decay


def compute_multistage_loss(
    model: torch.nn.Module,
    criterion: SILogLoss,
    images: torch.Tensor,
    depth_1: torch.Tensor,
    depth_2: torch.Tensor,
    decoder_w: float,
    bottleneck_w: float,
    use_ckpt: bool,
    precomputed_dino: torch.Tensor | None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    out1 = model(
        images,
        target_layer=1,
        return_intermediate=True,
        use_checkpointing=use_ckpt,
        precomputed_dino=precomputed_dino,
    )
    out2 = model(
        images,
        target_layer=2,
        return_intermediate=True,
        use_checkpointing=use_ckpt,
        precomputed_dino=precomputed_dino,
    )

    l1_b = criterion(out1["bottleneck"], depth_1)
    l1_d = criterion(out1["decoder"], depth_1)
    l1_f = criterion(out1["final"], depth_1)
    l2_b = criterion(out2["bottleneck"], depth_2)
    l2_d = criterion(out2["decoder"], depth_2)
    l2_f = criterion(out2["final"], depth_2)

    total = (
        l1_f
        + decoder_w * l1_d
        + bottleneck_w * l1_b
        + l2_f
        + decoder_w * l2_d
        + bottleneck_w * l2_b
    )
    stats = {
        "l1_b": float(l1_b.detach().item()),
        "l1_d": float(l1_d.detach().item()),
        "l1_f": float(l1_f.detach().item()),
        "l2_b": float(l2_b.detach().item()),
        "l2_d": float(l2_d.detach().item()),
        "l2_f": float(l2_f.detach().item()),
    }
    return total, stats


def save_checkpoint(
    path: Path,
    epoch: int,
    best_val_loss: float,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": 2,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
        },
        path,
    )


def maybe_resume(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    if not path.exists():
        return 0, float("inf")

    checkpoint = torch.load(path, map_location=device)
    model_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    opt_key = "optimizer_state" if "optimizer_state" in checkpoint else "optimizer_state_dict"
    sch_key = "scheduler_state" if "scheduler_state" in checkpoint else "scheduler_state_dict"
    sca_key = "scaler_state" if "scaler_state" in checkpoint else "scaler_state_dict"

    model.load_state_dict(checkpoint[model_key])
    optimizer.load_state_dict(checkpoint[opt_key])
    scheduler.load_state_dict(checkpoint[sch_key])
    scaler.load_state_dict(checkpoint[sca_key])

    best_val = checkpoint.get("best_val_loss", checkpoint.get("best_loss", float("inf")))
    return int(checkpoint["epoch"]) + 1, float(best_val)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(int(config["experiment"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() and config["hardware"]["device"] == "cuda" else "cpu")
    amp_enabled = bool(config["hardware"].get("amp", True)) and device.type == "cuda"
    main_process = is_main_process()
    log_cfg = config.get("logging", {})
    log_to_terminal = bool(log_cfg.get("log_to_terminal", True))
    verbose_components = bool(log_cfg.get("verbose_components", True))

    model = DINOSFIN_Architecture_NEW(
        strategy=config["architecture"]["strategy"],
        base_channels=int(config["architecture"]["base_channels"]),
        num_sfin=int(config["architecture"]["num_sfin"]),
        num_rhag=int(config["architecture"]["num_rhag"]),
        window_size=int(config["architecture"]["window_size"]),
        dino_embed_dim=int(config["architecture"]["dino_embed_dim"]),
        fft_mode=config["architecture"]["fft_mode"],
        fft_pad_size=int(config["architecture"]["fft_pad_size"]),
        use_precomputed_dino=bool(config["data"]["use_precomputed_dino"]),
    ).to(device)

    if bool(config["hardware"].get("compile_model", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, config)
    criterion = SILogLoss()
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)

    train_loader, val_loader = get_dataloaders(config)

    ckpt_cfg = config["training"]["checkpoint"]
    ckpt_dir = Path(ckpt_cfg["dir"])
    latest_ckpt = ckpt_dir / ckpt_cfg["latest_name"]
    best_ckpt = ckpt_dir / ckpt_cfg["best_name"]

    start_epoch, best_val_loss = maybe_resume(latest_ckpt, model, optimizer, scheduler, scaler, device)
    run = setup_wandb(config, model=model if main_process else None)

    accum_steps = int(config["training"]["accum_steps"])
    max_epochs = int(config["training"]["epochs"])
    log_every = int(log_cfg.get("train_log_every_steps", log_cfg.get("log_every_steps", 20)))
    use_ckpt = bool(config["architecture"].get("use_gradient_checkpointing", False))
    grad_clip_norm = float(config["training"].get("grad_clip_norm", 1.0))
    global_step = start_epoch * max(1, len(train_loader))

    log_terminal(
        log_to_terminal,
        (
            f"[train] device={device.type} amp={amp_enabled} compile={bool(config['hardware'].get('compile_model', False))} "
            f"precomputed_dino={bool(config['data'].get('use_precomputed_dino', False))} start_epoch={start_epoch}"
        ),
    )
    if run is not None:
        log_terminal(log_to_terminal, f"[train] W&B enabled: project={log_cfg.get('project')} run={run.name}")
    log_terminal(log_to_terminal, f"[train] Logging cadence: every {log_every} train steps")

    sched_cfg = config["training"]["scheduler"]
    if sched_cfg["name"].lower() == "cosine":
        t_max_epochs = int(sched_cfg["t_max_epochs"])
        if t_max_epochs != max_epochs:
            log_terminal(
                log_to_terminal,
                f"[warn] scheduler.t_max_epochs ({t_max_epochs}) != training.epochs ({max_epochs}); verify annealing intent",
            )

    for epoch in range(start_epoch, max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_train_loss = 0.0
        step_counter = 0
        decoder_w, bottleneck_w = curriculum_weights(epoch, max_epochs, config)

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device, non_blocking=True)
            depth_1 = batch["depth_1"].to(device, non_blocking=True)
            depth_2 = batch["depth_2"].to(device, non_blocking=True)
            precomputed_dino = batch.get("dino_features")
            if precomputed_dino is not None:
                precomputed_dino = precomputed_dino.to(device, non_blocking=True)

            amp_ctx = torch.autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            with amp_ctx:
                loss, components = compute_multistage_loss(
                    model,
                    criterion,
                    images,
                    depth_1,
                    depth_2,
                    decoder_w,
                    bottleneck_w,
                    use_ckpt,
                    precomputed_dino,
                )
                scaled_loss = loss / accum_steps

            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Non-finite training loss detected. "
                    "Try safer numerics for server runs (e.g., architecture.fft_mode=fp32 and/or hardware.amp=false). "
                    f"epoch={epoch+1} step={step+1} components={components}; "
                    f"{tensor_stats(images, 'images')}; {tensor_stats(depth_1, 'depth_1')}; "
                    f"{tensor_stats(depth_2, 'depth_2')}; {tensor_stats(precomputed_dino, 'precomputed_dino')}"
                )

            scaler.scale(scaled_loss).backward()

            should_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm * accum_steps)
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"Non-finite gradient norm detected at epoch={epoch+1} step={step+1}: grad_norm={float(grad_norm)}"
                    )

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    log_terminal(
                        log_to_terminal,
                        f"[warn] AMP overflow/skip at epoch={epoch+1} step={step+1}; scaler {scale_before} -> {scaler.get_scale()}",
                    )
                optimizer.zero_grad(set_to_none=True)
            else:
                grad_norm = None

            global_step += 1

            loss_value = float(loss.detach().item())
            epoch_train_loss += loss_value
            step_counter += 1

            if ((step + 1) % log_every == 0):
                log_payload = {
                    "train/loss": loss_value,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/decoder_weight": decoder_w,
                    "train/bottleneck_weight": bottleneck_w,
                    "train/epoch": epoch,
                    "train/step_in_epoch": step,
                    **{f"train/{k}": v for k, v in components.items()},
                }
                if grad_norm is not None:
                    log_payload["train/grad_norm"] = float(grad_norm.item())

                if run is not None and main_process:
                    run.log(log_payload, step=global_step)

                if main_process and log_to_terminal:
                    base_msg = (
                        f"[train][epoch={epoch+1}/{max_epochs} step={step+1}/{len(train_loader)} global_step={global_step}] "
                        f"loss={loss_value:.5f} lr={scheduler.get_last_lr()[0]:.3e} "
                        f"grad_norm={float(grad_norm.item()):.4f}" if grad_norm is not None else
                        f"[train][epoch={epoch+1}/{max_epochs} step={step+1}/{len(train_loader)} global_step={global_step}] "
                        f"loss={loss_value:.5f} lr={scheduler.get_last_lr()[0]:.3e}"
                    )
                    if verbose_components:
                        comp_msg = " ".join(
                            [f"{k}={v:.5f}" for k, v in components.items()]
                        )
                        log_terminal(True, f"{base_msg} {comp_msg}")
                    else:
                        log_terminal(True, base_msg)

        train_loss = epoch_train_loss / max(1, step_counter)

        model.eval()
        val_running = 0.0
        val_steps = 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device, non_blocking=True)
                depth_1 = batch["depth_1"].to(device, non_blocking=True)
                depth_2 = batch["depth_2"].to(device, non_blocking=True)
                precomputed_dino = batch.get("dino_features")
                if precomputed_dino is not None:
                    precomputed_dino = precomputed_dino.to(device, non_blocking=True)

                amp_ctx = torch.autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
                with amp_ctx:
                    val_loss, _ = compute_multistage_loss(
                        model,
                        criterion,
                        images,
                        depth_1,
                        depth_2,
                        decoder_w,
                        bottleneck_w,
                        use_ckpt,
                        precomputed_dino,
                    )
                if not torch.isfinite(val_loss):
                    raise RuntimeError(
                        "Non-finite validation loss detected. "
                        f"epoch={epoch+1} val_step={val_steps+1} val_loss={float(val_loss.detach().item())}; "
                        f"{tensor_stats(images, 'images')}; {tensor_stats(depth_1, 'depth_1')}; "
                        f"{tensor_stats(depth_2, 'depth_2')}; {tensor_stats(precomputed_dino, 'precomputed_dino')}"
                    )
                val_running += float(val_loss.detach().item())
                val_steps += 1

        val_loss = val_running / max(1, val_steps)
        if not np.isfinite(train_loss) or not np.isfinite(val_loss):
            raise RuntimeError(
                f"Epoch aggregate loss became non-finite at epoch={epoch+1}: train_loss={train_loss}, val_loss={val_loss}"
            )
        scheduler.step()

        if run is not None and main_process:
            run.log(
                {
                    "epoch": epoch,
                    "train/epoch_loss": train_loss,
                    "val/epoch_loss": val_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                },
                step=global_step,
            )

        log_terminal(
            log_to_terminal,
            (
                f"[epoch {epoch+1}/{max_epochs}] train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
                f"lr={scheduler.get_last_lr()[0]:.3e} decoder_w={decoder_w:.4f} bottleneck_w={bottleneck_w:.4f}"
            ),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(best_ckpt, epoch, best_val_loss, model, optimizer, scheduler, scaler)
            log_terminal(log_to_terminal, f"[ckpt] Saved new best checkpoint: {best_ckpt} (val_loss={val_loss:.5f})")

        if (epoch + 1) % int(ckpt_cfg.get("save_every_epochs", 1)) == 0:
            save_checkpoint(latest_ckpt, epoch, best_val_loss, model, optimizer, scheduler, scaler)
            log_terminal(log_to_terminal, f"[ckpt] Updated latest checkpoint: {latest_ckpt}")

    if run is not None:
        run.finish()
        log_terminal(log_to_terminal, "[wandb] Run finished and synced.")


if __name__ == "__main__":
    main()