#!/usr/bin/env python3
"""Fast model/training sanity check for a handful of steps using existing modules."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict
import sys

import torch
import torch.optim as optim
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import get_dataloaders
from models.wrapper import DINOSFIN_Architecture_NEW
from utils.losses import SILogLoss
from train import build_scheduler, compute_multistage_loss, curriculum_weights, save_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick training sanity check")
    p.add_argument("--config", required=True, help="Config path")
    p.add_argument("--train-steps", type=int, default=3, help="Train steps to execute")
    p.add_argument("--val-steps", type=int, default=1, help="Validation steps to execute")
    return p.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid config root: {path}")
    return cfg


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")
    amp_enabled = bool(cfg["hardware"].get("amp", True)) and device.type == "cuda"

    model = DINOSFIN_Architecture_NEW(
        strategy=cfg["architecture"]["strategy"],
        base_channels=int(cfg["architecture"]["base_channels"]),
        num_sfin=int(cfg["architecture"]["num_sfin"]),
        num_rhag=int(cfg["architecture"]["num_rhag"]),
        window_size=int(cfg["architecture"]["window_size"]),
        dino_embed_dim=int(cfg["architecture"]["dino_embed_dim"]),
        fft_mode=cfg["architecture"]["fft_mode"],
        fft_pad_size=int(cfg["architecture"]["fft_pad_size"]),
        use_precomputed_dino=bool(cfg["data"].get("use_precomputed_dino", False)),
    ).to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    scheduler = build_scheduler(optimizer, cfg)
    criterion = SILogLoss()
    scaler = torch.amp.GradScaler(device=device.type, enabled=amp_enabled)

    train_loader, val_loader = get_dataloaders(cfg)
    decoder_w, bottleneck_w = curriculum_weights(0, max(1, int(cfg["training"]["epochs"])), cfg)
    grad_clip_norm = float(cfg["training"].get("grad_clip_norm", 1.0))

    print(f"[info] device={device.type} amp={amp_enabled} train_steps={args.train_steps} val_steps={args.val_steps}")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(train_loader):
        if i >= args.train_steps:
            break

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
                use_ckpt=False,
                precomputed_dino=precomputed_dino,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        print(
            f"[train-step {i+1}] loss={float(loss.item()):.5f} "
            f"grad_norm={float(grad_norm.item()):.4f} "
            f"l1_f={components['l1_f']:.5f} l2_f={components['l2_f']:.5f}"
        )

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.val_steps:
                break

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
                    use_ckpt=False,
                    precomputed_dino=precomputed_dino,
                )
            print(f"[val-step {i+1}] loss={float(val_loss.item()):.5f}")

    scheduler.step()

    out_dir = Path(cfg["training"]["checkpoint"]["dir"]) / "quickcheck"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "sanity_roundtrip.pth"
    save_checkpoint(ckpt_path, 0, float("inf"), model, optimizer, scheduler, scaler)
    _ = torch.load(ckpt_path, map_location="cpu")
    print(f"[OK] checkpoint roundtrip: {ckpt_path}")

    print("[done] quickcheck_train passed")


if __name__ == "__main__":
    main()
