#!/usr/bin/env python3
"""Evaluate trained model on synthetic validation split with dense depth metrics."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import get_dataloaders
from models.wrapper import DINOSFIN_Architecture_NEW
from utils.losses import SILogLoss
from utils.metrics import RunningAverage, compute_depth_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate synthetic dense depth metrics")
    p.add_argument("--config", required=True, help="Config path")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path")
    p.add_argument("--max-batches", type=int, default=0, help="Limit validation batches (0=all)")
    p.add_argument("--output", required=True, help="Output JSON path")
    return p.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    return cfg


def build_model(cfg: Dict[str, Any], device: torch.device) -> torch.nn.Module:
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
    return model


def load_checkpoint_model(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[model_key])


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")
    amp_enabled = bool(cfg["hardware"].get("amp", True)) and device.type == "cuda"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _, val_loader = get_dataloaders(cfg)
    model = build_model(cfg, device)
    load_checkpoint_model(model, checkpoint_path, device)
    model.eval()

    criterion = SILogLoss()

    silog_l1 = RunningAverage()
    silog_l2 = RunningAverage()
    abs_rel_l1 = RunningAverage()
    abs_rel_l2 = RunningAverage()
    rmse_l1 = RunningAverage()
    rmse_l2 = RunningAverage()
    delta1_l1 = RunningAverage()
    delta1_l2 = RunningAverage()

    batches_done = 0
    with torch.no_grad():
        for batch in val_loader:
            if args.max_batches > 0 and batches_done >= args.max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            depth_1 = batch["depth_1"].to(device, non_blocking=True)
            depth_2 = batch["depth_2"].to(device, non_blocking=True)
            precomputed_dino = batch.get("dino_features")
            if precomputed_dino is not None:
                precomputed_dino = precomputed_dino.to(device, non_blocking=True)

            amp_ctx = torch.autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            with amp_ctx:
                pred_1 = model(images, target_layer=1, return_intermediate=False, precomputed_dino=precomputed_dino)
                pred_2 = model(images, target_layer=2, return_intermediate=False, precomputed_dino=precomputed_dino)

            l1 = float(criterion(pred_1, depth_1).item())
            l2 = float(criterion(pred_2, depth_2).item())
            silog_l1.update(l1)
            silog_l2.update(l2)

            m1 = compute_depth_metrics(pred_1, depth_1)
            m2 = compute_depth_metrics(pred_2, depth_2)
            abs_rel_l1.update(m1["abs_rel"])
            abs_rel_l2.update(m2["abs_rel"])
            rmse_l1.update(m1["rmse"])
            rmse_l2.update(m2["rmse"])
            delta1_l1.update(m1["delta1"])
            delta1_l2.update(m2["delta1"])

            batches_done += 1

    report = {
        "split": cfg["data"]["val_split"],
        "dataset": cfg["data"]["val_dataset_name"],
        "batches_evaluated": batches_done,
        "silog_layer1": silog_l1.mean,
        "silog_layer2": silog_l2.mean,
        "silog_mean": (silog_l1.mean + silog_l2.mean) * 0.5,
        "abs_rel_layer1": abs_rel_l1.mean,
        "abs_rel_layer2": abs_rel_l2.mean,
        "rmse_layer1": rmse_l1.mean,
        "rmse_layer2": rmse_l2.mean,
        "delta1_layer1": delta1_l1.mean,
        "delta1_layer2": delta1_l2.mean,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[eval-synth] done")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
