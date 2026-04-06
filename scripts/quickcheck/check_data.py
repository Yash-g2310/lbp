#!/usr/bin/env python3
"""Fast data-path sanity check for one train and one val batch."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml
from lbp_project.data.dataset import get_dataloaders


REQUIRED_BATCH_KEYS = ("sample_id", "image", "depth_1", "depth_2")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick-check dataloaders")
    p.add_argument("--config", required=True, help="Config path")
    return p.parse_args()


def inspect_batch(name: str, batch: Dict[str, Any]) -> None:
    for key in REQUIRED_BATCH_KEYS:
        if key not in batch:
            raise KeyError(f"{name} batch missing key '{key}'")

    image = batch["image"]
    d1 = batch["depth_1"]
    d2 = batch["depth_2"]
    if image.ndim != 4 or d1.ndim != 4 or d2.ndim != 4:
        raise ValueError(f"{name} batch tensor rank mismatch: image={image.shape}, d1={d1.shape}, d2={d2.shape}")

    if image.shape[-2:] != (224, 224):
        raise ValueError(f"{name} image shape mismatch: expected 224x224, got {image.shape[-2:]}")

    if d1.shape[-2:] != (224, 224) or d2.shape[-2:] != (224, 224):
        raise ValueError(f"{name} depth shape mismatch: d1={d1.shape[-2:]}, d2={d2.shape[-2:]}")

    print(
        f"[OK] {name}: image={tuple(image.shape)} depth_1={tuple(d1.shape)} depth_2={tuple(d2.shape)} "
        f"dtype={image.dtype}"
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    print(f"[info] loading dataloaders for {Path(args.config)}")
    train_loader, val_loader = get_dataloaders(cfg)

    train_batch = next(iter(train_loader))
    inspect_batch("train", train_batch)

    val_batch = next(iter(val_loader))
    inspect_batch("val", val_batch)

    use_precomputed = bool(cfg["data"].get("use_precomputed_dino", False))
    if use_precomputed:
        if "dino_features" not in train_batch:
            raise KeyError("use_precomputed_dino=true but dino_features missing from train batch")
        if train_batch["dino_features"].ndim != 4:
            raise ValueError(f"dino_features expected 4D tensor, got {train_batch['dino_features'].shape}")
        print(f"[OK] precomputed dino: {tuple(train_batch['dino_features'].shape)}")

    print("[done] quickcheck_data passed")


if __name__ == "__main__":
    main()
