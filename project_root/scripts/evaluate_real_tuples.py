#!/usr/bin/env python3
"""Evaluate tuple-wise benchmark metrics (pairs/trips/quads) on real LayeredDepth."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict
import sys

import torch
import yaml
from datasets import load_dataset
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.wrapper import DINOSFIN_Architecture_NEW
from utils.metrics import evaluate_tuple_sample, merge_tuple_counts, summarize_tuple_counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate real benchmark tuple accuracy")
    p.add_argument("--config", required=True, help="Config path")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path")
    p.add_argument("--split", default="", help="Single split override (validation/test)")
    p.add_argument("--splits", default="", help="Comma-separated splits override, e.g. validation,test")
    p.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0=all)")
    p.add_argument("--layer-key", default="", help="Single tuple layer key override")
    p.add_argument("--layer-keys", default="", help="Comma-separated layer keys, e.g. layer_all,layer_first")
    p.add_argument("--target-layer", type=int, default=-1, help="Model target layer override")
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
        use_precomputed_dino=False,
    ).to(device)
    return model


def load_checkpoint_model(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[model_key])


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    eval_cfg = cfg.get("evaluation", {})

    if args.splits:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    elif args.split:
        splits = [args.split]
    else:
        splits = eval_cfg.get("real_splits", [eval_cfg.get("real_split", "validation")])

    if args.layer_keys:
        layer_keys = [k.strip() for k in args.layer_keys.split(",") if k.strip()]
    elif args.layer_key:
        layer_keys = [args.layer_key]
    else:
        layer_keys = eval_cfg.get("real_layer_keys", [eval_cfg.get("real_layer_key", "layer_all")])

    if not splits:
        raise ValueError("No evaluation splits configured")
    if not layer_keys:
        raise ValueError("No layer keys configured")

    target_layer = args.target_layer if args.target_layer > 0 else int(eval_cfg.get("target_layer", 1))
    max_samples = args.max_samples if args.max_samples > 0 else int(eval_cfg.get("real_max_samples", 0))
    dataset_name = str(eval_cfg.get("real_dataset_name", "princeton-vl/LayeredDepth"))

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")
    amp_enabled = bool(cfg["hardware"].get("amp", True)) and device.type == "cuda"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(cfg, device)
    load_checkpoint_model(model, checkpoint_path, device)
    model.eval()

    tfm = image_transform()

    per_eval: Dict[str, Any] = {}
    warnings: list[str] = []
    aggregate_counts: Dict[str, Dict[str, int]] = {
        "pairs": {"correct": 0, "total": 0},
        "trips": {"correct": 0, "total": 0},
        "quads": {"correct": 0, "total": 0},
        "all": {"correct": 0, "total": 0},
    }

    for split in splits:
        ds = load_dataset(dataset_name, split=split, cache_dir=cfg["data"]["cache_dir"], streaming=False)
        for layer_key in layer_keys:
            totals: Dict[str, Dict[str, int]] = {
                "pairs": {"correct": 0, "total": 0},
                "trips": {"correct": 0, "total": 0},
                "quads": {"correct": 0, "total": 0},
                "all": {"correct": 0, "total": 0},
            }

            processed = 0
            with torch.no_grad():
                for i, sample in enumerate(ds):
                    if max_samples > 0 and i >= max_samples:
                        break

                    img_key = "image.png" if "image.png" in sample else "image"
                    img_pil = sample[img_key].convert("RGB")
                    original_size = img_pil.size
                    img = tfm(img_pil).unsqueeze(0).to(device)

                    amp_ctx = torch.autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
                    with amp_ctx:
                        pred = model(img, target_layer=target_layer, return_intermediate=False)

                    counts = evaluate_tuple_sample(pred, sample, original_size=original_size, layer_key=layer_key)
                    merge_tuple_counts(totals, counts)
                    processed += 1

            key = f"{split}/{layer_key}"
            summary = summarize_tuple_counts(totals)
            per_eval[key] = {
                "split": split,
                "layer_key": layer_key,
                "samples_evaluated": processed,
                **summary,
            }
            if per_eval[key].get("all_total", 0.0) == 0.0:
                warnings.append(
                    f"No tuple annotations available for {key} in accessible data (all_total=0)."
                )
            merge_tuple_counts(aggregate_counts, totals)

    aggregate_summary = summarize_tuple_counts(aggregate_counts)
    report = {
        "dataset": dataset_name,
        "splits": splits,
        "layer_keys": layer_keys,
        "target_layer": target_layer,
        "per_eval": per_eval,
        "aggregate": aggregate_summary,
        "official_quadruplet_accuracy": aggregate_summary.get("quads_acc", 0.0),
        "warnings": warnings,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[eval-tuples] done")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
