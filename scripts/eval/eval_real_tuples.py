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
from datasets import load_dataset
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml
from lbp_project.data.dataset import PrecomputedFeatureStore, _extract_sample_id
from lbp_project.models.factory import build_depth_model
from lbp_project.utils.checkpoints import load_checkpoint_model
from lbp_project.utils.metrics import evaluate_tuple_sample, merge_tuple_counts, summarize_tuple_counts


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
    p.add_argument(
        "--use-precomputed-dino",
        action="store_true",
        help="Use cached DINO features for real eval (default: disabled)",
    )
    p.add_argument(
        "--precomputed-index-path",
        default="",
        help="Override precomputed feature index path for real eval",
    )
    p.add_argument("--output", required=True, help="Output JSON path")
    return p.parse_args()


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
    data_cfg = cfg.get("data", {})

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
    use_precomputed_dino = bool(eval_cfg.get("real_use_precomputed_dino", False) or args.use_precomputed_dino)
    feature_store = None
    if use_precomputed_dino:
        index_path = (
            args.precomputed_index_path.strip()
            or str(eval_cfg.get("real_precomputed_index_path", data_cfg.get("precomputed_index_path", "")))
        )
        if not index_path:
            raise ValueError("Precomputed real-eval requested but no index path configured")
        feature_store = PrecomputedFeatureStore(
            index_path,
            max_cached_shards=int(data_cfg.get("precomputed_max_cached_shards", 1)),
        )

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_depth_model(cfg, device, use_precomputed_dino=use_precomputed_dino)
    missing, unexpected = load_checkpoint_model(
        model,
        checkpoint_path,
        device=device,
        strict=False,
        allow_missing_prefixes=("encoder.",),
    )
    if missing:
        raise RuntimeError(
            "Checkpoint is missing non-encoder weights required for evaluation: "
            f"{missing[:20]}"
        )
    if unexpected:
        raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected[:20]}")
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

                    precomputed = None
                    if use_precomputed_dino:
                        sample_id = _extract_sample_id(sample, i)
                        precomputed = feature_store.get(split, sample_id).unsqueeze(0).to(device)

                    amp_ctx = torch.autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
                    with amp_ctx:
                        pred = model(
                            img,
                            target_layer=target_layer,
                            return_intermediate=False,
                            precomputed_dino=precomputed,
                        )

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
        "use_precomputed_dino": use_precomputed_dino,
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
