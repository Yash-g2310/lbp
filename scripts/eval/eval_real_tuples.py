#!/usr/bin/env python3
"""Evaluate tuple-wise benchmark metrics (pairs/trips/quads) on real LayeredDepth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import sys

import torch
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml
from lbp_project.data.dataset import PrecomputedFeatureStore, _extract_sample_id
from lbp_project.data.hf_loading import load_dataset_split_with_policy
from lbp_project.models.factory import build_depth_model
from lbp_project.utils.checkpoints import load_checkpoint_model
from lbp_project.utils.eval_inference import predict_depth_for_eval, resolve_flow_eval_settings
from lbp_project.utils.eval_layers import resolve_required_layers
from lbp_project.utils.metrics import (
    evaluate_tuple_sample,
    evaluate_tuple_sample_multi_layer,
    merge_tuple_counts,
    summarize_tuple_counts,
)
from lbp_project.utils.run_manifest import build_run_manifest


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
        "--disable-multi-pass",
        action="store_true",
        help="Disable tuple-driven multi-pass layer inference and force single-pass target layer mode",
    )
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
    flow_eval = resolve_flow_eval_settings(cfg)

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
    multi_pass_eval = bool(eval_cfg.get("multi_pass_real_eval", True)) and not args.disable_multi_pass
    max_samples = args.max_samples if args.max_samples > 0 else int(eval_cfg.get("real_max_samples", 0))
    dataset_name = str(eval_cfg.get("real_dataset_name", "princeton-vl/LayeredDepth"))
    allow_hf_downloads = bool(data_cfg.get("allow_hf_downloads", True))
    allow_partial_local_shards = bool(data_cfg.get("allow_partial_local_shards", False))
    partial_local_min_shards = int(data_cfg.get("partial_local_shards_min_per_split", 1))
    repair_hf_cache_once = bool(
        data_cfg.get(
            "repair_hf_cache_once",
            allow_hf_downloads and not allow_partial_local_shards,
        )
    )

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

    model_cfg = dict(cfg)
    model_cfg["architecture"] = dict(cfg.get("architecture", {}))
    if bool(flow_eval["flow_mode"]):
        model_cfg["architecture"]["enable_velocity_head"] = True

    model = build_depth_model(model_cfg, device, use_precomputed_dino=use_precomputed_dino)
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
        ds = load_dataset_split_with_policy(
            dataset_name,
            split,
            cache_dir=str(cfg["data"]["cache_dir"]),
            allow_downloads=allow_hf_downloads,
            allow_cache_repair=repair_hf_cache_once,
            allow_partial_local_shards=allow_partial_local_shards,
            partial_local_min_shards=partial_local_min_shards,
            log_prefix="[eval-tuples][dataset]",
        )
        for layer_key in layer_keys:
            totals: Dict[str, Dict[str, int]] = {
                "pairs": {"correct": 0, "total": 0},
                "trips": {"correct": 0, "total": 0},
                "quads": {"correct": 0, "total": 0},
                "all": {"correct": 0, "total": 0},
            }
            missing_layer_tuples_total = 0

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
                        if feature_store is None:
                            raise RuntimeError("Precomputed feature store was not initialized")
                        sample_id = _extract_sample_id(sample, i)
                        precomputed = feature_store.get(split, sample_id).unsqueeze(0).to(device)

                    if multi_pass_eval:
                        required_layers = resolve_required_layers(
                            sample,
                            layer_key=layer_key,
                            target_layer_override=None,
                        )
                        if not required_layers:
                            processed += 1
                            continue

                        pred_by_layer: Dict[int, torch.Tensor] = {}
                        for layer_id in required_layers:
                            pred_by_layer[int(layer_id)] = predict_depth_for_eval(
                                model,
                                img,
                                target_layer=int(layer_id),
                                precomputed_dino=precomputed,
                                device=device,
                                amp_enabled=amp_enabled,
                                use_flow_inference=bool(flow_eval["use_flow_inference"]),
                                flow_steps=int(flow_eval["flow_steps"]),
                                flow_t_low=float(flow_eval["flow_t_low"]),
                                flow_t_high=float(flow_eval["flow_t_high"]),
                                flow_init=str(flow_eval["flow_init"]),
                                depth_clip_min=float(flow_eval["depth_clip_min"]),
                                depth_clip_max=float(flow_eval["depth_clip_max"]),
                            )

                        counts = evaluate_tuple_sample_multi_layer(
                            pred_by_layer,
                            sample,
                            original_size=original_size,
                            layer_key=layer_key,
                        )
                        missing_layer_tuples_total += int(
                            counts.get("missing_layer_tuples", {}).get("total", 0)
                        )
                    else:
                        pred = predict_depth_for_eval(
                            model,
                            img,
                            target_layer=target_layer,
                            precomputed_dino=precomputed,
                            device=device,
                            amp_enabled=amp_enabled,
                            use_flow_inference=bool(flow_eval["use_flow_inference"]),
                            flow_steps=int(flow_eval["flow_steps"]),
                            flow_t_low=float(flow_eval["flow_t_low"]),
                            flow_t_high=float(flow_eval["flow_t_high"]),
                            flow_init=str(flow_eval["flow_init"]),
                            depth_clip_min=float(flow_eval["depth_clip_min"]),
                            depth_clip_max=float(flow_eval["depth_clip_max"]),
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
                "missing_layer_tuples": float(missing_layer_tuples_total),
                **summary,
            }
            if per_eval[key].get("all_total", 0.0) == 0.0:
                warnings.append(
                    f"No tuple annotations available for {key} in accessible data (all_total=0)."
                )
            merge_tuple_counts(aggregate_counts, totals)

    aggregate_summary = summarize_tuple_counts(aggregate_counts)
    report = {
        "metadata": build_run_manifest(
            cfg,
            config_path=args.config,
            stage_mode=str(cfg.get("experiment", {}).get("stage_mode", "unknown")),
            project_root=PROJECT_ROOT,
            extra={
                "script": "eval_real_tuples",
                "checkpoint": str(checkpoint_path),
                "splits": splits,
                "layer_keys": layer_keys,
                "max_samples": int(max_samples),
            },
        ),
        "dataset": dataset_name,
        "splits": splits,
        "layer_keys": layer_keys,
        "target_layer": target_layer,
        "multi_pass_real_eval": multi_pass_eval,
        "flow_mode": bool(flow_eval["flow_mode"]),
        "flow_inference_enabled": bool(flow_eval["use_flow_inference"]),
        "flow_inference_steps": int(flow_eval["flow_steps"]),
        "flow_inference_init": str(flow_eval["flow_init"]),
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
