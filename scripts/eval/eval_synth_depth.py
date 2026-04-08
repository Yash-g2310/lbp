#!/usr/bin/env python3
"""Evaluate trained model on synthetic validation split with dense depth metrics."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml
from lbp_project.data.dataset import get_dataloaders
from lbp_project.models.factory import build_depth_model
from lbp_project.utils.checkpoints import load_checkpoint_model
from lbp_project.utils.losses import SILogLoss
from lbp_project.utils.metrics import RunningAverage, compute_depth_metrics
from lbp_project.utils.run_manifest import build_run_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate synthetic dense depth metrics")
    p.add_argument("--config", required=True, help="Config path")
    p.add_argument("--checkpoint", required=True, help="Checkpoint path")
    p.add_argument("--max-batches", type=int, default=0, help="Limit validation batches (0=all)")
    p.add_argument("--output", required=True, help="Output JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")
    amp_enabled = bool(cfg["hardware"].get("amp", True)) and device.type == "cuda"

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    _, val_loader = get_dataloaders(cfg)
    model = build_depth_model(cfg, device)
    missing, unexpected = load_checkpoint_model(model, checkpoint_path, device=device, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/state mismatch for synthetic eval. missing={missing[:10]} unexpected={unexpected[:10]}"
        )
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

    silog_all = RunningAverage()
    abs_rel_all = RunningAverage()
    rmse_all = RunningAverage()
    delta1_all = RunningAverage()

    per_layer: dict[int, dict[str, RunningAverage]] = {}

    def layer_meter(layer_id: int) -> dict[str, RunningAverage]:
        if layer_id not in per_layer:
            per_layer[layer_id] = {
                "silog": RunningAverage(),
                "abs_rel": RunningAverage(),
                "rmse": RunningAverage(),
                "delta1": RunningAverage(),
            }
        return per_layer[layer_id]

    batches_done = 0
    with torch.no_grad():
        for batch in val_loader:
            if args.max_batches > 0 and batches_done >= args.max_batches:
                break

            images = batch["image"].to(device, non_blocking=True)
            depth_1 = batch["depth_1"].to(device, non_blocking=True)
            depth_2 = batch["depth_2"].to(device, non_blocking=True)
            depth_targets = batch.get("depth_targets")
            depth_layer_mask = batch.get("depth_layer_mask")
            if depth_targets is not None:
                depth_targets = depth_targets.to(device, non_blocking=True)
            if depth_layer_mask is not None:
                depth_layer_mask = depth_layer_mask.to(device, non_blocking=True)
            precomputed_dino = batch.get("dino_features")
            if precomputed_dino is not None:
                precomputed_dino = precomputed_dino.to(device, non_blocking=True)

            if depth_targets is not None and depth_layer_mask is not None:
                num_layers = depth_targets.shape[1]
                for layer_idx in range(num_layers):
                    sample_mask = depth_layer_mask[:, layer_idx].to(dtype=torch.bool)
                    if not bool(sample_mask.any()):
                        continue

                    sample_idx = torch.nonzero(sample_mask, as_tuple=False).squeeze(1)
                    images_sub = images.index_select(0, sample_idx)
                    target_sub = depth_targets.index_select(0, sample_idx)[:, layer_idx]
                    precomputed_sub = (
                        precomputed_dino.index_select(0, sample_idx)
                        if precomputed_dino is not None
                        else None
                    )
                    target_layer = torch.full(
                        (images_sub.shape[0],),
                        layer_idx + 1,
                        device=device,
                        dtype=torch.long,
                    )

                    amp_ctx = (
                        torch.autocast(device_type="cuda", enabled=amp_enabled)
                        if device.type == "cuda"
                        else nullcontext()
                    )
                    with amp_ctx:
                        pred = model(
                            images_sub,
                            target_layer=target_layer,
                            return_intermediate=False,
                            precomputed_dino=precomputed_sub,
                        )

                    layer_id = layer_idx + 1
                    meter = layer_meter(layer_id)
                    silog_val = float(criterion(pred, target_sub).item())
                    metric_vals = compute_depth_metrics(pred, target_sub)
                    meter["silog"].update(silog_val)
                    meter["abs_rel"].update(metric_vals["abs_rel"])
                    meter["rmse"].update(metric_vals["rmse"])
                    meter["delta1"].update(metric_vals["delta1"])

                    silog_all.update(silog_val)
                    abs_rel_all.update(metric_vals["abs_rel"])
                    rmse_all.update(metric_vals["rmse"])
                    delta1_all.update(metric_vals["delta1"])

                    if layer_id == 1:
                        silog_l1.update(silog_val)
                        abs_rel_l1.update(metric_vals["abs_rel"])
                        rmse_l1.update(metric_vals["rmse"])
                        delta1_l1.update(metric_vals["delta1"])
                    if layer_id == 2:
                        silog_l2.update(silog_val)
                        abs_rel_l2.update(metric_vals["abs_rel"])
                        rmse_l2.update(metric_vals["rmse"])
                        delta1_l2.update(metric_vals["delta1"])
            else:
                amp_ctx = (
                    torch.autocast(device_type="cuda", enabled=amp_enabled)
                    if device.type == "cuda"
                    else nullcontext()
                )
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

                silog_all.update((l1 + l2) * 0.5)
                abs_rel_all.update((m1["abs_rel"] + m2["abs_rel"]) * 0.5)
                rmse_all.update((m1["rmse"] + m2["rmse"]) * 0.5)
                delta1_all.update((m1["delta1"] + m2["delta1"]) * 0.5)

            batches_done += 1

    report = {
        "metadata": build_run_manifest(
            cfg,
            config_path=args.config,
            stage_mode=str(cfg.get("experiment", {}).get("stage_mode", "unknown")),
            project_root=PROJECT_ROOT,
            extra={
                "script": "eval_synth_depth",
                "checkpoint": str(checkpoint_path),
                "max_batches": int(args.max_batches),
            },
        ),
        "split": cfg["data"]["val_split"],
        "dataset": cfg["data"]["val_dataset_name"],
        "batches_evaluated": batches_done,
        "silog_layer1": silog_l1.mean,
        "silog_layer2": silog_l2.mean,
        "silog_mean": silog_all.mean,
        "abs_rel_layer1": abs_rel_l1.mean,
        "abs_rel_layer2": abs_rel_l2.mean,
        "abs_rel_mean": abs_rel_all.mean,
        "rmse_layer1": rmse_l1.mean,
        "rmse_layer2": rmse_l2.mean,
        "rmse_mean": rmse_all.mean,
        "delta1_layer1": delta1_l1.mean,
        "delta1_layer2": delta1_l2.mean,
        "delta1_mean": delta1_all.mean,
        "layer_metrics": {
            str(layer_id): {
                "silog": meters["silog"].mean,
                "abs_rel": meters["abs_rel"].mean,
                "rmse": meters["rmse"].mean,
                "delta1": meters["delta1"].mean,
            }
            for layer_id, meters in sorted(per_layer.items())
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[eval-synth] done")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
