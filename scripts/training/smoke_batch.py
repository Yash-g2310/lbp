from __future__ import annotations

import argparse
from contextlib import nullcontext
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml
from lbp_project.data.dataset import get_dataloaders
from lbp_project.data.preflight import build_download_matrix, enforce_startup_preflight, format_download_matrix
from lbp_project.models.factory import build_depth_model
from lbp_project.utils.losses import SILogLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-batch numeric smoke test")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--max-batch-size", type=int, default=1, help="Clamp smoke-test batch size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    print(format_download_matrix(build_download_matrix(cfg), prefix="[smoke-startup]"), flush=True)
    warnings = enforce_startup_preflight(
        cfg,
        strict_server_policy=bool(cfg.get("data", {}).get("require_local_staging", False)),
    )
    for warning in warnings:
        print(f"[smoke-startup][warn] {warning}", flush=True)

    # Smoke test uses very conservative memory settings by default.
    cfg["data"]["batch_size"] = min(int(cfg["data"]["batch_size"]), int(args.max_batch_size))
    cfg["data"]["val_batch_size"] = min(int(cfg["data"].get("val_batch_size", 1)), int(args.max_batch_size))
    cfg["hardware"]["num_workers"] = 0

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")
    amp_enabled = bool(cfg["hardware"].get("amp", True)) and device.type == "cuda"
    amp_dtype_name = str(cfg["hardware"].get("amp_dtype", "float16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in {"bf16", "bfloat16"} else torch.float16

    model = build_depth_model(cfg, device)
    model.train()

    criterion = SILogLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    train_loader, _ = get_dataloaders(cfg)
    batch = next(iter(train_loader))

    images = batch["image"].to(device, non_blocking=True)
    depth_1 = batch["depth_1"].to(device, non_blocking=True).float()
    depth_2 = batch["depth_2"].to(device, non_blocking=True).float()
    precomputed_dino = batch.get("dino_features")
    if precomputed_dino is not None:
        precomputed_dino = precomputed_dino.to(device, non_blocking=True)

    amp_ctx = (
        torch.autocast(device_type="cuda", enabled=amp_enabled, dtype=amp_dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with amp_ctx:
        out1 = model(images, target_layer=1, return_intermediate=True, use_checkpointing=bool(cfg["architecture"].get("use_gradient_checkpointing", False)), precomputed_dino=precomputed_dino)
        out2 = model(images, target_layer=2, return_intermediate=True, use_checkpointing=bool(cfg["architecture"].get("use_gradient_checkpointing", False)), precomputed_dino=precomputed_dino)

        l1_b = criterion(out1["bottleneck"], depth_1)
        l1_d = criterion(out1["decoder"], depth_1)
        l1_f = criterion(out1["final"], depth_1)
        l2_b = criterion(out2["bottleneck"], depth_2)
        l2_d = criterion(out2["decoder"], depth_2)
        l2_f = criterion(out2["final"], depth_2)
        total = l1_b + l1_d + l1_f + l2_b + l2_d + l2_f

    components = {
        "l1_b": float(l1_b.detach().item()),
        "l1_d": float(l1_d.detach().item()),
        "l1_f": float(l1_f.detach().item()),
        "l2_b": float(l2_b.detach().item()),
        "l2_d": float(l2_d.detach().item()),
        "l2_f": float(l2_f.detach().item()),
        "total": float(total.detach().item()),
    }

    for key, value in components.items():
        if not torch.isfinite(torch.tensor(value)):
            raise RuntimeError(f"Non-finite component detected: {key}={value}")
        if value <= 1e-10:
            raise RuntimeError(f"Near-zero component detected: {key}={value}")

    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["training"].get("grad_clip_norm", 1.0)))
    if not torch.isfinite(grad_norm):
        raise RuntimeError(f"Non-finite grad norm in smoke test: {float(grad_norm)}")

    mem_msg = "cpu"
    if device.type == "cuda":
        allocated_gb = torch.cuda.memory_allocated(device) / (1024 ** 3)
        reserved_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
        mem_msg = f"cuda_allocated_gb={allocated_gb:.2f} cuda_reserved_gb={reserved_gb:.2f}"

    print("[smoke] PASS one-batch numeric test")
    print(f"[smoke] components={components}")
    print(f"[smoke] grad_norm={float(grad_norm):.6f} {mem_msg}")


if __name__ == "__main__":
    main()
