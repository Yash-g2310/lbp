#!/usr/bin/env python3
"""Fast FFT-path sanity check for the FourierUnit under expected feature-map sizes."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml
from lbp_project.models.components import FourierUnit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick-check FFT stability against fp32 reference")
    p.add_argument("--config", required=True, help="Config path")
    p.add_argument("--batch-size", type=int, default=2, help="Synthetic batch size")
    p.add_argument("--fail-rel-l2", type=float, default=1.5, help="Fail if relative L2 exceeds this value")
    return p.parse_args()


def rel_l2(ref: torch.Tensor, out: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(ref).item()
    num = torch.linalg.vector_norm(out - ref).item()
    if denom == 0.0:
        return 0.0 if num == 0.0 else float("inf")
    return float(num / denom)


def run_one_shape(
    channels: int,
    h: int,
    w: int,
    batch_size: int,
    fft_pad_size: int,
    device: torch.device,
    fail_rel_l2: float,
) -> None:
    fu_ref = FourierUnit(channels, channels, fft_mode="fp32", fft_pad_size=fft_pad_size).to(device)
    fu_pad = FourierUnit(channels, channels, fft_mode="pad_fp16", fft_pad_size=fft_pad_size).to(device)
    fu_pad.load_state_dict(fu_ref.state_dict())
    fu_ref.eval()
    fu_pad.eval()

    x = torch.randn(batch_size, channels, h, w, device=device, dtype=torch.float32)

    with torch.no_grad():
        y_ref = fu_ref(x).float()
        x_pad = x.half() if device.type == "cuda" else x
        y_pad = fu_pad(x_pad).float()

    if y_ref.shape != y_pad.shape:
        raise RuntimeError(f"Shape mismatch for ({h}x{w}): ref={tuple(y_ref.shape)} pad={tuple(y_pad.shape)}")
    if not torch.isfinite(y_ref).all() or not torch.isfinite(y_pad).all():
        raise RuntimeError(f"Non-finite values detected for ({h}x{w})")

    delta_rel_l2 = rel_l2(y_ref, y_pad)
    delta_max_abs = float((y_ref - y_pad).abs().max().item())

    print(
        f"[OK] shape=({h},{w}) rel_l2={delta_rel_l2:.6f} max_abs={delta_max_abs:.6f} "
        f"dtype_pad={'fp16' if device.type == 'cuda' else 'fp32'}"
    )

    if delta_rel_l2 > fail_rel_l2:
        raise RuntimeError(
            f"FFT quickcheck failed for ({h}x{w}): rel_l2={delta_rel_l2:.6f} > fail_rel_l2={fail_rel_l2}"
        )


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    use_cuda = torch.cuda.is_available() and cfg["hardware"].get("device") == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")

    base_channels = int(cfg["architecture"].get("base_channels", 32))
    channels = max(8, base_channels)
    fft_pad_size = int(cfg["architecture"].get("fft_pad_size", 256))

    # Shapes reflect key downsampling stages seen by SFIN FFT blocks.
    shapes: List[Tuple[int, int]] = [(224, 224), (112, 112), (56, 56)]

    print(
        f"[info] fft quickcheck device={device.type} channels={channels} "
        f"fft_pad_size={fft_pad_size} batch={args.batch_size}"
    )

    for h, w in shapes:
        run_one_shape(
            channels=channels,
            h=h,
            w=w,
            batch_size=args.batch_size,
            fft_pad_size=fft_pad_size,
            device=device,
            fail_rel_l2=float(args.fail_rel_l2),
        )

    print("[done] quickcheck_fft passed")


if __name__ == "__main__":
    main()
