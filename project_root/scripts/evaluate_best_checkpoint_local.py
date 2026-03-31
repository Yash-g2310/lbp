#!/usr/bin/env python3
"""Evaluate a checkpoint on locally cached real LayeredDepth shards.

This script is local-cache-first and does not require re-downloading missing shards.
It evaluates tuple metrics (pairs/trips/quads/all) and reports quadruplet accuracy
for validation/test based on whichever local arrow shards are present.
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path
import re
from typing import Any, Dict, List
import sys

import torch
import yaml
from datasets import Dataset, concatenate_datasets
import torchvision.transforms as transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.wrapper import DINOSFIN_Architecture_NEW
from utils.metrics import evaluate_tuple_sample, merge_tuple_counts, summarize_tuple_counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate checkpoint on local validation/test shards")
    p.add_argument("--config", required=True, help="Path to project YAML config")
    p.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint path (default: training.checkpoint.dir + evaluation.checkpoint_name)",
    )
    p.add_argument(
        "--cache-root",
        default="",
        help=(
            "Path to local HF datasets cache root for real dataset. "
            "Default: ~/.cache/huggingface/datasets/princeton-vl___layered_depth"
        ),
    )
    p.add_argument(
        "--splits",
        default="validation,test",
        help="Comma-separated splits to evaluate from local shards",
    )
    p.add_argument(
        "--layer-keys",
        default="",
        help="Comma-separated tuple layer keys (default from config evaluation.real_layer_keys)",
    )
    p.add_argument(
        "--target-layer",
        type=int,
        default=-1,
        help="Model target layer override (default from config)",
    )
    p.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help="Limit samples per split (0 means all locally available samples)",
    )
    p.add_argument(
        "--output",
        default="",
        help="Output JSON path (default: evaluation.report_dir/local_best_ckpt_eval.json)",
    )
    return p.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    return cfg


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_key = "model_state" if "model_state" in checkpoint else "model_state_dict"
    if model_key not in checkpoint:
        raise KeyError(f"Checkpoint missing '{model_key}'")
    state = checkpoint[model_key]
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state dict has unexpected type")
    return state


def infer_arch_overrides_from_state(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    overrides: Dict[str, int] = {}

    input_conv_key = None
    for candidate in ("decoder.input_conv.weight", "decoder.input_proj.weight"):
        if candidate in state:
            input_conv_key = candidate
            break
    if input_conv_key is not None and hasattr(state[input_conv_key], "shape"):
        w = state[input_conv_key]
        if len(w.shape) >= 2:
            base_channels = int(w.shape[0])
            fused_in_channels = int(w.shape[1])
            dino_embed_dim = fused_in_channels - 4
            if dino_embed_dim > 0:
                overrides["base_channels"] = base_channels
                overrides["dino_embed_dim"] = dino_embed_dim

    sfin_re = re.compile(r"^decoder\.encoder1\.blocks\.(\d+)\.")
    rhag_re = re.compile(r"^decoder\.bottleneck\.blocks\.(\d+)\.")
    sfin_idxs: set[int] = set()
    rhag_idxs: set[int] = set()
    for k in state.keys():
        m = sfin_re.match(k)
        if m:
            sfin_idxs.add(int(m.group(1)))
        m = rhag_re.match(k)
        if m:
            rhag_idxs.add(int(m.group(1)))

    if sfin_idxs:
        overrides["num_sfin"] = max(sfin_idxs) + 1
    if rhag_idxs:
        overrides["num_rhag"] = max(rhag_idxs) + 1

    return overrides


def _resolve_existing_path(path_str: str) -> Path | None:
    p = Path(path_str)
    candidates = [p]
    if not p.is_absolute():
        candidates.append(PROJECT_ROOT / p)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def resolve_checkpoint(cfg: Dict[str, Any], checkpoint_arg: str) -> Path:
    ckpt_cfg = cfg.get("training", {}).get("checkpoint", {})
    eval_cfg = cfg.get("evaluation", {})

    candidate_strings: List[str] = []

    if checkpoint_arg.strip():
        candidate_strings.append(checkpoint_arg.strip())
    else:
        ckpt_dir_cfg = str(ckpt_cfg.get("dir", "./checkpoints"))
        ckpt_dir = Path(ckpt_dir_cfg)
        if not ckpt_dir.is_absolute():
            ckpt_dir = PROJECT_ROOT / ckpt_dir

        configured_name = str(eval_cfg.get("checkpoint_name", ckpt_cfg.get("best_name", "best_checkpoint.pth")))
        latest_name = str(ckpt_cfg.get("latest_name", "latest_checkpoint.pth"))

        candidate_strings.extend(
            [
                str(ckpt_dir / configured_name),
                str(ckpt_dir / ckpt_cfg.get("best_name", "best_checkpoint.pth")),
                str(ckpt_dir / latest_name),
            ]
        )

    # Common alternate names/locations used across this project.
    candidate_strings.extend(
        [
            "/checkpoints/best-checkpoint.pth",
            "/checkpoints/best_checkpoint.pth",
            str(PROJECT_ROOT / "checkpoints" / "best_checkpoint.pth"),
            str(PROJECT_ROOT / "checkpoints" / "best-checkpoint.pth"),
            str(PROJECT_ROOT / "checkpoints" / "latest_checkpoint.pth"),
        ]
    )

    seen: set[str] = set()
    unique_candidates: List[str] = []
    for c in candidate_strings:
        if c and c not in seen:
            unique_candidates.append(c)
            seen.add(c)

    for c in unique_candidates:
        resolved = _resolve_existing_path(c)
        if resolved is not None:
            return resolved

    looked = "\n".join(f"  - {c}" for c in unique_candidates)
    raise FileNotFoundError(
        "Checkpoint not found. Tried these paths:\n"
        f"{looked}\n"
        "Tip: pass --checkpoint with the exact .pth path."
    )


def build_model(
    cfg: Dict[str, Any],
    device: torch.device,
    arch_overrides: Dict[str, int] | None = None,
) -> torch.nn.Module:
    arch = cfg["architecture"]
    data_cfg = cfg.get("data", {})
    arch_overrides = arch_overrides or {}

    base_channels = int(arch_overrides.get("base_channels", arch["base_channels"]))
    num_sfin = int(arch_overrides.get("num_sfin", arch["num_sfin"]))
    num_rhag = int(arch_overrides.get("num_rhag", arch["num_rhag"]))
    dino_embed_dim = int(arch_overrides.get("dino_embed_dim", arch["dino_embed_dim"]))

    model = DINOSFIN_Architecture_NEW(
        strategy=arch["strategy"],
        base_channels=base_channels,
        num_sfin=num_sfin,
        num_rhag=num_rhag,
        window_size=int(arch["window_size"]),
        dino_embed_dim=dino_embed_dim,
        fft_mode=arch["fft_mode"],
        fft_pad_size=int(arch["fft_pad_size"]),
        use_precomputed_dino=bool(data_cfg.get("use_precomputed_dino", False)),
    ).to(device)
    return model


def load_checkpoint_model(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint/model shape mismatch. Use the config that matches training architecture, "
            "or let this script auto-infer architecture from the checkpoint. "
            f"Original error: {exc}"
        ) from exc

    missing_non_encoder = [k for k in missing if not k.startswith("encoder.")]
    if missing_non_encoder:
        raise RuntimeError(
            "Checkpoint is missing non-encoder weights required for evaluation: "
            f"{missing_non_encoder[:20]}"
        )
    if unexpected:
        raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected[:20]}")

    if missing:
        print(
            f"[eval-local] warning: missing {len(missing)} encoder keys in checkpoint; "
            "using freshly loaded frozen DINO encoder"
        )


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def find_local_arrow_shards(cache_root: Path, split: str) -> List[Path]:
    pattern = f"layered_depth-{split}-*.arrow"
    shards = sorted(cache_root.rglob(pattern))
    return [p for p in shards if p.is_file()]


def load_local_dataset_from_shards(shards: List[Path]) -> Dataset:
    if not shards:
        raise FileNotFoundError("No local shards provided")
    datasets = [Dataset.from_file(str(shard)) for shard in shards]
    if len(datasets) == 1:
        return datasets[0]
    return concatenate_datasets(datasets)


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    eval_cfg = cfg.get("evaluation", {})

    print("[eval-local] starting local checkpoint evaluation")
    print(f"[eval-local] config={args.config}")

    checkpoint_path = resolve_checkpoint(cfg, args.checkpoint)
    print(f"[eval-local] checkpoint={checkpoint_path}")

    force_fp32_impl = bool(cfg.get("hardware", {}).get("force_fp32_impl", False))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")
    amp_enabled = (not force_fp32_impl) and bool(cfg["hardware"].get("amp", True)) and device.type == "cuda"
    print(
        f"[eval-local] device={device} amp_enabled={amp_enabled} "
        f"force_fp32_impl={force_fp32_impl}"
    )

    checkpoint_state = load_checkpoint_state(checkpoint_path, device)
    arch_overrides = infer_arch_overrides_from_state(checkpoint_state)
    if arch_overrides:
        print(f"[eval-local] inferred architecture overrides from checkpoint: {arch_overrides}")

    model = build_model(cfg, device, arch_overrides=arch_overrides)
    load_checkpoint_model(model, checkpoint_state)
    model.eval()

    splits = parse_csv_list(args.splits) if args.splits.strip() else ["validation", "test"]
    if args.layer_keys.strip():
        layer_keys = parse_csv_list(args.layer_keys)
    else:
        layer_keys = eval_cfg.get("real_layer_keys", [eval_cfg.get("real_layer_key", "layer_all")])

    target_layer = args.target_layer if args.target_layer > 0 else int(eval_cfg.get("target_layer", 1))

    default_cache_root = Path.home() / ".cache" / "huggingface" / "datasets" / "princeton-vl___layered_depth"
    cache_root = Path(args.cache_root.strip()) if args.cache_root.strip() else default_cache_root
    print(f"[eval-local] cache_root={cache_root}")
    if not cache_root.exists():
        print("[eval-local] warning: cache_root does not exist; split scans may be empty")

    max_samples = int(args.max_samples_per_split)
    tfm = image_transform()

    report: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "cache_root": str(cache_root),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "target_layer": target_layer,
        "layer_keys": [str(x) for x in layer_keys],
        "splits": {},
        "aggregate": {},
        "official_quadruplet_accuracy": 0.0,
        "notes": [],
    }

    aggregate_counts: Dict[str, Dict[str, int]] = {
        "pairs": {"correct": 0, "total": 0},
        "trips": {"correct": 0, "total": 0},
        "quads": {"correct": 0, "total": 0},
        "all": {"correct": 0, "total": 0},
    }

    with torch.no_grad():
        for split in splits:
            shards = find_local_arrow_shards(cache_root, split)
            split_key = str(split)
            print(f"[eval-local] split={split_key} local_shards={len(shards)}")
            report["splits"][split_key] = {
                "available_shards": len(shards),
                "shard_paths": [str(p) for p in shards],
                "per_layer_key": {},
            }

            if not shards:
                report["notes"].append(f"No local shards found for split={split}")
                continue

            ds = load_local_dataset_from_shards(shards)
            if max_samples > 0:
                ds = ds.select(range(min(max_samples, len(ds))))
            print(f"[eval-local] split={split_key} samples_to_eval={len(ds)}")

            for layer_key in layer_keys:
                print(f"[eval-local] split={split_key} layer_key={layer_key} evaluating...")
                counts: Dict[str, Dict[str, int]] = {
                    "pairs": {"correct": 0, "total": 0},
                    "trips": {"correct": 0, "total": 0},
                    "quads": {"correct": 0, "total": 0},
                    "all": {"correct": 0, "total": 0},
                }

                processed = 0
                for sample in ds:
                    img_key = "image.png" if "image.png" in sample else "image"
                    img_pil = sample[img_key].convert("RGB")
                    original_size = img_pil.size
                    img = tfm(img_pil).unsqueeze(0).to(device)

                    amp_ctx = torch.autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
                    with amp_ctx:
                        pred = model(img, target_layer=target_layer, return_intermediate=False)

                    sample_counts = evaluate_tuple_sample(pred, sample, original_size=original_size, layer_key=str(layer_key))
                    merge_tuple_counts(counts, sample_counts)
                    processed += 1

                split_summary = summarize_tuple_counts(counts)
                report["splits"][split_key]["per_layer_key"][str(layer_key)] = {
                    "samples_evaluated": processed,
                    **split_summary,
                    "quadruplet_accuracy": split_summary.get("quads_acc", 0.0),
                }
                print(
                    f"[eval-local] split={split_key} layer_key={layer_key} "
                    f"quads_acc={split_summary.get('quads_acc', 0.0):.4f} "
                    f"all_acc={split_summary.get('all_acc', 0.0):.4f}"
                )

                merge_tuple_counts(aggregate_counts, counts)

    agg_summary = summarize_tuple_counts(aggregate_counts)
    report["aggregate"] = agg_summary
    report["official_quadruplet_accuracy"] = agg_summary.get("quads_acc", 0.0)

    out_path = (
        Path(args.output.strip())
        if args.output.strip()
        else Path(str(eval_cfg.get("report_dir", "./artifacts/reports"))) / "local_best_ckpt_eval.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[eval-local] done")
    print(json.dumps(report, indent=2))
    print(f"[eval-local] report: {out_path}")


if __name__ == "__main__":
    main()
