"""Offline precomputation for frozen DINOv2 features into sharded .pt files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset
import torchvision.transforms as transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute DINO features into sharded tensors")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store shards and index")
    parser.add_argument("--shard_size", type=int, default=2048, help="Samples per shard")
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def image_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_sample_id(sample: Dict[str, Any], index: int) -> str:
    for key in ("sample_id", "id", "uid", "image_id"):
        if key in sample:
            return str(sample[key])
    return str(index)


def run_split(
    encoder: torch.nn.Module,
    dataset_name: str,
    split: str,
    cache_dir: str,
    output_dir: Path,
    shard_size: int,
    device: torch.device,
) -> Dict[str, Dict[str, str]]:
    tfm = image_transform()
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir, streaming=False)

    samples_index: Dict[str, Dict[str, str]] = {}
    shard_data: Dict[str, torch.Tensor] = {}
    shard_id = 0

    for i, sample in enumerate(ds):
        image_key = "image.png" if "image.png" in sample else "image"
        image = tfm(sample[image_key].convert("RGB")).unsqueeze(0).to(device)
        sample_id = get_sample_id(sample, i)

        with torch.no_grad():
            feats = encoder.forward_features(image)["x_norm_patchtokens"]
            b, n, c = feats.shape
            hw = int(n ** 0.5)
            dino = feats.permute(0, 2, 1).view(b, c, hw, hw).squeeze(0).cpu().contiguous()

        key = sample_id
        shard_data[key] = dino

        if len(shard_data) >= shard_size:
            shard_path = output_dir / f"{split}_shard_{shard_id:05d}.pt"
            torch.save(shard_data, shard_path)
            for sample_key in shard_data:
                samples_index[sample_key] = {
                    "shard_path": str(shard_path),
                    "feature_key": sample_key,
                    "split": split,
                }
            shard_data = {}
            shard_id += 1

    if shard_data:
        shard_path = output_dir / f"{split}_shard_{shard_id:05d}.pt"
        torch.save(shard_data, shard_path)
        for sample_key in shard_data:
            samples_index[sample_key] = {
                "shard_path": str(shard_path),
                "feature_key": sample_key,
                "split": split,
            }

    return samples_index


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["hardware"]["device"] == "cuda" else "cpu")
    encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    encoder.eval()

    full_index: Dict[str, Dict[str, Dict[str, str]]] = {"samples": {}}
    for split_cfg in (
        (cfg["data"]["train_dataset_name"], cfg["data"]["train_split"]),
        (cfg["data"]["val_dataset_name"], cfg["data"]["val_split"]),
    ):
        split_name = split_cfg[1]
        part_index = run_split(
            encoder=encoder,
            dataset_name=split_cfg[0],
            split=split_name,
            cache_dir=cfg["data"]["cache_dir"],
            output_dir=out_dir,
            shard_size=args.shard_size,
            device=device,
        )
        full_index["samples"][split_name] = part_index

    index_path = out_dir / "index.json"
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(full_index, f)


if __name__ == "__main__":
    main()
