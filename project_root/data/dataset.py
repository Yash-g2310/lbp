"""Data pipeline with local staging and optional precomputed DINO feature loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# ==========================================
# 1. TRANSFORMS
# ==========================================
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_depth = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def _resolve_key(sample: Dict[str, Any], candidates: tuple[str, ...], feature_name: str) -> str:
    for key in candidates:
        if key in sample:
            return key
    if "tuples.json" in sample and feature_name in {"depth_1", "depth_2"}:
        raise KeyError(
            "Validation sample contains 'tuples.json' ranking annotations but no dense depth maps. "
            "For supervised validation loss, use a split with dense depth targets (e.g., "
            "val_dataset_name=princeton-vl/LayeredDepth-Syn and val_split=validation)."
        )
    available = ", ".join(sorted(sample.keys()))
    raise KeyError(
        f"Missing '{feature_name}' field. Tried keys {candidates}. Available keys: [{available}]"
    )


class PrecomputedFeatureStore:
    """Loads feature tensors from sharded .pt files with sample_id lookup."""

    def __init__(self, index_path: str, max_cached_shards: int = 1) -> None:
        index_file = Path(index_path)
        if not index_file.exists():
            raise FileNotFoundError(f"Precomputed feature index not found: {index_file}")

        with index_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.samples: Dict[str, Any] = payload["samples"]
        self._shard_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_order: list[str] = []
        self.max_cached_shards = max(0, int(max_cached_shards))

    def _load_shard(self, shard_path: str) -> Dict[str, torch.Tensor]:
        try:
            return torch.load(shard_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Backward compatibility for older torch versions without weights_only.
            try:
                return torch.load(shard_path, map_location="cpu")
            except Exception as exc:
                raise RuntimeError(f"Failed to load precomputed shard '{shard_path}': {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load precomputed shard '{shard_path}': {exc}") from exc

    def get(self, split_name: str, sample_id: str) -> torch.Tensor:
        split_samples = self.samples.get(split_name, self.samples)
        meta = split_samples.get(str(sample_id))
        if meta is None:
            raise KeyError(f"sample_id '{sample_id}' not found in feature index for split '{split_name}'")

        shard_path = meta["shard_path"]
        feature_key = meta["feature_key"]
        if shard_path not in self._shard_cache:
            self._shard_cache[shard_path] = self._load_shard(shard_path)
            self._cache_order.append(shard_path)

            # Bound per-worker shard cache size to avoid host RAM explosions.
            if self.max_cached_shards == 0:
                self._shard_cache.clear()
                self._cache_order.clear()
            else:
                while len(self._cache_order) > self.max_cached_shards:
                    evict = self._cache_order.pop(0)
                    self._shard_cache.pop(evict, None)

        return self._shard_cache[shard_path][feature_key].clone()


def _extract_sample_id(sample: Dict[str, Any], index: int) -> str:
    for key in ("sample_id", "id", "uid", "image_id"):
        if key in sample:
            return str(sample[key])
    return str(index)


class LayeredDepthDataset(Dataset):
    def __init__(self, hf_dataset: Any, feature_store: Optional[PrecomputedFeatureStore], split_name: str) -> None:
        self.ds = hf_dataset
        self.feature_store = feature_store
        self.split_name = split_name

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.ds[index]
        sample_id = _extract_sample_id(sample, index)

        image_key = _resolve_key(sample, ("image.png", "image", "rgb", "rgb_image"), "image")
        depth_1_key = _resolve_key(
            sample,
            ("depth_1.png", "depth_1", "depth1", "layer_1_depth", "depth_layer_1"),
            "depth_1",
        )
        depth_2_key = _resolve_key(
            sample,
            ("depth_2.png", "depth_2", "depth2", "layer_2_depth", "depth_layer_2"),
            "depth_2",
        )

        item = {
            "sample_id": sample_id,
            "image": transform_img(sample[image_key].convert("RGB")),
            "depth_1": transform_depth(sample[depth_1_key]),
            "depth_2": transform_depth(sample[depth_2_key]),
        }
        if self.feature_store is not None:
            item["dino_features"] = self.feature_store.get(self.split_name, sample_id)

        return item

# ==========================================
# 2. BATCH COLLATION
# ==========================================
def collate_fn(batch):
    output = {
        "sample_id": [b["sample_id"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        "depth_1": torch.stack([b["depth_1"] for b in batch]),
        "depth_2": torch.stack([b["depth_2"] for b in batch]),
    }
    if "dino_features" in batch[0]:
        output["dino_features"] = torch.stack([b["dino_features"] for b in batch])
    return output

# ==========================================
# 3. HIGH-PERFORMANCE DATALOADERS
# ==========================================
def get_dataloaders(cfg: Dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    hw_cfg = cfg["hardware"]

    cache_dir = Path(data_cfg["cache_dir"])
    staged_root = Path(data_cfg["staged_root"])

    cache_dir.mkdir(parents=True, exist_ok=True)
    if data_cfg.get("require_local_staging", False) and not staged_root.exists():
        raise FileNotFoundError(f"Required staged_root is missing: {staged_root}")

    feature_store = None
    if data_cfg.get("use_precomputed_dino", False):
        # With multi-worker loading, each worker has its own feature store; keep
        # shard caches small to avoid host-memory OOMs from duplicated caches.
        max_cached_shards = int(data_cfg.get("precomputed_max_cached_shards", 1))
        feature_store = PrecomputedFeatureStore(
            data_cfg["precomputed_index_path"],
            max_cached_shards=max_cached_shards,
        )

    train_hf = load_dataset(
        data_cfg["train_dataset_name"],
        split=data_cfg["train_split"],
        cache_dir=str(cache_dir),
        streaming=False,
    )
    val_hf = load_dataset(
        data_cfg["val_dataset_name"],
        split=data_cfg["val_split"],
        cache_dir=str(cache_dir),
        streaming=False,
    )

    train_dataset = LayeredDepthDataset(train_hf, feature_store, data_cfg["train_split"])
    val_dataset = LayeredDepthDataset(val_hf, feature_store, data_cfg["val_split"])

    num_workers = int(hw_cfg["num_workers"])
    persistent_workers = bool(data_cfg.get("persistent_workers", False) and num_workers > 0)
    loader_common: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(data_cfg.get("pin_memory", True)),
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        loader_common["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 1))

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_cfg["batch_size"]),
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        **loader_common,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(data_cfg.get("val_batch_size", 1)),
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        **loader_common,
    )

    return train_loader, val_loader