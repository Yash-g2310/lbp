"""Data pipeline with local staging and optional precomputed DINO feature loading."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from lbp_project.data.hf_loading import load_dataset_split_with_policy

# ==========================================
# 1. TRANSFORMS
# ==========================================
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_depth = transforms.Compose([
    # Use nearest for depth to avoid blending valid and invalid pixels.
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST, antialias=False),
    transforms.ToTensor()
])


def convert_depth_tensor(depth_image: Any, depth_scale: float, depth_clip_min: float, depth_clip_max: float) -> torch.Tensor:
    """Convert raw depth image to float32 meters with optional clipping.

    LayeredDepth-Syn depth images are stored as I;16 and commonly represent millimeters.
    """
    depth = transform_depth(depth_image).float() / max(depth_scale, 1e-12)

    # Preserve invalid pixels (zeros / non-finite) as zeros instead of raising them to clip_min.
    valid = torch.isfinite(depth) & (depth > 0)
    depth_out = torch.zeros_like(depth)
    if valid.any():
        if depth_clip_max > 0:
            depth_out[valid] = torch.clamp(depth[valid], min=depth_clip_min, max=depth_clip_max)
        else:
            depth_out[valid] = torch.clamp(depth[valid], min=depth_clip_min)
    return depth_out


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


_DEPTH_LAYER_PATTERNS = (
    re.compile(r"^depth_(\d+)(?:\.png)?$", flags=re.IGNORECASE),
    re.compile(r"^depth(\d+)$", flags=re.IGNORECASE),
    re.compile(r"^layer_(\d+)_depth$", flags=re.IGNORECASE),
    re.compile(r"^depth_layer_(\d+)$", flags=re.IGNORECASE),
)


def _parse_depth_layer_id(key: str) -> Optional[int]:
    for pattern in _DEPTH_LAYER_PATTERNS:
        match = pattern.match(key)
        if match is None:
            continue
        layer_id = int(match.group(1))
        return layer_id if layer_id > 0 else None
    return None


def _discover_depth_layer_keys(sample: Dict[str, Any]) -> Dict[int, str]:
    layer_to_key: Dict[int, str] = {}
    for key in sample.keys():
        layer_id = _parse_depth_layer_id(key)
        if layer_id is None:
            continue

        chosen = layer_to_key.get(layer_id)
        if chosen is None:
            layer_to_key[layer_id] = key
            continue

        # Prefer canonical png-style keys when duplicates exist.
        chosen_rank = 1 if chosen.lower().endswith(".png") else 0
        key_rank = 1 if key.lower().endswith(".png") else 0
        if key_rank > chosen_rank:
            layer_to_key[layer_id] = key
    return layer_to_key


class PrecomputedFeatureStore:
    """Loads feature tensors from sharded .pt files with sample_id lookup."""

    def __init__(self, index_path: str, max_cached_shards: int = 1) -> None:
        index_file = Path(index_path)
        if not index_file.exists():
            raise FileNotFoundError(f"Precomputed feature index not found: {index_file}")

        with index_file.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        self.samples: Dict[str, Any] = payload["samples"]
        self.metadata: Dict[str, Any] = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        self._shard_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._cache_order: list[str] = []
        self.max_cached_shards = max(0, int(max_cached_shards))

    def _load_shard(self, shard_path: str) -> Dict[str, torch.Tensor]:
        try:
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        except TypeError:
            # Backward compatibility for older torch versions without weights_only.
            try:
                shard = torch.load(shard_path, map_location="cpu")
            except Exception as exc:
                raise RuntimeError(f"Failed to load precomputed shard '{shard_path}': {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to load precomputed shard '{shard_path}': {exc}") from exc

        if not isinstance(shard, dict):
            raise RuntimeError(f"Precomputed shard '{shard_path}' must contain a dict, got {type(shard)}")

        for key, tensor in shard.items():
            if not torch.is_tensor(tensor):
                raise RuntimeError(f"Shard '{shard_path}' key '{key}' is not a tensor: {type(tensor)}")
            if tensor.numel() == 0:
                raise RuntimeError(f"Shard '{shard_path}' key '{key}' is empty")
            non_finite = int((~torch.isfinite(tensor)).sum().item())
            if non_finite > 0:
                raise RuntimeError(
                    f"Shard '{shard_path}' key '{key}' has non-finite values: {non_finite}/{tensor.numel()}"
                )
            if bool(torch.all(tensor == 0)):
                raise RuntimeError(f"Shard '{shard_path}' key '{key}' is all zeros (corrupted feature cache)")

        return shard

    def get(self, split_name: str, sample_id: str) -> torch.Tensor:
        # Support both split-scoped index payloads ({"train": {...}, "validation": {...}})
        # and flat payloads ({sample_id: meta, ...}). Never silently fall back across splits.
        first_val = next(iter(self.samples.values()), None)
        is_flat_payload = bool(
            isinstance(first_val, dict)
            and "shard_path" in first_val
            and "feature_key" in first_val
        )

        if split_name in self.samples:
            split_samples = self.samples[split_name]
        elif is_flat_payload:
            split_samples = self.samples
        else:
            available_splits = ", ".join(sorted(str(k) for k in self.samples.keys()))
            raise KeyError(
                f"split '{split_name}' not found in feature index. Available splits: [{available_splits}]"
            )

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


def _infer_depth_defaults(data_cfg: Dict[str, Any]) -> tuple[float, float, float]:
    """Infer sane depth defaults from dataset names when config values are absent.

    We keep LayeredDepth-Syn behavior aligned with paper/eval conventions while
    avoiding accidental global clipping for unrelated datasets.
    """
    train_name = str(data_cfg.get("train_dataset_name", "")).lower()
    val_name = str(data_cfg.get("val_dataset_name", "")).lower()
    layered_syn = "layereddepth-syn" in train_name and "layereddepth-syn" in val_name
    if layered_syn:
        return 1000.0, 1e-3, 30.0
    return 1.0, 1e-3, 0.0


class LayeredDepthDataset(Dataset):
    def __init__(
        self,
        hf_dataset: Any,
        feature_store: Optional[PrecomputedFeatureStore],
        split_name: str,
        depth_scale: float,
        depth_clip_min: float,
        depth_clip_max: float,
        max_layers: Optional[int] = None,
    ) -> None:
        self.ds = hf_dataset
        self.feature_store = feature_store
        self.split_name = split_name
        self.depth_scale = depth_scale
        self.depth_clip_min = depth_clip_min
        self.depth_clip_max = depth_clip_max
        self.max_layers = int(max_layers) if max_layers is not None else None
        if self.max_layers is not None and self.max_layers < 1:
            raise ValueError(f"max_layers must be >= 1 when set, got {self.max_layers}")

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.ds[index]
        sample_id = _extract_sample_id(sample, index)

        image_key = _resolve_key(sample, ("image.png", "image", "rgb", "rgb_image"), "image")

        layer_keys = _discover_depth_layer_keys(sample)
        if self.max_layers is not None:
            layer_keys = {k: v for k, v in layer_keys.items() if k <= self.max_layers}
        if not layer_keys:
            if "tuples.json" in sample:
                raise KeyError(
                    "Validation sample contains 'tuples.json' ranking annotations but no dense depth maps. "
                    "For supervised validation loss, use a split with dense depth targets (e.g., "
                    "val_dataset_name=princeton-vl/LayeredDepth-Syn and val_split=validation)."
                )
            available = ", ".join(sorted(sample.keys()))
            raise KeyError(f"No depth layer fields discovered. Available keys: [{available}]")

        depth_layers: Dict[int, torch.Tensor] = {}
        for layer_id, key in sorted(layer_keys.items()):
            depth_layers[layer_id] = convert_depth_tensor(
                sample[key],
                self.depth_scale,
                self.depth_clip_min,
                self.depth_clip_max,
            )

        if 1 not in depth_layers:
            available = sorted(depth_layers.keys())
            raise KeyError(
                "Missing required depth layer 1 in sample after key discovery. "
                f"Discovered layers: {available}"
            )

        depth_1 = depth_layers[1]
        depth_2 = depth_layers.get(2, torch.zeros_like(depth_1))

        item = {
            "sample_id": sample_id,
            "image": transform_img(sample[image_key].convert("RGB")),
            "depth_1": depth_1,
            "depth_2": depth_2,
            "depth_layers": depth_layers,
            "available_layers": sorted(depth_layers.keys()),
        }
        if self.feature_store is not None:
            item["dino_features"] = self.feature_store.get(self.split_name, sample_id)

        return item

# ==========================================
# 2. BATCH COLLATION
# ==========================================
def collate_fn(batch):
    depth_1 = torch.stack([b["depth_1"] for b in batch])
    depth_2 = torch.stack([b["depth_2"] for b in batch])

    available_layers = [list(b.get("available_layers", [])) for b in batch]
    max_layer = max(1, max((max(layers) if layers else 0) for layers in available_layers))

    bsz = len(batch)
    layer_shape = depth_1.shape[1:]
    depth_targets = depth_1.new_zeros((bsz, max_layer, *layer_shape))
    depth_layer_mask = torch.zeros((bsz, max_layer), dtype=torch.bool)
    depth_valid_pixels = torch.zeros((bsz, max_layer, *layer_shape), dtype=torch.bool)

    for i, sample in enumerate(batch):
        for layer_id, layer_depth in sample.get("depth_layers", {}).items():
            if layer_id < 1 or layer_id > max_layer:
                continue
            slot = layer_id - 1
            depth_targets[i, slot] = layer_depth
            valid = torch.isfinite(layer_depth) & (layer_depth > 0)
            depth_valid_pixels[i, slot] = valid
            if bool(valid.any()):
                depth_layer_mask[i, slot] = True

    # Empty-space contract for layered supervision:
    # where front layer is missing but layer-2 exists, set L1 := L2 so early-layer flow targets stay coherent.
    if max_layer >= 2:
        l1_invalid = ~depth_valid_pixels[:, 0]
        l2_valid = depth_valid_pixels[:, 1]
        l1_copy_mask = l1_invalid & l2_valid
        if bool(l1_copy_mask.any()):
            depth_targets[:, 0] = torch.where(l1_copy_mask, depth_targets[:, 1], depth_targets[:, 0])
            depth_valid_pixels[:, 0] = depth_valid_pixels[:, 0] | l1_copy_mask
            depth_layer_mask[:, 0] = depth_valid_pixels[:, 0].reshape(bsz, -1).any(dim=1)

            # Keep backward-compatible keys consistent with updated multilayer tensors.
            depth_1 = depth_targets[:, 0]
            depth_2 = depth_targets[:, 1]

    output = {
        "sample_id": [b["sample_id"] for b in batch],
        "image": torch.stack([b["image"] for b in batch]),
        # Backward-compatible keys used by existing scripts.
        "depth_1": depth_1,
        "depth_2": depth_2,
        # Dynamic multilayer payload used by new training/eval paths.
        "depth_targets": depth_targets,
        "depth_layer_mask": depth_layer_mask,
        "depth_valid_pixels": depth_valid_pixels,
        "available_layers": available_layers,
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

    allow_hf_downloads = bool(data_cfg.get("allow_hf_downloads", True))
    allow_partial_local_shards = bool(data_cfg.get("allow_partial_local_shards", False))
    partial_local_min_shards = int(data_cfg.get("partial_local_shards_min_per_split", 1))
    enable_cache_repair = bool(
        data_cfg.get(
            "repair_hf_cache_once",
            allow_hf_downloads and not allow_partial_local_shards,
        )
    )

    train_hf = load_dataset_split_with_policy(
        data_cfg["train_dataset_name"],
        data_cfg["train_split"],
        cache_dir,
        allow_downloads=allow_hf_downloads,
        allow_cache_repair=enable_cache_repair,
        allow_partial_local_shards=allow_partial_local_shards,
        partial_local_min_shards=partial_local_min_shards,
        log_prefix="[dataset]",
    )
    val_hf = load_dataset_split_with_policy(
        data_cfg["val_dataset_name"],
        data_cfg["val_split"],
        cache_dir,
        allow_downloads=allow_hf_downloads,
        allow_cache_repair=enable_cache_repair,
        allow_partial_local_shards=allow_partial_local_shards,
        partial_local_min_shards=partial_local_min_shards,
        log_prefix="[dataset]",
    )

    default_depth_scale, default_depth_clip_min, default_depth_clip_max = _infer_depth_defaults(data_cfg)
    depth_scale = float(data_cfg.get("depth_scale", default_depth_scale))
    if depth_scale <= 0:
        raise ValueError("data.depth_scale must be > 0")
    depth_clip_min = float(data_cfg.get("depth_clip_min", default_depth_clip_min))
    depth_clip_max = float(data_cfg.get("depth_clip_max", default_depth_clip_max))
    max_layers_cfg = int(data_cfg.get("max_layers", cfg.get("architecture", {}).get("max_layer_id", 0)))
    max_layers = max_layers_cfg if max_layers_cfg > 0 else None

    train_dataset = LayeredDepthDataset(
        train_hf,
        feature_store,
        data_cfg["train_split"],
        depth_scale,
        depth_clip_min,
        depth_clip_max,
        max_layers=max_layers,
    )
    val_dataset = LayeredDepthDataset(
        val_hf,
        feature_store,
        data_cfg["val_split"],
        depth_scale,
        depth_clip_min,
        depth_clip_max,
        max_layers=max_layers,
    )

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