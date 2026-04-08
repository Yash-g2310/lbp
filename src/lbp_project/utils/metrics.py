"""Evaluation metrics for dense depth and tuple-wise benchmark annotations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch


@dataclass
class RunningAverage:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * int(n)
        self.count += int(n)

    @property
    def mean(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def compute_depth_metrics(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0e-6) -> Dict[str, float]:
    """Compute standard depth metrics on valid pixels (target > 0)."""
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch in compute_depth_metrics: pred={pred.shape}, target={target.shape}")

    valid = target > 0
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return {"abs_rel": 0.0, "rmse": 0.0, "delta1": 0.0, "delta2": 0.0, "valid_count": 0.0}

    p = pred[valid].clamp_min(eps)
    t = target[valid].clamp_min(eps)

    abs_rel = torch.mean(torch.abs(p - t) / t)
    rmse = torch.sqrt(torch.mean((p - t) ** 2))

    ratio = torch.maximum(p / t, t / p)
    delta1 = (ratio < 1.25).float().mean()
    delta2 = (ratio < (1.25 ** 2)).float().mean()

    return {
        "abs_rel": float(abs_rel.item()),
        "rmse": float(rmse.item()),
        "delta1": float(delta1.item()),
        "delta2": float(delta2.item()),
        "valid_count": float(valid_count),
    }


def _extract_points(tuple_item: Dict[str, Any]) -> List[List[float]]:
    points: List[Tuple[str, List[float]]] = []
    for key, value in tuple_item.items():
        if key.startswith("p") and isinstance(value, list) and len(value) >= 3:
            points.append((key, value))
    points.sort(key=lambda x: x[0])
    return [p for _, p in points]


def _tuple_correct(depth_array: np.ndarray, points: List[List[float]], scale_x: float, scale_y: float) -> Tuple[bool, bool]:
    # Return (is_valid, is_correct)
    if len(points) < 2:
        return False, False

    layers = [int(p[2]) for p in points]

    pred_h, pred_w = depth_array.shape
    pred_depths: List[float] = []
    for p in points:
        x = min(max(int(float(p[0]) * scale_x), 0), pred_w - 1)
        y = min(max(int(float(p[1]) * scale_y), 0), pred_h - 1)
        pred_depths.append(float(depth_array[y, x]))

    # Pairwise relative-depth constraints are robust even when multiple points share the same layer id.
    # For each pair with different gt layers: smaller layer id should have smaller predicted depth.
    constraints = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            li, lj = layers[i], layers[j]
            if li == lj:
                continue
            constraints += 1
            if li < lj:
                if not (pred_depths[i] < pred_depths[j]):
                    return True, False
            else:
                if not (pred_depths[j] < pred_depths[i]):
                    return True, False

    if constraints == 0:
        return False, False
    return True, True


def evaluate_tuple_sample(
    predicted_depth_map: torch.Tensor,
    sample: Dict[str, Any],
    original_size: Tuple[int, int],
    layer_key: str = "layer_all",
) -> Dict[str, Dict[str, int]]:
    """Evaluate tuple-wise accuracy for one sample.

    Returns counts for pairs, trips, quads, and overall.
    """
    if predicted_depth_map.ndim != 4 or predicted_depth_map.shape[0] != 1:
        raise ValueError("predicted_depth_map must have shape [1, 1, H, W] or [1, C, H, W]")

    annotation_key = "tuples.json" if "tuples.json" in sample else None
    if annotation_key is None:
        raise KeyError("Sample does not contain tuples.json annotations")

    tuples_root = sample[annotation_key]
    if layer_key not in tuples_root:
        raise KeyError(f"Layer key '{layer_key}' not found in tuples.json")

    tuple_group = tuples_root[layer_key]
    split_keys = {
        "pairs": tuple_group.get("pairs", []),
        "trips": tuple_group.get("trips", []),
        "quads": tuple_group.get("quads", []),
    }

    orig_w, orig_h = original_size
    pred_h, pred_w = predicted_depth_map.shape[2], predicted_depth_map.shape[3]
    scale_x, scale_y = pred_w / max(1, orig_w), pred_h / max(1, orig_h)
    depth_array = predicted_depth_map[0, 0].detach().cpu().numpy()

    out: Dict[str, Dict[str, int]] = {
        "pairs": {"correct": 0, "total": 0},
        "trips": {"correct": 0, "total": 0},
        "quads": {"correct": 0, "total": 0},
        "all": {"correct": 0, "total": 0},
    }

    for metric_key, tuples_list in split_keys.items():
        for item in tuples_list:
            if not item.get("is_real", True):
                continue
            points = _extract_points(item)
            is_valid, is_correct = _tuple_correct(depth_array, points, scale_x, scale_y)
            if not is_valid:
                continue
            out[metric_key]["total"] += 1
            out["all"]["total"] += 1
            if is_correct:
                out[metric_key]["correct"] += 1
                out["all"]["correct"] += 1

    return out


def extract_required_layer_ids(sample: Dict[str, Any], layer_key: str = "layer_all") -> List[int]:
    annotation_key = "tuples.json" if "tuples.json" in sample else None
    if annotation_key is None:
        raise KeyError("Sample does not contain tuples.json annotations")

    tuples_root = sample[annotation_key]
    if layer_key not in tuples_root:
        raise KeyError(f"Layer key '{layer_key}' not found in tuples.json")

    tuple_group = tuples_root[layer_key]
    layer_ids: set[int] = set()
    for metric_key in ("pairs", "trips", "quads"):
        for item in tuple_group.get(metric_key, []):
            if not item.get("is_real", True):
                continue
            points = _extract_points(item)
            for p in points:
                layer_id = int(p[2])
                if layer_id > 0:
                    layer_ids.add(layer_id)
    return sorted(layer_ids)


def _tuple_correct_multi_layer(
    depth_maps: Dict[int, np.ndarray],
    points: List[List[float]],
    scale_map: Dict[int, Tuple[float, float]],
) -> Tuple[bool, bool, bool]:
    # Return (is_valid, is_correct, missing_layer_prediction)
    if len(points) < 2:
        return False, False, False

    layers = [int(p[2]) for p in points]
    pred_depths: List[float] = []

    for p, layer_id in zip(points, layers):
        layer_depth = depth_maps.get(layer_id)
        scales = scale_map.get(layer_id)
        if layer_depth is None or scales is None:
            return False, False, True

        pred_h, pred_w = layer_depth.shape
        scale_x, scale_y = scales
        x = min(max(int(float(p[0]) * scale_x), 0), pred_w - 1)
        y = min(max(int(float(p[1]) * scale_y), 0), pred_h - 1)
        pred_depths.append(float(layer_depth[y, x]))

    constraints = 0
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            li, lj = layers[i], layers[j]
            if li == lj:
                continue
            constraints += 1
            if li < lj:
                if not (pred_depths[i] < pred_depths[j]):
                    return True, False, False
            else:
                if not (pred_depths[j] < pred_depths[i]):
                    return True, False, False

    if constraints == 0:
        return False, False, False
    return True, True, False


def evaluate_tuple_sample_multi_layer(
    predicted_depth_by_layer: Dict[int, torch.Tensor],
    sample: Dict[str, Any],
    original_size: Tuple[int, int],
    layer_key: str = "layer_all",
) -> Dict[str, Dict[str, int]]:
    """Evaluate tuple accuracy using per-layer prediction maps from multi-pass inference."""
    if not predicted_depth_by_layer:
        raise ValueError("predicted_depth_by_layer is empty")

    annotation_key = "tuples.json" if "tuples.json" in sample else None
    if annotation_key is None:
        raise KeyError("Sample does not contain tuples.json annotations")

    tuples_root = sample[annotation_key]
    if layer_key not in tuples_root:
        raise KeyError(f"Layer key '{layer_key}' not found in tuples.json")

    tuple_group = tuples_root[layer_key]
    split_keys = {
        "pairs": tuple_group.get("pairs", []),
        "trips": tuple_group.get("trips", []),
        "quads": tuple_group.get("quads", []),
    }

    orig_w, orig_h = original_size
    depth_maps: Dict[int, np.ndarray] = {}
    scale_map: Dict[int, Tuple[float, float]] = {}
    for layer_id, depth_tensor in predicted_depth_by_layer.items():
        if depth_tensor.ndim != 4 or depth_tensor.shape[0] != 1:
            raise ValueError(
                f"predicted depth for layer {layer_id} must have shape [1,1,H,W] or [1,C,H,W], got {tuple(depth_tensor.shape)}"
            )
        pred_h, pred_w = depth_tensor.shape[2], depth_tensor.shape[3]
        depth_maps[int(layer_id)] = depth_tensor[0, 0].detach().cpu().numpy()
        scale_map[int(layer_id)] = (pred_w / max(1, orig_w), pred_h / max(1, orig_h))

    out: Dict[str, Dict[str, int]] = {
        "pairs": {"correct": 0, "total": 0},
        "trips": {"correct": 0, "total": 0},
        "quads": {"correct": 0, "total": 0},
        "all": {"correct": 0, "total": 0},
    }
    missing_layer_tuples = 0

    for metric_key, tuples_list in split_keys.items():
        for item in tuples_list:
            if not item.get("is_real", True):
                continue
            points = _extract_points(item)
            is_valid, is_correct, missing_layer = _tuple_correct_multi_layer(depth_maps, points, scale_map)
            if missing_layer:
                missing_layer_tuples += 1
                continue
            if not is_valid:
                continue

            out[metric_key]["total"] += 1
            out["all"]["total"] += 1
            if is_correct:
                out[metric_key]["correct"] += 1
                out["all"]["correct"] += 1

    out["missing_layer_tuples"] = {"correct": 0, "total": int(missing_layer_tuples)}
    return out


def summarize_tuple_counts(counts: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    summary: Dict[str, float] = {}
    for key, entry in counts.items():
        total = max(0, int(entry.get("total", 0)))
        correct = max(0, int(entry.get("correct", 0)))
        acc = (100.0 * correct / total) if total > 0 else 0.0
        summary[f"{key}_correct"] = float(correct)
        summary[f"{key}_total"] = float(total)
        summary[f"{key}_acc"] = float(acc)
    return summary


def merge_tuple_counts(accum: Dict[str, Dict[str, int]], inc: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    for key in ("pairs", "trips", "quads", "all"):
        accum.setdefault(key, {"correct": 0, "total": 0})
        accum[key]["correct"] += int(inc.get(key, {}).get("correct", 0))
        accum[key]["total"] += int(inc.get(key, {}).get("total", 0))
    return accum
