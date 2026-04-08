"""HuggingFace dataset loading helpers with local/offline shard policies."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Iterable, List

from datasets import DownloadConfig, load_dataset


def _normalize_token(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _camel_to_snake(value: str) -> str:
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower()


def _dataset_cache_dir_candidates(cache_root: Path, dataset_name: str) -> List[Path]:
    if not cache_root.exists():
        return []

    owner, sep, dataset = dataset_name.partition("/")
    owner = owner.strip().lower()
    dataset = dataset.strip()

    candidates: List[Path] = []
    if sep and owner and dataset:
        dataset_lower = dataset.lower()
        dataset_snake = "-".join(_camel_to_snake(part) for part in dataset.split("-"))
        cache_names = {
            f"{owner}___{dataset_lower}",
            f"{owner}___{dataset_snake}",
        }

        for child in cache_root.iterdir():
            if not child.is_dir():
                continue
            child_name = child.name.lower()
            if child_name in cache_names:
                candidates.append(child)

        if candidates:
            return sorted(candidates)

        # Fallback for cache directory variants that append suffixes.
        for child in cache_root.iterdir():
            if not child.is_dir():
                continue
            child_name = child.name.lower()
            if any(child_name.startswith(name) for name in cache_names):
                candidates.append(child)
        if candidates:
            return sorted(candidates)

    return []


def discover_local_arrow_shards(cache_dir: str | Path, dataset_name: str, split_name: str) -> List[Path]:
    """Discover local Arrow shard files for a dataset/split under HuggingFace cache_dir."""
    cache_root = Path(cache_dir)
    if not cache_root.exists():
        return []

    dataset_norm = _normalize_token(dataset_name)
    split_marker = f"-{str(split_name).lower()}-"

    candidates = _dataset_cache_dir_candidates(cache_root, dataset_name)

    if not candidates:
        candidates = [cache_root]

    matches: List[Path] = []
    seen: set[str] = set()
    for root in candidates:
        for arrow_path in root.rglob("*.arrow"):
            name = arrow_path.name.lower()
            if split_marker not in name:
                continue
            if root == cache_root:
                normalized_path = _normalize_token(str(arrow_path))
                if dataset_norm and dataset_norm not in normalized_path:
                    continue
            key = str(arrow_path)
            if key in seen:
                continue
            seen.add(key)
            matches.append(arrow_path)

    return sorted(matches)


def _load_arrow_dataset_from_files(arrow_files: Iterable[Path]) -> Any:
    files = [str(path) for path in arrow_files]
    if not files:
        raise ValueError("arrow_files must contain at least one file")
    return load_dataset("arrow", data_files=files, split="train")


def load_dataset_split_with_policy(
    dataset_name: str,
    split_name: str,
    cache_dir: str | Path,
    allow_downloads: bool,
    allow_cache_repair: bool,
    allow_partial_local_shards: bool,
    partial_local_min_shards: int,
    log_prefix: str = "[dataset]",
) -> Any:
    """Load a dataset split under explicit local/download policy.

    Policy order:
    1) If partial local shards are allowed and enough local Arrow shards exist, load directly
       from those Arrow files (never touches remote).
    2) Otherwise load via dataset builder with `local_files_only` governed by `allow_downloads`.
    3) Optional one-time cache-repair retry is allowed only when downloads are enabled.
    """
    cache_root = Path(cache_dir)
    min_shards = max(1, int(partial_local_min_shards))

    if allow_partial_local_shards:
        local_shards = discover_local_arrow_shards(cache_root, dataset_name, split_name)
        shard_count = len(local_shards)
        if shard_count >= min_shards:
            print(
                f"{log_prefix} using local Arrow shards for {dataset_name}:{split_name} "
                f"(count={shard_count}, downloads_disabled={not allow_downloads})"
            )
            return _load_arrow_dataset_from_files(local_shards)

        if not allow_downloads:
            raise FileNotFoundError(
                "Local partial-shard mode is enabled but insufficient local Arrow shards were found for "
                f"{dataset_name}:{split_name}. found={shard_count} required_min={min_shards} "
                f"cache_dir={cache_root}"
            )

        print(
            f"{log_prefix} local partial-shard mode found {shard_count} shard(s) for "
            f"{dataset_name}:{split_name} (required_min={min_shards}); falling back to builder load"
        )

    download_cfg = DownloadConfig(local_files_only=not allow_downloads)
    load_kwargs = {
        "path": dataset_name,
        "split": split_name,
        "cache_dir": str(cache_root),
        "streaming": False,
        "download_config": download_cfg,
    }

    try:
        return load_dataset(**load_kwargs)
    except (FileNotFoundError, OSError) as exc:
        if not allow_cache_repair or not allow_downloads:
            raise

        error_text = str(exc)
        cache_file_issue = ".arrow" in error_text or "Failed to open local file" in error_text
        if not cache_file_issue:
            raise

        print(
            f"{log_prefix} detected broken local HuggingFace cache shard for "
            f"{dataset_name}:{split_name}; retrying once with force_redownload"
        )
        return load_dataset(
            **load_kwargs,
            download_mode="force_redownload",
            download_config=DownloadConfig(local_files_only=False),
        )
