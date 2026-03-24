#!/usr/bin/env python3
"""Server environment diagnostics for auth, dataset access, and filesystem readiness."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict
import sys

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check server env readiness")
    p.add_argument("--config", required=True, help="Config path")
    return p.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    return cfg


def has_wandb_auth() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path.home() / ".netrc"
    if not netrc.exists():
        return False
    content = netrc.read_text(encoding="utf-8", errors="ignore")
    return "api.wandb.ai" in content


def has_hf_auth() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"):
        return True
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    return token_file.exists()


def check_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    data_cfg = cfg.get("data", {})
    auth_cfg = cfg.get("auth", {})

    cache_dir = Path(str(data_cfg.get("cache_dir", "")))
    staged_root = Path(str(data_cfg.get("staged_root", "")))
    index_path = Path(str(data_cfg.get("precomputed_index_path", "")))

    print("[doctor] server environment diagnostics")
    print(f"[doctor] user={os.environ.get('USER', '<unknown>')}")
    print(f"[doctor] cwd={Path.cwd()}")
    print(f"[doctor] python={sys.executable}")

    missing_mods = []
    for mod in ("numpy", "torch", "yaml", "wandb"):
        try:
            __import__(mod)
        except Exception:
            missing_mods.append(mod)
    if missing_mods:
        raise SystemExit(f"[FAIL] Missing Python packages in current env: {', '.join(missing_mods)}")

    print(f"[doctor] cache_dir={cache_dir} writable={check_writable(cache_dir)}")
    print(f"[doctor] staged_root={staged_root} exists={staged_root.exists()}")
    print(f"[doctor] precomputed_index_path={index_path} exists={index_path.exists()}")

    wandb_required = bool(auth_cfg.get("require_wandb_login", False))
    hf_required = bool(auth_cfg.get("require_hf_login", False))

    wandb_ok = has_wandb_auth()
    hf_ok = has_hf_auth()

    print(f"[doctor] wandb_auth={wandb_ok} required={wandb_required}")
    print(f"[doctor] hf_auth={hf_ok} required={hf_required}")

    if wandb_required and not wandb_ok:
        raise SystemExit("[FAIL] Missing W&B auth. Run: wandb login")
    if hf_required and not hf_ok:
        raise SystemExit("[FAIL] Missing Hugging Face auth. Run: huggingface-cli login")

    print("[OK] server_env_doctor passed")


if __name__ == "__main__":
    main()
