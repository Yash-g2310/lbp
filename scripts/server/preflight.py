#!/usr/bin/env python3
"""Server preflight validation against cluster policy and config requirements."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lbp_project.config.io import load_yaml
from lbp_project.config.stage_policy import validate_stage_policy
from lbp_project.data.preflight import (
    build_download_matrix,
    enforce_hardware_profile,
    enforce_startup_preflight,
    format_download_matrix,
    format_hardware_profile,
)
from lbp_project.stage_gate import evaluate_stage_b_gate, format_stage_b_gate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preflight checks for server config and slurm script")
    p.add_argument("--config", required=True, help="Server YAML config")
    p.add_argument("--sbatch", required=False, help="Optional sbatch file to validate")
    return p.parse_args()


def parse_sbatch(path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    patt = re.compile(r"^#SBATCH\s+--([^=\s]+)(?:=(.+))?$")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = patt.match(line.strip())
        if not m:
            continue
        key = m.group(1)
        value = (m.group(2) or "").strip()
        cfg[key] = value
    return cfg


def fail(msg: str) -> None:
    raise SystemExit(f"[FAIL] {msg}")


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


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    print(format_download_matrix(build_download_matrix(cfg)), flush=True)

    hardware_profile = enforce_hardware_profile(cfg, strict=True)
    print(format_hardware_profile(hardware_profile), flush=True)

    preflight_warnings = enforce_startup_preflight(cfg, strict_server_policy=True)
    for warning in preflight_warnings:
        print(f"[WARN] {warning}", flush=True)

    for warning in validate_stage_policy(cfg, stage_mode="stage_b", strict=False):
        print(f"[WARN] {warning}", flush=True)

    try:
        stage_b_gate = evaluate_stage_b_gate(cfg)
    except Exception as exc:
        fail(f"Unable to evaluate Stage-B promotion gate: {exc}")
    print(format_stage_b_gate(stage_b_gate), flush=True)
    if not stage_b_gate.enabled:
        fail("evaluation.stage_b_gate.enabled must be true for Stage-B preflight")
    if not stage_b_gate.passed:
        fail("Stage-B promotion gate failed.\n" + format_stage_b_gate(stage_b_gate))

    hw_cfg = cfg["hardware"]
    log_cfg = cfg.get("logging", {})
    auth_cfg = cfg.get("auth", {})

    if hw_cfg.get("device") != "cuda":
        fail("hardware.device must be 'cuda' for server runs")

    if int(hw_cfg.get("num_workers", 0)) > 40:
        fail("num_workers must be <= 40 to respect CPU thread policy")

    ckpt_dir = Path(cfg.get("training", {}).get("checkpoint", {}).get("dir", ""))
    if not str(ckpt_dir):
        fail("training.checkpoint.dir must be set")
    try:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        fail(f"checkpoint dir cannot be created: {ckpt_dir} ({exc})")
    if not os.access(ckpt_dir, os.W_OK):
        fail(f"checkpoint dir not writable: {ckpt_dir}")

    if bool(log_cfg.get("use_wandb", False)) and bool(auth_cfg.get("require_wandb_login", False)):
        if not has_wandb_auth():
            fail("W&B auth not found. Run 'wandb login' or set WANDB_API_KEY before submission")

    if bool(auth_cfg.get("require_hf_login", False)) and not has_hf_auth():
        fail("Hugging Face auth not found. Run 'huggingface-cli login' or set HF_TOKEN")

    if args.sbatch:
        sbatch_cfg = parse_sbatch(Path(args.sbatch))

        gpus = sbatch_cfg.get("gres", "")
        if "gpu:1" not in gpus and sbatch_cfg.get("gpus", "") not in {"1", ""}:
            fail("SBATCH must request exactly one GPU (e.g., --gres=gpu:1)")

        cpus = int(sbatch_cfg.get("cpus-per-task", "0") or "0")
        if cpus > 40:
            fail("SBATCH cpus-per-task must be <= 40")

        time_str = sbatch_cfg.get("time", "")
        if time_str:
            h, m, s = [int(x) for x in time_str.split(":")]
            total_hours = h + m / 60 + s / 3600
            if total_hours > 24:
                fail("SBATCH time must be <= 24:00:00")

        mem_str = sbatch_cfg.get("mem", "")
        if cpus > 0 and mem_str.endswith("G"):
            mem_gb = float(mem_str[:-1])
            if mem_gb > cpus * 4:
                fail("SBATCH mem exceeds 4GB per requested CPU policy")

    print("[OK] preflight_server passed")


if __name__ == "__main__":
    main()
