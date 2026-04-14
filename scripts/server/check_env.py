#!/usr/bin/env python3
"""Server environment diagnostics for auth, dataset access, and filesystem readiness."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
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
    p = argparse.ArgumentParser(description="Check server env readiness")
    p.add_argument("--config", required=True, help="Config path")
    return p.parse_args()


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


def _stage_b_checkpoint_candidates(cfg: dict) -> list[Path]:
    eval_cfg = cfg.get("evaluation", {})
    train_ckpt_cfg = cfg.get("training", {}).get("checkpoint", {})
    if not isinstance(train_ckpt_cfg, dict):
        return []

    ckpt_dir = Path(str(train_ckpt_cfg.get("dir", "./runs/current/checkpoints")))
    ckpt_name = str(eval_cfg.get("checkpoint_name", train_ckpt_cfg.get("best_name", "best_checkpoint.pth")))
    latest_name = str(train_ckpt_cfg.get("latest_name", "latest_checkpoint.pth"))

    candidates = [ckpt_dir / ckpt_name, ckpt_dir / latest_name]
    deduped: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _allow_stage_b_bootstrap_without_checkpoint(cfg: dict, stage_b_gate) -> tuple[bool, str]:
    eval_cfg = cfg.get("evaluation", {})
    gate_cfg = eval_cfg.get("stage_b_gate", {})
    if not isinstance(gate_cfg, dict):
        gate_cfg = {}

    if not bool(gate_cfg.get("allow_bootstrap_without_checkpoint", True)):
        return False, "evaluation.stage_b_gate.allow_bootstrap_without_checkpoint=false"

    existing = [p for p in _stage_b_checkpoint_candidates(cfg) if p.exists()]
    if existing:
        return False, "existing checkpoints present"

    no_evidence_issue = any("No stage evidence reports found" in issue for issue in stage_b_gate.issues)
    if not no_evidence_issue:
        return False, "gate failed for reasons other than missing evidence"

    return True, "no existing checkpoints and no prior stage evidence"


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    print(format_download_matrix(build_download_matrix(cfg), prefix="[doctor]"), flush=True)

    hardware_profile = enforce_hardware_profile(
        cfg,
        strict=bool(cfg.get("data", {}).get("require_local_staging", False)),
    )
    print(format_hardware_profile(hardware_profile, prefix="[doctor][hardware]"), flush=True)

    warnings = enforce_startup_preflight(
        cfg,
        strict_server_policy=bool(cfg.get("data", {}).get("require_local_staging", False)),
    )
    for warning in warnings:
        print(f"[doctor][warn] {warning}", flush=True)

    for warning in validate_stage_policy(cfg, stage_mode="stage_b", strict=False):
        print(f"[doctor][warn] {warning}", flush=True)

    try:
        stage_b_gate = evaluate_stage_b_gate(cfg)
    except Exception as exc:
        raise SystemExit(f"[FAIL] Unable to evaluate Stage-B promotion gate: {exc}")
    print(format_stage_b_gate(stage_b_gate, prefix="[doctor][stage-b-gate]"), flush=True)
    if not stage_b_gate.enabled:
        raise SystemExit("[FAIL] evaluation.stage_b_gate.enabled must be true for Stage-B checks")
    if not stage_b_gate.passed:
        allow_bootstrap, reason = _allow_stage_b_bootstrap_without_checkpoint(cfg, stage_b_gate)
        if allow_bootstrap:
            print(
                "[doctor][stage-b-gate][bootstrap] bypassing pre-train gate once: "
                f"{reason}. Fresh training run will create new checkpoints/evidence.",
                flush=True,
            )
        else:
            raise SystemExit(
                "[FAIL] Stage-B promotion gate failed.\n"
                + format_stage_b_gate(stage_b_gate, prefix="[doctor][stage-b-gate]")
            )

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
