"""Structured experiment logging with optional Weights & Biases integration."""

from __future__ import annotations

import os
from typing import Any, Dict


def setup_wandb(cfg: Dict[str, Any], model=None):
    log_cfg = cfg.get("logging", {})
    if not log_cfg.get("use_wandb", False):
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled in config but not installed.") from exc

    mode = str(log_cfg.get("mode", "online")).lower()
    valid_modes = {"online", "offline", "dryrun", "disabled"}
    if mode not in valid_modes:
        raise ValueError(f"Invalid wandb mode '{mode}'. Expected one of: {sorted(valid_modes)}")

    # Keep behavior deterministic across local/server by honoring config mode explicitly.
    os.environ["WANDB_MODE"] = mode

    run = wandb.init(
        project=log_cfg.get("project"),
        entity=log_cfg.get("entity"),
        name=log_cfg.get("run_name"),
        mode=mode,
        config=cfg,
    )

    if run is not None:
        print(f"[wandb] Initialized run: {run.name} ({run.id})", flush=True)
        if getattr(run, "url", None):
            print(f"[wandb] URL: {run.url}", flush=True)

    if run is not None and model is not None and bool(log_cfg.get("watch_model", True)):
        watch_log = str(log_cfg.get("watch_log", "gradients"))
        watch_log_freq = int(log_cfg.get("watch_log_freq", 100))
        run.watch(model, log=watch_log, log_freq=watch_log_freq)

    return run
