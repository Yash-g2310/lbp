"""Config I/O helpers used by CLI and scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from lbp_project.config.validation import validate_config_dict


def load_yaml(path: str | Path, validate: bool = True) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {cfg_path}")
    if validate:
        validate_config_dict(data, source=cfg_path)
    return data
