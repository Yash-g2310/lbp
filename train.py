#!/usr/bin/env python3
"""Root training wrapper that dispatches to lbp_project.training.main."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


if __name__ == "__main__":
    runpy.run_module("lbp_project.training.main", run_name="__main__")
