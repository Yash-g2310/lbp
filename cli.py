#!/usr/bin/env python3
"""Root CLI wrapper that dispatches to lbp_project.cli."""

from __future__ import annotations

import sys
from pathlib import Path
import runpy

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


if __name__ == "__main__":
    runpy.run_module("lbp_project.cli", run_name="__main__")
