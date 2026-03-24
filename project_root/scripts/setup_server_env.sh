#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a server-side conda environment for this project.
# Usage:
#   bash scripts/setup_server_env.sh [env_name]

ENV_NAME="${1:-monodepth}"
PYTHON_VERSION="3.10"

if ! command -v conda >/dev/null 2>&1; then
  echo "[FAIL] conda not found. Load conda module or install Miniconda first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[info] conda env '$ENV_NAME' already exists"
else
  echo "[info] creating conda env '$ENV_NAME' (python=$PYTHON_VERSION)"
  conda create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
fi

conda activate "$ENV_NAME"

echo "[info] installing python dependencies"
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install datasets pyyaml wandb huggingface_hub tqdm pillow numpy

echo "[info] validating toolchain"
python - <<'PY'
from importlib.util import find_spec
mods = ["torch", "datasets", "yaml", "wandb", "huggingface_hub", "torchvision", "numpy"]
missing = [m for m in mods if find_spec(m) is None]
if missing:
    raise SystemExit(f"[FAIL] missing modules: {missing}")
print("[OK] python dependencies are ready")
PY

echo "[next] authenticate tools if needed:"
echo "  wandb login"
echo "  huggingface-cli login"
echo "[done] environment '$ENV_NAME' is ready"
