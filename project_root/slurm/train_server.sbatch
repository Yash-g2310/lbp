#!/bin/bash
#SBATCH --job-name=ld-train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=23:55:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/train.py" ]]; then
	ROOT_DIR="$PWD/project_root"
fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
	PYTHON_BIN="$PYTHON_BIN"
elif [[ -x "$HOME/.conda/envs/monodepth/bin/python" ]]; then
	PYTHON_BIN="$HOME/.conda/envs/monodepth/bin/python"
elif command -v python >/dev/null 2>&1; then
	PYTHON_BIN="$(command -v python)"
else
	echo "[FAIL] No python executable found. Set PYTHON_BIN explicitly." >&2
	exit 1
fi
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/server.yaml}"

mkdir -p "$ROOT_DIR/logs"
cd "$ROOT_DIR"

# Helps reduce fragmentation-related CUDA OOMs in long FFT-heavy runs.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[env] Using PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
mods = ("numpy", "torch", "yaml", "wandb")
missing = []
for m in mods:
	try:
		__import__(m)
	except Exception:
		missing.append(m)
if missing:
	raise SystemExit(f"[FAIL] Python env missing required packages: {', '.join(missing)}; executable={sys.executable}")
print(f"[env] python={sys.executable}")
PY

"$PYTHON_BIN" scripts/preflight_server.py --config "$CONFIG_PATH" --sbatch "$ROOT_DIR/slurm/train_server.sbatch"
"$PYTHON_BIN" scripts/server_env_doctor.py --config "$CONFIG_PATH"

echo "[cmd] $PYTHON_BIN scripts/smoke_one_batch.py --config $CONFIG_PATH --max-batch-size ${PRECHECK_MAX_BATCH:-1}"
"$PYTHON_BIN" scripts/smoke_one_batch.py --config "$CONFIG_PATH" --max-batch-size "${PRECHECK_MAX_BATCH:-1}"
echo "[OK] one-batch precheck passed"

"$PYTHON_BIN" scripts/train_and_eval.py --config "$CONFIG_PATH" --python "$PYTHON_BIN"
