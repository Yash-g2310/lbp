#!/bin/bash
#SBATCH --job-name=ld-train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=23:55:00
#SBATCH --output=runs/current/logs/%x-%j.out
#SBATCH --error=runs/current/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/cli.py" ]]; then
	echo "[FAIL] cli.py not found in ROOT_DIR=$ROOT_DIR" >&2
	exit 1
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
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/server/default.yaml}"

mkdir -p "$ROOT_DIR/runs/current/logs"
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

"$PYTHON_BIN" scripts/server/preflight.py --config "$CONFIG_PATH" --sbatch "$ROOT_DIR/slurm/templates/train.sbatch"
"$PYTHON_BIN" scripts/server/check_env.py --config "$CONFIG_PATH"

echo "[cmd] $PYTHON_BIN scripts/training/smoke_batch.py --config $CONFIG_PATH --max-batch-size ${PRECHECK_MAX_BATCH:-1}"
"$PYTHON_BIN" scripts/training/smoke_batch.py --config "$CONFIG_PATH" --max-batch-size "${PRECHECK_MAX_BATCH:-1}"
echo "[OK] one-batch precheck passed"

STAGE_B_PERIODIC_EVAL_EVERY="${STAGE_B_PERIODIC_EVAL_EVERY:-10}"
"$PYTHON_BIN" cli.py stage-b --config "$CONFIG_PATH" --python "$PYTHON_BIN" --periodic-eval-every "$STAGE_B_PERIODIC_EVAL_EVERY"
