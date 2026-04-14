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

LBP_JOB_START_TS="${LBP_JOB_START_TS:-$(date +%s)}"
export LBP_JOB_START_TS

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
PY_ENV_ROOT="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
if [[ -f "$PY_ENV_ROOT/lib/libstdc++.so.6" ]]; then
	export LD_PRELOAD="$PY_ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/server/default.yaml}"

mkdir -p "$ROOT_DIR/runs/current/logs"
cd "$ROOT_DIR"

ORIG_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
if [[ "${LBP_AUTOSELECT_GPU:-1}" == "1" ]] && command -v nvidia-smi >/dev/null 2>&1; then
	BEST_GPU="$(nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader,nounits | awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3); if (NR==1 || $2 > best_free) {best_free=$2; best_idx=$1; best_used=$3}} END{print best_idx}')"
	if [[ -n "$BEST_GPU" ]]; then
		export CUDA_VISIBLE_DEVICES="$BEST_GPU"
		if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import sys
import torch
ok = torch.cuda.is_available() and torch.cuda.device_count() >= 1
sys.exit(0 if ok else 1)
PY
		then
			export CUDA_VISIBLE_DEVICES="$ORIG_CUDA_VISIBLE_DEVICES"
			echo "[gpu] auto-select fallback: selected GPU not accessible under current allocation; preserving original CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
		else
			echo "[gpu] auto-selected physical GPU with highest free memory: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
		fi
	fi
fi

# Helps reduce fragmentation-related CUDA OOMs in long FFT-heavy runs.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[env] Using PYTHON_BIN=$PYTHON_BIN"
echo "[env] LD_PRELOAD=${LD_PRELOAD:-<empty>}"
echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "[env] LBP_JOB_START_TS=$LBP_JOB_START_TS"
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

if [[ -z "${PRECHECK_MAX_BATCH:-}" ]]; then
	PRECHECK_MAX_BATCH="$($PYTHON_BIN - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
print(int(cfg.get("data", {}).get("batch_size", 1)))
PY
)"
fi

echo "[cmd] $PYTHON_BIN scripts/training/smoke_batch.py --config $CONFIG_PATH --max-batch-size ${PRECHECK_MAX_BATCH}"
"$PYTHON_BIN" scripts/training/smoke_batch.py --config "$CONFIG_PATH" --max-batch-size "${PRECHECK_MAX_BATCH}"
echo "[OK] one-batch precheck passed"

STAGE_B_PERIODIC_EVAL_EVERY="${STAGE_B_PERIODIC_EVAL_EVERY:-10}"
"$PYTHON_BIN" cli.py stage-b --config "$CONFIG_PATH" --python "$PYTHON_BIN" --periodic-eval-every "$STAGE_B_PERIODIC_EVAL_EVERY"
