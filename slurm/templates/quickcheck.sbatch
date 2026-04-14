#!/bin/bash
#SBATCH --job-name=ld-quickcheck
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=runs/current/logs/%x-%j.out
#SBATCH --error=runs/current/logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/cli.py" ]]; then
	echo "[FAIL] cli.py not found in ROOT_DIR=$ROOT_DIR" >&2
	exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/server/quickcheck.yaml}"
PY_ENV_ROOT="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
if [[ -f "$PY_ENV_ROOT/lib/libstdc++.so.6" ]]; then
	export LD_PRELOAD="$PY_ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

mkdir -p "$ROOT_DIR/runs/current/logs"
cd "$ROOT_DIR"

echo "[env] Using PYTHON_BIN=$PYTHON_BIN"
echo "[env] LD_PRELOAD=${LD_PRELOAD:-<empty>}"

"$PYTHON_BIN" scripts/server/preflight.py --config "$CONFIG_PATH" --sbatch "$ROOT_DIR/slurm/templates/quickcheck.sbatch"
"$PYTHON_BIN" scripts/server/check_env.py --config "$CONFIG_PATH"
"$PYTHON_BIN" cli.py quickcheck --config "$CONFIG_PATH" --python "$PYTHON_BIN"
