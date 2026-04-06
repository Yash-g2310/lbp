#!/bin/bash
#SBATCH --job-name=ld-quickcheck
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$PWD}"
if [[ ! -f "$ROOT_DIR/cli.py" ]]; then
	echo "[FAIL] cli.py not found in ROOT_DIR=$ROOT_DIR" >&2
	exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/server/quickcheck.yaml}"

mkdir -p "$ROOT_DIR/logs"
cd "$ROOT_DIR"

"$PYTHON_BIN" scripts/server/preflight.py --config "$CONFIG_PATH" --sbatch "$ROOT_DIR/slurm/templates/quickcheck.sbatch"
"$PYTHON_BIN" scripts/server/check_env.py --config "$CONFIG_PATH"
"$PYTHON_BIN" cli.py quickcheck --config "$CONFIG_PATH" --python "$PYTHON_BIN"
