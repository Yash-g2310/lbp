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
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/server.yaml}"

mkdir -p "$ROOT_DIR/logs"
cd "$ROOT_DIR"

"$PYTHON_BIN" scripts/preflight_server.py --config "$CONFIG_PATH" --sbatch "$ROOT_DIR/slurm/train_server.sbatch"
"$PYTHON_BIN" scripts/server_env_doctor.py --config "$CONFIG_PATH"
"$PYTHON_BIN" scripts/train_and_eval.py --config "$CONFIG_PATH" --python "$PYTHON_BIN"
