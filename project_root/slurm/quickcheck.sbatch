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
if [[ ! -f "$ROOT_DIR/train.py" ]]; then
	ROOT_DIR="$PWD/project_root"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/quickcheck_server.yaml}"

mkdir -p "$ROOT_DIR/logs"
cd "$ROOT_DIR"

"$PYTHON_BIN" scripts/preflight_server.py --config "$CONFIG_PATH" --sbatch "$ROOT_DIR/slurm/quickcheck.sbatch"
"$PYTHON_BIN" scripts/server_env_doctor.py --config "$CONFIG_PATH"
bash scripts/run_quickcheck.sh "$CONFIG_PATH"
