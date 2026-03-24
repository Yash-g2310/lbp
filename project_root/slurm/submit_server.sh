#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-train}"

case "$MODE" in
  quickcheck)
    SBATCH_FILE="$ROOT_DIR/slurm/quickcheck.sbatch"
    CONFIG_PATH="${2:-$ROOT_DIR/configs/quickcheck_server.yaml}"
    ;;
  train)
    SBATCH_FILE="$ROOT_DIR/slurm/train_server.sbatch"
    CONFIG_PATH="${2:-$ROOT_DIR/configs/server.yaml}"
    ;;
  *)
    echo "usage: $0 [quickcheck|train] [config_path]" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" "$ROOT_DIR/scripts/preflight_server.py" --config "$CONFIG_PATH" --sbatch "$SBATCH_FILE"

sbatch --export=ALL,ROOT_DIR="$ROOT_DIR",PYTHON_BIN="$PYTHON_BIN",CONFIG_PATH="$CONFIG_PATH" "$SBATCH_FILE"
