#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODE="${1:-train}"
CHECKPOINT_PATH="${3:-${CHECKPOINT_PATH:-}}"

case "$MODE" in
  quickcheck)
    SBATCH_FILE="$ROOT_DIR/slurm/templates/quickcheck.sbatch"
    CONFIG_PATH="${2:-$ROOT_DIR/configs/server/quickcheck.yaml}"
    ;;
  train)
    SBATCH_FILE="$ROOT_DIR/slurm/templates/train.sbatch"
    CONFIG_PATH="${2:-$ROOT_DIR/configs/server/default.yaml}"
    ;;
  eval)
    SBATCH_FILE="$ROOT_DIR/slurm/templates/eval_real.sbatch"
    CONFIG_PATH="${2:-$ROOT_DIR/configs/server/default.yaml}"
    ;;
  *)
    echo "usage: $0 [quickcheck|train|eval] [config_path] [checkpoint_path_for_eval_optional]" >&2
    exit 2
    ;;
esac

"$PYTHON_BIN" "$ROOT_DIR/scripts/server/preflight.py" --config "$CONFIG_PATH" --sbatch "$SBATCH_FILE"

mkdir -p "$ROOT_DIR/runs/current/logs"

sbatch \
  --chdir="$ROOT_DIR" \
  --output="$ROOT_DIR/runs/current/logs/%x-%j.out" \
  --error="$ROOT_DIR/runs/current/logs/%x-%j.err" \
  --export=ALL,ROOT_DIR="$ROOT_DIR",PYTHON_BIN="$PYTHON_BIN",CONFIG_PATH="$CONFIG_PATH",CHECKPOINT_PATH="$CHECKPOINT_PATH" \
  "$SBATCH_FILE"
