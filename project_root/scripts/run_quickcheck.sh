#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/quickcheck_local.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT_DIR"

echo "[stage 1/5] audit configs"
"$PYTHON_BIN" scripts/audit_configs.py --configs "$CONFIG_PATH"

echo "[stage 2/5] audit dataset and shards"
"$PYTHON_BIN" scripts/audit_shards.py --config "$CONFIG_PATH" --verify-index-coverage

echo "[stage 3/5] data quickcheck"
"$PYTHON_BIN" scripts/quickcheck_data.py --config "$CONFIG_PATH"

echo "[stage 4/5] train quickcheck"
"$PYTHON_BIN" scripts/quickcheck_train.py --config "$CONFIG_PATH" --train-steps 3 --val-steps 1

echo "[stage 5/5] real tuple eval quickcheck"
QC_CKPT="$ROOT_DIR/artifacts/quickcheck/checkpoints/quickcheck/sanity_roundtrip.pth"
"$PYTHON_BIN" scripts/evaluate_real_tuples.py \
	--config "$CONFIG_PATH" \
	--checkpoint "$QC_CKPT" \
	--splits validation,test \
	--layer-keys layer_all,layer_first \
	--target-layer 1 \
	--max-samples 20 \
	--output "$ROOT_DIR/artifacts/quickcheck/reports/real_tuple_eval_quickcheck.json"

echo "[done] quickcheck pipeline passed for $CONFIG_PATH"
