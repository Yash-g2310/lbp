#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/configs/local/quickcheck.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$ROOT_DIR"

echo "[stage 1/6] audit configs"
"$PYTHON_BIN" scripts/config/validate_yaml.py --configs "$CONFIG_PATH"

echo "[stage 2/6] audit dataset and shards"
"$PYTHON_BIN" scripts/config/validate_dataset.py --config "$CONFIG_PATH" --verify-index-coverage --progress-every 1000

echo "[stage 3/6] data quickcheck"
"$PYTHON_BIN" scripts/quickcheck/check_data.py --config "$CONFIG_PATH"

echo "[stage 4/6] fft quickcheck"
"$PYTHON_BIN" scripts/quickcheck/check_fft.py --config "$CONFIG_PATH"

echo "[stage 5/6] train quickcheck"
"$PYTHON_BIN" scripts/quickcheck/check_train.py --config "$CONFIG_PATH" --train-steps 3 --val-steps 1

EVAL_SPLITS="${EVAL_SPLITS:-}"
EVAL_LAYER_KEYS="${EVAL_LAYER_KEYS:-}"
TARGET_LAYER="${TARGET_LAYER:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

if [[ -z "$EVAL_SPLITS" ]]; then
	EVAL_SPLITS="$($PYTHON_BIN - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
ev = cfg.get("evaluation", {})
splits = ev.get("real_splits")
if not splits:
    splits = [ev.get("real_split", "validation")]
print(",".join(str(s) for s in splits))
PY
)"
fi

if [[ -z "$EVAL_LAYER_KEYS" ]]; then
	EVAL_LAYER_KEYS="$($PYTHON_BIN - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
ev = cfg.get("evaluation", {})
keys = ev.get("real_layer_keys")
if not keys:
    keys = [ev.get("real_layer_key", "layer_all")]
print(",".join(str(k) for k in keys))
PY
)"
fi

if [[ -z "$TARGET_LAYER" ]]; then
	TARGET_LAYER="$($PYTHON_BIN - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
ev = cfg.get("evaluation", {})
print(int(ev.get("target_layer", 1)))
PY
)"
fi

if [[ -z "$MAX_SAMPLES" ]]; then
	MAX_SAMPLES="$($PYTHON_BIN - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
ev = cfg.get("evaluation", {})
print(int(ev.get("real_max_samples", 20)))
PY
)"
fi

echo "[stage 6/6] real tuple eval quickcheck"
QC_CKPT="$ROOT_DIR/artifacts/quickcheck/checkpoints/quickcheck/sanity_roundtrip.pth"
"$PYTHON_BIN" scripts/eval/eval_real_tuples.py \
	--config "$CONFIG_PATH" \
	--checkpoint "$QC_CKPT" \
	--splits "$EVAL_SPLITS" \
	--layer-keys "$EVAL_LAYER_KEYS" \
	--target-layer "$TARGET_LAYER" \
	--max-samples "$MAX_SAMPLES" \
	--output "$ROOT_DIR/artifacts/quickcheck/reports/real_tuple_eval_quickcheck.json"

echo "[done] quickcheck pipeline passed for $CONFIG_PATH"
