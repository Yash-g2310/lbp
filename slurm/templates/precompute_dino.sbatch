#!/bin/bash
#SBATCH --job-name=ld-precompute-dino
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
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
PY_ENV_ROOT="$(cd "$(dirname "$PYTHON_BIN")/.." && pwd)"
if [[ -f "$PY_ENV_ROOT/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="$PY_ENV_ROOT/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/server/default.yaml}"

if [[ -z "${FEATURES_DIR:-}" ]]; then
    FEATURES_DIR="$($PYTHON_BIN - "$CONFIG_PATH" <<'PY'
import os
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
idx = str(cfg["data"]["precomputed_index_path"]).strip()
if not idx:
    raise SystemExit("[FAIL] data.precomputed_index_path is empty")
print(os.path.dirname(idx))
PY
)"
fi

SHARD_SIZE="${SHARD_SIZE:-2048}"
LOG_EVERY="${LOG_EVERY:-200}"
KEEP_OLD_FEATURE_BACKUP="${KEEP_OLD_FEATURE_BACKUP:-0}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TMP_DIR="${FEATURES_DIR}.rebuild-${TS}"
BACKUP_DIR="${FEATURES_DIR}.bak-${TS}"
SWAPPED="0"

rollback_if_needed() {
    status=$?
    if [[ $status -ne 0 && "$SWAPPED" == "1" && -d "$BACKUP_DIR" ]]; then
        echo "[rollback] restoring previous feature directory from $BACKUP_DIR"
        rm -rf "$FEATURES_DIR"
        mv "$BACKUP_DIR" "$FEATURES_DIR"
    fi
    exit $status
}
trap rollback_if_needed ERR

mkdir -p "$ROOT_DIR/runs/current/logs"
cd "$ROOT_DIR"

echo "[env] Using PYTHON_BIN=$PYTHON_BIN"
echo "[env] LD_PRELOAD=${LD_PRELOAD:-<empty>}"
echo "[env] CONFIG_PATH=$CONFIG_PATH"
echo "[env] FEATURES_DIR=$FEATURES_DIR"
echo "[env] TMP_DIR=$TMP_DIR"

mkdir -p "$TMP_DIR"

echo "[step] precomputing ConvNeXt-distilled DINOv3 features"
"$PYTHON_BIN" src/lbp_project/data/precompute_dino.py \
    --config "$CONFIG_PATH" \
    --output_dir "$TMP_DIR" \
    --shard_size "$SHARD_SIZE" \
    --log_every "$LOG_EVERY"

TMP_INDEX="$TMP_DIR/index.json"
if [[ ! -f "$TMP_INDEX" ]]; then
    echo "[FAIL] expected index not found after precompute: $TMP_INDEX" >&2
    exit 1
fi

echo "[step] strict compatibility check for rebuilt index"
"$PYTHON_BIN" - "$CONFIG_PATH" "$TMP_INDEX" <<'PY'
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve().parents[2]
src_root = project_root / 'src'
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from lbp_project.config.io import load_yaml
from lbp_project.data.preflight import validate_precomputed_index_compatibility

cfg = load_yaml(sys.argv[1])
index_path = sys.argv[2]
warns = validate_precomputed_index_compatibility(cfg, strict=True, index_path=index_path)
print(f"[OK] strict compatibility passed for {index_path}; warnings={len(warns)}")
PY

if [[ -d "$FEATURES_DIR" ]]; then
    echo "[step] backing up current feature dir to $BACKUP_DIR"
    mv "$FEATURES_DIR" "$BACKUP_DIR"
fi

mv "$TMP_DIR" "$FEATURES_DIR"
SWAPPED="1"

echo "[step] normalizing index shard paths to active feature directory"
"$PYTHON_BIN" - "$FEATURES_DIR" "$TMP_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

features_dir = Path(sys.argv[1]).resolve()
old_prefix = str(Path(sys.argv[2]).resolve())
index_path = features_dir / "index.json"

payload = json.loads(index_path.read_text(encoding="utf-8"))
samples = payload.get("samples", {})
if not isinstance(samples, dict):
    raise SystemExit(f"[FAIL] invalid index format at {index_path}: missing samples mapping")

rewritten = 0
for split_map in samples.values():
    if not isinstance(split_map, dict):
        continue
    for meta in split_map.values():
        if not isinstance(meta, dict):
            continue
        shard_path = str(meta.get("shard_path", ""))
        if not shard_path:
            continue
        if shard_path.startswith(old_prefix + os.sep):
            rel = shard_path[len(old_prefix + os.sep):]
            meta["shard_path"] = str(features_dir / rel)
            rewritten += 1

index_path.write_text(json.dumps(payload), encoding="utf-8")
print(f"[OK] normalized index shard paths in {index_path}; rewritten={rewritten}")
PY

echo "[step] validating dataset/index coverage on active index path"
"$PYTHON_BIN" scripts/config/validate_dataset.py \
    --config "$CONFIG_PATH" \
    --verify-index-coverage \
    --progress-every 5000

if [[ "$KEEP_OLD_FEATURE_BACKUP" != "1" && -d "$BACKUP_DIR" ]]; then
    echo "[step] removing backup directory to reclaim disk: $BACKUP_DIR"
    rm -rf "$BACKUP_DIR"
fi

trap - ERR
echo "[OK] rebuilt feature index is active at $FEATURES_DIR"
if [[ "$KEEP_OLD_FEATURE_BACKUP" == "1" && -d "$BACKUP_DIR" ]]; then
    echo "[info] backup retained at $BACKUP_DIR"
fi
