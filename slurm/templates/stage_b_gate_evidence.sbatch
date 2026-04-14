#!/bin/bash
#SBATCH --job-name=ld-stageb-gate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
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
GATE_REPORT_DIR="${GATE_REPORT_DIR:-$ROOT_DIR/runs/current/reports/stage_b_gate}"
MAX_SAMPLES="${GATE_MAX_SAMPLES:-200}"

mkdir -p "$ROOT_DIR/runs/current/logs"
mkdir -p "$GATE_REPORT_DIR"
cd "$ROOT_DIR"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
if [[ -z "$CHECKPOINT_PATH" ]]; then
    if [[ -f "$ROOT_DIR/runs/current/checkpoints/best_checkpoint.pth" ]]; then
        CHECKPOINT_PATH="$ROOT_DIR/runs/current/checkpoints/best_checkpoint.pth"
    elif [[ -f "$ROOT_DIR/runs/current/checkpoints/latest_checkpoint.pth" ]]; then
        CHECKPOINT_PATH="$ROOT_DIR/runs/current/checkpoints/latest_checkpoint.pth"
    else
        echo "[FAIL] No checkpoint found for gate evidence generation" >&2
        exit 1
    fi
fi

echo "[env] Using PYTHON_BIN=$PYTHON_BIN"
echo "[env] LD_PRELOAD=${LD_PRELOAD:-<empty>}"
echo "[env] CONFIG_PATH=$CONFIG_PATH"
echo "[env] CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "[env] GATE_REPORT_DIR=$GATE_REPORT_DIR"
echo "[env] GATE_MAX_SAMPLES=$MAX_SAMPLES"

REPORT1="$GATE_REPORT_DIR/real_tuple_eval_epoch_1.json"
REPORT2="$GATE_REPORT_DIR/real_tuple_eval_epoch_2.json"
FINAL="$GATE_REPORT_DIR/final_report.json"

echo "[step] generating stage-b gate evidence report epoch_1"
"$PYTHON_BIN" scripts/eval/eval_real_tuples.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --splits validation,test \
    --layer-keys layer_all,layer_first \
    --target-layer 1 \
    --max-samples "$MAX_SAMPLES" \
    --output "$REPORT1"

echo "[step] generating stage-b gate evidence report epoch_2"
"$PYTHON_BIN" scripts/eval/eval_real_tuples.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --splits validation,test \
    --layer-keys layer_all,layer_first \
    --target-layer 1 \
    --max-samples "$MAX_SAMPLES" \
    --output "$REPORT2"

cp "$REPORT2" "$FINAL"

echo "[step] validating stage-b gate on generated evidence"
"$PYTHON_BIN" - "$CONFIG_PATH" <<'PY'
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve().parents[2]
src_root = project_root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from lbp_project.config.io import load_yaml
from lbp_project.stage_gate import evaluate_stage_b_gate, format_stage_b_gate

cfg = load_yaml(sys.argv[1])
result = evaluate_stage_b_gate(cfg)
print(format_stage_b_gate(result))
if not result.passed:
    raise SystemExit("[FAIL] stage-b gate failed on generated evidence")
print("[OK] stage-b gate passed on generated evidence")
PY

echo "[OK] stage-b gate evidence ready"
