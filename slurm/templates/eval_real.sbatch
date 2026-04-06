#!/bin/bash
#SBATCH --job-name=ld-eval-real
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
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

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/server/default.yaml}"
EVAL_SPLITS="${EVAL_SPLITS:-}"
EVAL_LAYER_KEYS="${EVAL_LAYER_KEYS:-}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/runs/current/checkpoints/best_checkpoint.pth}"

REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/runs/current/reports}"
mkdir -p "$ROOT_DIR/runs/current/logs" "$REPORT_DIR"
REPORT_PATH="${REPORT_PATH:-$REPORT_DIR/real_tuple_eval_${SLURM_JOB_ID:-manual}.json}"

cd "$ROOT_DIR"

echo "[env] Using PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import sys
mods = ("numpy", "torch", "yaml", "datasets", "torchvision")
missing = []
for m in mods:
	try:
		__import__(m)
	except Exception:
		missing.append(m)
if missing:
	raise SystemExit(f"[FAIL] Python env missing required packages: {', '.join(missing)}; executable={sys.executable}")
print(f"[env] python={sys.executable}")
PY

"$PYTHON_BIN" scripts/server/preflight.py --config "$CONFIG_PATH" --sbatch "$ROOT_DIR/slurm/templates/eval_real.sbatch"
"$PYTHON_BIN" scripts/server/check_env.py --config "$CONFIG_PATH"

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

if [[ -z "$MAX_SAMPLES" ]]; then
	MAX_SAMPLES="$($PYTHON_BIN - "$CONFIG_PATH" <<'PY'
import sys
import yaml

cfg = yaml.safe_load(open(sys.argv[1], "r", encoding="utf-8"))
ev = cfg.get("evaluation", {})
print(int(ev.get("real_max_samples", 0)))
PY
)"
fi

cmd=(
	"$PYTHON_BIN"
	"cli.py"
	"eval-real"
	"--config" "$CONFIG_PATH"
	"--checkpoint" "$CHECKPOINT_PATH"
	"--splits" "$EVAL_SPLITS"
	"--layer-keys" "$EVAL_LAYER_KEYS"
	"--max-samples" "$MAX_SAMPLES"
	"--output" "$REPORT_PATH"
)

if [[ -n "${TARGET_LAYER:-}" ]]; then
	cmd+=("--target-layer" "$TARGET_LAYER")
fi

printf '[cmd] '
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo "[summary] extracting key metrics from $REPORT_PATH"
"$PYTHON_BIN" - "$REPORT_PATH" "$CHECKPOINT_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
checkpoint = sys.argv[2]
if not path.exists():
	raise SystemExit(f"[FAIL] report not found: {path}")

report = json.loads(path.read_text(encoding="utf-8"))
print(f"[summary] checkpoint={checkpoint}")

per_eval = report.get("per_eval", {})
if not per_eval:
	print("[summary] per_eval unavailable")
for eval_key, metrics in per_eval.items():
	print(
		f"[summary] eval={eval_key} "
		f"samples={metrics.get('samples_evaluated', 0)} "
		f"quads_acc={metrics.get('quads_acc', 0.0):.4f} "
		f"all_acc={metrics.get('all_acc', 0.0):.4f}"
	)

agg = report.get("aggregate", {})
print(f"[summary] aggregate quads_acc={agg.get('quads_acc', 0.0):.4f} all_acc={agg.get('all_acc', 0.0):.4f}")
print(f"[summary] official_quadruplet_accuracy={report.get('official_quadruplet_accuracy', 0.0):.4f}")
print(f"[summary] report={path}")
PY

echo "[OK] eval completed"