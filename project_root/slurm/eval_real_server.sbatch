#!/bin/bash
#SBATCH --job-name=ld-eval-real
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

CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/server.yaml}"
EVAL_SPLITS="${EVAL_SPLITS:-validation,test}"
EVAL_LAYER_KEYS="${EVAL_LAYER_KEYS:-layer_all}"
MAX_SAMPLES_PER_SPLIT="${MAX_SAMPLES_PER_SPLIT:-0}"

REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/artifacts/reports}"
mkdir -p "$ROOT_DIR/logs" "$REPORT_DIR"
REPORT_PATH="${REPORT_PATH:-$REPORT_DIR/local_best_ckpt_eval_${SLURM_JOB_ID:-manual}.json}"

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

"$PYTHON_BIN" scripts/preflight_server.py --config "$CONFIG_PATH" --sbatch "$ROOT_DIR/slurm/eval_real_server.sbatch"
"$PYTHON_BIN" scripts/server_env_doctor.py --config "$CONFIG_PATH"

cmd=(
	"$PYTHON_BIN"
	"scripts/evaluate_best_checkpoint_local.py"
	"--config" "$CONFIG_PATH"
	"--splits" "$EVAL_SPLITS"
	"--layer-keys" "$EVAL_LAYER_KEYS"
	"--max-samples-per-split" "$MAX_SAMPLES_PER_SPLIT"
	"--output" "$REPORT_PATH"
)

if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
	cmd+=("--checkpoint" "$CHECKPOINT_PATH")
fi

if [[ -n "${CACHE_ROOT:-}" ]]; then
	cmd+=("--cache-root" "$CACHE_ROOT")
fi

if [[ -n "${TARGET_LAYER:-}" ]]; then
	cmd+=("--target-layer" "$TARGET_LAYER")
fi

printf '[cmd] '
printf '%q ' "${cmd[@]}"
printf '\n'
"${cmd[@]}"

echo "[summary] extracting key metrics from $REPORT_PATH"
"$PYTHON_BIN" - "$REPORT_PATH" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
	raise SystemExit(f"[FAIL] report not found: {path}")

report = json.loads(path.read_text(encoding="utf-8"))
print(f"[summary] checkpoint={report.get('checkpoint', '<unknown>')}")

for split_name, split_data in report.get("splits", {}).items():
	layer_data = split_data.get("per_layer_key", {}).get("layer_all")
	if not layer_data:
		print(f"[summary] split={split_name} layer_all unavailable")
		continue
	print(
		f"[summary] split={split_name} "
		f"samples={layer_data.get('samples_evaluated', 0)} "
		f"quads_acc={layer_data.get('quads_acc', 0.0):.4f} "
		f"all_acc={layer_data.get('all_acc', 0.0):.4f}"
	)

agg = report.get("aggregate", {})
print(f"[summary] aggregate quads_acc={agg.get('quads_acc', 0.0):.4f} all_acc={agg.get('all_acc', 0.0):.4f}")
print(f"[summary] official_quadruplet_accuracy={report.get('official_quadruplet_accuracy', 0.0):.4f}")
print(f"[summary] report={path}")
PY

echo "[OK] eval completed"