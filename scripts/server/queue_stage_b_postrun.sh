#!/bin/bash
set -euo pipefail

usage() {
	echo "Usage: $0 <current_train_job_id> [after_eval|after_train]" >&2
	echo "  after_eval (default): queue fine-tune after eval job succeeds" >&2
	echo "  after_train: queue fine-tune directly after train job succeeds" >&2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
	usage
	exit 2
fi

CURRENT_JOB_ID="$1"
if [[ ! "$CURRENT_JOB_ID" =~ ^[0-9]+$ ]]; then
	echo "[FAIL] current_train_job_id must be numeric, got: $CURRENT_JOB_ID" >&2
	exit 2
fi

FINETUNE_DEP_MODE="${2:-${FINETUNE_DEP_MODE:-after_eval}}"
ROOT_DIR="${ROOT_DIR:-$PWD}"

if [[ ! -f "$ROOT_DIR/cli.py" ]]; then
	echo "[FAIL] cli.py not found in ROOT_DIR=$ROOT_DIR" >&2
	exit 1
fi

EVAL_TEMPLATE="${EVAL_TEMPLATE:-$ROOT_DIR/slurm/templates/eval_real.sbatch}"
FINETUNE_TEMPLATE="${FINETUNE_TEMPLATE:-$ROOT_DIR/slurm/templates/train_stage_b_finetune.sbatch}"
if [[ ! -f "$EVAL_TEMPLATE" ]]; then
	echo "[FAIL] eval template not found: $EVAL_TEMPLATE" >&2
	exit 1
fi
if [[ ! -f "$FINETUNE_TEMPLATE" ]]; then
	echo "[FAIL] fine-tune template not found: $FINETUNE_TEMPLATE" >&2
	exit 1
fi

EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-$ROOT_DIR/configs/server/default.yaml}"
FINETUNE_CONFIG_PATH="${FINETUNE_CONFIG_PATH:-$ROOT_DIR/configs/server/stage_b_finetune.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/runs/current/checkpoints/best_checkpoint.pth}"
EVAL_REPORT_DIR="${EVAL_REPORT_DIR:-$ROOT_DIR/runs/current/reports/postrun}"

mkdir -p "$ROOT_DIR/runs/current/logs" "$EVAL_REPORT_DIR"
RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_PATH="${REPORT_PATH:-$EVAL_REPORT_DIR/real_tuple_eval_full_${CURRENT_JOB_ID}_${RUN_STAMP}.json}"

echo "[queue] current_train_job_id=$CURRENT_JOB_ID"
echo "[queue] eval config=$EVAL_CONFIG_PATH"
echo "[queue] fine-tune config=$FINETUNE_CONFIG_PATH"

EVAL_JOB_ID="$(
	sbatch --parsable \
		--dependency=afterok:${CURRENT_JOB_ID} \
		--job-name=ld-eval-real-full \
		--export=ALL,ROOT_DIR=${ROOT_DIR},CONFIG_PATH=${EVAL_CONFIG_PATH},CHECKPOINT_PATH=${CHECKPOINT_PATH},MAX_SAMPLES=0,REPORT_PATH=${REPORT_PATH} \
		"$EVAL_TEMPLATE"
)"

echo "[queue] submitted full real-eval job: $EVAL_JOB_ID (afterok:$CURRENT_JOB_ID)"
echo "[queue] full real-eval report will be written to: $REPORT_PATH"

case "$FINETUNE_DEP_MODE" in
	after_eval)
		FINETUNE_DEP_JOB_ID="$EVAL_JOB_ID"
		;;
	after_train)
		FINETUNE_DEP_JOB_ID="$CURRENT_JOB_ID"
		;;
	*)
		echo "[FAIL] unsupported dependency mode: $FINETUNE_DEP_MODE (expected after_eval or after_train)" >&2
		exit 2
		;;
esac

FINETUNE_JOB_ID="$(
	sbatch --parsable \
		--dependency=afterok:${FINETUNE_DEP_JOB_ID} \
		--job-name=ld-stageb-ft \
		--export=ALL,ROOT_DIR=${ROOT_DIR},CONFIG_PATH=${FINETUNE_CONFIG_PATH} \
		"$FINETUNE_TEMPLATE"
)"

echo "[queue] submitted Stage-B aux fine-tune job: $FINETUNE_JOB_ID (afterok:$FINETUNE_DEP_JOB_ID)"
echo "[queue] done"

echo "EVAL_JOB_ID=$EVAL_JOB_ID"
echo "FINETUNE_JOB_ID=$FINETUNE_JOB_ID"
echo "EVAL_REPORT_PATH=$REPORT_PATH"
