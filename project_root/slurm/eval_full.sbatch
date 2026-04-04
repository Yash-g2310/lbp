#!/bin/bash
# SLURM batch script: full real-tuple evaluation
# Rename/adjust SBATCH directives to match your cluster if needed.
# Submit with: sbatch slurm/eval_full.sbatch

#SBATCH --job-name=ld-eval-full
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=23:55:00
#SBATCH --output=logs/eval_full_%j.out
#SBATCH --error=logs/eval_full_%j.err

set -euo pipefail

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

PYTHON=/mnt/home2/home/yash_g/.conda/envs/monodepth/bin/python
SCRIPT=scripts/evaluate_real_tuples.py
CONFIG=configs/server.yaml
CKPT=checkpoints/best_checkpoint.pth
OUT=artifacts/reports/real_tuple_eval_latest.json

mkdir -p logs artifacts/reports

echo "Starting full real-tuple evaluation"
echo "Command: $PYTHON $SCRIPT --config $CONFIG --checkpoint $CKPT --output $OUT"

${PYTHON} ${SCRIPT} --config ${CONFIG} --checkpoint ${CKPT} --output ${OUT}

echo "Evaluation finished; report written to ${OUT}"
