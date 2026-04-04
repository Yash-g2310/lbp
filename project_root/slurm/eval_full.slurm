#!/bin/bash
# SLURM job script: full real-tuple evaluation
# Adjust partition, time, and resources to your cluster policy.
# Submit with: sbatch slurm/eval_full.slurm

#SBATCH --job-name=ld-eval-full
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
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
