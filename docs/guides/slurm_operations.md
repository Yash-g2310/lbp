# Slurm Operations

This guide maps standard cluster workflow to project commands.

## Policy Constraints

1. One GPU per job.
2. Maximum 40 CPU threads per job.
3. Maximum memory allocation of 4 GB per requested CPU.
4. Maximum walltime 24 hours.

## Validation Before Submit

```bash
cd /home/yash_gupta/lbp/lbp
python scripts/server/preflight.py --config configs/server/default.yaml --sbatch slurm/templates/train.sbatch
python scripts/server/check_env.py --config configs/server/default.yaml
```

## Typical Submit Flow

```bash
cd /home/yash_gupta/lbp/lbp
bash slurm/submit.sh quickcheck configs/server/quickcheck.yaml
bash slurm/submit.sh train configs/server/default.yaml
bash slurm/submit.sh eval configs/server/default.yaml /path/to/checkpoint.pth
```

## Operational Commands

```bash
squeue -u <username>
sinfo
scancel <job_id>
```

## Template Defaults

- `slurm/templates/train.sbatch`: `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=32G`, `--time=23:55:00`
- `slurm/templates/quickcheck.sbatch`: `--gres=gpu:1`, `--cpus-per-task=4`, `--mem=16G`, `--time=00:20:00`
- `slurm/templates/eval_real.sbatch`: `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=32G`, `--time=23:55:00`

## Runtime Output Paths

- Slurm stdout/stderr defaults to `runs/current/logs/`.
- Checkpoint defaults are under `runs/current/checkpoints/`.
- Evaluation reports default to `runs/current/reports/`.
