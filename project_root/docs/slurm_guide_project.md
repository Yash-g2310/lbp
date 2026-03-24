# Slurm Mapping for This Project

This document maps your cluster policy to project settings.

## Cluster constraints to enforce

1. 1 GPU per job.
2. Maximum 40 CPU threads per job.
3. Maximum memory allocation of 4 GB per requested CPU.
4. Maximum walltime 24 hours.

## Project enforcement points

- `scripts/preflight_server.py` validates:
  - config server requirements (`cuda`, precomputed dino index path, staged root),
  - sbatch policy (`--gres=gpu:1`, `--cpus-per-task<=40`, `--time<=24:00:00`, memory cap check).
- `scripts/server_env_doctor.py` checks:
  - W&B auth (`WANDB_API_KEY` or `~/.netrc`),
  - optional HF auth (`HF_TOKEN` or cache token),
  - write access for cache paths,
  - staged root and precomputed index visibility.

- `slurm/train_server.sbatch` defaults:
  - `--gres=gpu:1`
  - `--cpus-per-task=8`
  - `--mem=32G`
  - `--time=23:55:00`

## Typical workflow

```bash
cd /home/yash_gupta/lbp/project_root
bash slurm/submit_server.sh quickcheck configs/quickcheck_server.yaml
bash slurm/submit_server.sh train configs/server.yaml
```

## How to verify server storage paths

```bash
echo "$USER"
ls -ld /scratch /mnt/home2/home/$USER
mkdir -p /mnt/home2/home/$USER/tmp_write_test && rmdir /mnt/home2/home/$USER/tmp_write_test
test -d /scratch && mkdir -p /scratch/$USER/tmp_write_test && rmdir /scratch/$USER/tmp_write_test || true
```

## Operational commands

```bash
squeue -u <username>
sinfo
scancel <job_id>
```

## Storage note

Keep quickcheck artifacts under `artifacts/quickcheck` to avoid mixing with production outputs.
