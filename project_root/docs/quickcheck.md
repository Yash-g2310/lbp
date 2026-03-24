# Quickcheck Runbook

This runbook validates the full training pipeline quickly (target 2-3 minutes) without modifying the main local/server training flow.

## What it checks

1. YAML schema sanity and required keys.
2. Dataset split loading and sample/key visibility.
3. Optional precomputed-feature index and shard-path checks.
4. Data batch shapes and required tensors.
5. Model forward/backward for 6-head loss.
6. Optimizer/scheduler/grad-clip and checkpoint roundtrip.
7. Real benchmark tuple evaluation smoke check (pairs/trips/quads, including quadruplet accuracy).

## Local run

```bash
cd /home/yash_gupta/lbp/project_root
PYTHON_BIN=/home/yash_gupta/miniconda3/envs/monodepth/bin/python bash scripts/run_quickcheck.sh configs/quickcheck_local.yaml
```

## Server dry validation (no sbatch submit)

```bash
cd /home/yash_gupta/lbp/project_root
/home/yash_gupta/miniconda3/envs/monodepth/bin/python scripts/preflight_server.py --config configs/quickcheck_server.yaml --sbatch slurm/quickcheck.sbatch
```

## Server environment bootstrap

If your server is missing `wandb`, `huggingface-cli`, or `datasets`, run:

```bash
cd /path/to/project_root
bash scripts/setup_server_env.sh layered_depth
```

## Server submit

```bash
cd /home/yash_gupta/lbp/project_root
bash slurm/submit_server.sh quickcheck configs/quickcheck_server.yaml
bash slurm/submit_server.sh train configs/server.yaml
```

## Authentication setup

W&B (recommended for server training logs):

```bash
wandb login
```

Hugging Face (only needed if you use private datasets/models):

```bash
huggingface-cli login
```

Token-through-env alternative (non-interactive jobs):

```bash
export WANDB_API_KEY=<your_key>
export HF_TOKEN=<your_hf_token>
```

## Data-source clarity

	- `princeton-vl/LayeredDepth-Syn:train`
	- `princeton-vl/LayeredDepth-Syn:validation`
	- `princeton-vl/LayeredDepth:validation`
	- `princeton-vl/LayeredDepth:test`
	- and report both `layer_all` and `layer_first` views.

## Notes

- Quickcheck configs use `logging.mode: dryrun` to avoid polluting online W&B dashboards.
- Server quickcheck requires `use_precomputed_dino: true` and a valid `precomputed_index_path`.
- If you want online W&B for quickcheck, set mode to `online` in quickcheck config.
- Full server training now uses a train+eval wrapper that writes final reports to `artifacts/reports/final_report.json`.
