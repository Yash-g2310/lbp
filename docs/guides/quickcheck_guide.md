# Quickcheck Guide

Quickcheck validates the end-to-end training stack quickly (target 2-3 minutes locally, longer on remote queues).

## What It Checks

1. YAML schema sanity and required keys.
2. Dataset split loading and sample/key visibility.
3. Optional precomputed-feature index and shard-path checks.
4. Data batch shapes and required tensors.
5. FFT path stability check.
6. Short train/val sanity with checkpoint roundtrip.
7. Real tuple smoke evaluation.

## Local Run

```bash
cd /home/yash_gupta/lbp/lbp
python cli.py quickcheck --config configs/local/quickcheck.yaml
```

## Server Dry Validation (No Submit)

```bash
cd /home/yash_gupta/lbp/lbp
python scripts/server/preflight.py --config configs/server/quickcheck.yaml --sbatch slurm/templates/quickcheck.sbatch
```

## Server Environment Bootstrap

```bash
cd /home/yash_gupta/lbp/lbp
bash scripts/server/bootstrap_env.sh layered_depth
```

## Server Submit

```bash
cd /home/yash_gupta/lbp/lbp
bash slurm/submit.sh quickcheck configs/server/quickcheck.yaml
bash slurm/submit.sh train configs/server/default.yaml
```

## Notes

- Quickcheck derives real-eval splits and layer keys from config by default.
- Override quickcheck real-eval behavior with env vars when needed:
  - `EVAL_SPLITS`
  - `EVAL_LAYER_KEYS`
  - `TARGET_LAYER`
  - `MAX_SAMPLES`
- Quickcheck configs use `logging.mode: dryrun` by default.
