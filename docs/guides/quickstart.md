# Quickstart

Use this guide for first-time local setup and first validation run.

## Prerequisites

1. Python environment with project dependencies.
2. Working CUDA setup for GPU training.
3. Access to HuggingFace datasets if running full data pulls.

## Local Setup

```bash
cd /home/yash_gupta/lbp/lbp
conda activate monodepth
```

## First Validation Run

Run the project quickcheck pipeline:

```bash
python cli.py quickcheck --config configs/local/quickcheck.yaml
```

## First Local Train+Eval Run

```bash
python cli.py train-eval --config configs/local/dev.yaml
```

## Where To Go Next

- Quickcheck details: `guides/quickcheck_guide.md`
- Slurm execution: `guides/slurm_operations.md`
- Data split and indexing details: `reference/data_configuration.md`
