#!/bin/bash
#SBATCH --job-name=aerocast-bench-cv-convlstm
#SBATCH --partition=nextgen      # gpu/longgpu unavailable to this account right now
# No --account here: pass yours at submit time, e.g. `sbatch -A <account> scripts/benchmark_array_convlstm.sbatch`
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --array=0-3               # 4 folds, convlstm only
#SBATCH --output=logs/%x-%a.out

export PYTHONUNBUFFERED=1   # stream print()s to the log immediately instead of buffering

# ConvLSTM only, run separately from scripts/benchmark_array.sbatch's other 6 models - one
# task per fold. --model makes run_benchmark.py resumable per fold, so a fold that already
# has results (see EXPERIMENTS.md) just skips its finished runs.
FOLDS=(1 2 3 4)
fold=${FOLDS[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
uv run python scripts/run_benchmark.py \
  --config "configs/experiments/benchmark_equates_fold${fold}.yaml" \
  --model convlstm
