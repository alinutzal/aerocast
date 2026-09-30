#!/bin/bash
#SBATCH --job-name=aerocast-bench-cv
#SBATCH --partition=nextgen      # gpu/longgpu unavailable to this account right now
# No --account here: pass yours at submit time, e.g. `sbatch -A <account> scripts/benchmark_array.sbatch`
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --array=0-23%4            # 4 folds x 6 models (convlstm excluded); %4 caps it at 4 GPUs running at once
#SBATCH --output=logs/%x-%a.out

export PYTHONUNBUFFERED=1   # stream print()s to the log immediately instead of buffering

# Controlled 7-model benchmark (configs/experiments/benchmark_equates_fold{1,2,3,4}.yaml),
# run per fold x model (28 tasks) so several GPUs can work in parallel instead of one GPU
# doing it serially; the array's %4 cap above controls how many run at once. Each task runs
# that model's tuning trials, then the final (target-mode x seed) runs, for one fold.
# --model makes run_benchmark.py resumable per (fold, model), same as before.
FOLDS=(1 2 3 4)
MODELS=(unet fno swin swin_unet gnn mamba)   # convlstm excluded
n_models=${#MODELS[@]}
fold=${FOLDS[$(( SLURM_ARRAY_TASK_ID / n_models ))]}
model=${MODELS[$(( SLURM_ARRAY_TASK_ID % n_models ))]}

cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
uv run python scripts/run_benchmark.py \
  --config "configs/experiments/benchmark_equates_fold${fold}.yaml" \
  --model "$model"
