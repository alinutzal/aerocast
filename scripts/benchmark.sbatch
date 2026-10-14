#!/bin/bash
#SBATCH --job-name=aerocast-bench
#SBATCH --partition=nextgen      # gpu/longgpu unavailable to this account right now
# No --account here: pass yours at submit time, e.g. `sbatch -A <account> scripts/benchmark.sbatch`
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00        # nextgen's max
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

export PYTHONUNBUFFERED=1   # stream print()s to the log immediately instead of buffering

cd "$SLURM_SUBMIT_DIR"   # submit from the repo root

# run_benchmark.py's own default (configs/experiments/benchmark.yaml) is BAAQMD, and only one
# real BAAQMD day is on disk here - it will always hit "No windows for split(s) ['val', 'test']".
# Require --config explicitly so a bare `sbatch scripts/benchmark.sbatch` fails fast instead.
case " $* " in
  *" --config "*) ;;
  *) echo "usage: sbatch scripts/benchmark.sbatch --config configs/experiments/benchmark_equates_fold{1,2,3,4}.yaml [--model NAME ...]" >&2
     exit 1 ;;
esac

config=""
models=()
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  case "${args[i]}" in
    --config) config="${args[i+1]}" ;;
    --model) models+=("${args[i+1]}") ;;
  esac
done
scope="$(basename "${config%.yaml}")-${models[*]:-all}"
scope="${scope// /_}"
uv run python scripts/job_storage.py --config "$config" --scope "$scope" -- \
  python scripts/run_benchmark.py "$@"
