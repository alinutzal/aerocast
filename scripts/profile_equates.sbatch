#!/bin/bash
#SBATCH --job-name=aerocast-profile-equates
#SBATCH --partition=nextgen      # gpu/longgpu unavailable to this account right now
# No --account here: pass yours at submit time, e.g. `sbatch -A <account> scripts/profile_equates.sbatch`
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

export PYTHONUNBUFFERED=1   # stream print()s to the log immediately instead of buffering

# Profile all 7 models on the EQUATES pilot grid (72x72 read area, 12 km). Written to its own
# file: results/profile.csv holds the 224x164 BAAQMD timings, and profile_models.py replaces the
# rows of every model it profiles. Tables pick this up with
#   make_tables.py --cv --dataset equates_pilot --profile results/profile_equates.csv ...
cd "$SLURM_SUBMIT_DIR"   # submit from the repo root
uv run python scripts/profile_models.py \
  --config configs/data/equates_2019_07_ca12km_fold3.yaml \
  --out results/profile_equates.csv
