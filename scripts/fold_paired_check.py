"""Paired Ox RMSE differences per blocked-CV fold, as a check on the day-block bootstrap.

make_tables.py --cv resamples the 28 pooled test days (4 folds x 7 days). Days within a week
are autocorrelated, so that bootstrap can be narrower than the real uncertainty. This script
gives the per-fold view: for each model, the seed-mean Ox RMSE on each fold's test week minus
the reference model's on the same week (same days, so paired). Four folds give four paired
differences; their mean and sd (t-interval, 3 df) are a wider check on the day-bootstrap CI.

It reads results/results.csv (for the run ids) and the run directories under runs/, so it
needs the same inputs as make_tables.py --cv. Scores ox target mode only.

    uv run python scripts/fold_paired_check.py --reference gnn
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tables import LABELS, MODES, Runs, bootstrap_counts  # noqa: E402

from aerocast.config import load_config  # noqa: E402
from aerocast.metrics import F  # noqa: E402

T_975_3DF = 3.182  # two-sided 95% t quantile, 3 degrees of freedom (4 folds)
FOLD_CONFIGS = "configs/data/equates_2019_07_ca12km_fold{}.yaml"


def fold_test_starts():
    """First test day of each fold, from the fold data configs."""
    return {f: str(load_config(FOLD_CONFIGS.format(f))["split"]["test"][0]) for f in range(1, 5)}


def fold_rmse(runs, source, starts):
    """Seed-mean Ox RMSE on each fold's test week: {fold: rmse}."""
    ends = list(starts.values())[1:] + ["9999-12-31"]
    out = {}
    per_seed_vals = {f: [] for f in starts}
    for days, arr in runs.daily(source, "ox"):
        for fold, start, end in zip(starts, starts.values(), ends):
            idx = [i for i, d in enumerate(days) if start <= d < end]
            sums = arr[idx].sum(axis=(0, 1))  # features summed over the fold's days and leads
            per_seed_vals[fold].append(np.sqrt(sums[F["ee"]] / sums[F["n"]]))
    for fold, vals in per_seed_vals.items():
        out[fold] = float(np.mean(vals))
    return out


def day_bootstrap_ci(runs, source, reference, n_samples=1000, seed=0):
    """95% CI of the seed-mean RMSE difference from the day-block bootstrap (as in make_tables).
    source and reference are (model, mode) pairs."""
    days = runs.daily(*source)[0][0]
    counts = bootstrap_counts(len(days), n_samples, seed)

    def rmse(parts):
        values = []
        for _, arr in parts:
            sums = np.einsum("bd,dlf->blf", counts, arr).sum(axis=1)
            values.append(np.sqrt(sums[:, F["ee"]] / sums[:, F["n"]]))
        return np.mean(values, axis=0)

    diff = rmse(runs.daily(*source)) - rmse(runs.daily(*reference))
    return tuple(np.percentile(diff, [2.5, 97.5]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--reference", default="gnn", help="model the differences are against (Ox mode)")
    args = parser.parse_args()

    runs = Runs(args.results, args.runs_dir, dataset="equates_pilot", cv=True)
    starts = fold_test_starts()
    ref = fold_rmse(runs, args.reference, starts)
    ref_key = (args.reference, "ox")

    print(f"Ox RMSE difference vs {LABELS.get(args.reference, args.reference)} (ppbV), paired on each fold's test week")
    print(f"{'model':<12} " + " ".join(f"fold{f:>2}" for f in starts)
          + f" {'mean':>7} {'t-CI 95%':>17} {'day-boot CI':>17}")
    for source in dict.fromkeys(m for m, mode in runs.groups if mode in MODES):
        if source == args.reference or (source, "ox") not in runs.groups:
            continue
        fold = fold_rmse(runs, source, starts)
        diffs = np.array([fold[f] - ref[f] for f in starts])
        half = T_975_3DF * diffs.std(ddof=1) / np.sqrt(len(diffs))
        lo, hi = day_bootstrap_ci(runs, (source, "ox"), ref_key)
        print(f"{LABELS.get(source, source):<12} " + " ".join(f"{d:+6.2f}" for d in diffs)
              + f" {diffs.mean():+7.2f} [{diffs.mean() - half:+6.2f}, {diffs.mean() + half:+6.2f}]"
              + f" [{lo:+6.2f}, {hi:+6.2f}]")


if __name__ == "__main__":
    main()
