"""Aggregate the blocked cross-validation results (4 folds x 3 seeds x 7 models, ox target
mode) into mean +/- spread per model, plus a per-fold breakdown to see weather-regime
variation separately from seed variation.

Rows come from two naming schemes, both matched: convlstm/unet/fno's first CV pass was run
directly with `aerocast-train --config configs/data/equates_2019_07_ca12km_fold{N}.yaml`
(run_id contains "equates_2019_07_ca12km_fold{N}-{model}_"); the full 7-model sweep is run
with `scripts/run_benchmark.py --config configs/experiments/benchmark_equates_fold{N}.yaml`
(run_id contains "benchmark_equates_fold{N}-final-{model}-{mode}-s{seed}_"). Only the ox mode
is aggregated here - species/multitask are 3 more comparable numbers per model, not more
weather regimes, and mixing them into this fold/seed spread would conflate the two. The full
per-mode tables come from scripts/make_tables.py.

Usage: python scripts/summarize_equates_cv.py
       python scripts/summarize_equates_cv.py --run-date 20260929   # only that day's run_ids
                                                                     # (run_id is timestamped
                                                                     # YYYYMMDD-HHMMSS_...) -
                                                                     # use this to exclude an
                                                                     # earlier, differently-
                                                                     # tuned CV pass for the
                                                                     # same model/fold.
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "results/results.csv"
OUT_DIR = Path("data/pilot_checks")
# fold_old/mode_old: aerocast-train direct runs (ox only, no mode in the name).
# fold_new/mode_new: run_benchmark.py final runs (any of the 3 target modes).
RUN_RE = re.compile(
    r"equates_2019_07_ca12km_fold(?P<fold_old>\d)-[a-z_]+_"
    r"|benchmark_equates_fold(?P<fold_new>\d)-final-[a-z_]+-(?P<mode_new>ox|species|multitask)-s\d+_"
)
MODELS = ["convlstm", "unet", "fno", "swin", "swin_unet", "gnn", "mamba"]
METRICS = ["rmse", "mae", "bias", "pearson_r"]  # unprefixed = Ox, scored for every target mode


def parse_run(run_id):
    """(fold, mode) from a run_id matching either naming scheme, or None if it matches neither."""
    m = RUN_RE.search(run_id)
    if not m:
        return None
    if m.group("fold_old"):
        return int(m.group("fold_old")), "ox"
    return int(m.group("fold_new")), m.group("mode_new")


def load_cv_rows(run_date=None):
    """{(model, fold): {metric: [values across seeds]}}, ox target mode only. run_date (a
    "YYYYMMDD" prefix of run_id) restricts to runs started that day."""
    by_cell = defaultdict(lambda: defaultdict(list))
    with open(RESULTS, newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] != "equates_pilot" or row["split"] != "test" or row["lead_hour"] != "all":
                continue
            if row["model"] not in MODELS or row["metric"] not in METRICS:
                continue
            if run_date and not row["run_id"].startswith(run_date):
                continue
            parsed = parse_run(row["run_id"])
            if not parsed or parsed[1] != "ox":
                continue
            fold, _ = parsed
            by_cell[(row["model"], fold)][row["metric"]].append(float(row["value"]))
    return by_cell


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-date", default=None, help="only run_ids timestamped this day (YYYYMMDD)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_cell = load_cv_rows(args.run_date)
    models = [m for m in MODELS if any((m, fold) in by_cell for fold in range(1, 5))]
    if not models:
        scope = f" on {args.run_date}" if args.run_date else ""
        raise SystemExit(f"No blocked-CV rows found in {RESULTS}{scope} (ox mode, dataset equates_pilot).")

    suffix = f"_{args.run_date}" if args.run_date else ""
    scope_note = f", run_id timestamped {args.run_date} only" if args.run_date else ""
    report = [f"# EQUATES pilot: blocked cross-validation (4 folds x 3 seeds, ox target mode{scope_note})\n",
              "Each fold holds out a different week of July as `test` (with a 5-day `val` block "
              "and 1-day buffers separating train/val/test), so the number below is a mean +/- "
              "spread over four different weather regimes and three seeds (up to 12 runs per "
              "model), not one number from one episode. See configs/data/equates_2019_07_ca12km_"
              "fold{1..4}.yaml for the exact day blocks. Models with fewer than 4 folds present "
              "are still in progress.\n"]

    report.append("## Per-fold mean (over 3 seeds), test Ox RMSE (ppbV)\n")
    header = "| model | " + " | ".join(f"fold {k}" for k in range(1, 5)) + " | mean of folds | fold std |"
    report.append(header)
    report.append("|---" * 6 + "|")
    overall = {}
    for model in models:
        fold_means = []
        for fold in range(1, 5):
            vals = by_cell[(model, fold)].get("rmse", [])
            fold_means.append(np.mean(vals) if vals else float("nan"))
        overall[model] = fold_means
        row = f"| {model} | " + " | ".join(f"{v:.3f}" if not np.isnan(v) else "-" for v in fold_means) + \
              f" | {np.nanmean(fold_means):.3f} | {np.nanstd(fold_means):.3f} |"
        report.append(row)
        print(f"{model}: per-fold RMSE {[f'{v:.3f}' for v in fold_means]}, "
              f"mean {np.nanmean(fold_means):.3f} +/- {np.nanstd(fold_means):.3f} (fold std)")

    report.append("\n## All runs per model (up to 4 folds x 3 seeds), test Ox RMSE (ppbV)\n")
    report.append("| model | mean | std | min | max | n |")
    report.append("|---|---|---|---|---|---|")
    all_vals = {}
    for model in models:
        vals = [v for fold in range(1, 5) for v in by_cell[(model, fold)].get("rmse", [])]
        all_vals[model] = vals
        if not vals:
            continue
        report.append(f"| {model} | {np.mean(vals):.3f} | {np.std(vals):.3f} | {np.min(vals):.3f} | "
                       f"{np.max(vals):.3f} | {len(vals)} |")
        print(f"{model}: all-run RMSE mean {np.mean(vals):.3f} +/- {np.std(vals):.3f} (n={len(vals)})")

    for metric in ["mae", "bias", "pearson_r"]:
        report.append(f"\n## {metric} (mean +/- std over all runs per model)\n")
        report.append("| model | mean | std |")
        report.append("|---|---|---|")
        for model in models:
            vals = [v for fold in range(1, 5) for v in by_cell[(model, fold)].get(metric, [])]
            if not vals:
                continue
            report.append(f"| {model} | {np.mean(vals):.3f} | {np.std(vals):.3f} |")

    # --- figure: RMSE per fold per model, with seed spread ---------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        xs = np.arange(1, 5) + (i - (len(models) - 1) / 2) * width
        means = [np.mean(by_cell[(model, f)].get("rmse", [np.nan])) for f in range(1, 5)]
        stds = [np.std(by_cell[(model, f)].get("rmse", [np.nan])) for f in range(1, 5)]
        ax.errorbar(xs, means, yerr=stds, fmt="o", capsize=4, label=model, color=cmap(i % 10))
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("fold (held-out week)")
    ax.set_ylabel("test Ox RMSE (ppbV)")
    ax.set_title("Blocked CV: test RMSE per fold (points = mean over seeds, bars = seed std)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_name = f"cv_rmse_by_fold{suffix}.png"
    fig.savefig(OUT_DIR / fig_name, dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR / fig_name}")
    report.append(f"\n## RMSE by fold, all models\n\n![cv rmse by fold]({fig_name})\n")

    summary_name = f"cv_summary{suffix}.md"
    (OUT_DIR / summary_name).write_text("\n".join(report))
    print(f"\nWrote {OUT_DIR / summary_name}")


if __name__ == "__main__":
    main()
