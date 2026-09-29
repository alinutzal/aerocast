"""Aggregate the blocked cross-validation results (4 folds x 3 seeds x {convlstm, unet, fno})
into mean +/- spread per model, plus a per-fold breakdown to see weather-regime variation
separately from seed variation.

Usage: python scripts/summarize_equates_cv.py
"""
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
RUN_RE = re.compile(r"equates_2019_07_ca12km_fold(\d)-([a-z_]+)_")
MODELS = ["convlstm", "unet", "fno"]
METRICS = ["rmse", "mae", "bias", "pearson_r"]


def load_cv_rows():
    """{(model, fold): {metric: [values across seeds]}}"""
    by_cell = defaultdict(lambda: defaultdict(list))
    with open(RESULTS, newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"] != "equates_pilot" or row["split"] != "test" or row["lead_hour"] != "all":
                continue
            m = RUN_RE.search(row["run_id"])
            if not m or row["metric"] not in METRICS:
                continue
            fold, model = int(m.group(1)), m.group(2)
            if model != row["model"] or model not in MODELS:
                continue
            by_cell[(model, fold)][row["metric"]].append(float(row["value"]))
    return by_cell


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_cell = load_cv_rows()

    report = ["# EQUATES pilot: blocked cross-validation (4 folds x 3 seeds)\n",
              "Each fold holds out a different week of July as `test` (with a 5-day `val` block "
              "and 1-day buffers separating train/val/test), so the number below is a mean +/- "
              "spread over four different weather regimes and three seeds (12 runs per model), "
              "not one number from one episode. See configs/data/equates_2019_07_ca12km_fold{1..4}.yaml "
              "for the exact day blocks.\n"]

    report.append("## Per-fold mean (over 3 seeds), test Ox RMSE (ppbV)\n")
    header = "| model | " + " | ".join(f"fold {k}" for k in range(1, 5)) + " | mean of folds | fold std |"
    report.append(header)
    report.append("|---" * 6 + "|")
    overall = {}
    for model in MODELS:
        fold_means = []
        for fold in range(1, 5):
            vals = by_cell[(model, fold)].get("rmse", [])
            fold_means.append(np.mean(vals) if vals else float("nan"))
        overall[model] = fold_means
        row = f"| {model} | " + " | ".join(f"{v:.3f}" for v in fold_means) + \
              f" | {np.nanmean(fold_means):.3f} | {np.nanstd(fold_means):.3f} |"
        report.append(row)
        print(f"{model}: per-fold RMSE {[f'{v:.3f}' for v in fold_means]}, "
              f"mean {np.nanmean(fold_means):.3f} +/- {np.nanstd(fold_means):.3f} (fold std)")

    report.append("\n## All 12 runs per model (4 folds x 3 seeds), test Ox RMSE (ppbV)\n")
    report.append("| model | mean | std | min | max | n |")
    report.append("|---|---|---|---|---|---|")
    all_vals = {}
    for model in MODELS:
        vals = [v for fold in range(1, 5) for v in by_cell[(model, fold)].get("rmse", [])]
        all_vals[model] = vals
        report.append(f"| {model} | {np.mean(vals):.3f} | {np.std(vals):.3f} | {np.min(vals):.3f} | "
                       f"{np.max(vals):.3f} | {len(vals)} |")
        print(f"{model}: all-run RMSE mean {np.mean(vals):.3f} +/- {np.std(vals):.3f} (n={len(vals)})")

    for metric in ["mae", "bias", "pearson_r"]:
        report.append(f"\n## {metric} (mean +/- std over all 12 runs per model)\n")
        report.append("| model | mean | std |")
        report.append("|---|---|---|")
        for model in MODELS:
            vals = [v for fold in range(1, 5) for v in by_cell[(model, fold)].get(metric, [])]
            report.append(f"| {model} | {np.mean(vals):.3f} | {np.std(vals):.3f} |")

    # --- figure: RMSE per fold per model, with seed spread ---------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"convlstm": "tab:blue", "unet": "tab:orange", "fno": "tab:green"}
    width = 0.25
    for i, model in enumerate(MODELS):
        xs = np.arange(1, 5) + (i - 1) * width
        means = [np.mean(by_cell[(model, f)].get("rmse", [np.nan])) for f in range(1, 5)]
        stds = [np.std(by_cell[(model, f)].get("rmse", [np.nan])) for f in range(1, 5)]
        ax.errorbar(xs, means, yerr=stds, fmt="o", capsize=4, label=model, color=colors[model])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("fold (held-out week)")
    ax.set_ylabel("test Ox RMSE (ppbV)")
    ax.set_title("Blocked CV: test RMSE per fold (points = mean over 3 seeds, bars = seed std)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cv_rmse_by_fold.png", dpi=150)
    plt.close(fig)
    print(f"wrote {OUT_DIR / 'cv_rmse_by_fold.png'}")
    report.append(f"\n## RMSE by fold, all models\n\n![cv rmse by fold](cv_rmse_by_fold.png)\n")

    (OUT_DIR / "cv_summary.md").write_text("\n".join(report))
    print(f"\nWrote {OUT_DIR / 'cv_summary.md'}")


if __name__ == "__main__":
    main()
