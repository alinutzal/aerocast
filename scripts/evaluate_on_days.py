"""Score already-trained benchmark models on a different test period, without retraining.

For each fold and model, the saved checkpoint is copied with its normalization stats and run
record into --out, together with a config whose test block is --test-start..--test-end. Input
history may come from earlier days (split.inputs_from_any_split), so windows whose targets fall
in the test period keep their full history. The original run directories are not touched: their
daily_stats.csv and results.csv rows, which the CV tables read, stay as they were. Scores go to
--out/results.csv, and the per-run summaries are written in the copies.

The models were trained with the run's own input and horizon (6 h of history, 10 h of targets),
so this is not a 7-day-history test. The training and validation blocks of each run are kept, so
the high-Ox threshold and the climatology baseline are the ones the run was tuned with.

    uv run python scripts/evaluate_on_days.py --test-start 2019-08-27 --test-end 2019-08-29 \
        --out /fs/scratch/<project>/$USER/AeroCast/abnormal_eval
"""
import argparse
import csv
import shutil
from pathlib import Path

import yaml

from aerocast.evaluate import evaluate_run

def find_run(runs_dir, fold, model, mode, seed):
    """Latest final run of this fold/model/mode/seed with saved test maps, or None."""
    name = f"benchmark_equates_fold{fold}-final-{model}-{mode}-s{seed}"
    hits = sorted(p for p in Path(runs_dir).glob(f"*_{name}_*") if (p / "best.pt").exists())
    return hits[-1] if hits else None


def make_copy(src, dst, test_start, test_end, experiments_md):
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "norm_stats.json", "run.json"):
        shutil.copy2(src / name, dst / name)
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    cfg["data"]["dates"] = None  # the run's explicit training-day list would hide the test days
    cfg["split"]["test"] = [test_start, test_end]
    cfg["split"]["inputs_from_any_split"] = True
    cfg["output"]["experiments_md"] = str(experiments_md)
    cfg["output"]["note"] = f"re-scored on {test_start}..{test_end} from {src.name}"
    (dst / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))


def summarize(results_csv, out_md, models):
    """RMSE by lead hour, averaged over the folds scored (baselines are scored once per run)."""
    with open(results_csv, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test" and r["metric"] == "rmse"]
    leads = [str(k) for k in range(1, 11)] + ["all"]
    lines = ["# Trained models on the test days (Ox RMSE, ppbV, mean over folds, seed 42, ox mode)", "",
             "| model | " + " | ".join("all" if k == "all" else f"+{k}h" for k in leads) + " |",
             "|---" * (len(leads) + 1) + "|"]
    for model in ["persistence", "climatology", *models]:
        cells = []
        for lead in leads:
            vals = [float(r["value"]) for r in rows if r["model"] == model and r["lead_hour"] == lead]
            cells.append(f"{sum(vals) / len(vals):.2f}" if vals else "-")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    out_md.write_text("\n".join(lines) + "\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-start", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=["unet", "fno", "swin", "swin_unet", "gnn", "mamba", "convlstm"])
    parser.add_argument("--mode", default="ox")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results_csv = out / "results.csv"
    experiments_md = out / "EXPERIMENTS.md"
    for model in args.models:
        for fold in args.folds:
            src = find_run(args.runs_dir, fold, model, args.mode, args.seed)
            if src is None:
                print(f"skip fold {fold} {model}: no checkpoint in {args.runs_dir}")
                continue
            dst = out / "runs" / f"fold{fold}-{model}-{args.mode}-s{args.seed}"
            make_copy(src, dst, args.test_start, args.test_end, experiments_md)
            evaluate_run(dst, results_csv=results_csv, labels=["test"], rewrite_test=True)
    print(summarize(results_csv, out / "summary.md", args.models))


if __name__ == "__main__":
    main()
