"""Score a trained run and the baselines on identical windows, in the target's native units.

Appends rows (run_id, model, split, lead_hour, metric, value, config_hash, git_commit) to
results.csv, and writes per_window_rmse.csv and summary.txt into the run directory.
"""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aerocast.baselines import DiurnalClimatology, persistence
from aerocast.config import load_config
from aerocast.metrics import LeadMetrics
from aerocast.models import build_model
from aerocast.normalize import NormStats
from aerocast.splits import load_and_split
from aerocast.train import predict_batch, window_datasets, write_json

RESULT_COLUMNS = ["run_id", "model", "split", "lead_hour", "metric", "value", "config_hash", "git_commit"]
MODELS = ("convlstm", "persistence", "climatology")
NOTES = {"smoke_test": "IN-SAMPLE smoke test, not a held-out result",
         "legacy": "in-sample, refactor check only"}


def append_results(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with open(path, newline="") as f:
            header = next(csv.reader(f), None)
        if header != RESULT_COLUMNS:
            raise ValueError(f"{path} has columns {header}, expected {RESULT_COLUMNS}")
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerows(rows)


def format_summary(rows, unit, n_windows, pred_len):
    def value(sub, model, lead, metric):
        return next(float(r["value"]) for r in sub if r["model"] == model and r["lead_hour"] == lead and r["metric"] == metric)

    lines = []
    for label in dict.fromkeys(r["split"] for r in rows):
        sub = [r for r in rows if r["split"] == label]
        note = f"  [{NOTES[label]}]" if label in NOTES else ""
        lines += ["", f"== {label}: {n_windows[label]} windows, Ox in {unit}{note}",
                  f"{'model':<12} {'RMSE':>7} {'MAE':>7} {'bias':>7} {'r':>6}"]
        lines += [f"{m:<12} {value(sub, m, 'all', 'rmse'):7.3f} {value(sub, m, 'all', 'mae'):7.3f} "
                  f"{value(sub, m, 'all', 'bias'):7.3f} {value(sub, m, 'all', 'pearson_r'):6.3f}" for m in MODELS]
        lines += ["RMSE by lead hour", f"{'model':<12} " + " ".join(f"{f'+{k}h':>6}" for k in range(1, pred_len + 1))]
        lines += [f"{m:<12} " + " ".join(f"{value(sub, m, k, 'rmse'):6.2f}" for k in range(1, pred_len + 1)) for m in MODELS]
    return "\n".join(lines)


def evaluate_run(run_dir, results_csv=None):
    """Evaluate runs/<run_id>; returns the rows appended to results.csv."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    meta = json.loads((run_dir / "run.json").read_text())
    stats = NormStats.load(run_dir / "norm_stats.json")
    pred_len = cfg["data"]["pred_len"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hourly, splits = load_and_split(cfg)
    if stats.channels != hourly.channels:
        raise ValueError(f"Run was trained on channels {stats.channels}, data has {hourly.channels}")
    model = build_model(cfg["model"], len(hourly.channels)).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model_state_dict"])
    model.eval()
    climatology = DiurnalClimatology(hourly.y, hourly.times, splits.train_hours)
    datasets = window_datasets(hourly, stats, cfg, splits.evaluate)

    rows, per_window = [], []
    for label, starts in splits.evaluate.items():
        if label == "test" and splits.mode != "dates":
            raise RuntimeError("Only a date-block split may produce test rows")
        metrics = {name: LeadMetrics(pred_len) for name in MODELS}
        loader = DataLoader(datasets[label], batch_size=cfg["train"]["batch_size"])
        i = 0
        with torch.no_grad():
            for batch in loader:
                pred, _ = predict_batch(model, batch, device, pred_len)
                for model_pred in stats.denormalize_y(pred.cpu().numpy()):
                    t = int(starts[i])
                    obs = hourly.y[t:t + pred_len]
                    forecasts = {"convlstm": model_pred,
                                 "persistence": persistence(hourly.y, t, pred_len),
                                 "climatology": climatology.predict(hourly.times[t:t + pred_len])}
                    for name, forecast in forecasts.items():
                        metrics[name].update(forecast[None], obs[None])
                        rmse = float(np.sqrt(np.mean((forecast.astype(np.float64) - obs) ** 2)))
                        per_window.append({"split": label, "window": i, "first_target_hour_utc": str(hourly.times[t]),
                                           "model": name, "rmse": f"{rmse:.6g}"})
                    i += 1
        for name in MODELS:
            rows += [{"run_id": meta["run_id"], "model": name, "split": label, "lead_hour": lead, "metric": metric,
                      "value": f"{value:.6g}", "config_hash": meta["config_hash"], "git_commit": meta["git_commit"]}
                     for lead, metric, value in metrics[name].results()]

    results_csv = Path(results_csv or cfg["output"]["results_csv"])
    append_results(results_csv, rows)
    with open(run_dir / "per_window_rmse.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_window[0]))
        writer.writeheader()
        writer.writerows(per_window)
    summary = format_summary(rows, hourly.target_unit, {k: len(v) for k, v in splits.evaluate.items()}, pred_len)
    (run_dir / "summary.txt").write_text(summary.lstrip() + "\n")
    meta["evaluation"] = {"evaluated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "labels": list(splits.evaluate), "target_unit": hourly.target_unit,
                          "results_csv": str(results_csv)}
    write_json(run_dir / "run.json", meta)
    print(summary)
    print(f"\nAppended {len(rows)} rows to {results_csv}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained run and the baselines; append to results.csv.")
    parser.add_argument("run_dir", help="run directory written by aerocast-train, e.g. runs/<run_id>")
    parser.add_argument("--results-csv", default=None, help="default: output.results_csv from the run's config")
    args = parser.parse_args(argv)
    evaluate_run(args.run_dir, args.results_csv)


if __name__ == "__main__":
    main()
