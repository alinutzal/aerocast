"""Score a trained run and the baselines on identical windows, in the targets' native units.

Scores Ox for every target mode, and NO2/O3 for species and multitask (plus the Ox formed from
the species heads in multitask, as oxsum_*). Adds high-Ox skill (threshold = percentile of Ox
on training days), a smoothing check (radially averaged power spectra at chosen lead hours),
and 95% CIs by block bootstrap over days. Appends rows (run_id, model, split, lead_hour,
metric, value, config_hash, git_commit) to results.csv, writes per-run files (per-window RMSE,
per-day sums for paired comparisons, spectra, error maps, summary) and records the run in
EXPERIMENTS.md. Test rows are written once per config hash.
"""
import argparse
import csv
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from aerocast.baselines import DiurnalClimatology, persistence
from aerocast.config import load_config
from aerocast.metrics import FEATURES, ForecastScores, radial_bins
from aerocast.models import build_model, model_spec
from aerocast.normalize import NormStats
from aerocast.splits import load_and_split
from aerocast.targets import physical
from aerocast.train import predict_batch, window_datasets, write_json

RESULT_COLUMNS = ["run_id", "dataset", "model", "split", "lead_hour", "metric", "value", "config_hash", "git_commit"]
BASELINES = ("persistence", "climatology")
NOTES = {"smoke_test": "IN-SAMPLE smoke test, not a held-out result",
         "legacy": "in-sample, refactor check only"}
EXPERIMENT_HEADER = ["date", "run id", "dataset", "model", "target mode", "seed", "config hash", "val Ox RMSE", "test Ox RMSE", "note"]
EVAL_DEFAULTS = {"high_ox_percentile": 95, "spectrum_leads": [1, 5, 10], "bootstrap_samples": 1000,
                 "bootstrap_seed": 0, "cell_km": 1.0}


def metric_prefix(variable):
    return {"Ox": "", "Ox_sum": "oxsum_"}.get(variable, f"{variable.lower()}_")


def append_results(path, rows):
    """Append rows under an exclusive lock, so concurrent runs can share one results.csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        header = next(csv.reader(f), None)
        if header is not None and header != RESULT_COLUMNS:
            raise ValueError(f"{path} has columns {header}, expected {RESULT_COLUMNS}")
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if header is None:
            writer.writeheader()
        writer.writerows(rows)


def has_rows(path, config_hash, split):
    path = Path(path)
    if not path.exists():
        return False
    with open(path, newline="") as f:
        return any(r["config_hash"] == config_hash and r["split"] == split for r in csv.DictReader(f))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def log_experiment(path, meta, rows):
    """Add or replace this run's line in the EXPERIMENTS.md table."""
    def ox_rmse(split):
        return next((float(r["value"]) for r in rows if r["model"] == meta["model"] and r["split"] == split
                     and r["lead_hour"] == "all" and r["metric"] == "rmse"), None)

    notes = [f"{label} Ox RMSE {ox_rmse(label):.3f} (in-sample)" for label in NOTES if ox_rmse(label) is not None]
    if meta.get("note"):
        notes.append(meta["note"])
    dataset = rows[0]["dataset"] if rows else "baaqmd"
    cells = [meta["created"][:10], meta["run_id"], dataset, meta["model"], meta.get("target_mode", "ox"), str(meta["seed"]),
             meta["config_hash"], *(f"{v:.3f}" if v is not None else "–" for v in (ox_rmse("val"), ox_rmse("test"))),
             "; ".join(notes)]
    line = "| " + " | ".join(cells) + " |"
    path = Path(path)
    with open(path, "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        lines = f.read().splitlines()
        if not lines:
            lines = ["# Experiments", "", "One line per run, written by aerocast-evaluate.", "",
                     "| " + " | ".join(EXPERIMENT_HEADER) + " |", "|" + "---|" * len(EXPERIMENT_HEADER)]
        marker = f"| {meta['run_id']} |"
        lines = [line if marker in existing else existing for existing in lines]
        if line not in lines:
            lines.append(line)
        f.seek(0)
        f.truncate()
        f.write("\n".join(lines) + "\n")


def format_summary(rows, unit, info, pred_len, sources, variables, spectrum_leads):
    def value(sub, source, lead, metric):
        return next((float(r["value"]) for r in sub if r["model"] == source and r["lead_hour"] == lead
                     and r["metric"] == metric), float("nan"))

    lines = []
    for label in dict.fromkeys(r["split"] for r in rows):
        sub = [r for r in rows if r["split"] == label]
        note = f"  [{NOTES[label]}]" if label in NOTES else ""
        lines += ["", f"== {label}: {info[label]['windows']} windows over {info[label]['days']} day(s), "
                      f"native units ({unit}); high Ox > {info['threshold']:.1f}{note}"]
        spectral = " ".join(f"{f'SR+{k}':>6}" for k in spectrum_leads)
        lines.append(f"{'Ox':<12} {'RMSE':>7} {'MAE':>7} {'bias':>7} {'r':>6} {'CSI95':>6} {'RMSE>95':>8} {spectral}")
        for s in sources:
            ratios = " ".join(f"{value(sub, s, k, 'spectral_ratio'):6.2f}" for k in spectrum_leads)
            lines.append(f"{s:<12} {value(sub, s, 'all', 'rmse'):7.3f} {value(sub, s, 'all', 'mae'):7.3f} "
                         f"{value(sub, s, 'all', 'bias'):7.3f} {value(sub, s, 'all', 'pearson_r'):6.3f} "
                         f"{value(sub, s, 'all', 'csi_p95'):6.3f} {value(sub, s, 'all', 'rmse_above_p95'):8.3f} {ratios}")
        lines += ["Ox RMSE by lead hour", f"{'model':<12} " + " ".join(f"{f'+{k}h':>6}" for k in range(1, pred_len + 1))]
        lines += [f"{s:<12} " + " ".join(f"{value(sub, s, k, 'rmse'):6.2f}" for k in range(1, pred_len + 1)) for s in sources]
        for variable in [v for v in variables if v != "Ox"]:
            p = metric_prefix(variable)
            names = [s for s in sources if not np.isnan(value(sub, s, "all", p + "rmse"))]
            lines.append(f"{variable:<12} " + " ".join(f"{s}: RMSE {value(sub, s, 'all', p + 'rmse'):.3f}, "
                                                       f"r {value(sub, s, 'all', p + 'pearson_r'):.3f}" for s in names))
    return "\n".join(lines)


def evaluate_run(run_dir, results_csv=None, labels=None, rewrite_test=False):
    """Evaluate runs/<run_id>; returns the rows appended to results.csv."""
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    meta = json.loads((run_dir / "run.json").read_text())
    stats = NormStats.load(run_dir / "norm_stats.json")
    spec = model_spec(cfg)
    pred_len = spec.t_out
    ecfg = {**EVAL_DEFAULTS, **(cfg.get("evaluate") or {})}
    spectrum_leads = [k for k in ecfg["spectrum_leads"] if 1 <= k <= pred_len]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg["train"].get("amp", "none")

    hourly, splits = load_and_split(cfg)
    if stats.channels != hourly.channels:
        raise ValueError(f"Run was trained on channels {stats.channels}, data has {hourly.channels}")
    model = build_model(cfg, stats).to(device)
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device)["model_state_dict"])
    model.eval()
    model_name = cfg["model"]["name"]
    sources = (model_name, *BASELINES)
    species = [c for c in spec.target_channels if c != "Ox"]
    variables = ["Ox", *species]

    train_hours = splits.train_hours
    ox = hourly.targets["Ox"]
    threshold = float(np.percentile(ox[train_hours], ecfg["high_ox_percentile"]))
    shifts = {v: float(hourly.targets[v][train_hours].mean()) for v in variables}
    climatology = {v: DiurnalClimatology(hourly.targets[v], hourly.times, train_hours) for v in variables}
    bins = radial_bins(*ox.shape[1:], ecfg["cell_km"])

    results_csv = Path(results_csv or cfg["output"]["results_csv"])
    labels = [label for label in splits.evaluate if label in (labels or splits.evaluate)]
    if "test" in labels and not rewrite_test and has_rows(results_csv, meta["config_hash"], "test"):
        print(f"Test rows for config {meta['config_hash']} already exist in {results_csv}; not rewriting them.")
        labels.remove("test")
    datasets = window_datasets(hourly, stats, cfg, {label: splits.evaluate[label] for label in labels})

    rows, per_window, daily_rows, spectra_rows, info = [], [], [], [], {"threshold": threshold}
    for label in labels:
        starts = splits.evaluate[label]
        scores = {(s, v): ForecastScores(pred_len, shifts[v], threshold if v == "Ox" else None,
                                         spectrum_leads if v == "Ox" else (), bins)
                  for s in sources for v in variables}
        if "Ox" in spec.target_channels and species:
            scores[(model_name, "Ox_sum")] = ForecastScores(pred_len, shifts["Ox"], threshold, (), bins)
        peak = int(np.argmax([ox[t:t + pred_len].mean() for t in starts]))  # highest-Ox window, for error maps
        maps = {}
        loader = DataLoader(datasets[label], batch_size=cfg["train"]["batch_size"])
        i = 0
        with torch.no_grad():
            for batch in loader:
                pred, _ = predict_batch(model, batch, device, amp)
                fields = {k: v.cpu().numpy() for k, v in physical(pred, stats, spec.target_channels).items()}
                for b in range(len(pred)):
                    t = int(starts[i])
                    day, hours = hourly.days[t], hourly.times[t:t + pred_len]
                    for (source, variable), score in scores.items():
                        truth = hourly.targets["Ox" if variable == "Ox_sum" else variable]
                        if source == model_name:
                            forecast = fields[variable][b]
                        elif source == "persistence":
                            forecast = persistence(truth, t, pred_len)
                        else:
                            forecast = climatology[variable].predict(hours)
                        score.update(forecast, truth[t:t + pred_len], day)
                        if variable == "Ox":
                            error = forecast.astype(np.float64) - truth[t:t + pred_len]
                            per_window.append({"split": label, "window": i, "first_target_hour_utc": str(hours[0]),
                                               "day": str(day), "model": source, "rmse": f"{np.sqrt(np.mean(error ** 2)):.6g}"})
                            if i == peak:
                                maps["truth"] = truth[t:t + pred_len][[k - 1 for k in spectrum_leads]]
                                maps[source] = forecast[[k - 1 for k in spectrum_leads]]
                                maps["first_target_hour_utc"] = str(hours[0])
                    i += 1

        for (source, variable), score in scores.items():
            prefix = metric_prefix(variable)
            for lead, metric, value in (score.rows(prefix)
                                        + score.bootstrap_rows(ecfg["bootstrap_samples"], ecfg["bootstrap_seed"], prefix)):
                rows.append({"run_id": meta["run_id"], "dataset": cfg["data"].get("dataset", "baaqmd"), "model": source,
                             "split": label, "lead_hour": lead, "metric": metric, "value": f"{value:.6g}",
                             "config_hash": meta["config_hash"], "git_commit": meta["git_commit"]})
            days, daily = score.daily()
            daily_rows += [{"split": label, "model": source, "variable": variable, "day": d, "lead_hour": k + 1,
                            **{f: float(daily[j, k, n]) for n, f in enumerate(FEATURES)}}
                           for j, d in enumerate(days) for k in range(pred_len)]
            for lead in score.spectrum_leads:
                power_pred, power_true = score.spectra(lead)
                spectra_rows += [{"split": label, "model": source, "lead_hour": lead, "k_cycles_per_km": f"{k:.6g}",
                                  "power_pred": f"{pp:.6g}", "power_true": f"{pt:.6g}"}
                                 for k, pp, pt in zip(bins[1], power_pred, power_true)]
        np.savez_compressed(run_dir / f"error_maps_{label}.npz", leads=np.array(spectrum_leads),
                            **{k: np.asarray(v) for k, v in maps.items()})
        info[label] = {"windows": len(starts), "days": len({str(hourly.days[t]) for t in starts})}

    if not rows:
        print("Nothing to evaluate.")
        return rows
    append_results(results_csv, rows)
    write_csv(run_dir / "per_window_rmse.csv", per_window)
    write_csv(run_dir / "daily_stats.csv", daily_rows)
    write_csv(run_dir / "spectra.csv", spectra_rows)
    summary = format_summary(rows, hourly.target_unit, info, pred_len, sources, variables, spectrum_leads)
    (run_dir / "summary.txt").write_text(summary.lstrip() + "\n")
    meta["evaluation"] = {"evaluated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "labels": labels, "target_unit": hourly.target_unit, "results_csv": str(results_csv),
                          "high_ox_percentile": ecfg["high_ox_percentile"], "high_ox_threshold": threshold,
                          "windows": {k: v for k, v in info.items() if k != "threshold"}}
    write_json(run_dir / "run.json", meta)
    log_experiment(cfg["output"]["experiments_md"], meta, rows)
    print(summary)
    print(f"\nAppended {len(rows)} rows to {results_csv}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate a trained run and the baselines; append to results.csv.")
    parser.add_argument("run_dir", help="run directory written by aerocast-train, e.g. runs/<run_id>")
    parser.add_argument("--results-csv", default=None, help="default: output.results_csv from the run's config")
    parser.add_argument("--splits", nargs="+", default=None, help="evaluate only these splits, e.g. --splits val")
    parser.add_argument("--rewrite-test", action="store_true", help="write test rows even if this config has them")
    args = parser.parse_args(argv)
    evaluate_run(args.run_dir, args.results_csv, args.splits, args.rewrite_test)


if __name__ == "__main__":
    main()
