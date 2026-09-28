"""Benchmark tables and figure data from results.csv and the run directories.

Only test rows are used; smoke-test and legacy rows never enter the tables. For each model x
target mode, metrics are averaged over seeds. 95% CIs come from a block bootstrap over test
days applied to the seed-averaged metric, and differences vs the U-Net of the same target mode
reuse the same resampled days (paired).

Writes to --out (default results/benchmark):
  main.md, main.tex            persistence and climatology first, then model x target mode:
                               Ox RMSE, MAE, MB, r (each with a 95% CI), CSI (p95), spectral
                               ratio at leads 1/5/10, parameters, inference time, and the RMSE
                               difference vs U-Net with its CI
  lead_rmse.md, lead_rmse.tex  Ox RMSE by lead hour
  figures/lead_rmse.csv        RMSE and CI by lead hour
  figures/spectra.csv          radially averaged power of forecasts and truth
  figures/error_maps/*.csv     truth, forecast and error on the highest-Ox test window
  figures/*.png                quick previews

    uv run python scripts/make_tables.py
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from aerocast.metrics import FEATURES, bootstrap_counts, metrics_from_sums

BASELINES = ("persistence", "climatology")
MODES = ("ox", "species", "multitask")
CI_METRICS = ("rmse", "mae", "bias", "pearson_r")
LABELS = {"convlstm": "ConvLSTM", "unet": "U-Net", "fno": "FNO", "swin": "SwinUNETR", "swin_unet": "Swin U-Net",
          "gnn": "GNN", "mamba": "Mamba", "persistence": "Persistence", "climatology": "Climatology"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class Runs:
    """Test rows, run metadata and per-day sums, grouped by (source, target mode)."""

    def __init__(self, results_csv, runs_dir):
        self.rows = [r for r in read_csv(results_csv) if r["split"] == "test"]
        if not self.rows:
            raise SystemExit(f"No test rows in {results_csv}. Tables are built from date-split test rows only; "
                             "smoke-test rows are never used.")
        self.meta = {}
        for run_id in dict.fromkeys(r["run_id"] for r in self.rows):
            path = Path(runs_dir) / run_id / "run.json"
            if path.exists():
                self.meta[run_id] = json.loads(path.read_text())
        self.values = {}
        for r in self.rows:  # last write wins if a run was re-evaluated with --rewrite-test
            self.values[(r["run_id"], r["model"], r["lead_hour"], r["metric"])] = float(r["value"])
        self.groups = defaultdict(list)  # (model, mode) -> run ids (one per seed)
        for run_id, meta in self.meta.items():
            self.groups[(meta["model"], meta["target_mode"])].append(run_id)
        self.runs_dir = Path(runs_dir)
        self._daily = {}

    def sources(self, order):
        models = [m for m in order if any(key[0] == m for key in self.groups)]
        models += sorted({m for m, _ in self.groups} - set(models))
        rows = [(b, None) for b in BASELINES]
        rows += [(m, mode) for m in models for mode in MODES if (m, mode) in self.groups]
        return rows

    def runs_for(self, source, mode):
        """Run ids scoring this source: its own runs, or (baselines) the U-Net ox runs, else any runs."""
        if mode is not None:
            return sorted(self.groups[(source, mode)], key=lambda r: self.meta[r]["seed"])
        for key in (("unet", "ox"), *sorted(self.groups)):
            if key in self.groups:
                return sorted(self.groups[key], key=lambda r: self.meta[r]["seed"])[:1]
        return []

    def value(self, source, mode, metric, lead="all"):
        values = [self.values.get((r, source, str(lead), metric)) for r in self.runs_for(source, mode)]
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else float("nan")

    def daily(self, source, mode):
        """Per-seed (days, D x leads x features) Ox sums on the test split."""
        out = []
        for run_id in self.runs_for(source, mode):
            key = (run_id, source)
            if key not in self._daily:
                path = self.runs_dir / run_id / "daily_stats.csv"
                rows = [r for r in read_csv(path) if r["split"] == "test" and r["model"] == source and r["variable"] == "Ox"]
                days = sorted({r["day"] for r in rows})
                leads = sorted({int(r["lead_hour"]) for r in rows})
                array = np.zeros((len(days), len(leads), len(FEATURES)))
                for r in rows:
                    array[days.index(r["day"]), leads.index(int(r["lead_hour"]))] = [float(r[f]) for f in FEATURES]
                self._daily[key] = (days, array)
            out.append(self._daily[key])
        return out


def bootstrap(runs, source, mode, reference, n_samples, seed):
    """Resampled seed-mean metrics (overall and per lead) for a source, and minus the reference."""
    daily = runs.daily(source, mode)
    if not daily or len(daily[0][0]) < 2:
        return None
    days = daily[0][0]
    counts = bootstrap_counts(len(days), n_samples, seed)

    def resampled(per_seed):
        overall, per_lead = [], []
        for seed_days, array in per_seed:
            if seed_days != days:
                raise SystemExit(f"{source}/{mode}: runs were scored on different test days")
            sums = np.einsum("bd,dlf->blf", counts, array)
            overall.append(metrics_from_sums(sums.sum(axis=1)))
            per_lead.append(metrics_from_sums(sums))
        mean = lambda parts: {m: np.mean([p[m] for p in parts], axis=0) for m in parts[0]}  # noqa: E731
        return mean(overall), mean(per_lead)

    overall, per_lead = resampled(daily)
    out = {"overall": overall, "per_lead": per_lead}
    if reference is not None:
        ref = runs.daily(*reference)
        if ref:
            ref_overall, _ = resampled(ref)
            out["diff_rmse"] = overall["rmse"] - ref_overall["rmse"]
    return out


def ci(samples):
    lo, hi = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    return lo, hi


def fmt(value, digits=2):
    return "–" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def profile_times(path):
    if not Path(path).exists():
        return {}
    return {r["model"]: r for r in read_csv(path)}


def build_tables(runs, order, profile, n_samples, seed):
    main, lead_rows, lead_csv = [], [], []
    for source, mode in runs.sources(order):
        is_baseline = mode is None
        reference = ("unet", mode or "ox") if source != "unet" and ("unet", mode or "ox") in runs.groups else None
        boot = bootstrap(runs, source, mode, reference, n_samples, seed)
        cells = {"source": source, "mode": mode or "–",
                 "name": LABELS.get(source, source) + ("" if is_baseline else f" ({mode})")}
        for m in CI_METRICS:
            cells[m] = runs.value(source, mode, m)
            cells[f"{m}_ci"] = ci(boot["overall"][m]) if boot else (None, None)
        cells["csi"] = runs.value(source, mode, "csi_p95")
        cells["spectral"] = [runs.value(source, mode, "spectral_ratio", k) for k in (1, 5, 10)]
        run_ids = runs.runs_for(source, mode)
        cells["params"] = None if is_baseline else runs.meta[run_ids[0]].get("parameters")
        cells["infer_ms"] = None if is_baseline else float(profile.get(source, {}).get("inference_ms_mean") or "nan")
        cells["diff"] = cells["rmse"] - runs.value(*reference, "rmse") if reference else None
        cells["diff_ci"] = ci(boot["diff_rmse"]) if boot and "diff_rmse" in boot else (None, None)
        cells["seeds"] = len(run_ids) if not is_baseline else 1
        main.append(cells)
        leads = sorted({int(k[2]) for k in runs.values if k[1] == source and k[2] != "all"})
        lead_values = [runs.value(source, mode, "rmse", k) for k in leads]
        lead_rows.append((cells["name"], lead_values))
        lo, hi = ci(boot["per_lead"]["rmse"]) if boot else ([None] * len(leads), [None] * len(leads))
        lead_csv += [{"model": source, "target_mode": mode or "", "lead_hour": k, "rmse": f"{v:.6g}",
                      "ci95_lo": "" if lo[i] is None else f"{lo[i]:.6g}", "ci95_hi": "" if hi[i] is None else f"{hi[i]:.6g}"}
                     for i, (k, v) in enumerate(zip(leads, lead_values))]
    return main, lead_rows, lead_csv


def with_ci(value, interval):
    lo, hi = interval
    return fmt(value) if lo is None else f"{fmt(value)} [{fmt(lo)}, {fmt(hi)}]"


def main_markdown(main, unit):
    head = ["Model", "Seeds", f"RMSE ({unit})", "MAE", "MB", "r", "CSI p95", "Spectral ratio +1/+5/+10",
            "Params (M)", "Inference (ms)", "ΔRMSE vs U-Net"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for c in main:
        lines.append("| " + " | ".join([
            c["name"], str(c["seeds"]), with_ci(c["rmse"], c["rmse_ci"]), with_ci(c["mae"], c["mae_ci"]),
            with_ci(c["bias"], c["bias_ci"]), with_ci(c["pearson_r"], c["pearson_r_ci"]), fmt(c["csi"]),
            "/".join(fmt(v) for v in c["spectral"]), fmt(c["params"] / 1e6 if c["params"] else None),
            fmt(c["infer_ms"], 1), with_ci(c["diff"], c["diff_ci"]) if c["diff"] is not None else "–"]) + " |")
    return "\n".join(lines) + "\n"


def latex_escape(text):
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("Δ", r"$\Delta$")


def main_latex(main, unit):
    def cell(value, interval):
        lo, hi = interval
        return fmt(value) if lo is None else f"{fmt(value)} \\scriptsize[{fmt(lo)}, {fmt(hi)}]"

    lines = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{lrllllrlrrl}", r"\toprule",
             f"Model & Seeds & RMSE ({latex_escape(unit)}) & MAE & MB & $r$ & CSI$_{{95}}$ & SR$_{{1/5/10}}$ & "
             r"Params (M) & Inference (ms) & $\Delta$RMSE vs U-Net \\", r"\midrule"]
    for i, c in enumerate(main):
        if i == len(BASELINES):
            lines.append(r"\midrule")
        lines.append(" & ".join([
            latex_escape(c["name"]), str(c["seeds"]), cell(c["rmse"], c["rmse_ci"]), cell(c["mae"], c["mae_ci"]),
            cell(c["bias"], c["bias_ci"]), cell(c["pearson_r"], c["pearson_r_ci"]), fmt(c["csi"]),
            "/".join(fmt(v) for v in c["spectral"]), fmt(c["params"] / 1e6 if c["params"] else None),
            fmt(c["infer_ms"], 1), cell(c["diff"], c["diff_ci"]) if c["diff"] is not None else "--"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Test-set Ox scores (mean over seeds; 95\% CIs by block bootstrap over test days; "
              r"$\Delta$RMSE is paired against the U-Net of the same target mode).}", r"\label{tab:benchmark}",
              r"\end{table}"]
    return "\n".join(lines).replace("–", "--") + "\n"


def lead_tables(lead_rows):
    leads = len(lead_rows[0][1])
    head = ["Model"] + [f"+{k}h" for k in range(1, leads + 1)]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join([name] + [fmt(v) for v in values]) + " |" for name, values in lead_rows]
    tex = [r"\begin{table}[t]", r"\centering\small", r"\begin{tabular}{l" + "r" * leads + "}", r"\toprule",
           " & ".join(["Model"] + [f"+{k}h" for k in range(1, leads + 1)]) + r" \\", r"\midrule"]
    tex += [" & ".join([latex_escape(name)] + [fmt(v) for v in values]) + r" \\" for name, values in lead_rows]
    tex += [r"\bottomrule", r"\end{tabular}", r"\caption{Test-set Ox RMSE by lead hour (mean over seeds).}",
            r"\label{tab:lead-rmse}", r"\end{table}"]
    return "\n".join(md) + "\n", "\n".join(tex).replace("–", "--") + "\n"


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figure_data(runs, order, figures):
    spectra, maps = [], {}
    for source, mode in runs.sources(order):
        per_seed = []
        for run_id in runs.runs_for(source, mode):
            path = runs.runs_dir / run_id / "spectra.csv"
            if path.exists():
                per_seed.append([r for r in read_csv(path) if r["split"] == "test" and r["model"] == source])
        if per_seed:
            for i, r in enumerate(per_seed[0]):
                spectra.append({"model": source, "target_mode": mode or "", "lead_hour": r["lead_hour"],
                                "k_cycles_per_km": r["k_cycles_per_km"],
                                "power_pred": f"{np.mean([float(s[i]['power_pred']) for s in per_seed]):.6g}",
                                "power_true": r["power_true"]})
        run_ids = runs.runs_for(source, mode)
        path = runs.runs_dir / run_ids[0] / "error_maps_test.npz" if run_ids else None
        if path is not None and path.exists():
            with np.load(path) as z:
                if source in z:
                    maps[(source, mode)] = (z["truth"], z[source], z["leads"], str(z["first_target_hour_utc"]))
    if spectra:
        write_csv(figures / "spectra.csv", spectra)
    error_dir = figures / "error_maps"
    for (source, mode), (truth, forecast, leads, start) in maps.items():
        error_dir.mkdir(parents=True, exist_ok=True)
        tag = source if mode is None else f"{source}_{mode}"
        for i, lead in enumerate(leads):
            np.savetxt(error_dir / f"truth_lead{lead}.csv", truth[i], delimiter=",", fmt="%.4f")
            np.savetxt(error_dir / f"{tag}_lead{lead}_forecast.csv", forecast[i], delimiter=",", fmt="%.4f")
            np.savetxt(error_dir / f"{tag}_lead{lead}_error.csv", forecast[i] - truth[i], delimiter=",", fmt="%.4f")
    if maps:
        (error_dir / "README.txt").write_text(
            f"Highest-Ox test window, first target hour {next(iter(maps.values()))[3]} UTC.\n"
            "Rows are grid rows (south to north), columns grid columns (west to east); values in native units.\n")
    return spectra, maps


def previews(figures, lead_csv, spectra, maps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key in dict.fromkeys((r["model"], r["target_mode"]) for r in lead_csv):
        rows = [r for r in lead_csv if (r["model"], r["target_mode"]) == key]
        ax.plot([int(r["lead_hour"]) for r in rows], [float(r["rmse"]) for r in rows], marker="o", ms=3,
                label=LABELS.get(key[0], key[0]) + (f" ({key[1]})" if key[1] else ""))
    ax.set(xlabel="Lead hour", ylabel="Ox RMSE", title="Test Ox RMSE by lead hour")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "lead_rmse.png", dpi=120)
    plt.close(fig)

    if spectra:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        lead = "5" if any(r["lead_hour"] == "5" for r in spectra) else spectra[0]["lead_hour"]
        truth_drawn = False
        for key in dict.fromkeys((r["model"], r["target_mode"]) for r in spectra):
            rows = [r for r in spectra if (r["model"], r["target_mode"]) == key and r["lead_hour"] == lead]
            k = [float(r["k_cycles_per_km"]) for r in rows]
            if not truth_drawn:
                ax.loglog(k, [float(r["power_true"]) for r in rows], "k-", lw=2, label="truth")
                truth_drawn = True
            ax.loglog(k, [float(r["power_pred"]) for r in rows], lw=1,
                      label=LABELS.get(key[0], key[0]) + (f" ({key[1]})" if key[1] else ""))
        ax.set(xlabel="Wavenumber (cycles/km)", ylabel="Power", title=f"Radially averaged Ox power, lead +{lead} h")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(figures / "spectra.png", dpi=120)
        plt.close(fig)

    if maps:
        items = list(maps.items())
        fig, axes = plt.subplots(1, len(items) + 1, figsize=(2.6 * (len(items) + 1), 3.2), squeeze=False)
        truth = items[0][1][0][0]
        axes[0, 0].imshow(truth, origin="lower")
        axes[0, 0].set_title("truth +1 h", fontsize=8)
        limit = max(np.abs(f[0] - t[0]).max() for (t, f, _, _) in maps.values())
        for ax, ((source, mode), (t, f, _, _)) in zip(axes[0, 1:], items):
            ax.imshow(f[0] - t[0], origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit)
            ax.set_title(LABELS.get(source, source) + (f"\n{mode}" if mode else ""), fontsize=8)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(figures / "error_maps.png", dpi=120)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark tables and figure data from results.csv.")
    parser.add_argument("--results", default="results/results.csv")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--profile", default="results/profile.csv")
    parser.add_argument("--out", default="results/benchmark")
    parser.add_argument("--order", nargs="+", default=["unet", "convlstm", "fno", "swin", "swin_unet", "gnn", "mamba"])
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args(argv)

    runs = Runs(args.results, args.runs_dir)
    out = Path(args.out)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    unit = next(iter(runs.meta.values())).get("target_unit", "")
    main_rows, lead_rows, lead_csv = build_tables(runs, args.order, profile_times(args.profile),
                                                  args.bootstrap_samples, args.seed)
    (out / "main.md").write_text(main_markdown(main_rows, unit))
    (out / "main.tex").write_text(main_latex(main_rows, unit))
    lead_md, lead_tex = lead_tables(lead_rows)
    (out / "lead_rmse.md").write_text(lead_md)
    (out / "lead_rmse.tex").write_text(lead_tex)
    write_csv(figures / "lead_rmse.csv", lead_csv)
    spectra, maps = figure_data(runs, args.order, figures)
    if not args.no_previews:
        previews(figures, lead_csv, spectra, maps)
    print((out / "main.md").read_text())
    print(f"Wrote tables and figure data to {out}")


if __name__ == "__main__":
    main()
