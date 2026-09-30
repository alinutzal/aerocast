"""Maps of the forecasts on the highest-Ox test window of one CV fold: truth, the baselines
and every model, at the lead hours evaluate.py saved (evaluate.spectrum_leads, default +1/+5/+10 h).

Each final run keeps its test-window truth and forecast in runs/<run_id>/error_maps_test.npz;
the window is picked from the truth, so every model on a fold shows the same hours. Writes
two figures: forecasts (shared colour scale) and forecast - truth (symmetric scale), on the
grid's own Lambert Conformal projection with state lines and the 64x64 core box outlined.

    uv run python scripts/plot_prediction_maps.py --fold 3
    uv run python scripts/plot_prediction_maps.py --fold 1 --mode species --seed 43 --models gnn mamba unet
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

LABELS = {"convlstm": "ConvLSTM", "unet": "U-Net", "fno": "FNO", "swin": "SwinUNETR", "swin_unet": "Swin U-Net",
          "gnn": "GNN", "mamba": "Mamba", "persistence": "Persistence", "climatology": "Climatology"}
MODELS = ["gnn", "mamba", "swin_unet", "unet", "fno", "convlstm", "swin"]
STATIC = "/fs/ess/PLS0144/data/aerocast_equates/pilot/files/static.nc"
RING = 4  # boundary cells around the 64x64 core (configs/data/equates_2019_07_ca12km.yaml)


def find_run(runs_dir, fold, model, mode, seed):
    """Latest final run of this fold/model/mode/seed, or None."""
    name = f"benchmark_equates_fold{fold}-final-{model}-{mode}-s{seed}"
    hits = sorted(p for p in Path(runs_dir).glob(f"*_{name}_*") if (p / "error_maps_test.npz").exists())
    return hits[-1] if hits else None


def load_panels(runs_dir, fold, mode, seed, models, baselines):
    panels, window, leads, unit = [], None, None, ""
    for model in models:
        run = find_run(runs_dir, fold, model, mode, seed)
        if run is None:
            print(f"skip {model}: no finished fold {fold} {mode} seed {seed} run")
            continue
        with np.load(run / "error_maps_test.npz") as z:
            start = str(z["first_target_hour_utc"])
            if window is None:
                window, leads, truth = start, z["leads"], z["truth"]
                unit = json.loads((run / "run.json").read_text()).get("target_unit", "")
                panels += [(LABELS[b], z[b]) for b in baselines if b in z.files]
            elif start != window:
                print(f"skip {model}: its test window starts {start}, not {window}")
                continue
            panels.append((LABELS.get(model, model), z[model]))
    if window is None:
        raise SystemExit(f"No runs with error maps for fold {fold}, mode {mode}, seed {seed} in {runs_dir}.")
    return truth, panels, leads, window, unit


def map_axes(fig, n_rows, n_cols, projection):
    return np.array([[fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection=projection)
                      for c in range(n_cols)] for r in range(n_rows)])


def decorate(ax, lon, lat, borders):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    if borders:
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.4, edgecolor="0.3")
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5)
    core = (slice(RING, -RING), slice(RING, -RING))
    edge_lon = np.concatenate([lon[core][0], lon[core][:, -1], lon[core][-1][::-1], lon[core][::-1, 0]])
    edge_lat = np.concatenate([lat[core][0], lat[core][:, -1], lat[core][-1][::-1], lat[core][::-1, 0]])
    ax.plot(edge_lon, edge_lat, "k--", lw=0.6, transform=ccrs.PlateCarree())
    xy = ax.projection.transform_points(ccrs.PlateCarree(), lon, lat)  # extent in map coordinates: no margin
    ax.set_extent([xy[..., 0].min(), xy[..., 0].max(), xy[..., 1].min(), xy[..., 1].max()], crs=ax.projection)


def plot(fields, truth, leads, lon, lat, title, path, error, unit, borders):
    import cartopy.crs as ccrs
    rows = fields if error else [("Truth (CMAQ)", truth)] + fields
    projection = ccrs.LambertConformal(central_longitude=-97, standard_parallels=(33, 45))
    fig = plt.figure(figsize=(3.0 * len(leads) + 1, 2.9 * len(rows)), layout="constrained")
    axes = map_axes(fig, len(rows), len(leads), projection)
    if error:
        limit = np.nanpercentile(np.abs(np.stack([f - truth for _, f in rows])), 99)
        kw = dict(cmap="RdBu_r", vmin=-limit, vmax=limit)
    else:
        vmin, vmax = np.nanpercentile(np.stack([truth] + [f for _, f in rows]), [1, 99])
        kw = dict(cmap="viridis", vmin=vmin, vmax=vmax)
    for r, (name, field) in enumerate(rows):
        for c, lead in enumerate(leads):
            ax = axes[r, c]
            shown = field[c] - truth[c] if error else field[c]
            mesh = ax.pcolormesh(lon, lat, shown, transform=ccrs.PlateCarree(), shading="auto", **kw)
            decorate(ax, lon, lat, borders)
            rmse = "" if name.startswith("Truth") else f"  RMSE {np.sqrt(np.mean((field[c] - truth[c]) ** 2)):.2f}"
            ax.set_title(f"{name}, +{lead} h{rmse}", fontsize=8)
    fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.4, pad=0.02,
                 label=f"{'Forecast - truth' if error else 'Ox'} ({unit})")
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fold", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--mode", default="ox", choices=["ox", "species", "multitask"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--baselines", nargs="*", default=["persistence", "climatology"])
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--static", default=STATIC, help="NetCDF with LAT/LON on the model grid")
    parser.add_argument("--out", default="results/benchmark_cv/figures/maps")
    parser.add_argument("--no-borders", action="store_true", help="skip state lines (they need Natural Earth data)")
    args = parser.parse_args(argv)

    truth, panels, leads, window, unit = load_panels(args.runs_dir, args.fold, args.mode, args.seed,
                                                     args.models, args.baselines)
    with xr.open_dataset(args.static) as ds:
        lat, lon = ds["LAT"].values.squeeze(), ds["LON"].values.squeeze()
    if lat.shape != truth.shape[1:]:
        raise SystemExit(f"{args.static} grid {lat.shape} does not match the maps {truth.shape[1:]}.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"fold{args.fold}_{args.mode}_s{args.seed}"
    title = f"Highest-Ox test window, first target hour {window} UTC (fold {args.fold}, {args.mode} mode, seed {args.seed})"
    plot(panels, truth, leads, lon, lat, title, out / f"forecast_{tag}.png", False, unit, not args.no_borders)
    plot(panels, truth, leads, lon, lat, title, out / f"error_{tag}.png", True, unit, not args.no_borders)


if __name__ == "__main__":
    main()
