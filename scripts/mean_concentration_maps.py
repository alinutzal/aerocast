"""Period-mean surface maps of O3, NO2 and Ox for the EQUATES pilot.

Ox = O3 + NO2 (ppbV). Each map is the mean over every hour of the days in its period, on the
core 64x64 box (or the full 72x72 read area with --include-ring), or with --region bay on the
Bay Area block (37.2-38.7 N, 121.2-123.1 W, the grid cells that fall in that box). Rows are periods, columns are
species, and every column shares one colour scale across the periods, so colours compare
across rows. Each panel shows the box mean, min and max. Full grids are written as CSV.

    uv run python scripts/mean_concentration_maps.py
    uv run python scripts/mean_concentration_maps.py --include-ring
    uv run python scripts/mean_concentration_maps.py --region bay
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RING = 4
OUT_DIR = Path("data/pilot_checks")
PERIODS = [("July 2019, all days", "2019-07-01", "2019-07-31"),
           ("Aug 14-16 2019", "2019-08-14", "2019-08-16"),
           ("Aug 27-29 2019", "2019-08-27", "2019-08-29")]
SPECIES = ["O3", "NO2", "Ox"]


def day_of(path):  # equates_conc_YYYYMMDD.nc
    stamp = path.name[len("equates_conc_"):len("equates_conc_") + 8]
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}"


BAY_BOX = {"lat": (37.2, 38.7), "lon": (-123.1, -121.2)}


def bay_window(lat, lon):
    """Row and column slices of the grid cells inside the Bay Area box (bounding block)."""
    lo_lat, hi_lat = BAY_BOX["lat"]
    lo_lon, hi_lon = BAY_BOX["lon"]
    inside = (lat >= lo_lat) & (lat <= hi_lat) & (lon >= lo_lon) & (lon <= hi_lon)
    rows, cols = np.where(inside.any(axis=1))[0], np.where(inside.any(axis=0))[0]
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def period_means(data_dir, start, end, window):
    """{species: mean grid} over every hour of the days in [start, end]. window is a (rows, cols) pair."""
    files = [p for p in sorted(data_dir.glob("equates_conc_*.nc")) if start <= day_of(p) <= end]
    if not files:
        raise SystemExit(f"No concentration files for {start}..{end} in {data_dir}")
    sums = {s: None for s in SPECIES}
    hours = 0
    for path in files:
        with xr.open_dataset(path) as c:
            o3 = c["O3"].isel(LAY=0, ROW=window[0], COL=window[1]).values
            no2 = c["NO2"].isel(LAY=0, ROW=window[0], COL=window[1]).values
        fields = {"O3": o3, "NO2": no2, "Ox": o3 + no2}
        for s in SPECIES:
            sums[s] = fields[s].sum(axis=0) if sums[s] is None else sums[s] + fields[s].sum(axis=0)
        hours += o3.shape[0]
    return {s: sums[s] / hours for s in SPECIES}, hours, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="default: $AEROCAST_EQUATES_DIR/pilot/files")
    parser.add_argument("--include-ring", action="store_true", help="map the full 72x72 read area, not the 64x64 core")
    parser.add_argument("--region", choices=["core", "bay"], default="core",
                        help="core: the 64x64 box (default); bay: the Bay Area block")
    args = parser.parse_args()
    data_dir = Path(args.data_dir or os.path.join(os.environ.get("AEROCAST_EQUATES_DIR", ""), "pilot", "files"))
    with xr.open_dataset(data_dir / "static.nc") as s:
        full_lat = s["LAT"].isel(TSTEP=0, LAY=0).values
        full_lon = s["LON"].isel(TSTEP=0, LAY=0).values
    if args.region == "bay":
        rows, cols = bay_window(full_lat, full_lon)
        region_tag, region = "bay_area", "Bay Area (37.2-38.7 N, 121.2-123.1 W)"
    else:
        ring = slice(0, None) if args.include_ring else slice(RING, -RING)
        rows, cols = ring, ring
        region_tag = "full_read_area" if args.include_ring else "core_box"
        region = "full 72x72 read area" if args.include_ring else "64x64 core box"
    window = (rows, cols)
    lat, lon = full_lat[rows, cols], full_lon[rows, cols]
    # Projected grid coordinates (km), so the map is not skewed by the Lambert projection.
    with xr.open_dataset(sorted(data_dir.glob("equates_conc_*.nc"))[0]) as first:
        attrs = first.attrs
    n_row, n_col = full_lat.shape
    x_all = (attrs["XORIG"] + (np.arange(n_col) + 0.5) * attrs["XCELL"]) / 1000.0
    y_all = (attrs["YORIG"] + (np.arange(n_row) + 0.5) * attrs["YCELL"]) / 1000.0
    x, y = x_all[cols], y_all[rows]

    results = [(label, *period_means(data_dir, start, end, window)) for label, start, end in PERIODS]
    tag = region_tag
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, means, hours, days in results:
        slug = label.split(",")[0].split(" ")[0].lower() + "_" + label.split()[1].replace("-", "_") if "Aug" in label else "july_2019"
        for s in SPECIES:
            path = OUT_DIR / f"mean_{s.lower()}_{slug}_{tag}.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["row", "col", "lat", "lon", f"mean_{s.lower()}_ppbv"])
                for i in range(lat.shape[0]):
                    for j in range(lat.shape[1]):
                        writer.writerow([i, j, f"{lat[i, j]:.4f}", f"{lon[i, j]:.4f}", f"{means[s][i, j]:.3f}"])
        print(f"{label}: {days} days, {hours} hours | " + "; ".join(
            f"{s} mean {means[s].mean():.1f} (min {means[s].min():.1f}, max {means[s].max():.1f})" for s in SPECIES))

    limits = {s: (min(r[1][s].min() for r in results), max(r[1][s].max() for r in results)) for s in SPECIES}
    fig, axes = plt.subplots(len(PERIODS), len(SPECIES), figsize=(14, 13.5), constrained_layout=True)
    for row, (label, means, hours, days) in enumerate(results):
        for col, s in enumerate(SPECIES):
            ax = axes[row, col]
            vmin, vmax = limits[s]
            mesh = ax.pcolormesh(x, y, means[s], cmap="viridis", vmin=vmin, vmax=vmax, shading="auto")
            g = means[s]
            ax.set_title(f"{label}: {s} (ppbV)\nmean {g.mean():.1f}, min {g.min():.1f}, max {g.max():.1f}", fontsize=9)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel("y, km (Lambert grid)", fontsize=8)
            if row == len(PERIODS) - 1:
                ax.set_xlabel("x, km (Lambert grid)", fontsize=8)
            fig.colorbar(mesh, ax=ax, shrink=0.8, label="ppbV")
    fig.suptitle(f"EQUATES pilot, {region}: period-mean surface O3, NO2 and Ox (one colour scale per species)",
                 fontsize=12)
    fig_path = OUT_DIR / f"mean_concentrations_{tag}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"figure -> {fig_path}")


if __name__ == "__main__":
    main()
