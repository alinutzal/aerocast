"""Aug 14-16 episode vs the August mean on the EQUATES pilot read area, in the layout of the
CA 4 km figure: MDA8 O3, daily max 2 m temperature and 24 h mean 10 m wind, for Aug 15, the
Aug 14-16 mean and the August mean (Aug 1-30).

The daily metrics are derived from the pilot's hourly fields, on local standard time (UTC-8):
  MDA8 O3       max over the day's 8 h running means of surface O3 (ppbV)
  daily max T   max of TEMP2 over the day (deg C)
  24 h mean wind mean of WSPD10 over the day (m/s)
Periods average the daily values. The pilot covers only part of the figure's domain (its read
area is about 34-43.5 N, 116-128 W), so the rest of the frame stays blank. The default frame is
read from the figure (about 125 to 114 W, 32.5 to 42.5 N); pass --extent to change it.

    uv run python scripts/episode_vs_august_pilot.py
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

OUT_DIR = Path("data/pilot_checks")
UTC_OFFSET = -8  # local standard time for California
EPISODE = (14, 16)
AUGUST = (1, 30)
PANELS = [("MDA8 O$_3$ (ppb)", "o3", (20, 90)),
          ("Daily max 2-m T (°C)", "t", (10, 45)),
          ("24-h mean 10-m wind (m s$^{-1}$)", "wind", (0, 8))]


def hourly_fields(data_dir, days):
    """Hourly O3, TEMP2 and WSPD10 (UTC) on the full read area for the given August days."""
    o3, temp, wind = [], [], []
    for d in days:
        with xr.open_dataset(data_dir / f"equates_conc_201908{d:02d}.nc") as c:
            o3.append(c["O3"].isel(LAY=0).values)
        with xr.open_dataset(data_dir / f"equates_meteo_201908{d:02d}.nc") as m:
            temp.append(m["TEMP2"].isel(LAY=0).values - 273.15)
            wind.append(m["WSPD10"].isel(LAY=0).values)
    return np.concatenate(o3), np.concatenate(temp), np.concatenate(wind)


def daily_metrics(o3, temp, wind, day):
    """MDA8, daily max T and 24 h mean wind for one local day (day counted from 1 August)."""
    start = (day - 1) * 24 - UTC_OFFSET  # first UTC hour of the local day
    window = o3[start:start + 24 + 7]
    running8 = np.stack([window[i:i + 8].mean(axis=0) for i in range(24)])  # 8 h means starting in the day
    return {"o3": running8.max(axis=0),
            "t": temp[start:start + 24].max(axis=0),
            "wind": wind[start:start + 24].mean(axis=0)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="default: $AEROCAST_EQUATES_DIR/pilot/files")
    parser.add_argument("--extent", nargs=4, type=float, default=[-125.0, -114.0, 32.5, 42.5],
                        metavar=("LON_W", "LON_E", "LAT_S", "LAT_N"))
    args = parser.parse_args()
    data_dir = Path(args.data_dir or os.path.join(os.environ.get("AEROCAST_EQUATES_DIR", ""), "pilot", "files"))

    with xr.open_dataset(data_dir / "static.nc") as s:
        lat = s["LAT"].isel(TSTEP=0, LAY=0).values
        lon = s["LON"].isel(TSTEP=0, LAY=0).values
    o3, temp, wind = hourly_fields(data_dir, range(1, 32))
    daily = {d: daily_metrics(o3, temp, wind, d) for d in range(AUGUST[0], AUGUST[1] + 1)}

    def average(days):
        return {k: np.mean([daily[d][k] for d in days], axis=0) for k in ("o3", "t", "wind")}

    columns = [("Aug 15", daily[15]),
               (f"Aug {EPISODE[0]}-{EPISODE[1]} mean", average(range(EPISODE[0], EPISODE[1] + 1))),
               (f"August mean (1-{AUGUST[1]})", average(range(AUGUST[0], AUGUST[1] + 1)))]

    fig, axes = plt.subplots(3, 3, figsize=(13.5, 12), constrained_layout=True)
    west, east, south, north = args.extent
    for row, (label, key, (vmin, vmax)) in enumerate(PANELS):
        for col, (title, values) in enumerate(columns):
            ax = axes[row, col]
            mesh = ax.pcolormesh(lon, lat, values[key], cmap="jet", vmin=vmin, vmax=vmax, shading="auto")
            ax.set_xlim(west, east)
            ax.set_ylim(south, north)
            ax.set_aspect(1 / np.cos(np.deg2rad((south + north) / 2)))
            ax.set_title(title, fontsize=10)
            ax.tick_params(labelsize=7)
            if col == 0:
                ax.set_ylabel(label, fontsize=9)
            if row == 2:
                ax.set_xlabel("longitude", fontsize=8)
        fig.colorbar(mesh, ax=axes[row, :], shrink=0.9, extend="both", label=label)
        values = [v[key] for _, v in columns]
        print(f"{label}: " + " | ".join(f"{t} mean {v.mean():.1f} (min {v.min():.1f}, max {v.max():.1f})"
                                       for (t, _), v in zip(columns, values)))
    fig.suptitle("Aug 14-16, 2019 episode vs August mean, EQUATES pilot read area (12 km)", fontsize=12)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "episode_vs_august_pilot.png"
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


if __name__ == "__main__":
    main()
