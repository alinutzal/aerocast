"""Candidate high-Ox days for the abnormal-weather test, from the EQUATES pilot August files.

Per day, over the 64x64 core box at the surface layer: the peak hourly area-mean Ox (NO2 + O3,
ppbV), the 95th-percentile cell-hour Ox, and the meteorological drivers - peak area-mean 2 m
temperature (heat), mean wind speed and peak PBL height (stagnation). A day is a candidate when
its peak Ox is high and it is driven by heat or stagnation, using percentiles of the month
itself. The script only proposes days. Fire or smoke exclusion needs an outside fire record
that is not in this repository, so the fire_excluded column is left blank for a person to fill.

    uv run python scripts/abnormal_weather_candidates.py --month 8 --ox-percentile 85
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

RING = 4   # boundary cells on each side of the 64x64 core (configs/data/equates_2019_07_ca12km.yaml)
CORE = slice(RING, RING + 64)
OUT_DIR = Path("data/pilot_checks")


def daily_stats(conc_path, met_path):
    with xr.open_dataset(conc_path) as c, xr.open_dataset(met_path) as m:
        ox = (c["O3"] + c["NO2"]).isel(LAY=0, ROW=CORE, COL=CORE).values          # (24, 64, 64)
        temp = m["TEMP2"].isel(LAY=0, ROW=CORE, COL=CORE).values - 273.15         # K -> C
        wind = m["WSPD10"].isel(LAY=0, ROW=CORE, COL=CORE).values
        pbl = m["PBL"].isel(LAY=0, ROW=CORE, COL=CORE).values
    ox_area = ox.mean(axis=(1, 2))
    return {
        "ox_max_ppb": float(ox_area.max()),
        "ox_max_hour_index": int(ox_area.argmax()),
        "ox_p95_cell_ppb": float(np.percentile(ox, 95)),
        "t2_max_c": float(temp.mean(axis=(1, 2)).max()),
        "wind_mean_ms": float(wind.mean()),
        "calm_hours": int((wind.mean(axis=(1, 2)) < 2.0).sum()),
        "pbl_max_m": float(pbl.mean(axis=(1, 2)).max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--month", type=int, default=8)
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--ox-percentile", type=float, default=85, help="high-Ox threshold: this percentile of daily peak Ox")
    parser.add_argument("--data-dir", default=None, help="default: $AEROCAST_EQUATES_DIR/pilot/files")
    args = parser.parse_args()
    data_dir = Path(args.data_dir or os.path.join(os.environ.get("AEROCAST_EQUATES_DIR", ""), "pilot", "files"))
    if not data_dir.is_dir():
        raise SystemExit(f"No pilot files at {data_dir}; export AEROCAST_EQUATES_DIR or pass --data-dir.")

    stamp_prefix = f"{args.year}{args.month:02d}"
    days = sorted(p.name[len("equates_conc_"):-3] for p in data_dir.glob(f"equates_conc_{stamp_prefix}*.nc"))
    if not days:
        raise SystemExit(f"No concentration files for {stamp_prefix} in {data_dir}")
    rows = []
    for stamp in days:
        stats = daily_stats(data_dir / f"equates_conc_{stamp}.nc", data_dir / f"equates_meteo_{stamp}.nc")
        rows.append({"date": f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:]}", **stats})

    ox = np.array([r["ox_max_ppb"] for r in rows])
    t2 = np.array([r["t2_max_c"] for r in rows])
    wind = np.array([r["wind_mean_ms"] for r in rows])
    pbl = np.array([r["pbl_max_m"] for r in rows])
    thresholds = {
        "ox": np.percentile(ox, args.ox_percentile),
        "t2": np.percentile(t2, 90),
        "wind": np.percentile(wind, 25),
        "pbl": np.percentile(pbl, 25),
    }
    for r in rows:
        r["high_ox"] = r["ox_max_ppb"] >= thresholds["ox"]
        r["heat"] = r["t2_max_c"] >= thresholds["t2"]
        r["stagnation"] = r["wind_mean_ms"] <= thresholds["wind"] or r["pbl_max_m"] <= thresholds["pbl"]
        r["candidate"] = r["high_ox"] and (r["heat"] or r["stagnation"])
        r["fire_excluded"] = ""

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["date", "ox_max_ppb", "ox_max_hour_index", "ox_p95_cell_ppb", "t2_max_c", "wind_mean_ms",
              "calm_hours", "pbl_max_m", "high_ox", "heat", "stagnation", "candidate", "fire_excluded"]
    csv_path = OUT_DIR / f"abnormal_weather_candidates_{stamp_prefix}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)

    dates = [r["date"][-2:] for r in rows]
    cand = np.array([r["candidate"] for r in rows])
    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    series = [
        ("peak area-mean Ox (ppbV)", ox, thresholds["ox"], "high-Ox threshold"),
        ("peak area-mean 2 m T (°C)", t2, thresholds["t2"], "heat (90th pct)"),
        ("mean wind speed (m/s)", wind, thresholds["wind"], "stagnation (25th pct)"),
        ("peak PBL height (m)", pbl, thresholds["pbl"], "stagnation (25th pct)"),
    ]
    x = np.arange(len(rows))
    for ax, (label, values, threshold, note) in zip(axes, series):
        ax.plot(x, values, marker="o", ms=3, color="0.3", lw=1)
        ax.axhline(threshold, ls="--", color="tab:blue", lw=1, label=note)
        ax.scatter(x[cand], values[cand], color="tab:red", zorder=3, s=40, label="candidate")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(dates, fontsize=7)
    axes[-1].set_xlabel(f"day of {args.year}-{args.month:02d} (core 64x64 box, surface layer)")
    axes[0].set_title("EQUATES pilot: candidate high-Ox days (before fire exclusion)")
    fig.tight_layout()
    fig_path = OUT_DIR / f"abnormal_weather_candidates_{stamp_prefix}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"{len(rows)} days. Thresholds: Ox >= {thresholds['ox']:.1f} ppbV, T2 >= {thresholds['t2']:.1f} C, "
          f"wind <= {thresholds['wind']:.2f} m/s, PBL <= {thresholds['pbl']:.0f} m")
    print("Candidates:")
    for r in rows:
        if r["candidate"]:
            drivers = "/".join(d for d, on in (("heat", r["heat"]), ("stagnation", r["stagnation"])) if on)
            print(f"  {r['date']}  Ox {r['ox_max_ppb']:6.1f} ppbV  T2 {r['t2_max_c']:5.1f} C  "
                  f"wind {r['wind_mean_ms']:4.2f} m/s  PBL {r['pbl_max_m']:6.0f} m  [{drivers}]")
    print(f"Wrote {csv_path} and {fig_path}")


if __name__ == "__main__":
    main()
