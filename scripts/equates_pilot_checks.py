"""Sanity checks and figures for the EQUATES pilot dataset. Prints the checks and writes a
short report with figures to data/pilot_checks/.

Usage: python scripts/equates_pilot_checks.py --config configs/data/equates_2019_07_ca12km.yaml
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aerocast.config import load_config  # noqa: E402
from aerocast.data import available_days, load_hourly  # noqa: E402

PLAUSIBLE_RANGES = {
    "conc:NO": (0, 200, "ppbV"), "conc:NO2": (0, 150, "ppbV"), "conc:O3": (0, 150, "ppbV"),
    "conc:ATOTIJ": (0, 200, "ug/m3"),
    "meteo:TEMP2": (250, 330, "K"), "meteo:WSPD10": (0, 30, "m/s"),
    "meteo:PBL": (0, 4000, "m"), "meteo:RGRND": (0, 1200, "W/m2"), "meteo:Q2": (0, 0.03, "kg/kg"),
}
OUT_DIR = Path("data/pilot_checks")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/equates_2019_07_ca12km.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg, features_cfg = cfg["data"], cfg["features"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    days = available_days(data_cfg)
    print(f"Found {len(days)} days: {days[0]} .. {days[-1]}" if days else "No days found.")
    hd = load_hourly(data_cfg, features_cfg, days)
    H, W = hd.state.shape[-2:]
    report = [f"# EQUATES pilot checks\n\n{len(days)} days, {hd.state.shape[0]} hours, grid {H}x{W}.\n"]

    # --- identical timestamps across streams ---------------------------------------------
    # load_hourly already intersects hours across all configured sources per day and raises
    # if a day shares no hours; report the per-day hour counts actually used.
    counts = {}
    for day in days:
        counts[str(day)] = int(np.sum(hd.days == day))
    short_days = {d: c for d, c in counts.items() if c != 24}
    line = ("All streams share all 24 hours on every day (identical TFLAG timestamps after "
            "alignment)." if not short_days else f"Days with fewer than 24 shared hours: {short_days}")
    print(line)
    report.append(f"## Timestamp alignment\n\n{line}\n")

    # --- NaNs, negatives, ranges ------------------------------------------------------------
    report.append("## Value checks\n\n| channel | NaNs | negative count | min | max | plausible range |\n"
                   "|---|---|---|---|---|---|")
    all_channels = {**{n: hd.state[:, i] for i, n in enumerate(hd.channels["state"])},
                     **{n: hd.forcing[:, i] for i, n in enumerate(hd.channels["forcing"])}}
    for name, arr in all_channels.items():
        n_nan = int(np.isnan(arr).sum())
        n_neg = int((arr < 0).sum())
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        plausible = PLAUSIBLE_RANGES.get(name)
        flag = ""
        if plausible:
            plo, phi, unit = plausible
            flag = "OK" if plo - 1e-6 <= lo and hi <= phi + 1e-6 else "OUT OF RANGE"
            plausible_str = f"{plo}-{phi} {unit} [{flag}]"
        else:
            plausible_str = "-"
        print(f"{name:16s} NaNs={n_nan:6d}  negatives={n_neg:8d}  range=[{lo:9.3f}, {hi:9.3f}]  {plausible_str}")
        report.append(f"| {name} | {n_nan} | {n_neg} | {lo:.3f} | {hi:.3f} | {plausible_str} |")
    total_nan = sum(int(np.isnan(a).sum()) for a in all_channels.values())
    report.append(f"\nTotal NaNs across all channels: {total_nan}.\n")

    # --- diurnal cycle (local solar time) ---------------------------------------------------
    lon = None
    static_path = Path(data_cfg["data_dir"]) / data_cfg["static_file"]
    with xr.open_dataset(static_path, engine="netcdf4") as sds:
        lon = np.squeeze(np.asarray(sds["LON"].values))
        lat = np.squeeze(np.asarray(sds["LAT"].values))

    utc_hour = (hd.times.astype(np.int64) % 24)
    local_hour = np.round((utc_hour[:, None, None] + lon[None, :, :] / 15.0) % 24).astype(int) % 24

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, values in [("O3", hd.targets["O3"]), ("NO2", hd.targets["NO2"]), ("Ox", hd.targets["Ox"])]:
        means = [values[local_hour == h].mean() for h in range(24)]
        ax.plot(range(24), means, marker="o", ms=3, label=name)
    ax.set_xlabel("local solar hour"); ax.set_ylabel("domain-mean (ppbV)")
    ax.set_title("Domain-mean diurnal cycle (local solar time), July 2019")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / "diurnal_cycle.png", dpi=150); plt.close(fig)
    print("wrote", OUT_DIR / "diurnal_cycle.png")

    o3_peak_hour = int(np.argmax([hd.targets["O3"][local_hour == h].mean() for h in range(24)]))
    no2_peak_hour = int(np.argmax([hd.targets["NO2"][local_hour == h].mean() for h in range(24)]))
    cycle_note = (f"O3 peaks at local hour {o3_peak_hour}:00, NO2 peaks at local hour {no2_peak_hour}:00 "
                  "(domain mean over the whole read area, so urban NO2 morning-peak and rural/mixed "
                  "O3 afternoon-peak signals are blended together, not separated by land use).")
    print(cycle_note)
    report.append(f"## Diurnal cycle\n\n![diurnal cycle](diurnal_cycle.png)\n\n{cycle_note}\n")

    # --- July-mean maps -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (name, values) in zip(axes, [("O3 (ppbV)", hd.targets["O3"]), ("NO2 (ppbV)", hd.targets["NO2"]),
                                          ("Ox (ppbV)", hd.targets["Ox"])]):
        im = ax.pcolormesh(values.mean(axis=0), cmap="viridis")
        ax.set_title(f"July-mean {name}")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("July 2019 mean concentration maps, EQUATES pilot domain (row/col indices, not lat/lon)")
    fig.tight_layout(); fig.savefig(OUT_DIR / "july_mean_maps.png", dpi=150); plt.close(fig)
    print("wrote", OUT_DIR / "july_mean_maps.png")
    report.append("## July-mean maps\n\n![july mean maps](july_mean_maps.png)\n\n"
                   "NO emissions map skipped: this pilot has no emissions stream (see DATA_CARD.md "
                   "'Known gaps' - the only 2019 gridded emissions found on AWS cover 10 scattered "
                   "July days, not the full month).\n")

    o3_map = hd.targets["O3"].mean(axis=0)
    no2_map = hd.targets["NO2"].mean(axis=0)
    corner_desc = (f"O3 is higher in the {'north' if o3_map[-8:,:].mean() > o3_map[:8,:].mean() else 'south'} "
                   f"of the domain than the {'south' if o3_map[-8:,:].mean() > o3_map[:8,:].mean() else 'north'} "
                   f"(mean {o3_map[-8:,:].mean():.1f} vs {o3_map[:8,:].mean():.1f} ppbV in the north/south 8-cell "
                   f"edge strips); NO2 is higher in the "
                   f"{'north' if no2_map[-8:,:].mean() > no2_map[:8,:].mean() else 'south'} similarly "
                   f"(mean {no2_map[-8:,:].mean():.1f} vs {no2_map[:8,:].mean():.1f} ppbV). This reflects "
                   "the mix of urban (Bay Area/Sacramento) and rural/valley cells in the read area, not "
                   "a single clean urban-vs-rural gradient - see the map for the actual spatial pattern "
                   "rather than assuming a textbook NO2-source/O3-downwind split.")
    print(corner_desc)
    report.append(corner_desc + "\n")

    (OUT_DIR / "report.md").write_text("\n".join(report))
    print(f"\nWrote {OUT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
