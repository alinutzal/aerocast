"""Write synthetic CMAQ-like day files for tests and dry runs.

Same file names, variable names, dims (TSTEP, LAY, ROW, COL), units and time stamps as the
BAAQMD data: METCRO2D and emissions have 25 hourly steps stamped 08Z-08Z (00-24 local
standard time), out.combine has 24 steps stamped 00Z-23Z that hold 00-23 local time.

    uv run python scripts/make_synthetic_data.py --out-dir /path/to/synthetic --days 3
"""
import argparse
import zlib
from pathlib import Path

import numpy as np
import xarray as xr

UTC_OFFSET = 8  # local standard time (PST) = UTC - 8
METEO = {"TEMP2": "K", "WSPD10": "M/S", "WDIR10": "DEGREES"}
CONC = {"NO": "ppbV", "NO2": "ppbV", "PM25_CL": "ug m-3", "O3": "ppbV"}
EMIS_SCALE = {"NO": 0.3, "NO2": 0.03, "ALK1": 0.5, "OLE1": 0.004, "ARO1": 0.006, "ARO2": 0.005,
              "TERP": 0.0005, "ISOP": 0.00008}


def _noise(kind, name, hours, shape, seed):
    """Deterministic per (variable, hour), so an hour repeated in two files gets the same values."""
    code = zlib.crc32(f"{kind}:{name}".encode())
    return np.stack([np.random.default_rng([seed, code, int(g)]).normal(size=shape) for g in hours])


def _fields(kind, names, hours, rows, cols, seed):
    """Diurnal cycles x spatial patterns + noise. `hours` count local hours from the first midnight."""
    g = np.asarray(hours)[:, None, None]
    h, day = g % 24, g // 24
    yy, xx = np.meshgrid(np.linspace(0, 1, rows), np.linspace(0, 1, cols), indexing="ij")
    urban = np.exp(-((yy - 0.4) ** 2 + (xx - 0.6) ** 2) / 0.05)

    def cycle(peak_hour):
        return np.sin(2 * np.pi * (h - peak_hour + 6) / 24)  # maximum at peak_hour

    fields = {}
    for name in names:
        noise = _noise(kind, name, hours, (rows, cols), seed)
        if kind == "meteo":
            fields[name] = {
                "TEMP2": 283 + 5 * cycle(15) + 3 * xx + 0.5 * day + 0.3 * noise,
                "WSPD10": np.abs(3 + 1.5 * cycle(15) + 2 * xx + 0.3 * noise),
                "WDIR10": (230 + 40 * np.sin(2 * np.pi * g / 36) + 30 * xx + 5 * noise) % 360,
            }[name]
        elif kind == "conc":
            fields[name] = {
                "NO": np.maximum(0.01, 2 + 6 * urban * np.exp(-(((h - 8) / 2.5) ** 2)) + 0.2 * noise),
                "NO2": 9 + 6 * urban + 3 * cycle(20) + 0.5 * day + 0.5 * noise,
                "PM25_CL": 0.08 + 0.05 * urban + 0.01 * np.abs(noise),
                "O3": 31 + 7 * cycle(13) - 5 * urban + 1.5 * day + 0.7 * noise,
            }[name]
        else:
            activity = 0.3 + np.clip(np.sin(np.pi * (h - 5) / 14), 0, None)
            fields[name] = np.maximum(0, EMIS_SCALE[name] * (urban * activity + 0.02 * noise))
        fields[name] = np.broadcast_to(fields[name], (len(hours), rows, cols)).astype(np.float32)
    return fields


def _ioapi_dataset(fields, units, first_stamp, n_steps, description):
    """IOAPI-style dataset: variables (TSTEP, LAY, ROW, COL), TFLAG, SDATE/STIME/TSTEP attributes."""
    stamps = first_stamp + np.arange(n_steps) * np.timedelta64(1, "h")
    days = stamps.astype("datetime64[D]")
    years = days.astype("datetime64[Y]")
    yyyyddd = (years.astype(int) + 1970) * 1000 + (days - years.astype("datetime64[D]")).astype(int) + 1
    hhmmss = (stamps - days).astype(int) * 10000
    tflag = np.repeat(np.stack([yyyyddd, hhmmss], axis=1)[:, None, :], len(fields), axis=1).astype(np.int32)
    data_vars = {"TFLAG": (("TSTEP", "VAR", "DATE-TIME"), tflag, {"units": "<YYYYDDD,HHMMSS>"})}
    for name, values in fields.items():
        data_vars[name] = (("TSTEP", "LAY", "ROW", "COL"), values[:, None], {"units": units[name], "long_name": name})
    rows, cols = next(iter(fields.values())).shape[1:]
    attrs = {"SDATE": np.int32(yyyyddd[0]), "STIME": np.int32(hhmmss[0]), "TSTEP": np.int32(10000),
             "NROWS": np.int32(rows), "NCOLS": np.int32(cols), "NLAYS": np.int32(1),
             "NVARS": np.int32(len(fields)), "FILEDESC": description}
    return xr.Dataset(data_vars, attrs=attrs)


def write_day(out_dir, day, day_index, rows, cols, seed):
    out_dir = Path(out_dir)
    stamp = str(day).replace("-", "")
    local_midnight_utc = day.astype("datetime64[h]") + UTC_OFFSET
    hours_25 = day_index * 24 + np.arange(25)  # 00..24 local
    hours_24 = hours_25[:24]                   # 00..23 local

    meteo = _fields("meteo", METEO, hours_25, rows, cols, seed)
    _ioapi_dataset(meteo, METEO, local_midnight_utc, 25, "synthetic METCRO2D").to_netcdf(
        out_dir / f"METCRO2D_{stamp}.nc")

    conc = _fields("conc", CONC, hours_24, rows, cols, seed)
    # Like the real out.combine files: local-time data stamped as if it were 00Z-23Z.
    _ioapi_dataset(conc, CONC, day.astype("datetime64[h]"), 24, "synthetic out.combine").to_netcdf(
        out_dir / f"out.combine_{stamp}.nc")

    emis = _fields("emis", EMIS_SCALE, hours_25, rows, cols, seed)
    _ioapi_dataset(emis, {name: "moles/s" for name in EMIS_SCALE}, local_midnight_utc, 25,
                   "synthetic emissions").to_netcdf(out_dir / f"egts_l.{stamp}.1.1km.synthetic.ncf")


def make_synthetic_data(out_dir, start="2018-11-13", n_days=3, rows=16, cols=12, seed=0):
    """Write n_days consecutive days of files to out_dir; returns the dates written."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    first = np.datetime64(start, "D")
    days = [first + i for i in range(n_days)]
    for i, day in enumerate(days):
        write_day(out_dir, day, i, rows, cols, seed)
    return [str(d) for d in days]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--start", default="2018-11-13", help="first date, YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    days = make_synthetic_data(args.out_dir, args.start, args.days, args.rows, args.cols, args.seed)
    print(f"Wrote {len(days)} days ({days[0]} .. {days[-1]}) to {args.out_dir}")


if __name__ == "__main__":
    main()
