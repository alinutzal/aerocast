"""Load CMAQ day files, align them on common hours, cache the arrays and build windows.

Each day has three files (meteorology, concentrations, emissions). Their TFLAG
timestamps, shifted by `data.time_offset_hours`, are intersected so every hour kept
has all three sources. Days are concatenated in time order; a window is only built
over consecutive hours, so gaps between days are never bridged.

Channels are named "source:VAR". File sources are meteo (METCRO2D), conc (out.combine)
and emis (emissions), so the emission and concentration NO/NO2 stay distinct. Derived
channels: meteo:U10/meteo:V10 (from WSPD10 and WDIR10), time:HOUR_SIN/time:HOUR_COS
(local hour of day) and grid:X/grid:Y (grid coordinates scaled to [0, 1]).
"""
import fnmatch
import glob
import hashlib
import json
import os
import re
import warnings
from dataclasses import dataclass
from functools import reduce
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

SOURCES = ("meteo", "conc", "emis")
DERIVED_WIND = ("U10", "V10")
TIME_CHANNELS = ("HOUR_SIN", "HOUR_COS")
GRID_CHANNELS = ("X", "Y")
HOUR = np.timedelta64(1, "h")
CACHE_VERSION = 2


def to_day(value):
    """'2018-11-13', '20181113', a date or a datetime64 -> numpy datetime64[D]."""
    text = str(value)
    if re.fullmatch(r"\d{8}", text):
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return np.datetime64(text, "D")


def _stamp(day):
    return str(to_day(day)).replace("-", "")


def hour_of_day(times):
    """Hour of day (0-23) of datetime64[h] timestamps."""
    return ((times - times.astype("datetime64[D]")) // HOUR).astype(np.int64)


def split_channel(name):
    """'emis:NO' -> ('emis', 'NO')."""
    if not isinstance(name, str):
        raise TypeError(f"Channel {name!r} is not a string; quote it in the YAML")
    source, sep, var = name.partition(":")
    if not sep or not var or source not in SOURCES + ("time", "grid"):
        raise ValueError(f"Channel {name!r} must look like source:VAR with source in {SOURCES + ('time', 'grid')}")
    if source == "time" and var not in TIME_CHANNELS:
        raise ValueError(f"Unknown time channel {name!r}; expected one of {TIME_CHANNELS}")
    if source == "grid" and var not in GRID_CHANNELS:
        raise ValueError(f"Unknown grid channel {name!r}; expected one of {GRID_CHANNELS}")
    return source, var


def channel_layout(data_cfg, features_cfg):
    """Channel names in model order: {"state": [...], "forcing": [...], "static": [...]}."""
    inputs = data_cfg["inputs"]
    state = list(inputs["state"])
    if features_cfg.get("include_o3_input") and "conc:O3" not in state:
        state.append("conc:O3")
    layout = {"state": state, "forcing": list(inputs["forcing"]), "static": list(inputs.get("static") or [])}
    for group, names in layout.items():
        for name in names:
            source, _ = split_channel(name)
            if (source == "grid") != (group == "static"):
                raise ValueError(f"{name}: grid channels, and only they, belong in data.inputs.static")
            if source == "time" and group != "forcing":
                raise ValueError(f"{name}: time channels are forcings")
    return layout


def source_variables(data_cfg, features_cfg):
    """Raw variables to read from each file: file-backed channels, WSPD10/WDIR10 for the
    derived winds, and the target species (from conc)."""
    wanted = {source: [] for source in SOURCES}
    layout = channel_layout(data_cfg, features_cfg)
    for name in layout["state"] + layout["forcing"]:
        source, var = split_channel(name)
        if source == "meteo" and var in DERIVED_WIND:
            wanted["meteo"] += ["WSPD10", "WDIR10"]
        elif source in SOURCES:
            wanted[source].append(var)
    wanted["conc"] += [str(v) for v in data_cfg["target"]]
    return {source: list(dict.fromkeys(names)) for source, names in wanted.items()}


def day_files(data_cfg, day):
    """Paths of the meteo/conc/emis files for one day; each pattern must match exactly one file."""
    files = {}
    for source in SOURCES:
        pattern = str(Path(data_cfg["data_dir"]) / data_cfg["files"][source].format(date=_stamp(day)))
        matches = sorted(glob.glob(pattern))
        if len(matches) != 1:
            raise FileNotFoundError(f"{source} file for {to_day(day)}: expected one match for {pattern}, found {len(matches)}")
        files[source] = Path(matches[0])
    return files


def available_days(data_cfg):
    """Days in data_dir for which all three files exist."""
    meteo_pattern = data_cfg["files"]["meteo"]
    candidates = set()
    for path in glob.glob(str(Path(data_cfg["data_dir"]) / meteo_pattern.format(date="[0-9]" * 8))):
        name = Path(path).name
        for stamp in re.findall(r"(?=(\d{8}))", name):
            if fnmatch.fnmatch(name, meteo_pattern.format(date=stamp)):
                candidates.add(to_day(stamp))
    days = []
    for day in sorted(candidates):
        try:
            day_files(data_cfg, day)
        except FileNotFoundError:
            continue
        days.append(day)
    return days


def ioapi_times(ds):
    """UTC timestamp (datetime64[h]) of each TSTEP, from TFLAG or the SDATE/STIME/TSTEP attributes."""
    if "TFLAG" in ds:
        flags = np.asarray(ds["TFLAG"].values)[:, 0, :].astype(np.int64)
        yyyyddd, hhmmss = flags[:, 0], flags[:, 1]
    else:
        if int(ds.attrs["TSTEP"]) != 10000:
            raise ValueError(f"Only hourly files are supported (TSTEP={ds.attrs['TSTEP']})")
        n = ds.sizes["TSTEP"]
        start = _ioapi_to_datetime(np.array([int(ds.attrs["SDATE"])]), np.array([int(ds.attrs["STIME"])]))[0]
        return start + np.arange(n) * HOUR
    if np.any(hhmmss % 10000):
        raise ValueError("Only whole-hour time steps are supported")
    return _ioapi_to_datetime(yyyyddd, hhmmss)


def _ioapi_to_datetime(yyyyddd, hhmmss):
    years = (yyyyddd // 1000 - 1970).astype("datetime64[Y]")
    days = years.astype("datetime64[D]") + (yyyyddd % 1000 - 1).astype("timedelta64[D]")
    return days.astype("datetime64[h]") + (hhmmss // 10000).astype("timedelta64[h]")


def _read_source(path, source, variables, offset_hours):
    with xr.open_dataset(path, engine="netcdf4", decode_times=False) as ds:
        times = ioapi_times(ds) + int(offset_hours) * HOUR
        if np.any(np.diff(times) <= np.timedelta64(0, "h")):
            raise ValueError(f"{path.name}: time steps are not increasing")
        fields, units = {}, {}
        for name in variables:
            if name not in ds:
                raise KeyError(f"Variable {name} not found in {path.name}")
            da = ds[name]
            if "LAY" in da.dims:
                da = da.isel(LAY=0)  # surface layer
            if da.ndim != 3:
                raise ValueError(f"{name} in {path.name} has unexpected dims {da.dims}")
            fields[f"{source}:{name}"] = np.asarray(da.values, dtype=np.float32)
            units[f"{source}:{name}"] = str(da.attrs.get("units", "")).strip()
    return times, fields, units


def _add_wind_components(fields, units):
    """meteo:U10/V10 from 10 m speed and direction (meteorological convention: the direction
    the wind blows from, clockwise from north). The grid's rotation from true north is under
    2 degrees on this domain and is ignored, so U10/V10 are treated as grid-relative."""
    if "meteo:WSPD10" in fields and "meteo:WDIR10" in fields:
        speed, direction = fields["meteo:WSPD10"], np.deg2rad(fields["meteo:WDIR10"])
        fields["meteo:U10"] = (-speed * np.sin(direction)).astype(np.float32)
        fields["meteo:V10"] = (-speed * np.cos(direction)).astype(np.float32)
        units["meteo:U10"] = units["meteo:V10"] = units["meteo:WSPD10"]


def _cache_path(data_cfg, day, files, variables, offsets):
    if not data_cfg.get("cache_dir"):
        return None
    key = {
        "version": CACHE_VERSION,
        "day": str(to_day(day)),
        "files": {s: [str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns] for s, p in files.items()},
        "variables": variables,
        "offsets": offsets,
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return Path(data_cfg["cache_dir"]) / f"{_stamp(day)}_{digest}.npz"


def load_day(data_cfg, features_cfg, day):
    """One day's fields ("source:VAR") on the hours common to all three files:
    {"times", "fields", "units"}. Cached as .npz under data.cache_dir, so NetCDF is read once."""
    files = day_files(data_cfg, day)
    variables = source_variables(data_cfg, features_cfg)
    offsets = {s: int((data_cfg.get("time_offset_hours") or {}).get(s, 0)) for s in SOURCES}
    cache = _cache_path(data_cfg, day, files, variables, offsets)
    if cache is not None and cache.exists():
        with np.load(cache) as z:
            return {
                "times": z["times"].astype("datetime64[h]"),
                "fields": {k[2:]: z[k] for k in z.files if k.startswith("f_")},
                "units": json.loads(str(z["units"])),
            }

    read = {s: _read_source(files[s], s, variables[s], offsets[s]) for s in SOURCES}
    common = reduce(np.intersect1d, [times for times, _, _ in read.values()])
    if len(common) == 0:
        raise ValueError(f"{to_day(day)}: the three files share no hours; check data.time_offset_hours")
    if len(common) < 24:
        counts = {s: len(times) for s, (times, _, _) in read.items()}
        warnings.warn(f"{to_day(day)}: only {len(common)} hours common to all files {counts}; check data.time_offset_hours")

    fields, units = {}, {}
    for times, source_fields, source_units in read.values():
        index = np.searchsorted(times, common)
        for name, values in source_fields.items():
            fields[name] = values[index]
        units.update(source_units)
    _add_wind_components(fields, units)
    shapes = {values.shape[1:] for values in fields.values()}
    if len(shapes) != 1:
        raise ValueError(f"{to_day(day)}: grids differ between files: {shapes}")

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "wb") as f:
            np.savez(f, times=common.astype(np.int64), units=json.dumps(units),
                     **{f"f_{name}": values for name, values in fields.items()})
        os.replace(tmp, cache)
    return {"times": common, "fields": fields, "units": units}


@dataclass
class HourlyData:
    times: np.ndarray    # (T,) datetime64[h] UTC after time offsets, strictly increasing
    days: np.ndarray     # (T,) datetime64[D] date of the file each hour came from
    state: np.ndarray    # (T, C_state, H, W) float32, native units
    forcing: np.ndarray  # (T, C_forcing, H, W) float32, native units
    static: np.ndarray   # (C_static, H, W) float32
    targets: dict        # "NO2", "O3", ..., "Ox" -> (T, H, W) float32, native units
    channels: dict       # {"state": [...], "forcing": [...], "static": [...]}
    target_unit: str


def grid_channels(names, height, width):
    """grid:X (column) and grid:Y (row) scaled to [0, 1]: (len(names), H, W)."""
    yy, xx = np.meshgrid(np.linspace(0, 1, height), np.linspace(0, 1, width), indexing="ij")
    values = {"grid:X": xx, "grid:Y": yy}
    return np.stack([values[n] for n in names]).astype(np.float32) if names else np.zeros((0, height, width), np.float32)


def load_hourly(data_cfg, features_cfg, days):
    """Concatenate the given days into one hourly series of state, forcing, static and targets."""
    days = sorted(to_day(d) for d in days)
    layout = channel_layout(data_cfg, features_cfg)
    parts = [load_day(data_cfg, features_cfg, day) for day in days]

    times = np.concatenate([p["times"] for p in parts])
    if np.any(np.diff(times) <= np.timedelta64(0, "h")):
        raise ValueError("Hours overlap between days; check data.time_offset_hours")
    day_of_hour = np.concatenate([np.full(len(p["times"]), day) for p, day in zip(parts, days)])
    height, width = next(iter(parts[0]["fields"].values())).shape[1:]

    local_hour = hour_of_day(times + int(data_cfg.get("local_utc_offset_hours", 0)) * HOUR)
    angle = 2 * np.pi * local_hour / 24
    time_values = {"time:HOUR_SIN": np.sin(angle).astype(np.float32), "time:HOUR_COS": np.cos(angle).astype(np.float32)}

    def column(name):
        if name in time_values:
            return np.broadcast_to(time_values[name][:, None, None], (len(times), height, width))
        return np.concatenate([p["fields"][name] for p in parts])

    def stack(names):
        if not names:
            return np.zeros((len(times), 0, height, width), dtype=np.float32)
        return np.stack([column(name) for name in names], axis=1).astype(np.float32)

    species = [str(v) for v in data_cfg["target"]]
    targets = {v: np.concatenate([p["fields"][f"conc:{v}"] for p in parts]) for v in species}
    targets["Ox"] = sum(targets[v] for v in species).astype(np.float32)
    target_units = {parts[0]["units"][f"conc:{v}"] for v in species}
    if len(target_units) != 1:
        raise ValueError(f"Target variables {species} have different units: {target_units}")
    return HourlyData(times, day_of_hour, stack(layout["state"]), stack(layout["forcing"]),
                      grid_channels(layout["static"], height, width), targets, layout, target_units.pop())


def window_starts(times, seq_len, pred_len, labels=None, label=None):
    """Index t of the first target hour of every window whose hours t-seq_len .. t+pred_len-1
    are consecutive and, if labels are given, all equal to `label`."""
    span = seq_len + pred_len
    starts = []
    for s in range(len(times) - span + 1):
        end = s + span - 1
        if times[end] - times[s] != (span - 1) * HOUR:
            continue
        if labels is not None and not np.all(labels[s:end + 1] == label):
            continue
        starts.append(s + seq_len)
    return np.asarray(starts, dtype=np.int64)


class WindowDataset(Dataset):
    """Samples {"state", "forcing", "static", "target"} from normalized hourly arrays.

    state: (T_in, C_state, H, W) for the input hours t-T_in .. t-1.
    forcing: (T_in + T_out, C_forcing, H, W) for hours t-T_in .. t+T_out-1; without
        future_forcings the forecast hours repeat hour t-1, so no future values are seen.
    static: (C_static, H, W). target: (T_out, K, H, W) from targets (T, K, H, W).
    """

    def __init__(self, state, forcing, static, targets, starts, t_in, t_out, future_forcings=True):
        self.state, self.forcing, self.static, self.targets = state, forcing, torch.from_numpy(static), targets
        self.starts = np.asarray(starts, dtype=np.int64)
        self.t_in, self.t_out, self.future_forcings = t_in, t_out, future_forcings

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = int(self.starts[i])
        if self.future_forcings:
            forcing = self.forcing[t - self.t_in:t + self.t_out]
        else:
            forcing = np.concatenate([self.forcing[t - self.t_in:t], np.repeat(self.forcing[t - 1:t], self.t_out, axis=0)])
        return {
            "state": torch.from_numpy(np.ascontiguousarray(self.state[t - self.t_in:t])),
            "forcing": torch.from_numpy(np.ascontiguousarray(forcing)),
            "static": self.static,
            "target": torch.from_numpy(np.ascontiguousarray(self.targets[t:t + self.t_out])),
        }
