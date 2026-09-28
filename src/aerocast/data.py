"""Load CMAQ day files, align them on common hours, cache the arrays and build windows.

Each day has three files (meteorology, concentrations, emissions). Their TFLAG
timestamps, shifted by `data.time_offset_hours`, are intersected so every hour kept
has all three sources. Days are concatenated in time order; a window is only built
over consecutive hours, so gaps between days are never bridged.
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
HOUR = np.timedelta64(1, "h")
CACHE_VERSION = 1


def to_day(value):
    """'2018-11-13', '20181113', a date or a datetime64 -> numpy datetime64[D]."""
    text = str(value)
    if re.fullmatch(r"\d{8}", text):
        text = f"{text[:4]}-{text[4:6]}-{text[6:]}"
    return np.datetime64(text, "D")


def _stamp(day):
    return str(to_day(day)).replace("-", "")


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


def source_variables(data_cfg):
    """Variables read from each source: the inputs plus the target (and O3 for the O3-input flag)."""
    inputs = data_cfg["inputs"]
    variables = {
        "meteo": list(inputs["meteo"]),
        "conc": list(dict.fromkeys(list(inputs["state"]) + list(data_cfg["target"]) + ["O3"])),
        "emis": list(inputs["emis"]),
    }
    seen = {}
    for source, names in variables.items():
        for name in names:
            if not isinstance(name, str):
                raise TypeError(f"Variable name {name!r} in data.inputs/{source} is not a string; quote it in the YAML (NO parses as false)")
            if seen.setdefault(name, source) != source:
                raise ValueError(f"Variable {name} is configured for both {seen[name]} and {source}")
    return variables


def input_channels(data_cfg, features_cfg):
    """Channel names in model order, and the indices of the forcing (meteorology + emission) channels."""
    inputs = data_cfg["inputs"]
    meteo, state, emis = list(inputs["meteo"]), list(inputs["state"]), list(inputs["emis"])
    if features_cfg.get("include_o3_input") and "O3" not in state:
        state.append("O3")
    channels = meteo + state + emis
    forcing = list(range(len(meteo))) + list(range(len(meteo) + len(state), len(channels)))
    return channels, forcing


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


def _read_source(path, variables, offset_hours):
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
            fields[name] = np.asarray(da.values, dtype=np.float32)
            units[name] = str(da.attrs.get("units", "")).strip()
    return times, fields, units


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


def load_day(data_cfg, day):
    """One day's variables on the hours common to all three files: {"times", "fields", "units"}.

    Results are cached as .npz under data.cache_dir, so each NetCDF file is read once.
    """
    files = day_files(data_cfg, day)
    variables = source_variables(data_cfg)
    offsets = {s: int((data_cfg.get("time_offset_hours") or {}).get(s, 0)) for s in SOURCES}
    cache = _cache_path(data_cfg, day, files, variables, offsets)
    if cache is not None and cache.exists():
        with np.load(cache) as z:
            return {
                "times": z["times"].astype("datetime64[h]"),
                "fields": {k[2:]: z[k] for k in z.files if k.startswith("f_")},
                "units": json.loads(str(z["units"])),
            }

    read = {s: _read_source(files[s], variables[s], offsets[s]) for s in SOURCES}
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
    times: np.ndarray       # (T,) datetime64[h] UTC after time offsets, strictly increasing
    days: np.ndarray        # (T,) datetime64[D] date of the file each hour came from
    x: np.ndarray           # (T, C, H, W) float32 input channels, native units
    y: np.ndarray           # (T, H, W) float32 target (Ox = sum of data.target), native units
    channels: list
    forcing_channels: list  # indices of meteorology + emission channels
    target_unit: str


def load_hourly(data_cfg, features_cfg, days):
    """Concatenate the given days into one hourly series of inputs and target."""
    days = sorted(to_day(d) for d in days)
    channels, forcing = input_channels(data_cfg, features_cfg)
    parts = [load_day(data_cfg, day) for day in days]

    times = np.concatenate([p["times"] for p in parts])
    if np.any(np.diff(times) <= np.timedelta64(0, "h")):
        raise ValueError("Hours overlap between days; check data.time_offset_hours")
    day_of_hour = np.concatenate([np.full(len(p["times"]), day) for p, day in zip(parts, days)])

    x = np.stack([np.concatenate([p["fields"][c] for p in parts]) for c in channels], axis=1)
    target = list(data_cfg["target"])
    y = sum(np.concatenate([p["fields"][v] for p in parts]) for v in target).astype(np.float32)
    target_units = {parts[0]["units"][v] for v in target}
    if len(target_units) != 1:
        raise ValueError(f"Target variables {target} have different units: {target_units}")
    return HourlyData(times, day_of_hour, x, y, channels, forcing, target_units.pop())


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
    """Samples (x, y) or, with forcing_channels, (x, y, future_x) from (normalized) hourly arrays.

    future_x[k] is the rollout frame for hour t+k (k < pred_len - 1): forcing channels
    from that hour, all other channels held at the last input hour t-1.
    """

    def __init__(self, x, y, starts, seq_len, pred_len, forcing_channels=None):
        self.x, self.y = x, y
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len, self.pred_len = seq_len, pred_len
        self.forcing_channels = forcing_channels

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        t = int(self.starts[i])
        x = torch.from_numpy(np.ascontiguousarray(self.x[t - self.seq_len:t]))
        y = torch.from_numpy(np.ascontiguousarray(self.y[t:t + self.pred_len]))
        if self.forcing_channels is None:
            return x, y
        future = np.repeat(self.x[t - 1:t], self.pred_len - 1, axis=0)
        future[:, self.forcing_channels] = self.x[t:t + self.pred_len - 1][:, self.forcing_channels]
        return x, y, torch.from_numpy(future)
