"""Assign windows to train/val/test by contiguous date blocks.

Modes (split.mode):
  dates       honest split: every hour takes the split of its file date, and a window is
              kept only if all its input and target hours share one split, so no window
              crosses a split boundary.
  smoke_test  single-day pipeline check: train, early stopping and evaluation all use
              every window. Rows are labelled smoke_test, never test.
  legacy      refactor check against legacy/grid_forcast.py: same windows as smoke_test,
              rows labelled legacy.
"""
import warnings
from dataclasses import dataclass

import numpy as np

from aerocast.data import available_days, load_hourly, to_day, window_starts

SPLITS = ("train", "val", "test")
MODES = ("dates", "smoke_test", "legacy")


class SmokeTestWarning(UserWarning):
    """Evaluation windows were also used for training: the numbers are in-sample."""


@dataclass
class Splits:
    mode: str
    train: np.ndarray        # window starts used to fit the model and the normalization stats
    val: np.ndarray          # window starts for early stopping and checkpoint selection
    evaluate: dict           # results label -> window starts to score
    train_hours: np.ndarray  # (T,) bool, hours of training days (used by climatology)


def date_blocks(spec):
    """[start, end] or [[start, end], ...] (inclusive) -> [(datetime64[D], datetime64[D])]."""
    if not spec:
        return []
    pairs = spec if isinstance(spec[0], (list, tuple)) else [spec]
    blocks = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Date block {pair!r} must be [start, end]")
        start, end = to_day(pair[0]), to_day(pair[1])
        if end < start:
            raise ValueError(f"Date block {pair!r} ends before it starts")
        blocks.append((start, end))
    return blocks


def check_blocks(split_cfg):
    blocks = [(name, s, e) for name in SPLITS for s, e in date_blocks(split_cfg.get(name))]
    for i, (name_a, start_a, end_a) in enumerate(blocks):
        for name_b, start_b, end_b in blocks[i + 1:]:
            if name_a != name_b and start_a <= end_b and start_b <= end_a:
                raise ValueError(f"Split date blocks overlap: {name_a} {start_a}..{end_a} and {name_b} {start_b}..{end_b}")


def hour_labels(days, split_cfg):
    """Split name of each hour, from its file date ('' if the date is in no block)."""
    labels = np.full(len(days), "", dtype=object)
    for name in SPLITS:
        for start, end in date_blocks(split_cfg.get(name)):
            labels[(days >= start) & (days <= end)] = name
    return labels


def days_to_load(data_cfg, split_cfg):
    """data.dates if given, else every available day (inside the split blocks in dates mode)."""
    if data_cfg.get("dates"):
        days = sorted({to_day(d) for d in data_cfg["dates"]})
    else:
        days = available_days(data_cfg)
        if split_cfg["mode"] == "dates":
            blocks = [b for name in SPLITS for b in date_blocks(split_cfg.get(name))]
            days = [d for d in days if any(start <= d <= end for start, end in blocks)]
    if not days:
        raise FileNotFoundError(f"No days to load from {data_cfg['data_dir']}")
    return days


def make_splits(hourly, split_cfg, seq_len, pred_len):
    mode = split_cfg["mode"]
    if mode == "dates":
        check_blocks(split_cfg)
        labels = hour_labels(hourly.days, split_cfg)
        windows = {name: window_starts(hourly.times, seq_len, pred_len, labels, name) for name in SPLITS}
        empty = [name for name in SPLITS if len(windows[name]) == 0]
        if empty:
            loaded = sorted({str(d) for d in hourly.days})
            hint = " With a single day there is no clean held-out split: use split.mode: smoke_test." if len(loaded) == 1 else ""
            raise ValueError(f"No windows for split(s) {empty}; loaded days: {loaded}.{hint}")
        return Splits(mode, windows["train"], windows["val"],
                      {"val": windows["val"], "test": windows["test"]}, labels == "train")

    if mode in ("smoke_test", "legacy"):
        windows = window_starts(hourly.times, seq_len, pred_len)
        if len(windows) == 0:
            raise ValueError(f"No complete {seq_len}+{pred_len} h window in the loaded hours")
        if mode == "smoke_test":
            warnings.warn(
                "SMOKE TEST: training, early stopping and evaluation use the same windows "
                f"({len(windows)} from {len(np.unique(hourly.days))} day(s)). Results are in-sample, "
                "labelled smoke_test, and are not a held-out test.", SmokeTestWarning, stacklevel=2)
        else:
            warnings.warn("LEGACY mode: training and evaluation use the same windows (refactor check only).",
                          SmokeTestWarning, stacklevel=2)
        return Splits(mode, windows, windows, {mode: windows}, np.ones(len(hourly.times), dtype=bool))

    raise ValueError(f"Unknown split.mode {mode!r}; expected one of {MODES}")


def load_and_split(cfg):
    """Load the configured days and split their windows: returns (HourlyData, Splits)."""
    data_cfg = cfg["data"]
    days = days_to_load(data_cfg, cfg["split"])
    hourly = load_hourly(data_cfg, cfg["features"], days)
    return hourly, make_splits(hourly, cfg["split"], data_cfg["seq_len"], data_cfg["pred_len"])
