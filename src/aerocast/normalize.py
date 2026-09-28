"""Normalization statistics fitted on training windows only.

Per-channel input mean/std and a scalar target mean/std, computed exactly as the
legacy script did over its stack of windows (hours weighted by how many training
windows contain them, unbiased std, +1e-6), but without materializing the stack.
The stats are saved as JSON with every run and applied unchanged to val/test.
"""
import json
from dataclasses import asdict, dataclass

import numpy as np

MODES = ("none", "predictors", "full")
STD_EPS = 1e-6


def window_hour_counts(n_hours, starts, lo, hi):
    """Number of windows covering each hour, where the window at t covers hours t+lo .. t+hi-1."""
    counts = np.zeros(n_hours, dtype=np.int64)
    for t in starts:
        counts[t + lo:t + hi] += 1
    return counts


def _mean_std(values, counts):
    """Per-channel mean and unbiased std of values (T, C, ...), each hour repeated counts[t] times."""
    hours = np.nonzero(counts)[0]
    if len(hours) == 0:
        raise ValueError("No training windows to fit normalization stats on")
    flat = values.reshape(values.shape[0], values.shape[1], -1)
    n = counts[hours].sum() * flat.shape[2]
    total = sum(counts[h] * flat[h].sum(axis=1, dtype=np.float64) for h in hours)
    mean = total / n
    sq = sum(counts[h] * ((flat[h] - mean[:, None]) ** 2).sum(axis=1) for h in hours)
    return mean, np.sqrt(sq / (n - 1))


@dataclass
class NormStats:
    mode: str
    channels: list
    x_mean: list = None
    x_std: list = None
    y_mean: float = None
    y_std: float = None

    def normalize_x(self, x):
        if self.x_mean is None:
            return x
        mean = np.asarray(self.x_mean, dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.asarray(self.x_std, dtype=np.float32).reshape(1, -1, 1, 1)
        return ((x - mean) / std).astype(np.float32)

    def normalize_y(self, y):
        if self.y_mean is None:
            return y
        return ((y - np.float32(self.y_mean)) / np.float32(self.y_std)).astype(np.float32)

    def denormalize_y(self, y):
        if self.y_mean is None:
            return y
        return y * np.float32(self.y_std) + np.float32(self.y_mean)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def fit_stats(hourly, train_starts, seq_len, pred_len, mode):
    """Fit stats on the training windows: inputs from their input hours, target from their target hours."""
    if mode not in MODES:
        raise ValueError(f"Unknown normalization {mode!r}; expected one of {MODES}")
    stats = NormStats(mode=mode, channels=list(hourly.channels))
    n_hours = len(hourly.times)
    if mode in ("predictors", "full"):
        mean, std = _mean_std(hourly.x, window_hour_counts(n_hours, train_starts, -seq_len, 0))
        stats.x_mean, stats.x_std = mean.tolist(), (std + STD_EPS).tolist()
    if mode == "full":
        mean, std = _mean_std(hourly.y[:, None], window_hour_counts(n_hours, train_starts, 0, pred_len))
        stats.y_mean, stats.y_std = float(mean[0]), float(std[0] + STD_EPS)
    return stats
