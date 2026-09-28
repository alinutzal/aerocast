"""Normalization statistics fitted on training windows only.

Per-channel mean/std for state and forcing channels, and a scalar mean/std per target
(NO2, O3, Ox), computed as the legacy script did over its stack of windows (hours weighted
by how many training windows contain them, unbiased std, +1e-6) without materializing
the stack. Exceptions: the wind components share one scale and keep a zero mean, so wind
direction survives normalization; time and grid channels are already O(1) and are left
as they are. The stats are saved as JSON with every run and applied unchanged to val/test.
"""
import json
from dataclasses import asdict, dataclass

import numpy as np

MODES = ("none", "predictors", "full")
STD_EPS = 1e-6
VECTOR_PAIRS = (("meteo:U10", "meteo:V10"),)  # one shared scale, no mean shift
UNSCALED_SOURCES = ("time", "grid")


def window_hour_counts(n_hours, starts, lo, hi):
    """Number of windows covering each hour, where the window at t covers hours t+lo .. t+hi-1."""
    counts = np.zeros(n_hours, dtype=np.int64)
    for t in starts:
        counts[t + lo:t + hi] += 1
    return counts


def _moments(values, counts):
    """Per-channel mean, unbiased std and mean square of values (T, C, ...), hour t repeated counts[t] times."""
    hours = np.nonzero(counts)[0]
    if len(hours) == 0:
        raise ValueError("No training windows to fit normalization stats on")
    flat = values.reshape(values.shape[0], values.shape[1], -1)
    n = counts[hours].sum() * flat.shape[2]
    mean = sum(counts[h] * flat[h].sum(axis=1, dtype=np.float64) for h in hours) / n
    sq = sum(counts[h] * ((flat[h] - mean[:, None]) ** 2).sum(axis=1) for h in hours)
    return mean, np.sqrt(sq / (n - 1)), sq / n + mean ** 2


def _channel_stats(values, counts, names):
    mean, std, mean_square = _moments(values, counts)
    std = std + STD_EPS
    for i, name in enumerate(names):
        if name.split(":")[0] in UNSCALED_SOURCES:
            mean[i], std[i] = 0.0, 1.0
    for a, b in VECTOR_PAIRS:
        if a in names and b in names:
            i, j = names.index(a), names.index(b)
            mean[i] = mean[j] = 0.0
            std[i] = std[j] = np.sqrt((mean_square[i] + mean_square[j]) / 2) + STD_EPS
    return mean.tolist(), std.tolist()


@dataclass
class NormStats:
    mode: str
    channels: dict            # {"state": [...], "forcing": [...], "static": [...]}
    state_mean: list = None
    state_std: list = None
    forcing_mean: list = None
    forcing_std: list = None
    target_mean: dict = None  # target name -> scalar
    target_std: dict = None

    def normalize(self, group, values):
        """Normalize (T, C, H, W) state or forcing values; static (grid) channels pass through."""
        mean, std = getattr(self, f"{group}_mean", None), getattr(self, f"{group}_std", None)
        if mean is None:
            return values.astype(np.float32)
        mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
        return ((values - mean) / std).astype(np.float32)

    def normalize_target(self, name, values):
        if self.target_mean is None:
            return values.astype(np.float32)
        return ((values - np.float32(self.target_mean[name])) / np.float32(self.target_std[name])).astype(np.float32)

    def denormalize_target(self, name, values):
        """Normalized target -> native units (numpy or torch)."""
        if self.target_mean is None:
            return values
        return values * float(self.target_std[name]) + float(self.target_mean[name])

    def save(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(**json.load(f))


def fit_stats(hourly, train_starts, t_in, t_out, mode, future_forcings=True):
    """Fit stats on the training windows: state on their input hours, forcing on the hours it is
    fed for (input + forecast hours with future_forcings), targets on their target hours."""
    if mode not in MODES:
        raise ValueError(f"Unknown normalization {mode!r}; expected one of {MODES}")
    stats = NormStats(mode=mode, channels={k: list(v) for k, v in hourly.channels.items()})
    n_hours = len(hourly.times)
    if mode in ("predictors", "full"):
        counts = window_hour_counts(n_hours, train_starts, -t_in, 0)
        stats.state_mean, stats.state_std = _channel_stats(hourly.state, counts, hourly.channels["state"])
        counts = window_hour_counts(n_hours, train_starts, -t_in, t_out if future_forcings else 0)
        stats.forcing_mean, stats.forcing_std = _channel_stats(hourly.forcing, counts, hourly.channels["forcing"])
    if mode == "full":
        counts = window_hour_counts(n_hours, train_starts, 0, t_out)
        stats.target_mean, stats.target_std = {}, {}
        for name, values in hourly.targets.items():
            mean, std, _ = _moments(values[:, None], counts)
            stats.target_mean[name], stats.target_std[name] = float(mean[0]), float(std[0] + STD_EPS)
    return stats


def normalized_arrays(hourly, stats, target_names):
    """Normalized model inputs and the stacked targets (T, K, H, W) for WindowDataset."""
    return {
        "state": stats.normalize("state", hourly.state),
        "forcing": stats.normalize("forcing", hourly.forcing),
        "static": hourly.static,
        "targets": np.stack([stats.normalize_target(n, hourly.targets[n]) for n in target_names], axis=1),
    }
