"""Reference forecasts, scored on exactly the same windows as the model (native units)."""
import warnings

import numpy as np


def hour_of_day(times):
    """UTC hour of day of datetime64[h] timestamps. Local standard time is a fixed offset,
    so grouping by UTC hour is the same as grouping by local hour."""
    return ((times - times.astype("datetime64[D]")) // np.timedelta64(1, "h")).astype(np.int64)


def persistence(y, t, pred_len):
    """Ox at the last input hour (t-1), held for every lead hour: (pred_len, H, W)."""
    return np.repeat(y[t - 1:t], pred_len, axis=0)


class DiurnalClimatology:
    """Per-cell mean Ox for each hour of day, fitted on training hours only."""

    def __init__(self, y, times, train_hours):
        train_hours = np.asarray(train_hours, dtype=bool)
        if not train_hours.any():
            raise ValueError("Climatology needs at least one training hour")
        hod = hour_of_day(times)
        self.table = np.empty((24,) + y.shape[1:], dtype=np.float32)
        self.missing_hours = []
        for h in range(24):
            sel = train_hours & (hod == h)
            if sel.any():
                self.table[h] = y[sel].mean(axis=0, dtype=np.float64)
            else:
                self.missing_hours.append(h)
        if self.missing_hours:
            # Only happens when training days are incomplete; fall back to the all-hours mean.
            self.table[self.missing_hours] = y[train_hours].mean(axis=0, dtype=np.float64)
            warnings.warn(f"Climatology: no training data for UTC hours {self.missing_hours}; using the all-hours mean")

    def predict(self, times):
        """Climatology for the given target hours: (len(times), H, W)."""
        return self.table[hour_of_day(times)]
