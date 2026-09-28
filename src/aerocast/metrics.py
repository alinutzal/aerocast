"""Streaming forecast metrics per lead hour and overall."""
import numpy as np

METRICS = ("rmse", "mae", "bias", "pearson_r")
_SUMS = ("e", "ae", "ee", "p", "o", "pp", "oo", "po")


class LeadMetrics:
    """Accumulates RMSE, MAE, mean bias (pred - obs) and Pearson r per lead hour."""

    def __init__(self, pred_len):
        self.pred_len = pred_len
        self.n = 0
        self.sums = {k: np.zeros(pred_len) for k in _SUMS}
        self.shift = None  # subtracted before the r sums to limit cancellation

    def update(self, pred, obs):
        """pred, obs: (B, pred_len, H, W) in native units."""
        p = np.asarray(pred, dtype=np.float64)
        o = np.asarray(obs, dtype=np.float64)
        if p.shape != o.shape or p.shape[1] != self.pred_len:
            raise ValueError(f"Shape mismatch: pred {p.shape}, obs {o.shape}")
        if self.shift is None:
            self.shift = float(o.mean())
        axes = (0, 2, 3)
        e = p - o
        p, o = p - self.shift, o - self.shift
        self.n += p.size // self.pred_len
        for key, values in (("e", e), ("ae", np.abs(e)), ("ee", e * e), ("p", p), ("o", o),
                            ("pp", p * p), ("oo", o * o), ("po", p * o)):
            self.sums[key] += values.sum(axis=axes)

    @staticmethod
    def _metrics(n, s):
        mean_p, mean_o = s["p"] / n, s["o"] / n
        var_p, var_o = s["pp"] / n - mean_p ** 2, s["oo"] / n - mean_o ** 2
        cov = s["po"] / n - mean_p * mean_o
        r = cov / np.sqrt(var_p * var_o) if var_p > 0 and var_o > 0 else float("nan")
        return {"rmse": float(np.sqrt(s["ee"] / n)), "mae": float(s["ae"] / n),
                "bias": float(s["e"] / n), "pearson_r": float(r)}

    def results(self):
        """[(lead_hour, metric, value)] for lead hours 1..pred_len, then 'all'."""
        rows = []
        for k in range(self.pred_len):
            lead = self._metrics(self.n, {key: v[k] for key, v in self.sums.items()})
            rows += [(k + 1, name, lead[name]) for name in METRICS]
        overall = self._metrics(self.n * self.pred_len, {key: v.sum() for key, v in self.sums.items()})
        rows += [("all", name, overall[name]) for name in METRICS]
        return rows
