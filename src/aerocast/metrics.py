"""Forecast scores in native units, accumulated per day and lead hour.

Per-day sums make every metric recomputable for any subset or resample of days, which is
what the block bootstrap (CIs over days) and paired differences between runs need.
"""
import numpy as np

METRICS = ("rmse", "mae", "bias", "pearson_r")
EXCEEDANCE = ("csi", "f1", "rmse_above")
# Per-day, per-lead sums: count, error, |error|, error^2, shifted pred/obs moments for r,
# and exceedance counts (hits, misses, false alarms) with the squared error above threshold.
FEATURES = ("n", "e", "ae", "ee", "p", "o", "pp", "oo", "po", "hits", "misses", "fa", "ee_above", "n_above")
F = {name: i for i, name in enumerate(FEATURES)}


def metrics_from_sums(s):
    """Metrics from summed features s (..., len(FEATURES)); works on stacked resamples."""
    with np.errstate(invalid="ignore", divide="ignore"):
        n = s[..., F["n"]]
        mean_p, mean_o = s[..., F["p"]] / n, s[..., F["o"]] / n
        var_p = s[..., F["pp"]] / n - mean_p ** 2
        var_o = s[..., F["oo"]] / n - mean_o ** 2
        cov = s[..., F["po"]] / n - mean_p * mean_o
        r = np.where((var_p > 0) & (var_o > 0), cov / np.sqrt(np.abs(var_p * var_o)), np.nan)
        hits, misses, fa = s[..., F["hits"]], s[..., F["misses"]], s[..., F["fa"]]
        return {
            "rmse": np.sqrt(s[..., F["ee"]] / n),
            "mae": s[..., F["ae"]] / n,
            "bias": s[..., F["e"]] / n,
            "pearson_r": r,
            "csi": hits / (hits + misses + fa),
            "f1": 2 * hits / (2 * hits + misses + fa),
            "rmse_above": np.sqrt(s[..., F["ee_above"]] / s[..., F["n_above"]]),
        }


def radial_bins(height, width, cell_km=1.0):
    """Radial wavenumber bin of every 2D FFT coefficient (-1 for the mean), and bin centres in cycles/km."""
    n_bins = min(height, width) // 2
    ky = np.fft.fftfreq(height)[:, None]
    kx = np.fft.fftfreq(width)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)  # cycles per cell
    edges = np.linspace(0, 0.5, n_bins + 1)
    index = np.digitize(k, edges[1:-1], right=True)
    index[(k == 0) | (k > 0.5)] = -1
    return index, (edges[:-1] + edges[1:]) / 2 / cell_km


def radial_power(fields, bins):
    """Radially averaged power spectrum of (N, H, W) fields: mean removed and a 2D Hann taper
    applied, so the non-periodic edges do not leak power. Returns (N, n_bins)."""
    index, centres = bins
    height, width = fields.shape[-2:]
    taper = np.outer(np.hanning(height), np.hanning(width))
    anomalies = (fields - fields.mean(axis=(-2, -1), keepdims=True)) * taper
    power = np.abs(np.fft.fft2(anomalies)) ** 2 / (height * width)
    keep = index >= 0
    counts = np.bincount(index[keep], minlength=len(centres))
    return np.stack([np.bincount(index[keep], weights=p[keep], minlength=len(centres)) / counts for p in power])


class ForecastScores:
    """Scores of one forecast source against one variable over many windows."""

    def __init__(self, pred_len, shift, threshold=None, spectrum_leads=(), bins=None):
        self.pred_len, self.shift, self.threshold = pred_len, float(shift), threshold
        self.days = {}  # day -> (pred_len, len(FEATURES)) sums
        self.spectrum_leads, self.bins = tuple(spectrum_leads), bins
        self.power = {lead: [0.0, 0.0] for lead in self.spectrum_leads}  # summed pred / true spectra
        self.windows = 0

    def update(self, pred, obs, day):
        """pred, obs: (pred_len, H, W) in native units for one window whose first target hour is on `day`."""
        p = np.asarray(pred, dtype=np.float64)
        o = np.asarray(obs, dtype=np.float64)
        e = p - o
        ps, os_ = p - self.shift, o - self.shift
        sums = np.zeros((self.pred_len, len(FEATURES)))
        sums[:, F["n"]] = e[0].size
        for name, values in (("e", e), ("ae", np.abs(e)), ("ee", e * e), ("p", ps), ("o", os_),
                             ("pp", ps * ps), ("oo", os_ * os_), ("po", ps * os_)):
            sums[:, F[name]] = values.sum(axis=(1, 2))
        if self.threshold is not None:
            p_hi, o_hi = p > self.threshold, o > self.threshold
            sums[:, F["hits"]] = (p_hi & o_hi).sum(axis=(1, 2))
            sums[:, F["misses"]] = (~p_hi & o_hi).sum(axis=(1, 2))
            sums[:, F["fa"]] = (p_hi & ~o_hi).sum(axis=(1, 2))
            sums[:, F["ee_above"]] = np.where(o_hi, e * e, 0).sum(axis=(1, 2))
            sums[:, F["n_above"]] = o_hi.sum(axis=(1, 2))
        day = str(day)
        self.days[day] = self.days.get(day, 0) + sums
        for lead in self.spectrum_leads:
            spectra = radial_power(np.stack([p[lead - 1], o[lead - 1]]), self.bins)
            self.power[lead][0] = self.power[lead][0] + spectra[0]
            self.power[lead][1] = self.power[lead][1] + spectra[1]
        self.windows += 1

    def daily(self):
        """(days, (D, pred_len, len(FEATURES))) with days sorted."""
        days = sorted(self.days)
        return days, np.stack([self.days[d] for d in days])

    def rows(self, prefix=""):
        """[(lead_hour, metric, value)] per lead and overall; exceedance metrics when a threshold is set."""
        _, daily = self.daily()
        total = daily.sum(axis=0)  # (pred_len, F)
        names = list(METRICS) + (list(EXCEEDANCE) if self.threshold is not None else [])
        suffix = {"csi": "csi_p95", "f1": "f1_p95", "rmse_above": "rmse_above_p95"}
        rows = []
        for lead, sums in [*((k + 1, total[k]) for k in range(self.pred_len)), ("all", total.sum(axis=0))]:
            values = metrics_from_sums(sums)
            rows += [(lead, prefix + suffix.get(m, m), float(values[m])) for m in names]
        for lead in self.spectrum_leads:
            rows.append((lead, prefix + "spectral_ratio", self.spectral_ratio(lead)))
        return rows

    def spectra(self, lead):
        """Mean radially averaged power (pred, true) at a lead hour."""
        pred, true = self.power[lead]
        return pred / self.windows, true / self.windows

    def spectral_ratio(self, lead):
        """Predicted / true power summed over the top third of wavenumbers (1 = no smoothing)."""
        pred, true = self.spectra(lead)
        top = slice(2 * len(true) // 3, None)
        return float(pred[top].sum() / true[top].sum())

    def bootstrap_rows(self, n_samples, seed, prefix=""):
        """95% CIs by block bootstrap over days (days resampled with replacement): overall
        rmse/mae/bias/r (and csi with a threshold), plus rmse per lead. Empty with < 2 days."""
        days, daily = self.daily()
        if len(days) < 2:
            return []
        counts = bootstrap_counts(len(days), n_samples, seed)          # (B, D)
        resampled = np.einsum("bd,dlf->blf", counts, daily)             # (B, pred_len, F)
        rows = []
        overall = metrics_from_sums(resampled.sum(axis=1))
        per_lead = metrics_from_sums(resampled)
        names = ["rmse", "mae", "bias", "pearson_r"] + (["csi"] if self.threshold is not None else [])
        for m in names:
            label = "csi_p95" if m == "csi" else m
            lo, hi = np.nanpercentile(overall[m], [2.5, 97.5])
            rows += [("all", f"{prefix}{label}_ci95_lo", float(lo)), ("all", f"{prefix}{label}_ci95_hi", float(hi))]
        lo, hi = np.nanpercentile(per_lead["rmse"], [2.5, 97.5], axis=0)
        for k in range(self.pred_len):
            rows += [(k + 1, f"{prefix}rmse_ci95_lo", float(lo[k])), (k + 1, f"{prefix}rmse_ci95_hi", float(hi[k]))]
        return rows


def bootstrap_counts(n_days, n_samples, seed):
    """How often each day is drawn in each bootstrap resample: (n_samples, n_days)."""
    draws = np.random.default_rng(seed).integers(0, n_days, size=(n_samples, n_days))
    return np.stack([np.bincount(row, minlength=n_days) for row in draws]).astype(np.float64)

