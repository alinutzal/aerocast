import numpy as np
import pytest

from aerocast.normalize import NormStats, fit_stats
from aerocast.splits import load_and_split

SEQ, PRED = 6, 10


def test_stats_ignore_val_and_test_values(make_cfg):
    hourly, splits = load_and_split(make_cfg())
    before = fit_stats(hourly, splits.train, SEQ, PRED, "full")
    held_out = ~splits.train_hours
    assert held_out.sum() == 48  # the val and test days
    hourly.x[held_out] = hourly.x[held_out] * 100 + 1e3
    hourly.y[held_out] = -50 * hourly.y[held_out]
    assert fit_stats(hourly, splits.train, SEQ, PRED, "full") == before


def test_stats_equal_the_legacy_stacked_window_stats(make_cfg):
    hourly, splits = load_and_split(make_cfg())
    stats = fit_stats(hourly, splits.train, SEQ, PRED, "full")
    X = np.stack([hourly.x[t - SEQ:t] for t in splits.train]).astype(np.float64)  # (N, T, C, H, W)
    Y = np.stack([hourly.y[t:t + PRED] for t in splits.train]).astype(np.float64)
    np.testing.assert_allclose(stats.x_mean, X.mean(axis=(0, 1, 3, 4)), rtol=1e-9)
    np.testing.assert_allclose(np.array(stats.x_std) - 1e-6, X.std(axis=(0, 1, 3, 4), ddof=1), rtol=1e-9)
    assert stats.y_mean == pytest.approx(Y.mean(), rel=1e-9)
    assert stats.y_std - 1e-6 == pytest.approx(Y.std(ddof=1), rel=1e-9)


def test_normalization_modes(make_cfg):
    hourly, splits = load_and_split(make_cfg())
    none = fit_stats(hourly, splits.train, SEQ, PRED, "none")
    predictors = fit_stats(hourly, splits.train, SEQ, PRED, "predictors")
    assert none.x_mean is None and none.y_mean is None
    assert predictors.x_mean is not None and predictors.y_mean is None
    np.testing.assert_array_equal(none.normalize_x(hourly.x), hourly.x)


def test_stats_round_trip_through_json(make_cfg, tmp_path):
    hourly, splits = load_and_split(make_cfg())
    stats = fit_stats(hourly, splits.train, SEQ, PRED, "full")
    stats.save(tmp_path / "norm_stats.json")
    assert NormStats.load(tmp_path / "norm_stats.json") == stats
