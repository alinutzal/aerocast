import numpy as np
import pytest

from aerocast.normalize import NormStats, fit_stats
from aerocast.splits import load_and_split

SEQ, PRED = 6, 10


def stack_windows(values, starts, lo, hi):
    return np.stack([values[t + lo:t + hi] for t in starts]).astype(np.float64)


def test_stats_ignore_val_and_test_values(make_cfg):
    hourly, splits = load_and_split(make_cfg())
    before = fit_stats(hourly, splits.train, SEQ, PRED, "full")
    held_out = ~splits.train_hours
    assert held_out.sum() == 48  # the val and test days
    hourly.state[held_out] = hourly.state[held_out] * 100 + 1e3
    hourly.forcing[held_out] = -hourly.forcing[held_out] * 7
    for values in hourly.targets.values():
        values[held_out] = -50 * values[held_out]
    assert fit_stats(hourly, splits.train, SEQ, PRED, "full") == before


def test_stats_equal_the_legacy_stacked_window_stats(make_cfg):
    hourly, splits = load_and_split(make_cfg())
    stats = fit_stats(hourly, splits.train, SEQ, PRED, "full")
    state = stack_windows(hourly.state, splits.train, -SEQ, 0)       # (N, T_in, C, H, W)
    forcing = stack_windows(hourly.forcing, splits.train, -SEQ, PRED)  # future forcings: 16 hours
    np.testing.assert_allclose(stats.state_mean, state.mean(axis=(0, 1, 3, 4)), rtol=1e-9)
    np.testing.assert_allclose(np.array(stats.state_std) - 1e-6, state.std(axis=(0, 1, 3, 4), ddof=1), rtol=1e-9)
    names = hourly.channels["forcing"]
    plain = [i for i, n in enumerate(names) if n.split(":")[0] in ("meteo", "emis") and n not in ("meteo:U10", "meteo:V10")]
    np.testing.assert_allclose(np.array(stats.forcing_mean)[plain], forcing.mean(axis=(0, 1, 3, 4))[plain], rtol=1e-9)
    for name in ("NO2", "O3", "Ox"):
        target = stack_windows(hourly.targets[name], splits.train, 0, PRED)
        assert stats.target_mean[name] == pytest.approx(target.mean(), rel=1e-9)
        assert stats.target_std[name] - 1e-6 == pytest.approx(target.std(ddof=1), rel=1e-9)


def test_wind_shares_one_scale_and_time_grid_pass_through(make_cfg):
    hourly, splits = load_and_split(make_cfg())
    stats = fit_stats(hourly, splits.train, SEQ, PRED, "full")
    names = hourly.channels["forcing"]
    u, v = names.index("meteo:U10"), names.index("meteo:V10")
    forcing = stack_windows(hourly.forcing, splits.train, -SEQ, PRED)
    rms = np.sqrt((np.mean(forcing[:, :, u] ** 2) + np.mean(forcing[:, :, v] ** 2)) / 2)
    assert stats.forcing_mean[u] == stats.forcing_mean[v] == 0.0
    assert stats.forcing_std[u] == stats.forcing_std[v] == pytest.approx(rms + 1e-6, rel=1e-9)
    for i, name in enumerate(names):
        if name.startswith("time:"):
            assert (stats.forcing_mean[i], stats.forcing_std[i]) == (0.0, 1.0)
    np.testing.assert_array_equal(stats.normalize("static", hourly.static), hourly.static)


def test_normalization_modes(make_cfg):
    hourly, splits = load_and_split(make_cfg())
    none = fit_stats(hourly, splits.train, SEQ, PRED, "none")
    predictors = fit_stats(hourly, splits.train, SEQ, PRED, "predictors")
    assert none.state_mean is None and none.target_mean is None
    assert predictors.forcing_mean is not None and predictors.target_mean is None
    np.testing.assert_array_equal(none.normalize("state", hourly.state), hourly.state)


def test_stats_round_trip_through_json(make_cfg, tmp_path):
    hourly, splits = load_and_split(make_cfg())
    stats = fit_stats(hourly, splits.train, SEQ, PRED, "full")
    stats.save(tmp_path / "norm_stats.json")
    assert NormStats.load(tmp_path / "norm_stats.json") == stats
