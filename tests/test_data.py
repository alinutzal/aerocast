import numpy as np
import xarray as xr

from aerocast.data import WindowDataset, hour_of_day
from aerocast.splits import load_and_split

SEQ, PRED = 6, 10


def test_channel_layout(make_cfg):
    hourly, _ = load_and_split(make_cfg())
    assert hourly.channels["state"] == ["conc:NO", "conc:NO2", "conc:PM25_CL", "conc:O3"]
    assert len(hourly.channels["forcing"]) == 15 and hourly.channels["static"] == ["grid:X", "grid:Y"]
    assert hourly.state.shape[1:2] == (4,) and hourly.forcing.shape[1:2] == (15,) and hourly.static.shape[0] == 2
    np.testing.assert_allclose(hourly.targets["Ox"], hourly.targets["NO2"] + hourly.targets["O3"], rtol=1e-6)


def test_emission_and_concentration_nox_stay_separate(make_cfg, synthetic_dir):
    hourly, _ = load_and_split(make_cfg())
    emis = xr.open_dataset(next(synthetic_dir.glob("egts_l.20181113.*.ncf")), decode_times=False)
    conc = xr.open_dataset(synthetic_dir / "out.combine_20181113.nc", decode_times=False)
    forcing, state = hourly.channels["forcing"], hourly.channels["state"]
    for name in ("NO", "NO2", "HONO"):
        np.testing.assert_array_equal(hourly.forcing[:24, forcing.index(f"emis:{name}")], emis[name].values[:24, 0])
    for name in ("NO", "NO2"):
        np.testing.assert_array_equal(hourly.state[:24, state.index(f"conc:{name}")], conc[name].values[:, 0])


def test_wind_components_follow_the_meteorological_convention(make_cfg, synthetic_dir):
    hourly, _ = load_and_split(make_cfg())
    met = xr.open_dataset(synthetic_dir / "METCRO2D_20181113.nc", decode_times=False)
    speed, direction = met["WSPD10"].values[:24, 0], np.deg2rad(met["WDIR10"].values[:24, 0])
    names = hourly.channels["forcing"]
    u, v = hourly.forcing[:24, names.index("meteo:U10")], hourly.forcing[:24, names.index("meteo:V10")]
    np.testing.assert_allclose(u, -speed * np.sin(direction), atol=1e-5)
    np.testing.assert_allclose(v, -speed * np.cos(direction), atol=1e-5)
    # A wind from the west (270 degrees) blows toward +x.
    np.testing.assert_allclose(-1.0 * np.sin(np.deg2rad(270.0)), 1.0)


def test_hour_channels_use_local_time_and_grid_is_unit_square(make_cfg):
    hourly, _ = load_and_split(make_cfg())
    names = hourly.channels["forcing"]
    local = (hour_of_day(hourly.times) - 8) % 24
    np.testing.assert_allclose(hourly.forcing[:, names.index("time:HOUR_SIN"), 0, 0], np.sin(2 * np.pi * local / 24), atol=1e-6)
    np.testing.assert_allclose(hourly.forcing[:, names.index("time:HOUR_COS"), 3, 5], np.cos(2 * np.pi * local / 24), atol=1e-6)
    assert local[0] == 0  # the first hour of each file day is local midnight
    x, y = hourly.static
    assert (x.min(), x.max(), y.min(), y.max()) == (0.0, 1.0, 0.0, 1.0)
    assert np.all(np.diff(x, axis=1) > 0) and np.all(np.diff(y, axis=0) > 0)


def test_forcing_covers_the_forecast_hours_only_with_future_forcings(make_cfg):
    hourly, splits = load_and_split(make_cfg())
    targets = hourly.targets["Ox"][:, None]
    t = int(splits.train[0])
    for future, expected in ((True, hourly.forcing[t - SEQ:t + PRED]),
                             (False, np.concatenate([hourly.forcing[t - SEQ:t], np.repeat(hourly.forcing[t - 1:t], PRED, 0)]))):
        sample = WindowDataset(hourly.state, hourly.forcing, hourly.static, targets, splits.train, SEQ, PRED, future)[0]
        np.testing.assert_array_equal(sample["forcing"], expected)
        np.testing.assert_array_equal(sample["state"], hourly.state[t - SEQ:t])
        assert sample["target"].shape == (PRED, 1) + hourly.targets["Ox"].shape[1:]
