import numpy as np

from aerocast.baselines import DiurnalClimatology, persistence
from aerocast.splits import load_and_split

PRED = 10


def test_persistence_holds_the_last_input_hour(make_cfg):
    hourly, splits = load_and_split(make_cfg())
    for t in splits.evaluate["test"]:
        forecast = persistence(hourly.targets["Ox"], t, PRED)
        assert forecast.shape == (PRED,) + hourly.targets["Ox"].shape[1:]
        for k in range(PRED):
            np.testing.assert_array_equal(forecast[k], hourly.targets["Ox"][t - 1])


def test_climatology_uses_training_days_only(make_cfg):
    hourly, splits = load_and_split(make_cfg())  # train = first day
    clim = DiurnalClimatology(hourly.targets["Ox"], hourly.times, splits.train_hours)
    altered = hourly.targets["Ox"].copy()
    altered[~splits.train_hours] += 1000.0
    np.testing.assert_array_equal(DiurnalClimatology(altered, hourly.times, splits.train_hours).table, clim.table)
    # One training day: each hour's climatology is that day's field at the same hour.
    train = np.nonzero(splits.train_hours)[0]
    np.testing.assert_allclose(clim.predict(hourly.times[train]), hourly.targets["Ox"][train], rtol=1e-6)


def test_climatology_averages_each_hour_over_training_days(make_cfg, days):
    hourly, _ = load_and_split(make_cfg())
    train_hours = hourly.days != np.datetime64(days[2])
    clim = DiurnalClimatology(hourly.targets["Ox"], hourly.times, train_hours)
    expected = (hourly.targets["Ox"][:24].astype(np.float64) + hourly.targets["Ox"][24:48]) / 2  # same hour, days 1 and 2
    np.testing.assert_allclose(clim.predict(hourly.times[48:]), expected, rtol=1e-5)
