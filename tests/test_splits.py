import numpy as np
import pytest

from aerocast.data import window_starts
from aerocast.splits import load_and_split

HOUR = np.timedelta64(1, "h")
SEQ, PRED = 6, 10


def hours(start, n):
    return np.datetime64(start, "h") + np.arange(n) * HOUR


def test_windows_only_span_consecutive_hours():
    times = np.concatenate([hours("2018-11-13T08", 24), hours("2018-11-15T08", 24)])  # a missing day
    starts = window_starts(times, SEQ, PRED)
    assert len(starts) == 2 * 9
    for t in starts:
        assert np.all(np.diff(times[t - SEQ:t + PRED]) == HOUR)


def test_no_window_crosses_a_split_boundary(make_cfg, days):
    hourly, splits = load_and_split(make_cfg())
    assert len(hourly.times) == 72 and np.all(np.diff(hourly.times) == HOUR)
    windows = {"train": splits.train, "val": splits.val, "test": splits.evaluate["test"]}
    for name, day in zip(("train", "val", "test"), days):
        assert len(windows[name]) == 9  # 24 - (6 + 10) + 1
        for t in windows[name]:
            assert np.all(hourly.days[t - SEQ:t + PRED] == np.datetime64(day)), (name, t)
    # 57 windows exist across the 3 contiguous days; the 30 straddling a day boundary are dropped.
    assert len(window_starts(hourly.times, SEQ, PRED)) == 57


def test_split_date_blocks_must_not_overlap(make_cfg, days):
    cfg = make_cfg({"split.train": [days[0], days[1]], "split.val": [days[1], days[2]]})
    with pytest.raises(ValueError, match="overlap"):
        load_and_split(cfg)


def test_single_day_requires_smoke_test_mode(make_cfg, days):
    with pytest.raises(ValueError, match="smoke_test"):
        load_and_split(make_cfg({"data.dates": [days[0]]}))


def test_conc_time_offset_aligns_the_files(make_cfg):
    hourly, _ = load_and_split(make_cfg())
    assert hourly.times[0] == np.datetime64("2018-11-13T08")  # 00 local = 08Z
    # Trusting the out.combine TFLAG leaves only 08Z-23Z in common each day, and says so.
    cfg = make_cfg({"data.time_offset_hours": {"meteo": 0, "conc": 0, "emis": 0}})
    with pytest.warns(UserWarning, match="only 16 hours"):
        hourly, _ = load_and_split(cfg)
    assert len(hourly.times) == 3 * 16
