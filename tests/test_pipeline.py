import json

import numpy as np
import pandas as pd
import pytest

from aerocast.config import save_config
from aerocast.data import WindowDataset
from aerocast.evaluate import RESULT_COLUMNS, evaluate_run
from aerocast.normalize import NormStats
from aerocast.splits import SmokeTestWarning, load_and_split
from aerocast.train import main as train_main
from aerocast.train import train

SEQ, PRED = 6, 10


def read_results(cfg):
    return pd.read_csv(cfg["output"]["results_csv"], dtype={"lead_hour": str})


def test_two_epoch_train_and_evaluate_writes_results(make_cfg, tmp_path):
    cfg = make_cfg()
    save_config(cfg, tmp_path / "config.yaml")
    train_main(["--config", str(tmp_path / "config.yaml")])  # trains, then evaluates

    (run_dir,) = (tmp_path / "runs").iterdir()
    for name in ("config.yaml", "run.json", "norm_stats.json", "best.pt", "history.csv", "per_window_rmse.csv"):
        assert (run_dir / name).exists(), name
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["epochs_run"] == 2 and meta["git_commit"] and meta["config_hash"]
    assert meta["evaluation"]["target_unit"] == "ppbV"

    results = read_results(cfg)
    assert list(results.columns) == RESULT_COLUMNS
    assert set(results.split) == {"val", "test"}
    assert set(results.model) == {"convlstm", "persistence", "climatology"}
    assert set(results.metric) == {"rmse", "mae", "bias", "pearson_r"}
    assert set(results.lead_hour) == {str(k) for k in range(1, PRED + 1)} | {"all"}
    assert len(results) == 2 * 3 * (PRED + 1) * 4
    assert (results.run_id == run_dir.name).all() and (results.config_hash == meta["config_hash"]).all()

    # Every model is scored on exactly the same windows.
    windows = pd.read_csv(run_dir / "per_window_rmse.csv")
    scored = {m: list(zip(g.split, g.first_target_hour_utc)) for m, g in windows.groupby("model")}
    assert len(scored) == 3 and len(scored["convlstm"]) == 18
    assert scored["persistence"] == scored["convlstm"] == scored["climatology"]


@pytest.mark.parametrize("mode", ["smoke_test", "legacy"])
def test_in_sample_modes_never_write_test_rows(make_cfg, days, mode):
    cfg = make_cfg({"split.mode": mode, "data.dates": [days[0]], "train.epochs": 1})
    with pytest.warns(SmokeTestWarning):
        run_dir = train(cfg)
    with pytest.warns(SmokeTestWarning):
        evaluate_run(run_dir)
    assert set(read_results(cfg).split) == {mode}


def test_feature_flags_train_and_evaluate(make_cfg):
    cfg = make_cfg({"features.include_o3_input": True, "features.future_forcings": True, "train.epochs": 1})
    run_dir = train(cfg)
    channels = NormStats.load(run_dir / "norm_stats.json").channels
    assert channels[:7] == ["TEMP2", "WSPD10", "WDIR10", "NO", "NO2", "PM25_CL", "O3"]
    evaluate_run(run_dir)
    assert set(read_results(cfg).model) == {"convlstm", "persistence", "climatology"}


def test_future_frames_carry_forecast_hour_forcings(make_cfg):
    hourly, splits = load_and_split(make_cfg({"features.include_o3_input": True}))
    forcing = hourly.forcing_channels
    state = [c for c in range(len(hourly.channels)) if c not in forcing]
    assert [hourly.channels[c] for c in state] == ["NO", "NO2", "PM25_CL", "O3"]
    dataset = WindowDataset(hourly.x, hourly.y, splits.train, SEQ, PRED, forcing)
    t = int(splits.train[0])
    _, _, future = dataset[0]
    assert future.shape == (PRED - 1,) + hourly.x.shape[1:]
    for k in range(PRED - 1):
        np.testing.assert_array_equal(future[k, forcing], hourly.x[t + k, forcing])
        np.testing.assert_array_equal(future[k, state], hourly.x[t - 1, state])


def test_fixed_seed_repeats_training(make_cfg):
    cfg = make_cfg({"train.epochs": 1})
    histories = [pd.read_csv(train(cfg) / "history.csv") for _ in range(2)]
    pd.testing.assert_frame_equal(*histories)
