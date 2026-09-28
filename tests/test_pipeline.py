import json
import multiprocessing

import pandas as pd
import pytest

from aerocast.config import save_config
from aerocast.evaluate import RESULT_COLUMNS, append_results, evaluate_run
from aerocast.normalize import NormStats
from aerocast.splits import SmokeTestWarning
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
    assert {"rmse", "mae", "bias", "pearson_r", "csi_p95", "spectral_ratio"} <= set(results.metric)
    assert set(results.lead_hour) == {str(k) for k in range(1, PRED + 1)} | {"all"}
    # Per model and split: 4 metrics + 3 high-Ox metrics per lead and overall, 3 spectral ratios.
    assert len(results) == 2 * 3 * ((PRED + 1) * 7 + 3)
    assert (results.run_id == run_dir.name).all() and (results.config_hash == meta["config_hash"]).all()

    # Every model is scored on exactly the same windows.
    windows = pd.read_csv(run_dir / "per_window_rmse.csv")
    scored = {m: list(zip(g.split, g.first_target_hour_utc)) for m, g in windows.groupby("model")}
    assert len(scored) == 3 and len(scored["convlstm"]) == 18
    assert scored["persistence"] == scored["convlstm"] == scored["climatology"]

    # One EXPERIMENTS.md line per run, with the val and test Ox RMSE.
    lines = [line for line in (tmp_path / "EXPERIMENTS.md").read_text().splitlines() if run_dir.name in line]
    val = results.query("model == 'convlstm' and split == 'val' and lead_hour == 'all' and metric == 'rmse'").value.item()
    assert len(lines) == 1 and f"| {val:.3f} |" in lines[0]


@pytest.mark.parametrize("mode", ["smoke_test", "legacy"])
def test_in_sample_modes_never_write_test_rows(make_cfg, days, mode):
    cfg = make_cfg({"split.mode": mode, "data.dates": [days[0]], "train.epochs": 1})
    with pytest.warns(SmokeTestWarning):
        run_dir = train(cfg)
    with pytest.warns(SmokeTestWarning):
        evaluate_run(run_dir)
    assert set(read_results(cfg).split) == {mode}


@pytest.mark.parametrize("flags", [(False, False), (True, True)])
def test_feature_flags_train_and_evaluate(make_cfg, flags):
    include_o3, future = flags
    cfg = make_cfg({"features.include_o3_input": include_o3, "features.future_forcings": future, "train.epochs": 1})
    run_dir = train(cfg)
    state = NormStats.load(run_dir / "norm_stats.json").channels["state"]
    assert state == ["conc:NO", "conc:NO2", "conc:PM25_CL"] + (["conc:O3"] if include_o3 else [])
    evaluate_run(run_dir)
    assert set(read_results(cfg).model) == {"convlstm", "persistence", "climatology"}


def test_fixed_seed_repeats_training(make_cfg):
    cfg = make_cfg({"train.epochs": 1})
    columns = ["train_loss", "val_loss", "val_ox_rmse"]
    histories = [pd.read_csv(train(cfg) / "history.csv")[columns] for _ in range(2)]
    pd.testing.assert_frame_equal(*histories)


def _append_block(args):
    path, i = args
    row = dict.fromkeys(RESULT_COLUMNS, "x")
    append_results(path, [dict(row, run_id=f"run{i}", lead_hour=k) for k in range(50)])


def test_concurrent_appends_share_one_header(tmp_path):
    path = tmp_path / "results.csv"
    with multiprocessing.get_context("fork").Pool(4) as pool:
        pool.map(_append_block, [(path, i) for i in range(8)])
    results = pd.read_csv(path)
    assert list(results.columns) == RESULT_COLUMNS
    assert len(results) == 8 * 50 and results.run_id.nunique() == 8
