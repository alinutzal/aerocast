import json

import numpy as np
import pandas as pd
import pytest

from aerocast.evaluate import evaluate_run
from aerocast.metrics import ForecastScores, radial_bins, radial_power
from aerocast.splits import load_and_split
from aerocast.train import train

LEADS, H, W = 10, 16, 12


def random_windows(n, seed=0, noise=2.0):
    rng = np.random.default_rng(seed)
    obs = 40 + 5 * rng.standard_normal((n, LEADS, H, W))
    return obs + noise * rng.standard_normal(obs.shape), obs


def test_day_sums_give_the_direct_metrics():
    pred, obs = random_windows(6)
    scores = ForecastScores(LEADS, shift=40.0, threshold=45.0)
    for i in range(6):
        scores.update(pred[i], obs[i], day=f"d{i % 3}")
    rows = {(lead, m): v for lead, m, v in scores.rows()}
    e = pred - obs
    assert rows[("all", "rmse")] == pytest.approx(np.sqrt(np.mean(e ** 2)))
    assert rows[("all", "mae")] == pytest.approx(np.mean(np.abs(e)))
    assert rows[("all", "bias")] == pytest.approx(np.mean(e))
    assert rows[(3, "pearson_r")] == pytest.approx(np.corrcoef(pred[:, 2].ravel(), obs[:, 2].ravel())[0, 1])
    hit, obs_hi = (pred > 45) & (obs > 45), obs > 45
    csi = hit.sum() / ((pred > 45) | obs_hi).sum()
    assert rows[("all", "csi_p95")] == pytest.approx(csi)
    assert rows[("all", "f1_p95")] == pytest.approx(2 * hit.sum() / ((pred > 45).sum() + obs_hi.sum()))
    assert rows[("all", "rmse_above_p95")] == pytest.approx(np.sqrt(np.mean(e[obs_hi] ** 2)))


def test_spectral_ratio_flags_smoothing():
    _, obs = random_windows(4)
    bins = radial_bins(H, W)
    exact = ForecastScores(LEADS, 40.0, spectrum_leads=(1, 5), bins=bins)
    smooth = ForecastScores(LEADS, 40.0, spectrum_leads=(1, 5), bins=bins)
    kernel = np.ones((3, 3)) / 9
    for o in obs:
        blurred = np.stack([sum(np.roll(np.roll(f, dy, 0), dx, 1) * kernel[dy + 1, dx + 1]
                                for dy in (-1, 0, 1) for dx in (-1, 0, 1)) for f in o])
        exact.update(o, o, "d0")
        smooth.update(blurred, o, "d0")
    assert exact.spectral_ratio(1) == pytest.approx(1.0)
    assert smooth.spectral_ratio(5) < 0.6


def test_radial_power_peaks_at_the_wave_number():
    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    wave = np.sin(2 * np.pi * xx * 4 / 32)  # 4 cycles across 32 cells = 0.125 cycles/cell
    index, centres = radial_bins(32, 32)
    power = radial_power(wave[None], (index, centres))[0]
    assert abs(centres[np.argmax(power)] - 0.125) < 0.5 / len(centres)


def test_block_bootstrap_brackets_the_estimate():
    pred, obs = random_windows(12, seed=1)
    scores = ForecastScores(LEADS, shift=40.0, threshold=45.0)
    for i in range(12):
        scores.update(pred[i], obs[i], day=f"2018-11-{13 + i // 2}")  # 6 days, 2 windows each
    point = {(lead, m): v for lead, m, v in scores.rows()}
    ci = {(lead, m): v for lead, m, v in scores.bootstrap_rows(500, seed=0)}
    assert ci[("all", "rmse_ci95_lo")] < point[("all", "rmse")] < ci[("all", "rmse_ci95_hi")]
    assert ci[(4, "rmse_ci95_lo")] < point[(4, "rmse")] < ci[(4, "rmse_ci95_hi")]
    assert scores.bootstrap_rows(500, seed=0) == scores.bootstrap_rows(500, seed=0)
    single = ForecastScores(LEADS, shift=40.0)
    single.update(pred[0], obs[0], day="2018-11-13")
    assert single.bootstrap_rows(500, seed=0) == []  # one day: no CI


@pytest.mark.parametrize("mode", ["ox", "species", "multitask"])
def test_evaluation_rows_and_files(make_cfg, mode):
    cfg = make_cfg({"target_mode": mode, "train.epochs": 1})
    run_dir = train(cfg)
    evaluate_run(run_dir)
    results = pd.read_csv(cfg["output"]["results_csv"], dtype={"lead_hour": str})
    metrics = set(results.metric)
    assert {"rmse", "mae", "bias", "pearson_r", "csi_p95", "f1_p95", "rmse_above_p95", "spectral_ratio"} <= metrics
    for source in ("convlstm", "persistence", "climatology"):
        assert len(results.query("model == @source and split == 'test' and metric == 'spectral_ratio'")) == 3
    if mode in ("species", "multitask"):
        for source in ("convlstm", "persistence", "climatology"):
            assert results.query("model == @source and metric in ['no2_rmse', 'o3_rmse']").shape[0] == 2 * 2 * 11
    assert ("oxsum_rmse" in metrics) == (mode == "multitask")
    for name in ("daily_stats.csv", "spectra.csv", "error_maps_test.npz", "per_window_rmse.csv"):
        assert (run_dir / name).exists(), name

    # The high-Ox threshold comes from training days only.
    hourly, splits = load_and_split(cfg)
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["evaluation"]["high_ox_threshold"] == pytest.approx(np.percentile(hourly.targets["Ox"][splits.train_hours], 95))
    maps = np.load(run_dir / "error_maps_test.npz")
    assert maps["truth"].shape == (3, 16, 12) and maps["convlstm"].shape == (3, 16, 12)


def test_test_rows_are_written_once(make_cfg):
    cfg = make_cfg({"train.epochs": 1})
    run_dir = train(cfg)
    evaluate_run(run_dir)
    first = pd.read_csv(cfg["output"]["results_csv"])
    evaluate_run(run_dir)  # re-evaluation: val again, test skipped
    second = pd.read_csv(cfg["output"]["results_csv"])
    assert (second.split == "test").sum() == (first.split == "test").sum()
    assert (second.split == "val").sum() == 2 * (first.split == "val").sum()
    evaluate_run(run_dir, labels=["val"])
    assert (pd.read_csv(cfg["output"]["results_csv"]).split == "test").sum() == (first.split == "test").sum()
