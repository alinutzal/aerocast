"""Benchmark runner: tuning trials, selection, final runs, resume, and the date-split rule."""
import json

import pandas as pd
import pytest
import yaml
from conftest import benchmark_args, load_script


def test_runner_stages(benchmark_dir):
    results = pd.read_csv(benchmark_dir / "results" / "results.csv", dtype={"lead_hour": str})
    runs = {p.parent.name: json.loads(p.read_text()) for p in (benchmark_dir / "runs").glob("*/run.json")}
    configs = {p.parent.name: yaml.safe_load(p.read_text()) for p in (benchmark_dir / "runs").glob("*/config.yaml")}
    assert len(runs) == 2 * 2 + 2 * 2 * 1  # 2 trials x 2 models, then 2 models x 2 modes x 1 seed
    tuning = [r for r, m in runs.items() if m["config_name"].startswith("tune-")]
    assert all(set(results[results.run_id == r].split) == {"val"} for r in tuning)  # tuning never writes test rows
    finals = [r for r in runs if r not in tuning]
    assert all(set(results[results.run_id == r].split) == {"val", "test"} for r in finals)
    for trial in range(2):  # every model gets the same lr and weight decay in trial i
        draws = {(configs[r]["train"]["lr"], configs[r]["train"]["weight_decay"])
                 for r in tuning if runs[r]["config_name"].endswith(f"-t{trial}")}
        assert len(draws) == 1
    assert all(configs[r]["train"]["epochs"] == 1 for r in runs)
    selection = json.loads((benchmark_dir / "results" / "benchmark_selection.json").read_text())
    assert set(selection) == {"unet", "fno"}
    for model, chosen in selection.items():  # final runs use the selected trial's settings
        final = [r for r in finals if runs[r]["model"] == model]
        assert all(configs[r]["train"]["lr"] == chosen["lr"] for r in final)


def test_runner_resumes_without_rerunning(benchmark_dir, five_days):
    results_csv = benchmark_dir / "results" / "results.csv"
    before = len(pd.read_csv(results_csv))
    load_script("run_benchmark").main(benchmark_args(five_days, benchmark_dir))
    assert len(pd.read_csv(results_csv)) == before


def test_benchmark_needs_a_date_split(five_days, tmp_path):
    with pytest.raises(SystemExit, match="split.mode: dates"):
        load_script("run_benchmark").main(benchmark_args(five_days, tmp_path, ["split.mode=smoke_test"]))
