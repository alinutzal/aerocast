import importlib.util
from pathlib import Path

import pytest

from aerocast.config import load_config, set_dotted

ROOT = Path(__file__).resolve().parents[1]
DAYS = ["2018-11-13", "2018-11-14", "2018-11-15"]


def _synthetic_writer():
    spec = importlib.util.spec_from_file_location("make_synthetic_data", ROOT / "scripts" / "make_synthetic_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_synthetic_data


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    _synthetic_writer()(out, start=DAYS[0], n_days=len(DAYS), rows=16, cols=12, seed=0)
    return out


@pytest.fixture
def days():
    return list(DAYS)


@pytest.fixture
def make_cfg(synthetic_dir, tmp_path):
    """configs/base.yaml on the synthetic days (train, val, test = one day each), outputs in tmp_path."""

    def make(overrides=None):
        cfg = load_config(ROOT / "configs" / "base.yaml")
        settings = {
            "data.data_dir": str(synthetic_dir),
            "data.cache_dir": str(tmp_path / "cache"),
            "split.train": [DAYS[0], DAYS[0]],
            "split.val": [DAYS[1], DAYS[1]],
            "split.test": [DAYS[2], DAYS[2]],
            "train.epochs": 2,
            "train.log_every": 1,
            "output.runs_dir": str(tmp_path / "runs"),
            "output.results_csv": str(tmp_path / "results" / "results.csv"),
            "output.experiments_md": str(tmp_path / "EXPERIMENTS.md"),
            # a small ConvLSTM: these tests exercise the pipeline, not the model
            "model.hidden_dims": [16, 8],
            "model.num_groups": 4,
            "model.grad_checkpointing": False,
        }
        settings.update(overrides or {})
        for key, value in settings.items():
            set_dotted(cfg, key, value)
        return cfg

    return make


BENCHMARK_DAYS = ["2018-11-13", "2018-11-14", "2018-11-15", "2018-11-16", "2018-11-17"]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_args(data_dir, tmp, extra=()):
    """run_benchmark.py arguments: 2 models, 2 target modes, 1 seed, 2 trials, 1 epoch, on
    synthetic days split 2 train / 1 val / 2 test (two test days, so bootstrap CIs exist)."""
    d = BENCHMARK_DAYS
    settings = [f"data.data_dir={data_dir}", f"data.cache_dir={tmp / 'cache'}",
                f"split.train=['{d[0]}', '{d[1]}']", f"split.val=['{d[2]}', '{d[2]}']", f"split.test=['{d[3]}', '{d[4]}']",
                "train.epochs=1", "train.patience=1", "benchmark.models=['unet', 'fno']",
                "benchmark.target_modes=['ox', 'species']", "benchmark.seeds=[1]", "benchmark.tuning.trials=2",
                f"output.runs_dir={tmp / 'runs'}", f"output.results_csv={tmp / 'results' / 'results.csv'}",
                f"output.experiments_md={tmp / 'EXPERIMENTS.md'}", *extra]
    return [arg for s in settings for arg in ("--set", s)]


@pytest.fixture(scope="session")
def five_days(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic5")
    _synthetic_writer()(out, start=BENCHMARK_DAYS[0], n_days=5, rows=16, cols=12, seed=1)
    return out


@pytest.fixture(scope="session")
def benchmark_dir(five_days, tmp_path_factory):
    """One small benchmark run (tune, select, final) shared by the runner and table tests."""
    tmp = tmp_path_factory.mktemp("benchmark")
    load_script("run_benchmark").main(benchmark_args(five_days, tmp))
    return tmp
