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
        }
        settings.update(overrides or {})
        for key, value in settings.items():
            set_dotted(cfg, key, value)
        return cfg

    return make
