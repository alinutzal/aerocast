"""Every model config, and every option of its tuning knob, stays within the parameter budget."""
from pathlib import Path

import pytest
import yaml

from aerocast.config import load_config
from aerocast.models import REGISTRY, build_model, count_parameters

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = yaml.safe_load((ROOT / "configs" / "experiments" / "benchmark.yaml").read_text())["benchmark"]
LOW, HIGH = BENCHMARK["parameter_budget"]


def knob_settings(tuning):
    keys = tuning["knob"] if isinstance(tuning["knob"], list) else [tuning["knob"]]
    for option in tuning["options"]:
        values = option if isinstance(tuning["knob"], list) else [option]
        yield dict(zip(keys, values))


def test_every_registered_model_has_a_config():
    assert set(BENCHMARK["models"]) == set(REGISTRY)
    for name in REGISTRY:
        assert (ROOT / "configs" / "models" / f"{name}.yaml").exists()


@pytest.mark.parametrize("name", sorted(REGISTRY))
@pytest.mark.parametrize("mode", ["ox", "multitask"])
def test_parameters_within_budget_for_every_knob_option(name, mode):
    cfg = load_config(ROOT / "configs" / "smoke.yaml", [f"target_mode={mode}"], model=name)
    for setting in knob_settings(cfg["tuning"]):
        model_cfg = {**cfg["model"], **setting, **({"backend": "mambapy"} if name == "mamba" else {})}
        params = count_parameters(build_model({**cfg, "model": model_cfg}))
        assert LOW <= params <= HIGH, f"{name} {setting}: {params / 1e6:.2f}M"
