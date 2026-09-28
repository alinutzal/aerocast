import pandas as pd
import pytest
import torch

from aerocast.evaluate import evaluate_run
from aerocast.normalize import NormStats
from aerocast.targets import TargetLoss, physical, physical_ox, target_channels
from aerocast.train import train

# Deliberately different scales per target, so a sum in normalized space would be visibly wrong.
STATS = NormStats(mode="full", channels={}, target_mean={"NO2": 10.0, "O3": 30.0, "Ox": 40.0},
                  target_std={"NO2": 5.0, "O3": 8.0, "Ox": 9.0})
LOSS_CFG = {"type": "mixed", "huber_beta": 0.5, "alpha": 0.7, "consistency_weight": 1.0}


def test_target_channels_per_mode(make_cfg):
    cfg = make_cfg()
    for mode, expected in (("ox", ["Ox"]), ("species", ["NO2", "O3"]), ("multitask", ["NO2", "O3", "Ox"])):
        assert target_channels(dict(cfg, target_mode=mode)) == expected
    with pytest.raises(ValueError):
        target_channels(dict(cfg, target_mode="nox"))


def test_species_mode_sums_in_physical_units():
    pred = torch.randn(2, 10, 2, 4, 3)
    ox = physical_ox(pred, STATS, ["NO2", "O3"])
    no2, o3 = pred[:, :, 0] * 5 + 10, pred[:, :, 1] * 8 + 30
    torch.testing.assert_close(ox, no2 + o3)
    # Summing in normalized space and denormalizing as Ox would give something else.
    assert not torch.allclose(ox, (pred[:, :, 0] + pred[:, :, 1]) * 9 + 40)


def test_multitask_scores_the_ox_head_and_reports_the_species_sum():
    pred = torch.randn(2, 10, 3, 4, 3)
    fields = physical(pred, STATS, ["NO2", "O3", "Ox"])
    torch.testing.assert_close(fields["Ox"], pred[:, :, 2] * 9 + 40)
    torch.testing.assert_close(fields["Ox_sum"], fields["NO2"] + fields["O3"])


def test_multitask_consistency_term_in_physical_units():
    names = ["NO2", "O3", "Ox"]
    loss = TargetLoss(LOSS_CFG, names, STATS)
    target = torch.zeros(1, 10, 3, 4, 3)
    consistent = target.clone()
    consistent[:, :, 2] = (10 + 30 - 40) / 9  # Ox head equals NO2 + O3 in ppb: zero gap
    base = TargetLoss(dict(LOSS_CFG, consistency_weight=0.0), names, STATS)
    assert loss(consistent, target).item() == pytest.approx(base(consistent, target).item())
    shifted = consistent.clone()
    shifted[:, :, 2] += 1.0  # Ox head 9 ppb above NO2 + O3: gap^2 / sigma_Ox^2 = 1
    assert loss(shifted, target).item() - base(shifted, target).item() == pytest.approx(1.0, rel=1e-5)
    assert TargetLoss(LOSS_CFG, ["NO2", "O3"], STATS).consistency == 0.0  # species mode has no Ox head


@pytest.mark.parametrize("mode", ["species", "multitask"])
def test_target_modes_train_and_evaluate(make_cfg, mode):
    cfg = make_cfg({"target_mode": mode, "train.epochs": 1})
    run_dir = train(cfg)
    evaluate_run(run_dir)
    results = pd.read_csv(cfg["output"]["results_csv"], dtype={"lead_hour": str})
    ox = results.query("model == 'convlstm' and metric == 'rmse' and lead_hour == 'all'")
    assert set(ox.split) == {"val", "test"} and ox.value.notna().all()
