"""Interface tests run for every registered model, on small versions of each so they fit a CPU."""
import pytest
import torch

from aerocast.models import REGISTRY, ModelSpec
from aerocast.models.common import crop, pad_to_multiple, stack_time
from aerocast.normalize import fit_stats, normalized_arrays
from aerocast.data import WindowDataset
from aerocast.splits import load_and_split
from aerocast.targets import base_loss

STATE = ("conc:NO", "conc:NO2", "conc:PM25_CL", "conc:O3")
FORCING = ("meteo:TEMP2", "meteo:WSPD10", "meteo:U10", "meteo:V10", "emis:NO", "emis:NO2", "emis:HONO",
           "emis:ALK1", "emis:OLE1", "emis:ARO1", "emis:ARO2", "emis:TERP", "emis:ISOP", "time:HOUR_SIN", "time:HOUR_COS")
STATIC = ("grid:X", "grid:Y")
T_IN, T_OUT = 6, 10

# Same architectures as configs/models/*.yaml, with fewer channels.
SMALL = {
    "convlstm": {"hidden_dims": [16, 8], "kernel_size": 3, "norm": "group", "num_groups": 4},
    "unet": {"width": 8, "levels": 4, "norm_groups": 4},
    "fno": {"hidden_channels": 16, "n_modes": 8, "n_layers": 4},
    "swin": {"feature_size": 12, "depths": [2, 2, 2, 1], "window_size": 7, "mlp_ratio": 2.0},
    "swin_unet": {"embed_dim": 16, "depths": [2, 2, 2], "num_heads": [2, 4, 8], "window_size": 7},
    "gnn": {"latent": 32, "processor_steps": 2},
    "mamba": {"dim": 16, "depth": 2, "d_state": 4, "skip_channels": 8, "backend": "mambapy"},
}
MODELS = sorted(SMALL)


def make_spec(targets=("Ox",)):
    return ModelSpec(T_IN, T_OUT, STATE, FORCING, STATIC, tuple(targets))


def make_model(name, targets=("Ox",), **overrides):
    torch.manual_seed(0)
    return REGISTRY[name](make_spec(targets), **{**SMALL[name], **overrides})


def random_batch(batch=2, height=16, width=12):
    generator = torch.Generator().manual_seed(1)
    return {"state": torch.randn(batch, T_IN, len(STATE), height, width, generator=generator),
            "forcing": torch.randn(batch, T_IN + T_OUT, len(FORCING), height, width, generator=generator),
            "static": torch.rand(batch, len(STATIC), height, width, generator=generator)}


@pytest.fixture
def tiny_batch(make_cfg):
    """Two real windows from the synthetic data (16 x 12 grid), normalized, as one batch."""
    hourly, splits = load_and_split(make_cfg())
    stats = fit_stats(hourly, splits.train, T_IN, T_OUT, "full")
    arrays = normalized_arrays(hourly, stats, ["Ox"])
    dataset = WindowDataset(**arrays, starts=splits.train[[0, 4]], t_in=T_IN, t_out=T_OUT)
    return {key: torch.stack([dataset[0][key], dataset[1][key]]) for key in dataset[0]}


def test_stack_time_keeps_hours_together():
    batch = random_batch()
    stacked = stack_time(batch)
    assert stacked.shape == (2, T_IN * len(STATE) + (T_IN + T_OUT) * len(FORCING) + len(STATIC), 16, 12)
    torch.testing.assert_close(stacked[:, :len(STATE)], batch["state"][:, 0])
    offset = T_IN * len(STATE) + len(FORCING) * (T_IN + 3)
    torch.testing.assert_close(stacked[:, offset:offset + len(FORCING)], batch["forcing"][:, T_IN + 3])
    torch.testing.assert_close(stacked[:, -len(STATIC):], batch["static"])


def test_pad_and_crop_round_trip():
    x = torch.randn(2, 3, 16, 12)
    padded, size = pad_to_multiple(x, 32)
    assert padded.shape[-2:] == (32, 32) and size == (16, 12)
    torch.testing.assert_close(crop(padded, size), x)
    torch.testing.assert_close(padded[..., -1, :12], x[..., -1, :])  # edge rows repeated


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("size", [(16, 12), (36, 28)])
@pytest.mark.parametrize("targets", [("Ox",), ("NO2", "O3", "Ox")])
def test_output_shape_on_any_grid(name, size, targets):
    model = make_model(name, targets).eval()
    with torch.no_grad():
        out = model(random_batch(height=size[0], width=size[1]))
    assert out.shape == (2, T_OUT, len(targets), *size)


@pytest.mark.parametrize("name", MODELS)
def test_gradients_reach_all_parameters(name):
    # 36 x 28 so every attention window holds more than one token (a lone token ignores its position bias)
    model = make_model(name).train()
    model(random_batch(height=36, width=28)).square().mean().backward()
    for pname, param in model.named_parameters():
        assert param.grad is not None, f"{pname} gets no gradient"
        assert torch.isfinite(param.grad).all() and param.grad.abs().sum() > 0, f"{pname} gradient is zero or not finite"


@pytest.mark.parametrize("name", MODELS)
def test_overfits_one_tiny_batch(name, tiny_batch):
    model = make_model(name).train()
    loss_fn = base_loss({"type": "mixed", "huber_beta": 0.5, "alpha": 0.7})
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(200):
        loss = loss_fn(model(tiny_batch), tiny_batch["target"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0], f"loss {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_convlstm_rollout_is_causal():
    """Forcing of hour t+k is fed before predicting hour t+k+1, so it cannot change leads 0..k."""
    model = make_model("convlstm").eval()
    batch = random_batch()
    with torch.no_grad():
        base = model(batch)
        changed = dict(batch, forcing=batch["forcing"].clone())
        changed["forcing"][:, T_IN + 3] += 5.0
        out = model(changed)
    torch.testing.assert_close(out[:, :4], base[:, :4])
    assert not torch.allclose(out[:, 4], base[:, 4])


def test_fno_runs_at_twice_the_resolution(tiny_batch):
    """Trained on 16 x 12, the FNO predicts on a 32 x 24 version of the same fields."""
    model = make_model("fno").train()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(5):
        loss = (model(tiny_batch) - tiny_batch["target"]).square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    fine = {key: torch.nn.functional.interpolate(value.flatten(0, -3)[None], scale_factor=2, mode="bilinear")[0]
            .reshape(*value.shape[:-2], 32, 24) for key, value in tiny_batch.items() if key != "target"}
    with torch.no_grad():
        out = model.eval()(fine)
    assert out.shape == (2, T_OUT, 1, 32, 24) and torch.isfinite(out).all()


@pytest.mark.parametrize("name", MODELS)
def test_checkpoint_round_trip_with_weights_only_loading(name, tmp_path):
    from aerocast.train import model_state
    model = make_model(name)
    torch.save({"model_state_dict": model_state(model)}, tmp_path / "best.pt")
    restored = make_model(name)
    restored.load_state_dict(torch.load(tmp_path / "best.pt", weights_only=True)["model_state_dict"])
    batch = random_batch()
    with torch.no_grad():
        torch.testing.assert_close(restored.eval()(batch), model.eval()(batch))


def test_gnn_mesh_links_cells_4_and_16_apart():
    import numpy as np
    from aerocast.models.gnn import build_graph
    g = build_graph(224, 164)
    src, dst = g["mesh_edges"]
    lengths = np.round(np.linalg.norm(g["mesh_pos"][dst] - g["mesh_pos"][src], axis=1), 3)
    assert set(lengths) == {4.0, round(4 * 2 ** 0.5, 3), 16.0, round(16 * 2 ** 0.5, 3)}
    assert 11 * 16 >= 100  # default processor steps x coarse spacing (km) covers 10 h of transport


def test_gnn_edge_wind_is_the_along_edge_component():
    model = make_model("gnn")
    batch = random_batch(height=36, width=28)
    u, v = model.wind
    batch["forcing"][:, :, u], batch["forcing"][:, :, v] = 1.0, 0.0  # uniform wind toward +x (from the west)
    graph = model.graph(36, 28, "cpu")
    captured = {}
    encoder = model.edge_encoders["mesh"]
    handle = encoder.register_forward_hook(lambda module, inputs, output: captured.update(features=inputs[0]))
    with torch.no_grad():
        model(batch)
    handle.remove()
    geometry, unit = graph["geometry"]["mesh"]
    along = captured["features"][0, :, 3:]
    torch.testing.assert_close(along, unit[:, :1].expand_as(along))  # wind . unit vector = unit x-component
    east = (unit[:, 0] == 1) & (unit[:, 1] == 0)
    north = (unit[:, 0] == 0) & (unit[:, 1] == 1)
    assert east.any() and north.any()
    assert torch.all(along[east] == 1) and torch.all(along[north] == 0)


needs_cuda_kernel = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU and the mamba-ssm kernel")


@needs_cuda_kernel
def test_mamba_cuda_kernel_matches_the_pytorch_scan():
    from aerocast.models.mamba import SelectiveScanCuda, selective_scan_cuda, selective_scan_torch
    if selective_scan_cuda is None:
        pytest.skip("mamba-ssm kernel not installed")
    torch.manual_seed(0)
    b, k, d, n, length = 2, 4, 8, 4, 50
    cuda = {"device": "cuda", "requires_grad": True}
    inputs = [torch.randn(b, k * d, length, **cuda), (0.1 * torch.randn(b, k * d, length, device="cuda")).requires_grad_(),
              (-torch.rand(k * d, n, device="cuda") - 0.5).requires_grad_(), torch.randn(b, k, n, length, **cuda),
              torch.randn(b, k, n, length, **cuda), torch.randn(k * d, **cuda), torch.randn(k * d, **cuda)]
    out_cuda = SelectiveScanCuda.apply(*inputs, True)
    out_ref = selective_scan_torch(*inputs, True)
    torch.testing.assert_close(out_cuda, out_ref, rtol=1e-4, atol=1e-4)
    for g_cuda, g_ref in zip(torch.autograd.grad(out_cuda.square().sum(), inputs),
                             torch.autograd.grad(out_ref.square().sum(), inputs)):
        torch.testing.assert_close(g_cuda, g_ref, rtol=1e-3, atol=1e-3)


@needs_cuda_kernel
def test_mamba_backends_agree_on_the_model():
    torch.manual_seed(0)
    reference = make_model("mamba").cuda().eval()
    kernel = make_model("mamba", backend="mamba_ssm").cuda().eval()
    kernel.load_state_dict(reference.state_dict())
    batch = {key: value.cuda() for key, value in random_batch(height=36, width=28).items()}
    with torch.no_grad():
        torch.testing.assert_close(kernel(batch), reference(batch), rtol=1e-4, atol=1e-4)
