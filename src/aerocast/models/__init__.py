"""Model registry.

Every model maps a batch {"state": (B, T_in, C_state, H, W), "forcing": (B, T_in + T_out,
C_forcing, H, W), "static": (B, C_static, H, W)} to (B, T_out, K, H, W) in normalized units.
Models are built from a ModelSpec (channel names, hours, K), never from H or W.
"""
from dataclasses import dataclass

from aerocast.data import channel_layout
from aerocast.models.convlstm import ConvLSTMCell, ConvLSTMForecaster, StackedConvLSTM
from aerocast.models.fno import FNOForecaster
from aerocast.models.gnn import MeshGNN
from aerocast.models.mamba import VMambaForecaster
from aerocast.models.swin import SwinUNet, SwinUNETRForecaster
from aerocast.models.unet import UNet
from aerocast.targets import target_channels


@dataclass(frozen=True)
class ModelSpec:
    t_in: int
    t_out: int
    state_channels: tuple
    forcing_channels: tuple
    static_channels: tuple
    target_channels: tuple

    @property
    def k(self):
        return len(self.target_channels)

    @property
    def frame_channels(self):
        """Channels of one hourly frame [state, forcing, static] (recurrent models)."""
        return len(self.state_channels) + len(self.forcing_channels) + len(self.static_channels)

    @property
    def stacked_channels(self):
        """Channels after stacking every hour into channels (2D models)."""
        return (self.t_in * len(self.state_channels) + (self.t_in + self.t_out) * len(self.forcing_channels)
                + len(self.static_channels))


REGISTRY = {"convlstm": ConvLSTMForecaster, "unet": UNet, "fno": FNOForecaster,
            "swin": SwinUNETRForecaster, "swin_unet": SwinUNet, "gnn": MeshGNN,
            "mamba": VMambaForecaster}


def model_spec(cfg):
    layout = channel_layout(cfg["data"], cfg["features"])
    return ModelSpec(cfg["data"]["seq_len"], cfg["data"]["pred_len"], tuple(layout["state"]),
                     tuple(layout["forcing"]), tuple(layout["static"]), tuple(target_channels(cfg)))


def build_model(cfg, stats=None):
    """`stats` (fitted NormStats) is only used when the model config sets enforce_positive:
    true, to compute each target channel's floor in normalized units (any model can opt into
    this, not just FNO - see FNOForecaster for what it does with it). Without stats (e.g. a
    model built to inspect its shape/parameters before data is loaded), enforce_positive
    still works, just with the floor defaulting to 0 until a real checkpoint is loaded."""
    params = dict(cfg["model"])
    name = params.pop("name")
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}; registered: {sorted(REGISTRY)}")
    spec = model_spec(cfg)
    if params.get("enforce_positive") and stats is not None and stats.target_mean is not None:
        params["target_floor"] = [-stats.target_mean[c] / stats.target_std[c] for c in spec.target_channels]
    return REGISTRY[name](spec, **params)


def count_parameters(model):
    """Trainable parameters, counting complex weights as two real numbers."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)


__all__ = ["ConvLSTMCell", "ConvLSTMForecaster", "FNOForecaster", "MeshGNN", "ModelSpec", "REGISTRY", "StackedConvLSTM", "SwinUNETRForecaster", "SwinUNet", "UNet", "VMambaForecaster",
           "build_model", "count_parameters", "model_spec"]
