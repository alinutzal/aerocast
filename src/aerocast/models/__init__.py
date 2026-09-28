"""Model registry.

Every model maps a batch {"state": (B, T_in, C_state, H, W), "forcing": (B, T_in + T_out,
C_forcing, H, W), "static": (B, C_static, H, W)} to (B, T_out, K, H, W) in normalized units.
Models are built from a ModelSpec (channel names, hours, K), never from H or W.
"""
from dataclasses import dataclass

from aerocast.data import channel_layout
from aerocast.models.convlstm import ConvLSTMCell, ConvLSTMForecaster, StackedConvLSTM
from aerocast.models.fno import FNOForecaster
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


REGISTRY = {"convlstm": ConvLSTMForecaster, "unet": UNet, "fno": FNOForecaster}


def model_spec(cfg):
    layout = channel_layout(cfg["data"], cfg["features"])
    return ModelSpec(cfg["data"]["seq_len"], cfg["data"]["pred_len"], tuple(layout["state"]),
                     tuple(layout["forcing"]), tuple(layout["static"]), tuple(target_channels(cfg)))


def build_model(cfg):
    params = dict(cfg["model"])
    name = params.pop("name")
    if name not in REGISTRY:
        raise ValueError(f"Unknown model {name!r}; registered: {sorted(REGISTRY)}")
    return REGISTRY[name](model_spec(cfg), **params)


def count_parameters(model):
    """Trainable parameters, counting complex weights as two real numbers."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)


__all__ = ["ConvLSTMCell", "ConvLSTMForecaster", "FNOForecaster", "ModelSpec", "REGISTRY", "StackedConvLSTM", "UNet",
           "build_model", "count_parameters", "model_spec"]
