"""Fourier neural operator (neuraloperator's FNO) on the time-stacked input.

Four Fourier layers keep n_modes modes per dimension (neuraloperator counts both signs along
the first axis and keeps n_modes // 2 + 1 of the real FFT along the last). The domain is not
periodic, so the lifted fields are padded by a fraction of the grid before the FFTs and
cropped after; the padding scales with the input, so the model runs on grids of any size,
including ones it was not trained on. Positional embedding is off: the static grid:X/grid:Y
channels already carry normalized coordinates. The whole operator runs in fp32 (FFTs do not
support bf16), even when training uses autocast.
"""
import torch
import torch.nn as nn
from neuralop.models import FNO

from aerocast.models.common import stack_time, unstack_output


class FNOForecaster(nn.Module):
    def __init__(self, spec, hidden_channels=64, n_modes=16, n_layers=4, domain_padding=0.125,
                 lifting_channel_ratio=2, projection_channel_ratio=2, grad_checkpointing=False):
        super().__init__()
        if grad_checkpointing:
            raise ValueError("fno: gradient checkpointing is not implemented")
        self.spec = spec
        self.fno = FNO(n_modes=(n_modes, n_modes), in_channels=spec.stacked_channels,
                       out_channels=spec.t_out * spec.k, hidden_channels=hidden_channels, n_layers=n_layers,
                       lifting_channel_ratio=lifting_channel_ratio, projection_channel_ratio=projection_channel_ratio,
                       positional_embedding=None, domain_padding=domain_padding, fno_block_precision="full")

    def forward(self, batch):
        x = stack_time(batch)
        with torch.autocast(device_type=x.device.type, enabled=False):
            y = self.fno(x.float())
        return unstack_output(y, self.spec.t_out, self.spec.k)
