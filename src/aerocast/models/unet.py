"""Plain 2D U-Net on the time-stacked input, the usual setup for CMAQ emulation.

Each level has two 3x3 conv + GroupNorm + ReLU layers; levels are joined by 2x2 max-pooling
on the way down and by a 1x1 channel reduction plus bilinear upsampling on the way up, with
skip connections concatenated. All 10 lead hours come out of one 1x1 head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from aerocast.models.common import crop, pad_to_multiple, stack_time, unstack_output


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_groups):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(norm_groups, out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(norm_groups, out_channels), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, spec, width=52, levels=4, norm_groups=4, grad_checkpointing=False):
        super().__init__()
        self.spec, self.grad_checkpointing = spec, grad_checkpointing
        widths = [width * 2 ** i for i in range(levels)]
        self.down = nn.ModuleList([ConvBlock(spec.stacked_channels, widths[0], norm_groups)]
                                  + [ConvBlock(widths[i - 1], widths[i], norm_groups) for i in range(1, levels)])
        self.reduce = nn.ModuleList([nn.Conv2d(widths[i], widths[i - 1], 1) for i in range(levels - 1, 0, -1)])
        self.up = nn.ModuleList([ConvBlock(2 * widths[i - 1], widths[i - 1], norm_groups) for i in range(levels - 1, 0, -1)])
        self.head = nn.Conv2d(widths[0], spec.t_out * spec.k, 1)
        self.multiple = 2 ** (levels - 1)

    def _block(self, block, x):
        if self.grad_checkpointing and self.training:
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, batch):
        x, size = pad_to_multiple(stack_time(batch), self.multiple)
        skips = []
        for i, block in enumerate(self.down):
            x = self._block(block, F.max_pool2d(x, 2) if i else x)
            skips.append(x)
        x = skips.pop()
        for reduce, block in zip(self.reduce, self.up):
            x = F.interpolate(reduce(x), scale_factor=2, mode="bilinear", align_corners=False)
            x = self._block(block, torch.cat([x, skips.pop()], dim=1))
        return unstack_output(crop(self.head(x), size), self.spec.t_out, self.spec.k)
