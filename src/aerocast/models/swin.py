"""Swin-style windowed-attention encoder-decoders on the time-stacked input.

swin       MONAI SwinUNETR (spatial_dims=2): shifted-window transformer encoder with a
           convolutional (residual U-Net) decoder. Most of its parameters sit in that decoder.
swin_unet  Swin U-Net: shifted-window transformer stages on both sides, built from MONAI's
           Swin stages; patch merging on the way down, pixel-shuffle patch expanding plus skip
           concatenation on the way up, and a final patch expansion back to the grid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR
from monai.networks.nets.swin_unetr import BasicLayer, PatchMergingV2

from aerocast.models.common import crop, pad_to_multiple, stack_time, unstack_output


class SwinUNETRForecaster(nn.Module):
    def __init__(self, spec, feature_size=24, depths=(2, 2, 2, 1), num_heads=(3, 6, 12, 24), window_size=7,
                 mlp_ratio=2.0, patch_size=2, grad_checkpointing=False):
        super().__init__()
        self.spec = spec
        self.net = SwinUNETR(in_channels=spec.stacked_channels, out_channels=spec.t_out * spec.k, patch_size=patch_size,
                             depths=tuple(depths), num_heads=tuple(num_heads), window_size=window_size,
                             mlp_ratio=mlp_ratio, feature_size=feature_size, spatial_dims=2,
                             use_checkpoint=grad_checkpointing)
        self.multiple = patch_size ** 5  # SwinUNETR needs spatial sizes divisible by patch_size**5

    def forward(self, batch):
        x, size = pad_to_multiple(stack_time(batch), self.multiple)
        if x.shape[-2] * x.shape[-1] <= self.multiple ** 2:
            # A one-cell bottleneck cannot be instance-normalized; grids this small only occur in tests.
            x = F.pad(x, (0, 0, 0, self.multiple), mode="replicate")
        return unstack_output(crop(self.net(x), size), self.spec.t_out, self.spec.k)


class ChannelLayerNorm(nn.LayerNorm):
    """LayerNorm over the channels of a (B, C, H, W) tensor."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class PatchExpand(nn.Module):
    """Upsample by `factor` with a 1x1 projection and pixel shuffle, then normalize channels."""

    def __init__(self, dim, out_dim, factor=2):
        super().__init__()
        self.proj = nn.Conv2d(dim, out_dim * factor ** 2, 1)
        self.shuffle = nn.PixelShuffle(factor)
        self.norm = ChannelLayerNorm(out_dim)

    def forward(self, x):
        return self.norm(self.shuffle(self.proj(x)))


class SwinUNet(nn.Module):
    def __init__(self, spec, embed_dim=64, depths=(2, 2, 4), num_heads=(2, 4, 8), window_size=7, patch_size=4,
                 mlp_ratio=4.0, grad_checkpointing=False):
        super().__init__()
        self.spec = spec
        stages = len(depths)
        dims = [embed_dim * 2 ** i for i in range(stages)]

        def stage(i):
            return BasicLayer(dim=dims[i], depth=depths[i], num_heads=num_heads[i], window_size=(window_size, window_size),
                              drop_path=0.0, mlp_ratio=mlp_ratio, qkv_bias=True, drop=0.0, attn_drop=0.0,
                              norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=grad_checkpointing)

        self.embed = nn.Conv2d(spec.stacked_channels, dims[0], patch_size, stride=patch_size)
        self.embed_norm = ChannelLayerNorm(dims[0])
        self.encoder = nn.ModuleList([stage(i) for i in range(stages)])
        self.merge = nn.ModuleList([PatchMergingV2(dims[i], spatial_dims=2) for i in range(stages - 1)])
        self.expand = nn.ModuleList([PatchExpand(dims[i], dims[i - 1]) for i in range(stages - 1, 0, -1)])
        self.fuse = nn.ModuleList([nn.Conv2d(2 * dims[i - 1], dims[i - 1], 1) for i in range(stages - 1, 0, -1)])
        self.decoder = nn.ModuleList([stage(i - 1) for i in range(stages - 1, 0, -1)])
        self.final = PatchExpand(dims[0], max(dims[0] // 4, 8), factor=patch_size)
        self.head = nn.Conv2d(max(dims[0] // 4, 8), spec.t_out * spec.k, 1)
        self.multiple = patch_size * 2 ** (stages - 1)

    def forward(self, batch):
        x, size = pad_to_multiple(stack_time(batch), self.multiple)
        x = self.embed_norm(self.embed(x))
        skips = []
        for i, stage in enumerate(self.encoder):
            if i:  # patch merging works on channels-last tensors
                x = self.merge[i - 1](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            x = stage(x)
            skips.append(x)
        x = skips.pop()
        for expand, fuse, stage in zip(self.expand, self.fuse, self.decoder):
            x = stage(fuse(torch.cat([expand(x), skips.pop()], dim=1)))
        return unstack_output(crop(self.head(self.final(x)), size), self.spec.t_out, self.spec.k)
