"""VMamba-style state-space model on the time-stacked input.

A patch-4 embedding turns the grid into a token grid. Each VSS block runs a 2D selective scan
(SS2D): the tokens are read in four orders (row-major, column-major, and both reversed), each
with its own selective-SSM parameters, and the four outputs are mapped back and summed, so
every token sees the whole grid. A pixel-shuffle head returns to the grid, refined with a
full-resolution convolution of the input.

The selective scan runs on one of two backends, recorded per run:
  mamba_ssm  the CUDA kernel from the mamba-ssm wheel, called directly (Linux + CUDA);
  mambapy    pure PyTorch using mambapy's parallel scan (anywhere; slower, fp32).
Benchmark numbers should all come from the same backend.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from aerocast.models.common import crop, pad_to_multiple, stack_time, unstack_output

try:
    import selective_scan_cuda  # compiled extension shipped in the mamba-ssm wheel
except ImportError:
    selective_scan_cuda = None


class SelectiveScanCuda(torch.autograd.Function):
    """mamba-ssm's selective-scan kernel without z gating: u, delta (b, d, l); A (d, n) fp32;
    B, C (b, g, n, l) grouped along d; D, delta_bias (d,) fp32. Mirrors mamba_ssm's SelectiveScanFn."""

    @staticmethod
    def forward(ctx, u, delta, A, B, C, D, delta_bias, delta_softplus):
        dtype = u.dtype
        u, delta, B, C = (t.to(dtype).contiguous() for t in (u, delta, B, C))
        A, D, delta_bias = (t.float().contiguous() for t in (A, D, delta_bias))
        out, x, *_ = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.delta_softplus = delta_softplus
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    def backward(ctx, dout):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *_ = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout.contiguous(), x, None, None, ctx.delta_softplus, False)
        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None


def selective_scan_torch(u, delta, A, B, C, D, delta_bias, delta_softplus):
    """Same computation in PyTorch (fp32) with mambapy's parallel scan:
    h_t = exp(delta_t A) h_{t-1} + delta_t B_t u_t,  y_t = C_t . h_t + D u_t."""
    from mambapy.pscan import pscan

    batch, dim, _ = u.shape
    groups = B.shape[1]
    u, delta = u.float(), delta.float() + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    B = B.float().repeat_interleave(dim // groups, dim=1)  # (b, d, n, l)
    C = C.float().repeat_interleave(dim // groups, dim=1)
    decay = torch.exp(delta.unsqueeze(-1) * A.float()[None, :, None, :])  # (b, d, l, n)
    drive = (delta * u).unsqueeze(-1) * B.transpose(2, 3)                   # (b, d, l, n)
    h = pscan(decay.transpose(1, 2), drive.transpose(1, 2))                 # (b, l, d, n)
    y = (h * C.permute(0, 3, 1, 2)).sum(-1).transpose(1, 2)                 # (b, d, l)
    return y + D.float()[None, :, None] * u


def resolve_backend(backend):
    if backend == "auto":
        return "mamba_ssm" if selective_scan_cuda is not None and torch.cuda.is_available() else "mambapy"
    if backend == "mamba_ssm" and selective_scan_cuda is None:
        raise ImportError("backend mamba_ssm needs the mamba-ssm wheel (Linux, CUDA)")
    if backend not in ("mamba_ssm", "mambapy"):
        raise ValueError(f"Unknown Mamba backend {backend!r}; expected auto, mamba_ssm or mambapy")
    return backend


class SS2D(nn.Module):
    """VMamba 2D selective scan over a (B, H, W, C) token grid, four scan orders."""

    DIRECTIONS = 4

    def __init__(self, dim, d_state=16, expand=2, d_conv=3, dt_min=0.001, dt_max=0.1, dt_floor=1e-4):
        super().__init__()
        inner, k = expand * dim, self.DIRECTIONS
        self.inner, self.d_state, self.dt_rank = inner, d_state, math.ceil(dim / 16)
        self.in_proj = nn.Linear(dim, 2 * inner, bias=False)
        self.conv = nn.Conv2d(inner, inner, d_conv, padding=d_conv // 2, groups=inner)
        bound = inner ** -0.5
        self.x_proj = nn.Parameter(torch.empty(k, self.dt_rank + 2 * d_state, inner).uniform_(-bound, bound))
        # Mamba's time-step initialisation: dt log-uniform in [dt_min, dt_max] through the softplus.
        self.dt_proj = nn.Parameter(torch.empty(k, inner, self.dt_rank).uniform_(-self.dt_rank ** -0.5, self.dt_rank ** -0.5))
        dt = torch.exp(torch.rand(k, inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(k * inner, 1))
        self.D = nn.Parameter(torch.ones(k * inner))
        self.out_norm = nn.LayerNorm(inner)
        self.out_proj = nn.Linear(inner, dim, bias=False)
        self.backend = "mambapy"

    def forward(self, x):
        b, h, w, _ = x.shape
        length = h * w
        x, z = self.in_proj(x).chunk(2, dim=-1)
        x = F.silu(self.conv(x.permute(0, 3, 1, 2)))                            # (b, inner, h, w)
        xs = torch.stack([x.flatten(2), x.transpose(2, 3).flatten(2)], dim=1)   # row-major, column-major
        xs = torch.cat([xs, xs.flip(-1)], dim=1)                                # and both reversed: (b, 4, inner, l)
        proj = torch.einsum("bkdl,kcd->bkcl", xs, self.x_proj)
        dts, Bs, Cs = torch.split(proj, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("bkrl,kdr->bkdl", dts, self.dt_proj)
        args = (xs.reshape(b, -1, length), dts.reshape(b, -1, length), -torch.exp(self.A_log.float()),
                Bs.contiguous(), Cs.contiguous(), self.D.float(), self.dt_bias.reshape(-1).float(), True)
        if self.backend == "mamba_ssm":
            if not x.is_cuda:
                raise ValueError("The mamba_ssm backend needs CUDA tensors")
            out = SelectiveScanCuda.apply(*args)
        else:
            out = selective_scan_torch(*args)
        out = out.view(b, self.DIRECTIONS, self.inner, length).float()
        rows = out[:, 0] + out[:, 2].flip(-1)
        cols = (out[:, 1] + out[:, 3].flip(-1)).view(b, self.inner, w, h).transpose(2, 3).reshape(b, self.inner, length)
        y = (rows + cols).transpose(1, 2).reshape(b, h, w, self.inner)
        return self.out_proj(self.out_norm(y) * F.silu(z))


class VSSBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, **ss2d):
        super().__init__()
        self.norm1, self.ss2d = nn.LayerNorm(dim), SS2D(dim, **ss2d)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(mlp_ratio * dim)), nn.GELU(), nn.Linear(int(mlp_ratio * dim), dim))

    def forward(self, x):
        x = x + self.ss2d(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VMambaForecaster(nn.Module):
    def __init__(self, spec, dim=160, depth=9, patch_size=4, d_state=16, expand=2, mlp_ratio=4.0, skip_channels=32,
                 backend="auto", grad_checkpointing=False):
        super().__init__()
        self.spec, self.patch_size, self.grad_checkpointing = spec, patch_size, grad_checkpointing
        self.backend = resolve_backend(backend)
        self.embed = nn.Conv2d(spec.stacked_channels, dim, patch_size, stride=patch_size)
        self.embed_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([VSSBlock(dim, mlp_ratio, d_state=d_state, expand=expand) for _ in range(depth)])
        for block in self.blocks:
            block.ss2d.backend = self.backend
        self.norm = nn.LayerNorm(dim)
        self.up = nn.Sequential(nn.Conv2d(dim, skip_channels * patch_size ** 2, 1), nn.PixelShuffle(patch_size))
        self.skip = nn.Conv2d(spec.stacked_channels, skip_channels, 3, padding=1)
        self.head = nn.Sequential(nn.Conv2d(2 * skip_channels, skip_channels, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(skip_channels, spec.t_out * spec.k, 1))

    def forward(self, batch):
        x, size = pad_to_multiple(stack_time(batch), self.patch_size)
        tokens = self.embed_norm(self.embed(x).permute(0, 2, 3, 1))  # (B, h, w, dim)
        for block in self.blocks:
            if self.grad_checkpointing and self.training:
                tokens = checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        features = self.up(self.norm(tokens).permute(0, 3, 1, 2))
        y = self.head(torch.cat([features, self.skip(x)], dim=1))
        return unstack_output(crop(y, size), self.spec.t_out, self.spec.k)
