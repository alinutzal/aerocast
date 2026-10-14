"""Universal Physics Transformer (Alkin et al., 2024, arXiv:2402.12365) on the time-stacked input.

UPT has three parts, and this is a regular-grid version of each:

  encoder      "supernode pooling": a coarser set of supernodes (one per `stride` cells) each
               gathers the cells in its (2 * radius + 1)^2 neighbourhood. A message is
               MLP(cell embedding + embedding of the cell's offset from the supernode), and a
               supernode averages the messages it receives. Transformer blocks then mix the
               supernode tokens. Because the tokens are defined by position, not by array
               index, the model runs on grids of any size.
  approximator the latent state is advanced one forecast hour at a time by transformer blocks
               (the UPT "latent rollout"); the hour is injected with a learned lead embedding.
               Nothing is decoded between steps, so the cost of a rollout does not grow with
               the number of grid cells.
  decoder      every grid cell is a query: a Fourier embedding of its coordinates (plus, if
               `local_skip`, a pointwise embedding of the cell's own last state, that hour's
               forcing and its static channels) cross-attends to the latent tokens of the
               lead being decoded.

`local_skip` is the one departure from the paper: UPT decodes from the latent alone, which on
1-12 km CMAQ fields would have to rebuild fine-scale structure from ~stride^2 fewer tokens
(the smoothing problem already seen in the U-Net). The skip plays the role U-Net's skip
connections play. Set it to false for the plain UPT decoder.

enforce_positive / target_floor behave exactly as in fno.py.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from aerocast.models.common import stack_time


class FourierPosition(nn.Module):
    """(..., 2) coordinates in [0, 1] -> (..., dim): sin/cos at octave frequencies, then a linear map."""

    def __init__(self, dim, n_freq=8):
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freq), persistent=False)
        self.proj = nn.Linear(2 + 4 * n_freq, dim)

    def forward(self, coords):
        angles = coords.unsqueeze(-1) * self.freqs                      # (..., 2, F)
        feats = torch.cat([coords, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)
        return self.proj(feats)


class Attention(nn.Module):
    """Multi-head attention; self-attention when `context` is omitted, cross-attention otherwise."""

    def __init__(self, dim, num_heads):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"upt: dim {dim} is not divisible by num_heads {num_heads}")
        self.heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, context=None):
        context = x if context is None else context
        split = lambda t: t.reshape(t.shape[0], t.shape[1], self.heads, -1).transpose(1, 2)  # noqa: E731
        k, v = self.kv(context).chunk(2, dim=-1)
        y = F.scaled_dot_product_attention(split(self.q(x)), split(k), split(v))
        return self.out(y.transpose(1, 2).flatten(2))


class Mlp(nn.Sequential):
    def __init__(self, dim, hidden):
        super().__init__(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))


class Block(nn.Module):
    """Pre-norm transformer block: self-attention, or cross-attention to `context`."""

    def __init__(self, dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x, context=None):
        x = x + self.attn(self.norm1(x), context)
        return x + self.mlp(self.norm2(x))


class SupernodePooling(nn.Module):
    """(B, C, H, W) cells -> (B, L, dim) supernode tokens on a stride-spaced lattice.

    Supernode (i, j) sits at cell (i * stride, j * stride) and gathers the cells within `radius`
    of it; cells outside the grid are masked out of the mean, not zero-padded into it."""

    def __init__(self, in_channels, dim, stride, radius):
        super().__init__()
        self.stride, self.radius, self.k = stride, radius, 2 * radius + 1
        self.cell = nn.Linear(in_channels, dim)
        self.offset = nn.Linear(2, dim)
        self.message = nn.Sequential(nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        axis = torch.arange(-radius, radius + 1, dtype=torch.float32) / max(radius, 1)
        self.register_buffer("offsets", torch.stack(torch.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2),
                             persistent=False)                                 # (k*k, 2), (d_row, d_col)

    def forward(self, x):
        b, _, height, width = x.shape
        cells = self.cell(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)           # (B, dim, H, W)
        args = dict(kernel_size=self.k, stride=self.stride, padding=self.radius)
        patches = F.unfold(cells, **args)                                       # (B, dim*k*k, L)
        valid = F.unfold(torch.ones(1, 1, height, width, device=x.device, dtype=x.dtype), **args)  # (1, k*k, L)
        dim = cells.shape[1]
        patches = patches.reshape(b, dim, self.k ** 2, -1).permute(0, 3, 2, 1)  # (B, L, k*k, dim)
        messages = self.message(patches + self.offset(self.offsets))
        valid = valid.permute(0, 2, 1).unsqueeze(-1)                            # (1, L, k*k, 1)
        tokens = (messages * valid).sum(dim=2) / valid.sum(dim=2)
        return self.norm(tokens)


class UPTForecaster(nn.Module):
    def __init__(self, spec, dim=224, depth=3, dec_depth=2, num_heads=4, mlp_ratio=4.0, stride=4, radius=2,
                 local_skip=True, grad_checkpointing=False, enforce_positive=False, target_floor=None):
        super().__init__()
        self.spec, self.stride, self.local_skip = spec, stride, local_skip
        self.grad_checkpointing = grad_checkpointing
        self.enforce_positive = enforce_positive
        if enforce_positive:
            floor = torch.zeros(spec.k) if target_floor is None else torch.as_tensor(target_floor, dtype=torch.float32)
            if floor.shape != (spec.k,):
                raise ValueError(f"target_floor must have {spec.k} values, got {floor.shape}")
            self.register_buffer("positivity_floor", floor)
        self.pool = SupernodePooling(spec.stacked_channels, dim, stride, radius)
        self.token_position = FourierPosition(dim)
        self.encoder = nn.ModuleList([Block(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.lead_embedding = nn.Parameter(torch.zeros(spec.t_out, dim))
        nn.init.normal_(self.lead_embedding, std=0.02)
        self.approximator = nn.ModuleList([Block(dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.cell_position = FourierPosition(dim)
        if local_skip:
            local = len(spec.state_channels) + len(spec.forcing_channels) + len(spec.static_channels)
            self.local = nn.Sequential(nn.Linear(local, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query_norm = nn.LayerNorm(dim)
        self.decoder = nn.ModuleList([Block(dim, num_heads, mlp_ratio) for _ in range(dec_depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, spec.k))

    @staticmethod
    def coordinates(rows, cols, step, height, width, device):
        """(rows*cols, 2) positions in [0, 1] of cells row = i*step, col = j*step."""
        r = torch.arange(rows, device=device, dtype=torch.float32) * step / max(height - 1, 1)
        c = torch.arange(cols, device=device, dtype=torch.float32) * step / max(width - 1, 1)
        return torch.stack(torch.meshgrid(r, c, indexing="ij"), -1).reshape(-1, 2)

    def _run(self, layer, *args):
        if self.grad_checkpointing and self.training:
            return checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)

    def _decode(self, latent, query, local):
        q = query if local is None else query + self.local(local)
        h = self.query_norm(q)
        for block in self.decoder:
            h = block(h, latent)
        return self.head(h)                                                    # (B, N, K)

    def forward(self, batch):
        x = stack_time(batch)
        b, _, height, width = x.shape
        rows, cols = math.ceil(height / self.stride), math.ceil(width / self.stride)
        tokens = self.pool(x) + self.token_position(self.coordinates(rows, cols, self.stride, height, width, x.device))
        for block in self.encoder:
            tokens = self._run(block, tokens)

        query = self.cell_position(self.coordinates(height, width, 1, height, width, x.device)).unsqueeze(0)
        t_in = self.spec.t_in
        static = batch["static"].flatten(2).transpose(1, 2)                    # (B, N, C_static)
        last = batch["state"][:, -1].flatten(2).transpose(1, 2)                # (B, N, C_state)
        outputs, latent = [], tokens
        for lead in range(self.spec.t_out):
            latent = latent + self.lead_embedding[lead]
            for block in self.approximator:
                latent = self._run(block, latent)
            local = None
            if self.local_skip:
                forcing = batch["forcing"][:, t_in + lead].flatten(2).transpose(1, 2)
                local = torch.cat([last, forcing, static], dim=-1)
            out = self._run(self._decode, latent, query, local)
            outputs.append(out.transpose(1, 2).reshape(b, self.spec.k, height, width))
        y = torch.stack(outputs, dim=1)                                        # (B, T_out, K, H, W)
        if self.enforce_positive:
            floor = self.positivity_floor.view(1, 1, -1, 1, 1)
            y = F.softplus(y - floor) + floor
        return y
