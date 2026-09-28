"""Encode-process-decode graph network (PyTorch Geometric) on the time-stacked input.

Grid cells are nodes whose features are all input hours stacked. An encoder passes them to
a coarser mesh (one node per 4x4 block, linked to the cells of its bilinear stencil), a
processor runs interaction-network steps on a multi-scale mesh (neighbours 4 cells apart and,
on every 4th mesh node, 16 cells apart, so each step can move information ~16-23 km), and a
decoder passes the result back to the cells. Every edge carries its offset and length plus
the wind component along the edge for each input and forecast hour.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import MessagePassing

from aerocast.models.common import stack_time, unstack_output


def mlp(in_dim, hidden, out_dim, norm=True):
    layers = [nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers, nn.LayerNorm(out_dim)) if norm else nn.Sequential(*layers)


class InteractionNetwork(MessagePassing):
    """Update each edge from (sender, receiver, edge), then each receiver from its mean incoming
    edge; both with residuals. Features may carry a leading batch dimension."""

    def __init__(self, latent, aggr="mean"):
        super().__init__(aggr=aggr, node_dim=-2)
        self.edge_mlp = mlp(3 * latent, latent, latent)
        self.node_mlp = mlp(2 * latent, latent, latent)

    def forward(self, x_src, x_dst, edge_index, edge_attr):
        aggregated = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr,
                                    size=(x_src.size(-2), x_dst.size(-2)))
        edges = edge_attr + self._edges
        return x_dst + self.node_mlp(torch.cat([x_dst, aggregated], dim=-1)), edges

    def message(self, x_i, x_j, edge_attr):
        self._edges = self.edge_mlp(torch.cat([x_j, x_i, edge_attr], dim=-1))
        return self._edges


def build_graph(height, width, stride=4, coarse_stride=16):
    """Mesh nodes at the centres of stride x stride blocks; edge index arrays and positions (row, col)."""
    rows, cols = math.ceil(height / stride), math.ceil(width / stride)
    a, b = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    mesh_pos = np.stack([a.ravel() * stride + (stride - 1) / 2, b.ravel() * stride + (stride - 1) / 2], axis=1)
    node = lambda i, j: i * cols + j  # noqa: E731

    mesh_edges = set()
    for step in (1, coarse_stride // stride):  # fine mesh, then every step-th mesh node
        for i in range(0, rows, step):
            for j in range(0, cols, step):
                for di in (-step, 0, step):
                    for dj in (-step, 0, step):
                        if (di or dj) and 0 <= i + di < rows and 0 <= j + dj < cols:
                            mesh_edges.add((node(i, j), node(i + di, j + dj)))
    mesh_edges = np.array(sorted(mesh_edges)).T if mesh_edges else np.zeros((2, 0), dtype=np.int64)

    r, c = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    r, c = r.ravel(), c.ravel()
    i0 = np.clip(np.floor((r - (stride - 1) / 2) / stride), 0, rows - 1).astype(np.int64)
    j0 = np.clip(np.floor((c - (stride - 1) / 2) / stride), 0, cols - 1).astype(np.int64)
    pairs = set()
    for di in (0, 1):
        for dj in (0, 1):
            mesh = node(np.clip(i0 + di, 0, rows - 1), np.clip(j0 + dj, 0, cols - 1))
            pairs.update(zip((r * width + c).tolist(), mesh.tolist()))
    g2m = np.array(sorted(pairs)).T  # (grid cell, mesh node)
    grid_pos = np.stack([r, c], axis=1).astype(np.float64)
    coarse = ((a.ravel() % (coarse_stride // stride) == 0) & (b.ravel() % (coarse_stride // stride) == 0)).astype(np.float64)
    return {"mesh_pos": mesh_pos, "grid_pos": grid_pos, "mesh_edges": mesh_edges, "g2m": g2m,
            "mesh_static": np.stack([mesh_pos[:, 1] / max(width - 1, 1), mesh_pos[:, 0] / max(height - 1, 1), coarse], axis=1),
            "mesh_shape": (rows, cols)}


def edge_geometry(src_pos, dst_pos, scale):
    """(E, 3) offsets (d_col, d_row) and length, divided by scale, and (E, 2) unit vectors (x=col, y=row)."""
    delta = dst_pos - src_pos
    length = np.linalg.norm(delta, axis=1, keepdims=True)
    unit = np.divide(delta[:, ::-1], length, out=np.zeros_like(delta), where=length > 0)  # (dx, dy)
    return np.concatenate([delta[:, ::-1], length], axis=1) / scale, unit


class MeshGNN(nn.Module):
    def __init__(self, spec, latent=224, processor_steps=11, stride=4, coarse_stride=16, grad_checkpointing=False):
        super().__init__()
        if "meteo:U10" not in spec.forcing_channels or "meteo:V10" not in spec.forcing_channels:
            raise ValueError("gnn: edge winds need meteo:U10 and meteo:V10 among the forcing channels")
        self.spec, self.stride, self.coarse_stride = spec, stride, coarse_stride
        self.grad_checkpointing = grad_checkpointing
        self.wind = (spec.forcing_channels.index("meteo:U10"), spec.forcing_channels.index("meteo:V10"))
        hours = spec.t_in + spec.t_out
        self.grid_encoder = mlp(spec.stacked_channels, latent, latent)
        self.mesh_encoder = mlp(3, latent, latent)
        self.edge_encoders = nn.ModuleDict({name: mlp(3 + hours, latent, latent) for name in ("g2m", "mesh", "m2g")})
        self.g2m = InteractionNetwork(latent)
        self.processor = nn.ModuleList([InteractionNetwork(latent) for _ in range(processor_steps)])
        self.m2g = InteractionNetwork(latent)
        self.decoder = mlp(latent, latent, spec.t_out * spec.k, norm=False)
        self._graphs = {}

    def graph(self, height, width, device):
        key = (height, width, str(device))
        if key not in self._graphs:
            g = build_graph(height, width, self.stride, self.coarse_stride)
            mesh_src, mesh_dst = g["mesh_edges"]
            grid, mesh = g["g2m"]
            geometry = {
                "g2m": edge_geometry(g["grid_pos"][grid], g["mesh_pos"][mesh], self.stride),
                "mesh": edge_geometry(g["mesh_pos"][mesh_src], g["mesh_pos"][mesh_dst], self.coarse_stride),
                "m2g": edge_geometry(g["mesh_pos"][mesh], g["grid_pos"][grid], self.stride),
            }
            tensor = lambda a, dtype=torch.float32: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)  # noqa: E731
            self._graphs[key] = {
                "g2m": tensor(np.stack([grid, mesh]), torch.long), "m2g": tensor(np.stack([mesh, grid]), torch.long),
                "mesh": tensor(g["mesh_edges"], torch.long), "mesh_static": tensor(g["mesh_static"]),
                "geometry": {k: (tensor(v[0]), tensor(v[1])) for k, v in geometry.items()},
                "mesh_shape": g["mesh_shape"],
            }
        return self._graphs[key]

    def edge_features(self, name, graph, grid_wind, mesh_wind):
        """Encoded edge features (B, E, latent): geometry and the along-edge wind for every hour."""
        geometry, unit = graph["geometry"][name]
        src, dst = graph[name]
        if name == "g2m":
            wind = grid_wind[..., src]
        elif name == "m2g":
            wind = grid_wind[..., dst]
        else:
            wind = (mesh_wind[..., src] + mesh_wind[..., dst]) / 2
        along = (wind * unit.T[None, :, None, :]).sum(dim=1).transpose(1, 2)  # (B, E, hours)
        features = torch.cat([geometry.expand(along.shape[0], -1, -1), along], dim=-1)
        return self.edge_encoders[name](features)

    def _step(self, layer, *args):
        if self.grad_checkpointing and self.training:
            return checkpoint(layer, *args, use_reentrant=False)
        return layer(*args)

    def forward(self, batch):
        x = stack_time(batch)
        b, _, height, width = x.shape
        graph = self.graph(height, width, x.device)
        forcing = batch["forcing"]
        grid_wind = torch.stack([forcing[:, :, self.wind[0]], forcing[:, :, self.wind[1]]], dim=1)  # (B, 2, hours, H, W)
        pooled = F.avg_pool2d(grid_wind.flatten(1, 2), self.stride, ceil_mode=True)                # block means
        mesh_wind = pooled.reshape(b, 2, -1, *graph["mesh_shape"]).flatten(-2)
        grid_wind = grid_wind.flatten(-2)                                                          # (B, 2, hours, N)

        grid = self.grid_encoder(x.flatten(2).transpose(1, 2))                                     # (B, N_grid, latent)
        mesh = self.mesh_encoder(graph["mesh_static"]).expand(b, -1, -1)
        edges = {name: self.edge_features(name, graph, grid_wind, mesh_wind) for name in ("g2m", "mesh", "m2g")}
        mesh, _ = self._step(self.g2m, grid, mesh, graph["g2m"], edges["g2m"])
        mesh_edges = edges["mesh"]
        for layer in self.processor:
            mesh, mesh_edges = self._step(layer, mesh, mesh, graph["mesh"], mesh_edges)
        grid, _ = self._step(self.m2g, mesh, grid, graph["m2g"], edges["m2g"])
        out = self.decoder(grid).transpose(1, 2).reshape(b, -1, height, width)
        return unstack_output(out, self.spec.t_out, self.spec.k)
