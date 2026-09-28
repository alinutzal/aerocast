"""ConvLSTM models from legacy/grid_forcast.py, plus the adapter to the shared batch interface."""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, norm_type: str = "batch", num_groups: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,  # ✅ dynamically set
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding
        )
        # Normalization on the concatenated gate tensor improves stability
        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=4 * hidden_dim)
        else:
            self.norm = nn.BatchNorm2d(4 * hidden_dim)

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)
        conv_out = self.norm(conv_out)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

class StackedConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_dims=[64, 32], kernel_size=3, out_channels=1, norm="batch", num_groups=8):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.grad_checkpointing = False  # recompute each time step in backward to save memory

        # norm="batch" is the legacy model. In train mode its BatchNorm before the output conv
        # fixes each step's batch-and-grid mean output to the conv bias, so it cannot follow the
        # diurnal change in domain-mean Ox across lead hours; norm="group" has no such coupling.
        def hidden_norm(channels):
            return nn.BatchNorm2d(channels) if norm == "batch" else nn.GroupNorm(num_groups, channels)

        self.cell1 = ConvLSTMCell(input_channels, hidden_dims[0], kernel_size, norm_type=norm, num_groups=num_groups)
        self.bn1 = hidden_norm(hidden_dims[0])

        self.cell2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1], kernel_size, norm_type=norm, num_groups=num_groups)
        self.bn2 = hidden_norm(hidden_dims[1])

        self.out_conv = nn.Conv2d(hidden_dims[1], out_channels, kernel_size=1)

    def _step(self, x_t, h1, c1, h2, c2):
        h1, c1 = self.cell1(x_t, h1, c1)
        h1_bn = self.bn1(h1)
        h2, c2 = self.cell2(h1_bn, h2, c2)
        h2 = self.bn2(h2)
        return h1, c1, h2, c2

    def _run_step(self, x_t, h1, c1, h2, c2):
        if self.grad_checkpointing and self.training:
            return checkpoint(self._step, x_t, h1, c1, h2, c2, use_reentrant=False)
        return self._step(x_t, h1, c1, h2, c2)

    def forward(self, x, pred_steps=1, future_x=None):
        # x: (B, T, C, H, W) - process all T time steps
        # future_x: optional (B, pred_steps - 1, C, H, W) frames fed during the rollout
        #           (features.future_forcings); None reuses the last input frame
        if x.dim() == 5:
            B, T, C, H, W = x.shape
        else:
            # If single time step, add time dimension
            x = x.unsqueeze(1)
            B, T, C, H, W = x.shape

        # Initialize hidden states
        h1 = torch.zeros(B, self.hidden_dims[0], H, W, device=x.device)
        c1 = torch.zeros_like(h1)
        h2 = torch.zeros(B, self.hidden_dims[1], H, W, device=x.device)
        c2 = torch.zeros_like(h2)

        # Process all time steps sequentially through ConvLSTM
        for t in range(T):
            x_t = x[:, t]  # (B, C, H, W)
            h1, c1, h2, c2 = self._run_step(x_t, h1, c1, h2, c2)

        # Generate predictions for multiple future steps
        # Continue evolving hidden states autoregressively
        predictions = []
        x_last = x[:, -1]  # Use last input time step for autoregression
        
        for step in range(pred_steps):
            out = self.out_conv(h2)  # (B, 1, H, W)
            predictions.append(out.squeeze(1))  # (B, H, W)
            
            if step < pred_steps - 1:  # Don't update for last step
                # Continue evolving hidden states using last input (or the next future frame)
                x_next = x_last if future_x is None else future_x[:, step]
                h1, c1, h2, c2 = self._run_step(x_next, h1, c1, h2, c2)
        
        if pred_steps == 1:
            return predictions[0]  # (B, H, W)
        else:
            return torch.stack(predictions, dim=1)  # (B, pred_steps, H, W)


class ConvLSTMForecaster(nn.Module):
    """StackedConvLSTM over hourly frames [state, forcing, static].

    It encodes the T_in input hours, then rolls out one hour at a time. The frame fed before
    predicting hour t+k+1 carries the forcing of hour t+k, with the state held at the last
    input hour (the model never sees future concentrations).
    """

    def __init__(self, spec, hidden_dims=(64, 32), kernel_size=3, norm="batch", num_groups=8, grad_checkpointing=False):
        super().__init__()
        self.spec = spec
        self.net = StackedConvLSTM(spec.frame_channels, list(hidden_dims), kernel_size, out_channels=spec.k,
                                   norm=norm, num_groups=num_groups)
        self.net.grad_checkpointing = grad_checkpointing

    def forward(self, batch):
        state, forcing, static = batch["state"], batch["forcing"], batch["static"].unsqueeze(1)
        t_in = state.shape[1]
        t_out = forcing.shape[1] - t_in
        frames = torch.cat([state, forcing[:, :t_in], static.expand(-1, t_in, -1, -1, -1)], dim=2)
        rollout = torch.cat([state[:, -1:].expand(-1, t_out - 1, -1, -1, -1),
                             forcing[:, t_in:t_in + t_out - 1],
                             static.expand(-1, t_out - 1, -1, -1, -1)], dim=2)
        out = self.net(frames, pred_steps=t_out, future_x=rollout)
        return out.reshape(out.shape[0], t_out, self.spec.k, *out.shape[-2:])
