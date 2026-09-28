"""ConvLSTM models, moved unchanged from legacy/grid_forcast.py."""
import torch
import torch.nn as nn


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
    def __init__(self, input_channels, hidden_dims=[64, 32], kernel_size=3):
        super().__init__()
        self.hidden_dims = hidden_dims

        self.cell1 = ConvLSTMCell(input_channels, hidden_dims[0], kernel_size)
        self.bn1 = nn.BatchNorm2d(hidden_dims[0])

        self.cell2 = ConvLSTMCell(hidden_dims[0], hidden_dims[1], kernel_size)
        self.bn2 = nn.BatchNorm2d(hidden_dims[1])

        self.out_conv = nn.Conv2d(hidden_dims[1], 1, kernel_size=1)

    def forward(self, x, pred_steps=1):
        # x: (B, T, C, H, W) - process all T time steps
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
            
            h1, c1 = self.cell1(x_t, h1, c1)
            h1_bn = self.bn1(h1)
            
            h2, c2 = self.cell2(h1_bn, h2, c2)
            h2 = self.bn2(h2)

        # Generate predictions for multiple future steps
        # Continue evolving hidden states autoregressively
        predictions = []
        x_last = x[:, -1]  # Use last input time step for autoregression
        
        for step in range(pred_steps):
            out = self.out_conv(h2)  # (B, 1, H, W)
            predictions.append(out.squeeze(1))  # (B, H, W)
            
            if step < pred_steps - 1:  # Don't update for last step
                # Continue evolving hidden states using last input
                h1, c1 = self.cell1(x_last, h1, c1)
                h1_bn = self.bn1(h1)
                h2, c2 = self.cell2(h1_bn, h2, c2)
                h2 = self.bn2(h2)
        
        if pred_steps == 1:
            return predictions[0]  # (B, H, W)
        else:
            return torch.stack(predictions, dim=1)  # (B, pred_steps, H, W)
