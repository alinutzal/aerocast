import xarray as xr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset

class GridForecastSequenceDataset(Dataset):
    def __init__(self, meteo_path, conc_path, emis_path, vars_X, var_NO2, var_O3, seq_len=3, t_max=None):
        self.ds_meteo = xr.open_dataset(meteo_path, engine="netcdf4")
        self.ds_conc = xr.open_dataset(conc_path, engine="netcdf4")
        self.ds_emis = xr.open_dataset(emis_path, engine="netcdf4")

        self.vars_X = vars_X
        self.var_NO2 = var_NO2
        self.var_O3 = var_O3
        self.seq_len = seq_len

        self.X_all = []
        self.Y_all = []
        
        # Stats for normalization
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None

        # Determine common number of time steps
        self.T = min(
            self.ds_meteo.sizes['TSTEP'],
            self.ds_conc.sizes['TSTEP'],
            self.ds_emis.sizes['TSTEP']
        )
        if t_max:
            self.T = min(self.T, t_max)

        for t in range(self.seq_len, self.T - 1):
            x_seq = []
            for dt in range(t - self.seq_len, t):
                x_vars = []
                for v in vars_X:
                    if v in self.ds_meteo:
                        val = self.ds_meteo[v].isel(TSTEP=dt).values
                    elif v in self.ds_conc:
                        val = self.ds_conc[v].isel(TSTEP=dt).values
                    elif v in self.ds_emis:
                        val = self.ds_emis[v].isel(TSTEP=dt).values
                    else:
                        raise ValueError(f"Variable {v} not found in any dataset.")
                    # Squeeze out singleton dimensions (e.g., LAY=1)
                    val = np.squeeze(val)
                    # Ensure 2D (H, W)
                    if val.ndim != 2:
                        raise ValueError(f"Variable {v} has unexpected shape {val.shape}")
                    x_vars.append(val)
                x_seq.append(np.stack(x_vars, axis=0))  # shape: (C, H, W)

            X_seq = np.stack(x_seq, axis=0)  # shape: (T, C, H, W)

            no2_next = np.squeeze(self.ds_conc[self.var_NO2].isel(TSTEP=t).values)
            o3_next = np.squeeze(self.ds_conc[self.var_O3].isel(TSTEP=t).values)
            y_tensor = no2_next + o3_next  # shape: (H, W)

            self.X_all.append(torch.tensor(X_seq, dtype=torch.float32))
            self.Y_all.append(torch.tensor(y_tensor, dtype=torch.float32))
        
        # Compute normalization stats per channel
        X_stack = torch.stack(self.X_all)  # (N, T, C, H, W)
        Y_stack = torch.stack(self.Y_all)  # (N, H, W)
        
        # Mean/std per channel across all samples, timesteps, and spatial dims
        self.X_mean = X_stack.mean(dim=(0, 1, 3, 4))  # (C,)
        self.X_std = X_stack.std(dim=(0, 1, 3, 4)) + 1e-6  # (C,)
        self.Y_mean = Y_stack.mean()
        self.Y_std = Y_stack.std() + 1e-6
        
        # Normalize - broadcast over T, H, W dimensions
        for i in range(len(self.X_all)):
            # self.X_all[i] shape: (T, C, H, W)
            # Reshape mean/std to (1, C, 1, 1) for broadcasting
            mean = self.X_mean.view(1, -1, 1, 1)
            std = self.X_std.view(1, -1, 1, 1)
            self.X_all[i] = (self.X_all[i] - mean) / std
            self.Y_all[i] = (self.Y_all[i] - self.Y_mean) / self.Y_std

    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        return self.X_all[idx], self.Y_all[idx]



# Paths to data files
meteo_path = "datasets/METCRO2D_20181113.nc"
conc_path = "datasets/out.combine_20181113.nc"
# Emissions file resides at repo root, not under datasets/
emis_path = "egts_l.20181113.1.1km.baaqmd2018_newngc2.ncf"

# Define predictor variables and targets
predictor_vars = ['TEMP2', 'WSPD10', 'WDIR10', 'NO', 'NO2', 'PM25_CL', 'ALK1', 'OLE1', 'ARO1', 'ARO2', 'TERP', 'ISOP']
target_no2 = 'NO2'
target_o3 = 'O3'

# Create dataset
dataset = GridForecastSequenceDataset(
    meteo_path=meteo_path,
    conc_path=conc_path,
    emis_path=emis_path,
    vars_X=['TEMP2', 'WSPD10', 'WDIR10', 'NO', 'NO2', 'PM25_CL', 'ALK1', 'OLE1', 'ARO1', 'ARO2', 'TERP', 'ISOP'],
    var_NO2='NO2',
    var_O3='O3',
    seq_len=3,  # You can set 6, 12, etc.
    t_max=50    # Optional: limit for debugging
)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)





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

    def forward(self, x):
        # x: (B, T, C, H, W)
        if x.dim() == 5:
            x_t = x[:, -1]  # (B, C, H, W)
        else:
            x_t = x  # already (B, C, H, W)

        B, C, H, W = x_t.shape

        h1 = torch.zeros(B, self.hidden_dims[0], H, W, device=x.device)
        c1 = torch.zeros_like(h1)

        h1, c1 = self.cell1(x_t, h1, c1)
        h1 = self.bn1(h1)

        h2 = torch.zeros(B, self.hidden_dims[1], H, W, device=x.device)
        c2 = torch.zeros_like(h2)

        h2, c2 = self.cell2(h1, h2, c2)
        h2 = self.bn2(h2)

        out = self.out_conv(h2)  # (B, 1, H, W)
        return out.squeeze(1)    # (B, H, W)

class ConvLSTMForecast(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3, norm_type: str = "batch", num_groups: int = 8, dropout_p: float = 0.0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size, norm_type=norm_type, num_groups=num_groups)
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p and dropout_p > 0 else nn.Identity()
        self.conv_out = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        # Use only the last time step t
        x_t = x[:, -1]  # shape: (B, C, H, W)

        h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        c = torch.zeros_like(h)

        h, c = self.cell(x_t, h, c)  # Only 1 forward step
        h = self.dropout(h)
        out = self.conv_out(h)      # (B, 1, H, W)
        return out.squeeze(1)       # (B, H, W)




print(predictor_vars)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Normalization and dropout configuration
NORM_TYPE = "group"   # "group" | "batch"
NUM_GROUPS = 8        # used when NORM_TYPE == "group"
DROPOUT_P = 0.1       # dropout after ConvLSTM cell; set 0.0 to disable

model = ConvLSTMForecast(
    input_channels=len(predictor_vars),
    hidden_channels=64,
    norm_type=NORM_TYPE,
    num_groups=NUM_GROUPS,
    dropout_p=DROPOUT_P,
).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Loss configuration
LOSS_TYPE = "mixed"  # "huber" or "mixed"
HUBER_BETA = 0.5     # 0.5 => more MAE-like, 1.0 => default SmoothL1
ALPHA = 0.7          # weight for Huber in mixed loss (Huber*ALPHA + MSE*(1-ALPHA))

huber_loss = nn.SmoothL1Loss(beta=HUBER_BETA)
mse_loss = nn.MSELoss()

def compute_loss(pred, target):
    if LOSS_TYPE == "mixed":
        return ALPHA * huber_loss(pred, target) + (1 - ALPHA) * mse_loss(pred, target)
    else:
        return huber_loss(pred, target)

LOSS_LABEL = (
    f"Mixed(Huber(beta={HUBER_BETA}), alpha={ALPHA}, + MSE)" if LOSS_TYPE == "mixed"
    else f"Huber(beta={HUBER_BETA})"
)

# Dataloader (you already created it)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"\nTraining on {device}")
print(f"Dataset size: {len(dataset)} samples")
print(f"Data normalized: X_mean={dataset.X_mean.mean():.4f}, Y_mean={dataset.Y_mean:.4f}\n")

# Training
for epoch in range(1000):
    model.train()
    total_loss = 0.0
    sse_sum = 0.0  # sum of squared errors across all elements
    n_elem = 0     # total number of elements (B*H*W per batch)

    for X_batch, Y_batch in dataloader:
        X_batch = X_batch.to(device)  # (B, T, C, H, W) - sequence data
        Y_batch = Y_batch.to(device)  # (B, H, W)

        pred = model(X_batch)         # (B, H, W)
        loss = compute_loss(pred, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        # accumulate SSE and count for RMSE
        sse_sum += (pred - Y_batch).pow(2).sum().item()
        n_elem += Y_batch.numel()

    avg_loss = total_loss / len(dataloader)
    rmse_epoch = (sse_sum / max(1, n_elem)) ** 0.5
    scheduler.step(avg_loss)
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1:03d} - {LOSS_LABEL}: {avg_loss:.4f} | RMSE: {rmse_epoch:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

# Visualize a sample after training
import random
sample_idx = random.randint(0, len(dataset) - 1)
X_sample, Y_sample = dataset[sample_idx]

print(f"\n[Visualization] Sample {sample_idx}")
print(f"  X_shape: {X_sample.shape}, Y_shape: {Y_sample.shape}")

# Generate prediction for the sample
model.eval()
with torch.no_grad():
    X_sample_batch = X_sample.unsqueeze(0).to(device)  # Add batch dim
    pred_sample = model(X_sample_batch).cpu().squeeze()
    
    # Denormalize for visualization
    Y_sample_denorm = Y_sample * dataset.Y_std + dataset.Y_mean
    pred_sample_denorm = pred_sample * dataset.Y_std + dataset.Y_mean
    # Compute RMSE on the sample (denormalized space)
    sample_rmse = ((pred_sample_denorm - Y_sample_denorm) ** 2).mean().sqrt().item()
    print(f"Sample RMSE (t+1): {sample_rmse:.3f}")

# Plot target vs prediction
plt.figure(figsize=(12, 4))

# Use same color scale for both plots
vmin = min(Y_sample_denorm.min().item(), pred_sample_denorm.min().item())
vmax = max(Y_sample_denorm.max().item(), pred_sample_denorm.max().item())

plt.subplot(1, 2, 1)
im1 = plt.imshow(Y_sample_denorm.squeeze(), cmap='viridis', vmin=vmin, vmax=vmax)
plt.title("Target: NO2 + O3 at t+1")
plt.colorbar(im1)

plt.subplot(1, 2, 2)
im2 = plt.imshow(pred_sample_denorm, cmap='viridis', vmin=vmin, vmax=vmax)
plt.title("Pred: NO2 + O3 at t+1")
plt.colorbar(im2)

plt.tight_layout()
output_path = "results/grid_forecast_sample.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {output_path}")
