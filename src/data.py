import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class GridForecastSequenceDataset(Dataset):
    def __init__(self, meteo_path, conc_path, emis_path, vars_X, var_NO2, var_O3, seq_len=3, pred_len=10, t_max=None, normalization_mode="full"):
        self.ds_meteo = xr.open_dataset(meteo_path, engine="netcdf4")
        self.ds_conc = xr.open_dataset(conc_path, engine="netcdf4")
        self.ds_emis = xr.open_dataset(emis_path, engine="netcdf4")

        self.vars_X = vars_X
        self.var_NO2 = var_NO2
        self.var_O3 = var_O3
        self.seq_len = seq_len
        self.pred_len = pred_len  # Number of future steps to predict
        self.normalization_mode = normalization_mode.lower()

        valid_modes = {"none", "predictors", "full"}
        if self.normalization_mode not in valid_modes:
            raise ValueError(
                f"Invalid normalization_mode '{normalization_mode}'. "
                f"Choose from: {sorted(valid_modes)}"
            )

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

        for t in range(self.seq_len, self.T - self.pred_len):
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

            # Generate targets for multiple future steps
            y_seq = []
            for future_t in range(t, t + self.pred_len):
                no2_future = np.squeeze(self.ds_conc[self.var_NO2].isel(TSTEP=future_t).values)
                o3_future = np.squeeze(self.ds_conc[self.var_O3].isel(TSTEP=future_t).values)
                y_future = no2_future + o3_future  # shape: (H, W)
                y_seq.append(y_future)
            Y_seq = np.stack(y_seq, axis=0)  # shape: (pred_len, H, W)

            self.X_all.append(torch.tensor(X_seq, dtype=torch.float32))
            self.Y_all.append(torch.tensor(Y_seq, dtype=torch.float32))

        if self.normalization_mode in {"predictors", "full"}:
            X_stack = torch.stack(self.X_all)  # (N, T, C, H, W)
            # Mean/std per channel across all samples, timesteps, and spatial dims
            self.X_mean = X_stack.mean(dim=(0, 1, 3, 4))  # (C,)
            self.X_std = X_stack.std(dim=(0, 1, 3, 4)) + 1e-6  # (C,)

            # Normalize predictors only
            for i in range(len(self.X_all)):
                # self.X_all[i] shape: (T, C, H, W)
                # Reshape mean/std to (1, C, 1, 1) for broadcasting
                mean = self.X_mean.view(1, -1, 1, 1)
                std = self.X_std.view(1, -1, 1, 1)
                self.X_all[i] = (self.X_all[i] - mean) / std

        if self.normalization_mode == "full":
            Y_stack = torch.stack(self.Y_all)  # (N, pred_len, H, W)
            self.Y_mean = Y_stack.mean()
            self.Y_std = Y_stack.std() + 1e-6

            for i in range(len(self.Y_all)):
                self.Y_all[i] = (self.Y_all[i] - self.Y_mean) / self.Y_std

    def __len__(self):
        return len(self.X_all)

    def __getitem__(self, idx):
        return self.X_all[idx], self.Y_all[idx]

    def denormalize_y(self, y):
        """Map targets/predictions back to physical units (only needed in full mode)."""
        if self.normalization_mode == "full":
            return y * self.Y_std + self.Y_mean
        return y


def build_dataset(data_cfg):
    return GridForecastSequenceDataset(
        meteo_path=data_cfg["meteo_path"],
        conc_path=data_cfg["conc_path"],
        emis_path=data_cfg["emis_path"],
        vars_X=data_cfg["predictor_vars"],
        var_NO2=data_cfg["target_no2"],
        var_O3=data_cfg["target_o3"],
        seq_len=data_cfg["seq_len"],
        pred_len=data_cfg["pred_len"],
        t_max=data_cfg.get("t_max"),
        normalization_mode=data_cfg["normalization_mode"],
    )


def build_dataloader(dataset, data_cfg):
    return DataLoader(dataset, batch_size=data_cfg["batch_size"], shuffle=data_cfg.get("shuffle", True))
