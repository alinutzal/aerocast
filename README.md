# AeroCast

Deep learning-based air quality forecasting using ConvLSTM neural networks for spatio-temporal prediction of NO2 and O3 concentrations.

## Overview

AeroCast is a PyTorch-based framework for forecasting air quality using Convolutional LSTM (ConvLSTM) models. The system processes meteorological data, concentration measurements, and emissions data to predict future NO2 + O3 concentrations across spatial grids.

### Key Features

- **Multi-step temporal prediction**: Predicts 10 future time steps from 6 past observations
- **Stacked ConvLSTM architecture**: 2-layer deep network (64→32 hidden channels)
- **Spatial grid forecasting**: Handles 224×164 spatial grids with 12 input features
- **Advanced training techniques**:
  - Mixed loss function (Huber + MSE)
  - Cosine annealing with warm restarts
  - Gradient clipping and weight decay
  - Early stopping with patience
  - GroupNorm and BatchNorm layers

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management:

```bash
# Install dependencies
uv sync

# Or run scripts directly
uv run ./src/grid_forcast.py
```

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- NetCDF4 datasets

## Data Format

The model expects three NetCDF files:

1. **Meteorological data** (`METCRO2D_*.nc`):
   - Variables: TEMP2, WSPD10, WDIR10
   
2. **Concentration data** (`out.combine_*.nc`):
   - Variables: NO, NO2, PM25_CL, ALK1, OLE1, ARO1, ARO2, TERP, ISOP, O3
   
3. **Emissions data** (`egts_l.*.ncf`):
   - Additional emission variables

All datasets should have dimensions: `(TSTEP, LAY, ROW, COL)` where LAY=1.

## Model Architecture

### StackedConvLSTM

```
Input: (Batch, 6 timesteps, 12 channels, 224, 164)
  ↓
ConvLSTMCell Layer 1 (12 → 64 channels)
  ↓
BatchNorm2d
  ↓
ConvLSTMCell Layer 2 (64 → 32 channels)
  ↓
BatchNorm2d
  ↓
Conv2d (32 → 1 channel)
  ↓
Output: (Batch, 10 timesteps, 224, 164)
```

### Training Configuration

- **Input sequence length**: 6 time steps
- **Prediction horizon**: 10 time steps
- **Batch size**: 4
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-5)
- **Loss function**: 0.7×Huber(β=0.5) + 0.3×MSE
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=50, T_mult=2)
- **Early stopping**: Patience of 30 epochs

## Usage

### Training

```bash
uv run ./src/grid_forcast.py
```

The script will:
1. Load NetCDF datasets from `datasets/` directory
2. Create sequences with 6 input and 10 output time steps
3. Train the StackedConvLSTM model
4. Generate visualizations for all samples

### Output

Training produces:

- **Console output**: Epoch-wise loss, RMSE, and learning rate
- **Visualization**: `results/grid_forecast_all_samples.png`
  - 8 rows (one per sample)
  - 10 columns (one per prediction step)
  - Side-by-side target vs prediction comparison

### Example Results

```
Best Loss: 0.0886 | RMSE: 0.31 (normalized)

Sample RMSE (denormalized):
  Sample 0: 4.79
  Sample 1: 4.52
  Sample 2: 4.31
  Sample 3: 4.27
  Sample 4: 4.33
  Sample 5: 4.15
  Sample 6: 3.92
  Sample 7: 3.70
```

## Project Structure

```
AeroCast/
├── src/
│   ├── grid_forcast.py      # Main training script with model
│   └── read_data.py          # Data loading utilities
├── datasets/
│   ├── METCRO2D_20181113.nc
│   ├── out.combine_20181113.nc
│   └── egts_l.20181113.1.1km.baaqmd2018_newngc2.ncf
├── results/
│   └── grid_forecast_all_samples.png
├── pyproject.toml
├── .gitignore
└── README.md
```

## Model Details

### Input Variables (12 channels)

- **Meteorology**: TEMP2, WSPD10, WDIR10
- **Concentrations**: NO, NO2, PM25_CL
- **Emissions**: ALK1, OLE1, ARO1, ARO2, TERP, ISOP

### Target Variable

Combined NO2 + O3 concentration at future time steps

### Normalization

- **Input (X)**: Per-channel normalization across all samples
- **Output (Y)**: Global normalization across all samples

## Performance

The model achieves:
- ~37% improvement over single-layer baseline
- Average RMSE of 4.15 across all samples
- Efficient training with early stopping (~170 epochs)

## Dependencies

See `pyproject.toml` for full dependency list. Key packages:

- PyTorch 2.0+
- xarray 2024.3+
- numpy 1.24+
- netCDF4 1.6+
- matplotlib 3.7+

## License

[Add your license information]

## Citation

If you use this code in your research, please cite:

```
[Add citation information]
```

## Contact

[Add contact information]