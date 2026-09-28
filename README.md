# AeroCast

Deep-learning forecasts of Ox (NO2 + O3) on the 1 km BAAQMD CMAQ grid (224 x 164): 6 hours of
chemical state plus meteorology and emissions for the input and forecast hours in, 10 hourly
fields out.

- **Phase 1** turned the original script (kept unchanged in `legacy/grid_forcast.py`) into a tested
  pipeline with date-based train/val/test splits and two baselines, persistence and diurnal
  climatology.
- **Phase 2** adds six more models to the ConvLSTM: U-Net, FNO, two Swin variants, a GNN and
  Mamba. It also adds three target modes and a controlled benchmark in which only the model, the
  target mode and the seed change between runs.

## Quick start

```bash
uv sync                                                # installs aerocast (editable) + pytest
uv run pytest                                          # synthetic-data tests, ~2 min on CPU

# One-day smoke test on the real data (GPU), any model from configs/models/
uv run aerocast-train --config configs/smoke.yaml --model unet
uv run aerocast-train --config configs/smoke.yaml --model fno --set target_mode=species

# Re-score a saved run
uv run aerocast-evaluate runs/<run_id>

# Controlled benchmark (needs several days of data and split.mode: dates)
uv run python scripts/run_benchmark.py --model unet    # resumable, one model at a time
uv run python scripts/make_tables.py
uv run python scripts/profile_models.py                # GPU cost of every model
```

Run commands from the repo root: data, cache and output paths in the configs are relative to it.
Any config value can be overridden on the command line (`--set train.epochs=2`), and `--note`
adds a note to the run's line in `EXPERIMENTS.md`. `aerocast-train` evaluates the best checkpoint
when training finishes; pass `--no-eval` to skip that step.

On OSC, get a GPU first: `salloc -C gpu -t 2:00:00 -c 4 -A <project> --gpus=1`.

## Layout

```
configs/
  base.yaml              data, inputs, target mode, splits, seed, training, evaluation, outputs
  smoke.yaml             single-day smoke test (inherits base.yaml)
  legacy.yaml            Phase 1 inputs and ConvLSTM, for the refactor check
  models/<name>.yaml     one per model: architecture, AMP, checkpointing, tuning knob
  experiments/benchmark.yaml   models x target modes x seeds, budget, search space
src/aerocast/
  config.py              YAML with `base:` inheritance, --model, --set overrides, config hash
  data.py                load days, align hours, derived channels, cache, windows
  splits.py              contiguous date-block splits; smoke_test and legacy modes
  normalize.py           stats from training windows only, saved as JSON
  targets.py             target modes: output channels, loss, native-unit conversion
  baselines.py           persistence and diurnal climatology
  metrics.py             per-day sums, metrics, high-Ox skill, spectra, block bootstrap
  models/                registry, shared helpers and one file per model
  train.py, evaluate.py  aerocast-train, aerocast-evaluate
scripts/
  make_synthetic_data.py CMAQ-like NetCDF days on a small grid (used by the tests)
  run_benchmark.py       tuning trials, selection, final runs (resumable)
  make_tables.py         Markdown/LaTeX tables and CSV figure data from results.csv
  profile_models.py      parameters, memory, training and inference time per model
tests/                   pytest suite (synthetic data; two tests need a CUDA GPU)
legacy/grid_forcast.py   original script, unchanged
EXPERIMENTS.md           one line per run, written by aerocast-evaluate
results/results.csv      appended by every evaluation
results/profile.csv      model cost on one GPU (CMAQ wall-clock column filled by hand)
datasets/, cache/, runs/ inputs, preprocessed arrays, run directories (gitignored)
```

## Data and inputs

Each day needs three files in `data.data_dir` (`{date}` = `YYYYMMDD`; patterns in `data.files`):
`METCRO2D_{date}.nc`, `out.combine_{date}.nc` and `egts_l.{date}.*.ncf`, with dims
`(TSTEP, LAY, ROW, COL)` (the surface layer is used).

Channels are named `source:VAR`, so the emission and concentration NO/NO2 stay distinct:

| Group | Channels | Hours |
|---|---|---|
| state | `conc:NO`, `conc:NO2`, `conc:PM25_CL`, `conc:O3` (the last via `features.include_o3_input`) | t−6 … t−1 |
| forcing | `meteo:TEMP2`, `meteo:WSPD10`, `meteo:U10`, `meteo:V10`; `emis:NO`, `emis:NO2`, `emis:HONO`, `emis:ALK1`, `OLE1`, `ARO1`, `ARO2`, `TERP`, `ISOP`; `time:HOUR_SIN`, `time:HOUR_COS` | t−6 … t+9 (`features.future_forcings`) |
| static | `grid:X`, `grid:Y` (grid coordinates scaled to [0, 1]) | – |

- **Wind:** `meteo:U10` and `meteo:V10` are derived from WSPD10 and WDIR10 (meteorological
  convention). They are treated as grid-relative; the projection's rotation from true north is
  under 2° on this domain.
- **Hour of day:** in local standard time (`data.local_utc_offset_hours: -8`).
- **Targets:** NO2 and O3 from out.combine; Ox = NO2 + O3, in native units (ppbV), recorded per
  run.
- **Without `future_forcings`:** the forecast hours repeat hour t−1, so no future values are seen.

**Normalization** (`data.normalization`: none | predictors | full) is fitted on training windows
only, using the legacy formula: hours are weighted by how many training windows contain them, and
the std is unbiased plus 1e-6. Val/test use the saved stats unchanged.
- **State and forcing:** per-channel mean and std, over the hours each is fed for.
- **Wind:** u and v share one scale with no mean shift, so wind direction survives.
- **Time and grid channels:** left as they are.
- **Targets:** a scalar mean and std per target (NO2, O3, Ox).

**Time alignment.** Hours are matched on each file's TFLAG plus `data.time_offset_hours`, keeping
only hours present in all three files.
- **METCRO2D and emissions:** stamped 08Z–08Z, which is 00–24 local standard time (PST = UTC−8).
- **out.combine:** stamped 00Z–23Z, but its O3/NO diurnal cycles show it holds 00–23 PST, so
  `conc` gets +8 h.

Each day then contributes 24 aligned hours. Without the offset only 16 hours overlap, and the
loader warns. If a future batch of files is stamped differently, adjust the offsets.

**More days.** Drop the files into `datasets/` and set the split date blocks. Days are
concatenated hourly, windows never bridge a missing day, and each day is read from NetCDF once
(cached under `cache/`).

**Other datasets.** The loader is config-driven, not tied to BAAQMD's file names or three-source
layout: `data.files` lists whichever sources a dataset has (a source with no file is set to
`null`, since config `base:` inheritance merges keys rather than removing them), and
`data.local_solar_time: true` (with `data.static_file` giving a NetCDF with a `LON` variable)
computes hour-of-day per cell instead of from one grid-wide UTC offset, for domains wide enough
in longitude that a single offset would be inaccurate. `configs/data/equates_2019_07_ca12km.yaml`
is the second dataset: a pilot built from EPA's public EQUATES CMAQ run (July 2019, central
California, 12 km, meteorology + concentrations, no emissions - see `DATA_CARD.md`). Built by
`scripts/extract_equates_pilot.py`, which reads the source files by byte range and never
downloads them whole (CONC3D alone is 386 GB). `results.csv` and `EXPERIMENTS.md` have a
`dataset` column/field so runs on different datasets share one results file.

## Splits

`split.mode` in the config:

- **`dates`** (default): `split.train`, `split.val` and `split.test` are contiguous, inclusive date
  blocks that must not overlap. A window (6 input + 10 target hours) is kept only if all 16 hours
  are consecutive and in the same split. Windows crossing a split boundary or a gap are dropped.
- **`smoke_test`**: for a single day, where no clean held-out split exists. Training, early
  stopping and evaluation use every window, with a warning. Rows are labelled `smoke_test`, never
  `test`. Climatology is fitted on the day it is scored on, so it is trivially perfect here.
- **`legacy`**: same windows as `smoke_test`, rows labelled `legacy`, for the refactor check.

Only `dates` mode produces `test` rows, and only `test` rows feed benchmark tables.

## Models

Every model maps a batch `{state (B, 6, 4, H, W), forcing (B, 16, 15, H, W), static (B, 2, H, W)}`
to `(B, 10, K, H, W)`. They are built from the registry (`build_model(cfg)`) and never assume H or W.
The 2D models stack all hours into channels (266 inputs), pad to what they need and crop back.

| Name | Model | Params | Notes |
|---|---|---|---|
| `convlstm` | stacked ConvLSTM, hidden [272, 136] | 4.87M | recurrent; GroupNorm; per-hour recomputation in backward |
| `unet` | plain 4-level 2D U-Net, widths 52–416 | 4.87M | GroupNorm, bilinear upsampling |
| `fno` | neuraloperator FNO, 4 layers, hidden 64, 16x16 modes | 4.80M | 12.5% domain padding; fp32; any grid size |
| `swin` | MONAI SwinUNETR (2D), feature 24 | 5.70M | ~70% of parameters in its conv decoder |
| `swin_unet` | Swin U-Net, transformer stages on both sides | 4.82M | built from MONAI's Swin stages |
| `gnn` | encode-process-decode GNN (PyTorch Geometric) | 4.97M | mesh links 4 and 16 cells apart; along-edge wind per hour |
| `mamba` | VMamba-style 2D selective scan, patch 4, 4 directions | 4.94M | backend `mamba_ssm` (CUDA) or `mambapy` |

- **Parameter budget:** all within 5M ±20%; complex FNO weights count as two parameters.
- **Tuning knob:** each `configs/models/<name>.yaml` has one width/depth knob with three options,
  all within budget (checked by `tests/test_budget.py`).
- **AMP and checkpointing:** bf16 autocast everywhere except the FNO, whose FFTs need fp32.
  Gradient checkpointing is on for the ConvLSTM only. Both settings are logged in `run.json`.
- **ConvLSTM normalization:** the legacy ConvLSTM used BatchNorm. Its BatchNorm right before the
  output conv fixed each lead's batch-mean output in train mode, so it could not fit the diurnal
  change across lead hours. The Phase 2 ConvLSTM uses GroupNorm; `legacy.yaml` keeps BatchNorm.
- **Mamba install:** `mamba-ssm` is pinned to its prebuilt wheel (Linux, CUDA 12, torch 2.9,
  Python 3.12). It is installed without the language-model dependencies it declares; aerocast
  calls only its compiled scan kernel. Elsewhere `mambapy` runs the same scan in PyTorch. The
  backend is recorded per run, and the benchmark uses `mamba_ssm` throughout.

### Target modes (`target_mode`)

- `ox`: one channel, Ox directly.
- `species`: NO2 and O3. Ox = NO2 + O3 is formed after denormalizing.
- `multitask`: NO2, O3 and Ox. The loss is L(NO2) + L(O3) + L(Ox) + λ · mean((Ox − (NO2 + O3))²).
  The last term uses physical units divided by the training Ox variance
  (`train.loss.consistency_weight` = λ = 1). Ox is scored from the Ox head, and the species sum is
  reported as `oxsum_*`.

## Training

- **Reproducibility:** fixed seed, deterministic cuDNN, seeded shuffling.
- **Early stopping:** early stopping and the saved checkpoint both use the validation loss.
- **Optimization:** Adam, CosineAnnealingWarmRestarts (T_0 50, T_mult 2), and per-channel
  0.7·Huber(β 0.5) + 0.3·MSE with grad-norm clip 1.0 and batch 4.
- **Budget:** the benchmark allows 150 epochs with patience 20; the smoke and base configs keep
  the legacy 500 and 30.

Each run writes `runs/<run_id>/` containing:
- `config.yaml` (resolved, including the days used);
- `run.json` (git commit, config hash, model, parameters, AMP, checkpointing, Mamba backend, GPU,
  peak memory, seconds per epoch, windows, target unit);
- `norm_stats.json`, `best.pt` (tensors only) and `history.csv`;
- the evaluation files below.

## Evaluation

`aerocast-evaluate` scores the model, persistence (hour t−1 held) and climatology (per-cell,
per-hour-of-day mean over training days) on exactly the same windows, in native units:

- **Core metrics:** RMSE, MAE, mean bias and Pearson r, per lead hour and overall.
- **Other variables:** NO2 and O3 metrics (`no2_*`, `o3_*`) in species and multitask modes, for
  the baselines too.
- **High-Ox skill:** the threshold is the 95th percentile of Ox on training days. Reported as CSI
  and F1 for exceedance, and RMSE on cells above the threshold (`csi_p95`, `f1_p95`,
  `rmse_above_p95`).
- **Smoothing check:** radially averaged power spectra (mean removed, Hann taper) at leads 1, 5
  and 10. `spectral_ratio` is predicted/true power over the top third of wavenumbers: below 1 is
  too smooth, above 1 is noisy or blocky.
- **Confidence intervals:** 95% CIs by block bootstrap over days (`*_ci95_lo/hi`, needs ≥ 2
  days).
- **Test rows** are written once per config hash; `--rewrite-test` overrides this.

Rows go to `results/results.csv` (`run_id, model, split, lead_hour, metric, value, config_hash,
git_commit`), and a line goes to `EXPERIMENTS.md`. The run directory also gets:
- `per_window_rmse.csv`;
- `daily_stats.csv` (per-day sums, for paired comparisons);
- `spectra.csv`;
- `error_maps_<split>.npz` (the highest-Ox window);
- `summary.txt`.

`git_commit` ends in `-dirty` when tracked files other than the run logs had uncommitted changes.

## Benchmark

`configs/experiments/benchmark.yaml` fixes everything except the model, its tuned settings, the
target mode and the seed.

1. **Tune:** 8 random-search trials per model in `ox` mode, writing validation rows only. The
   learning rate (1e-4 to 3e-3) and weight decay (1e-6 to 1e-3) are drawn log-uniform, and trial
   *i* uses the same draw for every model. The model's knob is drawn from its three options.
2. **Select:** the trial with the lowest validation Ox RMSE, written to
   `results/benchmark_selection.json`.
3. **Final:** the selected config × 3 target modes × 3 seeds, validation and test.

`scripts/run_benchmark.py` is resumable: runs whose config hash already has its rows are skipped.
`--model NAME` runs one model and `--dry-run` lists what would run. It checks the parameter
budget before every run and refuses anything but a `dates` split.

`scripts/make_tables.py` builds everything from test rows only; smoke-test rows are refused. It
writes `results/benchmark/`:
- `main.md`/`main.tex`: persistence and climatology first, then model × target mode, with:
  - Ox RMSE, MAE, MB and r, each with a 95% CI (block bootstrap over test days, seed-averaged);
  - CSI (p95) and spectral ratio at leads 1/5/10;
  - parameters and inference time;
  - the paired RMSE difference vs the U-Net of the same target mode, with its CI.
- `lead_rmse.md`/`.tex`: Ox RMSE by lead hour.
- `figures/`: CSV data for Prism (lead-time curves with CIs, spectra, truth/forecast/error maps
  of the highest-Ox test window) and quick PNG previews.

`scripts/profile_models.py` measures each model on one GPU and writes `results/profile.csv`:
parameters, peak training memory, time per training step and epoch, and inference time for one
10-hour forecast on the full grid (batch 1, 50 runs after warm-up). Fill in the CMAQ wall-clock
column by hand; reruns keep it.

## Testing

```bash
uv run pytest                     # 114 tests on CPU (~2 min); 2 more run only on a CUDA GPU
uv run pytest -k "models and unet"
```

The tests run on synthetic days from `scripts/make_synthetic_data.py`, which uses the real files'
variables, dims, units and time stamps. They cover:
- **Data:** split boundaries, time alignment, the derived channels, and train-only
  normalization.
- **Baselines and targets:** persistence, train-only climatology, the three target modes (species
  sums in physical units, the multitask consistency term), and every metric against a direct
  computation.
- **Every model:**
  - output shape on 16x12 and 36x28 grids;
  - gradients reaching all parameters;
  - overfitting one tiny batch (loss down > 90% in 200 steps);
  - weights-only checkpoint round trips;
  - the parameter budget.
- **Model-specific:**
  - FNO at 2x the training resolution;
  - the GNN mesh and edge winds;
  - ConvLSTM causality.
- **GPU only:** the Mamba kernel wrapper against the PyTorch scan.
- **Workflow:** an end-to-end benchmark (tune → select → final → resume) and the tables built from
  it.

## Refactor check

`configs/legacy.yaml` runs the Phase 1 inputs and ConvLSTM (BatchNorm, flags off) on every window of
2018-11-13. In Phase 1 the pipeline in this mode gave per-window RMSE of 3.60 ppbV on average over
windows 0–7, against 3.56 for a fresh run of `legacy/grid_forcast.py`, with no window differing by
more than 0.23. The legacy script's own run-to-run spread is about 1 ppbV. To rerun the old script
without touching the committed `results/*.png`:

```bash
mkdir -p runs/legacy_script && ln -s ../../datasets runs/legacy_script/datasets
(cd runs/legacy_script && ../../.venv/bin/python ../../legacy/grid_forcast.py)
uv run aerocast-train --config configs/legacy.yaml
```

## What the single-day numbers mean

With one day of data every run is a `smoke_test`: trained and scored on the same windows. The
smoke numbers in `EXPERIMENTS.md` show that each model runs end to end; they are not a comparison
between models. Held-out results need the `dates` split, which needs more days.

## License

[Add your license information]

## Citation

If you use this code in your research, please cite:

```
[Add citation information]
```

## Contact

[Add contact information]
