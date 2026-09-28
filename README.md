# AeroCast

ConvLSTM forecasting of Ox (NO2 + O3) on the 1 km BAAQMD CMAQ grid (224 x 164): 6 hours of
meteorology, chemical state and emissions in, 10 hourly Ox fields out.

Phase 1 turns the original single script (kept unchanged in `legacy/grid_forcast.py`) into a small,
tested pipeline with date-based train/val/test splits and two baselines (persistence and diurnal
climatology) scored on the same windows as the model.

## Quick start

```bash
uv sync                                   # installs aerocast (editable) + pytest
uv run pytest                             # synthetic-data tests, ~15 s on CPU

# Single-day smoke test on the real data (GPU recommended)
uv run aerocast-train --config configs/smoke.yaml

# Re-score a saved run (appends to results/results.csv again)
uv run aerocast-evaluate runs/<run_id>
```

Run commands from the repo root: data, cache and output paths in the configs are relative to it.
Any config value can be overridden on the command line, e.g.
`--set train.epochs=2 --set features.future_forcings=true`. `aerocast-train` evaluates the best
checkpoint when training finishes; pass `--no-eval` to skip that step.

On OSC, get a GPU first, e.g. `salloc -C gpu -t 2:00:00 -c 4 -A <project> --gpus=1`.

## Layout

```
configs/
  base.yaml          data paths, variables, seq/pred lengths, split dates, seed, training
  smoke.yaml         single-day smoke test (inherits base.yaml)
  legacy.yaml        refactor check against the legacy script
src/aerocast/
  config.py          YAML loading with `base:` inheritance, --set overrides, config hash
  data.py            load days by date, align hours, cache arrays, build windows
  splits.py          contiguous date-block splits; smoke_test and legacy modes
  normalize.py       stats from training windows only, saved as JSON
  baselines.py       persistence and diurnal climatology
  metrics.py         RMSE, MAE, mean bias, Pearson r per lead hour and overall
  models/convlstm.py StackedConvLSTM, moved unchanged from the legacy script
  train.py           aerocast-train
  evaluate.py        aerocast-evaluate
scripts/
  make_synthetic_data.py   CMAQ-like NetCDF days on a small grid (used by the tests)
tests/               pytest suite (synthetic data only)
legacy/
  grid_forcast.py    original script, unchanged
src/read_data.py     original exploratory loader (unchanged)
datasets/            NetCDF inputs (gitignored)
cache/               preprocessed per-day arrays (gitignored)
runs/                one directory per training run (gitignored)
results/results.csv  appended by every evaluation
```

## Data

Each day needs three files in `data.data_dir`, named by date (`{date}` = `YYYYMMDD`; patterns in
`data.files`, globs allowed):

| File | Variables used | Units |
|---|---|---|
| `METCRO2D_{date}.nc` | TEMP2, WSPD10, WDIR10 | K, m/s, degrees |
| `out.combine_{date}.nc` | NO, NO2, PM25_CL (inputs); NO2 + O3 (target) | ppbV, ug m-3 |
| `egts_l.{date}.*.ncf` | ALK1, OLE1, ARO1, ARO2, TERP, ISOP | moles/s |

All variables have dims `(TSTEP, LAY, ROW, COL)`; the surface layer is used. The target
Ox = NO2 + O3 is reported in the concentration file's units (ppbV), and each run records that
unit in `run.json`.

**Time alignment.** Hours are matched on each file's TFLAG plus `data.time_offset_hours`, and only
hours present in all three files are kept. METCRO2D and the emission files are stamped 08Z to 08Z
the next day, which is 00 to 24 local standard time (PST = UTC-8); their solar radiation and
rush-hour emissions confirm those stamps. The out.combine files are stamped 00Z to 23Z, but their
diurnal cycles (O3 peaking at step 13, NO at step 9) show they hold 00 to 23 PST, so `conc` gets
+8 h. With that offset each day contributes 24 aligned hours, identical to the index alignment the
legacy script used. Without it only 16 hours overlap per day, and the loader warns about it. If a
future batch of files is stamped differently, adjust the offsets.

**More days.** Drop the new files into `datasets/` and set the split date blocks in
`configs/base.yaml` (or a copy of it). Days are concatenated hourly, and windows never bridge a
missing day. The first read of each day is cached under `cache/`, keyed on the file names, sizes,
mtimes, variables and offsets.

## Splits

`split.mode` in the config:

- **`dates`** (default): `split.train`, `split.val` and `split.test` are contiguous, inclusive date
  blocks (`[start, end]` or a list of them) that must not overlap. Every hour takes the split of
  its file date. A window (6 input + 10 target hours) is kept only if all 16 hours are consecutive
  and in the same split; windows that cross a split boundary or a gap are dropped. The run fails if
  any split ends up with no windows.
- **`smoke_test`**: for a single day, where no clean held-out split exists. Training, early
  stopping and evaluation all use every window; the run prints a warning, and its rows are
  labelled `smoke_test`, never `test`. Climatology is fitted on the same day it is scored on, so
  it is trivially perfect in this mode.
- **`legacy`**: same windows as `smoke_test`, rows labelled `legacy`. Only for the refactor check
  below.

Only `dates` mode can produce rows labelled `test`.

## Training

- Fixed seed (`seed`), deterministic cuDNN, seeded shuffling.
- Normalization (`data.normalization`: none | predictors | full) is fitted on training windows
  only: per-channel input mean/std and a scalar target mean/std, using the legacy formula (hours
  weighted by how many training windows contain them, unbiased std + 1e-6). Val/test use the
  saved stats unchanged.
- Early stopping (`train.patience`) and the saved checkpoint both use the validation loss.
- Optimizer, scheduler, loss and clipping match the legacy script: Adam (1e-3, wd 1e-5),
  CosineAnnealingWarmRestarts (T_0 50, T_mult 2), 0.7 Huber(beta 0.5) + 0.3 MSE, grad-norm clip 1.0,
  batch 4, up to 500 epochs, patience 30.

Each run writes `runs/<run_id>/`: `config.yaml` (resolved, including the days used), `run.json`
(git commit, config hash, windows per split, channels, target unit, best epoch),
`norm_stats.json`, `best.pt` (weights plus stats and config), `history.csv`, and after evaluation
`per_window_rmse.csv` and `summary.txt`.

### Feature flags

Both flags are off by default, which reproduces the legacy inputs:

- `features.include_o3_input`: adds O3 to the past-state channels (NO, NO2, PM25_CL, O3).
- `features.future_forcings`: during the 10-step rollout, the frame fed before predicting hour
  t+k+1 carries the meteorology and emissions of hour t+k, with the state channels held at the last
  input hour. With the flag off, the last input frame is repeated, as in the legacy script.

## Evaluation

`aerocast-evaluate runs/<run_id>` (also run automatically by `aerocast-train`) scores three models
on exactly the same windows, in native units:

- `convlstm`: the best checkpoint, denormalized.
- `persistence`: Ox at the last input hour (t-1), held for all 10 lead hours.
- `climatology`: per-cell mean Ox for each hour of day, from training days only.

The metrics are RMSE, MAE, mean bias (forecast - truth) and Pearson r, per lead hour (1 to 10)
and overall (`lead_hour = all`). Rows are appended to `results/results.csv`:

```
run_id, model, split, lead_hour, metric, value, config_hash, git_commit
```

`split` is `val`/`test` in `dates` mode, otherwise `smoke_test` or `legacy`. `config_hash`
identifies the experiment settings (it ignores the run name, output paths and logging).
`git_commit` gets a `-dirty` suffix when tracked files had uncommitted changes.

## Refactor check

`configs/legacy.yaml` trains and scores on every window of 2018-11-13 with both flags off, and
should give per-window RMSE close to the legacy script's. To rerun the old script without
overwriting the committed PNGs in `results/`, run it from a scratch directory:

```bash
mkdir -p runs/legacy_script && ln -s ../../datasets runs/legacy_script/datasets
(cd runs/legacy_script && ../../.venv/bin/python ../../legacy/grid_forcast.py)
uv run aerocast-train --config configs/legacy.yaml
```

Known differences from the legacy script:
- The legacy `range(seq_len, T - pred_len)` skips the last valid window, so it has 8 windows per
  day where the pipeline has 9. Windows 0 to 7 are the same.
- The legacy script stops early on training loss and evaluates the final weights; the pipeline
  selects the best checkpoint on validation loss.
- The legacy script is unseeded.

Result on 2018-11-13 (A100, 2026-09-28). Per-window RMSE is in ppbV and in-sample:

| Window (first target hour) | Legacy script | Pipeline, `legacy` mode | Persistence |
|---|---|---|---|
| 0 (06 PST) | 4.28 | 4.35 | 6.01 |
| 1 (07 PST) | 4.11 | 3.99 | 6.71 |
| 2 (08 PST) | 3.83 | 3.70 | 7.34 |
| 3 (09 PST) | 3.47 | 3.50 | 7.34 |
| 4 (10 PST) | 3.14 | 3.30 | 6.67 |
| 5 (11 PST) | 2.99 | 3.22 | 5.30 |
| 6 (12 PST) | 3.16 | 3.31 | 3.94 |
| 7 (13 PST) | 3.53 | 3.47 | 4.07 |
| 8 (14 PST) | – | 3.67 | 5.26 |
| Mean of 0–7 | 3.56 | 3.60 | 5.92 |

Earlier runs of the unseeded legacy script averaged 4.25 (the numbers previously in this README)
and 4.69 (W&B, 2026-04-06). The run-to-run spread is larger than the old-vs-new difference.

## Model

`StackedConvLSTM`: two ConvLSTM cells (12 → 64 → 32 channels, 3x3 kernels, BatchNorm on the gates
and on each hidden state), then a 1x1 convolution to one channel. It encodes the 6 input hours, then
rolls out autoregressively in hidden-state space for 10 steps.

```
Input  (B, 6, C, 224, 164)   C = 12 (13 with include_o3_input)
Output (B, 10, 224, 164)     Ox in normalized units, denormalized for evaluation
```

## What the single-day numbers mean

The legacy script trained on the same 8 windows of one day that it reported on, and computed its
normalization over all of them, so its 3.7 to 4.8 ppbV RMSE is in-sample. The smoke test is
in-sample too. Even so, persistence beats the ConvLSTM at +1 h (1.97 vs 4.38) and +2 h (3.56 vs
4.30); the ConvLSTM only wins from +3 h on. None of these numbers are a held-out result. That
needs the `dates` split, which needs more days.

## License

[Add your license information]

## Citation

If you use this code in your research, please cite:

```
[Add citation information]
```

## Contact

[Add contact information]
