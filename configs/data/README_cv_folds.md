# EQUATES pilot: blocked cross-validation folds

`equates_2019_07_ca12km_fold{1,2,3,4}.yaml` replace the single train/val/test split
(`equates_2019_07_ca12km.yaml`, still used directly by `scripts/extract_equates_pilot.py`
and the data card) with 4-fold blocked CV: each fold's `test` block is a different week of
July, so scoring across all four folds gives a mean +/- spread over four different weather
regimes rather than one number from one 5-day episode. Run 3 seeds per fold (`--set
seed=42/43/44`); `scripts/summarize_equates_cv.py` aggregates the resulting 4 x 3 = 12 runs
per model.

Every fold keeps a 1-day buffer between every pair of differently-labeled blocks (so no
window's input/target hours span a split boundary - `split.mode: dates` already refuses to
build a window that isn't entirely inside one label, but the buffer additionally keeps
`train` days from sitting immediately next to `val`/`test` days, whose weather is often
autocorrelated with its neighbors). Training on days after the held-out week is fine (and
used, for folds 1-3) since this is an emulator, not a causal forecast.

| Fold | Train | Val | Test (the held-out week) |
|---|---|---|---|
| 1 | Jul 15-31 (17d, after) | Jul 9-13 | **Jul 1-7** |
| 2 | Jul 1-7 + Jul 23-31 (16d, both sides) | Jul 17-21 | **Jul 9-15** |
| 3 | Jul 1-15 + Jul 31 (16d, both sides) | Jul 25-29 | **Jul 17-23** |
| 4 | Jul 1-16 (16d, before) | Jul 18-22 | **Jul 24-30** |

Fold 4's val sits *before* test, not after - the only layout that fits within the month for
the last fold (test runs to Jul 30, one day short of month end, itself used as a trailing
buffer). Every other fold places val after test.

Day-by-day, with `T`=train, `v`=val (1-day buffer), `x`=test (1-day buffer), `.`=buffer:

```
Jul   1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
F1:   x  x  x  x  x  x  x  .  v  v  v  v  v  .  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T
F2:   T  T  T  T  T  T  T  .  x  x  x  x  x  x  x  .  v  v  v  v  v  .  T  T  T  T  T  T  T  T  T
F3:   T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  .  x  x  x  x  x  x  x  .  v  v  v  v  v  .  T
F4:   T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  T  .  v  v  v  v  v  .  x  x  x  x  x  x  x  .
```
