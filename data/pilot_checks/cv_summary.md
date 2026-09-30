# EQUATES pilot: blocked cross-validation (4 folds x 3 seeds, ox target mode)

Each fold holds out a different week of July as `test` (with a 5-day `val` block and 1-day buffers separating train/val/test), so the number below is a mean +/- spread over four different weather regimes and three seeds (up to 12 runs per model), not one number from one episode. See configs/data/equates_2019_07_ca12km_fold{1..4}.yaml for the exact day blocks. Models with fewer than 4 folds present are still in progress.

## Per-fold mean (over 3 seeds), test Ox RMSE (ppbV)

| model | fold 1 | fold 2 | fold 3 | fold 4 | mean of folds | fold std |
|---|---|---|---|---|---|
| convlstm | 2.743 | 4.025 | 3.285 | 4.707 | 3.690 | 0.743 |
| unet | 2.764 | 3.583 | 3.460 | 4.559 | 3.592 | 0.640 |
| fno | 2.998 | 4.049 | 3.644 | 4.906 | 3.900 | 0.692 |
| swin | 3.106 | 4.847 | 3.558 | 5.423 | 4.234 | 0.938 |
| swin_unet | 2.770 | 3.581 | 3.625 | 4.279 | 3.564 | 0.535 |
| gnn | 2.775 | 3.421 | 3.078 | 3.757 | 3.258 | 0.368 |
| mamba | 2.817 | 3.445 | 3.266 | 4.235 | 3.441 | 0.512 |

## All runs per model (up to 4 folds x 3 seeds), test Ox RMSE (ppbV)

| model | mean | std | min | max | n |
|---|---|---|---|---|---|
| convlstm | 3.690 | 0.752 | 2.687 | 4.804 | 12 |
| unet | 3.592 | 0.645 | 2.668 | 4.714 | 24 |
| fno | 4.028 | 0.689 | 2.975 | 5.303 | 21 |
| swin | 4.234 | 0.947 | 2.907 | 5.537 | 12 |
| swin_unet | 3.564 | 0.537 | 2.738 | 4.324 | 12 |
| gnn | 3.258 | 0.370 | 2.737 | 3.801 | 12 |
| mamba | 3.441 | 0.523 | 2.657 | 4.403 | 12 |

## mae (mean +/- std over all runs per model)

| model | mean | std |
|---|---|---|
| convlstm | 2.745 | 0.629 |
| unet | 2.655 | 0.544 |
| fno | 2.912 | 0.545 |
| swin | 3.274 | 0.799 |
| swin_unet | 2.542 | 0.363 |
| gnn | 2.286 | 0.239 |
| mamba | 2.412 | 0.373 |

## bias (mean +/- std over all runs per model)

| model | mean | std |
|---|---|---|
| convlstm | -0.391 | 1.721 |
| unet | -0.421 | 1.410 |
| fno | -0.345 | 1.473 |
| swin | -0.437 | 2.310 |
| swin_unet | -0.125 | 0.580 |
| gnn | -0.067 | 0.909 |
| mamba | -0.363 | 0.935 |

## pearson_r (mean +/- std over all runs per model)

| model | mean | std |
|---|---|---|
| convlstm | 0.949 | 0.009 |
| unet | 0.946 | 0.009 |
| fno | 0.932 | 0.014 |
| swin | 0.937 | 0.016 |
| swin_unet | 0.939 | 0.013 |
| gnn | 0.952 | 0.007 |
| mamba | 0.947 | 0.010 |

## RMSE by fold, all models

![cv rmse by fold](cv_rmse_by_fold.png)
