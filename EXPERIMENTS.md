# Experiments

One line per run, written by aerocast-evaluate.

| date | run id | dataset | model | target mode | seed | config hash | val Ox RMSE | test Ox RMSE | note |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-28 | 20260928-142657_smoke_0c29 | baaqmd | convlstm | ox | 42 | 8363268a89 | – | – | smoke_test Ox RMSE 1.518 (in-sample); step 1 inputs, Phase 1 ConvLSTM width |
| 2026-09-28 | 20260928-144523_smoke-unet_e660 | baaqmd | unet | ox | 42 | 6aa37cd5dc | – | – | smoke_test Ox RMSE 1.514 (in-sample); U-Net smoke, ox |
| 2026-09-28 | 20260928-144625_smoke-unet_92f5 | baaqmd | unet | species | 42 | 1f0953da95 | – | – | smoke_test Ox RMSE 2.150 (in-sample); U-Net smoke, species |
| 2026-09-28 | 20260928-144715_smoke-unet_0d3e | baaqmd | unet | multitask | 42 | 378f7ea64f | – | – | smoke_test Ox RMSE 1.343 (in-sample); U-Net smoke, multitask |
| 2026-09-28 | 20260928-145452_smoke-fno_cdca | baaqmd | fno | ox | 42 | 3a21e4c246 | – | – | smoke_test Ox RMSE 0.830 (in-sample); FNO smoke, ox |
| 2026-09-28 | 20260928-150200_smoke-swin_8572 | baaqmd | swin | ox | 42 | c300db0f7a | – | – | smoke_test Ox RMSE 1.300 (in-sample); swin smoke, ox |
| 2026-09-28 | 20260928-150326_smoke-swin_unet_472e | baaqmd | swin_unet | ox | 42 | 24cf249d62 | – | – | smoke_test Ox RMSE 1.909 (in-sample); swin_unet smoke, ox |
| 2026-09-28 | 20260928-150946_smoke-gnn_e784 | baaqmd | gnn | ox | 42 | 3cb2c059bf | – | – | smoke_test Ox RMSE 1.747 (in-sample); GNN smoke, ox |
| 2026-09-28 | 20260928-151841_smoke-mamba_f56f | baaqmd | mamba | ox | 42 | 8370e75848 | – | – | smoke_test Ox RMSE 1.046 (in-sample); Mamba smoke, ox (mamba_ssm kernel) |
| 2026-09-28 | 20260928-152118_smoke-convlstm_22f8 | baaqmd | convlstm | ox | 42 | a40a4cc013 | – | – | smoke_test Ox RMSE 1.042 (in-sample); ConvLSTM smoke, ox (scaled, GroupNorm) |
| 2026-09-28 | 20260928-172247_equates_2019_07_ca12km-convlstm_1ee3 | equates_pilot | convlstm | ox | 42 | 524ac3e71f | 3.516 | 3.629 | pilot: EQUATES 2019-07, CA 72x72 |
| 2026-09-28 | 20260928-174140_equates_2019_07_ca12km-unet_c6ce | equates_pilot | unet | ox | 42 | 5be7941917 | 3.594 | 3.940 | pilot: EQUATES 2019-07, CA 72x72 |
| 2026-09-28 | 20260928-174330_equates_2019_07_ca12km-fno_5938 | equates_pilot | fno | ox | 42 | 9cd3ee7d1f | 4.065 | 4.292 | pilot: EQUATES 2019-07, CA 72x72 |
