# Experiments

One line per run, written by aerocast-evaluate.

| date | run id | model | target mode | seed | config hash | val Ox RMSE | test Ox RMSE | note |
|---|---|---|---|---|---|---|---|---|
| 2026-09-28 | 20260928-142657_smoke_0c29 | convlstm | ox | 42 | 8363268a89 | – | – | smoke_test Ox RMSE 1.518 (in-sample); step 1 inputs, Phase 1 ConvLSTM width |
| 2026-09-28 | 20260928-144523_smoke-unet_e660 | unet | ox | 42 | 6aa37cd5dc | – | – | smoke_test Ox RMSE 1.514 (in-sample); U-Net smoke, ox |
| 2026-09-28 | 20260928-144625_smoke-unet_92f5 | unet | species | 42 | 1f0953da95 | – | – | smoke_test Ox RMSE 2.150 (in-sample); U-Net smoke, species |
| 2026-09-28 | 20260928-144715_smoke-unet_0d3e | unet | multitask | 42 | 378f7ea64f | – | – | smoke_test Ox RMSE 1.343 (in-sample); U-Net smoke, multitask |
| 2026-09-28 | 20260928-145452_smoke-fno_cdca | fno | ox | 42 | 3a21e4c246 | – | – | smoke_test Ox RMSE 0.830 (in-sample); FNO smoke, ox |
| 2026-09-28 | 20260928-150200_smoke-swin_8572 | swin | ox | 42 | c300db0f7a | – | – | smoke_test Ox RMSE 1.300 (in-sample); swin smoke, ox |
| 2026-09-28 | 20260928-150326_smoke-swin_unet_472e | swin_unet | ox | 42 | 24cf249d62 | – | – | smoke_test Ox RMSE 1.909 (in-sample); swin_unet smoke, ox |
| 2026-09-28 | 20260928-150946_smoke-gnn_e784 | gnn | ox | 42 | 3cb2c059bf | – | – | smoke_test Ox RMSE 1.747 (in-sample); GNN smoke, ox |
| 2026-09-28 | 20260928-151841_smoke-mamba_f56f | mamba | ox | 42 | 8370e75848 | – | – | smoke_test Ox RMSE 1.046 (in-sample); Mamba smoke, ox (mamba_ssm kernel) |
