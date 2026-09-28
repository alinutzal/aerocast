# Experiments

One line per run, written by aerocast-evaluate.

| date | run id | model | target mode | seed | config hash | val Ox RMSE | test Ox RMSE | note |
|---|---|---|---|---|---|---|---|---|
| 2026-09-28 | 20260928-142657_smoke_0c29 | convlstm | ox | 42 | 8363268a89 | – | – | smoke_test Ox RMSE 1.518 (in-sample); step 1 inputs, Phase 1 ConvLSTM width |
| 2026-09-28 | 20260928-144523_smoke-unet_e660 | unet | ox | 42 | 6aa37cd5dc | – | – | smoke_test Ox RMSE 1.514 (in-sample); U-Net smoke, ox |
| 2026-09-28 | 20260928-144625_smoke-unet_92f5 | unet | species | 42 | 1f0953da95 | – | – | smoke_test Ox RMSE 2.150 (in-sample); U-Net smoke, species |
| 2026-09-28 | 20260928-144715_smoke-unet_0d3e | unet | multitask | 42 | 378f7ea64f | – | – | smoke_test Ox RMSE 1.343 (in-sample); U-Net smoke, multitask |
| 2026-09-28 | 20260928-145452_smoke-fno_cdca | fno | ox | 42 | 3a21e4c246 | – | – | smoke_test Ox RMSE 0.830 (in-sample); FNO smoke, ox |
