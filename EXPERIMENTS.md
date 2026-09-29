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
| 2026-09-28 | 20260928-182619_equates_2019_07_ca12km-fno_04fe | equates_pilot | fno | ox | 42 | 635b85984a | 4.098 | 4.287 | pilot: EQUATES 2019-07, CA 72x72; FNO positivity constraint enabled |
| 2026-09-28 | 20260928-192607_equates_2019_07_ca12km_fold3-convlstm_a47d | equates_pilot | convlstm | ox | 42 | fc6b5b2e4c | 4.447 | 3.261 | blocked CV fold 3/4, seed 42 |
| 2026-09-28 | 20260928-192607_equates_2019_07_ca12km_fold4-convlstm_0fb3 | equates_pilot | convlstm | ox | 42 | a5821a7023 | 3.421 | 4.804 | blocked CV fold 4/4, seed 42 |
| 2026-09-28 | 20260928-192607_equates_2019_07_ca12km_fold1-convlstm_a30a | equates_pilot | convlstm | ox | 42 | 5ea8b7a33a | 3.205 | 2.783 | blocked CV fold 1/4, seed 42 |
| 2026-09-28 | 20260928-192607_equates_2019_07_ca12km_fold2-convlstm_dd2f | equates_pilot | convlstm | ox | 42 | 6db426aa32 | 2.940 | 3.829 | blocked CV fold 2/4, seed 42 |
| 2026-09-28 | 20260928-194004_equates_2019_07_ca12km_fold3-convlstm_5a01 | equates_pilot | convlstm | ox | 43 | b5bcab73af | 4.785 | 3.503 | blocked CV fold 3/4, seed 43 |
| 2026-09-28 | 20260928-194241_equates_2019_07_ca12km_fold4-convlstm_4370 | equates_pilot | convlstm | ox | 43 | e0c022d386 | 3.345 | 4.674 | blocked CV fold 4/4, seed 43 |
| 2026-09-28 | 20260928-194304_equates_2019_07_ca12km_fold1-convlstm_9f49 | equates_pilot | convlstm | ox | 43 | a6cedb3835 | 3.209 | 2.759 | blocked CV fold 1/4, seed 43 |
| 2026-09-28 | 20260928-194412_equates_2019_07_ca12km_fold2-convlstm_824d | equates_pilot | convlstm | ox | 43 | 1a61860766 | 2.994 | 4.143 | blocked CV fold 2/4, seed 43 |
| 2026-09-28 | 20260928-195724_equates_2019_07_ca12km_fold4-convlstm_46fb | equates_pilot | convlstm | ox | 44 | 8c9fc9389f | 3.171 | 4.643 | blocked CV fold 4/4, seed 44 |
| 2026-09-29 | 20260928-201522_equates_2019_07_ca12km_fold4-unet_7556 | equates_pilot | unet | ox | 42 | 360f41612c | 3.190 | 4.590 | blocked CV fold 4/4, seed 42 |
| 2026-09-29 | 20260928-201816_equates_2019_07_ca12km_fold4-unet_257b | equates_pilot | unet | ox | 43 | 4e98a00ced | 3.253 | 4.425 | blocked CV fold 4/4, seed 43 |
| 2026-09-29 | 20260928-202044_equates_2019_07_ca12km_fold4-unet_ba3d | equates_pilot | unet | ox | 44 | 3835250a2a | 3.407 | 4.623 | blocked CV fold 4/4, seed 44 |
| 2026-09-29 | 20260928-202233_equates_2019_07_ca12km_fold4-fno_5390 | equates_pilot | fno | ox | 42 | 749c27790e | 3.740 | 5.193 | blocked CV fold 4/4, seed 42 |
| 2026-09-29 | 20260928-202339_equates_2019_07_ca12km_fold4-fno_0956 | equates_pilot | fno | ox | 43 | 9b02022df6 | 3.805 | 5.303 | blocked CV fold 4/4, seed 43 |
| 2026-09-28 | 20260928-195335_equates_2019_07_ca12km_fold3-convlstm_d840 | equates_pilot | convlstm | ox | 44 | 234e908e87 | 4.212 | 3.092 | blocked CV fold 3/4, seed 44 |
| 2026-09-29 | 20260928-202447_equates_2019_07_ca12km_fold4-fno_c9bc | equates_pilot | fno | ox | 44 | 77db0cf0f9 | 3.672 | 5.278 | blocked CV fold 4/4, seed 44 |
| 2026-09-29 | 20260928-202533_equates_2019_07_ca12km_fold3-unet_700d | equates_pilot | unet | ox | 42 | 207dfb8b48 | 4.233 | 3.373 | blocked CV fold 3/4, seed 42 |
| 2026-09-29 | 20260928-202716_equates_2019_07_ca12km_fold3-unet_a400 | equates_pilot | unet | ox | 43 | 9df918cd3e | 4.456 | 3.544 | blocked CV fold 3/4, seed 43 |
| 2026-09-29 | 20260928-202844_equates_2019_07_ca12km_fold3-unet_aaed | equates_pilot | unet | ox | 44 | aae224ef8e | 4.167 | 3.430 | blocked CV fold 3/4, seed 44 |
| 2026-09-29 | 20260928-203014_equates_2019_07_ca12km_fold3-fno_7692 | equates_pilot | fno | ox | 42 | c2afbbfb58 | 5.403 | 3.818 | blocked CV fold 3/4, seed 42 |
| 2026-09-29 | 20260928-203140_equates_2019_07_ca12km_fold3-fno_5368 | equates_pilot | fno | ox | 43 | 29e8e99b96 | 5.002 | 3.703 | blocked CV fold 3/4, seed 43 |
| 2026-09-29 | 20260928-203248_equates_2019_07_ca12km_fold3-fno_6a6d | equates_pilot | fno | ox | 44 | 6c01478a6b | 5.080 | 3.740 | blocked CV fold 3/4, seed 44 |
| 2026-09-29 | 20260928-200459_equates_2019_07_ca12km_fold2-convlstm_c0a0 | equates_pilot | convlstm | ox | 44 | 97410aed35 | 2.915 | 4.103 | blocked CV fold 2/4, seed 44 |
| 2026-09-29 | 20260928-203611_equates_2019_07_ca12km_fold2-unet_85d6 | equates_pilot | unet | ox | 42 | fd7bcdf7f4 | 3.205 | 3.572 | blocked CV fold 2/4, seed 42 |
| 2026-09-29 | 20260928-203815_equates_2019_07_ca12km_fold2-unet_8fd5 | equates_pilot | unet | ox | 43 | f4aa4775bc | 3.035 | 3.605 | blocked CV fold 2/4, seed 43 |
| 2026-09-29 | 20260928-204216_equates_2019_07_ca12km_fold2-unet_456d | equates_pilot | unet | ox | 44 | 5b320fa33d | 3.188 | 3.533 | blocked CV fold 2/4, seed 44 |
| 2026-09-29 | 20260928-204419_equates_2019_07_ca12km_fold2-fno_46a3 | equates_pilot | fno | ox | 42 | d9b45dd10b | 3.439 | 4.431 | blocked CV fold 2/4, seed 42 |
| 2026-09-29 | 20260928-200240_equates_2019_07_ca12km_fold1-convlstm_429a | equates_pilot | convlstm | ox | 44 | 6903897197 | 3.219 | 2.687 | blocked CV fold 1/4, seed 44 |
| 2026-09-29 | 20260928-204552_equates_2019_07_ca12km_fold2-fno_1c45 | equates_pilot | fno | ox | 43 | 96535eaedf | 3.400 | 3.919 | blocked CV fold 2/4, seed 43 |
| 2026-09-29 | 20260928-204710_equates_2019_07_ca12km_fold2-fno_8944 | equates_pilot | fno | ox | 44 | 8a6829e457 | 3.337 | 4.170 | blocked CV fold 2/4, seed 44 |
| 2026-09-29 | 20260928-204614_equates_2019_07_ca12km_fold1-unet_46ca | equates_pilot | unet | ox | 42 | 7a1843bba1 | 3.246 | 2.767 | blocked CV fold 1/4, seed 42 |
| 2026-09-29 | 20260928-204933_equates_2019_07_ca12km_fold1-unet_276b | equates_pilot | unet | ox | 43 | 8e4543284f | 3.466 | 2.804 | blocked CV fold 1/4, seed 43 |
| 2026-09-29 | 20260928-205232_equates_2019_07_ca12km_fold1-unet_986f | equates_pilot | unet | ox | 44 | e73a70e184 | 3.189 | 2.668 | blocked CV fold 1/4, seed 44 |
| 2026-09-29 | 20260928-205448_equates_2019_07_ca12km_fold1-fno_d3f1 | equates_pilot | fno | ox | 42 | d4d6f5bc6e | 3.527 | 2.995 | blocked CV fold 1/4, seed 42 |
| 2026-09-29 | 20260928-205632_equates_2019_07_ca12km_fold1-fno_52c9 | equates_pilot | fno | ox | 43 | a755218e96 | 3.489 | 2.975 | blocked CV fold 1/4, seed 43 |
| 2026-09-29 | 20260928-205809_equates_2019_07_ca12km_fold1-fno_d35e | equates_pilot | fno | ox | 44 | 363769a931 | 3.564 | 3.025 | blocked CV fold 1/4, seed 44 |
