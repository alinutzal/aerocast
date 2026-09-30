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
| 2026-09-29 | 20260929-094432_equates_2019_07_ca12km-unet_843b | equates_pilot | unet | ox | 42 | 5be7941917 | 3.591 | – | pilot: EQUATES 2019-07, CA 72x72 |
| 2026-09-29 | 20260929-095311_benchmark_equates_fold4-tune-fno-t0_f9d0 | equates_pilot | fno | ox | 42 | 8259a2151a | 3.627 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095311_benchmark_equates_fold2-tune-fno-t0_2a00 | equates_pilot | fno | ox | 42 | 6c6fe4e0d3 | 3.372 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095357_benchmark_equates_fold3-tune-fno-t0_a030 | equates_pilot | fno | ox | 42 | 9bab3f2d6b | 5.027 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095410_benchmark_equates_fold4-tune-fno-t1_9060 | equates_pilot | fno | ox | 42 | 5fa5395f00 | 3.404 | – | tuning trial 1 |
| 2026-09-29 | 20260929-095452_benchmark_equates_fold3-tune-fno-t1_c149 | equates_pilot | fno | ox | 42 | 4e45f68bc3 | 4.856 | – | tuning trial 1 |
| 2026-09-29 | 20260929-095458_benchmark_equates_fold3-tune-unet-t0_20ec | equates_pilot | unet | ox | 42 | baf5479122 | 4.539 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095416_benchmark_equates_fold2-tune-fno-t1_b4f2 | equates_pilot | fno | ox | 42 | 491e19a8a8 | 3.249 | – | tuning trial 1 |
| 2026-09-29 | 20260929-095508_benchmark_equates_fold1-tune-unet-t0_c5bc | equates_pilot | unet | ox | 42 | 6b3f67f02d | 3.374 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095317_benchmark_equates_fold4-tune-swin-t0_507e | equates_pilot | swin | ox | 42 | e79f5aa560 | 3.426 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095610_benchmark_equates_fold3-tune-unet-t1_6a8a | equates_pilot | unet | ox | 42 | ee25e2083b | 4.583 | – | tuning trial 1 |
| 2026-09-29 | 20260929-095335_benchmark_equates_fold3-tune-swin-t0_4c04 | equates_pilot | swin | ox | 42 | edbca19a68 | 5.358 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095405_benchmark_equates_fold1-tune-gnn-t0_e62e | equates_pilot | gnn | ox | 42 | 31605c07b0 | 3.690 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095649_benchmark_equates_fold1-tune-unet-t1_d74e | equates_pilot | unet | ox | 42 | 0bc73f4ff9 | 3.391 | – | tuning trial 1 |
| 2026-09-29 | 20260929-095553_benchmark_equates_fold3-tune-fno-t2_4003 | equates_pilot | fno | ox | 42 | 508b04cf9b | 4.777 | – | tuning trial 2 |
| 2026-09-29 | 20260929-095335_benchmark_equates_fold3-tune-swin_unet-t0_1b11 | equates_pilot | swin_unet | ox | 42 | bdd83ac0f0 | 4.382 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095311_benchmark_equates_fold1-tune-swin-t0_d512 | equates_pilot | swin | ox | 42 | 7c31d25dd2 | 3.847 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095801_benchmark_equates_fold1-tune-unet-t2_c3a0 | equates_pilot | unet | ox | 42 | f6e9e5bc7a | 3.339 | – | tuning trial 2 |
| 2026-09-29 | 20260929-095804_benchmark_equates_fold3-tune-fno-t3_5202 | equates_pilot | fno | ox | 42 | dd27534b70 | 5.711 | – | tuning trial 3 |
| 2026-09-29 | 20260929-095720_benchmark_equates_fold3-tune-unet-t2_dbb8 | equates_pilot | unet | ox | 42 | e0750938a5 | 4.287 | – | tuning trial 2 |
| 2026-09-29 | 20260929-095900_benchmark_equates_fold3-tune-fno-t4_79e3 | equates_pilot | fno | ox | 42 | 14a8b757c5 | 4.820 | – | tuning trial 4 |
| 2026-09-29 | 20260929-095859_benchmark_equates_fold1-tune-unet-t3_9f5e | equates_pilot | unet | ox | 42 | 7a9462ca41 | 3.439 | – | tuning trial 3 |
| 2026-09-29 | 20260929-095914_benchmark_equates_fold3-tune-unet-t3_3890 | equates_pilot | unet | ox | 42 | 8774cf8325 | 4.432 | – | tuning trial 3 |
| 2026-09-29 | 20260929-095635_benchmark_equates_fold2-tune-fno-t2_665b | equates_pilot | fno | ox | 42 | a0eb008fea | 3.303 | – | tuning trial 2 |
| 2026-09-29 | 20260929-095317_benchmark_equates_fold2-tune-mamba-t0_81fc | equates_pilot | mamba | ox | 42 | acbdb1af2c | 2.937 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095529_benchmark_equates_fold4-tune-mamba-t0_09e9 | equates_pilot | mamba | ox | 42 | 8be89a2d30 | 3.289 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095949_benchmark_equates_fold3-tune-fno-t5_3b0d | equates_pilot | fno | ox | 42 | b02dffd434 | 5.628 | – | tuning trial 5 |
| 2026-09-29 | 20260929-095437_benchmark_equates_fold4-tune-gnn-t0_2e4e | equates_pilot | gnn | ox | 42 | 0eedda391b | 3.140 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095939_benchmark_equates_fold1-tune-fno-t0_d300 | equates_pilot | fno | ox | 42 | fc14deadf6 | 3.522 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095724_benchmark_equates_fold3-tune-swin-t1_85d6 | equates_pilot | swin | ox | 42 | d3fb297351 | 5.457 | – | tuning trial 1 |
| 2026-09-29 | 20260929-100014_benchmark_equates_fold3-tune-unet-t4_d3d3 | equates_pilot | unet | ox | 42 | d6ecafffd2 | 4.705 | – | tuning trial 4 |
| 2026-09-29 | 20260929-100044_benchmark_equates_fold3-tune-fno-t6_1db2 | equates_pilot | fno | ox | 42 | 47e9c8962b | 4.741 | – | tuning trial 6 |
| 2026-09-29 | 20260929-095611_benchmark_equates_fold1-tune-swin_unet-t0_39eb | equates_pilot | swin_unet | ox | 42 | f4746e6f6c | 3.628 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095708_benchmark_equates_fold2-tune-swin_unet-t0_2d2d | equates_pilot | swin_unet | ox | 42 | 6839f680d9 | 3.556 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095801_benchmark_equates_fold1-tune-gnn-t1_4af8 | equates_pilot | gnn | ox | 42 | 5b9e2db5df | 3.332 | – | tuning trial 1 |
| 2026-09-29 | 20260929-100021_benchmark_equates_fold2-tune-fno-t3_1447 | equates_pilot | fno | ox | 42 | 9917e00f46 | 3.576 | – | tuning trial 3 |
| 2026-09-29 | 20260929-095713_benchmark_equates_fold4-tune-swin-t1_1895 | equates_pilot | swin | ox | 42 | ecc70b7a1d | 3.727 | – | tuning trial 1 |
| 2026-09-29 | 20260929-100012_benchmark_equates_fold1-tune-unet-t4_825f | equates_pilot | unet | ox | 42 | fc68cbf063 | 3.413 | – | tuning trial 4 |
| 2026-09-29 | 20260929-100047_benchmark_equates_fold1-tune-fno-t1_4f9d | equates_pilot | fno | ox | 42 | c5c233b156 | 3.547 | – | tuning trial 1 |
| 2026-09-29 | 20260929-100143_benchmark_equates_fold3-tune-fno-t7_800e | equates_pilot | fno | ox | 42 | 43aca9856b | 5.428 | – | tuning trial 7 |
| 2026-09-29 | 20260929-100113_benchmark_equates_fold4-tune-unet-t0_1911 | equates_pilot | unet | ox | 42 | 9ca901b999 | 3.196 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095809_benchmark_equates_fold2-tune-swin-t0_5b23 | equates_pilot | swin | ox | 42 | 68047ae706 | 3.605 | – | tuning trial 0 |
| 2026-09-29 | 20260929-100126_benchmark_equates_fold3-tune-unet-t5_6d42 | equates_pilot | unet | ox | 42 | eaf48b29e9 | 4.446 | – | tuning trial 5 |
| 2026-09-29 | 20260929-095549_benchmark_equates_fold4-tune-fno-t2_2ba9 | equates_pilot | fno | ox | 42 | 06a36eeec2 | 3.501 | – | tuning trial 2 |
| 2026-09-29 | 20260929-100039_benchmark_equates_fold2-tune-unet-t0_4d59 | equates_pilot | unet | ox | 42 | d69c0c6ed0 | 3.170 | – | tuning trial 0 |
| 2026-09-29 | 20260929-095830_benchmark_equates_fold3-tune-swin_unet-t1_5cad | equates_pilot | swin_unet | ox | 42 | 6d8ed21bef | 4.577 | – | tuning trial 1 |
| 2026-09-29 | 20260929-095836_benchmark_equates_fold1-tune-swin-t1_09de | equates_pilot | swin | ox | 42 | 080325f0c9 | 4.232 | – | tuning trial 1 |
| 2026-09-29 | 20260929-100225_benchmark_equates_fold3-final-fno-ox-s42_4e2b | equates_pilot | fno | ox | 42 | 47e9c8962b | 4.741 | 3.502 | final, tuning trial 6 |
| 2026-09-29 | 20260929-100153_benchmark_equates_fold2-tune-fno-t4_8eb4 | equates_pilot | fno | ox | 42 | b0756e93b1 | 3.241 | – | tuning trial 4 |
| 2026-09-29 | 20260929-100227_benchmark_equates_fold4-tune-unet-t1_8c45 | equates_pilot | unet | ox | 42 | 98d06ea572 | 3.336 | – | tuning trial 1 |
| 2026-09-29 | 20260929-095739_benchmark_equates_fold3-tune-gnn-t0_4746 | equates_pilot | gnn | ox | 42 | 83cfde1aa0 | 4.904 | – | tuning trial 0 |
| 2026-09-29 | 20260929-100237_benchmark_equates_fold3-tune-unet-t6_0fdf | equates_pilot | unet | ox | 42 | e4509380b0 | 4.622 | – | tuning trial 6 |
| 2026-09-29 | 20260929-100044_benchmark_equates_fold4-tune-gnn-t1_d5e6 | equates_pilot | gnn | ox | 42 | 4a6ba68a95 | 3.258 | – | tuning trial 1 |
| 2026-09-29 | 20260929-100213_benchmark_equates_fold1-tune-unet-t5_254d | equates_pilot | unet | ox | 42 | 285ccf7980 | 3.574 | – | tuning trial 5 |
| 2026-09-29 | 20260929-100056_benchmark_equates_fold3-tune-swin-t2_929a | equates_pilot | swin | ox | 42 | 7b19eac53a | 5.674 | – | tuning trial 2 |
| 2026-09-29 | 20260929-100325_benchmark_equates_fold3-final-fno-ox-s43_4767 | equates_pilot | fno | ox | 43 | b531f165f0 | 4.760 | 3.595 | final, tuning trial 6 |
| 2026-09-29 | 20260929-100041_benchmark_equates_fold2-tune-mamba-t1_687a | equates_pilot | mamba | ox | 42 | 376e4c710d | 2.922 | – | tuning trial 1 |
| 2026-09-29 | 20260929-100555_benchmark_equates_fold1-tune-fno-t2_b9e3 | equates_pilot | fno | ox | 42 | 838d20426a | 3.462 | – | tuning trial 2 |
| 2026-09-29 | 20260929-100838_benchmark_equates_fold1-tune-fno-t3_c70b | equates_pilot | fno | ox | 42 | 04b8721c77 | 3.751 | – | tuning trial 3 |
| 2026-09-29 | 20260929-100601_benchmark_equates_fold1-tune-unet-t6_d531 | equates_pilot | unet | ox | 42 | ac210ba750 | 3.509 | – | tuning trial 6 |
| 2026-09-29 | 20260929-100952_benchmark_equates_fold1-tune-fno-t4_b1ea | equates_pilot | fno | ox | 42 | 72514f3179 | 3.474 | – | tuning trial 4 |
| 2026-09-29 | 20260929-100555_benchmark_equates_fold1-tune-swin-t2_9107 | equates_pilot | swin | ox | 42 | 2de4f2c0f7 | 4.259 | – | tuning trial 2 |
| 2026-09-29 | 20260929-101051_benchmark_equates_fold1-tune-fno-t5_e306 | equates_pilot | fno | ox | 42 | 21cb3c2b55 | 3.592 | – | tuning trial 5 |
| 2026-09-29 | 20260929-100557_benchmark_equates_fold1-tune-swin_unet-t1_94a4 | equates_pilot | swin_unet | ox | 42 | 0692b73d39 | 3.624 | – | tuning trial 1 |
| 2026-09-29 | 20260929-100953_benchmark_equates_fold1-tune-unet-t7_a840 | equates_pilot | unet | ox | 42 | 0a1c47881f | 3.425 | – | tuning trial 7 |
| 2026-09-29 | 20260929-101141_benchmark_equates_fold1-tune-fno-t6_1d11 | equates_pilot | fno | ox | 42 | 6fe62fbac2 | 3.398 | – | tuning trial 6 |
| 2026-09-29 | 20260929-101207_benchmark_equates_fold1-final-unet-ox-s42_1d81 | equates_pilot | unet | ox | 42 | f6e9e5bc7a | 3.444 | 2.726 | final, tuning trial 2 |
| 2026-09-29 | 20260929-101132_benchmark_equates_fold1-tune-swin-t3_830a | equates_pilot | swin | ox | 42 | 8852ba1b74 | 4.469 | – | tuning trial 3 |
| 2026-09-29 | 20260929-101351_benchmark_equates_fold1-tune-gnn-t2_6ad0 | equates_pilot | gnn | ox | 42 | 6599226d05 | 3.256 | – | tuning trial 2 |
| 2026-09-29 | 20260929-101154_benchmark_equates_fold1-tune-swin_unet-t2_8313 | equates_pilot | swin_unet | ox | 42 | fb2ae3b83a | 3.916 | – | tuning trial 2 |
| 2026-09-29 | 20260929-101443_benchmark_equates_fold1-final-unet-ox-s43_8f49 | equates_pilot | unet | ox | 43 | ed2abdad27 | 3.489 | 2.750 | final, tuning trial 2 |
| 2026-09-29 | 20260929-101807_benchmark_equates_fold1-final-unet-ox-s44_9eb2 | equates_pilot | unet | ox | 44 | 66e4e81282 | 3.254 | 2.868 | final, tuning trial 2 |
| 2026-09-29 | 20260929-101705_benchmark_equates_fold1-tune-gnn-t3_ccd9 | equates_pilot | gnn | ox | 42 | 0199fec9be | 3.404 | – | tuning trial 3 |
| 2026-09-29 | 20260929-101723_benchmark_equates_fold1-tune-swin_unet-t3_8038 | equates_pilot | swin_unet | ox | 42 | ccdea2d39b | 4.061 | – | tuning trial 3 |
| 2026-09-29 | 20260929-102035_benchmark_equates_fold1-final-unet-species-s42_6e22 | equates_pilot | unet | species | 42 | af7d0e643b | 3.752 | 2.831 | final, tuning trial 2 |
| 2026-09-29 | 20260929-101700_benchmark_equates_fold1-tune-swin-t4_2200 | equates_pilot | swin | ox | 42 | bbbb82a2de | 3.760 | – | tuning trial 4 |
| 2026-09-29 | 20260929-102222_benchmark_equates_fold1-final-unet-species-s43_50c0 | equates_pilot | unet | species | 43 | d9ca2b18fd | 3.546 | 2.820 | final, tuning trial 2 |
| 2026-09-29 | 20260929-102417_benchmark_equates_fold1-final-unet-species-s44_4439 | equates_pilot | unet | species | 44 | 891dc3e768 | 3.680 | 2.854 | final, tuning trial 2 |
| 2026-09-29 | 20260929-102106_benchmark_equates_fold1-tune-gnn-t4_9b40 | equates_pilot | gnn | ox | 42 | a2bbc01f71 | 3.728 | – | tuning trial 4 |
| 2026-09-29 | 20260929-102259_benchmark_equates_fold1-tune-swin-t5_3b3b | equates_pilot | swin | ox | 42 | 92682e6c20 | 4.001 | – | tuning trial 5 |
| 2026-09-29 | 20260929-102555_benchmark_equates_fold1-final-unet-multitask-s42_a427 | equates_pilot | unet | multitask | 42 | 70af3ee93d | 3.613 | 2.743 | final, tuning trial 2 |
| 2026-09-29 | 20260929-102127_benchmark_equates_fold1-tune-swin_unet-t4_ee7f | equates_pilot | swin_unet | ox | 42 | ad0def0d28 | 3.672 | – | tuning trial 4 |
| 2026-09-29 | 20260929-102638_benchmark_equates_fold1-tune-gnn-t5_b39a | equates_pilot | gnn | ox | 42 | ee7f072056 | 3.663 | – | tuning trial 5 |
| 2026-09-29 | 20260929-102742_benchmark_equates_fold1-final-unet-multitask-s43_2fd8 | equates_pilot | unet | multitask | 43 | cf5c9f5694 | 3.503 | 2.943 | final, tuning trial 2 |
| 2026-09-29 | 20260929-103012_benchmark_equates_fold1-final-unet-multitask-s44_e393 | equates_pilot | unet | multitask | 44 | 7a3763389d | 3.494 | 2.755 | final, tuning trial 2 |
| 2026-09-29 | 20260929-102652_benchmark_equates_fold1-tune-swin-t6_e36f | equates_pilot | swin | ox | 42 | a12a31bf52 | 3.815 | – | tuning trial 6 |
| 2026-09-29 | 20260929-103008_benchmark_equates_fold1-tune-gnn-t6_9e94 | equates_pilot | gnn | ox | 42 | 2ea590e56f | 3.652 | – | tuning trial 6 |
| 2026-09-29 | 20260929-102855_benchmark_equates_fold1-tune-swin_unet-t5_95d4 | equates_pilot | swin_unet | ox | 42 | 8f8038f2ca | 3.783 | – | tuning trial 5 |
| 2026-09-29 | 20260929-103215_benchmark_equates_fold1-tune-mamba-t0_1c0f | equates_pilot | mamba | ox | 42 | 4756f86c13 | 3.358 | – | tuning trial 0 |
| 2026-09-29 | 20260929-103314_benchmark_equates_fold1-tune-swin-t7_4ebd | equates_pilot | swin | ox | 42 | 161b1e592c | 3.822 | – | tuning trial 7 |
| 2026-09-29 | 20260929-103425_benchmark_equates_fold1-tune-swin_unet-t6_10de | equates_pilot | swin_unet | ox | 42 | da956331d3 | 3.580 | – | tuning trial 6 |
| 2026-09-29 | 20260929-103810_benchmark_equates_fold1-tune-mamba-t1_f7e6 | equates_pilot | mamba | ox | 42 | f8a2d6908c | 3.340 | – | tuning trial 1 |
| 2026-09-29 | 20260929-103422_benchmark_equates_fold1-tune-gnn-t7_8f0f | equates_pilot | gnn | ox | 42 | 464541a8d1 | 3.613 | – | tuning trial 7 |
| 2026-09-29 | 20260929-103825_benchmark_equates_fold1-final-swin-ox-s42_54b8 | equates_pilot | swin | ox | 42 | bbbb82a2de | 3.760 | 2.907 | final, tuning trial 4 |
| 2026-09-29 | 20260929-103959_benchmark_equates_fold1-tune-swin_unet-t7_660e | equates_pilot | swin_unet | ox | 42 | 661044dbf3 | 3.687 | – | tuning trial 7 |
| 2026-09-29 | 20260929-104255_benchmark_equates_fold1-final-gnn-ox-s42_a1ac | equates_pilot | gnn | ox | 42 | 6599226d05 | 3.250 | 2.836 | final, tuning trial 2 |
| 2026-09-29 | 20260929-104112_benchmark_equates_fold1-tune-mamba-t2_62d6 | equates_pilot | mamba | ox | 42 | 179292d979 | 3.489 | – | tuning trial 2 |
| 2026-09-29 | 20260929-104424_benchmark_equates_fold1-final-swin-ox-s43_11df | equates_pilot | swin | ox | 43 | 2f61fe5471 | 3.714 | 3.007 | final, tuning trial 4 |
| 2026-09-29 | 20260929-104643_benchmark_equates_fold1-tune-mamba-t3_a7b4 | equates_pilot | mamba | ox | 42 | 5f690b791d | 3.450 | – | tuning trial 3 |
| 2026-09-29 | 20260929-104616_benchmark_equates_fold1-final-gnn-ox-s43_378b | equates_pilot | gnn | ox | 43 | 88e84b5167 | 3.242 | 2.737 | final, tuning trial 2 |
| 2026-09-29 | 20260929-104453_benchmark_equates_fold1-final-swin_unet-ox-s42_cf60 | equates_pilot | swin_unet | ox | 42 | da956331d3 | 3.580 | 2.784 | final, tuning trial 6 |
| 2026-09-29 | 20260929-105000_benchmark_equates_fold1-tune-mamba-t4_15ff | equates_pilot | mamba | ox | 42 | a0ad64b42f | 3.310 | – | tuning trial 4 |
| 2026-09-29 | 20260929-104737_benchmark_equates_fold1-final-swin-ox-s44_780a | equates_pilot | swin | ox | 44 | 3aab62190d | 3.865 | 3.403 | final, tuning trial 4 |
| 2026-09-29 | 20260929-105021_benchmark_equates_fold1-final-gnn-ox-s44_111f | equates_pilot | gnn | ox | 44 | f247da549e | 3.131 | 2.753 | final, tuning trial 2 |
| 2026-09-29 | 20260929-105300_benchmark_equates_fold1-tune-mamba-t5_889d | equates_pilot | mamba | ox | 42 | 3d0101222e | 3.530 | – | tuning trial 5 |
| 2026-09-29 | 20260929-105028_benchmark_equates_fold1-final-swin_unet-ox-s43_9675 | equates_pilot | swin_unet | ox | 43 | cd5f204587 | 3.643 | 2.738 | final, tuning trial 6 |
| 2026-09-29 | 20260929-105419_benchmark_equates_fold1-final-gnn-species-s42_24fc | equates_pilot | gnn | species | 42 | 2773abdeeb | 3.389 | 2.648 | final, tuning trial 2 |
| 2026-09-29 | 20260929-105314_benchmark_equates_fold1-final-swin-species-s42_3326 | equates_pilot | swin | species | 42 | 271e41ca61 | 3.978 | 2.994 | final, tuning trial 4 |
| 2026-09-29 | 20260929-105546_benchmark_equates_fold1-tune-mamba-t6_26bf | equates_pilot | mamba | ox | 42 | 9c768fad0b | 3.359 | – | tuning trial 6 |
| 2026-09-29 | 20260929-105952_benchmark_equates_fold1-tune-mamba-t7_1ae1 | equates_pilot | mamba | ox | 42 | 442166ec09 | 3.373 | – | tuning trial 7 |
| 2026-09-29 | 20260929-105735_benchmark_equates_fold1-final-swin_unet-ox-s44_b5e6 | equates_pilot | swin_unet | ox | 44 | ce92775559 | 3.608 | 2.788 | final, tuning trial 6 |
| 2026-09-29 | 20260929-105919_benchmark_equates_fold1-final-gnn-species-s43_4c30 | equates_pilot | gnn | species | 43 | 74aba1d2a2 | 3.413 | 2.562 | final, tuning trial 2 |
| 2026-09-29 | 20260929-110241_benchmark_equates_fold1-final-mamba-ox-s42_3199 | equates_pilot | mamba | ox | 42 | a0ad64b42f | 3.363 | 2.761 | final, tuning trial 4 |
| 2026-09-29 | 20260929-105936_benchmark_equates_fold1-final-swin-species-s43_0fe1 | equates_pilot | swin | species | 43 | e941a5d107 | 3.718 | 3.098 | final, tuning trial 4 |
| 2026-09-29 | 20260929-110240_benchmark_equates_fold1-final-swin_unet-species-s42_d07f | equates_pilot | swin_unet | species | 42 | e1daae8e5c | 3.632 | 2.979 | final, tuning trial 6 |
| 2026-09-29 | 20260929-110411_benchmark_equates_fold1-final-gnn-species-s44_9ca8 | equates_pilot | gnn | species | 44 | 64aa2d12dc | 3.408 | 2.645 | final, tuning trial 2 |
| 2026-09-29 | 20260929-110540_benchmark_equates_fold1-final-mamba-ox-s43_2e8b | equates_pilot | mamba | ox | 43 | 0a7dfe34c7 | 3.360 | 2.657 | final, tuning trial 4 |
| 2026-09-29 | 20260929-110604_benchmark_equates_fold1-final-swin-species-s44_011e | equates_pilot | swin | species | 44 | 02580d763a | 3.952 | 3.325 | final, tuning trial 4 |
| 2026-09-29 | 20260929-111022_benchmark_equates_fold1-final-mamba-ox-s44_e68a | equates_pilot | mamba | ox | 44 | 5dac09a9af | 3.280 | 3.032 | final, tuning trial 4 |
| 2026-09-29 | 20260929-110856_benchmark_equates_fold1-final-gnn-multitask-s42_f1f3 | equates_pilot | gnn | multitask | 42 | 5bfa1b954a | 3.296 | 2.598 | final, tuning trial 2 |
| 2026-09-29 | 20260929-110824_benchmark_equates_fold1-final-swin_unet-species-s43_c1dd | equates_pilot | swin_unet | species | 43 | 4a0f40f2ba | 3.650 | 2.808 | final, tuning trial 6 |
| 2026-09-29 | 20260929-111211_benchmark_equates_fold1-final-swin-multitask-s42_4581 | equates_pilot | swin | multitask | 42 | 181efb98ad | 3.747 | 2.747 | final, tuning trial 4 |
| 2026-09-29 | 20260929-111330_benchmark_equates_fold1-final-gnn-multitask-s43_171f | equates_pilot | gnn | multitask | 43 | 790b043009 | 3.298 | 2.613 | final, tuning trial 2 |
| 2026-09-29 | 20260929-111325_benchmark_equates_fold1-final-mamba-species-s42_1cd0 | equates_pilot | mamba | species | 42 | d0f3f2539e | 3.601 | 2.630 | final, tuning trial 4 |
| 2026-09-29 | 20260929-111527_benchmark_equates_fold1-final-swin_unet-species-s44_f51a | equates_pilot | swin_unet | species | 44 | 45afd8ded6 | 3.501 | 2.897 | final, tuning trial 6 |
| 2026-09-29 | 20260929-111823_benchmark_equates_fold1-final-gnn-multitask-s44_55b7 | equates_pilot | gnn | multitask | 44 | c618e872f6 | 3.149 | 2.640 | final, tuning trial 2 |
| 2026-09-29 | 20260929-111849_benchmark_equates_fold1-final-mamba-species-s43_6a16 | equates_pilot | mamba | species | 43 | bd618fbccf | 3.377 | 2.678 | final, tuning trial 4 |
| 2026-09-29 | 20260929-111752_benchmark_equates_fold1-final-swin-multitask-s43_3756 | equates_pilot | swin | multitask | 43 | df936425ec | 3.945 | 3.107 | final, tuning trial 4 |
| 2026-09-29 | 20260929-112403_benchmark_equates_fold2-tune-unet-t1_6611 | equates_pilot | unet | ox | 42 | ace72e6e99 | 3.206 | – | tuning trial 1 |
| 2026-09-29 | 20260929-112052_benchmark_equates_fold1-final-swin_unet-multitask-s42_0573 | equates_pilot | swin_unet | multitask | 42 | a92c4d6f67 | 3.550 | 2.805 | final, tuning trial 6 |
| 2026-09-29 | 20260929-112544_benchmark_equates_fold2-tune-unet-t2_78a9 | equates_pilot | unet | ox | 42 | 79e8d775df | 3.239 | – | tuning trial 2 |
| 2026-09-29 | 20260929-112332_benchmark_equates_fold1-final-mamba-species-s44_2237 | equates_pilot | mamba | species | 44 | 664780e173 | 3.443 | 2.586 | final, tuning trial 4 |
| 2026-09-29 | 20260929-112707_benchmark_equates_fold2-tune-unet-t3_0911 | equates_pilot | unet | ox | 42 | 6891e5ed8a | 3.291 | – | tuning trial 3 |
| 2026-09-29 | 20260929-112823_benchmark_equates_fold2-tune-unet-t4_e347 | equates_pilot | unet | ox | 42 | bfb4cab7b2 | 3.196 | – | tuning trial 4 |
| 2026-09-29 | 20260929-112441_benchmark_equates_fold1-final-swin-multitask-s44_58c7 | equates_pilot | swin | multitask | 44 | 18c9b8495b | 4.085 | 3.415 | final, tuning trial 4 |
| 2026-09-29 | 20260929-113016_benchmark_equates_fold2-tune-unet-t5_6ab7 | equates_pilot | unet | ox | 42 | 7d04ff7bef | 3.201 | – | tuning trial 5 |
| 2026-09-29 | 20260929-112813_benchmark_equates_fold1-final-mamba-multitask-s42_de51 | equates_pilot | mamba | multitask | 42 | 57bace9749 | 3.424 | 2.659 | final, tuning trial 4 |
| 2026-09-29 | 20260929-113123_benchmark_equates_fold2-tune-fno-t5_5961 | equates_pilot | fno | ox | 42 | ca1731ead0 | 3.434 | – | tuning trial 5 |
| 2026-09-29 | 20260929-112637_benchmark_equates_fold1-final-swin_unet-multitask-s43_90de | equates_pilot | swin_unet | multitask | 43 | 4c282089b2 | 3.662 | 2.874 | final, tuning trial 6 |
| 2026-09-29 | 20260929-113150_benchmark_equates_fold2-tune-unet-t6_dd4b | equates_pilot | unet | ox | 42 | 2c1dbfe0fb | 3.158 | – | tuning trial 6 |
| 2026-09-29 | 20260929-113222_benchmark_equates_fold2-tune-fno-t6_0551 | equates_pilot | fno | ox | 42 | f130326e00 | 3.281 | – | tuning trial 6 |
| 2026-09-29 | 20260929-113328_benchmark_equates_fold2-tune-fno-t7_e518 | equates_pilot | fno | ox | 42 | d0ebecbbab | 3.382 | – | tuning trial 7 |
| 2026-09-29 | 20260929-113321_benchmark_equates_fold2-tune-unet-t7_5b8f | equates_pilot | unet | ox | 42 | f90b7225b3 | 3.217 | – | tuning trial 7 |
| 2026-09-29 | 20260929-113422_benchmark_equates_fold2-final-fno-ox-s42_6518 | equates_pilot | fno | ox | 42 | b0756e93b1 | 3.241 | 3.894 | final, tuning trial 4 |
| 2026-09-29 | 20260929-113450_benchmark_equates_fold2-final-unet-ox-s42_9228 | equates_pilot | unet | ox | 42 | 2c1dbfe0fb | 3.155 | 3.530 | final, tuning trial 6 |
| 2026-09-29 | 20260929-113542_benchmark_equates_fold2-final-fno-ox-s43_a9c7 | equates_pilot | fno | ox | 43 | 2ed0737e0a | 3.415 | 3.692 | final, tuning trial 4 |
| 2026-09-29 | 20260929-113218_benchmark_equates_fold1-final-mamba-multitask-s43_93f3 | equates_pilot | mamba | multitask | 43 | ced580d1b8 | 3.473 | 2.669 | final, tuning trial 4 |
| 2026-09-29 | 20260929-113230_benchmark_equates_fold1-final-swin_unet-multitask-s44_dfa4 | equates_pilot | swin_unet | multitask | 44 | 215659bfc4 | 3.562 | 2.963 | final, tuning trial 6 |
| 2026-09-29 | 20260929-113647_benchmark_equates_fold2-final-fno-ox-s44_bbed | equates_pilot | fno | ox | 44 | 6260ac53a3 | 3.350 | 4.190 | final, tuning trial 4 |
| 2026-09-29 | 20260929-113628_benchmark_equates_fold2-final-unet-ox-s43_3ed3 | equates_pilot | unet | ox | 43 | c4da3131a0 | 3.120 | 3.687 | final, tuning trial 6 |
| 2026-09-29 | 20260929-113759_benchmark_equates_fold2-final-fno-species-s42_b2ef | equates_pilot | fno | species | 42 | a325c5325c | 3.224 | 3.732 | final, tuning trial 4 |
| 2026-09-29 | 20260929-113802_benchmark_equates_fold2-final-unet-ox-s44_d141 | equates_pilot | unet | ox | 44 | 739de5ab3a | 3.243 | 3.573 | final, tuning trial 6 |
| 2026-09-29 | 20260929-113923_benchmark_equates_fold2-final-fno-species-s43_02a5 | equates_pilot | fno | species | 43 | 5e06819e55 | 3.252 | 3.803 | final, tuning trial 4 |
| 2026-09-29 | 20260929-113932_benchmark_equates_fold2-final-unet-species-s42_bad8 | equates_pilot | unet | species | 42 | 853fc91fe4 | 3.257 | 3.497 | final, tuning trial 6 |
| 2026-09-29 | 20260929-113713_benchmark_equates_fold1-final-mamba-multitask-s44_e059 | equates_pilot | mamba | multitask | 44 | 0d69b4ad78 | 3.370 | 2.588 | final, tuning trial 4 |
| 2026-09-29 | 20260929-114045_benchmark_equates_fold2-final-fno-species-s44_93e8 | equates_pilot | fno | species | 44 | 18aef65bcb | 3.228 | 3.680 | final, tuning trial 4 |
| 2026-09-29 | 20260929-114116_benchmark_equates_fold2-final-unet-species-s43_b319 | equates_pilot | unet | species | 43 | 71443fa64a | 3.187 | 3.530 | final, tuning trial 6 |
| 2026-09-29 | 20260929-114206_benchmark_equates_fold2-final-fno-multitask-s42_2010 | equates_pilot | fno | multitask | 42 | 9c866f1bc9 | 3.251 | 3.897 | final, tuning trial 4 |
| 2026-09-29 | 20260929-113830_benchmark_equates_fold2-tune-swin-t1_dde2 | equates_pilot | swin | ox | 42 | 8e5f34b2ec | 3.684 | – | tuning trial 1 |
| 2026-09-29 | 20260929-114325_benchmark_equates_fold2-final-fno-multitask-s43_1775 | equates_pilot | fno | multitask | 43 | 57e52ff4b7 | 3.272 | 3.823 | final, tuning trial 4 |
| 2026-09-29 | 20260929-114318_benchmark_equates_fold2-final-unet-species-s44_e9ea | equates_pilot | unet | species | 44 | d7cd72081b | 3.300 | 3.606 | final, tuning trial 6 |
| 2026-09-29 | 20260929-114452_benchmark_equates_fold2-final-fno-multitask-s44_9fda | equates_pilot | fno | multitask | 44 | efd32fb72f | 3.272 | 3.743 | final, tuning trial 4 |
| 2026-09-29 | 20260929-114149_benchmark_equates_fold2-tune-swin_unet-t1_eb41 | equates_pilot | swin_unet | ox | 42 | 134edd3455 | 3.555 | – | tuning trial 1 |
| 2026-09-29 | 20260929-114517_benchmark_equates_fold2-final-unet-multitask-s42_c60a | equates_pilot | unet | multitask | 42 | 2671078c99 | 3.176 | 3.424 | final, tuning trial 6 |
| 2026-09-29 | 20260929-114403_benchmark_equates_fold2-tune-swin-t2_34f5 | equates_pilot | swin | ox | 42 | 545f813e74 | 3.789 | – | tuning trial 2 |
| 2026-09-29 | 20260929-114710_benchmark_equates_fold2-final-unet-multitask-s43_dbd9 | equates_pilot | unet | multitask | 43 | ccb031d508 | 3.200 | 3.670 | final, tuning trial 6 |
| 2026-09-29 | 20260929-114906_benchmark_equates_fold2-final-unet-multitask-s44_355d | equates_pilot | unet | multitask | 44 | 8036035c00 | 3.180 | 3.694 | final, tuning trial 6 |
| 2026-09-29 | 20260929-114837_benchmark_equates_fold2-tune-swin-t3_823a | equates_pilot | swin | ox | 42 | d965eb6f87 | 3.857 | – | tuning trial 3 |
| 2026-09-29 | 20260929-114703_benchmark_equates_fold2-tune-swin_unet-t2_25dd | equates_pilot | swin_unet | ox | 42 | 46d889f517 | 3.564 | – | tuning trial 2 |
| 2026-09-29 | 20260929-114716_benchmark_equates_fold2-tune-gnn-t0_25ee | equates_pilot | gnn | ox | 42 | 769e4bde21 | 3.113 | – | tuning trial 0 |
| 2026-09-29 | 20260929-115312_benchmark_equates_fold2-tune-swin-t4_8dab | equates_pilot | swin | ox | 42 | 3c06898cbc | 3.473 | – | tuning trial 4 |
| 2026-09-29 | 20260929-115349_benchmark_equates_fold2-tune-swin_unet-t3_0cff | equates_pilot | swin_unet | ox | 42 | 881e697aed | 3.553 | – | tuning trial 3 |
| 2026-09-29 | 20260929-115442_benchmark_equates_fold2-tune-gnn-t1_e4e5 | equates_pilot | gnn | ox | 42 | af52067e6f | 2.997 | – | tuning trial 1 |
| 2026-09-29 | 20260929-115915_benchmark_equates_fold2-tune-swin-t5_c41c | equates_pilot | swin | ox | 42 | 991fd669f5 | 3.606 | – | tuning trial 5 |
| 2026-09-29 | 20260929-120036_benchmark_equates_fold2-tune-gnn-t2_9b6c | equates_pilot | gnn | ox | 42 | a7df23f6fa | 2.890 | – | tuning trial 2 |
| 2026-09-29 | 20260929-115120_benchmark_equates_fold2-tune-mamba-t2_1861 | equates_pilot | mamba | ox | 42 | 8dc488489e | 2.808 | – | tuning trial 2 |
| 2026-09-29 | 20260929-115951_benchmark_equates_fold2-tune-swin_unet-t4_c74e | equates_pilot | swin_unet | ox | 42 | 5e2afb7fa3 | 3.249 | – | tuning trial 4 |
| 2026-09-29 | 20260929-120550_benchmark_equates_fold2-tune-mamba-t3_d076 | equates_pilot | mamba | ox | 42 | 39683a2609 | 3.028 | – | tuning trial 3 |
| 2026-09-29 | 20260929-120523_benchmark_equates_fold2-tune-swin-t6_3071 | equates_pilot | swin | ox | 42 | 2311751418 | 3.580 | – | tuning trial 6 |
| 2026-09-29 | 20260929-120534_benchmark_equates_fold2-tune-gnn-t3_869f | equates_pilot | gnn | ox | 42 | fef71dc3ba | 3.017 | – | tuning trial 3 |
| 2026-09-29 | 20260929-120759_benchmark_equates_fold2-tune-swin_unet-t5_90d7 | equates_pilot | swin_unet | ox | 42 | baeeb16941 | 3.358 | – | tuning trial 5 |
| 2026-09-29 | 20260929-120948_benchmark_equates_fold2-tune-mamba-t4_a4e8 | equates_pilot | mamba | ox | 42 | 8657109b02 | 2.990 | – | tuning trial 4 |
| 2026-09-29 | 20260929-121038_benchmark_equates_fold2-tune-swin-t7_74cb | equates_pilot | swin | ox | 42 | e835019b22 | 3.594 | – | tuning trial 7 |
| 2026-09-29 | 20260929-121413_benchmark_equates_fold2-tune-mamba-t5_743e | equates_pilot | mamba | ox | 42 | 6521d36a55 | 3.265 | – | tuning trial 5 |
| 2026-09-29 | 20260929-121213_benchmark_equates_fold2-tune-gnn-t4_6785 | equates_pilot | gnn | ox | 42 | 06afab166c | 3.274 | – | tuning trial 4 |
| 2026-09-29 | 20260929-121316_benchmark_equates_fold2-tune-swin_unet-t6_1263 | equates_pilot | swin_unet | ox | 42 | fe5aec0894 | 3.456 | – | tuning trial 6 |
| 2026-09-29 | 20260929-121443_benchmark_equates_fold2-final-swin-ox-s42_9421 | equates_pilot | swin | ox | 42 | 3c06898cbc | 3.473 | 4.859 | final, tuning trial 4 |
| 2026-09-29 | 20260929-121624_benchmark_equates_fold2-tune-mamba-t6_37d4 | equates_pilot | mamba | ox | 42 | e3aef633ed | 2.938 | – | tuning trial 6 |
| 2026-09-29 | 20260929-121739_benchmark_equates_fold2-tune-gnn-t5_baef | equates_pilot | gnn | ox | 42 | 050f118705 | 3.217 | – | tuning trial 5 |
| 2026-09-29 | 20260929-121856_benchmark_equates_fold2-tune-swin_unet-t7_9926 | equates_pilot | swin_unet | ox | 42 | da691a0985 | 3.395 | – | tuning trial 7 |
| 2026-09-29 | 20260929-122204_benchmark_equates_fold2-tune-mamba-t7_5b4f | equates_pilot | mamba | ox | 42 | 6182b10fa1 | 3.131 | – | tuning trial 7 |
| 2026-09-29 | 20260929-122048_benchmark_equates_fold2-final-swin-ox-s43_e64a | equates_pilot | swin | ox | 43 | a3eab1eaec | 3.547 | 4.797 | final, tuning trial 4 |
| 2026-09-29 | 20260929-122233_benchmark_equates_fold2-tune-gnn-t6_59cb | equates_pilot | gnn | ox | 42 | 8ac195d9eb | 3.192 | – | tuning trial 6 |
| 2026-09-29 | 20260929-122549_benchmark_equates_fold2-final-swin-ox-s44_5b0c | equates_pilot | swin | ox | 44 | 70ed82557c | 3.458 | 4.887 | final, tuning trial 4 |
| 2026-09-29 | 20260929-122344_benchmark_equates_fold2-final-swin_unet-ox-s42_44bc | equates_pilot | swin_unet | ox | 42 | 5e2afb7fa3 | 3.249 | 3.517 | final, tuning trial 4 |
| 2026-09-29 | 20260929-122747_benchmark_equates_fold2-tune-gnn-t7_5388 | equates_pilot | gnn | ox | 42 | ef7156eddd | 3.108 | – | tuning trial 7 |
| 2026-09-29 | 20260929-123052_benchmark_equates_fold2-final-swin-species-s42_defc | equates_pilot | swin | species | 42 | 24c6fa4c01 | 3.786 | 5.470 | final, tuning trial 4 |
| 2026-09-29 | 20260929-122450_benchmark_equates_fold2-final-mamba-ox-s42_6e0f | equates_pilot | mamba | ox | 42 | 8dc488489e | 2.807 | 3.453 | final, tuning trial 2 |
| 2026-09-29 | 20260929-123152_benchmark_equates_fold2-final-swin_unet-ox-s43_b8ad | equates_pilot | swin_unet | ox | 43 | 1a5b83db23 | 3.340 | 3.583 | final, tuning trial 4 |
| 2026-09-29 | 20260929-123507_benchmark_equates_fold2-final-gnn-ox-s42_35bf | equates_pilot | gnn | ox | 42 | a7df23f6fa | 2.889 | 3.424 | final, tuning trial 2 |
| 2026-09-29 | 20260929-123642_benchmark_equates_fold2-final-swin-species-s43_b8a7 | equates_pilot | swin | species | 43 | f204917909 | 3.752 | 5.224 | final, tuning trial 4 |
| 2026-09-29 | 20260929-124008_benchmark_equates_fold2-final-gnn-ox-s43_5162 | equates_pilot | gnn | ox | 43 | 66a815e628 | 2.842 | 3.430 | final, tuning trial 2 |
| 2026-09-29 | 20260929-123936_benchmark_equates_fold2-final-mamba-ox-s43_605b | equates_pilot | mamba | ox | 43 | 088c25a16d | 3.001 | 3.384 | final, tuning trial 2 |
| 2026-09-29 | 20260929-123944_benchmark_equates_fold2-final-swin_unet-ox-s44_d0b9 | equates_pilot | swin_unet | ox | 44 | cf87bc71b2 | 3.272 | 3.644 | final, tuning trial 4 |
| 2026-09-29 | 20260929-124249_benchmark_equates_fold2-final-swin-species-s44_912e | equates_pilot | swin | species | 44 | cbf52afd10 | 3.743 | 5.172 | final, tuning trial 4 |
| 2026-09-29 | 20260929-124513_benchmark_equates_fold2-final-gnn-ox-s44_be3c | equates_pilot | gnn | ox | 44 | d031e6c8ac | 2.891 | 3.409 | final, tuning trial 2 |
| 2026-09-29 | 20260929-124939_benchmark_equates_fold2-final-gnn-species-s42_0757 | equates_pilot | gnn | species | 42 | 36b73937ea | 2.938 | 3.411 | final, tuning trial 2 |
| 2026-09-29 | 20260929-124752_benchmark_equates_fold2-final-swin_unet-species-s42_10b8 | equates_pilot | swin_unet | species | 42 | 4b1b98433d | 3.484 | 3.699 | final, tuning trial 4 |
| 2026-09-29 | 20260929-124919_benchmark_equates_fold2-final-swin-multitask-s42_50d0 | equates_pilot | swin | multitask | 42 | 66e812a856 | 3.555 | 4.970 | final, tuning trial 4 |
| 2026-09-29 | 20260929-125412_benchmark_equates_fold2-final-gnn-species-s43_ace6 | equates_pilot | gnn | species | 43 | d1f5b769d9 | 2.954 | 3.392 | final, tuning trial 2 |
| 2026-09-29 | 20260929-125443_benchmark_equates_fold2-final-swin_unet-species-s43_3d28 | equates_pilot | swin_unet | species | 43 | 813598def6 | 3.420 | 3.618 | final, tuning trial 4 |
| 2026-09-29 | 20260929-124722_benchmark_equates_fold2-final-mamba-ox-s44_9c52 | equates_pilot | mamba | ox | 44 | f94dc4d35d | 2.876 | 3.497 | final, tuning trial 2 |
| 2026-09-29 | 20260929-125831_benchmark_equates_fold2-final-gnn-species-s44_7e24 | equates_pilot | gnn | species | 44 | b1dd131804 | 2.948 | 3.639 | final, tuning trial 2 |
| 2026-09-29 | 20260929-125521_benchmark_equates_fold2-final-swin-multitask-s43_d4ed | equates_pilot | swin | multitask | 43 | 26a48b7e38 | 3.369 | 4.924 | final, tuning trial 4 |
| 2026-09-29 | 20260929-130339_benchmark_equates_fold2-final-gnn-multitask-s42_c2c3 | equates_pilot | gnn | multitask | 42 | 03798c26c8 | 2.926 | 3.671 | final, tuning trial 2 |
| 2026-09-29 | 20260929-130158_benchmark_equates_fold2-final-swin_unet-species-s44_3ce3 | equates_pilot | swin_unet | species | 44 | d46d9f2093 | 3.462 | 3.797 | final, tuning trial 4 |
| 2026-09-29 | 20260929-130817_benchmark_equates_fold2-final-gnn-multitask-s43_045f | equates_pilot | gnn | multitask | 43 | 57279b9259 | 2.904 | 3.508 | final, tuning trial 2 |
| 2026-09-29 | 20260929-130446_benchmark_equates_fold2-final-swin-multitask-s44_6b7e | equates_pilot | swin | multitask | 44 | 9d9f93b091 | 3.515 | 4.841 | final, tuning trial 4 |
| 2026-09-29 | 20260929-130858_benchmark_equates_fold2-final-swin_unet-multitask-s42_7590 | equates_pilot | swin_unet | multitask | 42 | b3b351cc0b | 3.527 | 3.615 | final, tuning trial 4 |
| 2026-09-29 | 20260929-131549_benchmark_equates_fold3-tune-unet-t7_7b13 | equates_pilot | unet | ox | 42 | 21f7c97d7b | 4.370 | – | tuning trial 7 |
| 2026-09-29 | 20260929-131241_benchmark_equates_fold2-final-gnn-multitask-s44_7da6 | equates_pilot | gnn | multitask | 44 | 4d42e35806 | 2.885 | 3.479 | final, tuning trial 2 |
| 2026-09-29 | 20260929-130238_benchmark_equates_fold2-final-mamba-species-s42_2082 | equates_pilot | mamba | species | 42 | 4a6db1cede | 2.845 | 3.309 | final, tuning trial 2 |
| 2026-09-29 | 20260929-131751_benchmark_equates_fold3-final-fno-ox-s44_6087 | equates_pilot | fno | ox | 44 | cba9b5e63d | 4.849 | 3.509 | final, tuning trial 6 |
| 2026-09-29 | 20260929-131719_benchmark_equates_fold3-final-unet-ox-s42_ba87 | equates_pilot | unet | ox | 42 | e0750938a5 | 4.476 | 3.418 | final, tuning trial 2 |
| 2026-09-29 | 20260929-131852_benchmark_equates_fold3-final-unet-ox-s43_0130 | equates_pilot | unet | ox | 43 | 8866aeb4e6 | 4.670 | 3.575 | final, tuning trial 2 |
| 2026-09-29 | 20260929-131846_benchmark_equates_fold3-final-fno-species-s42_f73e | equates_pilot | fno | species | 42 | c0f6db52f7 | 4.641 | 3.551 | final, tuning trial 6 |
| 2026-09-29 | 20260929-131956_benchmark_equates_fold3-final-fno-species-s43_0653 | equates_pilot | fno | species | 43 | 15087875e3 | 4.777 | 3.611 | final, tuning trial 6 |
| 2026-09-29 | 20260929-131955_benchmark_equates_fold3-final-unet-ox-s44_c8b6 | equates_pilot | unet | ox | 44 | 2c3604ebfe | 4.525 | 3.421 | final, tuning trial 2 |
| 2026-09-29 | 20260929-131514_benchmark_equates_fold2-final-swin_unet-multitask-s43_2261 | equates_pilot | swin_unet | multitask | 43 | 73479e86b6 | 3.480 | 3.857 | final, tuning trial 4 |
| 2026-09-29 | 20260929-132057_benchmark_equates_fold3-final-fno-species-s44_4b8b | equates_pilot | fno | species | 44 | 1edfdc59fa | 4.797 | 3.669 | final, tuning trial 6 |
| 2026-09-29 | 20260929-132206_benchmark_equates_fold3-final-fno-multitask-s42_be60 | equates_pilot | fno | multitask | 42 | 7bd3f85c38 | 5.175 | 3.767 | final, tuning trial 6 |
| 2026-09-29 | 20260929-132101_benchmark_equates_fold3-final-unet-species-s42_2e86 | equates_pilot | unet | species | 42 | 0569c8e4dc | 4.262 | 3.329 | final, tuning trial 2 |
| 2026-09-29 | 20260929-132312_benchmark_equates_fold3-final-fno-multitask-s43_39fb | equates_pilot | fno | multitask | 43 | 184c3d3d8f | 4.771 | 3.583 | final, tuning trial 6 |
| 2026-09-29 | 20260929-132417_benchmark_equates_fold3-final-fno-multitask-s44_25b5 | equates_pilot | fno | multitask | 44 | d8fa712ad0 | 4.733 | 3.590 | final, tuning trial 6 |
| 2026-09-29 | 20260929-132350_benchmark_equates_fold3-final-unet-species-s43_0c56 | equates_pilot | unet | species | 43 | 591bed4a4b | 4.701 | 3.548 | final, tuning trial 2 |
| 2026-09-29 | 20260929-132522_benchmark_equates_fold3-final-unet-species-s44_dcb2 | equates_pilot | unet | species | 44 | b7932060cb | 4.744 | 3.519 | final, tuning trial 2 |
| 2026-09-29 | 20260929-132108_benchmark_equates_fold2-final-swin_unet-multitask-s44_0c23 | equates_pilot | swin_unet | multitask | 44 | bd534fffd7 | 3.471 | 3.765 | final, tuning trial 4 |
| 2026-09-29 | 20260929-132603_benchmark_equates_fold3-tune-swin-t3_b101 | equates_pilot | swin | ox | 42 | 2ecfc82bb0 | 5.609 | – | tuning trial 3 |
| 2026-09-29 | 20260929-132702_benchmark_equates_fold3-final-unet-multitask-s42_a85a | equates_pilot | unet | multitask | 42 | 4ab86f390e | 4.139 | 3.192 | final, tuning trial 2 |
| 2026-09-29 | 20260929-133043_benchmark_equates_fold3-final-unet-multitask-s43_734f | equates_pilot | unet | multitask | 43 | 419264bb8a | 4.531 | 3.345 | final, tuning trial 2 |
| 2026-09-29 | 20260929-133016_benchmark_equates_fold3-tune-swin-t4_661c | equates_pilot | swin | ox | 42 | dade9e838d | 4.904 | – | tuning trial 4 |
| 2026-09-29 | 20260929-132829_benchmark_equates_fold3-tune-swin_unet-t2_757b | equates_pilot | swin_unet | ox | 42 | c8c08e83b5 | 4.958 | – | tuning trial 2 |
| 2026-09-29 | 20260929-131844_benchmark_equates_fold2-final-mamba-species-s43_e835 | equates_pilot | mamba | species | 43 | 4d28246fb9 | 2.868 | 3.362 | final, tuning trial 2 |
| 2026-09-29 | 20260929-133358_benchmark_equates_fold3-final-unet-multitask-s44_b632 | equates_pilot | unet | multitask | 44 | e210e4ebad | 4.317 | 3.300 | final, tuning trial 2 |
| 2026-09-29 | 20260929-133439_benchmark_equates_fold3-tune-swin-t5_c924 | equates_pilot | swin | ox | 42 | 6e61440521 | 5.078 | – | tuning trial 5 |
| 2026-09-29 | 20260929-133444_benchmark_equates_fold3-tune-swin_unet-t3_8cb3 | equates_pilot | swin_unet | ox | 42 | c464f153f0 | 4.856 | – | tuning trial 3 |
| 2026-09-29 | 20260929-133903_benchmark_equates_fold3-tune-swin-t6_554f | equates_pilot | swin | ox | 42 | 3523a0f302 | 4.993 | – | tuning trial 6 |
| 2026-09-29 | 20260929-133816_benchmark_equates_fold3-tune-gnn-t1_166c | equates_pilot | gnn | ox | 42 | f54f3be0ed | 3.942 | – | tuning trial 1 |
| 2026-09-29 | 20260929-133947_benchmark_equates_fold3-tune-swin_unet-t4_1976 | equates_pilot | swin_unet | ox | 42 | 8ccde142e3 | 4.524 | – | tuning trial 4 |
| 2026-09-29 | 20260929-134307_benchmark_equates_fold3-tune-swin-t7_cf85 | equates_pilot | swin | ox | 42 | ba7a8ff4dc | 5.122 | – | tuning trial 7 |
| 2026-09-29 | 20260929-134432_benchmark_equates_fold3-tune-gnn-t2_76be | equates_pilot | gnn | ox | 42 | 4d9046b6fa | 3.698 | – | tuning trial 2 |
| 2026-09-29 | 20260929-133606_benchmark_equates_fold2-final-mamba-species-s44_20ff | equates_pilot | mamba | species | 44 | ba2ae620ec | 2.916 | 3.308 | final, tuning trial 2 |
| 2026-09-29 | 20260929-134629_benchmark_equates_fold3-tune-swin_unet-t5_9f24 | equates_pilot | swin_unet | ox | 42 | b954afc9c8 | 4.509 | – | tuning trial 5 |
| 2026-09-29 | 20260929-134759_benchmark_equates_fold3-final-swin-ox-s42_4672 | equates_pilot | swin | ox | 42 | dade9e838d | 4.904 | 3.460 | final, tuning trial 4 |
| 2026-09-29 | 20260929-134910_benchmark_equates_fold3-tune-gnn-t3_935b | equates_pilot | gnn | ox | 42 | 8e91f2e052 | 3.963 | – | tuning trial 3 |
| 2026-09-29 | 20260929-135033_benchmark_equates_fold3-tune-swin_unet-t6_01bd | equates_pilot | swin_unet | ox | 42 | 15c4bf84da | 4.362 | – | tuning trial 6 |
| 2026-09-29 | 20260929-135224_benchmark_equates_fold3-final-swin-ox-s43_eece | equates_pilot | swin | ox | 43 | 4c454f2f5c | 5.112 | 3.521 | final, tuning trial 4 |
| 2026-09-29 | 20260929-135445_benchmark_equates_fold3-tune-gnn-t4_dae9 | equates_pilot | gnn | ox | 42 | 60955a374c | 5.188 | – | tuning trial 4 |
| 2026-09-29 | 20260929-135543_benchmark_equates_fold3-tune-swin_unet-t7_d9df | equates_pilot | swin_unet | ox | 42 | 7f18feca9b | 4.443 | – | tuning trial 7 |
| 2026-09-29 | 20260929-135747_benchmark_equates_fold3-final-swin-ox-s44_27da | equates_pilot | swin | ox | 44 | 129eee0940 | 5.073 | 3.695 | final, tuning trial 4 |
| 2026-09-29 | 20260929-135934_benchmark_equates_fold3-tune-gnn-t5_2945 | equates_pilot | gnn | ox | 42 | 5e3740c235 | 4.923 | – | tuning trial 5 |
| 2026-09-29 | 20260929-140018_benchmark_equates_fold3-final-swin_unet-ox-s42_6376 | equates_pilot | swin_unet | ox | 42 | 15c4bf84da | 4.362 | 3.641 | final, tuning trial 6 |
| 2026-09-29 | 20260929-134928_benchmark_equates_fold2-final-mamba-multitask-s42_2cf6 | equates_pilot | mamba | multitask | 42 | 3f0df007d1 | 2.799 | 3.565 | final, tuning trial 2 |
| 2026-09-29 | 20260929-140150_benchmark_equates_fold3-final-swin-species-s42_ee71 | equates_pilot | swin | species | 42 | ede37afedd | 5.413 | 3.701 | final, tuning trial 4 |
| 2026-09-29 | 20260929-140332_benchmark_equates_fold3-tune-gnn-t6_b63f | equates_pilot | gnn | ox | 42 | 036de2cda4 | 4.849 | – | tuning trial 6 |
| 2026-09-29 | 20260929-140533_benchmark_equates_fold3-final-swin_unet-ox-s43_460b | equates_pilot | swin_unet | ox | 43 | ed8e2d7c85 | 4.378 | 3.652 | final, tuning trial 6 |
| 2026-09-29 | 20260929-140651_benchmark_equates_fold3-final-swin-species-s43_cf7b | equates_pilot | swin | species | 43 | faa6931b7c | 5.749 | 3.785 | final, tuning trial 4 |
| 2026-09-29 | 20260929-140945_benchmark_equates_fold3-final-swin_unet-ox-s44_6626 | equates_pilot | swin_unet | ox | 44 | 128657475b | 4.293 | 3.583 | final, tuning trial 6 |
| 2026-09-29 | 20260929-140818_benchmark_equates_fold3-tune-gnn-t7_34bd | equates_pilot | gnn | ox | 42 | a3567637a6 | 4.897 | – | tuning trial 7 |
| 2026-09-29 | 20260929-141142_benchmark_equates_fold3-final-swin-species-s44_1b6e | equates_pilot | swin | species | 44 | 5fe5db71e4 | 5.471 | 3.682 | final, tuning trial 4 |
| 2026-09-29 | 20260929-141430_benchmark_equates_fold3-final-gnn-ox-s42_9c1f | equates_pilot | gnn | ox | 42 | 4d9046b6fa | 3.698 | 3.050 | final, tuning trial 2 |
| 2026-09-29 | 20260929-141417_benchmark_equates_fold3-final-swin_unet-species-s42_0eb8 | equates_pilot | swin_unet | species | 42 | 12997891d4 | 4.452 | 3.653 | final, tuning trial 6 |
| 2026-09-29 | 20260929-141628_benchmark_equates_fold3-final-swin-multitask-s42_de6d | equates_pilot | swin | multitask | 42 | eb3d8c94a9 | 5.252 | 3.587 | final, tuning trial 4 |
| 2026-09-29 | 20260929-141907_benchmark_equates_fold3-final-gnn-ox-s43_5446 | equates_pilot | gnn | ox | 43 | 5a25357d1c | 3.667 | 3.047 | final, tuning trial 2 |
| 2026-09-29 | 20260929-141959_benchmark_equates_fold3-final-swin_unet-species-s43_d4ef | equates_pilot | swin_unet | species | 43 | 2d636a3040 | 4.430 | 3.633 | final, tuning trial 6 |
| 2026-09-29 | 20260929-142108_benchmark_equates_fold3-final-swin-multitask-s43_65de | equates_pilot | swin | multitask | 43 | 526f0918d8 | 5.954 | 3.770 | final, tuning trial 4 |
| 2026-09-29 | 20260929-140633_benchmark_equates_fold2-final-mamba-multitask-s43_3057 | equates_pilot | mamba | multitask | 43 | 8c44104985 | 2.880 | 3.546 | final, tuning trial 2 |
| 2026-09-29 | 20260929-142353_benchmark_equates_fold3-final-gnn-ox-s44_bf4a | equates_pilot | gnn | ox | 44 | 9a50819ce2 | 3.673 | 3.138 | final, tuning trial 2 |
| 2026-09-29 | 20260929-142605_benchmark_equates_fold3-final-swin-multitask-s44_3e42 | equates_pilot | swin | multitask | 44 | a4222af46a | 5.554 | 3.804 | final, tuning trial 4 |
| 2026-09-29 | 20260929-142451_benchmark_equates_fold3-final-swin_unet-species-s44_4dbe | equates_pilot | swin_unet | species | 44 | e7198fc2a0 | 4.494 | 3.595 | final, tuning trial 6 |
| 2026-09-29 | 20260929-142900_benchmark_equates_fold3-final-gnn-species-s42_52fc | equates_pilot | gnn | species | 42 | e808b0e54c | 3.775 | 3.187 | final, tuning trial 2 |
| 2026-09-29 | 20260929-143034_benchmark_equates_fold3-tune-mamba-t0_d717 | equates_pilot | mamba | ox | 42 | 66b324f263 | 4.116 | – | tuning trial 0 |
| 2026-09-29 | 20260929-143035_benchmark_equates_fold3-final-swin_unet-multitask-s42_45c2 | equates_pilot | swin_unet | multitask | 42 | 1b29dffcf7 | 4.401 | 3.566 | final, tuning trial 6 |
| 2026-09-29 | 20260929-143451_benchmark_equates_fold3-tune-mamba-t1_2581 | equates_pilot | mamba | ox | 42 | 0c8efe8b23 | 4.031 | – | tuning trial 1 |
| 2026-09-29 | 20260929-143518_benchmark_equates_fold3-final-swin_unet-multitask-s43_369a | equates_pilot | swin_unet | multitask | 43 | 8c2da79731 | 4.629 | 3.530 | final, tuning trial 6 |
| 2026-09-29 | 20260929-143350_benchmark_equates_fold3-final-gnn-species-s43_3bfe | equates_pilot | gnn | species | 43 | d57cf96aca | 3.493 | 2.960 | final, tuning trial 2 |
| 2026-09-29 | 20260929-142617_benchmark_equates_fold2-final-mamba-multitask-s44_1863 | equates_pilot | mamba | multitask | 44 | 5dd1786845 | 2.860 | 3.329 | final, tuning trial 2 |
| 2026-09-29 | 20260929-144139_benchmark_equates_fold3-final-gnn-species-s44_e351 | equates_pilot | gnn | species | 44 | e05027cf75 | 3.721 | 3.215 | final, tuning trial 2 |
| 2026-09-29 | 20260929-144514_benchmark_equates_fold4-tune-unet-t2_45d9 | equates_pilot | unet | ox | 42 | cdd7be74d1 | 3.314 | – | tuning trial 2 |
| 2026-09-29 | 20260929-144628_benchmark_equates_fold4-tune-unet-t3_743a | equates_pilot | unet | ox | 42 | 811c002832 | 3.318 | – | tuning trial 3 |
| 2026-09-29 | 20260929-144746_benchmark_equates_fold4-tune-unet-t4_c08e | equates_pilot | unet | ox | 42 | 3c6edc506e | 3.397 | – | tuning trial 4 |
| 2026-09-29 | 20260929-143813_benchmark_equates_fold3-tune-mamba-t2_e8d3 | equates_pilot | mamba | ox | 42 | e86782106d | 4.062 | – | tuning trial 2 |
| 2026-09-29 | 20260929-144903_benchmark_equates_fold4-tune-unet-t5_04f3 | equates_pilot | unet | ox | 42 | 4f573fea90 | 3.332 | – | tuning trial 5 |
| 2026-09-29 | 20260929-144605_benchmark_equates_fold3-final-gnn-multitask-s42_6fc3 | equates_pilot | gnn | multitask | 42 | fdb3f59e64 | 3.709 | 3.230 | final, tuning trial 2 |
| 2026-09-29 | 20260929-145012_benchmark_equates_fold4-tune-unet-t6_882b | equates_pilot | unet | ox | 42 | 297c6b4cbc | 3.256 | – | tuning trial 6 |
| 2026-09-29 | 20260929-144033_benchmark_equates_fold3-final-swin_unet-multitask-s44_fdab | equates_pilot | swin_unet | multitask | 44 | 824c282bde | 4.394 | 3.598 | final, tuning trial 6 |
| 2026-09-29 | 20260929-145122_benchmark_equates_fold4-tune-unet-t7_150b | equates_pilot | unet | ox | 42 | 8b5a74648f | 3.336 | – | tuning trial 7 |
| 2026-09-29 | 20260929-145234_benchmark_equates_fold4-tune-fno-t3_37d9 | equates_pilot | fno | ox | 42 | 3b88e8bca0 | 3.856 | – | tuning trial 3 |
| 2026-09-29 | 20260929-145232_benchmark_equates_fold4-final-unet-ox-s42_0d94 | equates_pilot | unet | ox | 42 | 9ca901b999 | 3.187 | 4.592 | final, tuning trial 0 |
| 2026-09-29 | 20260929-144956_benchmark_equates_fold3-tune-mamba-t3_33dd | equates_pilot | mamba | ox | 42 | bd4fae8b45 | 4.622 | – | tuning trial 3 |
| 2026-09-29 | 20260929-145337_benchmark_equates_fold4-tune-fno-t4_5523 | equates_pilot | fno | ox | 42 | 8600db6c85 | 3.512 | – | tuning trial 4 |
| 2026-09-29 | 20260929-145344_benchmark_equates_fold4-final-unet-ox-s43_44c2 | equates_pilot | unet | ox | 43 | c1ee092d11 | 3.409 | 4.714 | final, tuning trial 0 |
| 2026-09-29 | 20260929-145436_benchmark_equates_fold4-tune-fno-t5_7f19 | equates_pilot | fno | ox | 42 | 2ede62ecbb | 3.984 | – | tuning trial 5 |
| 2026-09-29 | 20260929-145056_benchmark_equates_fold3-final-gnn-multitask-s43_5296 | equates_pilot | gnn | multitask | 43 | 91e6c39a28 | 3.642 | 3.062 | final, tuning trial 2 |
| 2026-09-29 | 20260929-145533_benchmark_equates_fold4-tune-fno-t6_118d | equates_pilot | fno | ox | 42 | c08f533e41 | 3.514 | – | tuning trial 6 |
| 2026-09-29 | 20260929-145512_benchmark_equates_fold4-final-unet-ox-s44_ef1e | equates_pilot | unet | ox | 44 | e254ec84a0 | 3.214 | 4.409 | final, tuning trial 0 |
| 2026-09-29 | 20260929-145631_benchmark_equates_fold4-tune-fno-t7_0b6f | equates_pilot | fno | ox | 42 | 050ba51d40 | 3.839 | – | tuning trial 7 |
| 2026-09-29 | 20260929-145348_benchmark_equates_fold3-tune-mamba-t4_1a21 | equates_pilot | mamba | ox | 42 | 0531c8928a | 4.031 | – | tuning trial 4 |
| 2026-09-29 | 20260929-145714_benchmark_equates_fold4-final-fno-ox-s42_2a11 | equates_pilot | fno | ox | 42 | 5fa5395f00 | 3.404 | 4.742 | final, tuning trial 1 |
| 2026-09-29 | 20260929-145820_benchmark_equates_fold4-final-fno-ox-s43_7e9d | equates_pilot | fno | ox | 43 | fe4e6ce42e | 3.446 | 4.572 | final, tuning trial 1 |
| 2026-09-29 | 20260929-145753_benchmark_equates_fold3-tune-mamba-t5_9b47 | equates_pilot | mamba | ox | 42 | 0fe59c566e | 4.548 | – | tuning trial 5 |
| 2026-09-29 | 20260929-145647_benchmark_equates_fold4-final-unet-species-s42_6405 | equates_pilot | unet | species | 42 | 5ca701559d | 3.205 | 4.943 | final, tuning trial 0 |
| 2026-09-29 | 20260929-145554_benchmark_equates_fold3-final-gnn-multitask-s44_de88 | equates_pilot | gnn | multitask | 44 | 4de2b5319d | 3.673 | 3.105 | final, tuning trial 2 |
| 2026-09-29 | 20260929-145928_benchmark_equates_fold4-final-fno-ox-s44_4754 | equates_pilot | fno | ox | 44 | 5fb3a16387 | 3.317 | 4.350 | final, tuning trial 1 |
| 2026-09-29 | 20260929-150048_benchmark_equates_fold4-final-unet-species-s43_3424 | equates_pilot | unet | species | 43 | 9b9828b8f5 | 3.381 | 4.909 | final, tuning trial 0 |
| 2026-09-29 | 20260929-150130_benchmark_equates_fold4-final-fno-species-s42_612c | equates_pilot | fno | species | 42 | de379f0eca | 3.498 | 4.489 | final, tuning trial 1 |
| 2026-09-29 | 20260929-150243_benchmark_equates_fold4-final-fno-species-s43_75a8 | equates_pilot | fno | species | 43 | e5e657ac39 | 3.442 | 4.571 | final, tuning trial 1 |
| 2026-09-29 | 20260929-150233_benchmark_equates_fold4-final-unet-species-s44_c81f | equates_pilot | unet | species | 44 | 161f4f5b15 | 3.198 | 4.830 | final, tuning trial 0 |
| 2026-09-29 | 20260929-145940_benchmark_equates_fold3-tune-mamba-t6_e31e | equates_pilot | mamba | ox | 42 | 3b1dcd3333 | 4.108 | – | tuning trial 6 |
| 2026-09-29 | 20260929-150142_benchmark_equates_fold4-tune-swin-t2_e8e0 | equates_pilot | swin | ox | 42 | 145d417a28 | 3.888 | – | tuning trial 2 |
| 2026-09-29 | 20260929-150411_benchmark_equates_fold4-final-fno-species-s44_bc85 | equates_pilot | fno | species | 44 | bb9d77d6d4 | 3.608 | 4.606 | final, tuning trial 1 |
| 2026-09-29 | 20260929-150438_benchmark_equates_fold3-tune-mamba-t7_27de | equates_pilot | mamba | ox | 42 | c556dfd126 | 4.670 | – | tuning trial 7 |
| 2026-09-29 | 20260929-150430_benchmark_equates_fold4-final-unet-multitask-s42_d865 | equates_pilot | unet | multitask | 42 | 86907edfdd | 3.164 | 4.525 | final, tuning trial 0 |
| 2026-09-29 | 20260929-150537_benchmark_equates_fold4-final-fno-multitask-s42_539d | equates_pilot | fno | multitask | 42 | 1b77242200 | 3.350 | 4.541 | final, tuning trial 1 |
| 2026-09-29 | 20260929-150628_benchmark_equates_fold4-final-unet-multitask-s43_e6ee | equates_pilot | unet | multitask | 43 | f0eee18009 | 3.256 | 5.207 | final, tuning trial 0 |
| 2026-09-29 | 20260929-150650_benchmark_equates_fold4-final-fno-multitask-s43_fbb8 | equates_pilot | fno | multitask | 43 | 24b8f9d367 | 3.514 | 4.559 | final, tuning trial 1 |
| 2026-09-29 | 20260929-150533_benchmark_equates_fold4-tune-swin-t3_ff7f | equates_pilot | swin | ox | 42 | eaab3408bc | 3.880 | – | tuning trial 3 |
| 2026-09-29 | 20260929-150629_benchmark_equates_fold3-final-mamba-ox-s42_ae4b | equates_pilot | mamba | ox | 42 | 0531c8928a | 4.038 | 3.346 | final, tuning trial 4 |
| 2026-09-29 | 20260929-150808_benchmark_equates_fold4-final-unet-multitask-s44_eac0 | equates_pilot | unet | multitask | 44 | 0bd888fcf9 | 3.242 | 4.828 | final, tuning trial 0 |
| 2026-09-29 | 20260929-150831_benchmark_equates_fold4-final-fno-multitask-s44_930e | equates_pilot | fno | multitask | 44 | 177251c993 | 3.375 | 4.465 | final, tuning trial 1 |
| 2026-09-29 | 20260929-150954_benchmark_equates_fold3-final-mamba-ox-s43_3430 | equates_pilot | mamba | ox | 43 | aa32bc8aed | 3.966 | 3.227 | final, tuning trial 4 |
| 2026-09-29 | 20260929-150917_benchmark_equates_fold4-tune-swin-t4_c019 | equates_pilot | swin | ox | 42 | b02fb42e5d | 3.367 | – | tuning trial 4 |
| 2026-09-29 | 20260929-151103_benchmark_equates_fold4-tune-gnn-t2_dce5 | equates_pilot | gnn | ox | 42 | 74aeb74803 | 2.972 | – | tuning trial 2 |
| 2026-09-29 | 20260929-151045_benchmark_equates_fold4-tune-swin_unet-t0_3aac | equates_pilot | swin_unet | ox | 42 | 40dcc1a87c | 3.654 | – | tuning trial 0 |
| 2026-09-29 | 20260929-151307_benchmark_equates_fold3-final-mamba-ox-s44_a6b4 | equates_pilot | mamba | ox | 44 | 09dc54fc9b | 3.968 | 3.225 | final, tuning trial 4 |
| 2026-09-29 | 20260929-151350_benchmark_equates_fold4-tune-swin-t5_3eef | equates_pilot | swin | ox | 42 | 959bc8965d | 3.404 | – | tuning trial 5 |
| 2026-09-29 | 20260929-151559_benchmark_equates_fold4-tune-gnn-t3_841c | equates_pilot | gnn | ox | 42 | 4f832d4275 | 3.129 | – | tuning trial 3 |
| 2026-09-29 | 20260929-151715_benchmark_equates_fold3-final-mamba-species-s42_7ee3 | equates_pilot | mamba | species | 42 | 0428fb9317 | 4.554 | 3.383 | final, tuning trial 4 |
| 2026-09-29 | 20260929-151721_benchmark_equates_fold4-tune-swin-t6_8e26 | equates_pilot | swin | ox | 42 | 4a08e63029 | 3.758 | – | tuning trial 6 |
| 2026-09-29 | 20260929-151651_benchmark_equates_fold4-tune-swin_unet-t1_ba02 | equates_pilot | swin_unet | ox | 42 | 5ccbebcee0 | 3.605 | – | tuning trial 1 |
| 2026-09-29 | 20260929-152101_benchmark_equates_fold3-final-mamba-species-s43_efd0 | equates_pilot | mamba | species | 43 | c12f351d4f | 4.363 | 3.620 | final, tuning trial 4 |
| 2026-09-29 | 20260929-152053_benchmark_equates_fold4-tune-gnn-t4_d069 | equates_pilot | gnn | ox | 42 | 84a7ecd287 | 3.551 | – | tuning trial 4 |
| 2026-09-29 | 20260929-152113_benchmark_equates_fold4-tune-swin-t7_41de | equates_pilot | swin | ox | 42 | fdd6968969 | 3.671 | – | tuning trial 7 |
| 2026-09-29 | 20260929-152250_benchmark_equates_fold4-tune-swin_unet-t2_e7c9 | equates_pilot | swin_unet | ox | 42 | dfd3300e25 | 3.839 | – | tuning trial 2 |
| 2026-09-29 | 20260929-152458_benchmark_equates_fold3-final-mamba-species-s44_610a | equates_pilot | mamba | species | 44 | 1c9a1bbcd0 | 4.148 | 3.526 | final, tuning trial 4 |
| 2026-09-29 | 20260929-152617_benchmark_equates_fold4-tune-gnn-t5_ed9b | equates_pilot | gnn | ox | 42 | 1f54a2b3d8 | 3.496 | – | tuning trial 5 |
| 2026-09-29 | 20260929-152705_benchmark_equates_fold4-final-swin-ox-s42_6481 | equates_pilot | swin | ox | 42 | b02fb42e5d | 3.367 | 5.275 | final, tuning trial 4 |
| 2026-09-29 | 20260929-152929_benchmark_equates_fold4-tune-gnn-t6_1b9e | equates_pilot | gnn | ox | 42 | deb69654ba | 3.453 | – | tuning trial 6 |
| 2026-09-29 | 20260929-152910_benchmark_equates_fold3-final-mamba-multitask-s42_5654 | equates_pilot | mamba | multitask | 42 | 4d99976eae | 4.190 | 3.465 | final, tuning trial 4 |
| 2026-09-29 | 20260929-152900_benchmark_equates_fold4-tune-swin_unet-t3_7ed6 | equates_pilot | swin_unet | ox | 42 | 465b8a78ab | 3.803 | – | tuning trial 3 |
| 2026-09-29 | 20260929-153142_benchmark_equates_fold4-final-swin-ox-s43_fa6f | equates_pilot | swin | ox | 43 | 8369f52072 | 3.468 | 5.455 | final, tuning trial 4 |
| 2026-09-29 | 20260929-153334_benchmark_equates_fold3-final-mamba-multitask-s43_3d3a | equates_pilot | mamba | multitask | 43 | 3b92df46d5 | 4.380 | 3.495 | final, tuning trial 4 |
| 2026-09-29 | 20260929-153602_benchmark_equates_fold4-final-swin-ox-s44_b9e7 | equates_pilot | swin | ox | 44 | e128f0dcb9 | 3.686 | 5.537 | final, tuning trial 4 |
| 2026-09-29 | 20260929-153319_benchmark_equates_fold4-tune-gnn-t7_cd5a | equates_pilot | gnn | ox | 42 | 3c81ddea56 | 3.239 | – | tuning trial 7 |
| 2026-09-29 | 20260929-153410_benchmark_equates_fold4-tune-swin_unet-t4_0d70 | equates_pilot | swin_unet | ox | 42 | d833ae7bce | 3.436 | – | tuning trial 4 |
| 2026-09-29 | 20260929-153738_benchmark_equates_fold3-final-mamba-multitask-s44_4a5f | equates_pilot | mamba | multitask | 44 | 4c68184c5d | 4.177 | 3.498 | final, tuning trial 4 |
| 2026-09-29 | 20260929-153922_benchmark_equates_fold4-final-swin-species-s42_2f07 | equates_pilot | swin | species | 42 | 8d40d9c5fa | 4.107 | 5.579 | final, tuning trial 4 |
| 2026-09-29 | 20260929-154045_benchmark_equates_fold4-final-gnn-ox-s42_0cdd | equates_pilot | gnn | ox | 42 | 74aeb74803 | 2.969 | 3.680 | final, tuning trial 2 |
| 2026-09-29 | 20260929-154114_benchmark_equates_fold4-tune-swin_unet-t5_bae4 | equates_pilot | swin_unet | ox | 42 | 57effc8e10 | 3.436 | – | tuning trial 5 |
| 2026-09-29 | 20260929-154221_benchmark_equates_fold4-tune-mamba-t1_6e9c | equates_pilot | mamba | ox | 42 | 97ed42897f | 3.061 | – | tuning trial 1 |
| 2026-09-29 | 20260929-154403_benchmark_equates_fold4-final-swin-species-s43_e5a6 | equates_pilot | swin | species | 43 | 2dab9777ce | 3.686 | 5.520 | final, tuning trial 4 |
| 2026-09-29 | 20260929-154541_benchmark_equates_fold4-final-gnn-ox-s43_49c3 | equates_pilot | gnn | ox | 43 | e6376ca898 | 2.985 | 3.789 | final, tuning trial 2 |
| 2026-09-29 | 20260929-154642_benchmark_equates_fold4-tune-swin_unet-t6_b392 | equates_pilot | swin_unet | ox | 42 | a3022ee22d | 3.534 | – | tuning trial 6 |
| 2026-09-29 | 20260929-154755_benchmark_equates_fold4-tune-mamba-t2_42a6 | equates_pilot | mamba | ox | 42 | f90fa06988 | 3.018 | – | tuning trial 2 |
| 2026-09-29 | 20260929-154950_benchmark_equates_fold4-final-swin-species-s44_01c8 | equates_pilot | swin | species | 44 | 0abbed15d5 | 3.576 | 5.486 | final, tuning trial 4 |
| 2026-09-29 | 20260929-155049_benchmark_equates_fold4-final-gnn-ox-s44_88a8 | equates_pilot | gnn | ox | 44 | ccc8a211c8 | 2.981 | 3.801 | final, tuning trial 2 |
| 2026-09-29 | 20260929-155216_benchmark_equates_fold4-tune-swin_unet-t7_0f87 | equates_pilot | swin_unet | ox | 42 | 3208332442 | 3.507 | – | tuning trial 7 |
| 2026-09-29 | 20260929-155421_benchmark_equates_fold4-tune-mamba-t3_850a | equates_pilot | mamba | ox | 42 | 6c5c9bfb83 | 3.342 | – | tuning trial 3 |
| 2026-09-29 | 20260929-155434_benchmark_equates_fold4-final-swin-multitask-s42_e73a | equates_pilot | swin | multitask | 42 | 614a4182f1 | 3.686 | 5.601 | final, tuning trial 4 |
| 2026-09-29 | 20260929-155738_benchmark_equates_fold4-tune-mamba-t4_fc5d | equates_pilot | mamba | ox | 42 | 09fd8903b2 | 3.234 | – | tuning trial 4 |
| 2026-09-29 | 20260929-155942_benchmark_equates_fold4-final-swin-multitask-s43_015e | equates_pilot | swin | multitask | 43 | e97391ae53 | 3.476 | 5.383 | final, tuning trial 4 |
| 2026-09-29 | 20260929-160104_benchmark_equates_fold4-tune-mamba-t5_1103 | equates_pilot | mamba | ox | 42 | 9b895f2944 | 3.375 | – | tuning trial 5 |
| 2026-09-29 | 20260929-155642_benchmark_equates_fold4-final-swin_unet-ox-s42_9998 | equates_pilot | swin_unet | ox | 42 | d833ae7bce | 3.436 | 4.222 | final, tuning trial 4 |
| 2026-09-29 | 20260929-155552_benchmark_equates_fold4-final-gnn-species-s42_3611 | equates_pilot | gnn | species | 42 | aaab29b6cd | 2.932 | 3.751 | final, tuning trial 2 |
| 2026-09-29 | 20260929-160255_benchmark_equates_fold4-final-swin-multitask-s44_d9b0 | equates_pilot | swin | multitask | 44 | eee15dc165 | 3.621 | 5.863 | final, tuning trial 4 |
| 2026-09-29 | 20260929-160320_benchmark_equates_fold4-tune-mamba-t6_ab94 | equates_pilot | mamba | ox | 42 | 084e98b65c | 3.053 | – | tuning trial 6 |
| 2026-09-29 | 20260929-160807_benchmark_equates_fold4-tune-mamba-t7_3b99 | equates_pilot | mamba | ox | 42 | 237fff940f | 3.461 | – | tuning trial 7 |
| 2026-09-29 | 20260929-160348_benchmark_equates_fold4-final-swin_unet-ox-s43_787d | equates_pilot | swin_unet | ox | 43 | 5a8a5a14f5 | 3.412 | 4.324 | final, tuning trial 4 |
| 2026-09-29 | 20260929-160628_benchmark_equates_fold4-final-gnn-species-s43_be68 | equates_pilot | gnn | species | 43 | 5f571df3ec | 2.987 | 3.709 | final, tuning trial 2 |
| 2026-09-29 | 20260929-161108_benchmark_equates_fold4-final-gnn-species-s44_4dd7 | equates_pilot | gnn | species | 44 | 6f6964751f | 3.030 | 3.779 | final, tuning trial 2 |
| 2026-09-29 | 20260929-161045_benchmark_equates_fold4-final-mamba-ox-s42_5741 | equates_pilot | mamba | ox | 42 | f90fa06988 | 3.018 | 4.159 | final, tuning trial 2 |
| 2026-09-29 | 20260929-161101_benchmark_equates_fold4-final-swin_unet-ox-s44_a109 | equates_pilot | swin_unet | ox | 44 | 30e41348ac | 3.507 | 4.292 | final, tuning trial 4 |
| 2026-09-29 | 20260929-161548_benchmark_equates_fold4-final-gnn-multitask-s42_52be | equates_pilot | gnn | multitask | 42 | 1fb30a5251 | 2.918 | 3.706 | final, tuning trial 2 |
| 2026-09-29 | 20260929-161717_benchmark_equates_fold4-final-mamba-ox-s43_e749 | equates_pilot | mamba | ox | 43 | 6758ff855b | 3.103 | 4.142 | final, tuning trial 2 |
| 2026-09-29 | 20260929-162007_benchmark_equates_fold4-final-gnn-multitask-s43_630a | equates_pilot | gnn | multitask | 43 | 240e6f3457 | 2.988 | 3.717 | final, tuning trial 2 |
| 2026-09-29 | 20260929-161842_benchmark_equates_fold4-final-swin_unet-species-s42_0815 | equates_pilot | swin_unet | species | 42 | cc9cab0164 | 3.554 | 4.521 | final, tuning trial 4 |
| 2026-09-29 | 20260929-162412_benchmark_equates_fold4-final-mamba-ox-s44_0d43 | equates_pilot | mamba | ox | 44 | 1093d18129 | 3.038 | 4.403 | final, tuning trial 2 |
| 2026-09-29 | 20260929-162504_benchmark_equates_fold4-final-gnn-multitask-s44_478a | equates_pilot | gnn | multitask | 44 | f3de1b7364 | 2.940 | 3.806 | final, tuning trial 2 |
| 2026-09-29 | 20260929-162621_benchmark_equates_fold4-final-swin_unet-species-s43_b77f | equates_pilot | swin_unet | species | 43 | 368997445a | 3.476 | 4.527 | final, tuning trial 4 |
| 2026-09-29 | 20260929-163252_benchmark_equates_fold4-final-swin_unet-species-s44_8c84 | equates_pilot | swin_unet | species | 44 | e36a8fa586 | 3.582 | 4.398 | final, tuning trial 4 |
| 2026-09-29 | 20260929-162949_benchmark_equates_fold4-final-mamba-species-s42_a0cf | equates_pilot | mamba | species | 42 | 3d62633fed | 3.036 | 4.152 | final, tuning trial 2 |
| 2026-09-29 | 20260929-163928_benchmark_equates_fold4-final-swin_unet-multitask-s42_41af | equates_pilot | swin_unet | multitask | 42 | 1f1eefced4 | 3.513 | 4.225 | final, tuning trial 4 |
| 2026-09-29 | 20260929-164639_benchmark_equates_fold4-final-swin_unet-multitask-s43_c405 | equates_pilot | swin_unet | multitask | 43 | 7f99653010 | 3.615 | 4.420 | final, tuning trial 4 |
| 2026-09-29 | 20260929-164113_benchmark_equates_fold4-final-mamba-species-s43_fbb0 | equates_pilot | mamba | species | 43 | 42a56db0b9 | 3.030 | 3.999 | final, tuning trial 2 |
| 2026-09-29 | 20260929-165329_benchmark_equates_fold4-final-swin_unet-multitask-s44_7aaa | equates_pilot | swin_unet | multitask | 44 | e9a0136339 | 3.626 | 4.308 | final, tuning trial 4 |
| 2026-09-29 | 20260929-165437_benchmark_equates_fold4-final-mamba-species-s44_0b16 | equates_pilot | mamba | species | 44 | 9e98403439 | 3.075 | 3.986 | final, tuning trial 2 |
| 2026-09-29 | 20260929-171039_benchmark_equates_fold4-final-mamba-multitask-s42_3569 | equates_pilot | mamba | multitask | 42 | d5ca23b76a | 3.091 | 4.019 | final, tuning trial 2 |
| 2026-09-29 | 20260929-171749_benchmark_equates_fold4-final-mamba-multitask-s43_cf6c | equates_pilot | mamba | multitask | 43 | 48ea9ae191 | 3.060 | 3.976 | final, tuning trial 2 |
| 2026-09-29 | 20260929-172441_benchmark_equates_fold4-final-mamba-multitask-s44_b160 | equates_pilot | mamba | multitask | 44 | 7129c95634 | 3.052 | 3.965 | final, tuning trial 2 |
| 2026-09-29 | 20260929-180233_benchmark_equates_fold1-tune-fno-t7_805e | equates_pilot | fno | ox | 42 | 5fed8505fd | 3.528 | – | tuning trial 7 |
| 2026-09-29 | 20260929-180406_benchmark_equates_fold1-final-fno-ox-s42_501a | equates_pilot | fno | ox | 42 | 6fe62fbac2 | 3.398 | 3.063 | final, tuning trial 6 |
| 2026-09-29 | 20260929-180531_benchmark_equates_fold1-final-fno-ox-s43_8da1 | equates_pilot | fno | ox | 43 | 2829545cdb | 3.353 | 2.866 | final, tuning trial 6 |
| 2026-09-29 | 20260929-180638_benchmark_equates_fold1-final-fno-ox-s44_acf6 | equates_pilot | fno | ox | 44 | fa607ea875 | 3.443 | 3.022 | final, tuning trial 6 |
| 2026-09-29 | 20260929-180749_benchmark_equates_fold1-final-fno-species-s42_4866 | equates_pilot | fno | species | 42 | 1ef99e7bc6 | 3.414 | 2.944 | final, tuning trial 6 |
| 2026-09-29 | 20260929-180905_benchmark_equates_fold1-final-fno-species-s43_ca7d | equates_pilot | fno | species | 43 | 1cd7175e9c | 3.498 | 2.895 | final, tuning trial 6 |
| 2026-09-29 | 20260929-181024_benchmark_equates_fold1-final-fno-species-s44_c5d2 | equates_pilot | fno | species | 44 | c20615ad06 | 3.386 | 2.900 | final, tuning trial 6 |
| 2026-09-29 | 20260929-181142_benchmark_equates_fold1-final-fno-multitask-s42_8e5e | equates_pilot | fno | multitask | 42 | 2a032420b9 | 3.367 | 3.078 | final, tuning trial 6 |
| 2026-09-29 | 20260929-181323_benchmark_equates_fold1-final-fno-multitask-s43_4735 | equates_pilot | fno | multitask | 43 | 0486c25da8 | 3.417 | 2.947 | final, tuning trial 6 |
| 2026-09-29 | 20260929-181451_benchmark_equates_fold1-final-fno-multitask-s44_2546 | equates_pilot | fno | multitask | 44 | b95f77dd2c | 3.419 | 3.044 | final, tuning trial 6 |
| 2026-09-29 | 20260929-180257_benchmark_equates_fold3-tune-convlstm-t0_ac6d | equates_pilot | convlstm | ox | 42 | ae660c1cc7 | 4.302 | – | tuning trial 0 |
| 2026-09-29 | 20260929-180251_benchmark_equates_fold4-tune-convlstm-t0_9efd | equates_pilot | convlstm | ox | 42 | 32d3701925 | 3.307 | – | tuning trial 0 |
| 2026-09-29 | 20260929-180251_benchmark_equates_fold1-tune-convlstm-t0_d45f | equates_pilot | convlstm | ox | 42 | b4cd994c14 | 3.132 | – | tuning trial 0 |
| 2026-09-29 | 20260929-180251_benchmark_equates_fold2-tune-convlstm-t0_fdf9 | equates_pilot | convlstm | ox | 42 | b77023d33a | 2.850 | – | tuning trial 0 |
| 2026-09-29 | 20260929-182029_benchmark_equates_fold1-tune-convlstm-t1_c5c1 | equates_pilot | convlstm | ox | 42 | c02e09a2be | 3.076 | – | tuning trial 1 |
| 2026-09-29 | 20260929-181856_benchmark_equates_fold3-tune-convlstm-t1_c319 | equates_pilot | convlstm | ox | 42 | 9ffbb43e90 | 4.577 | – | tuning trial 1 |
| 2026-09-29 | 20260929-181938_benchmark_equates_fold4-tune-convlstm-t1_e540 | equates_pilot | convlstm | ox | 42 | 00fa8e47b3 | 3.283 | – | tuning trial 1 |
| 2026-09-29 | 20260929-182035_benchmark_equates_fold2-tune-convlstm-t1_cfb2 | equates_pilot | convlstm | ox | 42 | e2f8f91fa8 | 2.940 | – | tuning trial 1 |
| 2026-09-29 | 20260929-183020_benchmark_equates_fold1-tune-convlstm-t2_1f8f | equates_pilot | convlstm | ox | 42 | 78f4a9f809 | 3.211 | – | tuning trial 2 |
| 2026-09-29 | 20260929-183440_benchmark_equates_fold4-tune-convlstm-t2_5167 | equates_pilot | convlstm | ox | 42 | c26bd033e2 | 3.231 | – | tuning trial 2 |
| 2026-09-29 | 20260929-183038_benchmark_equates_fold3-tune-convlstm-t2_47d5 | equates_pilot | convlstm | ox | 42 | 6c680b82ac | 4.860 | – | tuning trial 2 |
| 2026-09-29 | 20260929-183800_benchmark_equates_fold2-tune-convlstm-t2_a5e6 | equates_pilot | convlstm | ox | 42 | 5568f78385 | 3.034 | – | tuning trial 2 |
| 2026-09-29 | 20260929-184858_benchmark_equates_fold3-tune-convlstm-t3_e09b | equates_pilot | convlstm | ox | 42 | d1ad7e327d | 5.113 | – | tuning trial 3 |
| 2026-09-29 | 20260929-184739_benchmark_equates_fold4-tune-convlstm-t3_75e8 | equates_pilot | convlstm | ox | 42 | 1c240ec532 | 3.373 | – | tuning trial 3 |
| 2026-09-29 | 20260929-184313_benchmark_equates_fold1-tune-convlstm-t3_4dc1 | equates_pilot | convlstm | ox | 42 | 7e2c7df8fc | 3.341 | – | tuning trial 3 |
| 2026-09-29 | 20260929-185551_benchmark_equates_fold2-tune-convlstm-t3_ef42 | equates_pilot | convlstm | ox | 42 | 63ea2d3854 | 3.064 | – | tuning trial 3 |
| 2026-09-29 | 20260929-185729_benchmark_equates_fold3-tune-convlstm-t4_7214 | equates_pilot | convlstm | ox | 42 | 8eca26d76b | 4.277 | – | tuning trial 4 |
| 2026-09-29 | 20260929-190122_benchmark_equates_fold4-tune-convlstm-t4_2d43 | equates_pilot | convlstm | ox | 42 | 7228a9c8f2 | 3.185 | – | tuning trial 4 |
| 2026-09-29 | 20260929-190154_benchmark_equates_fold1-tune-convlstm-t4_e4ba | equates_pilot | convlstm | ox | 42 | 5c8796529f | 3.354 | – | tuning trial 4 |
| 2026-09-29 | 20260929-191054_benchmark_equates_fold3-tune-convlstm-t5_158a | equates_pilot | convlstm | ox | 42 | 8a5cf5d2b9 | 4.530 | – | tuning trial 5 |
| 2026-09-29 | 20260929-190726_benchmark_equates_fold2-tune-convlstm-t4_4d16 | equates_pilot | convlstm | ox | 42 | acb316078e | 2.961 | – | tuning trial 4 |
| 2026-09-29 | 20260929-191605_benchmark_equates_fold4-tune-convlstm-t5_5e2d | equates_pilot | convlstm | ox | 42 | b38444c11c | 3.216 | – | tuning trial 5 |
| 2026-09-29 | 20260929-192316_benchmark_equates_fold3-tune-convlstm-t6_a13c | equates_pilot | convlstm | ox | 42 | 147dbb50d1 | 4.348 | – | tuning trial 6 |
| 2026-09-29 | 20260929-191728_benchmark_equates_fold1-tune-convlstm-t5_aeca | equates_pilot | convlstm | ox | 42 | 0096431bc7 | 3.382 | – | tuning trial 5 |
| 2026-09-29 | 20260929-192914_benchmark_equates_fold2-tune-convlstm-t5_5b40 | equates_pilot | convlstm | ox | 42 | c23bd03e45 | 3.067 | – | tuning trial 5 |
| 2026-09-29 | 20260929-193434_benchmark_equates_fold4-tune-convlstm-t6_321b | equates_pilot | convlstm | ox | 42 | 525d411855 | 3.153 | – | tuning trial 6 |
| 2026-09-29 | 20260929-193435_benchmark_equates_fold3-tune-convlstm-t7_b48a | equates_pilot | convlstm | ox | 42 | 0a09571201 | 4.549 | – | tuning trial 7 |
| 2026-09-29 | 20260929-194732_benchmark_equates_fold4-tune-convlstm-t7_f9bb | equates_pilot | convlstm | ox | 42 | beded865bc | 3.338 | – | tuning trial 7 |
| 2026-09-29 | 20260929-193703_benchmark_equates_fold1-tune-convlstm-t6_348d | equates_pilot | convlstm | ox | 42 | dc1ae20b15 | 3.159 | – | tuning trial 6 |
| 2026-09-29 | 20260929-194810_benchmark_equates_fold3-final-convlstm-ox-s42_fcd6 | equates_pilot | convlstm | ox | 42 | 8eca26d76b | 4.277 | 3.455 | final, tuning trial 4 |
| 2026-09-29 | 20260929-194508_benchmark_equates_fold2-tune-convlstm-t6_b299 | equates_pilot | convlstm | ox | 42 | e491991ee0 | 2.910 | – | tuning trial 6 |
| 2026-09-29 | 20260929-195623_benchmark_equates_fold4-final-convlstm-ox-s42_d682 | equates_pilot | convlstm | ox | 42 | 525d411855 | 3.153 | 4.422 | final, tuning trial 6 |
| 2026-09-29 | 20260929-195912_benchmark_equates_fold1-tune-convlstm-t7_ab04 | equates_pilot | convlstm | ox | 42 | e3d53f1a3b | 3.087 | – | tuning trial 7 |
| 2026-09-30 | 20260929-200141_benchmark_equates_fold3-final-convlstm-ox-s43_f868 | equates_pilot | convlstm | ox | 43 | 3e92fc91fb | 4.356 | 3.318 | final, tuning trial 4 |
| 2026-09-30 | 20260929-200259_benchmark_equates_fold2-tune-convlstm-t7_5bf0 | equates_pilot | convlstm | ox | 42 | d68975c42d | 2.985 | – | tuning trial 7 |
| 2026-09-30 | 20260929-201229_benchmark_equates_fold1-final-convlstm-ox-s42_a596 | equates_pilot | convlstm | ox | 42 | c02e09a2be | 3.076 | 2.781 | final, tuning trial 1 |
| 2026-09-30 | 20260929-200928_benchmark_equates_fold4-final-convlstm-ox-s43_b4e7 | equates_pilot | convlstm | ox | 43 | 73af30c512 | 3.185 | 4.515 | final, tuning trial 6 |
| 2026-09-30 | 20260929-201719_benchmark_equates_fold3-final-convlstm-ox-s44_9e82 | equates_pilot | convlstm | ox | 44 | 76f39ca938 | 4.452 | 3.254 | final, tuning trial 4 |
| 2026-09-30 | 20260929-201857_benchmark_equates_fold2-final-convlstm-ox-s42_8069 | equates_pilot | convlstm | ox | 42 | b77023d33a | 2.850 | 3.904 | final, tuning trial 0 |
| 2026-09-30 | 20260929-202336_benchmark_equates_fold4-final-convlstm-ox-s44_91ce | equates_pilot | convlstm | ox | 44 | 9fa5e0e4c3 | 3.189 | 4.505 | final, tuning trial 6 |
| 2026-09-30 | 20260929-202226_benchmark_equates_fold1-final-convlstm-ox-s43_1bd8 | equates_pilot | convlstm | ox | 43 | d3dd9d3bca | 3.243 | 2.771 | final, tuning trial 1 |
| 2026-09-30 | 20260929-203216_benchmark_equates_fold3-final-convlstm-species-s42_3c8c | equates_pilot | convlstm | species | 42 | f1f640b931 | 3.858 | 3.055 | final, tuning trial 4 |
| 2026-09-30 | 20260929-203644_benchmark_equates_fold2-final-convlstm-ox-s43_b822 | equates_pilot | convlstm | ox | 43 | b6a76e0bb9 | 2.965 | 4.387 | final, tuning trial 0 |
| 2026-09-30 | 20260929-204147_benchmark_equates_fold1-final-convlstm-ox-s44_b86f | equates_pilot | convlstm | ox | 44 | 26f99d004a | 3.063 | 2.676 | final, tuning trial 1 |
| 2026-09-30 | 20260929-203745_benchmark_equates_fold4-final-convlstm-species-s42_0d8c | equates_pilot | convlstm | species | 42 | 052ded07e6 | 2.984 | 4.603 | final, tuning trial 6 |
| 2026-09-30 | 20260929-205535_benchmark_equates_fold2-final-convlstm-ox-s44_9901 | equates_pilot | convlstm | ox | 44 | 50fff03f12 | 2.949 | 3.627 | final, tuning trial 0 |
| 2026-09-30 | 20260929-205323_benchmark_equates_fold3-final-convlstm-species-s43_49c8 | equates_pilot | convlstm | species | 43 | 2e3eacaae7 | 4.387 | 3.207 | final, tuning trial 4 |
| 2026-09-30 | 20260929-205808_benchmark_equates_fold1-final-convlstm-species-s42_8008 | equates_pilot | convlstm | species | 42 | 2f1db62799 | 2.944 | 2.586 | final, tuning trial 1 |
| 2026-09-30 | 20260929-210629_benchmark_equates_fold4-final-convlstm-species-s43_67ee | equates_pilot | convlstm | species | 43 | d8066985c3 | 3.058 | 4.428 | final, tuning trial 6 |
| 2026-09-30 | 20260929-211112_benchmark_equates_fold2-final-convlstm-species-s42_2c6e | equates_pilot | convlstm | species | 42 | a717d4e042 | 2.975 | 3.774 | final, tuning trial 0 |
| 2026-09-30 | 20260929-211659_benchmark_equates_fold3-final-convlstm-species-s44_58fa | equates_pilot | convlstm | species | 44 | dc046f52a6 | 4.438 | 3.171 | final, tuning trial 4 |
| 2026-09-30 | 20260929-211734_benchmark_equates_fold1-final-convlstm-species-s43_53ec | equates_pilot | convlstm | species | 43 | 637cb8e274 | 3.103 | 2.698 | final, tuning trial 1 |
| 2026-09-30 | 20260929-212222_benchmark_equates_fold4-final-convlstm-species-s44_fa06 | equates_pilot | convlstm | species | 44 | ac8efa8e1f | 3.253 | 4.963 | final, tuning trial 6 |
| 2026-09-30 | 20260929-213030_benchmark_equates_fold2-final-convlstm-species-s43_59c7 | equates_pilot | convlstm | species | 43 | 5ba7bbebe7 | 2.940 | 4.375 | final, tuning trial 0 |
| 2026-09-30 | 20260929-213700_benchmark_equates_fold3-final-convlstm-multitask-s42_cb55 | equates_pilot | convlstm | multitask | 42 | 1343f68c45 | 4.448 | 3.336 | final, tuning trial 4 |
| 2026-09-30 | 20260929-213720_benchmark_equates_fold1-final-convlstm-species-s44_7403 | equates_pilot | convlstm | species | 44 | d07b08d845 | 3.186 | 2.814 | final, tuning trial 1 |
| 2026-09-30 | 20260929-214155_benchmark_equates_fold4-final-convlstm-multitask-s42_7e15 | equates_pilot | convlstm | multitask | 42 | ca28d89a5f | 3.128 | 4.482 | final, tuning trial 6 |
| 2026-09-30 | 20260929-214925_benchmark_equates_fold2-final-convlstm-species-s44_1b83 | equates_pilot | convlstm | species | 44 | b90c186806 | 2.968 | 3.718 | final, tuning trial 0 |
| 2026-09-30 | 20260929-215705_benchmark_equates_fold1-final-convlstm-multitask-s42_c0ad | equates_pilot | convlstm | multitask | 42 | d3d15873f8 | 3.026 | 2.523 | final, tuning trial 1 |
| 2026-09-30 | 20260929-215327_benchmark_equates_fold3-final-convlstm-multitask-s43_0d82 | equates_pilot | convlstm | multitask | 43 | 55f6198f10 | 4.087 | 3.156 | final, tuning trial 4 |
| 2026-09-30 | 20260929-215838_benchmark_equates_fold4-final-convlstm-multitask-s43_787f | equates_pilot | convlstm | multitask | 43 | 995524d46f | 3.119 | 4.527 | final, tuning trial 6 |
| 2026-09-30 | 20260929-221029_benchmark_equates_fold1-final-convlstm-multitask-s43_df39 | equates_pilot | convlstm | multitask | 43 | 60ed604faf | 3.035 | 2.605 | final, tuning trial 1 |
| 2026-09-30 | 20260929-220841_benchmark_equates_fold2-final-convlstm-multitask-s42_4afb | equates_pilot | convlstm | multitask | 42 | a94e09c260 | 2.890 | 3.848 | final, tuning trial 0 |
| 2026-09-30 | 20260929-221647_benchmark_equates_fold4-final-convlstm-multitask-s44_b871 | equates_pilot | convlstm | multitask | 44 | d0e257e48d | 3.228 | 4.709 | final, tuning trial 6 |
| 2026-09-30 | 20260929-221414_benchmark_equates_fold3-final-convlstm-multitask-s44_ab86 | equates_pilot | convlstm | multitask | 44 | a85599d85e | 4.368 | 3.252 | final, tuning trial 4 |
| 2026-09-30 | 20260929-222412_benchmark_equates_fold1-final-convlstm-multitask-s44_341c | equates_pilot | convlstm | multitask | 44 | df1ca089b2 | 3.215 | 2.556 | final, tuning trial 1 |
| 2026-09-30 | 20260929-222714_benchmark_equates_fold2-final-convlstm-multitask-s43_6e55 | equates_pilot | convlstm | multitask | 43 | be9b414be9 | 2.961 | 3.970 | final, tuning trial 0 |
| 2026-09-30 | 20260929-224443_benchmark_equates_fold2-final-convlstm-multitask-s44_aea6 | equates_pilot | convlstm | multitask | 44 | 9db75c0afe | 2.893 | 3.951 | final, tuning trial 0 |
