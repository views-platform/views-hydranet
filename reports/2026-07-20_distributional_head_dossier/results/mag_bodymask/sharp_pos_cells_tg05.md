# Sharpness scorecard

pred_dir: `/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260727_200938`

**CONFIG (auto, from sidecar):** output_distribution=nb · reg_activation=None · static_channels=[] · dropout_rate=0.15 · input_channels=3

| target | subset | FSS@[1, 3, 5, 11] | area_ratio | conc1% | MCR | max (explode?) |
|--------|--------|-----------|-----------|--------|-----|----------------|
| lr_sb_best | STEP-1 | 0.21/0.56/0.67/0.79 | 4.4× | 0.15 | 0.694 |  |
| lr_sb_best | FULL | 0.02/0.10/0.15/0.29 | 43.3× | 0.37 | 2.855 | 32887612.0 (≤5669750 💥EXPLODED) |
| lr_ns_best | STEP-1 | 0.19/0.51/0.63/0.75 | 2.7× | 0.16 | 0.648 |  |
| lr_ns_best | FULL | 0.01/0.05/0.09/0.20 | 50.3× | 0.51 | 3.220 | 23074802.0 (≤100000 💥EXPLODED) |
| lr_os_best | STEP-1 | 0.11/0.33/0.42/0.52 | 2.9× | 0.17 | 0.848 |  |
| lr_os_best | FULL | 0.01/0.05/0.09/0.21 | 5.7× | 0.55 | 0.718 | 21867586.0 (≤8922950 💥EXPLODED) |
