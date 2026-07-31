# Sharpness scorecard

pred_dir: `/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260727_200938`

**CONFIG (auto, from sidecar):** output_distribution=nb · reg_activation=None · static_channels=[] · dropout_rate=0.15 · input_channels=3

| target | subset | FSS@[1, 3, 5, 11] | area_ratio | conc1% | MCR | max (explode?) |
|--------|--------|-----------|-----------|--------|-----|----------------|
| lr_sb_best | STEP-1 | 0.21/0.56/0.67/0.78 | 6.3× | 0.17 | 0.672 |  |
| lr_sb_best | FULL | 0.02/0.09/0.15/0.29 | 45.0× | 0.38 | 2.967 | 32818746.0 (≤5669750 💥EXPLODED) |
| lr_ns_best | STEP-1 | 0.18/0.50/0.63/0.74 | 5.9× | 0.23 | 0.657 |  |
| lr_ns_best | FULL | 0.01/0.05/0.08/0.19 | 52.5× | 0.53 | 5.181 | 25112422.0 (≤100000 💥EXPLODED) |
| lr_os_best | STEP-1 | 0.12/0.33/0.42/0.51 | 5.3× | 0.25 | 0.857 |  |
| lr_os_best | FULL | 0.01/0.05/0.09/0.21 | 6.2× | 0.56 | 27.377 | 34387848.0 (≤8922950 💥EXPLODED) |
