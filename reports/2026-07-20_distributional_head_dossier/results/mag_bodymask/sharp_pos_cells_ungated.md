# Sharpness scorecard

pred_dir: `/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260727_200938`

**CONFIG (auto, from sidecar):** output_distribution=nb · reg_activation=None · static_channels=[] · dropout_rate=0.15 · input_channels=3

| target | subset | FSS@[1, 3, 5, 11] | area_ratio | conc1% | MCR | max (explode?) |
|--------|--------|-----------|-----------|--------|-----|----------------|
| lr_sb_best | STEP-1 | 0.19/0.54/0.67/0.80 | 33.0× | 0.25 | 0.757 |  |
| lr_sb_best | FULL | 0.02/0.07/0.11/0.23 | 54.4× | 0.38 | 3.036 | 32709072.0 (≤5669750 💥EXPLODED) |
| lr_ns_best | STEP-1 | 0.11/0.38/0.54/0.72 | 142.7× | 0.23 | 0.751 |  |
| lr_ns_best | FULL | 0.01/0.03/0.06/0.14 | 67.3× | 0.53 | 2.140 | 52920496.0 (≤100000 💥EXPLODED) |
| lr_os_best | STEP-1 | 0.07/0.28/0.39/0.53 | 90.5× | 0.29 | 1.250 |  |
| lr_os_best | FULL | 0.01/0.03/0.07/0.18 | 12.4× | 0.56 | 0.505 | 20316228.0 (≤8922950 💥EXPLODED) |
