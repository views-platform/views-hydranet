# Sharpness scorecard

pred_dir: `/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260727_194847`

**CONFIG (auto, from sidecar):** output_distribution=nb · reg_activation=None · static_channels=[] · dropout_rate=0.15 · input_channels=3

| target | subset | FSS@[1, 3, 5, 11] | area_ratio | conc1% | MCR | max (explode?) |
|--------|--------|-----------|-----------|--------|-----|----------------|
| lr_sb_best | STEP-1 | 0.04/0.17/0.29/0.51 | 4.8× | 0.49 | 0.030 |  |
| lr_sb_best | FULL | 0.01/0.01/0.02/0.03 | 1.3× | 0.90 | 0.004 | 16238372.0 (≤5669750 💥EXPLODED) |
| lr_ns_best | STEP-1 | 0.02/0.06/0.09/0.22 | 6.6× | 0.64 | 0.024 |  |
| lr_ns_best | FULL | 0.00/0.00/0.01/0.01 | 1.0× | 0.89 | 0.002 | 19242246.0 (≤100000 💥EXPLODED) |
| lr_os_best | STEP-1 | 0.02/0.12/0.26/0.54 | 10.0× | 0.53 | 0.092 |  |
| lr_os_best | FULL | 0.00/0.01/0.01/0.03 | 2.3× | 0.89 | 1.781 | 6258205.5 (≤8922950) |
