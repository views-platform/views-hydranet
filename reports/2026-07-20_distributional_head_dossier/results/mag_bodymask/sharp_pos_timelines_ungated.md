# Sharpness scorecard

pred_dir: `/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260727_203007`

**CONFIG (auto, from sidecar):** output_distribution=nb · reg_activation=None · static_channels=[] · dropout_rate=0.15 · input_channels=3

| target | subset | FSS@[1, 3, 5, 11] | area_ratio | conc1% | MCR | max (explode?) |
|--------|--------|-----------|-----------|--------|-----|----------------|
| lr_sb_best | STEP-1 | 0.08/0.31/0.46/0.66 | 8.3× | 0.45 | 0.295 |  |
| lr_sb_best | FULL | 0.01/0.02/0.03/0.04 | 3.7× | 0.72 | 0.029 | 16682724.0 (≤5669750 💥EXPLODED) |
| lr_ns_best | STEP-1 | 0.04/0.11/0.18/0.34 | 10.0× | 0.54 | 0.104 |  |
| lr_ns_best | FULL | 0.00/0.01/0.01/0.02 | 5.6× | 0.87 | 0.008 | 8676320.0 (≤100000 💥EXPLODED) |
| lr_os_best | STEP-1 | 0.04/0.17/0.30/0.54 | 18.8× | 0.34 | 0.470 |  |
| lr_os_best | FULL | 0.00/0.01/0.01/0.03 | 3.2× | 0.89 | 5.327 | 5192769.0 (≤8922950) |
