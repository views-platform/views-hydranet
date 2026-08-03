# Sharpness scorecard

pred_dir: `/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260727_194847`

**CONFIG (auto, from sidecar):** output_distribution=nb · reg_activation=None · static_channels=[] · dropout_rate=0.15 · input_channels=3

| target | subset | FSS@[1, 3, 5, 11] | area_ratio | conc1% | MCR | max (explode?) |
|--------|--------|-----------|-----------|--------|-----|----------------|
| lr_sb_best | STEP-1 | 0.01/0.01/0.01/0.01 | 0.1× | 0.49 | 0.009 |  |
| lr_sb_best | FULL | 0.01/0.01/0.01/0.01 | 0.0× | 0.83 | 0.003 | 9869756.0 (≤5669750 💥EXPLODED) |
| lr_ns_best | STEP-1 | 0.00/0.00/0.00/0.00 | 0.1× | 0.49 | 0.004 |  |
| lr_ns_best | FULL | 0.00/0.00/0.00/0.00 | 0.0× | 0.87 | 13.403 | 30509034.0 (≤100000 💥EXPLODED) |
| lr_os_best | STEP-1 | 0.00/0.00/0.00/0.00 | 0.2× | 0.58 | 0.015 |  |
| lr_os_best | FULL | 0.00/0.00/0.00/0.00 | 0.1× | 0.77 | 0.001 | 7099826.0 (≤8922950) |
