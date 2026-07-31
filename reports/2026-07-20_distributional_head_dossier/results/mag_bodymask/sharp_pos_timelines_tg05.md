# Sharpness scorecard

pred_dir: `/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260727_203007`

**CONFIG (auto, from sidecar):** output_distribution=nb · reg_activation=None · static_channels=[] · dropout_rate=0.15 · input_channels=3

| target | subset | FSS@[1, 3, 5, 11] | area_ratio | conc1% | MCR | max (explode?) |
|--------|--------|-----------|-----------|--------|-----|----------------|
| lr_sb_best | STEP-1 | 0.01/0.01/0.01/0.01 | 0.3× | 0.48 | 0.259 |  |
| lr_sb_best | FULL | 0.01/0.01/0.01/0.01 | 0.2× | 0.66 | 0.026 | 6492230.0 (≤5669750 💥EXPLODED) |
| lr_ns_best | STEP-1 | 0.00/0.00/0.00/0.00 | 0.3× | 0.45 | 0.086 |  |
| lr_ns_best | FULL | 0.00/0.00/0.00/0.00 | 0.1× | 0.77 | 0.107 | 14174386.0 (≤100000 💥EXPLODED) |
| lr_os_best | STEP-1 | 0.00/0.00/0.00/0.00 | 0.1× | 0.70 | 0.301 |  |
| lr_os_best | FULL | 0.00/0.00/0.00/0.00 | 0.1× | 0.84 | 3.140 | 2948480.5 (≤8922950) |
