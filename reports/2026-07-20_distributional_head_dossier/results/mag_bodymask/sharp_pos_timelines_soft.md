# Sharpness scorecard

pred_dir: `/home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260727_203007`

**CONFIG (auto, from sidecar):** output_distribution=nb · reg_activation=None · static_channels=[] · dropout_rate=0.15 · input_channels=3

| target | subset | FSS@[1, 3, 5, 11] | area_ratio | conc1% | MCR | max (explode?) |
|--------|--------|-----------|-----------|--------|-----|----------------|
| lr_sb_best | STEP-1 | 0.03/0.06/0.08/0.11 | 0.5× | 0.43 | 0.258 |  |
| lr_sb_best | FULL | 0.01/0.01/0.02/0.03 | 0.3× | 0.65 | 0.026 | 7664409.0 (≤5669750 💥EXPLODED) |
| lr_ns_best | STEP-1 | 0.07/0.17/0.25/0.38 | 0.5× | 0.48 | 0.053 |  |
| lr_ns_best | FULL | 0.01/0.01/0.02/0.03 | 0.2× | 0.76 | 0.061 | 14899548.0 (≤100000 💥EXPLODED) |
| lr_os_best | STEP-1 | 0.00/0.00/0.00/0.00 | 0.2× | 0.57 | 0.269 |  |
| lr_os_best | FULL | 0.00/0.00/0.00/0.01 | 0.1× | 0.76 | 6.227 | 4467379.5 (≤8922950) |
