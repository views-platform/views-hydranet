# C-224 tail diagnostic — `violet_visitor` vs the FAO-02 climatology

Epic #263 / S5 (#269). Target `sb`. Taillardat2023 §3.3: CRPS treated as a random variable, its **distribution** compared rather than its expectation.

> **`diag_Tu` is not used in any decision rule in this dossier.** It is a DIAGNOSTIC. `verdict_token` reads no `diag_*` key, and `test_no_diag_column_reaches_the_decision_rule` asserts that by inspecting its source.

| h | q | u | m (model) | m (ref) | gamma model | gamma ref | **diag_Tu** |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.99 | 0.9391 | 1365 | 2044 | 0.732 | 0.723 | **-0.2957** |
| 1 | 0.995 | 4.016 | 715 | 990 | 0.680 | 0.642 | **-1.0125** |
| 1 | 0.999 | 29.33 | 133 | 208 | 0.720 | 0.602 | **+0.5480** |
| 18 | 0.99 | 1 | 1264 | 2139 | 0.617 | 0.683 | **+0.1619** |
| 18 | 0.995 | 5 | 673 | 942 | 0.592 | 0.588 | **-1.0113** |
| 18 | 0.999 | 31.32 | 139 | 202 | 0.517 | 0.380 | **+0.4608** |
| 36 | 0.99 | 1.874 | 1441 | 1968 | 0.958 | 0.945 | **+0.4515** |
| 36 | 0.995 | 6 | 693 | 953 | 0.954 | 0.937 | **+0.8575** |
| 36 | 0.999 | 40.92 | 133 | 208 | 0.964 | 0.949 | **+0.5575** |

`T_u > 0` means the model's CRPS tail is *further* from a fitted GPD than the reference's. **It does NOT mean better.** Taillardat §3.3 is explicit that an inflated, mis-calibrated 'extremist' forecaster scores HIGH — pinned by the green test `test_extremist_forecast_gets_a_HIGH_index`, whose passing condition is that this metric is gameable.

`n/a` means the index is **undefined**, not bad: the pooled threshold left one arm with fewer than 50 exceedances, so no GPD could be fitted to it.
