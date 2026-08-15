# C-224 tail diagnostic — `violet_visitor` vs the FAO-02 climatology

Epic #263 / S5 (#269). Target `sb`. Taillardat2023 §3.3: CRPS treated as a random variable, its **distribution** compared rather than its expectation.

> **`diag_Tu` is not used in any decision rule in this dossier.** It is a DIAGNOSTIC. `verdict_token` reads no `diag_*` key, and `test_no_diag_column_reaches_the_decision_rule` asserts that by inspecting its source.

| h | q | u | m (model) | m (ref) | gamma model | gamma ref | **diag_Tu** |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.99 | 0.9079 | 1380 | 2029 | 0.733 | 0.711 | **+0.0406** |
| 1 | 0.995 | 4.145 | 713 | 992 | 0.686 | 0.642 | **-2.5651** |
| 1 | 0.999 | 29 | 134 | 206 | 0.718 | 0.593 | **+0.7690** |
| 18 | 0.99 | 1 | 1264 | 2121 | 0.617 | 0.673 | **+0.2413** |
| 18 | 0.995 | 5 | 673 | 934 | 0.592 | 0.573 | **-1.0423** |
| 18 | 0.999 | 32 | 135 | 205 | 0.512 | 0.419 | **+0.4015** |
| 36 | 0.99 | 1.83 | 1441 | 1968 | 0.958 | 0.945 | **+0.5800** |
| 36 | 0.995 | 6 | 693 | 953 | 0.954 | 0.939 | **+0.7702** |
| 36 | 0.999 | 41.25 | 133 | 208 | 0.964 | 0.954 | **+0.7262** |

`T_u > 0` means the model's CRPS tail is *further* from a fitted GPD than the reference's. **It does NOT mean better.** Taillardat §3.3 is explicit that an inflated, mis-calibrated 'extremist' forecaster scores HIGH — pinned by the green test `test_extremist_forecast_gets_a_HIGH_index`, whose passing condition is that this metric is gameable.

`n/a` means the index is **undefined**, not bad: the pooled threshold left one arm with fewer than 50 exceedances, so no GPD could be fitted to it.
