# C-224 tail diagnostic — `violet_visitor` vs the FAO-02 climatology

Epic #263 / S5 (#269). Target `sb`. Taillardat2023 §3.3: CRPS treated as a random variable, its **distribution** compared rather than its expectation.

> **`diag_Tu` is not used in any decision rule in this dossier.** It is a DIAGNOSTIC. `verdict_token` reads no `diag_*` key, and `test_no_diag_column_reaches_the_decision_rule` asserts that by inspecting its source.

| h | q | u | m (model) | m (ref) | gamma model | gamma ref | **diag_Tu** |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | 0.99 | 1 | 1194 | 2047 | 0.687 | 0.682 | **-7.5787** |
| 1 | 0.995 | 4.652 | 691 | 1013 | 0.700 | 0.596 | **-2.7783** |
| 1 | 0.999 | 30.82 | 129 | 212 | 0.734 | 0.522 | **-1.6630** |
| 18 | 0.99 | 1.254 | 1264 | 2140 | 0.637 | 0.653 | **-1.8059** |
| 18 | 0.995 | 5 | 673 | 1028 | 0.592 | 0.521 | **+0.1055** |
| 18 | 0.999 | 33 | 129 | 205 | 0.500 | 0.301 | **-0.4665** |
| 36 | 0.99 | 2 | 1201 | 1920 | 0.951 † | 0.936 † | **+0.4193** |
| 36 | 0.995 | 6.074 | 693 | 1008 | 0.954 † | 0.928 † | **-0.2967** |
| 36 | 0.999 | 43.08 | 128 | 213 | 0.965 † | 0.951 † | **-0.7905** |

`T_u > 0` means the model's CRPS tail is *further* from a fitted GPD than the reference's. **It does NOT mean better.** Taillardat §3.3 is explicit that an inflated, mis-calibrated 'extremist' forecaster scores HIGH — pinned by the green test `test_extremist_forecast_gets_a_HIGH_index`, whose passing condition is that this metric is gameable.

`n/a` means the index is **undefined**, not bad: the pooled threshold left one arm with fewer than 50 exceedances, so no GPD could be fitted to it.

**† marks a row where the fitted `gamma` is at the PWM saturation ceiling.** `gpd_pwm_fit` is a probability-weighted-moment estimator: it is consistent only for `gamma < 0.5`, and its `a1` moment ceases to exist at `gamma >= 1`. Measured on exact GPD quantiles it saturates — a true shape of 0.9/1.0/1.1/1.3 fits to 0.86/0.92/0.96/0.99. A marked `gamma` is therefore a **lower bound**, not an estimate, and cannot distinguish a heavy tail from an infinite-mean one (`gamma >= 1`) — which is the regime this diagnostic exists to probe. A scipy MLE fit valid across this range already exists at `reports/2026-07-15_volatility_ceiling_dossier/tools/s5_tail.py::gpd_xi`; nothing cross-checks the two (registered).
