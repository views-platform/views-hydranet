# 03 — Harness and invariants

**Date:** 2026-08-15 · **Epic:** #263 · **Status:** S0

---

## 1. The standing harness (already exists — reuse, do not rebuild)

This programme inherits an unusually strong harness. **Discovering what was already there is most of why this
epic is 19 hours and not 19 days.**

| Guardrail | Where | Status |
|---|---|---|
| Frozen metric primitives + a freeze self-test | `2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py::_selftest` | **frozen** |
| `h=1` byte-reproduces the frozen T=0 ruler | `tests/test_score_v2_horizons.py::test_h1_matches_frozen_lodestar` | green |
| Identical cross-arm support at every horizon | `rollout_skill_score._support_keys` + intersection | in both v2 tools |
| Cheap support enumeration (identifiers only, no cube load) | `_support_keys` | done |
| Ex-ante stratum, leakage-tested by outcome permutation | `gw_stratified.exante_stratum` | **C-248 CLOSED** |
| Origin-block bootstrap, asserted wider than iid under clustering | `gw_stratified._bootstrap_mean_ci(resample="origin")` | **C-253 CLOSED** |
| OOM guard — per-cell vectors, never cubes | `del g` at `score_v2_horizons:149`, `rollout_skill_score:177`, `gw_stratified:183` | **C-252 closed by construction** |
| 21-column metric row incl. the whole occurrence axis (`act_ratio`, `precision_at_k`, `mag_on_false_pos`) | `score_v2_horizons._metric_row` + `activation_metrics.py` | **DRAFT §4.1's occurrence axis, already built** |
| Degenerate-forecast red-team | `tests/test_activation_metrics.py` — 8 tests | **DRAFT §5, already implemented** |
| Pinned truth hash | `v2_ruler.V2_TRUTH_SHA256` | done |
| Repo-root resolution pattern (never absolute) | `gw_stratified.py:128` `parents[3]` + its guard test | done — copy it |
| Paired CRPS significance, CI-visible tests | `scripts/crps_significance.py` + 6 tests | the model for `scripts/rollout_ruler_core.py` |
| Exceedance-conditional tail diagnostic | `scripts/tail_scorecard.py` (tracked, C-224-labelled, **untested**) | exists — Taillardat is *additive* |

## 2. The harness gap this dossier closes

**CI has zero coverage of the ruler, silently.** `reports/` is gitignored, so both
`tests/test_gw_stratified.py` and `tests/test_score_v2_horizons.py` call
`pytest.skip(allow_module_level=True)` when the tools are absent — which is always, in a clean clone.

**Resolution without touching `views_hydranet/`:**

| Layer | Location | Tracked | Tests |
|---|---|---|---|
| **Pure functions** — no I/O, no paths, no cube reads | `scripts/rollout_ruler_core.py` | **yes** | `tests/test_rollout_ruler_core.py` — **runs in CI, never skips** |
| **Drivers** — cube paths, cross-repo config reads, CSV writing | `<this dossier>/tools/*.py` | no | `tests/test_rollout_ruler_trust.py` — skip-guarded |

`scripts/` is tracked, is not the package (no `__init__.py`, not in `[project]`), and already hosts exactly
this pattern. A later promote to `views_hydranet/evaluation/` is `git mv` + one import line, tests unchanged.

## 3. Invariant taxonomy

**Hard (never violated):** the frozen `lodestar_score` primitives · `h=1 ≡ frozen T=0` · identical cross-arm
support · per-cell vectors not cubes past the arm loop · pre-origin-only stratification and climatology.

**Intentionally added by this programme:** a reference forecast · a skill score · `zero_share_of_gap` on every
headline row · provenance assertions · one DIAGNOSTIC-only tail index.

**Respect while changing:** the `h=1 → frozen lodestar` provenance chain is the ruler's credibility — it is
chain-of-custody, not tidiness. Do not refactor the four tools into one (`SCOPE.md` #18).

## 4. The 11 cluster-16 invariants → guard map

| ID | Tier | Status today | This epic | Guard | Story |
|---|:--:|---|---|---|:--:|
| **C-217** | 2 | ✅ **ASSERTED (S2)** — was cleared by *reasoning* only | `partition_audit` raises on `min_emitted_month <= train_max`; 6/6 arms `leak: false` | 3 pytest | S2 |
| **C-218** | 2 | ✅ **ASSERTED (S2)** — was read by hand | raises unless `rollout_feedback == 'sample'` or `diagnostic_only=True`; 6/6 arms `sample` | 2 pytest (synthetic model dirs — the cross-repo read stays runtime-only) | S2 |
| **C-219** | 2 | ✅ **ENFORCED (S3)** — was a norm with no code | `require_headline_columns` **raises** on a row lacking the split, AP, CRPSS or `zero_share_of_gap` | 2 pytest, CI-visible | S3 |
| **C-220** | 3 | ✅ **ASSERTED (S2)** — was **never tested at all** | pure `assert_sample_cube` in tracked `scripts/`; 468 cubes checked, all `(N, 16)` | 4 pytest, **CI-visible (never skips)** | S2 |
| **C-221** | 3 | ✅ **MDE STATED (S4)** | `results/MDE.md` — MDE **0.00596** at sb/h36, 90%, P=13; the latent single-arm-support defect pinned as **C-277** (`xfail(strict=True)`, not fixed) | 1 pytest (xfail) + doc | S4 |
| **C-224** | **1** | ✅ **DIAGNOSTIC EXISTS (S5)** — Tier-1 governance ask **UNCHANGED, still open** | Taillardat §3.3 index; 9 numbers in `results/tail_index.md`; 3 structural railguards | 8 pytest, CI-visible; impl 109/120, test 118/120 | S5 |
| **C-231** | 3 | ✅ **METRIC EXISTS (S1)** — was a finding with no metric | `crps_gap_decomposition` → `zero_share_of_gap`; identity holds to 8.9e-16 | 6 pytest, CI-visible | S1, S3 |
| **C-248** | **1** | **CLOSED** — and now **inherited (S3)** | the climatology draws strictly pre-origin; proven byte-identical under post-origin outcome permutation | existing + 1 pytest | S3 |
| **C-252** | 3 | ✅ **EXPLICIT (S4)** — was implicit only | `tracemalloc` asserts the significance path grows far below an (N,16) cube at N=195k | 1 pytest | S4 |
| **C-253** | 2 | **CLOSED** | reuse, **do not touch** | existing | — |
| **C-254** | 3 | ✅ **POWER STATED (S4)** | `MDE.md` states the MDE and the Giacomini fixed-scheme sentence; a null can no longer be read as 'no difference' by default | doc | S4 |

**Accounting: 10 pytest guards + 1 runtime guard.** That is the honest reading of the epic's "10 unblocked
invariants" — C-218 cannot be a portable pytest without an env var that would make it skip in CI, which is the
problem, not the solution.

## 5. Pre-flight checklist

Before **any** cube is scored (S6):

- [x] `05_analysis_plan.md` carries `LOCKED 2026-08-15` — **S0**
- [x] `SCOPE.md` exists with the 22 exclusions and the PARKED table — **S0**
- [x] decomposition identity reconciles `<1e-9` on every archived V2 row (**8.9e-16**) — **S1**
- [x] `partition_audit.json` green for **6/6** arms: `leak: false`, `rollout_feedback: 'sample'`,
      truth sha matches the pin, one artifact per arm — **S2**
- [x] climatology is byte-identical under post-origin outcome permutation — **S3**
- [x] `crps_skill_score` raises on a 1-sample reference — **S3**
- [x] `MDE.md` states a number (**0.00596** at sb/h36) — **S4**
- [x] no `diag_*` key reaches the decision rule — **S5**
- [x] full suite green (1362); `scripts/`-backed tests **run, not skip** with `reports/` absent — **S4**

---

## 6. DRAFT §4 — the metric battery, with a scope column

Absorbed from `reports/2026-08-13_evaluation_pitfalls_and_metric_battery_DRAFT.md` §4. **The scope column is
the point** — the battery is a research agenda; this dossier delivers a bounded slice of it.

| # | Battery item (DRAFT §4) | Disposition here |
|:--:|---|---|
| 1 | **Decompose, always** — occurrence (`P(y>0)`, AP/AUPRC, recall/precision) · conditional magnitude · calibration | **PARTLY DELIVERED.** Occurrence axis **already exists** (`activation_metrics.py` + `_metric_row`'s `act_ratio`/`precision_at_k`/`mag_on_false_pos`). Conditional magnitude partly via `size_ratio`/`mcr_*`. Calibration (PIT) is **OUT** — `SCOPE.md` #14. |
| 2 | **Score the deployed object, per step, over the full horizon** — the sample field, never pooled | **DELIVERED** — C-220 assertion (S2) + 7 pre-registered horizons. *Not* all 36 (`SCOPE.md` #17). |
| 3 | **Spatial verification** — FSS, SAL/MODE, Gini/entropy | **OUT** — `SCOPE.md` #13. Not in cluster 16. |
| 4 | **Discrimination that survives sparsity** — AUPRC; check dynamic range before ranking | **PARTLY** — AP is in `_metric_row`; the dynamic-range check is what `zero_share_of_gap` operationalises. |
| 5 | **Proper scores WITH a power statement** — stratified GW on an ex-ante stratum, reporting the MDE | **DELIVERED** — `gw_stratified` (C-248/C-253 closed) + `MDE.md` (S4). Explicitly *not* upgraded to an HAC regression (`SCOPE.md` #12). |
| 6 | **Reproducibility gates** — ≥3 seeds, validation partition, determinism, cache fingerprinting | **PARTLY.** Truth-hash pin delivered (S2). ≥3 seeds: only 2 gated_NB seeds survive ⇒ results labelled INDICATIVE. Validation partition is **OUT** (`SCOPE.md` #10 — it would leak). |
| 7 | **Predictability baselines** — persistence/climatology confound, per-axis predictability | ✅ **DELIVERED (S3)** — `climatology_resample`, a scorer-side stand-in for views-baseline's `ConflictologyModel` (canonical params; 0.9591 vs its archived 0.9601). Drops into the frozen `_metric_row` unchanged. Duplication tracked as **C-279**. |
| 8 | **The oracle-input rollout, every time you diagnose a rollout** | **ALREADY EXISTS** — `rollout_feedback='teacher_forced'`; EXP-SS-2 (2026-08-14) used it to localise the root by elimination. Nothing to build. |
| 9 | **Implementation verification is part of the evaluation** — parity tests, sampler sanity, enough origins | **DELIVERED as process** — S2 runs before any scoring; every pure function ships with a hand-computed test. This is meta-pattern 8's countermeasure. |

## 7. DRAFT §5 — the degenerate-forecast red-team

The DRAFT's key methodological proposal: *"treat the metric suite itself as software under test, and unit-test
it with adversarial/degenerate forecasts whose true quality is known… A metric that cannot separate these is
blind and must not be used for selection."*

**Status: already 2/3 implemented.** `tests/test_activation_metrics.py` (8 tests) covers 4 of the 6
degenerates for the activation metrics. Per `SCOPE.md` #16, this dossier builds **only the two the new
functions need**:

| Degenerate | Covered by | Story |
|---|---|---|
| all-zero | **new** — `test_degenerate_all_zero_forecast_scores_badly_on_skill` | S3 |
| climatology | **new** — it *is* the reference (S3) | S3 |
| persistence | `_persistence_gathered` + existing tests | — |
| over-concentrated spiky (*the #258 failure*) | `test_activation_metrics.py` | — |
| frequency-right/magnitude-wrong | `test_activation_metrics.py` | — |
| magnitude-right/frequency-wrong | `test_activation_metrics.py` | — |

The all-zero test is load-bearing: it must show `crpss < 0` **while the raw `crps_all` looks unremarkable** —
that single assertion is what turns C-219/C-231 from a norm into a code invariant.

## 8. Metrics recorded as BLIND — barred from selection

Populated by the epic's own evidence. **A metric listed here must not be used to select or rank.**

| Metric | Blind to | Evidence | Story |
|---|---|---|---|
| **`crps_all` alone** | whether a "win" is event skill or confident zeros | At h=36, 12/12 arm×target rows clear FAO-02's 5% superiority bar on a statistically real gap while ranking conflict **worse**. `violet_visitor` sb h36: CRPSS +0.079, CI excludes 0, `zero_share` 0.835, ΔAP −0.051, `size_ratio` 0.0. | S1, S6 |
| **`crps_none`** | anything — it rewards being empty on an empty field | `violet_visitor` sb h36 `crps_none` = **1.85e-07**, i.e. numerically perfect, from a forecast that fires 14,000× too rarely | S6 |
| **`crps_events` as a selection metric** | — (improper by construction) | Conditioning on the outcome is the Forecaster's Dilemma (Lerch2017 §2.3, catalog C1). Already FAO-02-excluded; restated here because the split makes it *tempting*. Display only. | S0 |
| **`MCR` / `size_ratio` alone** | spatial over-concentration and activation frequency | DRAFT catalog A1/A2; `size_ratio` reads 0.0000 at h≥18 for every arm, which is *correct* and still says nothing about placement | absorbed |
| `diag_Tu` (Taillardat) | — **DIAGNOSTIC, never a selection metric** | Gameable by construction: an inflated mis-calibrated forecaster scores **higher** (Taillardat §3.3), pinned by a green test. `verdict_token` reads no `diag_*` key. | S5 |

**What survived as trustworthy for selection:** the `crps_all` / `crps_events` / `crps_none` **split** read
together, **AP** (occurrence ranking), **`crpss_vs_clim`** (a skill score with a real denominator), and
**`zero_share_of_gap`** — always jointly, never one alone. `require_headline_columns` enforces exactly this.
