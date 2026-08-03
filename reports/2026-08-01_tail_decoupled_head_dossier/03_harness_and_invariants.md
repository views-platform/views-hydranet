# 03 — Harness & invariants (the crown jewel)

**Date:** 2026-08-01 · **Verdict:** pre-flight checklist **RED** (build gaps C1–C4 open). Audit finding:
**~75% of the standing harness already exists**; the new work is the mixture family + its numerics/tests
and the proper-metric scorer extension.

## A. Invariant taxonomy

### Hard invariants — never break (violation ⇒ experiment invalid)
- **Baseline byte-identical when the new family is off.** Legacy + NB/ZINB paths unchanged; the mixture
  is inert unless `output_distribution="mixture_nb"`. (Proven by a parity anchor — see C4.)
- **Fail-loud, no silent clamp/degradation** (NaN/Inf in NLL or sampler → raise, not mask).
- **Proper-scoring discipline (FAO-02):** primary metric proper; **no selection on `crps_events`**;
  no twCRPS/PIT/LogScore. The GW test is on a stratum defined by an **ex-ante** covariate.
- **Reproducibility:** seed-locked, deterministic-algorithms on; 3 seeds; (bitwise GPU repro NOT
  guaranteed — 3-seed spread is the uncertainty).
- **Full suite + ruff green; CICs/ADR synced.**

### Deliberately changed by this program (behind the family flag)
- Adds a **5-param head** (vs NB 2 / ZINB 3) and a **mixture NLL** — a new `output_distribution` value,
  default-off. We intend to *replace the single mean-tied tail* with a mean-decoupled 2-component tail
  **for this arm only**; reviewers should not defend the single-NB tail as sacred here.
- Adds a **stratified-proper + GW** readout to the v2 ruler — an *addition*, existing columns unchanged.

### Respect while changing (breakable in passing)
- **Rollout stability (ADR-070 sample-feedback):** a heavier tail component fed back could re-arm the
  exposure-bias bloom (C-MR4). Guard: `crps_none` per-horizon must stay flat (the bloom signature).
- **D×K sampler determinism / per-(pass,step) seeding** — the mixture sampler must preserve it.
- **The gate composition** — the mixture replaces the *body*, not the gate; gate AP must not move for a
  reason other than a better body.

## B. Standing harness — what already EXISTS (reuse)
| Mechanism | Status | Where |
|---|---|---|
| Default-off feature flag | ✅ | `config_initializer.py` `output_distribution` (default `standard`) + `n_head_samples`/`max_posterior_cube_gb` |
| Extension seam (OCP) | ✅ | `distributions/registry.py` — "a new family is added in exactly ONE place" |
| Compose-NBCore template | ✅ | `zero_inflated_negative_binomial.py` (`logaddexp` NLL, `sample`/`sample_core`) — the mixture copies this |
| Family loss adapter | ✅ | `distributions/family_loss.py` (`FamilyLoss` over `family.nll`) |
| D×K sampler + disk guard | ✅ | `distributions/sampling.py`, `hydranet_inference.generate_posterior_samples`, `disk_guard.assert_cube_fits` |
| Param-health forensics | ✅ | `utils/training_forensics.py` `record_params` (+ `training_engine.py` site) — watch `w`, `μ2:μ1`, tail occupancy |
| Locked eval ruler + frozen truth | ✅ | `reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py` (h=1==T=0 anchor); `…/v2_truth/calibration_datafactory_df.parquet` |
| Reproducibility / parity discipline | ✅ (partial) | `tests/test_derivation_parity.py`, `test_score_v2_horizons.py`; determinism gate exists (confirm exact name during build) |
| Run discipline (one-at-a-time, trap-restore, inline-score-delete-cube) | ✅ | hardened driver pattern (`s8_population_ab.sh`) — **lives in `views-models`**, not this repo |

## C. New harness this program needs (gaps — gate the first run)
- **C1 — mixture NLL numerics + tests (blocker, the delicate part FIRST).** `logsumexp(log w + logNB₁,
  log(1−w) + logNB₂)`; hand-checked against a brute-force 2-NB mixture pmf on a small case; NaN/Inf
  guards; ordered-means `μ2=μ1+softplus(Δ)` unit-tested (μ2>μ1 always). Gradient finiteness at `w→{0,1}`.
- **C2 — mixture sampler + determinism test.** Component pick + `NBCore.sample`; preserves per-(pass,
  step) seeding; D×K cube shape/contract unchanged; `to_cube_samples` still valid.
- **C3 — ruler extension: stratified-proper column + GW test.** Add to `score_v2_horizons.py` (or a
  sibling): the ex-ante risk stratum (recent-intensity), per-decile proper `crps_all`, and the
  Giacomini–White conditional predictive ability statistic (NB vs mixture). Unit-test the GW stat
  against a known case; assert h=1 anchor unchanged.
- **C4 — parity anchor.** With `output_distribution` ≠ mixture, outputs byte-identical to the current
  baseline (the mixture path provably inert when off).

## D. Pre-flight checklist (must be GREEN before the FIRST 3-seed run)
- [ ] **C1** mixture NLL + ordered-means implemented + unit-tested (numerics first) — **blocker**
- [ ] registered via `DISTRIBUTION_REGISTRY` (OCP), dispatcher untouched
- [ ] behind `output_distribution="mixture_nb"`; **C4** baseline byte-identical when off
- [ ] fail-loud NaN/Inf + device guards; no silent degradation
- [ ] **C2** sampler + determinism test green; D×K contract intact
- [ ] full suite + ruff green; CIC/ADR synced
- [ ] **C3** ruler emits stratified-proper + GW; h=1==T=0 anchor still holds
- [ ] single-tile overfit smoke passes (can it fit one known heavy-conflict cell? — distinguishes
      "can't train it" from "no tail signal")
- [ ] `05` pre-registered (hypothesis + falsifiers + GW decision rule) vs the locked baseline
- [ ] C-MR1..C-MR6 noted for `register-risk`

## E. Rules of engagement
One variable (the head family) behind its flag · pre-register then run · **cheap readout first**
(single-tile overfit → only then 3×300) · falsifier honesty (a fired falsifier kills the hypothesis,
no ad-hoc rescue) · **magnitude gains must come from representation, never from a mask/clamp** (the mask
graveyard) · `crps_none` bloom guardrail live throughout the rollout.
