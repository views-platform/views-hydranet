# 06 — Bloom-Verification Verdict (Epic #193, S6 #199 → S7 #200)

**Date:** 2026-07-27 · **Status:** verdict recorded; S8 close-out (governance) pending user review.

## Question (verbatim, from the epic)
> "completely productionize the solution … run three seeds at every model config … count across
> those and see: have we fixed the bloom?"

## What ran
18 free-running 36-step rollouts, inference-only, on **freshly-retrained known-seed** artifacts
(S5b: 3×nb + 3×zinb, all at a **matched 40-lesson** budget; seeds 42/43/44 persisted in each
`.pt.config.json` sidecar — C-112 closed). Matrix:

| Arm | family + composition (ADR-069) | seeds | feedback |
|-----|--------------------------------|-------|----------|
| gated_NB | `nb` + `soft_gate` | 42,43,44 | mean, sample |
| th_gated_NB | `nb` + `threshold_gate` (**τ=0.5**, see note) | 42,43,44 | mean, sample |
| ZINB | `zinb` + `self_zeroed` | 42,43,44 | mean, sample |

Composition applied **in-model at emit** (ADR-069 composer). Scored on the frozen per-horizon ruler
`tools/rollout_skill_score.py` (reuses the lodestar crps/AP/Brier primitives verbatim; h=1 == lodestar
T=0 by construction — selftest-proven). Results: `results/bloomverify/{bloomverify,traj}_<arm>_<seed>_<mode>.csv`.

## Counted verdict

Bloom fingerprint = **field-wide** wrong-zero mass: `crps_none(h)` (CRPS on the zero cells, EXP-2's
signature) + `M_mean(h)` (field-mean emitted log1p magnitude). "Bounded" = max_h crps_none < 0.5 AND
max_h M_mean < 5.0 (worst target). Per (arm, seed), mean vs sample, worst-target max over h=1..36:

| Arm | seed | MEAN max crps_none | MEAN max M_mean | MEAN | SAMPLE max crps_none | SAMPLE max M_mean | SAMPLE |
|-----|------|-----:|-----:|:--:|-----:|-----:|:--:|
| gated_NB | 42 | 36.00 | 285.2 | **BLOOMS** | 0.003 | 0.03 | bounded |
| gated_NB | 43 | 58.31 | 462.1 | **BLOOMS** | 0.019 | 0.18 | bounded |
| gated_NB | 44 | 63.40 | 502.7 | **BLOOMS** | 0.004 | 0.03 | bounded |
| th_gated_NB | 42 | 36.09 | 285.9 | **BLOOMS** | 0.009 | 0.07 | bounded |
| th_gated_NB | 43 | 65.47 | 518.8 | **BLOOMS** | 0.018 | 0.17 | bounded |
| th_gated_NB | 44 | 63.65 | 504.7 | **BLOOMS** | 0.002 | 0.02 | bounded |
| ZINB | 42 | 43.28 | 342.7 | **BLOOMS** | 0.105 | 0.70 | bounded |
| ZINB | 43 | 50.48 | 400.0 | **BLOOMS** | 0.342 | 2.32 | bounded |
| ZINB | 44 | 94.73 | 751.1 | **BLOOMS** | 0.351 | 2.49 | bounded |

> **mean-feedback BLOOMS in 9/9 arms; sample-feedback BOUNDED in 9/9 arms.**
> **⇒ The bloom is FIXED by sample-feedback (the productionized default, ADR-070) across every
> deployable arm and all three seeds.** This reproduces and generalizes EXP-2 on retrained,
> known-seed models — retiring "the bloom is a τ tuning problem"; it was an exposure-bias BUG, now
> fixed at the feedback boundary.

## Pre-registered falsifiers (05e)
- **F-B1** (sample fails to bound) — **did not fire** (9/9 bounded).
- **F-B2** (T=0 leak): **flagged, root-caused, then FIXED.** The S6 cross-process eval showed h=1
  crps_all *not exactly* identical mean vs sample (median Δ=0.0023; one os outlier, gated_NB_44_mean
  h1=0.44 vs sample 0.03). Investigation: the T=0 **distribution** (emit-mean, gate, params) is
  byte-identical by construction (ADR-070 §3 — in-process CPU test byte-exact), but the scored D×K
  **sample cube** was not — `to_cube_samples` drew the whole 36-step trajectory from ONE shared
  `torch.Generator`, and torch's batched Gamma rejection coupled h=1's draws to the h≥2 params that
  feedback changes. The gated_NB_44 outlier was one anomalous zero-cell magnitude in that GPU run
  (crps_events identical; only crps_none differed). **Fixed** — per-`(pass, step)` sub-generator
  seeding, commit `66a95ea`; the h=1 cube is now byte-identical mean vs sample for all 3 deployable
  compositions (regression test in `test_rollout_feedback.py`). **F-B2 no longer fires.** Verdict
  unaffected (field-wide magnitude; S6 cubes not re-scored).

## Honesty note — a verdict-criterion deviation I caught (did NOT report as truth)
The S6 driver's throwaway inline auto-verdict (`s6_score_one.py`, tmp, **never committed**) *deviated
from the 05e criterion*: it used `M_max` (max over cells → a single outlier cell reads as millions
while the field `M_mean` is ~0) instead of the pre-registered `M_mean`, and applied
`crps_all(36) ≥ 5×crps_all(1)` **without** the pre-registered terminal-spike carve-out (crps_all
degradation at long h is truth-driven: `crps_events` h36 sb = 84.63 is *identical* for mean and
sample). By that deviating criterion every arm "BLOOMS" including sample — the opposite of the truth.
Applied **as written** (`M_mean` + carve-out) the pre-registered criterion gives 9/9 & 9/9;
corroborated by `crps_none`. Caught by not trusting the auto-print; the repo scorer
(`rollout_skill_score.py`) was never affected; the banked CSV data is correct.

## Close-out (S8, 2026-07-27)
- **τ=0.5** (corrected): th_gated_NB ran at the validated τ=0.5 (baserate is a documented no-op —
  #167 exp-log); 05e §arms wording corrected in place.
- **F-B2 resolved** by the sampler fix (`66a95ea`) — see above.
- **C-113 → evidenced mitigation** (bloom bounded 9/9 by default via ADR-070; io-gain>1 durable fix
  separate/open); **C-121 → resolved** (the regression guard now covers the actual fix).
- Tooling promoted to `tools/`; ADR-070 active; dossier `git add -f`'d; runbook written.
