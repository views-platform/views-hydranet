# ADR-070: `rollout_feedback` — inference-time sample-feedback as the C-113 bloom mitigation

**Status:** Active
**Date:** 2026-07-27 (accepted 2026-07-27)
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers
**Epic:** #193 · **Builds on:** ADR-067 (family subsystem), ADR-069 (composition axis) · **Mitigates:** C-113 (autoregressive runaway) · **Relates to:** C-121 (regression guard), `feedback_clamp_log1p` (safety rail), ADR-056/GTF (training-time, separate & parked)

## Summary (read this first — self-contained)

The 36-step autoregressive rollout **blooms** (C-113): the model feeds its own **diffuse emit-mean** `E[y]` back as the next input, that dense field is out-of-distribution vs the sparse (99.7%-zero) real data, and errors compound into a runaway. **`rollout_feedback`** governs what the AR loop feeds back:

- `mean` — the emit-mean `E[y]` (historical behavior; the bloom driver).
- `sample` — a single seeded, composition-aware **family draw** per cell (sparse, in-distribution) — the mitigation.
- `teacher_forced` — the realized `truth[o+t]` (oracle; diagnostic only, not deployable).

Feeding back a **sample** keeps every step sparse and in-distribution, which **bounds the rollout** (EXP-2: ~20× on zinb; held across models). **Decision:** `rollout_feedback` defaults to **`sample` for registered family heads** (`nb`/`zinb`), `mean`/raw for legacy heads. This is safe because **the fix is T=0-neutral by construction** (below), so the scored T=0 product is byte-unchanged.

## 1. Context

C-113 is the costliest recurring failure (open since June). Prior attempts: `freeze_h` (inert), `feedback_clamp_log1p` (a safety rail that pins to the ceiling → over-prediction, not a resolution). The bloom dossier (`reports/2026-07-25_t0_rollout_skill_dossier/`) diagnosed the true driver as **exposure bias**: the fed-back *mean* is a dense blur unlike real conflict. EXP-2/EXP-3 showed feeding back a **sample** eliminates the runaway (a fixable bug), while the residual long-horizon *skill* gap is a separate ceiling (out of scope here — this ADR governs **stability**, not accuracy).

## 2. Decision

Adopt **`rollout_feedback ∈ {mean, sample, teacher_forced}`** as an inference config axis (already scaffolded; this ADR governs it and sets the production default).

- **`sample`** draws one per-cell family sample (`family.sample`, k=1) each AR step, **composed exactly like the emit** (`forecast_composition`): `self_zeroed` (zinb native), `soft_gate`/`threshold_gate` mask the draw by the cls gate. Only the fed-back copy changes; the emitted/scored D×K cube is untouched.
- **Default:** resolves to **`sample`** when a **registered family** (`resolve_family` ≠ None) is active and the field is unset; **`mean`/raw** for legacy heads (byte-identical). Overridable for experiments.
- **Determinism:** the draw uses a `torch.Generator` seeded from `torch_seed` (+ MC-dropout pass index) — S2 #121 gate.
- **Fail-loud:** `sample` with a legacy `output_distribution` (no family) raises (a sample needs a family).
- **`teacher_forced`** feeds the real month-t input (oracle) — diagnostic only; never a production default.

**Valid family × composition arms** (per ADR-069): **gated_NB** (`nb`+`soft_gate`), **th_gated_NB** (`nb`+`threshold_gate`, **τ=0.5** — the validated ADR-068 value; τ=baserate is a documented no-op), **ZINB** (`zinb`+`self_zeroed`).

## 3. Rationale & integrity impact

**T=0-neutral by construction (the decisive property).** In `predict()`, the seed step (`t == origin`) emits the h=1 prediction (= the scored T=0) *before* any feedback value is computed; `rollout_feedback` only affects h≥2. Therefore defaulting `sample` on for family heads **cannot change the frozen-lodestar T=0 scores** — zero cost to the shipped product, all upside on rollout stability. This is what makes sample-on a safe default rather than a risky one.

> **Neutrality is byte-exact at two levels (S8 verification, 2026-07-27).** (1) The T=0 *distribution* — emit-mean `E[y]`, gate, activated params — is byte-identical mean vs sample by the ordering above (in-process test). (2) The scored D×K *sample cube* required one fix: `to_cube_samples` originally drew the whole 36-step trajectory from a single `torch.Generator`, and torch's batched Gamma rejection coupled h=1's draws to the feedback-changed h≥2 params, so the *scored* T=0 was not byte-invariant (S6 flagged this; pre-registered F-B2). Fixed by seeding a per-`(pass, step)` sub-generator (commit `66a95ea`) — the h=1 cube is now byte-identical mean vs sample across all three compositions (regression test in `test_rollout_feedback.py`). F-B2 no longer fires.

**Stability, not skill.** Bounding the rollout ≠ making it accurate. Long-horizon magnitude is a data ceiling (amount-ceiling wall); occurrence skill is recoverable-in-principle but not delivered by this ADR. `rollout_feedback=sample` is the *stability* fix; skill work (GTF/`ss_feedback`, magnitude head) is separate and out of scope.

## 4. Consequences

### ✅ Positive
- The bloom is mitigated **by default** for every family arm, at zero T=0 cost.
- One config axis, uniform across families/compositions; overridable; deterministic.
- Retires reliance on the degenerate `feedback_clamp_log1p` rail for the bloom.

### ⚠️ Negative
- Sample-feedback is a single stochastic path per MC-dropout pass; the honest predictive spread comes from the D×K ensemble, not one path (documented; the S-path ensemble is a future width knob).
- Legacy heads get no mitigation (they can't sample) — acceptable (legacy is scope-locked out).
- The mitigation is verified at T=0 calibration; validation-partition graduation is separate (M3).

## 5. Validation
Governed by Epic #193: unit + integration/regression tests that `sample` bounds where `mean` blooms across gated_NB/th_gated_NB/ZINB + legacy fail-loud; a pre-registered 3-seed × 3-arm counted verdict (`05e`). Full suite + ruff + determinism green.

**Outcome (S7 verdict, 2026-07-27; `reports/2026-07-25_t0_rollout_skill_dossier/06_bloom_verification_verdict.md`):** on 6 freshly-retrained known-seed artifacts (matched 40-lesson budget), 18 free-running 36-step rollouts (3 arms × 3 seeds × {mean, sample}) scored on the frozen per-horizon ruler — **mean-feedback blooms 9/9 arms; sample-feedback bounded 9/9** (field-wide `crps_none`: mean 36–95 vs sample 0.002–0.35; `M_mean`: mean 285–751 vs sample 0.02–2.49). The bloom is fixed by the sample-on default across every deployable arm and seed. F-B1 did not fire; F-B2 was flagged, root-caused, and fixed (§3). **Mitigates C-113 (evidenced mitigation — the deployed rollout no longer blooms; the underlying input→output io-gain>1 durable fix remains a separate open item); resolves C-121 (the regression guard now covers the fix, not just detection).**

## 6. Implementation notes
`views_hydranet/utils/hydranet_inference.py` (`_sample_feedback`, the `predict()` AR loop, the `__init__` validation); `config_initializer.py` (`rollout_feedback` field + validator + the family-default resolution); `docs/CICs/HydraNetConfig.md`. Determinism per S2 #121. Dossier: `reports/2026-07-25_t0_rollout_skill_dossier/` (EXP-2/EXP-3 evidence).

## Glossary
Follows `reports/GLOSSARY.md`. New: **rollout_feedback** (what the AR loop feeds back); **bloom** (C-113 runaway); **T=0-neutral** (a change that cannot affect the h=1 / scored T=0 output).
