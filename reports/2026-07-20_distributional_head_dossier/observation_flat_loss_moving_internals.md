# Observation: flat training loss, still-moving internals (ZINB head)

**Status:** WORKING HYPOTHESIS — *not* pinned down. Single run, eyeballed. Recorded so we can dig
deeper one day.
**Date:** 2026-07-24 · **Run:** ZINB 3×300, seed 43 (~lesson 215), on the C-213-fixed family-aware
forensics (each target shows its OWN μ/θ/π). Prompted by a user observation.

---

## What we think we SEE
- The **training loss plateaus around lesson ~60** and then stays flat: at lesson 216, reg NLL ≈ **22**,
  cls weighted-BCE ≈ **59**, MultiTask total ≈ **210** — visibly saturated on `biopsy_loss_curves`.
- But the model keeps **changing under the hood** long past lesson 60:
  - `μ̄` (per-target body mean = conditional magnitude) **climbs steadily** through lesson 215
    (ns/sb → ~3.0, os → ~2.5), with **no sign of a plateau**.
  - `π` (structural-zero prob) **rises toward ~0.99**.
  - The forecast `E[y] = (1−π)μ` (magnitude pulse) actually **shrinks** over training (π rises faster
    than μ), while its calibration ratio slowly converges toward the truth level.
- Eyeballing the biopsy grid + the new param-health forensics, the body **systematically low-balls the
  magnitude on ACTIVE cells** (the classic "timid body"), and that under-shoot is *slowly* shrinking as
  μ̄ climbs. (Note the *marginal* E[y] over all cells is slightly *over* the truth mean — low on the
  peaks, high on the spread; cf. C-211 diffuse body.)

## Where we think we SEE it
- `02_training_dynamics/biopsy_loss_curves` (Learning Dynamics), lesson 216.
- The family-aware `REGRESSION FORENSIC — horizon split` param-health rows (μ̄, θ-CoV, π) for
  lr_ns/os/sb, lessons 215–216 (the C-213 upgrade).
- The Stage-5 biopsy spatial grid (active-cell magnitude under-shoot).

## Why we think we see it (candidate explanations — NOT confirmed)
1. **Zero-dominated loss.** 99.7% of cells are zero, so the aggregate NLL is almost entirely the
   zero-cell likelihood; once the zeros are fit (~lesson 60), the loss saturates and is ~insensitive to
   further improvement on the 0.3% positive cells. The model keeps improving the positive-cell magnitude
   (μ̄↑) — real learning on the cells we care about — but it barely registers in the aggregate loss.
   ⇒ **the loss is the wrong convergence lens here; μ̄ / crps-events / size-ratio is the right one.**
2. **μ/π ridge degeneracy.** For sparse zero-inflated data, (high π, high μ) and (low π, low μ) give
   near-identical zero-cell likelihood — a near-flat ridge in (μ, π). The model appears to drift along
   it (π↑ *and* μ↑ together) post-plateau at ~constant loss. This matches the "under the hood it finds a
   different way to hold the same loss" intuition. (The C-199/C-200 μ/π ridge; the π-ridge penalty
   `pi_penalty_weight` exists to regularize it — currently OFF.)
3. **NOT grokking** (recorded because it came up in discussion). Grokking = delayed *validation*
   generalization long after *train* loss saturates near-perfect, driven by regularization finding a
   generalizing circuit. Here the train loss plateaus at a *non-zero* level and we don't track a val
   curve over training. The *spirit* ("flat loss ≠ learning stopped") matches; the mechanism does not.

## How to dig deeper one day
- **Run 600–1000 lessons** and watch whether μ̄ / crps-events / size-ratio keep improving after the loss
  is long-flat — **judge by those, NOT the aggregate loss** (it converged at ~60 and won't move). μ̄ had
  not plateaued at 215.
- **Watch π for saturation** (→1 ⇒ dead body, F1 falsifier). If longer training drifts up the ridge into
  π-saturation, that is the failure mode to catch.
- **Decompose the loss into zero-cell vs positive-cell NLL over training** — the cleanest test of
  hypothesis 1: does the positive-cell NLL keep dropping after the aggregate plateaus? (Needs the
  per-cell-class loss logged — cf. the C-215 observability gap.)
- **Toggle the C-200 π-ridge** (`pi_penalty_weight`): does pinning π change the μ trajectory / final
  magnitude — i.e., does regularizing the ridge free μ to fit the peaks, or just freeze it?
- **Track μ̄ vs π vs E[y] jointly** to characterize the ridge-drift direction empirically.

## Caveats
- Single run (seed 43), eyeballed — not multi-seed, not scored against the frozen ruler over training.
- Hypothesis 1 is currently *inferential*, not measured: the reg/cls and zero-vs-positive loss
  decomposition isn't logged (C-215), so we can't yet confirm the positive-cell NLL keeps dropping.
- Cross-refs: C-199/C-200 (μ/π ridge), C-211 (diffuse body / gate-independent crps), C-215 (loss not
  logged numerically), the "timid body" thread, `05_analysis_plan.md` (F1 π-collapse / θ-CoV falsifier).
