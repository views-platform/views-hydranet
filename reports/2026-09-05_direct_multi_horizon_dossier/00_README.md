# Direct multi-horizon — delete the recursion instead of mitigating it

**Epic:** [#324](https://github.com/views-platform/views-hydranet/issues/324) · **Tracking:** [#336](https://github.com/views-platform/views-hydranet/issues/336) · **Strategy decision:** [#310](https://github.com/views-platform/views-hydranet/issues/310)
**Status:** SCAFFOLDED — chair decision recorded (PURSUE), nothing implemented, nothing run

## Purpose

Seven interventions have attacked the training/deployment gap. **Every one of them mitigated the
autoregressive loop.** Six failed, four of them on the same lever (**M45**: AP loss scales with how
much the model fires), and the one that worked — clamping the cell state (**M48/M56**, +0.0591
AP@h36, shipped 2026-09-05 as ADR-027 §2.1) — works by *stopping the loop's damage*, not by
improving the forecast.

This programme deletes the loop. The ConvLSTM encoder still digests history; what goes is the
**state evolving during the forecast**. Every horizon is decoded from the origin state, with the
horizon as an input covariate. **No feedback ⇒ no exposure bias at all — not reduced, absent.**

## The framing that makes this coherent with what we just shipped

**Direct multi-horizon is the cell clamp taken to its limit.** The clamp says *stop the state
drifting*; this says *don't evolve it at all*. The intervention that works becomes the architecture.

That also corrects #310's own founding objection, which this dossier records as **overstated**:
a direct head does **not** forfeit the recurrent state. It keeps the encoder and the spatial map
the encoder builds (**M54**, **M55**: 2.97× fair persistence at h36 with the gap widening). What it
forfeits is the *evolution* of that state — the thing measured as broken (**M50**: `max|h|` drains
65.6 → 1.6, 41×).

## Document index

| # | file | status |
|---|---|---|
| 00 | `00_README.md` | **living** |
| 01 | `01_literature.md` | written — the blueprint, the platform precedent, the theory, and **one serious gap** |
| 02 | `02_design.md` | written — the shape, the seam, and the honest cost |
| 03 | `03_harness_and_invariants.md` | **TODO** — blocks S3 |
| 04 | `04_roadmap.md` | written |
| 05 | `05_analysis_plan.md` | **NOT WRITTEN** — pre-registration follows the design review |
| 06 | `06_glossary.md` | written |
| 07 | `07_experiment_log.md` | empty, append-only |

## What is different about this programme's economics

**M65 (2026-09-05, zero GPU):** the seed variance every recent screen was sized against (~20%,
C-119/C-184) is **stale by ~5×**. Measured on four same-config seeds at L=300: sd **0.0134 (4.1%)**
at h18. Consequences, at one-sided α=0.05 and 80% power:

| design | MDE at h18 | can it see the cell freeze (+0.0367)? |
|---|---|---|
| unpaired n=1 | 0.0473 | **no** |
| unpaired n=2 | 0.0334 | marginal |
| **paired n=2** | **0.0131** | **yes, 2.8× margin** |

**So this programme pre-registers a paired 2-seed design, not an n=1 screen.** Every prior screen in
this repo was a harm detector; this one can detect a benefit.

## The baseline to beat is no longer the unclamped rollout

ADR-027 §2.1 shipped the cell clamp today. **The honest control is now the clamped rollout**
(AP@h18 ≈ 0.362, h36 ≈ 0.284), not the unclamped one (0.326 / 0.225). A direct head that merely
matches the clamp has bought nothing. This must be in the pre-registration before any run.

## Current state & next actions

- [x] Chair decision recorded on #310 (PURSUE)
- [x] `01_literature`, `02_design`, `04_roadmap`, `06_glossary`
- [ ] **Fetch `Shi2017` (encoder-forecaster / nowcasting benchmark)** — the library holds 567 papers
      and **no primary ConvLSTM or nowcasting-architecture source**; we would be designing on our own
      backbone's paper being absent. Blocks the design freeze, not the epic.
- [ ] `03_harness_and_invariants` — wire every gate (none is repo-wide)
- [ ] `expert-method-review` on `02_design`
- [ ] `05_analysis_plan` pre-registration — **paired 2-seed, clamped control, non-inferiority floor at h1**
- [ ] Implement behind the registry seam; byte-identical when unselected
- [ ] Adversarial audit in a clean context by a non-author
- [ ] Smoke + potency at a trained checkpoint (C-324/C-325)
- [ ] Run, score, dispose

## Conventions

Numbered docs, `00` living. Git-tracked via `git add -f`. Archived on close. Vocabulary is
`reports/GLOSSARY.md`; new terms are defined once in `06`.
