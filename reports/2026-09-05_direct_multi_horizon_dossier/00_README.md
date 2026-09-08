# Direct multi-horizon — delete the recursion instead of mitigating it

**Epic:** [#324](https://github.com/views-platform/views-hydranet/issues/324) · **Tracking:** [#336](https://github.com/views-platform/views-hydranet/issues/336) · **Strategy decision:** [#310](https://github.com/views-platform/views-hydranet/issues/310)
**Status:** SCAFFOLDED — chair decision recorded (PURSUE), nothing implemented, nothing run

## Purpose

**The one measurement this programme rests on — M51.** Same model, same seed, emit-only, one flag
changed, arms **identical at h1** as they must be before feedback acts:

| what the model is fed each step | occurrence over 36 steps | magnitude | firing↔size alignment |
|---|---|---|---|
| **its own forecasts** (deployment) | **×0.036** — 28× fewer | ×0.222 | 66.6 → 4.3 |
| **real observations** (oracle) | **×1.19 — flat** | ×0.91 | ×1.12 |

**Feed the model real data at every step and the collapse does not happen at all.** The 36-month
degradation is not a property of the model, the horizon, or the data — it is a property of **what
the model is fed after step 1**. That is the train/deployment input mismatch, measured cleanly on
this vehicle.

**And the prize is the largest this programme has ever had in front of it.** AP@h18, `sb`, seed 42:

| arm | h1 | h18 | h36 |
|---|---|---|---|
| fed **real observations** (the ceiling) | 0.4779 | **0.4974** | **0.4667** |
| **clamped rollout** — what production ships today | 0.4779 | 0.3622 | 0.2828 |
| **gap** | — | **0.1352** | **0.1839** |

Note the ceiling **does not decay with horizon** (0.4779 → 0.4974 → 0.4667). Nothing about month 36
is intrinsically harder. **The gap at h18 is 3.7× the cell clamp — the best result this programme
has produced in its history — and at h36 it is 4.6×.**

This epic deletes the mechanism that creates that gap: the model stops being fed its own forecasts.
Every horizon is decoded from the origin state, with the horizon supplied as an input covariate.
**No feedback ⇒ no mismatch.**

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

Two things, and together they invert the usual problem.

**The effect we are chasing is large.** The gap to the oracle is **0.135 at h18**. Measured seed sd
is **0.0134** (M65, four same-config seeds at L=300). So even capturing **a quarter** of the gap
(0.034) is detectable at 2 unpaired seeds (MDE 0.0334); capturing half is detectable at n=1. **This
is the first programme here whose target effect is comfortably above its own noise floor** — every
prior screen was chasing effects at or below the MDE, which is why they could only detect disasters.

**But the pre-registered MDE in an earlier draft of this dossier was wrong** and is corrected here:
M65's *paired* sd (0.0075) came from an **emit-only** flag applied to one artifact. A new
architecture is a training-time treatment whose arms cannot share weights, so that pairing does not
transfer. Use the **unpaired 0.0134**, pair on **origins** (route-agnostic) rather than on seeds, and
size against a stated share of the 0.135 gap rather than against the clamp's +0.037.

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
