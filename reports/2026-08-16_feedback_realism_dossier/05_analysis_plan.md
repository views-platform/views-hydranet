# 05 — Analysis plan (pre-registered)

# **LOCKED 2026-08-16**

> Locked **before any arm runs**. Nothing below may be changed afterwards. If something here turns out to be
> wrong, that is a logged finding in `07_experiment_log.md` and a *new* pre-registration — not an edit here.

---

## The question

**Which statistic of the fed-back field does the rollout's skill actually depend on?**

`the feedback realism gap` (GLOSSARY, locked 2026-08-15) is the parent cause of the rollout failure. We have
only ever sampled its two endpoints: `teacher_forced` (a real field — gate AP 0.30 → 0.27 at h36) and fully
generated (0.30 → 0.01). The space between is unmeasured.

The damage decomposes as `Σᵢ [drift in statistic i] × [skill's sensitivity to statistic i]`. **E1 measures
the first factor, E2 the second.** Today we know exactly one drift number (`P(on|on)` 0.418 real vs 0.090
free-running) and **zero** sensitivities — yet distribution matching only fixes the statistics you name.

## The one variable

`feedback_transform` — an explicit diagnostic argument, no config key, default `None`. Nothing else moves:
same artifact, same data, same seed, `rollout_feedback='sample'` throughout, no retraining.

---

## Method

### Substrate
`truncated_smoke`, artifact `calibration_model_20260814_003058.pt` (the EXP-SS-2 artifact), target `sb`,
h = 1/6/12/18/24/30/36, 13 origins, ~27 min/arm. Scored with the unchanged
`score_v2_horizons.py` + `activation_metrics.py`, so every number is comparable to EXP-SS-2, EXP-01 and the
v2 board.

### The arms

**Batch 1 — measure the gap and settle the direction (5 arms)**

| arm | what it feeds | what it answers |
|---|---|---|
| `identity` | the model's own field, unchanged | **E1**: the fingerprint + the free-running control |
| `use_real` | the real field at this step | **F1 on real data** — must reproduce EXP-SS-2's `teacher_forced` |
| `wrong_month:-60` | a real field from 60 months earlier | **E3**: realistic but uninformative |
| `occurrence_real_magnitude_model` | real *where*, model *how much* | **E4**: which channel carries the poison |
| `occurrence_model_magnitude_real` | model *where*, real *how much* | **E4**, mirror |

**Batch 2 — dose-response on the oracle (8 arms).** Start from the field we know works and break exactly
one property.

| axis | arm specs |
|---|---|
| recall | `thin:0.25`, `thin:0.75` |
| precision | `inject:0.002`, `inject:0.01` |
| magnitude | `magnitude_perturb:0.5`, `magnitude_perturb:1.5` |
| spatial structure | `spatial_scramble` |
| temporal persistence | `shuffle_months` |

**All five axes are pre-registered and all are run** — no selection on batch 1's results, which would be a
garden of forking paths. GPU time is free tonight; buying the whole grid is cheaper than defending a
selection rule.

**E2 and E4 bracket the response curve.** E2 degrades from the good end, E4 repairs from the bad end. A
sensitivity measured only near the oracle need not describe the collapsed regime.

### Two implementation facts that shape the arms

1. **`shuffle_months`, not a spatial roll.** The temporal axis permutes *which step's real field is fed*, so
   each field stays real, realistic and correctly geo-located. A torus roll was built first and **rejected**:
   it preserves the statistics, but the grid is a map of Africa, not a torus — rolling a blob off the east
   edge lands it in another country while the static channels (coordinates, ADR-060/061) stay fixed, which
   confounds "persistence broken" with "field decoupled from geography".
2. **`spatial_scramble` carries an irreducible confound.** Destroying clustering necessarily breaks the
   field's alignment with the statics, because plausible locations *are* the clustering. Read that arm as
   "spatial structure **and its geographic grounding**", not "structure alone". This is stated before the
   run, not after.

### Every arm self-reports the field it fed
`fedfield_*.csv` records, per (origin, sample, step): active fraction, mean magnitude on actives,
`P(on|on)`, and neighbour-pairs-per-active. The fixture tests prove each transform moves its axis on a
hand-built field; this proves it moved on the **real** one. An arm whose statistics did not shift is a
**silent no-op** and its score is void, not evidence that the axis does not matter.

---

## Pre-registered predictions

| # | Prediction |
|---|---|
| **P1** | `P(on\|on)` drifts furthest of all measured statistics, relative to its real value. |
| **P2** | At least one E2 axis produces a **≥0.05 gate-AP drop at h18** — some named statistic is individually sufficient to cause meaningful damage. |
| **P3** | E1's field statistics **saturate on the same horizon as AP** (~h6), consistent with a fixed point rather than unbounded drift. |
| **P4** | `wrong_month:-60` retains gate AP **well above** the free-running arm at h18 — realism carries skill even when the field is uninformative. |

## Falsifiers (pre-committed)

| # | Fires if… | Consequence |
|---|---|---|
| **F1** | `use_real` does not reproduce EXP-SS-2's `teacher_forced` row (h1 0.298 / h18 0.301 / h36 0.271, ±0.005) | the transform reads the wrong month or channels ⇒ **all arms void**, stop |
| **F2** | any arm's static channels differ from the control's | channel-slice bug ⇒ void |
| **F3** | `identity` does not reproduce EXP-SS-2's free-`sample` row (0.298 / 0.007 / 0.008) | the seam perturbed the production path ⇒ void |
| **F4** | every E2 axis at its severest dose costs **< 0.01 AP at h18** | no single named statistic matters ⇒ **distribution matching is dead as a strategy**; the gap is joint/structural and only a critic (Professor Forcing) can address it. A real result. |
| **F5** | `wrong_month:-60` collapses to the free-running arm | realism is insufficient; the model needs *correct* history ⇒ downgrade distribution matching, re-scope to K=5 rollout training |
| **F6** | any arm's `fedfield_*.csv` shows its target statistic unmoved | that arm is a silent no-op ⇒ its score is void, fix and re-run |

**F1–F3 are also CI tests** (`tests/test_feedback_transform_seam.py`), each verified against a deliberate
sabotage. Passing them on a fixture is necessary, not sufficient — this repo has had fixture-level
correctness flip in production, which is why they are re-checked here against archived real numbers.

---

## Decision rule — pre-committed

Rank axes by **ΔAP at h18 under the severest dose**, versus the `use_real` oracle arm:

```
NAMED-STATISTIC-SUFFICIENT  iff  the top axis costs >= 0.05 AP at h18
                                 -> that statistic is what a distribution-matching loss penalises first
NO-SINGLE-STATISTIC         iff  every axis costs < 0.01 AP at h18   (F4)
                                 -> the gap is joint; only a critic can address it
INCONCLUSIVE                otherwise
```

The 0.05 threshold is FAO-02's superiority margin, reused rather than invented.

---

## Skepticism ledger

1. **40 lessons, seed 42, one origin set, one target, one vehicle. INDICATIVE.** The ordering of axes is the
   result; the magnitudes are not calibrated to anything.
2. **E2's doses are not a calibration of real-world unrealism** — only of skill's sensitivity to it. "How
   much does the model's field actually drift on this axis" is E1's job, and the two must be multiplied,
   not confused.
3. **The response is known to be nonlinear** (a cliff to h6, then saturation). Sensitivities measured near
   the oracle need not describe the collapsed regime; that is why E4 anchors the far end.
4. **Meta-pattern 8 (invalid knowledge from a bug).** Every guard here was checked against a deliberate
   sabotage before any arm ran, because this programme has twice produced a confident verdict from a wrong
   implementation.
5. **`identity` is not free of the composition defect.** Even here the fed-back field is only calibrated at
   h1; these arms isolate *which statistic matters*, not the whole problem.

## What this plan does NOT decide

Whether to build distribution matching, Professor Forcing, or K-step rollout training. It decides **what
they would have to target**.
