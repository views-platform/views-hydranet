# 07 — Experiment log (append-only)

Every run and outcome, **including negatives and postmortems**. Each entry links its pre-registration
(`05_analysis_plan.md`) and states its verdict against the pre-committed falsifiers. No success-only drift.

---

## EXP-00 — build + review + the stopped first launch · 2026-08-16

**Code:** `views_hydranet/utils/feedback_field_transforms.py`, `HydraNetInference.feedback_transform`,
drivers in `tools/`. 1468 tests, ruff clean.

### A design flaw caught by a failing test
The persistence axis was first built as a **torus roll** of the field. Its orthogonality test failed, and
investigating why exposed something worse than a bad metric: the grid is a map of Africa, not a torus, so
rolling a blob off the east edge lands it in another country while the coordinate/static channels stay
fixed. That confounds "persistence broken" with "field decoupled from geography". Replaced with
**`shuffle_months`** — permute which step's *real* field is fed, so every field stays real, realistic and
correctly geo-located, and only the temporal order moves.

### `/code-review medium`: 12 findings, 10 fixed, batch 1 stopped and relaunched
Batch 1 had been launched. It was **killed 13 minutes in** and relaunched on fixed code, because two
findings would have produced exactly the wrong-knowledge failure this programme exists to avoid:

| finding | why it mattered |
|---|---|
| `splice` **raised** on an all-zero donor | the donor is the model's own field, and the model going quiet **is** the phenomenon under study — the arm would have aborted an unattended batch at the moment the effect appeared |
| the transform RNG was seeded **identically** to the sample-feedback RNG | two Generators, one seed, one stream — the intervention would have been correlated with the quantity it measures (the C-113 coupling class) |
| `magnitude_perturb` used `exp(sigma*Z)`, mean `exp(s²/2)` | a **×2.5 inflation at sigma=1.5**, confounding magnitude *realism* with magnitude *inflation* on a model with documented runaway-feedback sensitivity. Now mean-one by construction |
| `wrong_month:0.5` truncated to 0 | silently runs `use_real` — the control scored as the treatment |
| clustering statistic used a pooled denominator; `active_fraction` pooled sb/ns/os | the statistics meant to catch a silent no-op were themselves wrong; now per (batch, target) |
| the log1p round-trip was never asserted | an `asinh` feature would have run every arm on mis-scaled counts and looked plausible |
| `shuffle_months` was not a derangement | ~1 fixed point in 35 steps = silent control steps inside the treatment arm |

Also fixed: `use_real` returns the full slice (so F1 does not depend on statics being time-invariant), and
month bounds are pre-flighted instead of failing ~30 steps into the first origin.

---

## EXP-01 — batch 1, arms 1-3 · 2026-08-16 · **BOTH VALIDITY FALSIFIERS CLEARED; P1 FALSIFIED; P4 CONFIRMED**

**Vehicle:** `truncated_smoke`, artifact `calibration_model_20260814_003058.pt`, target `sb`, 13 origins,
`rollout_feedback='sample'`, ~27 min/arm.

### Validity — checked on REAL data, not only on fixtures

| falsifier | check | result |
|---|---|---|
| **F3** | `identity` must reproduce the archived free-`sample` row | 0.2979 / 0.0070 / 0.0083 vs 0.298 / 0.007 / 0.008 — **does not fire** |
| **F1** | `use_real` must reproduce archived `teacher_forced` | 0.2979 / 0.3008 / 0.2711 vs 0.298 / 0.301 / 0.271 — **does not fire** |

F1 is the one the programme rests on: the transform reads the right month and the right channels, at
production scale. Every downstream arm is interpretable.

### E1 — the generated field vs the real one (**P1 FALSIFIED**)

Drift at the plateau (step >= 12), median over 13 origins × samples, target `sb`:

| statistic | generated | real | drift |
|---|--:|--:|--:|
| active fraction | 0.0028 | 0.0037 | **1.3×** |
| `P(on\|on)` persistence | 0.046 | 0.430 | **9.3×** |
| **spatial clustering** | 0.011 | 0.449 | **40.8×** |

**P1 predicted `P(on|on)` would drift furthest. It does not — clustering drifts 4.4× more.** Recorded as a
falsified prediction, not softened.

**The model emits roughly the right NUMBER of active cells and puts them in the wrong PLACES.** Active
fraction is the one property it nearly gets right (1.3× off); spatial coherence is destroyed.

Two independent validations fall out: the real field's `P(on|on)` measures **0.430** here against **0.418**
recorded in #258 via a completely different code path, and the real field's statistics are flat across all
36 steps — which is what real data should look like.

**P3 tracking toward confirmation:** the field statistics collapse over steps 1-6 and then plateau
(clustering 0.358 -> 0.016 by h6, flat thereafter), on the same horizon as the AP cliff. A fixed point, not
unbounded drift.

### E3 — realism vs correctness (**P4 CONFIRMED, F5 does not fire**)

A **real field from 60 months earlier** — perfectly realistic, carrying no information about the situation
being forecast:

| h | free-running | wrong-month | oracle | share of the gap |
|--:|--:|--:|--:|--:|
| 6 | 0.0284 | 0.1290 | 0.3043 | 36% |
| 18 | 0.0070 | **0.1322** | 0.3008 | **43%** |
| 36 | 0.0083 | 0.1199 | 0.2711 | 42% |

**19× the free-running AP at h18, stable across the whole horizon.** The damage decomposes roughly
**43% realism / 57% correctness**: nearly half of what the rollout loses is lost not because its input is
*wrong* but because it is *implausible*.

This is a direct endorsement of the distribution-matching family — from a run designed to be able to kill
it (F5). It also puts the state-freeze result in proportion: realism is worth 43%, holding the cell state
was worth 23%, and those are likely overlapping rather than additive.

`act_ratio` under wrong-month is 1.4-2.0 (versus 0.27 free-running): a plausible input keeps activation
calibrated even when uninformative.

### Standing caveat
40 lessons, seed 42, one origin set, one target, one vehicle. **INDICATIVE.** The ordering is the result;
the magnitudes are not calibrated. And **drift is not damage** — E1 measures how far each statistic moved,
E2 measures how much skill cares. The product is what matters, and batch 2 has not run.

---

## EXP-02 — batch 1 complete: the E4 channel split · 2026-08-16 · **OCCURRENCE, NOT MAGNITUDE**

| arm | AP h6 | AP h18 | AP h36 | share of the oracle gap @h18 |
|---|--:|--:|--:|--:|
| `identity` (free-running) | 0.0284 | 0.0070 | 0.0083 | 0% |
| `occurrence_model_magnitude_real` | 0.0599 | 0.0301 | 0.0160 | **8%** |
| `wrong_month:-60` | 0.1290 | 0.1322 | 0.1199 | **43%** |
| `occurrence_real_magnitude_model` | 0.2634 | 0.2674 | 0.2802 | **89%** |
| `use_real` (oracle) | 0.3043 | 0.3008 | 0.2711 | 100% |

### The decomposition

| what the fed field carries | share of the gap |
|---|--:|
| occurrence merely **plausible** (real field, wrong month) | 43% |
| occurrence **correct**, magnitudes the model's own | 89% |
| magnitude correct, occurrence the model's own | **8%** |

**Occurrence carries the damage; magnitude does not.** An 11:1 split. This is consistent with `size_ratio`
being 0.0000 in every arm of every rollout dossier, and with P3's prediction that occurrence and amount are
separate ceilings — but it is the first time magnitude's contribution has been *measured* as a share.

It also splits the realism question in two: **43% of the gap is bought by the occurrence field merely being
plausible**, and a further **46%** by it being correct. Nearly half the damage is not about being wrong; it
is about being implausible.

### A caveat on E4b, found by reading its own fed-field record

`occurrence_model_magnitude_real` is **not** the clean mirror of E4a. Its recorded field drifts to
**active 0.0167, clustering 0.948** by step 12 — denser and *more* clustered than the real field
(0.0037 / 0.449), not the free-running model's field (0.0028 / 0.011). The arm is a feedback loop: feeding
real (large) magnitudes drives the model to over-fire, so "model occurrence" here means *occurrence as it
evolves under real-magnitude feedback*, not free-running occurrence.

The conclusion survives — a field carrying real magnitudes at the model's own placements recovers 8% — but
the arm is not a symmetric complement of E4a and should not be reported as one. Recorded because the
fed-field instrumentation is what exposed it; the score alone would have read as a clean mirror.

### Verdicts so far
**F1, F2, F3, F5, F6 do not fire. P1 FALSIFIED** (clustering drifts 4.4× more than persistence).
**P3 tracking** (statistics saturate on the AP-cliff horizon). **P4 CONFIRMED** (43%).
P2 and F4 need batch 2.

### What this already implies for the fix
A distribution-matching loss should target **the occurrence field's spatial structure** and can largely
ignore magnitude. That is a much smaller and better-posed object than a realism critic on the full field,
and it agrees with E1's finding that clustering is the dominant drift (40.8×). Batch 2 tests whether skill's
*sensitivity* follows that drift, which is the other half of the product and is not yet measured.

---
