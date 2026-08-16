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

## EXP-03 — batch 2 dose-response · 2026-08-16 · **SPATIAL STRUCTURE, DECISIVELY**

Degrade the *real* field one axis at a time; ΔAP at h18 against the oracle (0.3008). Free-running is 0.0070.

| axis | ΔAP h18 | resulting AP | verdict |
|---|--:|--:|---|
| **`spatial_scramble`** | **−0.2911** | 0.0097 | clears the 0.05 bar 6× over |
| `thin:0.75` | −0.0764 | 0.2244 | clears |
| ~~`shuffle_months`~~ | ~~−0.0415~~ | ~~0.2593~~ | **VOID — F6 fires, see below** |
| `inject:0.01` | −0.0299 | 0.2709 | |
| `thin:0.25` | −0.0268 | 0.2740 | |
| `magnitude_perturb:1.5` | −0.0203 | 0.2805 | |
| `inject:0.002` | −0.0090 | 0.2918 | |
| `magnitude_perturb:0.5` | −0.0021 | 0.2987 | |

**Destroying spatial structure alone reproduces the collapse.** 0.3008 → 0.0097, within 0.003 of free-running
— from a field with **identical active count and identical magnitudes**. The fed-field record proves the
intervention was clean: active 0.00321 (unchanged), mean magnitude 11.87 (unchanged), clustering
0.447 → 0.009.

**P2 CONFIRMED. F4 does not fire.** Decision rule returns **NAMED-STATISTIC-SUFFICIENT**: the target is the
spatial coherence of the occurrence field.

### F6 FIRES on `shuffle_months` — that arm is VOID
It was to destroy temporal persistence. It moved it **0.424 → 0.404 (5%)**. The intervention barely
happened, and its −0.0415 is *not* the cost of breaking persistence. The score alone reads as a plausible
middling effect and would have been written up as "persistence matters somewhat"; only the fed-field record
exposed it.

**Why it failed is a finding about the DGP:** real conflict is geographically sticky over *years*, so
permuting months inside a 36-month window lands on a month whose conflict is in much the same places. The
temporal axis cannot be broken this way, by construction. `wrong_month:-60` is the arm that actually
decorrelates (persistence 0.527, from a genuinely different era).

### Orthogonality held on real data
`spatial_scramble` left persistence at **0.424** (oracle 0.424) while destroying clustering — the guarantee
the transform was designed around, confirmed outside the fixtures. `magnitude_perturb` left clustering at
0.447 (unchanged) and cost ~nothing: a clean control in the opposite direction.

### Confound, recorded
`thin` and `inject` both *also* damage clustering as a side effect (0.447 → 0.137 and → 0.135). Part of
their measured damage is likely clustering-mediated. This strengthens rather than weakens the reading — the
axis that disturbs **only** clustering does the most harm, the axis that disturbs **only** magnitude does
almost none.

---

## EXP-04 — the gate-structure probe · 2026-08-16 · **HYPOTHESIS FALSIFIED; A REFINED MECHANISM SURVIVES**

**Motivated by a maintainer challenge:** if the failure is spatial, why have repeated coordinate-channel
attempts (ADR-061, C-152) never helped? Proposed answer: coords act on the **marginals**, while
`compose_samples` draws `torch.bernoulli(gate)` **independently per cell** — so the *joint* structure is
discarded at sampling. Hypothesis: the gate still knows, and the sampler throws it away.

### The metric would have manufactured the expected answer
The first `topk` reference used `torch.topk`, which breaks ties by **flat index order** — on any uniform
region it returns a *contiguous run*, which the clustering statistic scores as highly structured. Measured
on hand-built gates: structured **1.00**, smeared **1.06**. Indistinguishable, and it would have reported
"the gate knows" from a pure artifact. Fixed with random tie-breaking; the probe now separates the two
cases 1.50 vs 0.025. **Caught by the discriminating test, before any real data.**

### The result

| step | oracle Moran's I | oracle coherent | free Moran's I | free coherent | free independent |
|--:|--:|--:|--:|--:|--:|
| 1 | 0.504 | 0.847 | 0.406 | 0.749 | 0.147 |
| 6 | 0.491 | 0.821 | **0.161** | **0.295** | 0.011 |
| 35 | 0.509 | 0.820 | **0.161** | 0.262 | 0.018 |

The oracle gate holds Moran's I ~0.50 across all 36 steps. **The free-running gate collapses to 0.16 by
step 6** — the same cliff as everything else. And a *maximally coherent* sampler on the free-running gate
reaches only 0.25 clustering, against real 0.449.

**The hypothesis as stated is FALSIFIED: the structure is not there to recover.**

### The refined mechanism, which survives and is actionable

| gate | coherent | independent | ratio |
|---|--:|--:|--:|
| oracle (sharp) | 0.819 | 0.317 | **2.6×** |
| free-running (diffuse) | 0.249 | 0.010 | **25×** |

**Independent sampling is ~10× more destructive on a diffuse gate than a sharp one.** Concentrated mass
still yields clumps under independent draws; smeared mass yields confetti. So two effects compound in a
loop: the gate smears → independent sampling converts a mildly-smeared gate into pure scatter → the scatter
is fed back → the gate smears further.

A coherent sampler alone would move the fed field from **0.010 to ~0.25** clustering (target 0.449) with
**no retraining**. Whether that breaks the loop is not measurable from this data — the loop has to be run.

### Why the control was necessary
Run only on the free-running arm, we would have seen a smeared gate *and* a scattered sample and could have
blamed either. The oracle arm shows independent sampling of a *healthy* gate yields 0.33 against a real
0.449 — the sampler is not inherently destructive, which is what makes the 25× amplification on a diffuse
gate the interesting quantity rather than a foregone conclusion.

### On the coordinate-channel puzzle
Coords improve *which cells are likely* — a marginal, per-cell property, and every metric they were judged
on is per-cell. What fails here is the **spatial coherence of the probability field** plus a sampler that
assumes independence. Coords were never touching the failing mechanism. C-152's "CoordConv is not a lever"
now has a mechanism behind it.

---

## EXP-05 — correlated feedback sampling · 2026-08-16 · **NULL, and it corrects EXP-03's framing**

**Motivated by EXP-04:** independent Bernoulli sampling is ~10× more destructive on a diffuse gate than a
sharp one. If coherent feedback broke the loop, it would be a fix needing **no retraining**.

**Built:** a Gaussian copula sampler (`views_hydranet/utils/correlated_bernoulli.py`) — smooth a white-noise
field, map through Φ, threshold at the gate probability. `P(active_i) == gate_i` **exactly**, whatever the
correlation, because Φ(z) is uniform for standard normal z. Applied to the **feedback path only**;
`to_cube_samples` keeps independent sampling, so any effect would be the model behaving differently rather
than the ruler being handed a prettier cube (asserted by a test).

### Calibration, on the CONTROL, before the treatment
Swept ℓ on the oracle gate against the real field's clustering (0.449). Result was already negative:

| sampler | clustering | n_active | error vs 0.449 |
|---|--:|--:|--:|
| **independent** | **0.329** | 184 | **0.120** |
| ℓ=1.0 | 0.663 | 181 | 0.214 |
| ℓ=3.0 | 0.958 | 176 | 0.509 |
| ℓ=8.0 | 0.805 | 150 | 0.356 |

On a *healthy* gate, independent sampling is **closer to realistic clustering than any correlated version** —
they all overshoot. `n_active` is flat across scales on the same gate, confirming the marginal guarantee on
real data. And no single ℓ can serve both regimes: the oracle gap is 1.4×, the free-running gap is 45×, and
the correlation needed depends on how concentrated the gate already is.

Because a single calibrated point would have been a weak test, the treatment was run as a **pre-committed
sweep with all arms reported**.

### The result

| arm | fed clustering | AP h6 | AP h18 | AP h36 |
|---|--:|--:|--:|--:|
| independent (control) | 0.011 | 0.0284 | 0.0070 | 0.0083 |
| ℓ=0.5 | 0.047 | 0.0230 | 0.0069 | 0.0084 |
| **ℓ=1.0** | **0.494** | 0.0200 | **0.0069** | 0.0083 |
| ℓ=3.0 | 1.064 | 0.0191 | 0.0075 | 0.0087 |
| oracle | 0.449 | 0.3043 | 0.3008 | 0.2711 |

**Clustering spans 100× — straight through the real value 0.449 — and gate AP does not move.** At ℓ=1.0 the
fed-back field has essentially perfect realistic clustering and skill is 0.0069 against a control of 0.0070.
If anything h6 degrades slightly with more clumping.

### This corrects EXP-03's framing
EXP-03 concluded "the target is the spatial coherence of the occurrence field". **Too loose.**

* EXP-03: correct places + destroyed clustering → collapse. Clustering matters *given* correct places.
* EXP-05: wrong places + correct clustering → **no recovery**. Clustering is worthless *without* them.

**Clustering was a proxy for correct placement, not an independently sufficient property.**
`spatial_scramble` hurt because it destroyed the *correctness* of placement; restoring clumpiness to an
already-misplaced field produces realistic-looking clumps in the wrong locations.

### What it rules out
**A distribution-matching loss that penalises the field's clustering statistic will not work.** It would
produce exactly this: matching statistics, no skill. The same objection applies to any realism critic that
can be satisfied by marginal or summary statistics rather than by placement — which is most of the cheap
formulations of #262's option 3.

The null is credible *because* the implementation was tight: marginals preserved (verified on the oracle at
fixed gate), the realistic clustering target actually hit, and the scored path untouched. A weaker sampler
would have left "perhaps it did not clump enough" open.

### One measurement note
`active_fraction` differs across treatment arms (0.0027–0.0038). That is **not** a marginal violation — each
arm's gate has evolved under a different history, so the gates differ. The marginal check is the oracle
calibration above, where the gate is held fixed and `n_active` is flat.

### Where this leaves the programme
The sampler is not the lever; the gate's own diffusion is. That points to the training-side work — Professor
Forcing on state trajectories, or K=5 rollout training — and away from inference-time patches. It also
narrows distribution matching to formulations that constrain *placement*, not summary statistics.

---
