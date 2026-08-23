# Pre-analysis plan — why does the model emit only 39% of the true occurrence at step 1?

**Status: LOCKED** — committed alone, before `tools/` existed.

## 0. The observation that prompted this (exploratory, already seen — disclosed)

Comparing the fed-back field's own statistics, model (`identity`) vs oracle (`use_real`), across the four
L=300 ε=0 seeds — **existing CSVs, no GPU, no new runs**:

| step | 1 | 2 | 3 | 4 | 5 | 6 |
|---|--:|--:|--:|--:|--:|--:|
| active_fraction, model/oracle | **0.392** | 0.267 | 0.206 | 0.161 | 0.129 | 0.101 |
| magnitude, model/oracle | 1.183 | 1.177 | 1.037 | 1.121 | 1.110 | 1.043 |

**Step 1 already emits 39% of the true active fraction** (per-seed 0.371 / 0.398 / 0.427 / 0.371,
sd 0.027), and it then decays ~**×0.78 per step**. **Magnitude is not the problem** — it is ~1.0
throughout, consistent with every prior magnitude finding.

⚠️ **This is exploratory and was seen before this plan was written.** Everything below is therefore **not
blind**; the thresholds are set from arithmetic that does not reference these numbers, and the expected
outcome is stated up front (C-305, C-309 discipline).

**Why step 1 matters:** at `t == origin` the model's input is **entirely real data**. Step 1's emission
precedes any feedback. **The rollout is amplifying an error that already exists before it starts.**

## 1. Hypothesis — the zero process is applied twice

`fullzero_*` runs `forecast_composition = 'soft_gate'` and `rollout_feedback = 'sample'`. Per
`composition.compose_samples`, soft_gate is:

```
mask = torch.bernoulli(gate)      # gate = per-cell P(y > 0)
emitted = body_sample * mask
```

The body is a **plain NB**, which **has its own probability mass at zero**. A cell is emitted active only
if the gate fires **AND** the NB draw is non-zero. So:

> **E[emitted occurrence] = E[ gate × P(NB draw > 0) ]**, not `E[gate]`.

**H:** the gate is approximately right and the attenuation is the body's zero mass — the occurrence
process is modelled once by the gate and then **silently applied a second time** by the body's zeros.

**H0:** the gate itself under-predicts occurrence, and the body's zero mass is incidental.

These make **different, separable predictions**, which is what makes this worth running.

## 2. The one variable

Nothing is trained or changed. This is a **decomposition** of an existing quantity into two measured
factors at a single step.

## 3. Method

One forward pass at the production origin (**335** — *not* `predict()`'s `seq_len - 1` fallback; that
default is C-308's second occurrence), full grid, seeds 42/43/44/45, `sb`. Record per cell:

* `gate` — the emitted `P(y>0)`;
* `p_nonzero` — the body's own `P(draw > 0)`, computed **analytically from the NB parameters**, not by
  sampling, so it carries no Monte-Carlo error;
* `truth` — the real active fraction at that step.

Then compare three numbers: **`mean(gate)`**, **`mean(gate × p_nonzero)`**, and **truth**.

## 4. Decision rule — registered before the measurement

Let `G = mean(gate) / truth` and `C = mean(gate × p_nonzero) / truth`. The exploratory emitted ratio is
**0.39**, so `C` is the quantity that must reproduce it.

| condition | verdict |
|---|---|
| `C` within **±0.05** of the observed emitted ratio **AND** `G ≥ 0.80` | **DOUBLE-COUNT CONFIRMED** — the gate is substantially right and the body's zeros cause the shortfall |
| `G ≤ 0.55` | **GATE UNDER-PREDICTS** — H0. The shortfall is the gate's, and the composition is not the lever |
| neither | **BOTH / INCONCLUSIVE** — report `G` and `C`; claim nothing |

**Justification of the two cuts, independent of the data.** `G ≥ 0.80` means the gate accounts for at
least 80% of true occurrence, i.e. it is *not* the dominant error. `G ≤ 0.55` is the mirror: the gate
alone already loses nearly half, so it *is*. The gap between them is left deliberately unresolvable
rather than split, because a design that cannot separate two causes must say so. `±0.05` on `C` is the
reproduction tolerance for a quantity we already measured to 3 significant figures with seed sd 0.027.

**No branch may be overridden by an argument not written here** (C-305).

## 5. Falsifiers — pre-committed

* **F1 — origin.** The manifest must record `origin == 335` and the measured sample period `371`. A run
  reporting `seq_len - 1` is void. *(Direct C-308 guard; it has already fired twice.)*
* **F2 — the decomposition must be exact.** `mean(gate × p_nonzero)` computed analytically must match a
  **sampled** emitted occurrence from the same forward pass to within **2%**. If it does not, the
  analytic `p_nonzero` is not describing the sampler that produces the field, and §4 is void.
* **F3 — truth is the same truth.** The true active fraction at step 1 must match the `use_real` arm's
  step-1 `active_fraction` in the existing CSVs. Otherwise this is not the quantity the exploration
  measured.
* **F4 — no NaN/inf** in `gate` or `p_nonzero`.

## 6. What each outcome buys

* **DOUBLE-COUNT CONFIRMED** ⇒ a **composition-layer** defect, not a training one, sitting upstream of
  every rollout result we have. It would also predict that `self_zeroed` (ZINB, no gate multiply) should
  *not* show the step-1 shortfall — a free cross-check against existing ZINB artifacts.
* **GATE UNDER-PREDICTS** ⇒ the lever is gate calibration at T=0, and the composition is exonerated.

## 7. False-negative mode and reopen trigger (C-307)

`p_nonzero` is computed at the **posterior mean** parameters. If the NB parameters are strongly
dispersed across posterior draws, the mean's zero-probability is not the draw-averaged zero-probability
(Jensen). **A "gate under-predicts" verdict would therefore not rule out a double-count** that this
estimator smears. **Reopen if** the verdict is H0 *and* the per-draw spread of `p_nonzero` exceeds its
mean.

## 8. Scope

One step (step 1), one target (`sb`), 4 seeds, no training, no config change. **This measures occurrence
only.** It says nothing about AP, and nothing about steps ≥ 2 beyond the compounding already shown.
