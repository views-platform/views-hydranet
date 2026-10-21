# Pre-registration — separating the TRAINED family from the EMITTED family

**2026-10-21.** ⛔ **Written before the arithmetic test runs and before any code is built.**

## 0. Where this comes from, and why it is not a new idea

`2026-10-13_composition_arm_outcome/OUTCOME.md:77-84`, eight days old and never acted on:

> **The gate was the asset, and retraining is what cost us.** Composing the *control's own* gate —
> which is better than the arm's — would have emitted ≈ 0.6050 against an observed 0.6438, with no
> training at all.
>
> **That is not expressible as a config today** … `output_distribution` selects one family that
> serves **both** the training loss (`utils/utils.py:186-187`) **and** the emit-time draw
> (`DISTRIBUTION_REGISTRY`, `distributions/registry.py:36-44`). There is no axis that says "train
> the body as `nb`, emit it zero-truncated". M1 computed that combination *analytically* from the
> control's parameters and got **76.3%**; the engine cannot currently *run* it.

Three independent measurements since then point the same way:

| | |
|---|---|
| the composition work | the body draws zero in **~57%** of draws, and `threshold_gate` discards a calibrated gate in favour of it |
| ⭐ **2026-10-21, §4.3** | a single theta rescale recovers **9.58% of composed CRPS held out**, and the emitted positive fraction rises 0.26× → 0.66× of observed. **The body's over-dispersion is what empties the map** |
| ⭐ **2026-10-21, within-window** | the `truncated_nb`-TRAINED body is flat with respect to where conflict is (1.0–1.1×) while the `nb`-trained body separates by 5–70× — **training on the truncated likelihood destroyed the body**, which is why the emit-time route matters: it takes the truncation without the training damage |

## 1. The hypothesis

**H1.** The emitted zero mass is a **composition** defect, not a defect of the learned body — so a
model **trained** as `nb` and **emitted** zero-truncated keeps the `nb` body's placement skill *and*
gains the truncated composition's occurrence realism, which training on `truncated_nb` buys only by
destroying the body.

## 2. ⛔ The order: measure first, build second

The engine change is small but it widens a config surface ADR-069 deliberately kept narrow. ⇒ **The
arithmetic test runs FIRST, on the frozen control cube, and the axis is built only if it pays.** The
cube carries `mu`, `theta` and `gate` per draw per cell-month, which is everything the emitted draw
needs; nothing about this requires the checkpoint, which is just as well because it is deleted.

**A 2 × 2, because the two levers interact and testing one alone cannot see that:**

| | NB emission (today) | zero-truncated emission |
|---|---|---|
| **model's own theta** | the baseline | **the arm** |
| **theta × §4.3's scale** | §4.3's result, already measured | the combination |

## 3. Pre-registered predictions

⚠️ **I am predicting against this repo's own headline number**, and saying so plainly: the 76.3%
figure is about **zero mass**, not about score, and nothing yet measures the score.

| # | quantity | prediction | confidence |
|---|---|---|---|
| **P-1** | emitted positive fraction under truncated emission | rises to **≈ the gate's own mean**, i.e. within 20% of the observed 0.0034 — by construction, since `P(body > 0) = 1` makes emitted `P(y>0) = gate` exactly | **90%** — this is arithmetic, and it is the manipulation check, not the result |
| **P-2** ⭐ **the hypothesis** | composed CRPS, truncated vs NB emission, model's theta | ⛔ **WORSE, by 5–30%** | **65%** |
| **P-3** | composed CRPS, truncated emission **with** theta recalibrated | better than truncated-at-model-theta, but **still worse than the best NB-emission cell** | 55% |
| **P-4** | where any CRPS loss comes from | **the true-zero cells**: 99.66% of out-of-sample cell-months have truth 0, and truncation forces a floor of 1 wherever the gate fires, so each such member costs ~1 where an NB draw would have cost 0 | 75% |

⛔ **Why P-2 predicts a LOSS against our own 76.3%.** Those are different quantities. Filling the
map is good for occurrence realism and, on a field that is 99.66% zeros, is paid for in CRPS on
every cell the gate fires where nothing happened. §4.3 already showed the two can point oppositely:
its gate arm improved CRPS **by emptying the map further**, which the 2026-08-15 verdict identifies
as an artifact. ⇒ **An improvement here would be a genuine surprise and a strong result.**

## 4. ⛔ Falsifiers, committed now

| ID | fires if | consequence |
|---|---|---|
| **F1** | the emitted positive fraction does **not** move materially under truncated emission | ⛔ **VOID** — the truncation was not applied; an implementation failure, not a result |
| **F2** | composed CRPS **improves** under truncated emission | ⇒ **P-2 refuted**, and the emit-time axis is worth building. ⛔ Report as a surprise against a 65% prior, not as a confirmation |
| **F3** | the CRPS change is **not** concentrated in true-zero cells | ⇒ P-4 refuted; the mechanism is not the one stated and the explanation must be rebuilt before any claim |
| **F4** | the 2 × 2 is not separable — the theta effect reverses sign across the emission column | ⇒ the two levers interact and **neither may be reported alone** |

## 5. ⛔ What this plan does NOT do

- ⛔ **It does not touch FAO-02.** CRPS is the locked primary metric and is used as such.
- ⛔ **It builds nothing yet.** If the arithmetic says no, no code is written and the idea is
  retired with a measurement instead of an opinion.
- ⛔ **It is not a claim about retraining.** It recalibrates the *emission* of delivered weights.
- ⚠️ **n = 1, one target (`sb`), one vehicle** — and that vehicle's checkpoint is deleted, so this
  is the last question the frozen cube can ever answer.

## 6. Slots (C-351) and method

| slot | value |
|---|---|
| population | all **64,818** land cells × the **13 out-of-sample months 457–469**. No admission filter; zeros are data |
| transform | none — count space, matching the frozen ruler |
| estimator | `crps_ensemble` copied verbatim from `lodestar_score.py:26-33`; **all four cells use m = 16**, so its biased `2/(m*m)` spread term cancels in every comparison made here |
| predictor | `soft_gate` composition via the repository's own `compose_samples`, 4 posterior draws × 4 head draws = the deployed 16 members |

**Zero-truncated draw:** rejection-free inverse-CDF on the NB conditioned to `y ≥ 1`, i.e. draw
`u ~ U(p0, 1)` and invert — ⛔ **not** rejection sampling, whose cost explodes exactly where `p0 → 1`,
which is most of this grid.
