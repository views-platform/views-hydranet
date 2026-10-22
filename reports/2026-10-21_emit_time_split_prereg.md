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

---

# OUTCOME — 2026-10-21. ⛔ P-2 REFUTED, F2 FIRED. Cost 107.3 s / 3.07 GiB.

Driver `2026-10-12_underfire_diagnosis/emit_time_split_test.py`, committed before it ran. Frozen
`nb` control cube, out of sample 457–469, 842,634 cell-months, common random numbers across arms.

| emission | theta | composed CRPS | vs base | emitted pos frac | CRPS on `y=0` | CRPS on `y>0` |
|---|---|---|---|---|---|---|
| NB | model's own | 0.050758 | — | 0.0009 | 0.000463 | 14.589 |
| **zero-truncated** | **model's own** | **0.047741** | **−5.94%** | **0.0037** | 0.006274 | **12.034** |
| NB | ×21.99 | 0.046048 | −9.28% | 0.0022 | 0.001052 | 13.053 |
| zero-truncated | ×21.99 | 0.046023 | −9.33% | 0.0037 | 0.001693 | 12.860 |

Observed positive fraction: **0.0034**.

## ⛔ P-2 was wrong, at 65% confidence, and the reasoning error is the useful part

I predicted composed CRPS would get **worse by 5–30%**. It is **5.94% better**. The argument was:
*"99.66% of out-of-sample cell-months have truth 0, so a floor of 1 wherever the gate fires is paid
for on every one of them."*

**Half of that is right and it is the wrong half.** Truncation does cost on true-zero cells —
0.000463 → **0.006274**, a 13.5× increase, exactly the direction P-4 named. But I reasoned about the
**count** of cells instead of their **contribution**, and those are not the same thing on a field
like this:

| | share of cells | per-cell CRPS | share of the pooled mean |
|---|---|---|---|
| truth = 0 | **99.66%** | 0.000463 | **~1%** |
| truth > 0 | 0.34% | **14.589** | **~99%** |

⇒ **The pooled CRPS is almost entirely the event cells**, and truncation improves those by **17.5%**
(14.589 → 12.034). The 13.5× penalty on the zeros is real and still nearly invisible. ⚠️ This is the
same error as reading a 97.8%-empty grid's `b` as a property of the model: a population can dominate
the *count* and contribute nothing to the *statistic*.

## ✅ P-1 confirmed, and it is arithmetic rather than a result

Emitted positive fraction **0.0009 → 0.0037** against an observed **0.0034** — within 9%, and it is
the manipulation check: with `P(body > 0) = 1`, emitted `P(y>0) = gate` exactly, so this measures
that the gate is calibrated, which `P2` already established at median `|gate − p| = 0.0054`.

## ⭐ The finding the 2×2 was built for: the two levers are SUBSTITUTES, not additive

| | |
|---|---|
| theta rescale under **NB** emission | **−9.28%** |
| theta rescale under **truncated** emission | **−3.60%** |
| truncation at the **model's** theta | **−5.94%** |
| truncation at **theta ×22** | **−0.05%** |

⇒ Each lever alone recovers 6–9%; together they recover 9.33%, barely more than the better one
alone. **They are doing the same job — moving mass off the zero atom — by two different routes.**

⛔ **F4 as worded did NOT fire** (the theta effect does not *reverse* sign, it shrinks), so the
falsifier is recorded as not-fired rather than stretched to cover this. But the substitution is
real, and the committed consequence applies: **neither number may be quoted as additive with the
other.**

## ⭐ Why truncation is nonetheless the better of the two, despite the smaller CRPS gain

1. ⛔ **It has NO free parameter.** The theta route needs a scale fitted on data (21.99, fitted in
   sample). Truncation is a change of emission rule with nothing to tune, nothing to overfit and
   nothing to re-fit per model.
2. **It lands the positive rate where the truth is.** 0.0037 against 0.0034 observed, versus the
   theta route's 0.0022 — which is still 35% short.
3. It is the combination `M1` computed analytically at **76.3%** of the zero-mass gap and that the
   engine has never been able to run.

## ⇒ Verdict and what it licenses

**The emit-time axis is worth building.** §2 committed to building only if the arithmetic paid, and
it pays: a parameter-free emission change recovers 5.94% of composed CRPS, fixes the emitted
positive rate to within 9% of observed, and needs no retraining.

⛔ **What this does NOT establish**, and none of it may be skipped when the axis is written up:

- **n = 1, one target (`sb`), one vehicle**, and that vehicle's checkpoint is **deleted** — this is
  the last question the frozen cube can answer.
- It is a change to **emission on delivered weights**. It says nothing about training.
- `crps_all` is one FAO-02 column. **AP, Brier, MCR and `crps_events` are unmeasured here**, and
  C-352 records that one of the platform's three AP constructions is coupled to exactly the
  dispersion this changes. ⇒ The build must be scored on the full battery, not on this number.
- ⚠️ The gain is concentrated in **0.34% of cells**. That is where the signal is, and it is also
  where n is smallest.
