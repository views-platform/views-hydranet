# RESULTS — the gate under-dispersion programme, complete measurement record

**Date:** 2026-10-18 · **Status:** Phase 1 complete. ⛔ **Nothing further runs until this document, `RND_PLAN.md` and the Phase-2 pre-registration exist** (Simon, 2026-10-18).

> ⛔ **The axis is now CLOSED for measurement — see `CONCLUSION.md` (2026-10-19), which carries the
> decision for every question in this record.** This file is the evidence; that file is the verdict.
> The one open result (H2, `weight_decay`, views-models #571/#572) is a named blank there, and the
> decision is the same under every outcome it can have. Phase 2 is **recommended not to run**, so the
> Phase-2 pre-registration named in the status line above is superseded rather than written.

**Executed by:** the `views-models` seat. **Recorded, verified and written by:** this seat (views-hydranet).
**Plan:** `PREREG.md` (14 amendments) · **Evaluation report:** `../2026-10-16_pos_weight_arm_outcome/OUTCOME.md` rev 3
**Scripts:** `results/` — `e1_sigma.py`, `e2_meanlogit.py`, `e6d_align.py`, `e6c_power.py`, `e6f_bn_reestimate.py`, `tier2_remedies.py`, plus `e6f_amber.log` and `tier2_copper.log`.

**Provenance rule for this document:** every number is tagged with who measured it and whether this seat reproduced it. ⛔ **Nothing here is "reported" — it is either MEASURED (with the script), REPRODUCED (independently here) or DERIVED (algebra, with the derivation).** That distinction is the Amendment 6 control applied to our own record.

---

## 0. In plain words

The gate's probabilities are **squashed**: when it says 2% the real rate is nearer 1.3%, and when it says 85% it is nearer 90%. We set out to find what causes that and whether it can be fixed.

**What we found:** the squashing is in the **network**, not in how we average its output. The one cheap fix we expected to work — re-estimating the BatchNorm statistics — **makes it dramatically worse.** A one-parameter recalibration (temperature scaling) **does** fix it on the shipped model, cutting calibration error by 71%, but **fails on the sibling model** whose defect has a second component. And the three models that use a hard threshold instead of a soft gate turn out to be **structurally immune** to the whole defect.

---

## 1. The fact being explained

Fit `logit(observed rate) = a + b·logit(gate)`; honest is `a = 0, b = 1`. h=1, 13 out-of-sample months (457–469), 842,634 cell-months per arm.

| | bin-free `a` | bin-free `b` | binned `b` (n≥1000 / all / n-wtd) |
|---|---|---|---|
| **amber** `pos_weight=2.0`, shipped | **+0.1584** | **1.1875** | 1.146 / 1.177 / 1.139 |
| **copper** `pos_weight=1.0`, killed arm | **+1.2980** | **1.3596** | 1.419 / 1.251 / 1.456 |

**REPRODUCED here** (bin-free fits, IRLS, this seat). `b > 1` in all fits: the gate is `under-dispersed` (GLOSSARY §5b).

⚠️ **`b` is a LOWER BOUND** on the true compression — fitting on a noisy binned predictor attenuates the slope toward 1 by two mechanisms (classical attenuation and noise-driven selection into extreme bins). **DERIVED.**

⚠️ **The binned intercepts are confounded** and must not be quoted: a bin-unweighted intercept is reproducible to 0.0001 from bin weighting plus the slope defect alone (OUTCOME §8.6). **The bin-free pair above is the usable one**, and the 8.2× asymmetry it shows is what explains F8 (§6).

---

## 2. E1 — across-draw logit sd. ⚠️ A MEASUREMENT, NOT A TEST.

| arm | pre-registered floor | median σ | mean | RMS | p75 | implied `T = √(1+πσ²/8)` |
|---|---|---|---|---|---|---|
| amber | ≥ 0.85 | **0.459** | 0.480 | 0.524 | 0.614 | 1.040 |
| copper | ≥ 1.20 | **0.594** | 0.625 | 0.686 | 0.806 | 1.067 |

**MEASURED** (`e1_sigma.py`). Below floor on every summary; the family does not cross it; zero saturated draws; per-cell σ spans 0.005–2.3.

⛔ **This falsifies nothing**, and the plan said so in advance (§4): E1's bands were derived by **inverting `b` values already published**, so agreement would have been consistency, not confirmation. It is the first honest number on that axis and is recorded as a measurement.

---

## 3. E2 — the MC-dropout averaging explains ~8%, not the defect

| arm | `mean(sigmoid)` | `sigmoid(mean logit)` | share of excess explained |
|---|---|---|---|
| amber | 1.1875 | 1.1713 | **8.6%** |
| copper | 1.3596 | 1.3335 | **7.2%** |

Binned `n≥1000`: 1.1462 → 1.1391 and 1.4190 → 1.3782. **MEASURED** (`e2_meanlogit.py`).

⛔ **The pre-registered SECOND outcome fires in both arms: the compression is in the NETWORK, not the reporting path.** R2 (`sigmoid(mean logit)`) survives its kill condition (`b < 1` did not fire) but recovers under a tenth of the defect and is **quantitatively pointless as a remedy.**

✅ **This confirms a derivation made before the measurement** (PREREG Amendment 1, ATTACK 1): training runs dropout **unlocked**, so the loss averages over masks and its stationary condition is `E_ε[sigmoid(z+ε)] = wp/(1+(w−1)p)` — **the mean-of-sigmoid that inference reports.** The Jensen compression is absorbed into the weights and contributes nothing independently. **Predicted qualitatively, measured at ~8%.**

**Residual the remaining candidates must explain: 1.171 (amber) / 1.334 (copper).**

---

## 4. The recalibration sample is degenerate — ⛔ a primary finding

| fact | value |
|---|---|
| `qualified_count` at each of the 30 recal lesson indices | **4 or 5 cells** |
| distinct `(target, threshold)` strata across those 30 indices | **6**, cycling five times |
| curriculum ratio at steps 0–29 | **0.665, flat** |
| the roof clip | `min(ratio, max_ratio × roof_ratio) = 0.95 × 0.7 = 0.665` |
| the clip releases at | **step 214 of 900** = lesson 71 of 300 |
| mean ratio over all 900 training steps | **0.3540** |
| ⛔ the recal sample is | **1.88× hotter than the average training step** |
| training below ratio 0.36 | **50.8%**; 25.0% sits at the 0.05 floor |

`qualified_count` and the strata count: **MEASURED** (`views-models` seat). The clip mechanism, the release step and the 1.88×: **REPRODUCED here** by derivation and recomputation (`b = −0.9/(900×0.75) = −0.001333`; `curriculum.py:91,102`).

⛔ **So the BatchNorm running statistics shipped inside both delivered artifacts were estimated over windows anchored on 4–5 cells per stratum, in 6 strata, drawn entirely from a flat region covering the hottest 23.8% of the schedule — and are then applied, frozen, to 64,818 land cells globally.** Not activity-enriched. **Degenerate.**

**This establishes H1's premise by direct measurement, far more strongly than E6 could have.** It is also a defect on its own terms, independent of calibration — **OUTCOME §9 issues 11 and 13.**

---

## 5. E6 — the window experiment. ⛔ NOT WORTH RUNNING, by measurement.

**Alignment proven first** (`e6d_align.py`, **MEASURED**): window at `(92, 496)`, the unique full-block match over all 300 timesteps × 11 channels. ⚠️ Single-slice matching would have been **insufficient** — channels 3 and 4 are constant-valued, giving 57 spurious candidates. Closed-form recovery `r0 = p_row + (h_max − DIM − new_row_offset)`, `c0 = new_col_offset − p_col`, validated against the content proof.

**E6c power** (`e6c_power.py`, **MEASURED**), locked rule `get_lesson(w mod 30)`, interior margin 13:

| N win | nominal | UNION | dup | cell-months | pos | `b_crop` | `sd(b)` window-boot | gate (<0.027) |
|---|---|---|---|---|---|---|---|---|
| 30 | 312 | 294 | 1.1× | 3,822 | 22 | 1.0380 | 0.0666 | **FAIL** |
| 60 | 556 | 528 | 1.1× | 6,864 | 125 | 1.1160 | 0.1269 | **FAIL** |
| 120 | 1,007 | 862 | 1.2× | 11,206 | 131 | 1.1677 | 0.1002 | **FAIL** |
| 300 | 2,421 | 1,838 | 1.3× | **23,894** | **314** | 1.1730 | **0.0470** | **FAIL** |

⛔ **Fails at every count**, and the `sd` sequence is **non-monotone**, so no sufficient count may be extrapolated (the plan's own sweep rule forbids it).

**Two pre-registered confounds measured far smaller than predicted:**
- **F5′ selection confound** `b_grid − b_crop(300)` = **+0.0145**, against A1's 0.099 worst case that forced the whole `b_crop` redesign — **~7× smaller.** The cells these windows select are essentially ordinary.
- **Duplication 1.1–1.3×**, against a prediction of heavy clustering from the 4–5-cell anchor set — **~4× smaller.** The ±32 jitter spreads the draws far more than the anchor count suggested.

⛔ **The term that actually killed E6 was named by neither seat: LAND FRACTION.** Of 10,800 nominal interior cells only **2,421 are land (22.4%)**, against 25.0% land globally — **recoverable from a figure already in the plan.** 314 positives, not the 140,400 cell-months this seat wrote into Amendment 5's own table.

---

## 6. E6f — the discriminator. ⛔ H1's remedy is DEAD.

All preconditions met: paired baseline `b_before = 1.1875` reproducing F3′'s target **exactly**; `_reset_bn_stats` over 15 layers, `momentum=None`; 336 full-grid 360×720 months over training months 121–456; dropout ON; no augmentation; `no_grad`; `_assert_bn_buffers_finite` ✅ finite. **MEASURED** (`e6f_bn_reestimate.py`, `e6f_amber.log`); 554 s baseline + 1,153 s re-estimation = **28.4 min at 3.43 s/month.**

| | `b` | `\|b−1\|` | pooled mean gate | max signed error | cells > 0.9 |
|---|---|---|---|---|---|
| before (as shipped) | **1.1875** | 0.1875 | 0.00417 (1.21× truth) | +0.0587 | 0.008% |
| after re-estimation | **0.7793** | **0.2207** ⬆ | 0.01981 (**5.75× truth**) | **+0.6872** | 0.44% |

⛔ **It did not approach 1.0 — it crossed and overshot.** `F1⁗`'s equivalence band `|b−1| ≤ 0.0214`: **NOT MET.** The after-curve over-states at **every** level (gate 0.24 → truth 0.023; gate 0.97 → truth 0.42).

⛔ **Re-estimating the BatchNorm statistics on the inference distribution produces a grossly over-confident gate and makes calibration worse in absolute terms.** A 5.75× mean gate would also wreck the firing rate and the composed forecast.

### What E6f established beyond the kill

✅ **H1's mechanism is confirmed with enormous leverage: 0.41 of slope movement from changing nothing but 15 layers' running means and variances** — more than twice the entire defect. **BatchNorm statistics are the dominant control on this gate's calibration.** ⚠️ And this is a **LOWER BOUND** on the train/deploy mismatch: statistics are only the *adjustable* channel; the **weights** were fitted on activity-seeking crops and no eval statistic touches that.

⛔ **The criterion was backwards, and no fixed statistic satisfies the corrected one.** `_recalibrate_bn` calls `train()`, so BatchNorm normalises **each batch by its own moments** during both training and recalibration. So the head was trained against activations standardised **per batch, on activity-selected crops** — and:

```
(a) running stats = E[training batch moments]  ->  x_hat matches what the head SAW
(b) running stats = eval-data moments          ->  x_hat correctly STANDARDISED for the data
```

**They coincide only when training batches are i.i.d. samples of the eval distribution. Activity-seeking crops are not.** So `b = 1.19` and `b = 0.78` are **both correct implementations of incompatible criteria** — ⛔ **the axis has no correct point, not merely no reachable one.** **DERIVED**, jointly, and it explains the sign of both endpoints:

| eval statistics | what `x̂ = 0` means to the head | effect | measured |
|---|---|---|---|
| hot moments on the global grid | global spread is **smaller** than assumed; `x̂` compresses | head **under**-reads | `b = 1.19` |
| global moments on the global grid | `x̂ = 0` now means *quiet globally*; `+2σ` is a stronger claim | head **over**-reads | `b = 0.78` |

⛔ **GOODHART BAN, pre-committed:** `b` crossed 1.0, so by continuity some blend of recal distributions gives `b = 1.000` exactly. **Tuning the recal distribution to hit it is forbidden** — it is a nuisance parameter fitted to the endpoint it is judged on, with no principled value, and it would move the firing rate as a side effect. **The ban extends to pre-specified blends**, because no fixed statistic is correct.

---

## 7. Tier 2 — the remedies. ✅ R1 CONFIRMED on the shipped model.

**AMBER** (**MEASURED**, `tier2_remedies.py`, 13 origin folds, fit on 12 evaluate on 1):

| | `b` held-out | `\|gate−obs\|` | AP | sharpness |
|---|---|---|---|---|
| raw | 1.1917 | 0.00981 | 0.54101 | 0.03439 |
| **R1** temperature, 1 param | **1.0378** | **0.00287** | 0.54101 | 0.03380 |
| R3 Platt, 2 param | 1.0037 | 0.00335 | 0.54101 | 0.03572 |
| R4 isotonic | 0.9907 | 0.00482 | 0.53837 | 0.03555 |

`T` mean **0.8708**, min 0.8674, max 0.8747 — **spread 0.8%.**

| | verdict |
|---|---|
| **F2** `\|b−1\| = 0.0378` vs 0.05 | ✅ PASS |
| **F9** `T` spread 0.8% vs 15% | ✅ PASS — **a single global temperature IS well-defined** |
| **F8** R4 0.00482 vs R1 0.00287 | ✅ PASS — **isotonic is 68% WORSE; the defect IS a pure slope** |
| **F7** `ΔAP = 0.00000` exactly | ⛔ **FIRES** |
| **F10** rank identity True/True, 14 new ties | ✅ PASS |

✅ **R1 cuts held-out calibration error by 71% with ONE parameter, stable to 0.8% across folds, and the non-parametric ceiling cannot beat it. The first intervention in this programme with a measured benefit.**

**COPPER — ⛔ R1 FAILS:**

| | `b` held-out | `\|gate−obs\|` |
|---|---|---|
| raw | 1.3633 | 0.00878 |
| **R1** | **1.2482** | **0.01136** ⬆ worse than raw |
| R3 | 1.0028 | 0.00929 |
| **R4** | 0.9928 | **0.00430** ← best |

**F2 FIRES** and **F8 FIRES.** `T = 0.9155` where the slope needs `1/1.36 = 0.73`: **NLL-fitted temperature optimises likelihood, not the slope.**

⛔ **So "the defect is a pure slope" is TRUE for the shipped model and FALSE for the arm** — and it was predictable from the bin-free intercepts in §1: amber **+0.16** (slope-only) against copper **+1.30** (slope **and** level), 8.2×. **One parameter cannot address two components; R3's two fix it exactly.**

### Aggregate firing rate under R1

| | raw | R1 |
|---|---|---|
| amber | 1.209× truth | **0.951×** ✅ improves |
| copper | 0.945× truth | **0.733×** ⛔ worsens |

**amber REPRODUCED here** (0.004167 → 0.003278, −21.3%). Cells above gate 0.5 invariant in both (1296, 777) — sign-preserving, a free confirmation of rank preservation.

⛔ **F7's verdict is NOT "cosmetic", and this seat retracts that word.** `ΔAP = 0` is a **theorem** under a monotone map, not a measurement. But the composed forecast is `gate × magnitude`, so changed probabilities **change predicted counts**, and AP structurally cannot see that. **R1 buys honest probabilities and nothing on the rank axis.** Whether that is worthless or the point is a **product question** — ruled on by Simon, 2026-10-18: *"yes 100%"*.

⚠️ ~~**F11 (the firing-rate check on a corrected gate) is UNRUN.** It needs the composed forecast, which Tier 2 does not build, and the uncomposed body output was not retained. **It is the last gate before production.**~~

✅ **SUPERSEDED 2026-10-19 by §7b — F11 is RUN, it needed no GPU, and it separates the two arms.**

---

## 7b. ✅ F11 — RUN 2026-10-19. ⛔ R1 PASSES on amber and FAILS on copper, on the composed forecast.

**MEASURED**, `results/f11_firing.py`, 13 origins × 3 targets × 64,818 cells × 16 draws = **13,482,144 cell-draws per target per model.** Truth is this session's own **FRESH** datafactory pull (`results/truth_pull.py`), which reproduced the Phase-1 cache **exactly** on all three targets — `act_true` sb 0.00344752, ns 0.00120100, os 0.00129119.

### ⛔ Why this needed no GPU, and the plan said it did

`RND_PLAN §5` states F11 *"requires a re-run for the kestrels"* because amber's `by_*` archive is 9% truncated and `DO_NOT_USE`. ⛔ **That is true of the 36-horizon archive and false at h=1.** A hand-rescued `gate_h1` slice survives for **both** kestrels — 13 origins × 3 targets × (64818, 16), every value finite per its own `SLICE_MANIFEST.json` — and the composed draws survive complete in `*_lr.tar.zst`. **Nothing at h=1 is missing.** `RND_PLAN §5` and §10 are corrected.

### The estimator — exact in expectation, not an approximation

Under `soft_gate`, `compose_samples` draws `Bernoulli(gate) × body_draw` per pass, so for cell `c`, pass `d`, the composed indicator `z = 1{composed > 0}` is Bernoulli with `E[z] = g·p`, where `p = P(body draw > 0)` is **untouched by a gate recalibration.** Therefore

```
E[ z · (g'/g) ] = g'·p = E[ z' ]            g' = sigmoid(logit(g)/T)
```

**The observed composed indicator, reweighted by the gate ratio, is an unbiased estimator of the corrected composed firing rate.** No per-cell inversion, no clipping, no imputation, and valid whether `g'` is above or below `g`. ⛔ **The body is never needed — which is exactly why the truncated `by_*` archive does not block this.** Max importance weight **1.0392** (amber) / **1.0249** (copper); the `g' > g` region is **≤ 0.16%** of cell-draws, so the variance cost is negligible.

### ✅ An independent check that the join is right, and it is a strong one

`Σz / Σg` estimates pooled `P(body draw > 0)`. The kestrels are `truncated_nb` — a **zero-truncated body cannot draw zero**, so this must be **1.0** exactly:

| | sb | ns | os |
|---|---|---|---|
| amber | **1.0070** | **1.0067** | **0.9956** |
| copper | **0.9986** | **0.9861** | **0.9964** |

**All six within ~1.5 Monte-Carlo SE of 1.0.** ⛔ **So at h=1 the composed firing indicator IS the gate's Bernoulli draw, and the body contributes nothing to occurrence.** The "F11 needs the composed forecast" obstacle was real for a `nb` body and vacuous for a truncated one.

### The result — `act_ratio` = P(pred>0)/P(truth>0), `activation_metrics.py:38`

The committed instrument's `act_pred` was recomputed independently and **matched the accumulation to every printed digit** on all six series.

| model | target | `act_true` | raw | **`act_ratio` raw** | R1 | **`act_ratio` R1** | T-sweep |
|---|---|---|---|---|---|---|---|
| **amber** | sb | 0.0034475 | 0.0042534 | **1.2338** | 0.0033573 | ✅ **0.9738** | 0.9681 – 0.9804 |
| **amber** | ns | 0.0012010 | 0.0016635 | **1.3851** | 0.0012585 | ✅ **1.0479** | 1.0407 – 1.0561 |
| **amber** | os | 0.0012912 | 0.0017868 | **1.3838** | 0.0012200 | ✅ **0.9449** | 0.9354 – 0.9558 |
| **amber** | **pooled** | 0.0019799 | 0.0025679 | **1.2970** | 0.0019452 | ✅ **0.9825** | |
| **copper** | sb | 0.0034475 | 0.0024119 | 0.6996 | 0.0020494 | ⛔ **0.5944** | single `T` on record |
| **copper** | ns | 0.0012010 | 0.0008145 | 0.6782 | 0.0006506 | ⛔ **0.5418** | |
| **copper** | os | 0.0012912 | 0.0009778 | 0.7573 | 0.0006874 | ⛔ **0.5323** | |
| **copper** | **pooled** | 0.0019799 | 0.0014014 | 0.7078 | 0.0011291 | ⛔ **0.5703** | |

✅ **AMBER: all three targets move from over-firing (1.23×–1.39×) to within 5.5% of honest, and the pooled rate lands at 0.9825.** The **sweep rule** is satisfied: `T` swept over its full measured fold range (0.8674–0.8747) moves `act_ratio` by at most 0.012 and **the family does not cross 1.0 on any target.** ⛔ **F11 does not fire. R1 improves amber's firing rate as well as its calibration.**

⛔ **COPPER: the opposite, and worse than §7's aggregate suggested.** Copper already **under**-fires at 0.68×–0.76×, and R1 drives it to **0.53×–0.59×** — *away* from honest on every target. **F11 FIRES on copper.** This is now the **third** independent axis on which R1 fails the arm (F2 held-out slope, F8 isotonic beats it, F11 firing rate), and it is the one with a product consequence: a 43% firing deficit on a composed count forecast.

### ⚠️ What F11 still cannot say, stated rather than glossed

- ⛔ **`precision_at_k` is INVARIANT under R1 by construction** — top-k by gate score is rank-preserving, so the fired set is identical up to the 14 new ties F10 recorded. **DERIVED, and consistent with the measured `ΔAP = 0.00000` exactly.** The standing rule *"read `act_ratio` with `precision_at_k`, always"* therefore has nothing to add here: R1 moves firing **volume** and **cannot** move placement. ⚠️ **So F11 establishes that the volume is right, not that the firing lands anywhere better.**
- ⛔ **CRPS on the composed forecast under the corrected gate is NOT recoverable from these archives.** The per-cell body sample exists only where a draw fired, and with a mean gate of ~0.004 the overwhelming majority of cells have **zero** positive draws out of 16 — so the body distribution is absent exactly where the gate is small, which is nearly everywhere. Substituting a pooled body sample would be a nuisance parameter standing in for the thing being measured, which this programme has banned twice. ⛔ **The CRPS half of `RND_PLAN §5` still needs a re-run; the `act_ratio` half never did.**
- ⚠️ **No numeric guardrail band for `act_ratio` is pre-registered anywhere.** `PREREG:186` says *"`act_ratio` leaves its guardrail band"* and never states the band; the prior arm used *"≤ control"* (`OUTCOME.md` P-6). **A falsifier with no threshold cannot fire on a number alone**, so the verdicts above are stated as direction and magnitude against 1.0, which is the one value `act_ratio` has a meaning at. **Recorded as a gap in this programme's own pre-registration.**
- ⚠️ **h=1 only.** The delivery is 36 months and the gate posterior is not retained beyond h=1 (#558 / C-161). **Unchanged by this measurement.**
- ⚠️ **§7's "amber REPRODUCED here (0.004167 → 0.003278)" is a DIFFERENT object from this table's 0.0042534 → 0.0033573.** §7's is a 4-pass posterior-**mean** gate from P2's scoring sweep; this is the **16 delivered draws** of the production run. The −20.5% here against §7's −21.3% is agreement between two different measurements, **not a reproduction**, and calling it one would be the provenance-mismatch class.

---

## 8. ⛔ The threshold-gate models are STRUCTURALLY IMMUNE — and their τ already absorbs the defect

**Composition modes across the roster** (**MEASURED** independently by both seats; we agree exactly). Only 10 models set `forecast_composition`:

| mode | models |
|---|---|
| **`soft_gate`** (7) | amber_kestrel, copper_kestrel (`truncated_nb`); blue_stranger, pink_pirate, purple_alien (`mixture_nb`); bright_starship, heavy_freighter (`nb`) |
| **`threshold_gate`** (3) | **violet_visitor** τ=0.20 (`nb`); **blazing_meteor** τ=0.16, **bold_comet** τ=0.14 (`mixture_nb`) |

⛔ **Two composition paths with different gate semantics**, per `composition.py`'s own contract:
- `compose_samples` (sample cube): `soft_gate` → `bernoulli(gate)`, a **0/1 draw**; `threshold_gate` → `(gate ≥ τ)`.
- `compose_mean` (point / AR-feedback): `soft_gate` → **`gate × mean`, multiplicative, never zeroes**; `threshold_gate` → `(gate ≥ τ) × mean`.

They agree in expectation (`gate × mean = E[Bernoulli(gate)·Y]`) and **differ per draw.**

### The invariance — DERIVED and verified exactly

Temperature is strictly increasing and `T > 0`, so every step is an iff:

```
sigmoid(logit(g)/T) ≥ τ   ⟺   logit(g) ≥ T·logit(τ)   ⟺   g ≥ sigmoid(T·logit(τ))
```

⛔ **Identical firing set. REPRODUCED here on 200,000 random draws: 200,000/200,000 identical decisions, exact.**

| model | τ now | equivalent τ on the raw gate (`T = 0.8708`) | shift |
|---|---|---|---|
| violet_visitor | 0.20 | **0.2302** | +15.1% |
| blazing_meteor | 0.16 | **0.1909** | +19.3% |
| bold_comet | 0.14 | **0.1707** | +21.9% |

**So calibrating the gate and moving τ are the same knob**, and since under-dispersion is a **monotone** distortion, ⛔ **a threshold gate is invariant to it up to the choice of τ. The three hard-gate models cannot be harmed by this defect in their firing set.**

### ⛔ And the sharper consequence this seat adds

"Immune" understates it. **τ was selected empirically** (#465 moved it off 0.5). ⛔ **So the tuning absorbed the distortion: those models have already had the correction applied, implicitly, through τ.** Two consequences:

1. **This is the mechanism for Simon's own observation that the threshold models "perhaps work better."** Not a coincidence, and not evidence that thresholding is a better design — **evidence that an empirically tuned threshold is robust to a monotone calibration defect that a soft gate passes straight through.**
2. ⛔ **Applying R1 to a threshold model while keeping τ would CHANGE its firing set for no reason** — it would move the effective threshold from 0.2302 back to 0.20. **R1 on a hard-gate model requires re-deriving τ, and buys nothing, because τ already encodes the correction.**

⚠️ **Simon, 2026-10-18: too soon to conclude, and he is not planning to change them — *"if anything the opposite."*** Recorded as his ruling, not as a recommendation.

---

## 9. ⛔ A GOVERNANCE DEFECT FOUND WHILE WRITING THIS RECORD

`RESULTS_LEDGER.md:613-623` defines the **eligibility conjunction** that decides whether a model may ship:

> 1. **Superiority:** CRPS ≥ **5%** better than baseline, *and*
> 2. **All guardrails non-inferior:** QS99 & Brier within 1%; MCR calibration-distance closer to 1.

⛔ **There is NO occurrence guardrail.** And `OUTCOME.md §3` measured that **halving the firing rate buys 37–83% CRPS.** So **a model that halves its firing rate satisfies criterion 1 outright**, and none of the three guardrails reliably catches it: MCR is the **pooled magnitude ratio that `audit_run.py:137` explicitly demotes** (*"DO NOT read as calibration"*), and Brier on a 99.7%-zero field is dominated by the zeros — the documented #258 blindness (*"the models under-activate 4–115× yet MCR ≈ 1"*).

⛔ **The shipping criteria are satisfiable by the exact artefact this programme found.** `OUTCOME.md §3.5`'s standing rule — *"CRPS rewards timid under-firing; always read it with `act_ratio`"* — is precisely the missing guardrail.

⚠️ **Also inconsistent in the same table:** it lists **MCR** as *"guardrail — magnitude calibration"*, which `audit_run.py:137` forbids in writing. **Flagged, not edited** — the eligibility criteria are governance and changing them is Simon's call, not this seat's. **Proposed as an issue in `RND_PLAN.md`.**

---

## 10. What this programme established, and what it did not

**Established:**
- The gate is **under-dispersed**, `b ≈ 1.19` on the shipped model, measured six ways and surviving ten rounds of two-seat adversarial review.
- The compression is **in the network** (~8% from the reporting path).
- **`pos_weight` cannot address it at any value** — `logit(s*) = log(w) + logit(p)` exactly, a pure intercept shift with zero slope effect.
- **BatchNorm statistics are the dominant adjustable control** (0.41 of leverage, a lower bound), and **no fixed choice of them is correct.**
- **R1 fixes the shipped model** (71%, one parameter, `T` stable to 0.8%, isotonic cannot beat it) and **fails the arm**, whose defect has a second component.
- **The three threshold-gate models are structurally immune**, and their tuned τ already absorbs the distortion.
- **13 issues**, several in production code.

**NOT established:**
- ⛔ **H3 — the resolution limit.** F7 firing is *consistent with* it but cannot establish it, because **AP is unchanged by construction under a monotone map: a test that cannot fail cannot confirm.** Live and untested.
- ⛔ **H2 — `weight_decay: 0.1`.** Live, and the premise this seat argued it from (that the weights pin the slope) is **damaged** by E6f's 0.41 of movement with the weights untouched.
- ✅ **The firing rate under a corrected gate (F11) — RUN 2026-10-19, §7b.** `act_ratio` pooled: amber **1.2970 → 0.9825** (F11 does not fire); copper **0.7078 → 0.5703** (⛔ F11 FIRES). **No GPU.** ⛔ **Still NOT established: CRPS on the composed forecast under the corrected gate** — the body sample is absent wherever the gate is small, which is nearly everywhere, and no substitute is admissible.
- ⛔ **That R1 fires in better PLACES.** `precision_at_k` is invariant under a monotone map by construction. R1 moves firing volume and cannot move placement.
- ⛔ **Anything at h ≥ 2.** Validated at h=1 only; the delivery is 36 months; #558 / C-161.
- ⛔ **Generalisation beyond amber and copper.** n=2, same architecture, one config key apart — **and the fix that works on one fails on the other.** No gate-calibration number exists for any served model.
- ⛔ **The root cause.** Train/deploy geometry; training-side; never scoped by this programme.

---

## 11. Error record for this programme

Both seats' withdrawn claims, because a record that keeps only the surviving ones is not a record.

**This seat withdrew 7:** the "second hidden bias" reading (rev 1 of OUTCOME); "`pos_weight` is inert" (too strong — the mechanism claim is correct, the artifact claim was not); the grid dimension (**180×180 — the africa figure carried into a global claim**, in two committed documents and project memory); the ±0.10 tolerance justification (a systematic range used to size a standard error); **two jointly-unsatisfiable corrections** (Amendment 3's interior rule against Amendment 2's power gate); the 300-window power conclusion; "C-184 is right by design"; the tag-propagation rule that forbade its own authority measurement; "nothing asked what the field name was asserting" (**it did — `EVIDENCE.md:17-20`**); and `sharpness_scorecard.py` "does not exist" (**it does — and the mechanism was a shell-logic error: `grep | head || find` never ran the `find`**).

**The other seat withdrew 5:** the 39% stationary-condition headline (Jensen bias, zero at `w=1` by construction); its own 0.0218 power bound (uniform subsampling of a clustered design); the H4 recurrent limb (**refuted by this repo's own registered-and-scored prediction**); the A1 worst case (0.099 against a realised +0.0145); and the clustered-overlap prediction (1.1–1.3× against "heavy").

**Blind predictions, scored:** this seat's O-3 (E6f would land at 1.10–1.16) — ⛔ **WRONG**, measured 0.7793, logged **W-25**. O-4 (cost in minutes not hours) — right as worded at 28.4 min, with the implicature wrong. ✅ The two that held: AP cannot move by a monotone recalibration (`ΔAP = 0.00000` exactly), and the Jensen contribution is non-independent (~8%).

**Nine cost or power estimates across both seats, nine optimistic.** ⛔ **Neither seat states a cost or a power figure again without measuring one first.**

---

## 12. Prior art found in this repository, unsearched

Five pieces, each found by the other seat **after** the fact:

| | what it was |
|---|---|
| `2026-06-20_gate_calibration_dossier/` + **`scripts/gate_reliability.py`** (`eb48924`) | a pre-registered gate-reliability plan and **a committed binned reliability diagram with bootstrap CIs.** P2 reinvented the instrument |
| `2026-08-23_state_range_dossier` AMENDMENT 3 + F3 | **the interior-vs-pooled boundary confound, measured at 142×** — in a file whose docstring explains why E6 as specified cannot work |
| the same dossier's AMENDMENT 4 | **the recurrent-reach argument, registered and scored WRONG** — which this seat then reproduced |
| `2026-08-03_lockeddropout_calibration_dossier` | a **LOCKED** plan on *"improving calibration if the current posterior is too tight"* |
| **`audit_run.py:55-60,130,135,137`** | **`range_compression`** (*">>1 = compression"*), **`gate_brier`**, and **the MCR demotion** |

⛔ **The mechanism for missing them was not laziness.** `MEMORY.md` filed the gate-calibration threads under *"Superseded/history"* — **a marker that said not to look**, and which was right about the *conclusions* while silently retiring the *instruments*. Three controls adopted: search `reports/` per candidate and record the terms; **search `scripts/` too**; and **a "superseded" marker retires a conclusion, never a tool, an instrument or a falsifier.** ⛔ **A fourth, added after `range_compression`: search the CONCEPT, not the file — metric definitions inside scoring functions are a third hiding place.**
