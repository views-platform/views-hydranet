# (c) The result — the instrument works, and ⛔ ITS NULL DOES NOT

**Date:** 2026-11-09 · `tools/simulate_ordering_discrimination.py` → `results/simulate_ordering_discrimination.out`
⛔ **PLANTED DATA ONLY.** No `mu`, no checkpoint, no arm, no real truth. **Nothing here is a finding
about the model**, and no `v1` decision is taken.

---

## ⭐⭐ THE HEADLINE, IN ONE LINE

**The estimator is correct and the mechanism is confirmed — but none of the four resampling schemes
recovers test size under joint dependence, so there is NO VALID NULL yet, and `DSC_ORD` on real `mu`
cannot be interpreted until there is.**

⇒ ⛔ **This blocks the measurement.** It is the step-(c) discipline doing exactly its job: a defect
that would have produced a confident, significant, wrong answer was found **before** `mu` was read.

---

## 1 Scoreboard — 7 held, 3 refuted, 1 part-tested

| id | prior | prediction | outcome |
|---|---|---|---|
| `P-81` | ~99% | `G5a`: `x=y` ⇒ `CRPS_ORD` exactly `0.0` | ✅ **HELD** — `0.0` bit-exact |
| ⭐ `P-82` | ~90% | `G5b`: `x=−y` ⇒ zero only *within tolerance* | ⛔ **REFUTED** — bit-exact. **The auditor was wrong and my first version was right** |
| `P-83` | ~85% | `G12` fires: dropped `sample_weight` changes `R*` | ✅ **HELD** — but see §4; my *first* `G12` could not fire |
| ⭐ `P-84` | ~95% | step 2 recovers `U/(U+Δ)` to <1e-10 | ✅ **HELD** — `5.6e-17`. **The auditor's algebra is exact** |
| ⭐ `P-85` | ~60% | null `R*` is of **order** `log n / n` | ✅ **HELD** — `0.00594` measured vs `log n/n = 0.00591` at `n=1200`. Constant ≈ **1.0** |
| `P-86` | ~70% | `PERMUTE` exact on both i.i.d.-`y` plantings | ✅ **HELD** — `0.050` and `0.050` against `α = 0.05` |
| `P-87` | ~65% | `PERMUTE` over-rejects when both fields are dependent | ✅ **HELD** — `0.917` |
| ⭐ `P-88` | ~45% | blocky-limit ratio in `[0.5b, b]` | ⛔ **REFUTED** — ratio ≈ **1.1–1.2 × b**, i.e. slightly **ABOVE** `b` |
| ⛔⛔ `P-89` | ~55% | some scheme recovers size on 4c (≤ `2α`) | ⛔ **REFUTED** — best is `LON_SHIFT` at **0.267**; `STRATIFIED` **0.883** |
| `P-90` | ~80% | `G7`/`G10` hold on every rung and both cut rules | ✅ **HELD** |
| `P-91` | ~50% | LOMO recovers planted skill **and** the gap sees drift | ⚠️ **PART**: the gap sees drift emphatically (`+1.90` vs `+0.016`); the *"within 20% of a planted value"* clause was **never tested** — no known OOS skill was planted |

⚠️ **Read this down, not across.** `P-81` is the self-test and was low-risk by design. The three
genuinely uncertain ones were `P-85`, `P-88` and `P-91`; one held dead-on, one was refuted, one is
half-tested.

---

## 2 ⛔⛔ THE BLOCKING FINDING: the three-point pattern reproduces, and no scheme survives it

| planted `H_0` | `PERMUTE` | `LON_SHIFT` | `STRATIFIED` | predicted |
|---|---|---|---|---|
| **4a** both i.i.d. | **0.050** | 0.050 | 0.050 | exact ✅ |
| **4b** `x` smooth, `y` i.i.d. — ⭐ the CONTROL | **0.050** | 0.050 | 0.083 | exact ✅ |
| **4c** both smooth | ⛔ **0.917** | ⚠️ 0.267 | ⛔ 0.883 | over-reject ✅ |

⭐⭐ **The mechanism is confirmed, not merely the direction.** All three rows behave as `02§4.4c`
predicted: exact, exact, over-reject. ⛔ And the control matters — had 4b shown distortion, the
finding would have been in the resampler, not the mechanism. It did not.

⇒ **`B2` stands, and it is enormous:** at `α = 0.05`, `PERMUTE` rejects a **true null 92% of the
time** under joint dependence. A permutation p-value on real `mu` would be meaningless in the
direction that waves `v1` through.

⛔⛔ **But `P-89` is refuted and that is the blocker.** `LON_SHIFT` is a **3.4× improvement** on
`PERMUTE` and still rejects at `0.267`. `STRATIFIED` barely helps. ⇒ **We do not have a valid null.**

⚠️⚠️ **And the honest limitation on that verdict.** My "spatial" planting is a **block-constant
surrogate** (`_smooth`: groups of `b` consecutive rows), **not** a real lat/lon grid with smooth
fields and a land mask. `LON_SHIFT` shifts within synthetic latitude bands (`lat = grp % 40`), so
consecutive groups land in *different* bands and the shift is not given a geometry it was designed
for. ⇒ **`LON_SHIFT`'s 0.267 may understate it badly, and `P-89`'s refutation is a statement about
this planting, not a proof that no shift scheme can work.** ⛔ The next step is a faithful grid —
**not** a conclusion that the null is impossible.

---

## 3 ⭐ The two measurements that justify the whole apparatus

### 3.1 The CRPS bug would have overstated out-of-sample skill by 83%

Step 2, with out-of-range mass actually planted (`Δ = 8.995`, factor `U/(U+Δ) = 0.547`):

| | |
|---|---|
| what v1's formula would have **reported** | `SKILL = 0.8818` |
| what is **true** | `SKILL = 0.4823` |
| the relation, recovered | `|true − predicted| = 5.6e-17` |

⇒ ⛔ A defect I found by inspection and the auditor confirmed with three parts would have **nearly
doubled** the headline. ⭐ And the correction is now verified to machine precision, so the auditor's
`U/(U+Δ)` algebra — which it flagged as derived-but-not-executed — is **exact**.

### 3.2 In-sample overstates what is usable by ~40%, with no drift at all

| | in-sample `R*` | LOMO | rolling-origin | gap |
|---|---|---|---|---|
| no drift | **0.957** | 0.573 | 0.556 | +0.016 |
| drift planted | 0.606 | 0.121 | ⛔ **−1.781** | ⭐ **+1.903** |

⭐⭐ **This is the audit's largest point, measured.** The in-sample statistic reads `0.957` where only
`0.573` generalises — and that is with a *stationary* process. ⇒ **The decision must hang on the
out-of-sample number**, exactly as `§6` now says.

⭐ And under planted drift, **rolling-origin goes to `−1.78`** — far worse than no model — while LOMO
still reads `+0.12`. The LOMO-minus-rolling gap is **116× larger** than its stationary value. ⇒ The
gap is a **working non-stationarity detector**, and over 13 months that is a thing we need before
building `v1`.

### 3.3 Three free parameters measured to matter, not argued to

| parameter | effect |
|---|---|
| ⚠️ **scale of `y`** | `R*` = **0.881** raw vs **0.763** on `log1p` ⇒ §7.2 confirmed: the answer is scale-dependent, so the scale is part of the claim |
| ⚠️ **cut rule** | at rung 4, `R*` = **0.349** (equal-count) vs **0.467** (equal-width); realised `d'` diverges at the top rungs (64/128 vs 58/108) ⇒ **swept, never selected** |
| ⚠️ **tail weight** | the **single largest gap carries 63.4%** of the CRPS integral (1,478 of 2,332) ⇒ §7.2's tail dominance is not a worry, it is the arithmetic |

---

## 4 ⛔ Three defects in MY OWN instrument, found by running it

| | what |
|---|---|
| ⛔⛔ | **`G12` could not fire as first written** — the gate built to catch the likeliest silent error. It used a 90/10 split of a **monotone** signal, so `r` was already feasible, **PAV pooled nothing**, and the weights could not matter. ⇒ Precisely the defect class the audit condemned `G9` for, committed by me one file later. Fixed with unequal blocks **plus** a monotonicity violation. ⚠️ **And the honest caveat on the fix:** in the repaired case the unweighted answer is `R* = −2.97`, which **`G1` would also catch** — so `G12` is demonstrated to *fire*, not demonstrated to catch something otherwise *silent*. A case where the wrong answer is plausible is still owed |
| ⛔ | **Step 2 was vacuous on the first run** — a random split put **no** test cell outside the training range, so `Δ = 0` and the relation under test was never exercised. The arithmetic agreed to `1e-16` and meant nothing. ⇒ `P-84` was **untested, not confirmed**, until the split was contrived to force out-of-range mass |
| ⛔ | **`n_eff_blocky` is meaningless on a continuous index** — it counts distinct `(x, y)` pairs, so on `plant_signal` it returns `n` and reports `b = 1.00`. It is exact only in the block-constant limit. ⇒ Labelled in the output. ⛔ **`02§7` step 0 calls `n_eff` the first number printed and the parameter that sets the distortion — and we have no estimator of it for real `mu`.** That is now an open item, not a solved one |

ℹ️ And a fourth, from the first crash: `IsotonicRegression.y_` **does not exist** in this sklearn, and
the obvious substitute `transform()`/`fit_transform()` is the **interpolating** path `P5` warned
about — it rebuilds the fit as `interp1d` over block corners, so an interior point gets a *ramped*
value instead of its block's constant. ⇒ Switched to the functional `isotonic_regression`, which
builds no interpolator. ⭐ The audit's warning was right and the API change made it sharper.

---

## 5 ⭐ What (c) settled about the AUDITORS

Three predictions existed to test the audit rather than the data. All three resolved:

| | |
|---|---|
| ⭐ `P-82` **REFUTED the auditor.** | It argued `G5b` would be zero only to rounding because the pooled weighted mean reaches `p[j]` *"by a different arithmetic path"*. It does not — both are `bincount` then divide, so the result is **bit-exact**. My original `G5a`/`G5b` pairing was right and the "correction" was wrong. |
| ✅ `P-84` **CONFIRMED the auditor**, to `5.6e-17`, on algebra it had explicitly flagged as unexecuted. |
| ⭐ `P-88` **REFUTED the auditor's caveat.** It retracted `b·(H_{n/b}/H_n)` for **exactly `b`**, noting that ties should pull the ratio **below** `b`. Measured: `2.41 / 4.01 / 7.79 / 18.65` at `b = 2/4/8/16` — **linear in `b` with a coefficient of ≈1.1–1.2, i.e. slightly ABOVE `b`.** ⇒ *"Of order `b`"* is right; *"bounded above by `b`"* is **wrong**, and `n/n_eff` is a **lower** bound on the distortion rather than an upper one. |

---

## 6 ⛔ What is NOT settled, and what must happen before real `mu`

1. ⛔⛔ **A valid null.** No tested scheme recovers size under joint dependence. Next: a **faithful
   lat/lon planting** with a land mask and genuinely smooth fields, to give `LON_SHIFT` the geometry
   it was designed for. ⛔ Until a scheme recovers size, `DSC_ORD` on `mu` has **no interpretable
   p-value**, and §6's out-of-sample `SKILL` — which needs no null for its level — is the only
   readout that could be quoted.
2. ⭐⭐ **An `n_eff` estimator for a continuous field — NOW ANSWERED, 2026-11-09.** `02§7` makes it
   the first number printed and the parameter that sets the distortion, and §4 records that
   `n_eff_blocky` cannot supply it. ⭐ **`Clifford1989` gives one:** `M̂ = 1 + σ̂_r^{-2}`, where
   `σ̂_r` is the estimated SD of the sample correlation, computed from the two fields' own
   autocorrelation functions — i.e. defined for **continuous** fields, which is exactly what
   `n_eff_blocky` is not. ⇒ This open item becomes **implementation work, not an unsolved problem.**
   ⚠️ With two cautions from the same audit: `Clifford1989`'s own Monte-Carlo missed the 5% target
   in **2 of 12** lattices (C-4683), and `Dutilleul1993` (C-4688) shows the *incomplete* estimator
   differing from the *complete* one by up to **8×** on small lattices. ⇒ The estimator needs its own
   validation before its output is quoted; it is not a plug-in.
3. ⚠️ **A `G12` case where the wrong answer is plausible** rather than caught by `G1`.
4. ⚠️ `P-91`'s first clause — recovery of a **known planted** out-of-sample skill — was never tested.
5. ⚠️ **The maxT band and the bootstrap were not run at all.** `02§4.2` and `H-c` are still owed.
6. ✅ **Clifford et al. 1989 / Dutilleul 1993 — ACQUIRED, indexed and locator-audited** by `library`
   on 2026-11-09 (with `Dormann2007` as a review). ⭐ `Clifford1989` **C-4681 independently confirms
   the anti-conservatism**, quantified at **25–55% rejection against a nominal 5%**. ⇒ `02§4.4`'s
   self-derived argument is **confirmed rather than replaced**: their route is the variance of the
   sample correlation, ours is tie-preserving permutation in the block limit — **same destination,
   different mechanism** — and §4.4(c)'s control row is not in their work at all.
   ⚠️ **Our 91.7% is well above their 25–55%**, and the two are not like-for-like: a permutation test
   rather than a `t`-test, on a deliberately extreme planting. The direction and mechanism agree;
   the magnitude is ours and harsher.
   ⚠️ **Scope limit** (`Dormann2007`, C-4690): these are **test-statistic** corrections and *"do not
   allow for a correction of the parameter estimates."* Fine here — we need a correctly sized test,
   not corrected coefficients — but they fix the inference, not the estimate.

---

## 7 ⛔ What this does NOT say

- ⛔ **Nothing about the model.** No `mu`, no checkpoint, no arm, no `v1` decision. Every number above
  is planted.
- ⛔ **Not that the null is impossible** — only that four schemes failed on **this** planting, whose
  geometry is a block-constant surrogate (§2).
- ⛔ **Not a verdict threshold.** What `R*` would justify building `v1` is a decision rule; it belongs
  in a pre-registration, and it is Simon's.
- ⚠️ `B = 99` in the size study and 60 trials per cell ⇒ Monte-Carlo SE on a rejection rate is
  **≈0.028**. The `0.917` and `0.050`s are far outside it; `STRATIFIED`'s `0.083` on 4b is **not**.


---

## 8 ⭐⭐ (c2), 2026-11-09 — THE BLOCKER IS CLEARED. The shift recovers size on a real grid.

`tools/spatial_null.py` → `results/spatial_null.out`. Pre-registered and committed before it ran.
⛔ Still **planted data only**: a faithful 360×720 raster with a 25.0% land mask (64,800 land cells
against the real 64,818), ~2,905 clustered event cells, two independent FFT-smoothed fields.

### 8.1 The result

| | `PERMUTE` | ⭐ `LON_SHIFT` | `LON_SHIFT_RANK` | `LAT_BLOCK` |
|---|---|---|---|---|
| control, `y` i.i.d. (3 lengths) | 0.00–0.10 ✅ | 0.03–0.07 ✅ | 0.03 ✅ | 0.00–0.07 ✅ |
| **both smooth**, `L = 2` | ⛔ 0.300 | ✅ **0.067** | ✅ 0.067 | ⛔ 0.333 |
| **both smooth**, `L = 6` | ⛔ 0.900 | ✅ **0.033** | ✅ 0.033 | ⛔ 0.833 |
| **both smooth**, `L = 18` | ⛔ 0.867 | ✅ **0.000** | ✅ 0.100 | ⛔ 0.633 |

⭐⭐ **`P-93` HELD. The longitude shift recovers size at every swept correlation length** (≤ `2α`), in
both its plain and rank-reimposed forms. `P-92` (control) and `P-94` (`PERMUTE` still badly broken)
also held.

⇒ ⛔ **`03_FINDING.md` §6.1's blocker is cleared, and §6.1's own caveat was the reason:** (c)'s
`0.267` *was* an artefact of the block-constant surrogate. Given the geometry the shift was designed
for, it drops to **0.00–0.07**.

⛔ **`LAT_BLOCK` fails on every row** (0.333 / 0.833 / 0.633) ⇒ spatial strata do **not** work here.
Of the three schemes, exactly one does.

### 8.2 ⭐⭐ Which reconciliation of `Fortin1996` survives — and the mechanism is VISIBLE

`library`'s reading wins (`02_PSEUDOCODE.md` §4.7): the direction is **statistic-dependent**.
`Fortin1996` holds for boundary overlap; our derivation holds for continuous association; **both stand
without conflict.** `opponent`'s prediction — a material residual over-rejection surviving on a
faithful grid — is **refuted**.

⭐ And `library`'s *mechanism* is directly legible in the `perm null R*` column, which was not what
the predictions were about:

| planted `L` | true null `R*` | `PERMUTE`'s null | `LON_SHIFT`'s null |
|---|---|---|---|
| 2 | 0.0069 | 0.0027 | **0.0058** |
| 6 | 0.0206 | 0.0026 | **0.0286** |
| 18 | 0.1068 | 0.0026 | **0.1145** |

⇒ ⭐⭐ **`PERMUTE`'s reference distribution is FROZEN at the independent level** (0.0026 at every
length) **while the truth rises 40-fold.** The shift's reference **tracks** the truth. That is
*"structure-destruction compresses"* measured directly, and it is why `PERMUTE` fails and the shift
does not.

⚠️ At `L = 18` the shift's null (0.1145) slightly **exceeds** the truth (0.0600) ⇒ mildly
**conservative** at the longest correlation length, which is why its rejection rate is 0.000.
Over-correction, not under-correction — the safe direction, but recorded.

### 8.3 ⛔ `P-95` REFUTED — Clifford's `M̂` does NOT predict the distortion

| `L` | measured distortion | `n / M̂` | ratio |
|---|---|---|---|
| 2 | 2.59 | 1.57 | 1.64 ✅ |
| 6 | 7.81 | 33.69 | ⛔ **0.23** |
| 18 | 41.52 | 215.43 | ⛔ **0.19** |

⇒ `M̂` **overstates the distortion by 4–5×** at realistic correlation lengths. `P-96` held (`M̂` falls
monotonically with correlation length), so it is a valid **ordinal** indicator and **not** a
quantitative predictor.

⛔ **This matters for the design.** `02§7` step 0 makes `n_eff` *"the first number printed and the
parameter that sets the distortion"*. ⇒ **That second clause is now withdrawn.** `n_eff` is a
**diagnostic that ranks how dependent a field is**, not an estimate of how wrong `PERMUTE` will be.
⚠️ Consistent with the locator audit's warnings, which is why they were printed beside the output:
`Dutilleul1993` C-4688 (the incomplete form can differ 8× from the complete one) and `Clifford1989`'s
own 2-of-12 misses (C-4683). ⇒ The **complete** estimator is now worth implementing; this was the
incomplete one.

### 8.4 ⚠️ What §8 does NOT say

- ⛔ **Nothing about the model.** No `mu`, no checkpoint. Independence here is **by construction**, so
  this shows the scheme is correctly sized on a null it was *built* to be correctly sized on.
- ⚠️ **`B = 49`, 30 reps ⇒ Monte-Carlo SE ≈ 0.040.** So *"recovers size"* means **not detectably
  over-rejecting at this resolution** — `0.100` and `0.067` are each within ~1.5 SE of `α`. A
  confirmation run at larger `B` is owed before the scheme is relied on.
- ⚠️ **A faithful GEOMETRY is not a simulation of conflict.** Real `mu` is smooth for reasons a
  Gaussian kernel does not reproduce, and real events are not a thresholded smooth field.
- ⛔ **Still owed:** the maxT band, the bootstrap (`H-c`), `opponent`'s replicate-independence check
  (adjacent shifts of a smooth field are near-identical, so the *effective* replicate count may be
  below the nominal one), and the **complete** Clifford estimator.
