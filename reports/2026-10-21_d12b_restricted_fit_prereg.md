# D12b — pre-registration: recover the nugget, which D12's F3 made inadmissible

**2026-10-21.** Written **before** anything is computed. Simon asked for it after D12's outcome.
⛔ **This file writes the plan only. Nothing is run.**

Reads `2026-10-12_underfire_diagnosis/d12_result.npz` (56 KB — the γ(d) curves D12 already computed
and committed, `48a8e0c`). **No new sweep, no forward pass, no pod, nothing fetched.** Seconds of CPU.

---

## 0. ⛔ THE HONESTY PROBLEM, STATED FIRST, BECAUSE IT WEAKENS EVERYTHING BELOW

**I have already seen the data this plan is written against.** `d12b_curve_readout.out` prints
`γ(d)` and `ρ(d)` at d = 1, 2, 4, 8, 16, 32 for all eight field-direction pairs. A pre-registration
written after seeing four of the eight points a fit will use is **not a test of a prediction** — it is
an **estimation plan with a pre-committed procedure**, and it must not later be quoted as a
prediction that survived.

⇒ **What is genuinely blind, and is therefore the whole value of D12b:** the fitted **nugget** is not
a quantity I can read off a printed curve, and the number it is compared against was measured by a
**different instrument** (D6-C's simulation through the truth's own estimator) **before** D12 existed.
That one comparison is a real test. Everything else here is description.

⚠️ **Two further weaknesses, named now rather than discovered later:**
1. The procedure below was chosen *because* D12's failed. Choosing an estimator after seeing one fail
   is a selection effect on the estimator, which is C-351's estimator slot pointed at myself.
2. ⛔ **Restricting the range does NOT give a plateau.** At d = 8 the truth's `ρ` is still 0.32 and the
   model's 0.57 — the curves are nowhere near flat. So a restricted fit does not fix F3; it **avoids
   needing the sill at all** (§2). Anyone reading D12b as "the sill comparison, now valid" has it
   backwards.

---

## 1. ⛔ A DEFECT IN D12's OWN F4, FOUND WHILE DERIVING THIS PLAN

D12's §4 committed this check: *"`n/(n+s)` for the truth should reproduce 0.154 / 0.081"* — D6-C's
measured estimator attenuation. **That comparison is of the wrong form, and the error is mine.**

D6-C measured lag-1 autocorrelation **dropping 0.8594 → 0.7053** (E–W) when the model's field is
pushed through the truth's 336-month count estimator. The variance decomposition makes attenuation
**multiplicative**, not additive:

> `ρ_Y(1) = ρ_m(1) · Var(m)/(Var(m)+σ²)`

so the noise share is `σ²/(Var(m)+σ²) = 1 − ρ_Y(1)/ρ_m(1)`, **not** `ρ_m(1) − ρ_Y(1)`.

| direction | ρ clean | ρ through the estimator | **multiplicative** noise share | D12's F4 used |
|---|---|---|---|---|
| E–W | 0.8594 | 0.7053 | **0.1793** | 0.154 |
| N–S | 0.7636 | 0.6824 | **0.1063** | 0.081 |

⇒ **D12b's target numbers are 0.1793 and 0.1063.** D12's F4 reported "⛔ FIRES, disagrees" against
0.154/0.081; the fit it tested was void anyway, so nothing downstream moves — but the check as written
would have been wrong even on an admissible fit. Logged as `W-46`.

---

## 2. THE INSTRUMENT: the nugget normalised by the field's OWN VARIANCE, never by a fitted sill

For a field `Z = m + e` with white `e`, the variogram's nugget **is** `σ²` and the field's total
variance **is** `Var(m) + σ²`. So

> **`n / var(Z)` is the white-noise share — and `var(Z)` is measured, not fitted.**

⇒ The quantity D12 needed never required a sill. D12 asked for `n/(n+s)` with `s` fitted, and the
unreachable asymptote took the whole fit down with it. ⚪ **One substitution removes the dependency**,
which is why D12b is a 40-line readout and not a new sweep.

### What is fitted, and over what

| slot | value |
|---|---|
| **population** | the same pairs D12 formed: all 64,818 land cells, no admission filter, zeros kept as data. In sample 421–456 and out of sample 457–469, reported separately, never pooled |
| **transform** | unchanged from D12 and §9.3 — `log(mu + ε)`, `logit(gate)`, `log(truth mean + ε)`, `ε = 1e-6` |
| **estimator** | least squares on the **already-computed** `γ̂(d)`, over `d = 1 … D_max^mono` only, where `D_max^mono` is the largest lag below the first non-monotone bin **in that direction and window** — a rule fixed here, read from the curve, not chosen per field |

**`D_max^mono`, read from `d12b_curve_readout.out` §2.** ⚠️ That readout prints the lag `d` at which
`γ(d+1) < γ(d)` **first** happens, so the printed lag is itself the last monotone one and is
admissible: E–W in sample **9**, N–S in sample ⚠️ **1**, E–W out of sample **19**, N–S out of sample
**14**, isotropic in sample **19**, out of sample **40**. (An earlier sentence of mine said "d ≤ 8";
the rule is the monotone range, and for in-sample E–W that is 9 lags, not 8.)

⛔ **The in-sample N–S cell is therefore DEAD ON ARRIVAL and is declared so now** — its first
non-monotone bin is d = 1 (the body's `ρ(1) = 0.761 < ρ(2) = 0.824` anomaly), leaving no usable lags.
⇒ **In-sample N–S is VOID before the run, not after.** The isotropic direction is reported alongside
E–W because it has the longest monotone reach, at the cost of mixing the two axes.

### Two nested models, because "is there a nugget" is a question about shape near the origin

| model | form | parameters |
|---|---|---|
| **M0** | `s·(1 − exp(−d/ξ))` | 2 — no nugget |
| **M1** | `n + s·(1 − exp(−d/ξ))` | 3 |

A nugget is identified by **curvature between the first few lags**, not by the asymptote: with no
nugget, `γ(1)/γ(2)` is a fixed function of `ξ` alone. ⇒ The nugget question becomes a nested model
comparison on 1 ≤ d ≤ `D_max^mono`, which is answerable without reaching a sill.

---

## 3. PRE-COMMITTED PREDICTIONS

⚠️ Only P-1 and P-2 are genuinely blind (§0). P-3 to P-5 are descriptions of data already seen and
are recorded so that a reader can tell which is which.

| # | quantity | prediction | basis | blind? |
|---|---|---|---|---|
| **P-1** | truth 336-mo, E–W: `n/var` | **0.179**, within 2 fit SE | §1's corrected D6-C attenuation, measured by simulation | ⭐ **YES** |
| **P-2** | truth 336-mo, isotropic: `n/var` | **between 0.106 and 0.179** (the two axes bracket it) | the E–W and N–S attenuations | ⭐ **YES** |
| **P-3** | model `log mu`, E–W: `n/var` | **≈ 0**, not distinguishable from 0 at 2 SE | `mu` is analytic in the emitted parameters; no count noise enters it | partly — D12's void fit said 0.273, which I have seen |
| **P-4** | M1 beats M0 for the truth, and does **not** for the model | the same asymmetry, stated as a model comparison | P-1 and P-3 together | no |
| **P-5** | fitted `ξ` ordering: model > truth in every admissible cell | D12's model-free ranges, 1.9–5.0× at every swept threshold | no — already measured |

⭐ **P-1 is the one that matters.** Two instruments — a 300-replicate simulation through the truth's
estimator, and a least-squares fit to a structure function — computed on different principles, two
days apart, with the second's target derived before it ran. Agreement would corroborate both. ⛔ And
this is the check D12 was supposed to deliver and could not.

---

## 4. ⛔ FALSIFIERS, committed now

| ID | fires if | consequence |
|---|---|---|
| **F1** | fitted `ξ` exceeds `D_max^mono` in a cell | the **range** is unidentified there; report it as a bound, never as a value — D12's F5 repeated at the restricted scale |
| **F2** | the **model's** `n/var` is distinguishable from 0 at 2 SE | ⛔ **either** the model's field carries genuine cell-scale noise — which would make the in-sample N–S inversion a mechanism rather than an anomaly — **or** M1 is still misspecified. Both are live; the truth's nugget may NOT be quoted until which one is settled |
| **F3** | truth `n/var` differs from §1's target by more than 2 fit SE | one of the two instruments is wrong. ⛔ Report both numbers, quote neither alone, and do not fix it by re-choosing the fit range |
| **F4** | M1 and M0 are statistically indistinguishable for **both** fields | the instrument has **no power** at integer lags on this grid ⇒ the nugget question is closed as unanswerable this way, and no further variogram work is proposed |
| **F5** | the fitted `n` is **negative**, or any fit SE exceeds its own parameter | that cell is VOID for lack of identification |

## 5. ⛔ VOID conditions

1. **In-sample N–S is already VOID** (§2) — one monotone lag. Declared before the run.
2. Fewer than **4** usable lags in a cell ⇒ that cell is VOID; a three-parameter fit on three points is not an estimate.
3. Non-convergence, or a covariance matrix the optimiser cannot return.
4. ⚠️ If **every** admissible cell turns out to be out-of-sample only, D12b's result is **out-of-sample
   only** and may not be quoted against §9.3's or D6-C's in-sample numbers. That would be a population
   mismatch of exactly the kind C-351 registers, and it is a real possibility here, since the two
   longest monotone reaches are both out of sample.

## 6. Decision rules

| outcome | what it means | what happens next |
|---|---|---|
| P-1 holds (F3 silent) | the truth's cell-scale noise share is **independently confirmed at ~0.18** by two instruments | the local half of the finding gets a second leg; D6-C's 0.06 residual stands on more than one measurement |
| F3 fires | a published number is wrong — either D6-C's attenuation or this fit | ⛔ stop; resolve before either is used again. **Do not** re-choose the fit range to make them agree |
| F2 fires | the model field has cell-scale structure it should not have | ⚪ that is a **bigger** result than the nugget — it would point at upsampling or patch stitching, and it has a cheap direct test (inspect a single emitted field's row/column differences) |
| F4 fires | the method is exhausted | ⛔ no more variogram work. The scale profile from D12 §3 is what we have, and it is enough for the loss arm |

## 7. Cost

⛔ **Measured, not estimated**, printed by the script: this reads a 56 KB `.npz` and runs ~12 least-squares
fits of ≤ 3 parameters on ≤ 40 points. ⚠️ Expected to be **under a second**; if it is not, something is
wrong and the number gets reported rather than explained away.

## 8. What D12b cannot do

- ⛔ It **cannot** rescue the sill, and does not try (§0.2). The sill stays VOID from D12's F3.
- ⛔ It **cannot** locate the defect in a component. `Cox1955` C-3611 forecloses that for every
  second-order statistic, and **C-353** now registers the class. A nugget that agrees with D6-C
  measures the **truth estimator's** noise, not the model's emission.
- ⛔ It **cannot** be read as a prediction that survived, except for P-1 and P-2 (§0).
- ⛔ It **does not** touch FAO-02, any config, or any trained artifact. Nothing here moves S1 or the
  loss-arm design.

---

# ⛔ OUTCOME — run 2026-10-21. P-1 and P-2, the only blind predictions, BOTH FAILED. The stop rule is in force.

`d12b_restricted_fit.py`, committed `3738dcf` **before** it ran. Cost measured: **0.10 s / 0.09 GiB**
— under the predicted second, as §7 required it to be reported either way. ⚠️ I first wrote "0.60 s /
0.19 GiB" into this section from memory and corrected it against the printed output; the rule is to
read the artifact, and for one sentence I did not.

## 1. ✅ The design worked — the nugget IS identified without a sill

Every admissible cell gives a truth nugget **30–45 standard errors from zero**, with M1 beating M0 at
`p < 1e-7`, on a curve that never plateaus. ⇒ **Dropping `n/(n+s)` for `n/var` did exactly what §2
claimed it would**, and that is the one thing here that went right (`R-21`).

## 2. ⛔ P-1 AND P-2 FAILED IN ALL FOUR ADMISSIBLE CELLS

| cell | truth `n/var` | ±2 SE | target (corrected D6-C) | |
|---|---|---|---|---|
| in sample, E–W | **0.2760** | 0.0127 | 0.1793 | ⛔ |
| in sample, isotropic | **0.3308** | 0.0166 | [0.1063, 0.1793] | ⛔ |
| out of sample, E–W | **0.2938** | 0.0156 | 0.1793 | ⛔ |
| out of sample, isotropic | **0.3736** | 0.0167 | [0.1063, 0.1793] | ⛔ |

⇒ **The variogram says the truth's white-noise share is 1.5–2.6× larger than D6-C's attenuation
implies.** The disagreement is 6–12 SE wide, so it does not go away under the SE caveat in §4 — and
that caveat runs the wrong way for a rescue, since the real uncertainty is *larger* than the printed
SE and F3 was already firing at the printed one.

**F3's pre-registered consequence, in force:** *"one of the two instruments is wrong. ⛔ Report both
numbers, quote neither alone, and do not fix it by re-choosing the fit range."* ⇒ **Neither 0.179 nor
0.28–0.37 is quotable**, and D6-C's **0.154 → 0.06 residual** now rests on a measurement another
instrument contradicts. Logged `W-47`.

## 3. ⛔ F2 FIRED TOO, AND IT GATES EVERYTHING ABOVE

The **model's** `log mu` nugget is **not** zero in three of four cells — `n/var` 0.0161–0.1305, at
**2.4–12.4 SE** from zero (P-3 ⛔, and P-4's predicted asymmetry fails with it in 3/4 cells).

⚠️ **And the split between heads is the informative part, which no prediction anticipated:**

| field | nugget | M1 vs M0 |
|---|---|---|
| model **gate logit** | **exactly 0**, at the bound, in **every** cell | `F = 0`, `p = 1` — M1 ≡ M0 |
| model **`log mu`** | 0.016–0.131 of variance | `p = 3e-9` … 0.046 |

Both fields come from the same forward pass. ⇒ **Whatever puts cell-scale noise in the body's field
does not touch the gate's.**

⭐ **The leading explanation is our own instrument, not the model.** `d4_result.npz` holds **4**
posterior draws under locked dropout and `mu` is their mean (`d4_mu_on_the_grid.py:107`). Residual
MC-dropout noise after averaging only four draws is **cell-independent by construction — i.e. exactly
a nugget** — and it enters `log mu` multiplicatively while the gate's sigmoid compresses it. ⛔ This
is a hypothesis, not a finding: it has a **direct test on data already on disk** (the per-cell
across-draw variance of `MU`, whose share of total variance should match the fitted nugget), and it
has **not been run**.

⇒ Until that is settled, F2's own rule holds: **the truth's nugget may not be quoted**, and §2's
disagreement in F3 may itself be an artifact of comparing a 4-draw field against a 300-replicate
simulation.

## 4. ⛔ A DEFECT IN MY OWN F5

F5 fires on *"a fit SE exceeds its own parameter"*, implemented as `SE/max(n, 1e-12)`. When the
nugget pins at its lower bound of exactly 0 — which is the **correct** answer for the gate — that
ratio is ~2e10 and F5 fires trivially, reporting VOID for the cleanest result in the table. ⇒ The
gate's eight "F5 FIRES" lines are **a falsifier defect, not a finding**; the substantive content is
`n = 0` at the boundary with M1 ≡ M0. Logged `W-49`. **F1's firings are genuine** and
pre-registered: the model's fitted `ξ` exceeds the fit reach, so those ranges are bounds, not values.

## 5. Where this leaves the programme

| | |
|---|---|
| ✅ **established** | the truth's field has a large, unambiguously identified cell-scale noise component; the `n/var` design works without a sill |
| ⛔ **in dispute, by two of our own instruments** | its *size* — 0.28–0.37 here against 0.179 from D6-C |
| ⛔ **blocked** | quoting either, per F3; and the 0.06 residual's single-instrument standing is now a known weakness rather than an assumption |
| ⚪ **next, cheap, and NOT run** | the across-draw variance of `MU` — resolves F2 and tells us whether §2's disagreement is a four-draw artifact. It needs no fit and no new sweep |
| ⛔ **unchanged** | D12's scale profile (§3 of the D12 outcome). It is a ratio of two `γ` curves at matched `d` and uses **no fitted parameter**, so nothing here touches it |

---

# ⛔ D12c ADDENDUM — run 2026-10-21. My explanation for F2 is HALF right, which kills the route

`d12c_draw_noise.py`, committed `4fe27da` before it ran. Cost **0.77 s / 0.17 GiB**. ⛔ Post-hoc,
and narrower than F2: it tests only whether the model's nugget is our own 4-draw MC-dropout noise.

⚠️ **Two provenance facts, because both cut against this section's authority.** (1) The `(1 − ρ_D)`
white-share line in §1 was **added after the first run** and the script re-run, so that one number
is post-hoc even within D12c; the split-half total and the lag-1 reading are from the committed
version. (2) I first wrote this commit's hash as `1e5cbb5` from memory — it is `4fe27da`, and that
is the second artifact detail I stated without reading it in this hour (the other was D12b's cost).
⛔ Same reflex, two instances, both caught only because something else made me open the file.

**Split-half:** build the same field twice from disjoint draw pairs. For iid draws
`Var(F₁₂ − F₃₄) = s²` while the 4-draw field carries `s²/4`, so one subtraction measures the MC noise
of the field D12 actually fitted, in the same transform, on the same cells.

## 1. ⭐ The magnitude matches. The structure does not.

| field | white part of MC noise | D12b's fitted nugget |
|---|---|---|
| body `log mu`, in sample | **0.0710** | **0.0789** |
| body `log mu`, out of sample | 0.0320 | 0.0161 |
| gate logit, in sample | 0.0128 | **0.0000** |
| gate logit, out of sample | 0.0164 | **0.0000** |

⇒ **In sample the body's entire fitted nugget is accounted for by our own four draws, to within 10%.**

⛔ **But the difference field is NOT WHITE** — its lag-1 autocorrelation is **+0.3654 / +0.3630**
(body, in sample) and **+0.4576 / +0.4088** (gate), in all four cases, both directions. So
MC-dropout noise is **spatially correlated at one cell**, which is what should have been expected:
dropout removes *channels*, and convolutions spread the perturbation across cells. ⇒ It is **not a
pure nugget**, and the magnitude agreement above rests on a **crude** `(1 − ρ_D)` split that the
script labels as such.

⇒ **F2 is not resolved.** The noise is real, of the right size, and in the wrong shape. Per F2's own
rule the truth's nugget **remains unquotable**. Logged `W-50`.

## 2. ⛔ And the gate exposes a detection floor, which closes the method

The gate's field provably contains **1.3–1.6%** white MC noise, and the fit returned **exactly 0** in
every cell with `M1 ≡ M0` (`F = 0`, `p = 1`). ⇒ **A nugget of a few percent of variance is below this
instrument's detection floor on this grid**, so "the gate has no cell-scale noise" was a **false
negative** — and D12b's cleanest-looking result is the one that is wrong.

⇒ **That is F4's condition in substance:** the nested comparison has no power at the scale of the
effect. **F4's pre-registered consequence applies — no further variogram work.** The nugget/sill
decomposition cannot be done with this instrument on this grid, and that is now measured rather than
suspected.

## 3. ⚪ One by-product worth more than the nugget, and NOT claimed as a finding

**Our MC-dropout posterior spread is spatially correlated noise, not independent per-cell
uncertainty** — ρ ≈ 0.27–0.46 between neighbouring cells, measured in four independent cells.
⛔ Not pursued here: it bears on how posterior spread is reported in the product, which is a different
question from over-smoothing, and it has had exactly one measurement.

✅ **Both halves of it are now tracked where they belong, not in this addendum** (2026-10-21):
**C-128** (locked-dropout posterior calibration unvalidated) carries the magnitude and spatial table
plus ⭐ **the cheap cost test nobody has run** — score a dropout-OFF run against the current one at
**matched total sample count** on the frozen ruler, since every MC draw costs a full forward pass and
head draws are free. **C-209** (epistemic and aleatoric folded onto one flat `S`) carries the
consequence that the two components differ in **spatial structure**, not only in size, so no stored
frame tells a consumer which part of its spread is shareable across neighbouring cells — which is
**C-110**'s aggregation concern with a measured number attached for the first time. ⛔ The dropout-off
test is **not proposed as a decision**: it changes what inference delivers, so it is the owner's.

## 4. Where the D12 programme actually lands

| | |
|---|---|
| ⭐ **durable** | **D12's scale profile** — the deficit is monotone in scale, worst at one cell (body 0.391, gate 0.112 of their due share), nearly closed by 32 cells. ⛔ It is a ratio of two γ curves at matched `d` and uses **no fitted parameter**, so none of D12b's or D12c's failures touch it |
| ⭐ **durable** | the correlation-length excess is **real in sign everywhere** and **1.9–7.0× in size**, not the 12.8× an exponential extrapolation gave |
| ⛔ **closed as unanswerable** | the nugget/sill decomposition, by F4's condition — detection floor above the effect, and our own instrument's MC noise sitting at the same magnitude |
| ⛔ **still in dispute, unresolved** | the truth's noise share: 0.28–0.37 here against D6-C's 0.179. ⚠️ And D12c records an untested **candidate design error** in that comparison — D6-C pushed the *model's* field through the truth's estimator, so its attenuation depends on the *model's* rate field while the nugget depends on the *truth's*, and firing differs 4.18× between them. They may never have been the same quantity |
| ⛔ **foreclosed, not pending** | the component attribution (`Cox1955` C-3611, registered as C-353) |
