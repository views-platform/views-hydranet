# (a) The math — what quantity we want, exactly

**Date:** 2026-11-08 · **Status:** step (a) of three · ⛔ **REVISED 2026-11-08 after two independent
audits. See §0 for what changed.** ⛔ **No design decision is taken in this file.** (b) pseudocode and
(c) a synthetic simulation with a planted answer come next, and only then is any choice here final.

⛔ **Why this file exists.** Simon, 2026-11-08: nothing is decided until there is *rigorous math,
pseudocode, and a simulation proving it works the way I think it does for the reasons I think it
does.* Measured on this codebase: on **"what"** questions I score ~**6/10** from memory alone; on
**"why"** questions **0/10**. ⇒ Every mechanism claim below is either **derived here** or **labelled
as a hypothesis for (c)**.

---

## 0 ⛔ THE AUDIT, AND WHAT IT BROKE

Two seats, run in parallel on deliberately different jobs so neither could ratify the other.

| seat | job | result |
|---|---|---|
| `library` | locator audit — does each paper say this, at this location | ⭐ **11 of 11 CONFIRMED**, page numbers given; one scope note (§3.2) |
| `opponent` | adversarial maths — are the derivations valid | ⛔ **two BREAKS**, one sub-claim broken, one statistic shown near-vacuous, plus six additions |

⭐ **Both breaks were re-derived here before being accepted.** Neither is taken on the auditor's word.

| # | what was wrong in v1 of this file | where |
|---|---|---|
| ⛔⛔ **B1** | **§4.1 misread the `Henzi2021` quote it reproduced.** I claimed a distinct-valued scalar index gives `R* = 1` "from nothing". The quote's first extreme requires *"a perfect monotonic relationship to the response"* — a perfect score from **perfect information**. Its second extreme ("no order relations at all") is an **antichain**, which needs a *partial* order and is **unreachable under the total order on R**. ⇒ The null bias is **`O(log n / n)`**, not maximal — I overstated it by ~3 orders of magnitude, in the direction that makes my own instrument look useless | §4.1 |
| ⛔⛔ **B2** | **§5 asserted exactness from the wrong premises.** Preserving both marginals, `n` and `d` is bookkeeping, not exactness. Exactness needs the **randomisation hypothesis**, which needs exchangeability — and our cells are a **spatial panel**. ⇒ The test is **anti-conservative**, in the one direction that waves `v1` through on a null index. ⛔ Worse: (c) as I specified it would have planted *exchangeable* data and returned a clean bill, **planting the assumption under test** | §5 |
| ⛔ **B3** | **§3.3's gloss was a false biconditional.** `DSC_ORD = 0` does **not** mean "no ranking information": `x = −y` gives `DSC_ORD = 0` with perfect information. The statistic is one-sided | §3.3 |
| ⛔ **B4** | **§4.3's "bias is monotone in `d`" is false** — `d` is not a sufficient statistic for fitting freedom. Replaced by a stronger statement on the refinement chain | §4.3 |
| ⚠️ **B5** | **§4.4's `DSC_MAX` is near-vacuous in the regime we expect.** As `d → n` it → `UNC_0`, so for a continuous `mu` on 2,905 cells it adds nothing. Replaced by a coarsening sweep that is strictly more informative | §4.4 |
| ⚠️ **B6** | **§3.2 mis-attributed `R*`.** `Gneiting2023`'s general form is `(DSC − MCB)/UNC` and **can go negative**; `DSC/UNC` is the special case where `MCB` is absent | §3.2 |

⭐ **And the audit's largest contribution was not a break but an omission: the decision is a
generalisation question and every statistic in v1 of this file was in-sample.** See §6.

---

## 1 Setup and notation

Cells `i = 1, …, n`, each carrying truth `y_i ∈ Z≥0` and a **candidate ordering index** `x_i ∈ R`.
Optionally a predictive distribution `F_i`.

⭐ **`x` is the object under test, not `F`.** `v1` keeps the model's ordering and replaces its
magnitudes, so the question is about an ordering, and an ordering is a scalar per cell.

**Populations.** Three, never combinable (`C-351`):

| | definition | selected on | licenses |
|---|---|---|---|
| `P_all` | all 842,634 cell-months | nothing | occurrence + magnitude |
| `P_open` | `(draws > 0).any()` | the **prediction** | calibration (`Gneiting2011`) |
| ⭐ `P_event` | `y_i > 0` — 2,905 event cells | the **outcome** | ⛔ **discrimination only** |

⇒ Everything here is on `P_event`. ⛔ Outcome-selected ⇒ **no calibration claim** may be read off any
number defined in this file.

✅ **Checked, not assumed** (`opponent` raised it; `e1_data.py:13`): at h1 *"the origins are months
457..469, **one month per origin**"* ⇒ `(cell, target month)` is **unique**, so there are no duplicate
or near-duplicate rows inflating `n`. ⚠️ **This is an h1 property only** — at h36 several origins reach
the same target month and the duplicate hazard is real.

---

## 2 TWO ORDER STRUCTURES, ONE STATISTIC

### 2.1 `DSC_ISO` — Arnold's form

`Arnold2024` §3.1 eq. (24)–(25) decomposes **the score of a forecast we already have**, with IDR
covariates `F_1, …, F_n` *themselves* under the **stochastic order** `≤_st`:

> *"we apply IDR to the data (F_1, y_1), …, (F_n, y_n) … where the covariates are F_1, …, F_n and the
> partial order on the covariate space is the stochastic order"* — `library`: **CONFIRMED**, p. 5005

### 2.2 `DSC_ORD(x)` — our form

IDR of `y` on the **scalar** `x` under the **total order `≤` on R**. Plain `Henzi2021` IDR.

⛔ **CORRECTED (B-audit, item 1a).** v1 of this file said these *"are not the same number and neither
is an estimate of the other"*. **Too strong.** If `F_i = F(· ; x_i)` is any family **stochastically
increasing** in `x`, then `F_i ≤_st F_j ⟺ x_i ≤ x_j`, the induced order **is** the total order on `x`,
and the two IDR problems are *literally identical*. ⇒ It is **one statistic under two order
structures**, not two statistics.

⭐ That reframing is not cosmetic — it is what **licenses** §3.3's use of `Arnold2024` Prop. 3.1, which
this file already depends on.

⚠️ **And it imposes a condition §4.2 was treating as settled.** `DSC_ISO(F) ≥ DSC_ORD(mu)` requires
the orders be **nested** — `mu` must be an order-preserving functional of `F`. If `mu` is the
regression head's mean while the delivered draws pass through gating and composition, `F_i ≤_st F_j`
need **not** give `mu_i ≤ mu_j`, the orders are **not nested**, and **neither `DSC` dominates**. ⇒ One
line of code settles it. **Until it is run, no comparison between the two is licensed.**

### 2.3 ⭐⭐ Why `DSC_ORD(x)` is a genuine ceiling for `v1` — by construction, not analogy

This is the strongest result the audit added, and it is what makes the measurement decision-relevant.

`v1` emits, per cell, the distribution of the `k`-th order statistic of draws from a fixed pool, where
`k` is the model's rank. The `k`-th order statistic of `n` i.i.d. draws is **stochastically increasing
in `k`**. ⇒ `v1`'s family is stochastically monotone in `x` ⇒ it is **feasible** for the IDR
minimisation ⇒

```
            CRPS(v1)  ≥  CRPS_ORD(x)        hence        DSC_ORD(x)  ≥  DSC(v1)
```

⭐ **`DSC_ORD(x)` is the best any rank-coupled emission built on `x` can possibly do.** A low value is
therefore a *hard* stop on `v1`, not a suggestive one. ∎

---

## 3 The estimator

### 3.1 IDR on a totally ordered covariate

`Henzi2021` Thm 2.1 gives existence and uniqueness (`library`: CONFIRMED, p. 8). For a total order
with `x_1 < … < x_n`, eq. (10) gives it in closed form, credited to Barlow et al. (1972) (CONFIRMED,
p. 8):

```
            F̂_i(z) =  min      max     (1/(j−k+1)) · Σ_{l=k..j} 1{y_l ≤ z}
                     k=1..i   j=k..n
```

and §2.3 eq. (15) gives the cheaper equivalent — **one isotonic regression per threshold** (CONFIRMED,
p. 10):

```
            F̂(z) = argmin  Σ_i ( η_i − 1{y_i ≤ z} )²     over η ∈ [0,1] non-increasing in x
```

Two facts that make it computable, both verbatim in §2.3:

- ⭐ The objective is **constant between the distinct values of `y`**, so only the `m` distinct truths
  need thresholds; quantile-level approaches are *"computationally prohibitive"*.
- ⭐ For a total order, **IDR ≡ PAV** (§2.2, citing Barlow Thm 1.12) ⇒ `sklearn.isotonic` is an
  admissible implementation at each threshold.

### 3.2 The components

With `Ĝ` the **empirical CDF of `y_1, …, y_n`** (the marginal — the no-model forecast):

```
  CRPS_ORD(x) = (1/n) Σ_i crps( F̂_i , y_i )
  UNC_0       = (1/n) Σ_i crps( Ĝ   , y_i )
  DSC_ORD(x)  = UNC_0 − CRPS_ORD(x)                  ⭐ THE QUANTITY
  R*(x)       = DSC_ORD(x) / UNC_0 ∈ [0, 1]
```

⚠️ **CORRECTED (`library`, scope note).** `Gneiting2023`'s universal coefficient is
**`R* = (DSC − MCB)/UNC`** (§3.4 eq. (32); `R* ∈ [0,1]` is Thm 3.6, p. 3256) and **can go negative out
of sample**. `DSC/UNC` is the **special case where `MCB` is absent by construction**, which is ours.
⛔ v1 of this file implied `DSC/UNC` was his definition. It is not.

### 3.3 ⭐ `DSC_ORD` is exactly the threshold-integral of CORP discrimination

Since `CRPS = ∫ BS(z) dz` (`C-55`), eq. (15)'s threshold-`z` fit **is** the PAV/CORP recalibration of
`1{y ≤ z}`, and eq. (15)'s minimisations are **separable across `z`**:

```
            DSC_ORD(x)  =  ∫ [ BS(Ĝ at z) − BS(PAV fit at z) ] dz  =  ∫ DSC_CORP(z) dz     exactly
```

⭐⭐ **Payoff: one fit yields a `DSC` CURVE over magnitude thresholds, free.** And that — not the
scalar — is what answers *"does the ordering carry **magnitude** information"* rather than *"any
information"*. An index that separates 1 from 2 and is noise above 10 has its curve **piled at low
`z`**, visibly. ⇒ **The curve is the headline; `R*` is its summary.** Given that `UNC_0` on a 1–2,482
outcome is dominated by a handful of events (§7.2), aggregating first would hide exactly the failure
we care about.

### 3.4 The endpoints, and ⛔ the one that was glossed wrongly

`Arnold2024` Prop. 3.1 (`library`: CONFIRMED, both iff directions, p. 5005):

- `DSC = 0` ⟺ `F̂_i = Ĝ` for all `i` — the recalibrated forecast collapsed into the no-model forecast.
- `DSC ≥ 0` is a **theorem**: `Ĝ` repeated `n` times is admissible, so the minimum cannot exceed it.
  The paper's own proof says so (*"A further admissible n-tuple … is the constant n-tuple with
  entries `F̂_mg`, whence `DSC_ISO ≥ 0`"*). ⇒ **A gate, not a result.**

⛔⛔ **B3 — CORRECTED.** v1 of this file glossed the first bullet as *"the index carries no ranking
information. Not weak — none."* **That is false.** Counterexample: `x = −y` exactly. IDR constrains `y`
to be *increasing* in `x`, so the fit is flat, `F̂_i = Ĝ`, `DSC_ORD = 0` — while `x` determines `y`
**perfectly**. The paper's `iff` is about the *recalibrated forecast*; my gloss added a claim about
*information* that does not follow.

⇒ **`DSC_ORD = 0` means no isotone-INCREASING information.** ⛔ Mitigation, and it is a free parameter
until written down: **the sign of `x` is pre-declared**, and `DSC_ORD(x)` and `DSC_ORD(−x)` are both
computed. This removes a whole class of false negatives on indices whose natural sign is inverted (an
inverse intensity, a distance, a non-occurrence logit).

### 3.5 `MCB` is deferred, not undefined

v1 said `MCB` is undefined here. ⚠️ **Better: not defined *yet*.** The moment the rank→magnitude map is
named, §2.3's monotonicity makes `v1`'s IDR recalibration *be* `F̂`, so `Arnold2024`'s decomposition
applies verbatim:

```
            MCB_v1 := CRPS(v1) − CRPS_ORD(x)  ≥ 0            and
            CRPS(v1) = UNC_0 − DSC_ORD(x) + MCB_v1
```

⭐ `MCB_v1` is precisely *how much of the available ordering information our chosen emission gives
away* ⇒ **the same instrument that decides whether to build `v1` grades `v1` once built.**

---

## 4 The bias — and both of v1's claims about it were wrong

### 4.1 ⛔⛔ B1 — the misread quote, and the corrected magnitude

`Henzi2021` §3 opening (`library`: CONFIRMED verbatim, ellipsis honest, p. 15):

> *"In the extreme case of distinct, totally ordered covariate values `x_1, …, x_n ∈ X` **and a perfect
> monotonic relationship to the response values** `y_1, …, y_n`, the IDR distribution associated with
> `x_i` is simply the point measure in `y_i`. The same happens in the other extreme, when there are no
> order relations at all. Hence, the partial order serves to regularize the IDR solution."*

⛔ **v1 of this file conflated the two extremes and concluded `R* = 1` "from nothing".** Reading the
bolded clause: the first extreme is a perfect score from **perfect information**. And the second
extreme — *no order relations at all* — is an **antichain**, which requires a **partial** order; under
the total order on a scalar every pair is comparable, so **that extreme cannot arise in §2.2's own
setting.**

**What actually happens, derived.** By eq. (15), IDR at threshold `z` is the PAV regression of
`1{y ≤ z}` on `x`. PAV's blocks are the vertices of the greatest convex minorant of the partial-sum
path of the responses in `x`-order. Under `H_0` that path is an **exchangeable random walk**, and by
**Sparre Andersen** the vertex count of such a walk's convex minorant is **distribution-free**, with
the law of the cycle count of a uniform random permutation and mean the harmonic number
`H_n ≈ log n + γ`. ⇒ Under the null the fit pools into **`O(log n)` blocks, not `n`**. (Bernoulli
increments carry ties, and ties only *merge* vertices, so the count is bounded above by the continuous
case — the direction is safe.) Per threshold the in-sample explained Brier is `O(log n / n)` of that
threshold's `UNC`, and by §3.3 the bound carries through the integral:

```
            DSC_ORD  ≤  O(log n / n) · UNC_0
```

⇒ ⭐ **At `n = 2,905`, `log n / n ≈ 2.7 × 10⁻³`.** The statistic is close to **unbiased** at our sample
size — the same adaptivity as the isotonic LSE's `log n / n` risk at a flat truth. ⛔ v1's framing
(*"a bare figure is uninterpretable; its scale is set by the fitting freedom"*) **understated this
instrument by roughly its entire range**, and would have had a reader dismiss a real small `R*` as
"just the bias".

⚠️ **This is a SHARP PREDICTION FOR (c), not a measurement.** `opponent` states, and I agree, that the
constant is not derived and the tie correction is argued by direction only. ⇒ **(c) must measure the
null `R*`.** If it lands near 1, the fit is buggy. If it lands at `10⁻³`–`10⁻²`, the derivation held.
⭐ Either way (c) becomes a **mechanism** test, not merely a sizing test — which is what the rule
asks for.

⇒ The *conclusion* (report against a null) stands. Only the magnitude claim is withdrawn.

### 4.2 The inclusion lemma — HOLDS

If `⪯_1 ⊆ ⪯_2` as sets of pairs then `C_2 ⊆ C_1`, so `min_{C_1} ≤ min_{C_2}`: a **weaker** order gives
a **smaller** `CRPS_ORD` and a **larger** `DSC`. ∎ (`opponent`: HOLDS, both algebra and direction.)
⚠️ Its *application* to `F` vs `mu` needs §2.2's nesting check, which is not done.

### 4.3 ⛔ B4 — "bias monotone in `d`" is false; the refinement chain is stronger

`d` is **not** a sufficient statistic for fitting freedom: two indices with the same `d` but different
block patterns and orderings have different biases. v1 asserted monotonicity from two endpoints.

⭐ **What is true, from §4.2.** Let `x'` be a **coarsening** of `x` (merge adjacent blocks). Merging
turns one-directional pairs into mutual ones, so `⪯_x ⊆ ⪯_{x'}`, so `C_{x'} ⊆ C_x`, so

```
            DSC_ORD(x)  ≥  DSC_ORD(x')      for every realisation — PATHWISE
```

Pathwise domination ⇒ it holds for the null **expectation and for every quantile** of the null
distribution. And `DSC_MAX` is non-decreasing under refinement (§4.4). ⇒ *"bias and ceiling move in
opposite directions"* is a **theorem along a refinement chain** with stochastically ordered nulls —
far stronger than two endpoints. ⛔ It **cannot** be exported to unrelated indices with similar `d`.

⛔ **And "each against its own null" is necessary but NOT sufficient** — my own worry, confirmed.
Margins over different nulls have different variances and skew, and a p-value is not an effect size
(floored at `1/(B+1)`, so two good indices both saturate and become indistinguishable).

⇒ **For genuinely non-nested indices, use a PAIRED permutation:** permute `y` **once** per replicate
and evaluate **both** indices on the same permuted vector, giving the null of the **difference**
directly. Absorbs both biases and their correlation. ⭐ Same move as 2026-11-07's paired bootstrap,
and for the same reason. ⚠️ `M98`/`M99`'s `mean_draw` (1,321 distinct values) vs `gate` (≈64,732)
comparison has exactly this shape — **flagged, not re-opened**: the crossing is a different estimator
and whether the argument transfers is a separate question.

### 4.4 The ceiling — valid, and ⚠️ near-vacuous where we expect to be

Let `G_1, …, G_d` be `x`'s tie blocks. ⭐ `x_i = x_j` gives **both** order relations, so `F̂_i ≥ F̂_j`
*and* `F̂_j ≥ F̂_i` pointwise ⇒ `F̂` is **forced** constant on blocks, not merely allowed to be.

**The properness step, with the condition v1 omitted.** CRPS is **strictly** proper relative to the
class of probability measures with **finite first mean** (`Gneiting2007_StrictlyProperScoringRules`).
Properness quantifies over **all** `Q` in that class, not only over data-generating laws, and

```
            E_{Y ~ Ĝ_k}[ crps(P, Y) ]  =  (1/|G_k|) Σ_{i∈G_k} crps(P, y_i)      identically
```

so applying it with `Q = Ĝ_k` is legitimate — *"the true distribution" is a red herring*. Strictness
gives uniqueness of the minimiser. Relaxing the between-block constraint can only lower the objective:

```
            DSC_ORD(x)  ≤  DSC_MAX(x) := UNC_0 − (1/n) Σ_k Σ_{i∈G_k} crps( Ĝ_k , y_i )
```

At `d = 1`, `Ĝ_1 = Ĝ` and `DSC_MAX = 0` exactly, recovering §3.4's endpoint. ∎

⚠️⚠️ **B5 — but the regime defeats it.** As `d → n` blocks become singletons, `Ĝ_k = δ_{y_i}`,
`crps = 0`, so `DSC_MAX → UNC_0` and the ratio `DSC_ORD/DSC_MAX → R*`. A continuous `mu` on 2,905
event cells is **nearly all-distinct** ⇒ the ceiling **adds nothing in exactly the regime we expect**.
It informs only for `d ≪ n`.

⭐ **Replacement, strictly more informative: the coarsening sweep.** Compute `DSC_ORD` on `x`
coarsened to `d' = 2, 4, 8, …, d`. By §4.3 that curve is **monotone by construction** with
stochastically ordered nulls at each rung; its top end *is* the old bound; and its **shape** separates
*"no resolution"* from *"resolution, wrong ordering"* at **every** resolution instead of at one. Cost:
`log₂ d` fits of a one-pass object. ℹ️ Same sweep shape as `~/brain/4_notes/2026-11-06_resolution_not_ranking.md`
(verified to exist) — worth reading before naming it.

⚠️ Two notes if `DSC_MAX` is reported anyway: it is itself an in-sample fitted quantity, and it is
**not** permutation-invariant for `1 < d < n` (permutation preserves block **sizes**, not **contents**),
so its null expectation is not 0 and the ratio would need its own null. ⇒ Prefer reporting the
numerator's margin-over-null plus `d` and the block-size distribution.

---

## 5 ⛔⛔ B2 — THE NULL. The largest break, and it is in the dangerous direction

### 5.1 v1's grounds were the wrong grounds

v1 argued: permutation preserves `y`'s marginal, `x`'s tie structure, `n` and `d` ⇒ *"an exact
finite-sample test"*. ⛔ **The implication does not go through.** Those four are necessary bookkeeping.
Exactness requires the **randomisation hypothesis** — under `H_0`, `(x, y_π) =_d (x, y)` for every `π`
— which holds if `y ⟂ x` **and at least one of the two vectors is exchangeable**. `H_0: y ⟂ x` alone
is strictly weaker and does not suffice.

### 5.2 ⛔ And on our data the assumption fails, in the direction that waves `v1` through

**Counterexample in exactly our shape.** Let `x` and `y` be two **independent** spatially
autocorrelated fields on the same geometry. `H_0` holds exactly; neither vector is exchangeable. The
realised alignment of two smooth fields has **far larger variance** than the alignment of a smooth
field with a random relabelling, because a smooth field's effective sample size is far below `n`.
Permutation destroys `x`'s autocorrelation and therefore **understates the null variance**.

⇒ ⛔ **The test is ANTI-CONSERVATIVE. It rejects too often.**

⛔⛔ **REBUILT 2026-11-09.** v1 rested this on Clifford/Richardson/Hémon (1989) and Dutilleul (1993).
⛔ `library` reports the library holds **neither** — no PDF, no metadata. A finding that changed the
design cannot rest on a reference nobody here has opened. ⇒ **The claim is now derived from this
dossier's own lemmas**, in four parts with three epistemic statuses, in `02_PSEUDOCODE.md` §4.4:

| | | status |
|---|---|---|
| **(a)** validity | the randomisation hypothesis is **false** when neither vector is exchangeable | ⭐ **logic alone** — no direction, no citation, no empirical input |
| **(b)** direction | in the `b`-fold autocorrelated limit the permutation null is centred too low by **exactly the block size**: permutation preserves `x`'s ties, so both configurations keep `G` blocks of size `b` and only the within-block response variance changes (`p(1−p)` vs `p(1−p)/b`) ⇒ ratio `= n/G = b`. ⭐⭐ **Both the undetermined constant AND the harmonic number cancel**, and `b = n/n_eff` | ⭐ **DERIVED**, inheriting no constant and no asymptotic. ⚠️ `b` is an **upper bound** — the observed case carries more ties, which merge vertices |
| **(c)** mechanism | exactness fails only when **NEITHER** vector is exchangeable ⇒ smoothness in **one** field alone is harmless, and the distortion needs **JOINT** autocorrelation ⇒ a **three-point** prediction whose middle case is a **control** (`x` smooth, `y` i.i.d. ⇒ still **exact**; distortion there would be an implementation fault) | ⭐ derived; makes the test of the mechanism real |
| **(d)** magnitude | the effective `b` on our data is **unknown** | ⛔ **UNMEASURED** — (c)'s step 4 is the evidence |

⭐⭐ And (d) promotes **`n_eff = n/b`** to the parameter that *sets* the distortion ⇒ it is the **first
number (c) prints**, because it estimates how badly `PERMUTE` is wrong before a single replicate is
spent. ℹ️ The two papers are owed as **historical attribution** only, and stay unquoted until acquired.

We are squarely inside it: `mu` is a smooth spatial field by construction (shared features,
neighbourhood structure), and fatalities are strongly clustered in space and time. ⇒ The error runs
**toward building `v1` on a null index** — the worst available asymmetry.

### 5.3 ⛔⛔ And (c) as I specified it would have planted the assumption under test

v1's `H-e` read *"the permutation test is correctly sized at our `n`, ties and skew"* — **dependence is
not named**. A simulation planting `H_0` with **exchangeable** data would find the test exact, because
under exchangeability **it is**, and the clean bill would be **vacuous**.

⇒ ⭐⭐ **(c) must plant two independent fields with OUR spatial and temporal autocorrelation on OUR
cell geometry, and measure the FPR there.** That, not the exchangeable version, is the test of `H-e`.

⛔ This is the same defect class the repo keeps finding: a gate whose failure condition is *"my own
account is incomplete"* (`feedback_gate_against_your_own_explanation`). Had the pre-registration gone
out as written, it would have **locked the vacuous version in.**

### 5.4 The candidate fixes, cheapest first

| # | scheme | preserves | cost |
|---|---|---|---|
| ⭐ 1 | **cyclic / toroidal shift** — shift `x` as a whole against `y` (in space, or pair month `t`'s `x` with month `t+k`'s `y`) | `x`'s autocorrelation **exactly**; needs **no model** of the dependence | ⚠️ under a time shift the `y` multiset is **not** fixed (end months drop), so `UNC_0` **moves per replicate** and §5.5's invariance dies ⇒ recompute `UNC_0` inside the loop and say so |
| 2 | **restricted (stratified) permutation** within a **SPATIAL** stratum (country / region) | exact under *within-stratum* exchangeability — far more defensible | power |

⛔ **CORRECTED 2026-11-09.** v1 offered *"or within month"*. That addresses nothing: `B2` is about
**spatial** autocorrelation, and permuting within a month destroys the within-month spatial alignment
that **is** the problem. The strata must be spatial.
| 3 | report `n_eff` | — | tells the reader the test's real resolution |

⛔ **Whichever is chosen, the exchangeability unit IS the null** and belongs in §8's pre-declaration
list beside `α` and `B`.

### 5.5 One sub-claim that HOLDS, with a corollary

`UNC_0` is a function of the multiset of `y` alone, which any label permutation fixes ⇒ `R*` is
comparable across replicates. ⭐ **Corollary:** `R*` is then a strictly decreasing function of
`CRPS_ORD` across replicates, so the permutation test on `R*` and on `DSC_ORD` are **the same test with
the same p-value** — reporting both is interpretation, not extra inference. ⚠️ Dies under a shift
scheme (§5.4 note).

---

## 6 ⭐⭐ THE OMISSION THE AUDIT FOUND, AND IT OUTRANKS BOTH BREAKS

**All of §4 and §5 exist only because the statistic is in-sample.**

`Arnold2024`'s decomposition is *deliberately* in-sample — it attributes a **realised** score; it does
not estimate generalisation. ⛔ But our decision is go/no-go on **building** `v1`, which is a
**generalisation** question. ⇒ v1 of this file built an elaborate bias-and-null apparatus around a
quantity that answers an adjacent question.

⭐ **The direct instrument: a temporal split.** Fit IDR on months `≤ T`; evaluate CRPS on months `> T`
against the **training** marginal. Then:

- the number is **unbiased** — no `log n / n` argument needed;
- **no null is needed for the level** (beating climatology out of sample is the claim);
- it has the **same structure deployment will face**;
- and ⇒ `UNC_0`'s own in-sample optimism (`Ĝ` is the ECDF of the very `y` it scores, §7.3) **vanishes**.

⛔ **CORRECTED 2026-11-09 — v1 attributed this design to the paper and `library` found that wrong.**
What is `Henzi2021`'s (§5.4): the **temporal train/test split** and reporting out-of-sample CRPS. ⛔
What is **NOT** theirs: the climatology comparison. Their references are the **raw ensemble** and
competing postprocessing methods (BMA, EMOS, HCLR); the marginal appears as a reference **nowhere in
the paper**, and they report raw CRPS, not a skill score against a marginal. ⇒ **The temporal split is
their idiom; `SKILL = 1 − CRPS_oos/UNC_oos` against the training marginal is OURS.** Same error class
as `N10`: my own design attributed to a source.

⇒ **Both are reported, and they answer different questions:**

| | question | needs |
|---|---|---|
| in-sample `DSC_ORD` + null | **is there signal at all?** | §5's null, done right |
| ⭐ **out-of-sample IDR skill** | **how much can `v1` actually use?** | a temporal split |

⛔⛔ **The decision hangs on the second.** §2.2's claim to be *"the statistic the programme actually
asks for"* belongs to the out-of-sample form.

---

## 7 Three further hazards the audit added

### 7.1 Non-nested orders — see §2.2's ⚠️. One line of code, currently assumed.

### 7.2 ⚠️ `R*` is NOT invariant to the scale of `y`

`DSC_ORD` is invariant to monotone transforms of **`x`** (§2.2 — a feature, and §2.3 is why).
⛔ It is **not** invariant to monotone transforms of **`y`**: on `log1p(y)` the `UNC_0`, `CRPS_ORD` and
`R*` all differ. ⇒ *"Does the ordering carry magnitude information"* has a **scale-dependent answer**.

On the raw count scale with `y` up to 2,482, `UNC_0` and `DSC` are dominated by a handful of events —
an **effective leverage sample of a few cells**, whatever `n` says. Consequences:

- the null distribution is **skewed and heavy-tailed**, so a single `1−α` quantile at modest `B` is
  unstable in the tail ⇒ **report the whole permutation distribution**, not one quantile;
- the bootstrap CI will be **wide** — consistent with 2026-11-07's crossing bootstrap landing at the
  **optimistic edge** (`P(boot ≤ point)` = 0.190, 0.043);
- ⇒ ⭐ use §3.3's **threshold curve** so bulk and tail are **separable** rather than aggregated.

⛔ **The scale of `y` is pre-declared** (§8).

### 7.3 ⚠️ `UNC_0` is itself fitted in-sample

`Ĝ` is the ECDF of the same `y` it is scored against ⇒ slightly optimistic as a no-model reference,
biasing `R*` upward through the **denominator** as well as the numerator. Small, same error class as
the numerator's — and it **vanishes** under §6's split.

---

## 8 ⛔ The pre-declaration list

Each of these **changes the null**, so each is a free parameter until written down. Declared **before
(c) runs**, not after:

1. `α`, and any multiplicity correction over thresholds / coarsening rungs.
2. `B`, the replicate count — with §7.2's warning that a tail quantile needs `B` large.
3. ⭐ **the exchangeability unit** and the resampling scheme (§5.4) — *this is the null*.
4. ⭐ **the sign of `x`** (§3.4) — otherwise a false negative is free.
5. ⭐ **the scale of `y`** — raw counts or `log1p` (§7.2).
6. `d` and the block-size distribution of `x` (§4.4).
7. The split point `T` for §6's out-of-sample readout.

**Gates** (two, by different code paths, so a disagreement localises the bug):

- `DSC_ORD ≥ 0` — a **theorem** (§3.4), so a violation is an implementation bug.
- `DSC_ORD ≤ DSC_MAX` — §4.4. ⭐ Two gates that **disagree** are worth more than two that agree.

---

## 9 ⚠️ What remains a hypothesis for (c), and not to be settled by argument

| # | hypothesis | status after (a) + audit |
|---|---|---|
| H-a | On `P_all`, `DSC_ORD` is dominated by the `y = 0` block and re-measures occurrence | ⭐ plausible, **UNTESTED**. §1's restriction sidesteps rather than refutes it |
| H-b | A weaker (partial) order inflates `DSC` | ⭐ **direction DERIVED** (§4.2) · ⛔ magnitude unknown · ⚠️ and its application to `F` vs `mu` needs §2.2's nesting check first |
| H-c | `DSC/UNC_0` has large sampling variance on a 1–2,482 tail | ⛔ **UNTESTED**. `N4` precedent: `MCB/UNC` 4.32% against a floor of 11.91% |
| H-d | `mu` is dense enough to order 2,905 events | ⛔ **UNTESTED AND UNOBSERVED** — `mu` has never been on disk. ⚠️ And §4.3/§4.4 show both of v1's guesses about the consequence were wrong, so this is to be **measured** |
| ⭐⭐ H-e | **(REWRITTEN.)** The resampling scheme is correctly sized **under our spatial and temporal dependence** | ⛔ **UNTESTED**, and ⛔ the originally planned test of it was **vacuous** (§5.3). Must plant dependent fields |
| ⭐ H-f | **(NEW, REWORDED 2026-11-09.)** The in-sample null bias on **`R*`** is of **order `log n / n`** — constant **not derived**, so expect `10⁻³`–`10⁻²`, and ⭐ **the prediction under test is the ORDER, not the value.** ⛔ v1 wrote `≈ 2.7 × 10⁻³`, which silently sets the constant to 1; `opponent` flagged that I was over-claiming its own result, which is exactly the error class this dossier exists to prevent. ⭐ And state it on **`R*`**, not `DSC_ORD`: the per-threshold null inflation is proportional to that threshold's own `p_j(1−p_j)`, so the `w_j` weights **cancel** in the ratio ⇒ the prediction is robust to §7.2's tail dominance even though `DSC_ORD` is not | ⛔ **A SHARP, FALSIFIABLE PREDICTION** (§4.1); tie correction argued by direction only |
| H-g | **(NEW.)** The out-of-sample IDR skill (§6) is the decision-relevant number and is positive | ⛔ **UNTESTED** — and it is what the decision hangs on |

⇒ ⛔ **No design decision is taken.** (b) is pseudocode for §3, §5.4 and §6; (c) plants known answers
and asks whether H-a…H-g behave as stated, **for the reasons stated**.

---

## 10 Sources, as read

| | file | what was read |
|---|---|---|
| `Arnold2024` | `papers/Arnold2024_CRPSDecompositions.pdf` | §3.1 eq. (23)–(25); Prop. 3.1 + proof; abstract's term expansion. `library`: 4/4 CONFIRMED, pp. 4992, 5005–6 |
| `Henzi2021` | `papers/Henzi2021_IsotonicDistributionalRegression.pdf` | Thm 2.1; eq. (10); §2.2 IDR≡PAV; §2.3 eq. (15) + distinct-`y`; ⭐ §3 opening. `library`: 4/4 CONFIRMED, pp. 8, 10, 15 — ⭐ ellipsis verified honest |
| `Gneiting2023` | `papers/Gneiting2023_ConditionalCalibration.pdf` | `R*` §3.4 eq. (32), Thm 3.6 p. 3256, `R²` nesting p. 3255; decomposition = ⛔ **Thm 2.23** (p. 3243), **not** 2.26 (p. 3246, a different result about calibration under translation). `library`: CONFIRMED + the scope note in §3.2 |
| `Gneiting2007` | `papers/Gneiting2007_StrictlyProperScoringRules.pdf` | strict properness of CRPS on the finite-first-moment class (§4.4) |
| ⛔ **NOT HELD** | Clifford/Richardson/Hémon 1989; Dutilleul 1993 | §5.2. `library` 2026-11-09: **the library holds NEITHER** — no PDF, no metadata. ⇒ The claim no longer rests on them (§5.2 is **re-derived** from this file's own lemmas, see `02_PSEUDOCODE.md` §4.4); they are owed as **historical attribution** and stay **unquoted** until acquired and locator-audited |
| in-repo | `reports/2026-09-30_evaluation_learning_report.md` | `N4`, `N11` |
| in-repo | `reports/2026-11-04_rank_coupled_body_programme.md` | §2.3 and its *"before the first fit"* deadline |
| in-repo | `reports/2026-11-04_delivered_gate_lookup/tools/e1_data.py:13` | ✅ one month per origin at h1 ⇒ no duplicate rows |
