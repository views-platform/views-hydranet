# (a) The math — what quantity we want, exactly

**Date:** 2026-11-08 · **Status:** step (a) of three. ⛔ **No design decision is taken in this file.**
(b) pseudocode and (c) a synthetic simulation with a planted answer come next, and only then is any
choice here final.

⛔ **Why this file exists.** Simon, 2026-11-08: nothing is decided until there is *rigorous math,
pseudocode, and a simulation proving it works the way I think it does for the reasons I think it
does.* Measured on this codebase: on **"what"** questions I score ~**6/10** from memory alone; on
**"why"** questions **0/10**. ⇒ Every mechanism claim below is either **derived here** or **labelled
as a hypothesis for (c)**. Nothing is asserted from recall.

ℹ️ Every citation below was read from the PDF text in `~/brain/9_library/papers/_extracted_text/`
while writing this file, not recalled.

---

## 1 Setup and notation

A **population** of cells indexed `i = 1, …, n`, each carrying

| symbol | meaning | type |
|---|---|---|
| `y_i` | the truth — observed fatalities | `Z≥0` |
| `x_i` | a **candidate ordering index** — the thing whose ranking we are testing | `R` |
| `F_i` | *optionally*, an actual predictive distribution for cell `i` | CDF |

⭐ **`x` is the object under test, not `F`.** The question `v1` needs answered is *"does this ordering
carry information about magnitude"*, and an ordering is a scalar per cell. This distinction is
§2 and it is the main content of this file.

**Populations.** Three, never combinable (`C-351`):

| | definition | selected on | licenses |
|---|---|---|---|
| `P_all` | all 842,634 cell-months | nothing | occurrence + magnitude |
| `P_open` | `(draws > 0).any()` | the **prediction** | calibration (`Gneiting2011`) |
| ⭐ `P_event` | `y_i > 0` — the 2,905 event cells | the **outcome** | ⛔ **discrimination only** |

⇒ **Everything in this file is on `P_event`**, because `v1` reorders magnitudes *among cells that
fire*, and occurrence is already measured separately (AP 0.5678, `M100` axis 1). ⛔ Outcome-selected
⇒ no calibration claim may be read off any number defined here, ever.

---

## 2 ⛔⛔ TWO DIFFERENT STATISTICS, AND THEY MUST NOT SHARE A NAME

This is the first substantive result of doing the math, and it is a correction to how I described the
plan in conversation.

### 2.1 `DSC_ISO` — Arnold's statistic

`Arnold2024_CRPSDecompositions` §3.1 eq. (24)–(25) decomposes **the score of a forecast we already
have**. Its IDR covariate is `F_1, …, F_n` *themselves*, ordered by the **stochastic order** `≤_st`:

> *"we apply IDR to the data (F_1, y_1), …, (F_n, y_n) … where the covariates are F_1, …, F_n and the
> partial order on the covariate space is the stochastic order"*

⇒ `CRPS = MCB_ISO − DSC_ISO + UNC_0`. The subject is **the delivered forecast**.

### 2.2 `DSC_ORD(x)` — the statistic §2.3 of the programme actually asks for

What we want is different: *given only the ordering induced by `x`, how much discrimination is
available?* That is IDR with a **scalar** covariate under the **total order `≤` on R** — plain
`Henzi2021` IDR, no distributions in the covariate at all.

⛔ **These are not the same number and neither is an estimate of the other.** Quoting one as the other
is the `C-351` defect in its exact form: same estimator, different population and different transform.

⇒ **Naming rule for this dossier, to be added to `GLOSSARY.md`:**

| name | covariate | order | subject |
|---|---|---|---|
| `DSC_ISO` | the forecast `F_i` | `≤_st` (partial) | ⭐ how good **this forecast** is |
| ⭐ `DSC_ORD(x)` | the index `x_i ∈ R` | `≤` (total) | ⭐ how good **this ordering** is |

⚠️ A third quantity is constructible — IDR of `y` on `x` used to decompose `F`'s score — and it is
**not defined here and not to be computed**. It would mix a forecast's miscalibration with a
different variable's ordering.

---

## 3 The estimator, written out

### 3.1 IDR on a totally ordered covariate

`Henzi2021` Theorem 2.1 gives existence and uniqueness of the CRPS-based isotonic regression `F̂`.
For a **total order** with `x_1 < … < x_n`, their eq. (10) gives it **in closed form** — a max–min
formula over the response indicators, credited to Barlow et al. (1972) eqs. (1.9)–(1.13):

```
            F̂_i(z) =  min      max     (1/(j−k+1)) · Σ_{l=k..j} 1{y_l ≤ z}
                     k=1..i   j=k..n
```

and `Henzi2021` §2.3 eq. (15) gives the equivalent, and cheaper, characterisation — **one isotonic
regression per threshold**:

```
            F̂(z) = argmin  Σ_i ( η_i − 1{y_i ≤ z} )²        over η ∈ [0,1] non-increasing in x
```

Two facts that make this computable rather than theoretical, both stated in §2.3:

- ⭐ The objective is **constant between the distinct values of `y`**, so only the `m` distinct truths
  need thresholds — not all `i/j` quantile levels, which the paper calls *"computationally
  prohibitive"*. On `P_event`, `m` is the number of distinct fatality counts among 2,905 events.
- ⭐ For a total order, **IDR and the PAV algorithm coincide** (§2.2, citing Barlow Thm 1.12) ⇒
  `sklearn.isotonic.IsotonicRegression` is an admissible implementation at each threshold, and the
  per-threshold fits are automatically nested into a valid CDF.

### 3.2 The three components

With `F̂_i` the IDR solution and `Ĝ` the **empirical CDF of `y_1, …, y_n`** (the marginal — the
"no-model" forecast, the same distribution for every cell):

```
  CRPS_ORD(x) = (1/n) Σ_i crps( F̂_i , y_i )          the recalibrated score
  UNC_0       = (1/n) Σ_i crps( Ĝ   , y_i )          the marginal's score
  DSC_ORD(x)  = UNC_0 − CRPS_ORD(x)                  ⭐ THE QUANTITY
  R*(x)       = DSC_ORD(x) / UNC_0 ∈ [0, 1]          ⭐ THE HEADLINE, dimensionless
```

⛔ **`MCB` is NOT defined for `DSC_ORD`.** There is no forecast in the construction, so there is
nothing to be miscalibrated. ⇒ This instrument answers *"is there an ordering to preserve"* and says
**nothing** about whether any forecast built on it would be calibrated. Stating otherwise is the
`N4`-class error.

`R*` is `Gneiting2023_ConditionalCalibration`'s universal coefficient, which nests classical `R²`.
`R* = 0` ⇔ the ordering buys nothing; `R* = 1` ⇔ the recalibrated forecast is a point mass at the
truth in every cell.

### 3.3 The two `iff` endpoints, and why they are the reason to bother

`Arnold2024` Prop. 3.1, applied to §3.2's construction:

- `DSC = 0` ⟺ `F̂_i = Ĝ` for all `i` ⇒ ⭐ **the recalibrated forecast collapsed into the no-model
  forecast.** The index carries **no** ranking information. Not weak — none.
- Non-negativity `DSC ≥ 0` is a **theorem, not an empirical finding**: `Ĝ` repeated `n` times is
  itself an admissible candidate in the constrained minimisation (it satisfies every order relation
  trivially), so the minimum cannot exceed it. ⇒ A negative `DSC` is an implementation bug, which
  makes it a **gate**, not a result.

---

## 4 ⛔ The bias, derived — and this is where the conversation was wrong in both directions

### 4.1 `DSC_ORD > 0` for a random index. This is not a worry, it is a theorem-adjacent certainty

`Henzi2021` §3, opening paragraph, states both extremes explicitly:

> *"In the extreme case of distinct, totally ordered covariate values `x_1, …, x_n` and a perfect
> monotonic relationship to the response values `y_1, …, y_n`, the IDR distribution associated with
> `x_i` is simply the point measure in `y_i` … The same happens in the other extreme, when there are
> no order relations at all. Hence, **the partial order serves to regularize the IDR solution.**"*

⇒ Reading that off: `crps(δ_{y_i}, y_i) = 0`, so in either extreme `CRPS_ORD = 0` and
`DSC_ORD = UNC_0`, i.e. `R* = 1` — **a perfect score, from nothing.** The staircase is a fitted object
with up to `n` steps and it will fit noise.

⛔ ⇒ **A bare `DSC_ORD` figure is uninterpretable.** It is not "slightly optimistic"; its scale is set
by the fitting freedom, which is a property of `x`'s tie structure and `n`, not of the model. This is
the formal version of the evaluation report's `N4`.

### 4.2 ⛔ MY HAZARD (b) WAS RIGHT IN DIRECTION AND FOR A WEAKER REASON THAN I GAVE

In conversation I claimed: our sparse 16-draw CDFs cross ⇒ the stochastic order is partial ⇒ IDR has
more freedom ⇒ `DSC_ISO` inflates. The **direction is correct and provable**, and it is also stated by
the authors — *"the partial order serves to regularize"*, with no order relations being the extreme
that overfits perfectly.

**The derivation** (not recall): if two order relations satisfy `⪯_1 ⊆ ⪯_2` as sets of pairs, then
the feasible set `C_2 ⊆ C_1`, so `min_{C_1} ≤ min_{C_2}`. A weaker order ⇒ smaller `CRPS_ISO` ⇒
**larger** `DSC`. ∎

⛔ **But the reason I gave for why it matters was wrong.** I presented it as an argument for using a
scalar covariate *to avoid overfitting*. §4.1 shows the scalar/total-order case **also** overfits to
point masses when `x` is distinct-valued. ⇒ The total order is the right choice because it answers the
right question (§2.2), **not** because it is safe. Both need the null.

⚠️ **And the magnitude is still unknown.** The direction is derived; how much `DSC` inflates on our
sparsity, our `n`, our tie structure, is **not derivable and is a hypothesis for (c)**.

### 4.3 ⛔⛔ MY HAZARD (d) WAS BACKWARDS, and this is the real find of step (a)

I said: if `mu` turns out heavily tied, that is bad news for the index. ⛔ **Ties cut both ways, and
`Henzi2021` §3 says which way is which** — ties are *order relations*, and order relations
**regularise**. So:

| `x`'s tie structure | fitting freedom | `DSC_ORD` bias | ceiling on `DSC_ORD` |
|---|---|---|---|
| all values distinct | **maximal** (`n` steps) | ⛔ large | none — can reach `R* = 1` |
| heavily tied | **small** (`d` steps) | ⭐ small | ⛔ capped — see §4.4 |
| one value (`d = 1`) | none | **0** | **0 exactly** |

⇒ ⭐⭐ **The bias and the ceiling move in OPPOSITE directions with `d`.** A tied index is a
*less* overfit but *more* capped estimate. So "heavy ties" is not a verdict on the index — it is a
statement about which of two failure modes applies.

⛔⛔ **The decision consequence, and it is the one I would have got wrong:** two candidate indices
with different tie structures carry **differently-biased** `DSC_ORD` values. ⇒ **They cannot be
compared to each other directly.** Each must be compared **against its own permutation null**, and the
reported quantity is the *margin over its own null*, never the raw level. ⚠️ `M98`/`M99`'s crossing
comparison between `mean_draw` (1,321 distinct values) and `gate` (≈64,732) has exactly this shape —
**flagged, not re-opened here**; the crossing is a different estimator and whether the same argument
transfers is a separate question.

### 4.4 The ceiling, derived — a computable upper bound from ties alone

Let `G_1, …, G_d` be `x`'s tie blocks. `F̂` is constant on each block by the IDR constraint
(`P_i = P_j` when `x_i = x_j`). CRPS is a proper scoring rule, so the per-block minimiser of
`Σ_{i∈G_k} crps(P, y_i)` is the block's own empirical CDF `Ĝ_k`. Dropping the monotonicity constraint
between blocks can only lower the objective, hence

```
  CRPS_ORD(x)  ≥  (1/n) Σ_k Σ_{i∈G_k} crps( Ĝ_k , y_i )
  ⇒  DSC_ORD(x)  ≤  DSC_MAX(x)  :=  UNC_0 − (1/n) Σ_k Σ_{i∈G_k} crps( Ĝ_k , y_i )
```

⭐ `DSC_MAX(x)` is the most discrimination **any** monotone fit could extract given `x`'s resolution,
**ignoring whether the ordering is any good**. It costs one pass and it is the rigorous replacement for
my hand-waved "tie diagnostic". At `d = 1` it is exactly `0`, which recovers §3.3's endpoint. ∎

⇒ Report `DSC_ORD(x)`, `DSC_MAX(x)`, and the ratio. A small `DSC_ORD` with a small `DSC_MAX` means
*the index has no resolution*; a small `DSC_ORD` with a large `DSC_MAX` means *the index has
resolution and orders wrongly*. ⛔ These are different findings with different consequences for `v1`,
and no single number separates them.

---

## 5 The null, and why permutation is the right one

**`H_0`:** `y ⟂ x` — the index carries no information about magnitude.

Under `H_0` the labels are exchangeable, so for a permutation `π` of `1..n`, `DSC_ORD(x_π)` has the
same distribution as `DSC_ORD(x)`. Permuting:

- preserves the **marginal of `y`** ⇒ `UNC_0` is unchanged, so `R*` is comparable across replicates;
- preserves the **tie structure of `x`** ⇒ the fitting freedom of §4.3 is held fixed, which is
  precisely what makes the null measure the right bias;
- preserves `n` and the distinct-value count `d`.

⇒ An **exact** finite-sample test, conditional on both marginals. Report `DSC_ORD(x)` against the
`1−α` quantile of `B` permutation replicates.

⛔ **Two things this null does NOT do**, stated so they are not assumed away:

1. It gives **no confidence interval** on `DSC_ORD(x)` itself. That needs a **bootstrap** over cells —
   owed separately, and known to matter here: on 2026-11-07 the bootstrap on the related min-type
   `crossing` statistic put both point estimates at the **optimistic edge** of their own sampling
   distributions (`P(boot ≤ point)` = 0.190 and 0.043).
2. It does not fix the **sizing of the test**. ⇒ (c) must measure the false-positive rate at a planted
   `H_0` and show it is ≈ `α`. ⚠️ Precedent: the first threshold-reliability run used the 95th
   percentile and a perfectly calibrated synthetic forecast came back `MISCALIBRATED 1.1x` on the
   first seed — behaving exactly as specified, and useless. This one declares
   **`α` and any multiplicity correction before it runs.**

---

## 6 ⚠️ What remains a hypothesis, to be settled by (c) and not by argument

| # | hypothesis | status after (a) |
|---|---|---|
| H-a | On `P_all`, `DSC_ORD` is dominated by the `y = 0` block and mostly re-measures occurrence | ⭐ **plausible but UNTESTED.** §2's restriction to `P_event` sidesteps it rather than refuting it |
| H-b | Sparse, crossing predictive CDFs inflate `DSC_ISO` | ⭐ **direction DERIVED** (§4.2) · ⛔ **magnitude unknown** |
| H-c | `DSC_ORD / UNC_0` has large sampling variance on a 1–2,482 tail | ⛔ **UNTESTED.** `N4` precedent: a `MCB/UNC` of 4.32% against a noise floor of 11.91% |
| H-d | `mu` is dense enough to order 2,905 events | ⛔ **UNTESTED AND UNOBSERVED** — `mu` has never been on disk. ⚠️ And §4.3 shows I had the sign of the consequence backwards, so this must be *measured*, not reasoned about |
| H-e | The permutation test is correctly sized at our `n`, ties and skew | ⛔ **UNTESTED** — and §5's precedent says assume nothing |

⇒ ⛔ **No design decision is taken.** (b) is the pseudocode for §3–§5; (c) plants a known answer and
asks whether each of H-a…H-e behaves as §4 says, **for the reason §4 says**.

---

## 7 Sources, as read

| | file | what was read |
|---|---|---|
| `Arnold2024` | `papers/Arnold2024_CRPSDecompositions.pdf` | §3.1 eq. (23)–(25), Prop. 3.1 and its proof; the abstract's `MCB`/`DSC`/`UNC` expansion |
| `Henzi2021` | `papers/Henzi2021_IsotonicDistributionalRegression.pdf` | Thm 2.1; eq. (10) max–min under a total order; §2.2 IDR≡PAV; §2.3 eq. (15) threshold form and the distinct-`y` reduction; ⭐ §3 opening — the regularisation statement both extremes rest on |
| `Gneiting2023` | `papers/Gneiting2023_ConditionalCalibration.pdf` | `R* = DSC/UNC`; the decomposition is ⛔ **Thm 2.23**, not 2.26 (per this repo's own `E.0` correction) |
| in-repo | `reports/2026-09-30_evaluation_learning_report.md` | `N4` (every score needs its own null floor) and `N11` (the terms, the two papers) |
| in-repo | `reports/2026-11-04_rank_coupled_body_programme.md` | §2.3 — the measurement and its *"before the first fit"* deadline |
