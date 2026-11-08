# (b) Pseudocode — how §01's math becomes code

**Date:** 2026-11-09 · **Status:** step (b) of three · ⛔ **No design decision is taken here, and no
real data is touched.** (c) plants known answers and tests whether each procedure below behaves as
`01_MATH.md` says, for the reason it says.

⇒ Read `01_MATH.md` first. Every section number below in the form `§N` refers to that file.

⭐ **One result makes this whole file short:** the CRPS of a step CDF against an observation is an
**exact finite sum** over the gaps between distinct truths (§1.3). That makes §3.3's threshold
integral exact rather than approximate, so the `DSC` **curve** falls out of the same arithmetic as the
scalar — no extra pass, no quadrature, no bandwidth.

---

## 0 The objects, once

```
  n            number of cells in the population            (P_event: 2,905)
  y[1..n]      truth, raw counts, y_i >= 1 on P_event
  x[1..n]      the candidate ordering index                  (mu, or gate, or ...)
  t[1..n]      the target month of cell i                    (needed by §5 and §6 only)

  ỹ[1..m]      the DISTINCT values of y, ascending           (§3.1: thresholds needed, and no more)
  w[1..m-1]    w_j = ỹ[j+1] - ỹ[j]                           ⭐ the threshold weights
```

⚠️ **`w` is where the tail dominance of §7.2 physically lives.** On raw counts with `max y = 2482`,
the top gaps are enormous, so a handful of thresholds carry most of the integral. ⇒ This is visible in
the code rather than hidden in an aggregate, which is the point of reporting the curve.

---

## 1 Primitives

### 1.1 Collapse `x` to its tie blocks — ⛔ do this FIRST, do not delegate it to a library

```
  PROCEDURE blocks(x):
      xs[1..d]  <- sorted distinct values of x                    # d = distinct-value count
      g[i]      <- index in 1..d such that xs[g[i]] == x[i]
      size[k]   <- #{ i : g[i] == k }
      RETURN xs, g, size
```

⭐ **Why explicitly.** §4.4: `x_i = x_j` imposes **both** order relations, so `F̂` is **forced** equal
on a block. Collapsing to `d` blocks and running PAV on block means with weights `size` is *exactly*
equivalent and makes that constraint structural. ⛔ Trusting a library's internal tie handling is the
`C-324`-shaped risk — a constraint we believe in that is never actually applied. `G4` (§6) checks it
anyway.

### 1.2 Weighted PAV at one threshold

```
  PROCEDURE pav_at(z, y, g, size, direction = DECREASING_IN_X):
      r[k]  <- ( Σ_{i: g[i]==k} 1{ y[i] <= z } ) / size[k]        # block mean of the indicator
      f[k]  <- weighted isotonic regression of r on 1..d,
                   weights = size, constrained NON-INCREASING in k
      RETURN f[1..d]                                              # f[k] = F̂(z) for every cell in block k
```

ℹ️ Non-increasing because larger `x` ⇒ stochastically larger ⇒ **smaller** CDF. (`sklearn.isotonic`
with `increasing=False`, or any PAV; §3.1 — for a total order `Henzi2021` §2.2 gives IDR ≡ PAV.)

### 1.3 ⭐ CRPS of a step CDF — an exact finite sum, and the Brier decomposition for free

`Henzi2021` §2.2: `F̂_i` is *"piecewise constant with any points of discontinuity at `y_1,…,y_n`"*.
With `F(z) = 0` below `ỹ_1` and `1` at or above `ỹ_m` (where the indicator agrees), the CRPS integral
`∫ (F(z) − 1{y ≤ z})² dz` has **support only on `[ỹ_1, ỹ_m]`** and is constant on each gap:

```
            crps(F, y)  =  Σ_{j=1}^{m-1}  w[j] · ( F(ỹ[j]) − 1{ y <= ỹ[j] } )²
```

⇒ **Exact, not a discretisation.** And it is literally a weighted sum of Brier scores at the
thresholds `ỹ[j]`, which is §3.3's identity `DSC_ORD = ∫ DSC_CORP(z) dz` made arithmetic:

```
            BS[j](F)  =  (1/n) Σ_i ( F_i(ỹ[j]) − 1{ y[i] <= ỹ[j] } )²
            CRPS(F)   =  Σ_j w[j] · BS[j](F)
```

### 1.4 The marginal, in closed form

`Ĝ(ỹ[j]) = (1/n) Σ_i 1{y_i ≤ ỹ[j]} =: p[j]`, the same for every cell, so

```
            BS[j](Ĝ)  =  p[j] · (1 - p[j])          ⭐ the per-threshold UNC
            UNC_0     =  Σ_j w[j] · p[j] · (1 - p[j])
```

⇒ No fit is needed for the reference. ⭐ This is also `G6`'s hook: `UNC_0` depends on the multiset of
`y` **alone**, so it must be **bit-identical** under any relabelling of `x` (§5.5).

---

## 2 The core measurement

```
  PROCEDURE dsc_ord(x, y):
      ỹ, w          <- thresholds and gaps from y                        # §0
      xs, g, size   <- blocks(x)                                         # §1.1
      p[j]          <- (1/n) Σ_i 1{ y[i] <= ỹ[j] }                       # §1.4

      FOR j = 1 .. m-1:
          f         <- pav_at(ỹ[j], y, g, size)                          # §1.2
          F[:, j]   <- f[ g[:] ]                                         # broadcast block -> cell
          BS_fit[j] <- (1/n) Σ_i ( F[i,j] - 1{ y[i] <= ỹ[j] } )²
          BS_mg[j]  <- p[j] * (1 - p[j])
          dsc_curve[j] <- BS_mg[j] - BS_fit[j]                           # ⭐ DSC_CORP at threshold ỹ[j]

      UNC_0    <- Σ_j w[j] * BS_mg[j]
      CRPS_ORD <- Σ_j w[j] * BS_fit[j]
      DSC_ORD  <- UNC_0 - CRPS_ORD                    # == Σ_j w[j] * dsc_curve[j], asserted by G7
      R_star   <- DSC_ORD / UNC_0

      RETURN DSC_ORD, R_star, UNC_0, dsc_curve, F, (xs, g, size)
```

⭐⭐ **`dsc_curve` is the headline** (§3.3), `R_star` its summary. The curve is what distinguishes *"the
ordering carries **magnitude** information"* from *"the ordering carries **any** information"*: an
index that separates 1 from 2 and is noise above 10 has its curve piled at low `ỹ`, visibly.

**Cost.** `m-1` PAV fits on `d` points: `O(m·d)`. At `m ≈ a few hundred` and `d ≤ 2,905`, one call is
milliseconds. ⚠️ `m` is **unmeasured** — it is the distinct-truth count on `P_event` and it sets the
whole budget. (c) measures it before anything else is sized.

### 2.1 The two signs — ⛔ not optional, and pre-declared (§3.4, break B3)

```
  PROCEDURE dsc_ord_signed(x, y, declared_sign):
      ASSERT declared_sign was written down before this ran            # else it is a free parameter
      primary  <- dsc_ord(declared_sign * x, y)
      mirror   <- dsc_ord(-declared_sign * x, y)
      RETURN primary, mirror
```

`DSC_ORD = 0` means *no isotone-**increasing** information*, not *no information*: `x = −y` scores
exactly 0 while determining `y` perfectly. Both signs are computed; the declared one is the
hypothesis, the mirror is the diagnostic.

---

## 3 The ceiling and the coarsening sweep (§4.4)

```
  PROCEDURE dsc_max(x, y):                                  # the ceiling - KEPT, but see the warning
      xs, g, size <- blocks(x)
      FOR each block k:
          Ĝ_k     <- ECDF of { y[i] : g[i] == k }
      oracle      <- (1/n) Σ_k Σ_{i in block k} crps(Ĝ_k, y[i])        # §1.3, per block
      RETURN UNC_0 - oracle
```

⚠️ **Near-vacuous when `d → n`** (§4.4, break B5): singleton blocks give `crps = 0`, so
`DSC_MAX → UNC_0`. Kept **only** as gate `G2`'s counterpart, computed by a different code path so a
disagreement localises the bug. ⛔ Not reported as a finding.

```
  PROCEDURE coarsen(x, d_target):                           # ⭐ the replacement readout
      xs, g, size <- blocks(x)
      cuts        <- d_target - 1 cut points placed at EQUAL CUMULATIVE COUNT over size
      x_coarse[i] <- index of the cut interval containing g[i]
      RETURN x_coarse

  PROCEDURE resolution_sweep(x, y, rungs = [2, 4, 8, 16, ..., d]):
      FOR each r in rungs:
          out[r] <- dsc_ord(coarsen(x, r), y)
      RETURN out
```

⭐ By §4.3 this curve is **monotone non-decreasing in `r` pathwise**, with stochastically ordered nulls
at each rung — asserted as `G7`. Its shape separates *"no resolution"* from *"resolution, wrong
ordering"* at **every** resolution instead of at one.

⚠️ **`coarsen`'s cut rule is a free parameter** and this repo has been bitten three times by exactly
that (`F24`, `M97`, `P-66`/`P-67`/`P-68` — subjects that turned out to be properties of `RATIO_EDGES`).
⇒ Equal-cumulative-count is declared here; (c) sweeps at least one alternative (equal-width in `x`)
and the sweep is reported, not selected from.

---

## 4 ⛔ The resampling schemes (§5) — three, and WHICH ONE IS THE NULL

```
  PROCEDURE resample(scheme, x, y, t, rng):
      CASE scheme OF

        PERMUTE:                                            # ⛔ exact only under exchangeability
            RETURN x[ rng.permutation(n) ], y, UNC_0_cached

        MONTH_SHIFT(k):                                     # ⭐ preserves x's dependence EXACTLY
            # pair month u's x with month (u + k)'s y, over the overlapping months only
            pairs <- { (i, i') : t[i'] == t[i] + k,  cell(i') == cell(i) }
            ⚠️ the y-multiset CHANGES (end months drop)
               => UNC_0 MUST be recomputed inside the loop; §5.5's invariance does NOT hold
            RETURN x over pairs, y over pairs, RECOMPUTE UNC_0

        STRATIFIED(strata):                                 # exact under within-stratum exchangeability
            FOR each stratum s:  permute x within s only
            RETURN x_permuted, y, UNC_0_cached               # y untouched => UNC_0 invariant
```

⛔⛔ **The chosen scheme IS the null hypothesis, not an implementation detail** (§5.2). `PERMUTE`
destroys `x`'s spatial autocorrelation and therefore **understates the null variance**, making the
test **anti-conservative** — it rejects too often, in the direction that waves `v1` through on a null
index. ⇒ **The scheme, and the exchangeability unit it implies, are pre-declared (§8.3).**

```
  PROCEDURE null_distribution(x, y, t, scheme, B, rng):
      obs <- dsc_ord(x, y).DSC_ORD
      FOR b = 1 .. B:
          x_b, y_b, unc_b <- resample(scheme, x, y, t, rng)
          rep[b]          <- dsc_ord(x_b, y_b).DSC_ORD       # with unc_b where the scheme moved it
      RETURN obs, rep[1..B]                                  # ⭐ the WHOLE distribution, not one quantile
```

⚠️ **Return the whole vector.** §7.2: a tail-dominated statistic has a skewed, heavy-tailed null, so a
single `1−α` quantile at modest `B` is unstable exactly where it is read. The report shows the
distribution; `p = (1 + #{rep ≥ obs}) / (1 + B)` is quoted **with** its floor `1/(1+B)` named.

**Cost.** `B` × `O(m·d)`. At `m ≈ 300`, `d ≈ 2,905`, `B = 999` this is ~3·10⁵ PAV calls. ⚠️ Budget
**unverified** — (c) times one replicate before `B` is chosen, and `B` is declared before the run.

### 4.1 Comparing two indices — paired, never by levels (§4.3)

```
  PROCEDURE paired_null(x_A, x_B, y, t, scheme, B, rng):
      obs <- dsc_ord(x_A,y).DSC_ORD - dsc_ord(x_B,y).DSC_ORD
      FOR b = 1 .. B:
          ONE resample, applied to BOTH                      # ⭐ the same relabelling for A and B
          rep[b] <- dsc_ord(x_A_b, y_b).DSC_ORD - dsc_ord(x_B_b, y_b).DSC_ORD
      RETURN obs, rep
```

⛔ Two indices with different tie structures carry **differently biased** levels, so levels are not
comparable and margins-over-separate-nulls are not either (different variances; a p-value floored at
`1/(1+B)` saturates). The paired difference absorbs both biases **and their correlation**. ⭐ Same
move, same reason, as 2026-11-07's paired bootstrap.

---

## 5 ⭐⭐ The out-of-sample readout — the number the decision hangs on (§6)

```
  PROCEDURE idr_fit(x_tr, y_tr):
      ỹ, w        <- thresholds from y_tr                    # ⛔ TRAINING thresholds, frozen
      xs, g, size <- blocks(x_tr)
      FOR j: f[:, j] <- pav_at(ỹ[j], y_tr, g, size)
      RETURN model = (xs, f, ỹ, w, p_tr)                     # p_tr = the TRAINING marginal

  PROCEDURE idr_predict(model, x_new):                        # a monotone step function of x
      k <- searchsorted(model.xs, x_new, side = RIGHT) - 1    # right-continuous lookup
      ⛔ x_new below xs[1] or above xs[d] lies in Henzi2021 §2.4's ENVELOPE - an INTERVAL
         of admissible CDFs, not a point. DECLARED RULE: clamp to the nearest fitted block.
         (c) measures how many test cells clamp; if it is a large share, the rule is load-bearing
         and must be reported as such rather than buried.
      RETURN model.f[ k, : ]

  PROCEDURE out_of_sample(x, y, t, T):
      tr <- { i : t[i] <= T } ;  te <- { i : t[i] > T }
      ASSERT tr and te are DISJOINT in month, not only in row                      # G8
      model     <- idr_fit(x[tr], y[tr])
      F_te      <- idr_predict(model, x[te])  for each test cell
      CRPS_oos  <- Σ_j model.w[j] * (1/|te|) Σ_{i in te} (F_te[i,j] - 1{y[i] <= ỹ[j]})²
      UNC_oos   <- Σ_j model.w[j] * (1/|te|) Σ_{i in te} (p_tr[j]  - 1{y[i] <= ỹ[j]})²
                   # ⛔ the TRAINING marginal, scored on TEST truth: a real, honest competitor
      SKILL     <- 1 - CRPS_oos / UNC_oos
      RETURN SKILL, CRPS_oos, UNC_oos, curve_oos
```

⭐ **No null is needed for the level.** Beating a frozen climatology out of sample *is* the claim, and
`SKILL ≤ 0` is a direct negative. This is `Henzi2021`'s own usage, and it removes §4's bias argument
and §7.3's `UNC_0` optimism at a stroke.

⚠️ **Only 13 months exist at h1** (`e1_data.py`: months 457–469, one per origin). ⇒ A split leaves very
few test months and `|te|`'s event count is small. The split point `T` is **pre-declared** (§8.3), and
if the test fold is too thin to resolve anything **that is the finding** — reported as *"not
evaluable at this n"*, never as a null result. ℹ️ `R-66`'s device: *a prediction about a population
must state the minimum `n` at which it is evaluable.*

---

## 6 The gates — and three of them come straight out of the derivations

| id | assertion | why it can fire |
|---|---|---|
| ⭐ `G1` | `DSC_ORD ≥ 0` | a **theorem** (§3.4): `Ĝ` repeated `n` times is feasible. A violation is a bug, never a result |
| ⭐ `G2` | `DSC_ORD ≤ DSC_MAX` | §4.4, computed by a **different code path** ⇒ a disagreement localises the fault |
| `G3` | `F[i, ·]` non-decreasing in `j`, and in `[0,1]` | the per-threshold fits must compose into a valid CDF. Monotone because isotonic regression is order-preserving in its response and `1{y ≤ ỹ_j}` is non-decreasing in `j` — ⚠️ **asserted, not trusted** |
| ⭐ `G4` | `F[i, ·] == F[i', ·]` whenever `x[i] == x[i']` | the tie constraint of §4.4. Checks §1.1 actually applied it |
| ⭐⭐ `G5a` | `x = y` ⇒ `R* = 1.0` and `CRPS_ORD = 0` | **truth-as-index self-test.** Distinct `y` + perfect monotone ⇒ point masses (`Henzi2021` §3). Same shape as `G-P4` in the existing instruments |
| ⭐⭐ `G5b` | `x = −y` ⇒ `DSC_ORD = 0.0` exactly | pins the **one-sidedness** of break `B3` as a live property, not a comment |
| `G6` | `UNC_0` **bit-identical** across `PERMUTE` replicates | §5.5 / §1.4. ⛔ Must FAIL under `MONTH_SHIFT`, which is itself the check that the shift really moved the `y` multiset |
| ⭐ `G7` | `resolution_sweep` is monotone non-decreasing in `r`, **pathwise**, on every replicate | §4.3's refinement theorem. A violation falsifies the derivation, not the data |
| `G8` | train and test month sets are disjoint | §5. The `e1_data` G-4 pattern |
| `G9` | `Σ_j w[j] · dsc_curve[j] == DSC_ORD` to machine precision | §1.3's identity ⇒ the curve and the scalar are the same object, so the curve cannot drift from the headline |

⛔ **Every gate must be shown to FIRE by a test that breaks the code**, per this repo's standing rule.
`G5a`/`G5b` are the pair I would trust least on its own and most as a couple: one says the instrument
can see perfect information, the other says it is blind in exactly the declared direction.

---

## 7 Order of execution for (c), and the diagnostic that comes before any score

```
  0.  m, d, the block-size distribution of x, and ONE replicate's wall time
      ⛔ printed BEFORE anything is scored - they set the budget and they are all unmeasured
  1.  G5a / G5b / G1 / G3 / G4 / G9 on planted data                 # does the instrument work at all
  2.  H-f: the null R* under EXCHANGEABLE planted H_0               # expect O(log n / n) ~ 2.7e-3
  3.  ⭐⭐ H-e: the null R* under DEPENDENT planted H_0, our geometry # expect PERMUTE to over-reject
  4.      the same, under MONTH_SHIFT and STRATIFIED                # which scheme recovers size?
  5.  G7 + the coarsening sweep on planted partial information
  6.  H-c: bootstrap width on a planted tail                       # is the thing resolvable at all
  7.  H-g: the out-of-sample procedure on planted data              # does SKILL recover a planted value
```

⛔ **Steps 2 and 3 are the whole point.** Step 2 alone is the vacuous test §5.3 identified: planting
exchangeable `H_0` validates a test that is exact under exchangeability. **Step 3 is the test of the
assumption**, and `PERMUTE` is *predicted to fail it*. ⇒ If step 3 comes back clean for `PERMUTE`, the
break `B2` was wrong and §5 should be reverted — which is the form a real prediction takes.

---

## 8 What is still a free parameter after (b)

⚠️ Named so none of them is silently chosen by the implementation:

1. ⭐ the **resampling scheme** / exchangeability unit (§4) — *this is the null*
2. ⭐ the **sign** of `x` (§2.1)
3. ⭐ the **scale** of `y` — raw counts or `log1p`; `w` changes, so `R*` changes (§7.2)
4. `coarsen`'s cut rule (§3)
5. the **envelope rule** for out-of-range `x_new` (§5)
6. the split point `T` (§5)
7. `α`, `B`, and multiplicity over thresholds and rungs
8. `idr_predict`'s `side = RIGHT` convention — ⚠️ a tie at a training `x` value resolves one way or the
   other and nothing here forces the choice

⇒ **(c) declares 1–8 in writing before it runs**, and sweeps 3, 4 and 5 rather than selecting them.

---

## 9 What (b) deliberately does NOT contain

- ⛔ **No real data path.** No `mu`, no checkpoint forward pass, no arm loop. (c) runs on planted data
  only, so that a failure is a failure of the instrument and cannot be mistaken for a finding.
- ⛔ **No estimator imported from the existing dossiers.** `resolution_curve` / `crossing` answer a
  *different* question (§01 §4.3's flag) and nothing here should accidentally inherit their population.
- ⛔ **No verdict thresholds.** What `R*` would be large enough to justify `v1` is not decided in (a),
  (b) or (c) — it is a decision rule, it belongs in a pre-registration, and it is Simon's.
