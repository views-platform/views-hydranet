# (b) Pseudocode — how §01's math becomes code

**Date:** 2026-11-09 · **Status:** step (b) of three · ⛔ **REVISED after two audits — see §0.** ⛔ **No
design decision is taken here, no real data is touched, and no verdict threshold is set.**

⇒ Read `01_MATH.md` first. `§N` below refers to that file; `(b)§N` to this one.

---

## 0 ⛔ WHAT THE AUDITS CHANGED

| seat | job | result |
|---|---|---|
| `library` | locator audit | 1 CONFIRMED (the load-bearing one), **2 wrong**, **2 not found** |
| `opponent` | adversarial maths | **the bug I found is real and has 3 parts**, `MONTH_SHIFT` **dead twice over**, **3 of 10 gates cannot fire**, and ⭐ the two gates that would catch the most likely silent error **did not exist** |

⭐ **Where the derivations were checkable, I re-derived them before accepting.** The per-gap algebra,
the `U/(U+Δ)` relation, the `G3` monotonicity proof and the `G5a`/`G5b` asymmetry all hold.

| # | what was wrong in v1 | now |
|---|---|---|
| ⛔⛔ **P1** | **The exact-CRPS sum is wrong out of sample, in three ways** — above-range, below-range, and ⭐ an **interior** error whenever a test truth falls strictly inside a training gap | (b)§1.3 **replaced** by one formula correct in-sample and out |
| ⛔⛔ **P2** | **`MONTH_SHIFT` was unimplementable**, not merely weak: as written it required the same cell to be an event in both months ⇒ a persistence-biased subset that **excludes onsets** | (b)§4 **rebuilt**; primary null is now a **longitude-only, land-only, rank-reimposed** shift |
| ⛔⛔ **P3** | **`G9` was decorative — it could not fire.** Both sides are the same expression over the same two arrays | replaced by `G11`, an **independent recomputation**, which would have caught `P1`, `P4` and `P5` |
| ⛔⛔ **P4** | **Nothing forced `sample_weight` to be the block sizes.** `sklearn`'s default is equal weights, and the wrong answer passes every structural gate | `G12`, with deliberately unequal blocks |
| ⛔ **P5** | `sklearn`'s `.predict()` **linearly interpolates** between knots, silently turning a step CDF into a piecewise-linear one, invisibly to `G4` | ⛔ use `.y_`; never `.predict()` |
| ⛔ **P6** | `searchsorted(..., side=RIGHT) − 1` returns **`k = −1`** below the training range, which in numpy indexes the **LAST** block ⇒ the lowest cells silently get the **highest** CDF, the exact inverse of the declared clamp | `G13` asserts the clamp bounds and counts each end |
| ⛔ **P7** | A single train/test split wastes most of 13 months | **LOMO + rolling-origin**, both reported (b)§5 |
| ⚠️ **P8** | `G2` and `G4` also cannot fire as placed | `G2` moved to the low rungs; `G4` labelled for what it tests |
| ⚠️ **P9** | `Henzi2021` **§2.5**, not §2.4 — and the paper **prescribes** the prediction rule | adopted, (b)§5 |
| ⚠️ **P10** | Three places inferred their own integration grid | ⭐ one rule, (b)§1.0 |
| ⛔⛔ **P11** | **§2.5 contains TWO rules and this file merged them.** Metric interpolation uses `x`'s **values**; eq. (21)'s midpoint uses only the **order**. Under the first, out-of-sample `SKILL` is **not** invariant to a monotone reparameterisation of `x` ⇒ `mu`, `log mu` and `rank(mu)` would give **three different go/no-go answers**, with the in-sample half scale-free and the half the decision hangs on not | ⭐ **eq. (21)**, and it is a **derived** choice, (b)§5.1 |
| ⭐⭐ **P12** | **The anti-conservatism rested on two papers the library does not hold.** It is now **re-derived from this file's own lemmas**, with the undetermined constant **cancelling** | (b)§4.4 — and it yields a **three-point** prediction, not one |

---

## 1 Primitives

### 1.0 ⛔ THE RULE THAT KILLS A WHOLE DEFECT CLASS

> ⭐⭐ **`ỹ` and `w` are explicit arguments to every scoring call. No function EVER derives them from
> the distribution or the data it is handed.**

`P1` was one instance of a helper inferring its own grid. `opponent` found three more: `dsc_max`
deriving the grid from each block's own support (⇒ `G2` would compare two numbers **in different
units**); `MONTH_SHIFT` moving the grid per replicate (⇒ `rep[b]` not on the same scale as `obs`);
and the `log1p` sweep changing `w` while leaving `p[j]` invariant, so a cached `w` is silently wrong.

⇒ `w` is computed in **one place**, as a function of the declared scale of `y`, and passed down.

### 1.1 The objects — declared, including four that v1 omitted

```
  n            cells in the population                           (P_event: 2,905)
  y[1..n]      truth, raw counts (or log1p — DECLARED, §8.3)
  x[1..n]      the candidate ordering index
  t[1..n]      target month of cell i
  ⭐ cell[1..n]   priogrid id of cell i                            # NEW — (b)§4 needs it
  ⭐ X[c, u]      the FULL x field over all land cells x all months # NEW — (b)§4's shift needs it
  ⭐ lat[1..n]    latitude band of cell i                          # NEW — the shift axis
  ⭐ strat[1..n]  SPATIAL stratum (country/region) of cell i        # NEW — STRATIFIED needs it

  ỹ[1..m]      distinct values of y, ascending
  w[1..m-1]    w_j = ỹ[j+1] - ỹ[j]
```

⚠️ **`w` is where §7.2's tail dominance physically lives** — on raw counts to 2,482 the top gaps are
enormous, so a few thresholds carry most of the integral. Visible in the code, not hidden.

⛔ **CRPS convention, declared** (`opponent`, not-on-list 7): the **continuous** CRPS
`∫_R (F(z) − 1{y ≤ z})² dz`, which is what `Henzi2021`/`Arnold2024` use — **not** the discrete ranked
probability score summed over integers. `y` is integer-valued, so the two differ. ⇒ If (c) validates
against a library implementation using the other convention, the mismatch will present as a bug in the
new code.

### 1.2 Collapse `x` to tie blocks — ⛔ explicitly, and with the weights

```
  PROCEDURE blocks(x):
      xs[1..d]  <- sorted distinct values of x
      g[i]      <- index k with xs[k] == x[i]
      size[k]   <- #{ i : g[i] == k }
      RETURN xs, g, size
```

⭐ **Why explicitly.** §4.4: `x_i = x_j` imposes **both** order relations ⇒ `F̂` is **forced** equal on
a block. Collapsing to `d` blocks and running weighted PAV on block means is **exactly** equivalent.
`opponent`'s proof, re-derived here — the bias–variance split, with `η_i = f_{g(i)}`:

```
  Σ_{i∈G_k} (f_k − r_i)²  =  |G_k| (f_k − r̄_k)²  +  Σ_{i∈G_k} (r_i − r̄_k)²
```

and the second term is free of `f`. ∎ ⚠️ **Condition v1 omitted: this is a property of SQUARED loss**
(per threshold), via that split — it holds for Bregman divergences and **fails** for absolute loss. It
is *not* a CRPS-level equivalence.

### 1.3 Weighted PAV at one threshold — ⛔ three traps, each silent

```
  PROCEDURE pav_at(z, y, g, size):
      r[k]  <- ( Σ_{i: g[i]==k} 1{ y[i] <= z } ) / size[k]
      f[k]  <- weighted isotonic regression of r on 1..d,
                   ⛔ sample_weight = size          # P4 — NOT the default
                   constrained NON-INCREASING in k   # larger x => stochastically larger => smaller CDF
      RETURN f[1..d]                                 # ⛔ read .y_ , NEVER .predict()   # P5
```

⛔⛔ **P4 is the single most likely silent wrong number in this file.** `sklearn.isotonic` defaults to
`sample_weight=None`. Dropping it solves a **different problem** and the output is still monotone,
still in `[0,1]`, still block-constant, still curve-consistent — it passes `G1`, `G3`, `G4`, `G5a` and
(within tolerance) `G5b`. One keyword, wrong number, clean gates. ⇒ `G12`.

⛔ **P5:** `.predict()` interpolates linearly between knots, converting the step CDF into a
piecewise-linear one. `G4` cannot see it — at identical `x` interpolation returns identical values.

### 1.4 ⭐ CRPS of a step CDF — ONE formula, correct in-sample and out

`Henzi2021` §2.2 (`library`: **CONFIRMED verbatim, p. 8**, and stated **generally**, for partial orders
too): *"`F̂_i` is piecewise constant with any points of discontinuity at `y_1, …, y_n`"* — jumps occur
**only** at observed `y` values.

⛔⛔ **P1.** v1 wrote `crps = Σ_j w_j (F(ỹ_j) − 1{y ≤ ỹ_j})²`. That is right **in-sample only**. Out of
sample the model's grid is frozen at the *training* truths and a test `y` need not be on it. Three
failures, re-derived here:

| | condition | the term v1 omits |
|---|---|---|
| above-range | `y > ỹ_m` | `F = 1`, indicator `0` ⇒ `(y − ỹ_m)` |
| below-range | `y < ỹ_1` | `F = 0`, indicator `1` ⇒ `(ỹ_1 − y)`. ⚠️ Empty on `P_event` today, **not** empty under the `log1p` sweep or on `P_all` |
| ⭐ **interior** | `ỹ_j < y < ỹ_{j+1}` | the **indicator is not constant on the gap**. True term is `(y−ỹ_j)f_j² + (ỹ_{j+1}−y)(1−f_j)²`; v1 computes `w_j f_j²`. Error `(ỹ_{j+1}−y)(1−2f_j)` — **signed**: understates where `f_j < ½`, overstates where `f_j > ½` |

⭐ **The replacement — and v1's formula falls out of it as the in-sample special case:**

```
   a_ij = clip( y_i − ỹ_j , 0 , w_j )          # length of gap j lying BELOW y_i
   b_ij = w_j − a_ij                           # length lying at or above y_i

   crps(F_i, y_i) =  (ỹ_1 − y_i)⁺
                  +  Σ_{j=1}^{m-1} [ a_ij · f_ij²  +  b_ij · (1 − f_ij)² ]
                  +  (y_i − ỹ_m)⁺
```

**Verified in-sample:** `y_i` is a knot, so below it `a = w_j, b = 0 ⇒ w_j f²`, at or above it
`a = 0, b = w_j ⇒ w_j(1−f)²` — v1's formula term for term, with both out-of-range terms vanishing. ∎

⇒ ⛔ **This replaces v1's §1.3 everywhere, not just in the out-of-sample path.** No branch to forget.
Vectorises as `a = clip(y[:,None] − ỹ[None,:m−1], 0, w)`.

ℹ️ Equivalent alternative, kept as `G11`'s cross-check: evaluate on `sorted(unique(ỹ^tr ∪ y^te))`. On
that union grid both factors are constant on every gap and v1's identity holds verbatim.

### 1.5 The marginal, in closed form

`Ĝ(ỹ_j) = p[j] := (1/n) Σ_i 1{y_i ≤ ỹ_j}`, the same for every cell ⇒

```
   BS[j](Ĝ) = p[j](1 − p[j])          ⭐ the per-threshold UNC
   UNC_0    = Σ_j w[j] · p[j](1 − p[j])
```

⇒ No fit needed for the reference, and `UNC_0` depends on the multiset of `y` **alone** — which is
`G6`'s hook.

---

## 2 The core measurement

```
  PROCEDURE dsc_ord(x, y, ỹ, w):                     # ⛔ grid passed IN (b)§1.0
      xs, g, size <- blocks(x)
      p[j]        <- (1/n) Σ_i 1{ y[i] <= ỹ[j] }
      FOR j = 1 .. m-1:
          f            <- pav_at(ỹ[j], y, g, size)
          F[:, j]      <- f[ g[:] ]
      CRPS_ORD <- (1/n) Σ_i crps(F_i, y_i)            # (b)§1.4, the unified formula
      BS_fit[j]<- (1/n) Σ_i ( F[i,j] - 1{ y[i] <= ỹ[j] } )²
      BS_mg[j] <- p[j] (1 - p[j])
      dsc_curve[j] <- BS_mg[j] - BS_fit[j]            # ⭐ DSC_CORP at threshold ỹ[j]
      UNC_0    <- Σ_j w[j] BS_mg[j]
      DSC_ORD  <- UNC_0 - CRPS_ORD
      R_star   <- DSC_ORD / UNC_0
      RETURN DSC_ORD, R_star, UNC_0, CRPS_ORD, dsc_curve, F, (xs, g, size)
```

⭐⭐ `dsc_curve` is the headline (§3.3); `R_star` its summary. An index that separates 1 from 2 and is
noise above 10 has its curve **piled at low `ỹ`**, visibly.

⚠️ ⭐ **`CRPS_ORD` is now computed by the per-cell formula and `BS_fit` separately** — deliberately, so
`G11` can compare two paths that share no intermediate. In v1 they shared everything, which is why
`G9` was decorative.

**Cost.** `m−1` weighted PAV fits on `d` points: `O(m·d)`. ⚠️ `m` is **unmeasured** and sets the whole
budget — (c) prints it first ((b)§7 step 0).

### 2.1 The two signs — pre-declared (§3.4, break B3)

```
  PROCEDURE dsc_ord_signed(x, y, ỹ, w, declared_sign):
      ASSERT declared_sign was written down before this ran
      RETURN dsc_ord(declared_sign*x, …), dsc_ord(-declared_sign*x, …)
```

`DSC_ORD = 0` means no isotone-**increasing** information: `x = −y` scores exactly 0 while determining
`y` perfectly.

---

## 3 The resolution sweep (§4.4)

```
  PROCEDURE dsc_max(x, y, ỹ, w):                      # ⛔ GLOBAL grid, never the block's own (b)§1.0
      xs, g, size <- blocks(x)
      oracle      <- (1/n) Σ_k Σ_{i∈G_k} crps( ECDF{y : g==k} , y[i] ; ỹ, w )
      RETURN UNC_0 - oracle
```

⚠️ Near-vacuous as `d → n` (§4.4). Kept **only** as `G2`'s counterpart — and `G2` now runs at the
**low rungs**, where the bound actually binds.

```
  PROCEDURE coarsen(x, r):                            # r a DYADIC rung
      xs, g, size <- blocks(x)
      cuts        <- r-1 cut points at levels j/r of the CUMULATIVE COUNT over size
      x_coarse[i] <- index of the cut interval containing g[i]    # ⭐ a function of g, NEVER of x[i]
      RETURN x_coarse, realised_d'                    # ⭐ REALISED, not requested
```

⭐ **Why the ladder must be dyadic** (`opponent`, item 6): cuts at levels `j/r` **nest** across rungs
because `j/2 = 2j/4`, so rung `r`'s cut set is contained in rung `2r`'s ⇒ each rung is a genuine
coarsening of the next ⇒ **`G7`'s pathwise monotonicity is STRUCTURAL, not asserted.**

⛔ **Therefore: data-adaptive binning is excluded by rule** — `k`-means, Jenks, variance-minimising
cuts do **not** nest across rungs, and that is precisely what would quietly break the theorem.

⭐ `x_coarse` is a function of `g` alone, never of `x`'s value, which is why equal-count coarsening can
never *split* a block. ⛔ Keep that property explicit — it is the whole guarantee and one refactor from
being lost.

⚠️ **Realised `d'` can be less than `r`** when one block holds a large share of the mass and several
cuts collapse onto the same boundary ⇒ report realised `d'` at every rung, or the rungs are mislabelled.

⚠️ `coarsen`'s cut rule is a free parameter, and this repo has been bitten three times by exactly that
(`F24`, `M97`, `P-66`/`P-67`/`P-68` — subjects that turned out to be properties of `RATIO_EDGES`).
⇒ Equal-cumulative-count declared; equal-width-in-`x` swept as the alternative; **both dyadic**.

---

## 4 ⛔ The null — REBUILT. Which scheme is used IS the hypothesis

```
  PROCEDURE resample(scheme, …, k, rng):
      CASE scheme OF

        ⭐ LON_SHIFT(k):                      # PRIMARY. longitude-only, land-only, rank-reimposed
            FOR each latitude band:
                cyclically permute x among THAT BAND'S LAND CELLS by offset k
            x_sh[i] <- the shifted field at cell i
            ⭐ RANK-REIMPOSE: x_b <- the OBSERVED x multiset, assigned in x_sh's rank order
            RETURN x_b, y UNCHANGED, UNC_0 cached

        MONTH_SHIFT(k):                       # temporal DIAGNOSTIC only — p-floor 1/13
            x_b[i] <- X[ cell[i], ((t[i] + k - 457) mod 13) + 457 ]    # rows FIXED
            RETURN x_b, y UNCHANGED, UNC_0 cached

        STRATIFIED:                           # ⛔ SPATIAL strata ONLY, never month
            permute x within each spatial stratum
            RETURN x_b, y UNCHANGED, UNC_0 cached

        PERMUTE:                              # kept as the deliberately BROKEN comparator
            RETURN x[rng.permutation(n)], y, UNC_0 cached
```

### 4.1 Why each, and why v1's version was worse than weak

⛔⛔ **`MONTH_SHIFT` as v1 wrote it could not be implemented.** The pairing
`{(i,i') : t[i']=t[i]+k, cell(i')=cell(i)}` over the **outcome-selected** event rows requires the same
cell to be an event in **both** months. ⇒ The replicate population became the cells that fire twice `k`
apart — an unmeasured, `k`-dependent, **persistence-biased** subset that **systematically excludes
onsets**, which are exactly the cases a magnitude model most needs. And `n`, `ỹ`, `w`, `d`, `UNC_0`
all moved, differently per `k`. **That is not a null distribution.**

⭐ **The fix restores every invariance v1 thought it had to surrender:** hold the **rows** fixed and
replace each row's `x` with the value the *same cell* carried `k` months earlier, wrapping cyclically.
`y` untouched ⇒ `ỹ`, `w`, `UNC_0`, `n` all invariant, `G6` holds, the grid never moves, and `x` keeps
its real spatial autocorrelation because it **is** a real `x` field, from the wrong month. ⚠️ Still
`1/13` ⇒ **diagnostic, not primary.**

⭐⭐ **Why longitude-only.** The sphere is periodic in **longitude only**. A latitudinal shift maps the
Sahel onto Scandinavia. And longitude-only is conservative in the right way: it **preserves `x`'s
latitude distribution**, so the gross co-location of the two fields in the conflict band survives and
only the *fine-grained* alignment is destroyed — exactly what a null should break and keep.

⭐ **Why land-only, within bands.** A raw raster shift moves donors onto ocean (64,818 land of 259,200
raster cells). ⛔ Do **not** fix it by dropping rows — that changes `n` and the population, which is
`P2` again. Shift **within each latitude band's land cells**, so every event row keeps a defined donor,
`n` is exact, and within-band short-range autocorrelation largely survives.

⭐ **Why rank-reimposition.** A spatial shift changes the `x` **multiset** over the 2,905 rows, so `d`
and the tie structure would move per replicate — and holding those fixed is the point. Repair, free
because of §2.2's reparameterisation invariance: take the **shifted field's ordering**, then assign the
**observed** `x` values in that rank order. `DSC_ORD` depends on `x` only through ordering and ties, so
this changes **nothing** about the statistic while making the tie structure **exactly** preserved.
⚠️ Declare a tie-breaking rule for ties in the donor field.

⛔ **`STRATIFIED` by month does not address `B2` at all** — `B2` is about **spatial** autocorrelation,
and permuting within a month destroys the within-month spatial alignment that *is* the problem. Strata
must be **spatial**. ⇒ `01_MATH.md` §5.4's *"or within month"* is struck.

### 4.4 ⭐⭐ P12 — the anti-conservatism, re-derived from THIS FILE'S OWN LEMMAS

⛔ `library` round 2: the library **holds neither** Clifford/Richardson/Hémon (1989) **nor** Dutilleul
(1993) — no PDF, no metadata. ⇒ A finding that changed the design cannot rest on a reference nobody
here has opened (this repo's standing scar: an unsourced docstring sentence quoted as a measurement
across a repo boundary). ⭐ **So the claim is rebuilt on lemmas already derived in this dossier.** Four
parts, three epistemic statuses:

**(a) VALIDITY — broken on logic alone, no direction and no citation needed.** A permutation test is
exact *conditional on* the randomisation hypothesis `(x, y_π) =_d (x, y)` for every `π`. That is a
checkable statement about distributions, and it is **false** whenever neither vector is exchangeable.
§5.1's correction stands on this alone.

**(b) DIRECTION — derived, and the undetermined constant CANCELS.** Take the `b`-fold
perfectly-autocorrelated limit: partition `n` cells into `G = n/b` groups, with `x` **and** `y` each
constant within a group and independent across groups.

1. By **(b)§1.2's tie-collapse** (exact for squared loss), `x`'s blocks *are* the groups, so `DSC_ORD`
   on `n` cells equals the problem on `G` group summaries with equal weights `b`. And `UNC_0` is
   **identical** either way — repeating each `y` value `b` times changes neither the ECDF nor the
   threshold grid. ⇒ `R*` on `n` cells **is** `R*` on `G` observations, exactly.
2. By **§4.1's corrected bias**, the null level of `R*` on `G` exchangeable observations is `≈ c·H_G/G`.
3. A permutation of the `n` cell labels **scatters each `x` across groups**, destroying the block
   structure and leaving an exchangeable sequence of length `n` ⇒ permutation null level `≈ c·H_n/n`.
4. Ratio of true null location to permutation null location:

```
   (H_G/G) / (H_n/n)  =  b · ( H_{n/b} / H_n )  ≈  b · ( 1 − log b / log n )   >  1   for b > 1
```

⇒ ⛔ **The permutation null is centred TOO LOW by a factor of order `b`**, and the observed statistic is
judged against it. ⭐ **Over-rejection follows from a LOCATION argument** — stronger than the variance
argument §5.2 originally gave. ∎ ⭐⭐ And `H-f`'s unmeasured constant `c` **cancels** in the ratio (same
functional form, same `y` multiset, different length only), so this does not inherit it. At `n = 2,905`
and `b = 2` the factor is **1.83**; at `b = 100` it is **42**.

**(c) ⭐ A sharper condition — and it turns one prediction into a discriminating PATTERN.** Redo (b)
with `x` block-constant but `y` **i.i.d.**: PAV can then explain only *between-block* variance, whose
share of the total is `1/b`, giving `R*_null ≈ c·H_{n/b}/n` against the permutation's `c·H_n/n` — a
ratio **below 1**. ⛔ **That case is CONSERVATIVE, not anti-conservative.**

⇒ The distortion requires **JOINT** autocorrelation — roughly the *product* of the two fields'
autocorrelations at each lag, not either alone. Both are present here (`mu` smooth by construction,
fatalities clustered), so the direction stands, and (b)§7 step 4 becomes:

| planted `H_0` | predicted size of `PERMUTE` |
|---|---|
| both fields i.i.d. | **exact** |
| `x` smooth, `y` i.i.d. | ⛔ **UNDER**-rejects |
| ⭐ both smooth, our geometry | ⛔ **OVER**-rejects, by a factor of order `b` |

⭐⭐ **Reproducing that PATTERN confirms the mechanism, not merely the direction** — the *"for the
reason I think it does"* standard rather than *"it came out the way I expected"*. ⇒ A single
over-rejection on one planted case is **no longer enough** to claim it.

**(d) MAGNITUDE — unmeasured, and this is where the caveat belongs.** The effective `b` on our data is
unknown ⇒ step 4 is the **evidence**, not a confirmation of something already settled.

⭐⭐ **Which promotes `n_eff` from a nice-to-have to the parameter that SETS the distortion.**
`n_eff = n/b`, and (b) says the size distortion is of order `b`. ⇒ Measuring `n_eff` in (b)§7 **step 0**
gives an *a priori* estimate of how badly `PERMUTE` is wrong **before spending a single replicate**, and
says in advance whether any scheme can have power at this `n`. **It is the first number printed.**

ℹ️ Clifford et al. and Dutilleul are now worth acquiring as **historical attribution** — the claim no
longer waits on them, and until they are in the library and locator-audited the citations stay out.

### 4.2 The distribution, and the simultaneous band

```
  PROCEDURE null_distribution(x, y, ỹ, w, scheme, B, rng):
      obs <- dsc_ord(x, y, ỹ, w)
      FOR b = 1..B:  rep[b] <- dsc_ord(resample(scheme,…), ỹ, w)
      RETURN obs, rep[1..B]                           # ⭐ the WHOLE distribution
```

⚠️ Return the whole vector. §7.2: a tail-dominated statistic has a skewed, heavy-tailed null, so a
single `1−α` quantile at modest `B` is unstable exactly where it is read.
`p = (1 + #{rep ≥ obs})/(1 + B)`, quoted **with** its floor `1/(1+B)` named.

⭐⭐ **Multiplicity — dissolve most of it, then do the rest for free.**

1. **Pre-declare ONE primary endpoint:** the scalar `R*` at full resolution. The curve and the rungs
   are a **descriptive readout**. Multiplicity bites only if we *select* from them.
2. For genuine simultaneous inference, use a **maxT permutation band** (Westfall–Young): per replicate,
   standardise each threshold by its own permutation SD and take the max over thresholds and rungs; the
   `1−α` quantile gives an **exact family-wise** band that adapts to the dependence. ⭐ **Zero extra
   cost** — it reuses the replicates already computed. Bonferroni would be absurdly conservative here
   because the tests are strongly positively dependent (one fit, nested thresholds).

⚠️ **The trap in that standardisation:** at the top thresholds `p(1−p)` is tiny while `w` is enormous,
so the permutation SD can be numerically zero and the standardised statistic explodes. ⇒ **Declare, in
advance, a floor on `p[j](1−p[j])` below which a threshold is excluded from the family.** Choosing that
floor after seeing the curve is the same error class as choosing the cut rule after seeing the sweep.

### 4.3 Comparing two indices — paired, never by levels (§4.3)

```
  PROCEDURE paired_null(x_A, x_B, y, ỹ, w, scheme, B, rng):
      obs <- DSC(x_A) - DSC(x_B)
      FOR b: ONE resample, applied to BOTH => rep[b] <- DSC(x_A_b) - DSC(x_B_b)
```

⛔ Different tie structures ⇒ differently biased levels ⇒ levels not comparable, and
margins-over-separate-nulls not comparable either (different variances; a p-value floored at `1/(1+B)`
saturates). The paired difference absorbs both biases **and their correlation**.

---

## 5 ⭐⭐ Out of sample — the number the decision hangs on (§6)

### 5.1 The prediction rule is the paper's, not mine

`library`: the envelope is `Henzi2021` **§2.5 "Prediction"** (pp. 14–15), **not** §2.4 ("Consistency").
⭐ And the paper **prescribes** rules:

⛔⛔ **P11 — and §2.5 holds TWO rules with OPPOSITE invariance properties. This file merged them.**

| rule | weight | uses | scale-free in `x`? |
|---|---|---|---|
| pre-(20) **linear interpolation**, offered *"in the specific case `X = R`"* | `λ = (x − x_k)/(x_{k+1} − x_k)` | the **metric** on `x` | ⛔ **NO** |
| ⭐ **eq. (21)**, the pointwise average of the envelope bounds, which *"solely uses information supplied by the partial order"* | `λ = ½` always | the **order** only | ✅ **YES** |

⭐⭐ **§2.2's invariance decides it, and the stakes are the whole instrument.** `DSC_ORD` depends on `x`
only through its ordering and ties — that is *why* this measures an **ordering**. Under the metric rule
the out-of-sample forecast depends on `x`'s **values**, so §6's `SKILL` would **not** be invariant to a
monotone reparameterisation. ⛔ The in-sample half of this file would be scale-free and **the
out-of-sample half — the half the decision hangs on — would not**: `mu`, `log mu` and `rank(mu)` would
give **three different answers to the go/no-go question.**

⇒ **eq. (21). A DERIVED choice, not a declared one.** For a total order `s(x)` and `p(x)` hold at most
one element each, so eq. (21) is the **midpoint**, not an `x`-weighted interpolation:

```
  PROCEDURE idr_predict(model, x_new):
      k <- searchsorted(model.xs, x_new, side = RIGHT) - 1
      ASSERT -1 <= k <= d                                       # before clamping
      IF   x_new <  xs[1]  : RETURN f[1]                        # clamp — see below
      ELIF x_new >  xs[d]  : RETURN f[d]
      ELIF x_new == xs[k]  : RETURN f[k]
      ELSE                 : RETURN ½ ( f[k] + f[k+1] )         # ⭐ eq. (21)
```

ℹ️ This is **still a step function of `x`** — with `2d − 1` levels instead of `d`. So `searchsorted` was
nearly right; it needed the midpoint when `x_new` falls strictly between two training values. ⛔ `P6`'s
`k = −1` indexing bug is **unaffected** by any of this and `G13` still applies.

⚠️ **And interpolation of either kind leaves (b)§1.4 intact** — the interpolation is in `x`, the CRPS
integral is in `z`. A convex combination of two step functions **with the same knots** is a step
function with those same knots, so the integrand is still constant on every gap and the corrected
formula applies verbatim with `f_ij := λ f_k + (1−λ) f_{k+1}`. `G3` survives (a convex combination of
two vectors non-decreasing in `j` is non-decreasing in `j`), and so do §2.3's ceiling and §3.5's `MCB`
decomposition, because the interpolated family is still **stochastically monotone in `x`**.

✅ **The clamp is DOUBLY sanctioned**, which strengthens the `Δ` diagnostic rather than weakening it:
eq. (21)'s own boundary case (*"let `F` equal the available bound given by the non-empty set"*) gives
`F_1` below the range and `F_d` above it, identical to the pre-(20) sentence. ⇒ `Δ` is **not an artefact
of a convention we chose** — it is a property of the rule the source prescribes.

ℹ️ A consistency check worth keeping: the paper's fallback for a covariate incomparable to *all*
training points is **the empirical distribution of `y`** — exactly `UNC`'s forecast. It cannot arise
under a total order, so (b)§5 does not need it, but the source's own notion of *"no information"* is the
no-model forecast, which is what §3.4's `DSC = 0` endpoint says too.

⛔⛔ **P6 — the one-character bug.** `k = searchsorted(xs, x_new, side=RIGHT) − 1` returns `k = −1` for
`x_new < xs[1]`, and numpy indexes `−1` as the **LAST** block ⇒ the lowest test cells silently receive
the **highest** block's CDF, the exact inverse of the declared clamp. ⇒ `G13` asserts `0 ≤ k ≤ d−1`
*after* clamping and **counts each end separately** — the two ends mean different things.

### 5.2 ⛔ A single split is the wrong design — LOMO and rolling-origin, both

```
  PROCEDURE lomo(x, y, t, X…):                        # ⭐ every event cell is a test point ONCE
      FOR each month u in 457..469:
          tr <- {i : t[i] != u} ;  te <- {i : t[i] == u}
          model_u <- idr_fit(x[tr], y[tr])            # fold-specific grid AND fold-specific marginal
          accumulate  num += Σ_{i∈te} crps(predict(model_u, x[i]), y[i] ; ỹ^u, w^u)
                      den += Σ_{i∈te} crps(Ĝ^{tr(u)},              y[i] ; ỹ^u, w^u)
      SKILL_lomo <- 1 - num/den                       # pooled per-cell scores, NOT pooled fits

  PROCEDURE rolling_origin(x, y, t, T0):              # ⭐ DEPLOYMENT-shaped
      FOR T = T0 .. 468:  train 457..T, test T+1, accumulate as above
      SKILL_roll <- 1 - num/den
```

⭐ **LOMO pools per-cell test scores across 13 folds, so the pooled test `n` is the full 2,905**, not a
few hundred. ⚠️ But LOMO's training folds contain months *after* the test month ⇒ **not
deployment-shaped**. It answers *"how much signal is there"*. Rolling-origin answers *"what would
deployment get"*.

⭐⭐ **Report both, and the GAP between them is itself a finding** — it measures non-stationarity over
the 13 months, which is a thing we want to know before building `v1`.

⚠️ **LOMO does not remove the dependence**, it only blocks the temporal axis: a cell that is an event
in months `u` and `u+1` sits in train and test with highly autocorrelated `x` and `y`. ℹ️ The leakage
is weaker than in a cell-level model, because IDR learns only a **global** monotone `x → y` map with no
cell identity — but a persistent `x` and a persistent `y` still help it. ⇒ **Add a spatially blocked
variant (leave-one-region-out) as the robustness check.** If month-blocked and space-blocked agree,
dependence is not driving the number. ⭐ That is the cheap version of the whole `B2` problem, for one
extra loop.

### 5.3 ⛔ The tail is a one-cell-leverage statistic. Declare it unevaluable in advance

**The `Δ` relation, re-derived here.** The out-of-range omission is **identical** in numerator and
denominator, because the training marginal `p_tr` also saturates at 1 above `ỹ_m` and at 0 below `ỹ_1`.
With `Δ = (1/|te|) Σ_i [ (y_i − ỹ_m)⁺ + (ỹ_1 − y_i)⁺ ]` and `C`, `U` the uncorrected values:

```
   true SKILL  =  1 − (C+Δ)/(U+Δ)  =  (U−C)/(U+Δ)  =  reported SKILL × U/(U+Δ)
```

⇒ ⭐ **Exact multiplicative shrinkage toward zero.** Reported skill **overstates the magnitude** in
either direction, and ⭐ **cannot flip the sign** — so `SKILL ≤ 0` as a direct negative survives the
bug. `Δ` is driven by the single largest test event's excess over its fold's training maximum, divided
by `|te|`: **one cell can set it.**

⇒ **Report `Δ` and the factor `U/(U+Δ)` as a diagnostic even after the formula is fixed** — it is an
honest measure of how far outside its training range the instrument was asked to work.

⛔⛔ **The interior error does NOT cancel, and its differential is**

```
   error(CRPS) − error(UNC)  =  2 (ỹ_{j+1} − y) ( f_j − p_j )
```

⇒ **exactly proportional to the discrimination at that threshold.** The non-cancelling error is **zero
where the index is useless and grows precisely where it works.** ⚠️ Epistemically worse than the tail
term even if smaller: the tail term has a known sign and an exact correction; this has neither, and it
**scales with the signal being measured.** ⇒ This is why (b)§1.4 replaces the formula everywhere rather
than patching the out-of-range ends.

**Two instruments make the tail visible instead of hidden:**

1. the **out-of-sample threshold curve** — same §3.3 identity, free;
2. a **top-`k` jackknife** — drop the largest event, recompute, report the swing.

⇒ ⭐ **This is how to operationalise `R-66` properly.** Pre-declare not only a minimum test `n` but a
**minimum count of test events above a declared magnitude**, below which the tail claim is reported as
**not evaluable**. ⛔ Without that second number, `R-66` is satisfied by a bulk-only sample.

---

## 6 The gates — revised: three could not fire, and the two that matter most were missing

| id | assertion | class |
|---|---|---|
| `G1` | `DSC_ORD ≥ 0` | ⭐ **theorem** (§3.4) ⇒ a violation is a bug, never a result |
| `G2` | `DSC_ORD ≤ DSC_MAX` **at the LOW rungs** | ⚠️ **moved.** At full resolution `DSC_MAX → UNC_0` and the gate is vacuous; at low rungs the bound binds |
| `G3` | `F[i,·]` non-decreasing in `j`, within `[−tol, 1+tol]` | ⭐ **PROMOTED to theorem class.** Proof, one line from eq. (10): every block average of `1{y_l ≤ z}` is non-decreasing in `z`, and `min`/`max` are monotone ⇒ `F̂_i(ỹ_j)` non-decreasing in `j`. ∎ ⇒ a failure localises to the **solver** (inexact projection, interpolating `predict`, inconsistent tie-breaking), never to `x`. ⚠️ **needs a tolerance** — float PAV can return a `−1e-16` decrease, and an exact-equality gate that fires on noise is worse than none |
| `G4` | `F[i,·] == F[i',·]` when `x[i] == x[i']` | ⚠️ **tautological under (b)§1.2's own implementation** (`f[g[:]]` makes it true by construction). Valuable **only** against the library path (b)§1.3 warns about ⇒ labelled as testing an implementation we did not write |
| ⭐ `G5a` | `x = y` ⇒ `CRPS_ORD == 0.0` **exactly** | blocks of `x=y` are level sets of `y` ⇒ constant responses ⇒ point masses; **and the missing step: feasibility** — those point masses are stochastically increasing because `ỹ_k` is, so the relaxed optimum already satisfies the constraint and **is** the IDR solution. ⇒ `f` is exactly `0.0`/`1.0`, matching the indicator bit for bit ⇒ the sum is exactly `0.0`. ⭐ **Heavy ties in `y` HELP this gate**; `Henzi2021` §3's distinctness condition does not apply. Assert on the primitive, not on `R* == 1` |
| ⭐ `G5b` | `x = −y` ⇒ `DSC_ORD == 0` **within tolerance** | ⚠️ **v1 had the pair the wrong way round.** The projection is the constant weighted mean, which **equals** `p[j]` as a rational but reaches it by a different arithmetic path (`Σ size·r / Σ size` vs `(1/n) Σ 1{·}`) ⇒ zero to within rounding, not bit-identical. Pins break `B3`'s one-sidedness as a live property |
| `G6` | `UNC_0` **bit-identical** across replicates | ⭐ and it must **FAIL** by construction if a scheme ever moves `y` — the check that the scheme moved something real |
| `G7` | `resolution_sweep` monotone non-decreasing in realised `d'`, **pathwise, every replicate** | §4.3's refinement theorem ⇒ a violation falsifies the **derivation** |
| ⭐⭐ `G10` | `x_coarse(r)` **constant on each level set of** `x_coarse(2r)`, and realised `d'` non-decreasing in `r` | ⭐ **NEW.** Nestedness asserted **on the data** rather than inferred from a belief about the cut rule ⇒ makes `G7` a consequence rather than a hope |
| ⭐⭐ `G11` | `CRPS_ORD` recomputed **independently** — per-cell closed form, union grid, **no shared intermediate** — agrees to `tol` | ⭐⭐ **NEW, replacing the decorative `G9`.** `G9` compared `Σw·BS_mg − Σw·BS_fit` against `Σw·(BS_mg − BS_fit)`: the **same expression** over the **same two arrays**, so no code change breaks one without the other ⇒ it could not fire. `G11` is the **only gate that attacks the arithmetic rather than the structure**, and it would have caught `P1`, `P4`, `P5` and the grid-inference siblings |
| ⭐⭐ `G12` | on **deliberately UNEQUAL** block sizes, the weighted and unweighted PAV solutions **provably differ**, and the weighted one is returned | ⭐⭐ **NEW.** The only gate that can see `P4`, the most likely silent wrong number |
| ⭐ `G13` | `0 ≤ k ≤ d−1` after clamping, with the clamp count reported **per end** | ⭐ **NEW.** Catches `P6`. ⚠️ Every other gate but `G8` is in-sample, so nothing else can see it |
| `G8` | train and test month sets disjoint, per fold | the `e1_data` `G-4` pattern |

⛔ **Every gate must be shown to FIRE by a test that breaks the code** — this repo's standing rule, and
the rule that condemned `G9`. ⇒ `G2`, `G4` and `G9` all failed it in v1; two are repositioned and one
is replaced.

---

## 7 Execution order for (c)

```
  0.  ⭐⭐ n_eff FIRST — it is the parameter that SETS the size distortion ((b)§4.4d: n_eff = n/b
      and the distortion is of order b), so it estimates a priori how badly PERMUTE is wrong
      BEFORE one replicate is spent, and says whether any scheme can have power at this n.
      Then m, d, the block-size distribution, and ONE replicate's wall time.
      ⛔ All printed BEFORE anything is scored. All unmeasured. With 2,905 clustered cells the
         effective count may be nearer the number of distinct conflict EPISODES than n; if it is
         small the honest verdict is "not evaluable at this n".
  1.  G5a, G5b, G1, G3, G10, G11, G12, G13 on planted data        # does the instrument work at all
  2.  ⭐ plant a known Δ and recover U/(U+Δ) to machine precision   # tests (b)§5.3's algebra ITSELF
  3.  H-f: the null R* under EXCHANGEABLE planted H_0              # order log n / n
  4.  ⭐⭐ H-e: THE THREE-POINT PATTERN of (b)§4.4c, not one case:
         4a  both fields i.i.d.            -> PERMUTE predicted EXACT
         4b  x smooth, y i.i.d.            -> PERMUTE predicted to UNDER-reject
         4c  both smooth, our geometry      -> PERMUTE predicted to OVER-reject, by order b
  5.      the same under LON_SHIFT, MONTH_SHIFT, STRATIFIED        # which scheme recovers size?
  6.  G7 + the sweep on planted PARTIAL information
  7.  H-c: bootstrap width and the top-k jackknife on a planted tail
  8.  H-g: LOMO and rolling-origin recover a planted skill; and their gap recovers planted drift
```

⛔ **Steps 3 and 4 are the whole point.** Step 3 alone is the **vacuous** test §5.3 identified —
planting exchangeable `H_0` validates a test that is exact under exchangeability *by assumption*.

⭐⭐ **Step 4 is the test of the assumption, and it is now a PATTERN, not a single prediction.** ⛔ A
lone over-rejection at 4c is **no longer sufficient** — the mechanism claim requires 4a exact, 4b
under-rejecting and 4c over-rejecting. That is the *"for the reason I think it does"* standard. ⇒ If the
pattern does not reproduce, break `B2`'s **direction** is unsupported even if its **validity** half
(§4.4a) still stands on logic, and §5 is revised accordingly.

⭐ **Step 2 is new and it tests the auditor, not the data.** `opponent` flagged that the `U/(U+Δ)`
relation is algebra it derived and did not execute. ⇒ Plant a known `Δ`; if the recovered factor is not
`U/(U+Δ)` to machine precision, the corrected formula is wrong and so is the audit.

---

## 8 Free parameters after (b)

1. ⭐ the **resampling scheme** / exchangeability unit — *this is the null* ((b)§4)
2. ⭐ the **sign** of `x` ((b)§2.1)
3. ⭐ the **scale** of `y` — raw or `log1p`; `w` changes, so `R*` changes (§7.2) ⇒ **swept**
4. `coarsen`'s cut rule — equal-count vs equal-width, **both dyadic** ⇒ **swept**
5. ✅ the in-range prediction rule — ⛔ **NO LONGER FREE.** `Henzi2021` §2.5 offers two rules and §2.2's invariance **decides** it: eq. (21)'s midpoint, because the metric rule would make the out-of-sample half of the instrument scale-dependent ((b)§5.1). A **derived** decision, recorded as such
6. the LOMO / rolling-origin split points, and `T0`
7. `α`, `B`, the primary endpoint, and the `p(1−p)` floor for the maxT family
8. the donor tie-breaking rule in `LON_SHIFT`
9. ⭐ the **minimum test-event count above a declared magnitude** for a tail claim ((b)§5.3)

⇒ (c) declares all nine **in writing before it runs**, and sweeps 3 and 4.

---

## 9 What (b) deliberately does NOT contain

- ⛔ **No real data path.** No `mu`, no checkpoint forward. (c) runs on planted data only, so a failure
  is a failure **of the instrument** and cannot be mistaken for a finding.
- ⛔ **No estimator imported from the crossing dossiers** — they answer a different question (§4.3's
  flag) and nothing here should inherit their population.
- ⛔ **No verdict threshold.** What `R*` justifies building `v1` is a decision rule, it belongs in a
  pre-registration, and it is Simon's.

ℹ️ **Audit provenance.** Breaks `P1`–`P10` are `opponent`'s; `P9` and the `library` verdicts are
`library`'s. The per-gap algebra, `U/(U+Δ)`, `G3`'s proof and the `G5a`/`G5b` asymmetry were
re-derived here. ⚠️ `opponent` notes its own `Δ` algebra is underived-in-execution — hence step 2.
