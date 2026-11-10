# 02 — Design: the ranking body and the rank-coupled empirical draw

**Serves:** the mechanism half of Simon's question — *what exactly would a "ranking head" be*, what
it emits, and what it is trained against. Written 2026-11-10.

⛔ This document **specifies**; it does not adjudicate. Per `00 §6`, no sentence here concludes
against B — a conclusion requires a number from a trained ranking head, and none exists.

---

## 1 The two emissions, side by side

Write `p_i` for the occurrence probability at cell `i`, `τ` for the composition threshold
(`ADR-069`), and `S = {i : p_i ≥ τ}` for the **firing set**.

**A — today.** The body is a per-cell negative binomial with head-emitted `(μ_i, θ_i)`:

```
A_i^(s)  =  1{i ∈ S} · NB(μ_i, θ_i)^(s)                          s = 1 … S_draws
```

**B — proposed.** The body emits a single **score** `f_i` with no scale meaning; only its *order*
is used. Magnitude comes from the empirical distribution of what actually happened:

```
let  n = |S|,  and  π : {1…n} → S  the ordering of the firing set by f, ascending
draw z_1 … z_n  ~  Pool,  with replacement;  sort ascending → z_(1) ≤ … ≤ z_(n)

B_{π(k)}^(s)  =  z_(k)                 k = 1 … n          (one ensemble member)
B_i^(s)       =  0                      i ∉ S
```

`Pool` = the fit window's **strictly positive** truth counts. The construction is the **Schaake
shuffle** (Clark 2004) / **ensemble copula coupling** (Schefzik 2013), applied with the body's own
ranking as the coupling index rather than a historical analogue.

⭐ **Why this is the interesting swap.** A's body must answer *how big*, which is the question the
data answers worst — July's measurement put the magnitude ceiling at Spearman 0.303 against
persistence's 0.367, confound-clean. B's body answers *which is bigger*, and the size is supplied by
the empirical record. The body's job changes from estimation to ordering.

⚠️ **The one property that makes B comparable to A at all.** A's point forecast and B's ranking
index can be the *same* quantity, in which case both emissions carry identical ordering information
and differ only in how they spend it. #579 used exactly that construction, which is why its `ΔCRPS`
reduces to `MCB_B − MCB_A` as an identity.

---

## 2 ⭐⭐ The label transform — `r = ceil(log2(1 + y))`

### 2.1 Why a transform is required at all

The loss of §3 consumes **rating levels** `r`, in the graded-relevance sense of information
retrieval. Our truth is **raw fatality counts**. Using counts directly is not merely
ill-conditioned — it is not representable:

| | raw counts | `r = ceil(log2(1 + y))` |
|---|---|---|
| distinct levels, `\|R\|` | 115 | **10** |
| maximum level | 681 | **10** |
| gain weight `2^r − 1` | `2^681 ≈ 1.0 × 10^205` | **1023** |
| in float32 | ⛔ **`inf` for `r ≥ 128`** — **16 cells, 0.2392%** | ✅ finite |
| weight ratio, top cell : 1-fatality cell | ~`10^205` | **1023 : 1** |

Measured on the #579 fixed population — n = 6,689 cell-months, cells with `p ≥ 0.05` on arm
`ctlfullzero_fortytwo`, score months 463–469, truth `lr_sb_best` raw counts, delivered
autoregressive h = 1. Source: `../2026-11-10_rank_coupling_verdict/results/s11_tau_sweep.out:65`
(*"maximum 681 (115 distinct values, 114 gaps)"*); the level and weight columns are reproducible
from `../2026-11-10_rank_coupling_verdict/tools/tau_sweep_decomposition.py`'s `load_folds()`.

⚠️ **This is not a hypothetical overflow.** Training is **float32 throughout, with no mixed
precision**: a search for `autocast`, `GradScaler`, `float16`, `bfloat16` and `set_default_dtype`
across `views_hydranet/` returns zero hits, and the only `backward` is a plain
`w_loss.backward()` (`training_engine.py:1833`). The repo also carries a float32 overflow scar —
gradient norms rising to `9.4e9`, *"float32 gave out at lesson 48"*
(`training_engine.py:612-614`).

### 2.2 ⭐ Why the dyadic grade is principled and not a patch

The standard IR gain convention pairs grades `0…4` with weights `2^r − 1 = 0, 1, 3, 7, 15`. Grade
dyadically and that convention **inverts the transform**:

```
r = ceil(log2(1 + y))        ⇒        2^r − 1  ≈  y
```

Measured over the 1,392 positive cells of the population above:

| `r` | weight `2^r − 1` | counts mapped to it | n |
|---|---|---|---|
| 1 | 1 | 1 | 217 |
| 2 | 3 | 2 – 3 | 247 |
| 3 | 7 | 4 – 7 | 262 |
| 4 | 15 | 8 – 15 | 241 |
| 5 | 31 | 16 – 31 | 196 |
| 6 | 63 | 32 – 63 | 149 |
| 7 | 127 | 64 – 124 | 64 |
| 8 | 255 | 129 – 244 | 13 |
| 10 | 1023 | 547 – 681 | 3 |

`(2^r − 1) / y` over positives: **median 1.348, mean 1.335, range [1.000, 1.984]** — exact where
`y + 1` is a power of two, and never more than 2× otherwise.

⇒ **The gain weight is the death toll, to within a factor of two.** The loss weights each rating
level by roughly the number of people killed in it. That is the justification a reviewer will ask
for, and it is a property of the data meeting the convention, not a choice tuned to taste.

⭐ **Occupancy is near-uniform across seven levels** (217 · 247 · 262 · 241 · 196 · 149 · 64) with
**no population-dependent binning**. Quantile grades would also be balanced — by construction — but
their bin edges shift between folds, so each fold would be a different target. Dyadic edges are
fixed in advance and mean the same thing everywhere.

### 2.3 ⚠️ The level set is declared a priori, never read off the fold

Level **9 is empty** on this population: the observed levels run `0…8`, then `10`. Since `|R|` in
§3's loss counts *observed* levels, reading it per fold would make the normaliser `1/(|R| − 1)`
**drift between folds** — a different objective in each.

⇒ **Declare `R = {0, 1, …, ceil(log2(1 + y_max))}` once**, from the grid's truth maximum, and hold
`|R|` fixed across folds. Empty levels contribute a step with an empty `c_t` and therefore nothing.
This is a `C-351` requirement — the population and the transform are part of the number.

### 2.4 ⭐ The zero block is handled twice over, by construction

Zeros map to `r = 0`, so their gain weight `2^0 − 1` is **exactly 0**; and `r = 0` is the *lowest*
level, which §3's loss drops as its final step. On the full 360×720 land grid zeros are **99.6%** of
cell-months; on the firing-set population measured above they are **79.19%** (5,297 of 6,689). Both
figures are of different populations and both are stated because `C-351` requires it.

⇒ The enormous tied-at-zero block is not something to engineer around. It is zero-gradient under
the loss by two independent mechanisms.

---

## 3 The loss — Zhu & Klabjan 2020, Eq. 2

### 3.1 The objective, verbatim

Transcribed from `~/brain/9_library/papers/Zhu2020_ListwiseUniqueRatings.pdf` (library
`Zhu2020_ListwiseUniqueRatings`, claims `C-4769`–`C-4772`). Let `R` be the rating levels sorted
descending, `c_t` the set of cells at level `r_t`, and `s̃_t` the cells at levels strictly below
`r_t`:

```
                exp(f(d))
P_t(d)  =  ─────────────────────────────────                                        (1)
           exp(f(d)) + Σ_{d' ∈ s̃_t} exp(f(d'))


                   1        |R|−1
L(f; R, D)  =  − ───────  ·   Σ    (2^{r_t} − 1) ·  Σ        ln P_t(d)              (2)
                 |R| − 1     t=1                  d ∈ c_t
```

Two properties quoted from the paper, both load-bearing here:

> *"When all n documents have the same rating, we do not need a ranking function for sorting the
> documents, and thus the loss is 0. Similarly, at step `|R|` all documents in set `s_|R|` have the
> same rating. Therefore we only consider loss from steps 1 to `|R| − 1`."*

⇒ Steps are indexed by **rating level**, not by cell. With ~1,400 positives spread over 9 levels the
loss has **9 steps**, not 1,400 — and the zero level is the dropped final step.

### 3.2 ⭐⭐ Why this family, and not a pairwise loss

This is a **hard theory constraint**, verified against the proofs rather than the abstracts:

| source | result |
|---|---|
`Calauzenes2012_NonExistenceConvexRankingSurrogates`, **Corollary 4** | **No convex surrogate loss is calibrated** with respect to Average Precision, ERR, or **pairwise disagreement** |
same, **Remark 1** | ⭐ Convex calibrated surrogates **do** exist for **DCG** — the explicit exception |
`Duchi2010_ConsistencyRankingAlgorithms`, **Theorems 11–12** | Standard pairwise convex surrogates (hinge, logistic, margin-based) are **inconsistent even in low-noise settings** |
same, **Lemma 7** | finding the optimal `α` for a pairwise loss is **NP-hard** |
same, **Theorem 13 / Corollary 15** | a **value-regularized linear** loss **is** consistent under low noise |

⇒ A RankNet-style pairwise logistic objective is ruled out on theory, not on taste. Eq. 2 is a
**weighted multi-class classification over rating levels** — a listwise objective in the DCG family,
with the exponential gain the exception licenses.

⚠️ **The pair arithmetic says the same thing from the data side.** At 2,905 events in 842,634
cell-months the all-cell pair count decomposes as: zero–zero 352,571,976,856 (dropped);
**zero–positive 2,439,412,745 — 99.83% of surviving pairs**; positive–positive **4,218,060 —
0.17%**. An all-cell pairwise objective is therefore a *detection* objective with magnitude as a
rounding error. Level-indexing sidesteps this: the zero level is one step, not two billion pairs.

⚠️ **And it tells us the cost model does not transfer.** Mohler 2020's `O(M₀·M₁)`
*"approximately linear in practice"* is zero-cells × positive-cells — for us that is the 2.44
billion above, the dominant cost rather than a saving. That claim is about a city grid.

### 3.3 ⛔ Why not a top-k objective, although the obvious reading recommends one

The literature batch's own gap search suggested *"LambdaRank/LambdaLoss with truncated NDCG@k …
the tie block doesn't need to be ranked"*. **The closest published analogue rules that out.**

Mohler, Porter, Carter & LaFree 2020, *Crime Science* 9(3), CrimeRank — verified directly against
the PDF, lines 690–693, because the library's audit of this paper is still open and no claim is
taken from it:

> *"the highest grid cells of CrimeRank contain less incidents compared to methods that use maximum
> likelihood estimation. This is likely due to the fact that PAI is not changed by a re-ordering of
> the top grid cells ranking, but instead is sensitive to cells either being inside or outside of
> the top k."*

⚠️⚠️ **Read that passage in two halves, because only one of them is load-bearing here** — a
distinction forced by the audit that closed on 2026-11-10, whose finding `S-6` is that the library
claim `C-4803` renders the paper's *"likely"* as *"arises because"*, turning a conjecture into a
mechanism. ⇒ This document quotes the **passage**, hedge intact, and takes only the first half:

1. ⭐ **Definitional, not conjectural:** *PAI is unchanged by a re-ordering within the top k.* PAI is
   the fraction of events captured inside a set of `k` cells, so it is a function of **set
   membership** and cannot depend on the order inside the set. This is arithmetic, and it is the
   half our design rests on.
2. ⚠️ **The paper's own conjecture, which we do NOT use:** that this is *"likely"* why CrimeRank's
   very top cells hold fewer incidents than an MLE method's. An empirical pattern with a hedged
   explanation. Nothing here depends on it.

⇒ A top-k metric **cannot see re-ordering inside the top k**, so a head trained on one comes out
deliberately indifferent to it. But B assigns magnitude **by rank within the firing set** (§1): a
membership score gives the coupling nothing to couple on. ⭐ **The objective must reward ordering
*inside* the firing set, which is precisely what level-indexing in Eq. 2 does and what truncation
at `k` destroys.**

### 3.4 What the head emits

One scalar per cell per target: `n_params = 1`, against the NB body's 2. The score is
**scale-free** — only its order is read, by both the loss (Eq. 1 is shift-invariant but not
scale-invariant, so the scale is pinned by the objective's own optimum, not by any calibration
target) and the emission (§1). Nothing downstream interprets `f` as a count.

⇒ This enters through the `ADR-067` registry seam. ⚠️ The seam routes *families*, not loss
*shapes* — see `03` for the dispatch gap that the registry does not close.

---

## 4 ⛔ The ranked list: what set does the loss order?

### 4.1 What a training step actually contains — traced, not assumed

| fact | evidence |
|---|---|
one sample is `[B=1, T_train, C, 32, 32]` — a 32×32 patch carrying the **whole** training time axis | `volume_sampler.py:88` slices `[:, r0:r0+dim, c0:c0+dim, :]`, axis 0 untouched; layout at `volume_handler.py:365-393` |
**`B` is always 1** | `training_engine.py:1750` `get_batch(..., batch_size=1)`; `windows_per_lesson: 3` are *sequential* calls with gradient accumulation, not a batch dimension |
the criterion is called once per target per timestep — `pred_j [1,32,32,n_params]` against `target_j [1,32,32]`, reduced to a scalar | `training_engine.py:904`, `:916-924`; reduction at `nb_core.py:75-79` |
the full field is **never** a torch tensor inside the autograd graph | `to_pytorch` has three call sites; the only full-field one is `hydranet_inference.py:1605` — **inference only** |

⇒ **The ranked list is the 1,024-cell patch-month.** Declared here rather than discovered later.
⚠️ The intuitive repair — concatenate the batch so cross-patch pairs appear — **is not available**:
there is no batch dimension to concatenate.

### 4.2 ⭐ Three facts that make this much less bad than it looks

1. **Patches are activity-biased, not uniform.** The anchor is drawn from a full-grid event-count
   map under `sampling_strategy: 'sigmoid'`, `sampling_steepness: 1.0`, `min_events: 5`
   (`volume_sampler.py:79-81`, `:136`), then jittered. The all-zero patch — whose loss is exactly 0
   by §3.1 — is therefore rare **by construction**, not a coin flip.
2. **`body_supervision: 'all'`** on the live vehicle ⇒ all 1,024 cells reach the criterion, zeros
   included, **with no mask** (`training_engine.py:712-719`, `:966-967`). The zero block is fully
   present, which is exactly the regime §3.1's step truncation is built for.
3. **A graph-connected per-cell `weight` already exists** (`family_loss.py:35-42`,
   `nb_core.py:75-79`, `C-199`) — the seam a level-weighted loss needs, already carrying an
   all-zero-weight case that returns a graph-connected `0` instead of a detached one.

### 4.3 ⚠️ What it costs, stated before the run

A within-patch loss **never sees a pair whose two cells lie in different patches.** A CNN score is
translation-equivariant, so a *pure* local function of local features would sort globally for free.
Recurrent BatchNorm breaks that purity by giving the score a patch-dependent offset — and this repo
has already measured that exact pathology, in BN recalibration fitted on 32×32 windows and applied
to the 360×720 grid.

⇒ **The patch→global ordering gap is a pre-registered falsifier, not an assumption.** The instrument
exists and needs nothing built: `ρ_event`, measured globally over 7 folds. The finish line of
`00 §⭐⭐` is stated in those units for this reason.

---

## 5 ⭐ The emission half: the tie region is the measured cost

Clark et al. 2004 §4c, verified verbatim at lines 828–835 of
`~/brain/9_library/papers/Clark2004_SchaakeShuffle.pdf`:

> *"zero precipitation days will be matched with observed precipitation amounts, and their assignment
> to a given ensemble member is entirely random. **This latter problem is intractable.** In spite of
> such problems with daily precipitation data, however, the Schaake Shuffle does preserve most of the
> spatial correlation structure in precipitation fields and certainly provides a significant
> improvement over the unshuffled ensemble output."*

That is our failure mode, named in 2004, in a dry-basin regime, by the authors of the method. Cells
whose index reads zero get matched to positive amounts and the assignment among them is arbitrary.
**Its size is measurable and was measured** — the `tie@0` column of
`../2026-11-10_rank_coupling_verdict/results/s11_tau_sweep.out`:

| τ | 0.05 | 0.10 | 0.20 | 0.26 | 0.35 | 0.50 | 0.65 | 0.80 |
|---|---|---|---|---|---|---|---|---|
| `tie@0` — share of firing cells tied at index 0 | 0.635 | 0.386 | 0.057 | 0.023 | 0.008 | 0.002 | 0.000 | 0.000 |
| `ΔCRPS` = `CRPS_B − CRPS_A` | +11.977 | +6.055 | +2.814 | +1.981 | +1.292 | +0.716 | +0.352 | +0.157 |
| `MCB_B` | 12.094 | 6.172 | 2.952 | 2.148 | 1.499 | 1.014 | 0.728 | 0.544 |

Spearman(`tie@0`, `ΔCRPS`) = **0.994**, Pearson = **0.981** over the 8 rows.

⭐ **The tie block exists because the index was a count mean** — quantised at 1/16 with an atom at
zero, because A's body draws zero ~57% of the time. A continuous ranking score has **no atom at
zero**, so `tie@0 → 0` by construction. That is an argument for the ranking head drawn from the
literature and our own instrument rather than from intuition.

⛔⛔ **Two honesty constraints on reading that table, stated here so they cannot be dropped later:**

1. **It is an association in a confounded sweep at n = 8, not an identified effect.** τ moves
   `tie@0` *and* the scored population simultaneously — the firing set shrinks 6,689 → 303 across
   the row. Nothing here isolates the tie mechanism from the population change.
2. **There is a floor the tie region does not explain.** At `tie@0 = 0.000`, `MCB_B` is still
   **0.728** and **0.544**. ⇒ The prediction is that a continuous score removes *most* of B's
   disadvantage, **not all of it**, and whatever remains at `tie@0 = 0` is a separate question this
   design does not answer.

---

## 6 ⚠️ Limits of this design, stated before anything is built

1. **No published precedent for the whole construction.** Every Schaake/ECC design in the library
   trains stage 1 with a likelihood or a regression loss — Walz 2024's two-stage U-Net uses a CNN
   with MSE. Training stage 1 with a **ranking** loss and recovering magnitude nonparametrically at
   each rank was **not found** in the literature. The halves are standard; the join is ours.
2. **No published method combines tie-aware Plackett–Luce with a 99.6% tied-at-zero block.** The
   tie-handling papers assume a modest number of rating levels. §2.4 is a construction, not a
   citation.
3. **ECC assumes a perfect model for the rank dependence structure** (`Schefzik2013`), and the
   output ensemble size equals the raw one — **16** for us. Inherited, not solved.
4. **The ordering target is low-magnitude.** `N8b`: for pairs whose smaller event is ≥10 fatalities
   the model's ordering never reaches the 0.70 target. Any pooled ordering figure, including the
   `+0.3127` finish line, is a **low-magnitude** figure, and `02` does not claim otherwise.
5. **A's calibration blocks a proper-score comparison.** `Brocker2009` (`C-52`): proper scores
   cannot rank *miscalibrated* schemes, and A is miscalibrated — pooled `obs/pred` 1.2349–1.6848
   across ten arms. ⇒ Criteria must be **conditional**, never marginal; B is marginally calibrated
   *by construction* so a pooled magnitude check is a no-op in a guardrail's clothing.
6. **Nothing here is magnitude-neutral by accident.** Bai 2023 measured that a standard ranking loss
   improves NDCG and *degrades* regression accuracy. Simon has scoped magnitude out for now — *"Only
   ranking"* — so this is an accepted cost, named, not an oversight.

---

## 7 References

Library claims and local verification. ⚠️ Entries marked **audit pending** supply no claim to this
document; where one is quoted, it was verified directly against the PDF and says so.

| work | id | what this design takes from it |
|---|---|---|
Zhu & Klabjan 2020, *Listwise LTR by Exploring Unique Ratings* | `C-4769`–`C-4772` | §3.1 — the objective, level-indexed, with the step truncation that handles the zero block |
Calauzènes, Usunier & Gallinari 2012, NIPS | — | §3.2 — Cor. 4 (no convex surrogate for AP/ERR/pairwise), Remark 1 (DCG excepted) |
Duchi, Mackey & Jordan 2010, ICML | — | §3.2 — Thms 11–12 inconsistency, Lemma 7 NP-hardness, Thm 13 the consistent alternative |
Clark, Gangopadhyay, Hay, Rajagopalan & Wilby 2004, *J. Hydrometeorology* 5(1) | `C-4746`+ | §5 — the Schaake shuffle, and §4c's *"intractable"* zero-tie direction (quote verified locally, lines 828–835) |
Schefzik, Thorarinsdottir & Gneiting 2013 | — | §1, §6.3 — ECC as the framework, and the perfect-dependence assumption it carries |
Jagerman, Kiseleva & de Rijke 2017, ListPL | `C-4773`–`C-4775` | the cost of mishandling target-side ties: **overfitting**, visible as validation decay while training still improves |
Bai, Jagerman, Qin, Wang, Bendersky & Najork 2023, CIKM | — | §6.6 — the ranking/regression trade-off, measured |
Blondel, Teboul, Berthet & Djolonga 2020, ICML | — | the non-convex differentiable sort, held in reserve as the alternative that sidesteps §3.2 differently |
Mohler, Porter, Carter & LaFree 2020, *Crime Science* 9(3) | ⚠️ **audit pending** | §3.3 — the top-k blindness, **quoted from the PDF directly** (lines 690–693), no library claim taken |
Arnold, Lerch, Dey & Gneiting 2024 | `C-3572`, `C-3574` | the `MCB − DSC + UNC` decomposition behind §5's table |
Bröcker 2009 | `C-52` | §6.5 — proper scores cannot rank miscalibrated schemes |
Gneiting 2011 | `C-3531` | the scoring rule is declared **ex ante**; why `05` is a stub |

**In-repo, re-runnable.** `../2026-11-10_rank_coupling_verdict/tools/tau_sweep_decomposition.py`
(§2, §5 — 37.8 s CPU) · `../2026-11-08_ordering_discrimination/tools/simulate_ordering_discrimination.py`
(the `DSC`/`CRPS_ORD` primitives) · `../2026-11-04_delivered_gate_lookup/tools/e1_data.py` (the
fold loader and the delivered artifacts both depend on).

⚠️ **Noted, not fixed:** the `+8.52` figure that `../2026-11-10_rank_coupling_verdict/03_FINDING.md`
§3 reported as *"B loses to A, 7/7 folds"* is **WITHDRAWN** as an attribution — the number
reproduces, but it belongs to an unswept τ = 0.20 rather than to the emission, as §5's table shows.
