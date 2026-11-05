# Research programme: a hurdle with a rank-coupled positive body

**Date** 2026-11-04 · **Status** specification. Nothing implemented. Three decisions open.
**Relationship to the queue** this is `RP-3`'s direction, re-specified. It is **not** the
2026-11-03 non-parametric programme document, and it is **not** the gate-conditioned lookup of
epic #559's original S5, which is dead.

**Document structure.** 1 motivation, literature and inspiration · 2 open questions and unsettled
decisions · 3 the hard and unambiguous mathematics · 4 simplified pseudo-code · 5 expectations,
hopes and fears.

---

# 1 · Motivation, literature and inspiration

## 1.1 The defect, as measured

The body emits a negative binomial with a conditional mean `mu` and a shape parameter `theta`. It is
wrong in a specific, measured way, and the way it is wrong is not "too narrow" or "too wide".

**Among cells that have any ensemble spread it is over-dispersed in 9 of 10 bins** — `sd/RMSE` of
1.41 / 2.12 / 2.10 / 1.39 / 1.15 across bins 0–8 — **and under-dispersed only in the top spread
bin** (0.48 at 10 bins, 0.39 at 50). Stable at all four bin counts tested (`M90`(2),
`reports/2026-10-29_draw_count_sweep/`).

⇒ The correction the data requires is **non-monotone in the spread**. A single family with one
global shape parameter cannot bend both ways at once. That is the defect, stated without reference
to any remedy.

⚠️ The pooled dispersion ratio of 0.59 that was once quoted is a Simpson artifact: **99.42% of the
population is a point mass** — not a narrow distribution, no distribution at all — and the
under-dispersed top bin dominates the aggregate because event-cell errors are large.

## 1.2 ⭐⭐ The mechanism, and it is the whole reason for this programme

The negative binomial negative log-likelihood of a **zero** observation is

```
NLL(0 | mu, theta) = theta * log(1 + mu/theta)
```

monotone increasing in `theta` for every `mu > 0` (ADR-073 §0, `docs/ADRs/proposed/`). On a cell
whose observation is zero the loss-minimising shape is therefore `theta -> 0`, **at every `mu`, with
no interior optimum.**

**99.66% of cells are zero in every month.** The grid-median fitted `theta` has collapsed to
**3.6e-06**. The conditionally honest `theta` on event cells is **0.4323**
(`reports/2026-10-12_underfire_next_step.md:224` — ⚠️ referred to as `M6` in one dossier's local
numbering, which is not the `RESULTS_LEDGER` `M6`).

⇒ **Five orders of magnitude. The shape parameter is set by the zeros, not by the events.**

### Why that invalidates the existing record rather than merely explaining it

Every parametric remedy tried so far was fitted under this same objective on this same
zero-dominated population, so the mechanism applies to all of them:

- **Epic #230's two-component NB mixture.** Component 2 was alive (median `w|act` 0.71–0.92) with
  mean separation of **14x–690x** between components, **and the magnitude stayed capped.** Both
  components' `theta` were dragged by the same zero mass. ⇒ The result tests *"a mixture under this
  objective did not crack the wall"*, not *"a mixture cannot"*.
- **The ZINB with a hurdle mask.** Gradient starvation on 95% of cells, non-convergence,
  autoregressive divergence. One specific parameterisation failing, not zero-inflation as a concept.
- **The quantile and dial remedies** (`epic #458`, the winsorised τ-pinball dial): rescaling is not
  calibrating; a global point-dial cannot calibrate a heterogeneous body.

⇒ The record contains **no fair test of a flexible positive-count distribution**, because no
candidate was ever fitted where the zeros could not reach its shape.

### And the record already names what is missing

Epic #230's conclusion, after measuring that component 2 was alive and the cap held anyway:
**the gap is tail *shape* (`xi > 0`), not tail location.** Nothing tried since estimates a shape.

## 1.3 What is working, and is therefore worth building on

**The ranking.** On the delivered autoregressive forecast at h1, the gate's average precision is
**0.5667056384496834** over 842,634 cell-months (`exp03_pairs.json`, computed with the frozen ruler;
reproduced bit-exactly 2026-11-04). Against a positive rate of **0.3448%** that is roughly
**165x chance**.

**And placement, not recall, is what the model is for.** `thin_0.75` keeps **97%** of the oracle's
performance after deleting 25% of events, while scrambling positions costs **81%**
(`project_firing_is_not_the_lever`). ⇒ *Recall is free; placement is everything.*

**The gate is calibrated.** Median `|gate - observed|` of **0.0054** over 842,634 cell-months, and
measured again here: expected firing 0.004063 against an observed rate of 0.003448 over the 13
delivered h1 months — a ratio of **1.179**, i.e. it fires about 18% too often, not too rarely.

⇒ So the two halves are in different states. **The gate works. The body's shape is captured by data
the body should never have seen.**

## 1.4 The idea

Separate the two jobs completely, and give the magnitude distribution no access to zeros at all.

- the **gate** alone decides zero versus positive;
- a distribution on the **strictly positive counts** `{1, 2, 3, ...}` alone decides how large;
- that distribution is indexed by an **embedding** of what the network knows about the cell.

⭐⭐ **The structural consequence, which is the point:** a distribution supported on `{1,2,...}`
never sees a zero observation, so **ADR-073's zero-capture mechanism cannot occur.** It is absent by
construction, not mitigated by a regulariser, a bound or a mask.

⇒ This is why the programme is worth running even though the record contains a measured negative for
the nearest prior attempt (§1.6). The negative is about a conditioning variable; this is about where
the zeros go.

## 1.5 Literature and inspiration

### Simon's stated inspirations

> *"pivot away from parametric distributions and the regression heads to non-parametric
> distributions… The motivation was the observation that the ranking, which is learned on the
> regression head, is not bad. So if you could somehow preserve the ranking, but draw from a
> non-parametric distribution that was closely fitted to the empirical, instead of trying to impose
> parametric assumptions, then it would work better."*

Named: isotonic regression / EasyUQ, the analog ensemble, KDE interpolation.

### What the first mechanism actually is, named so reviewers recognise it

Drawing magnitudes from an empirical pool and assigning them to cells **in the model's rank order**
uses the **same coupling step** as the **Schaake shuffle** and **ensemble copula coupling** — rank
reordering that imposes a dependence structure while preserving the marginals exactly.

⛔ **But v1 is NOT a Schaake shuffle, and the distinction matters.** Schaake reorders an *existing
ensemble* to a template taken from *historical observations*; ECC uses the *raw dynamical ensemble*
as the template. ⇒ v1 has **neither an ensemble nor a historical template**: it draws iid from a pool
and orders by a single deterministic score field.

| | does it transfer to v1? |
|---|---|
| `Scheuerer2015` — a pooled score cannot separate a correct coupling from a **random** one | ✅ **yes.** That is a statement about *scoring sensitivity to rank structure*, which is template-agnostic |
| `Sperati2017` — the reordering *"recovers the spatial and temporal correlation"* | ⛔ **no.** That depends on the template being past analysis fields carrying real spatial structure. A score field's rank structure is a different object whose adequacy is an open empirical question |

⇒ ⭐ **v1's theoretical basis is §3.4's P1–P8 and nothing else.** The literature supplies a name for
the operation and a warning about how to score it — not a guarantee that the template is any good.

**Held in the library, verified verbatim 2026-11-04:**

| | |
|---|---|
| `Sperati2017_GriddedAnalogEnsemble` | the mechanism: *"Applying the Schaake Shuffle reordering technique, which **ranks and reorders** AnEn members using the rank structure of past analysis data, effectively recovers the spatial and temporal correlation."* |
| `Scheuerer2015_VariogramScore` | ⭐ the scoring warning. On 11-member wind ensembles the variogram score **clearly separates** ensemble copula coupling from **random** reordering while the energy score cannot — ESS 0.175 random against 0.184 ECC |
| `MacDonald2011` **C-4462** | the tail remedy: a kernel bulk below a threshold `u` with a **generalized Pareto** above, `u` estimated **jointly** with the bandwidth and `(sigma, xi)`, *"avoiding subjective threshold pre-selection and fully accounting for threshold uncertainty"* |
| `Jeworrek2023` **C-4568**, **C-4569** | the cheap tail alternative: members formed from the past analog **error**, which *"enables ensemble members to take values outside the range of the finite archived observational dataset, making them theoretically capable of forecasting record events that traditional analog methods cannot"*. Measured: probability of detection up **~30%** for 90th-percentile and **~60%** for 95th-percentile events |
| `DelleMonache2013` **C-4559** | the window cost: *"Reducing the training period from 12-15 months to 6-9 months degrades AnEn performance more than it degrades LR"* |
| `Walz2026` **C-3605** | why rank tolerances need care: Spearman and Kendall *"cannot attain their maximum value under perfect monotone dependence for non-continuous distributions"* |
| `Henzi2021` **C-4394** | why IDR is a benchmark and not the remedy: it uses only the stochastic order, so it is *"unable to distinguish situations where distributions agree in location but differ in spread or shape"* — which is §1.1's defect exactly |

**Relayed, not held — worth fetching:** Schefzik, Thorarinsdottir & Gneiting (2013), the primary
ECC reference, cited by `Scheuerer2015`; and Clark et al. (2004), the Schaake shuffle original.

## 1.6 Prior art inside this repository, which outranks the literature here

| artifact | what it was | what it showed |
|---|---|---|
| **`RP-3`** / epic **#468**, ladder `L1`–`L5` (#469–#473) | the family-free conditional CDF: bin cells by the model's index, use the empirical outcome distribution per bin | ⛔ **all five findings RETRACTED as forecast results** (`W-107`, `C-399`) — computed on the teacher-forced cube `d4_result.npz`, where the model is handed last month's truth at every step |
| **`M89`** (#489) | the same method on the **delivered** autoregressive forecast | ⛔ **it lost.** Best 0.054232 against the emission's own 0.052109, and worse than no model beyond `B >= 1,000`. ⭐ Mechanism measured: the **composed** index has 1,321 distinct values over 842,634 cell-months and is **exactly 0.0 on 99.6567%** of them, because `threshold_gate` zeroes the draw below tau=0.20. *"A method that works by sorting cannot work on a quantity that does not sort."* |
| **`L4`** | *"the extrapolation limit, measured"* | fit-window max **1,528** against score-window max **2,482** ⇒ the empirical pool's ceiling bit |
| **`L5b`** | the lookup's effective resolution | **~115 levels, not 8,000** |
| **views-baseline `B`, `C`, `E`** (ADR-023, epic #104) | per-cell climatologies, **all three positives-only hurdles**: zero mass from the cell's empirical zero rate, smoothing on the positives only | ⚠️ **nothing scored yet** — built and unit-tested only; evaluation is views-baseline#109. ⭐ One structural lesson from building them: a **per-cell** pool is usually too small — a cell needs ≥2 positives with spread or it falls back, and on pgm most cells are expected to fall back |

⇒ **What this programme inherits:** the instruments (frozen, self-tested) and `M89`'s negative.
**What it changes:** where the zeros live, and which quantity supplies the ordering.

⚠️ `M89`'s negative is **not** evidence against this programme. It measured a *conditioning
variable* whose support had been destroyed by a composition threshold. It says nothing about a
positives-only body.

## 1.7 ⛔ Attribution, stated plainly

- **The direction is Simon's** — the pivot away from parametric bodies, preserving the ranking, and
  the specification in §3 (the hurdle, the embedding, the positives-only body, the rank-coupled
  assignment, and the observation that `h` and `c` are already embeddings).
- **The first mechanism is the literature's**, not mine and not his: it is the Schaake shuffle /
  ECC device, and the binning variant was already built in this repository before either of us
  discussed it (`reports/2026-10-27_family_free_head/family_free_cdf.py`).
- ⚠️ **One correction on record.** I previously presented the binning implementation to Simon as
  "his idea". It is not. His contribution was the direction; the implementation is the repo's and the
  literature's. The conflation is recorded because it caused a week's work to be aimed at the wrong
  half of the model.

---

*Part 1 ends. Parts 2–5 follow.*

# 2 · Open questions and unsettled decisions

Each item gives the decision, the trade-off, a default where I have one, and who decides. Format
follows the repository's own §12 convention.

## 2.0 What is already settled, for reference

| | settled as |
|---|---|
| the shape of the model | hurdle: gate decides zero/positive, a positives-only distribution decides how large |
| the first mechanism | **rank coupling** — draw `n_t` values from the pool, sort, assign in rank order |
| the embedding, v1 | the scalar regression output; the general form keeps `h` and `c` available |
| sequencing | measure first, then **frozen head**, then **retrain** |
| the internal test | ⭐ **v1 against v1 with ranks permuted within the stratum** — identical marginals and totals, differing only in the coupling |
| the stopping rule | if v1 does not beat its own permutation on either score, the ranking carries nothing ⇒ drop it |

---

## 2.1 ⬜⬜ The reference set — window, stratification and tail threshold, decided TOGETHER

**Decision.** What `W` is, whether `z` stratifies the pool, and where the bulk/tail threshold `u`
sits — ⛔ **and these cannot be settled in sequence, because they trade against each other.**

**Measured, on this pull:**

| `W` | events | max | p99 of positives | ≈ exceedances above p99 |
|---|---|---|---|---|
| all history 121–456 | 45,612 | **48,183** | 308 | ~456 |
| last 120 months | 22,310 | 2,342 | 315 | ~223 |
| last 36 months | 8,208 | 1,528 | 314 | ~82 |
| last 6 months | 1,294 | 854 | 251 | ~13 |

⭐⭐ **The p99 barely moves — 251, 314, 315, 308 — while the maximum moves by a factor of 56.**
⇒ The window is almost purely a decision about **tail reach**, not about the bulk.

**The three-way tension, stated plainly:**

- a **narrow `z`** buys conditional relevance and starves the pool;
- a **broad `z`** buys pool size and dilutes the empirical CDF with irrelevant mass;
- a **long `W`** buys tail reach and exceedances for fitting `u`, and mixes conflict regimes;
- a **short `W`** keeps the regime current and caps what can ever be emitted.

⇒ ⭐ **The resolution is not to choose well; it is to split the choice across mechanisms** — the
**bulk conditioned narrowly** and the **tail pooled widely**. That is only available if the tail is a
separate component, which makes 2.2 a prerequisite rather than an enhancement.

### ⭐⭐ And the window carries a MEASURED bias, not just a reach trade-off

Expected top cell when $n^z_t=263$ values are drawn from each pool, against the score window's own
monthly maxima (**547, 638, 244, 99, 681, 148, 138**; mean **356**, median **244**):

| pool | $\mathbb{E}[\text{top cell}]$ | median | $P(\text{top cell} > 356)$ |
|---|---|---|---|
| 36 months | **676** | 639 | **0.893** |
| 120 months | **787** | 714 | 0.886 |

⇒ ⛔ **A longer window does not merely raise the ceiling; it systematically over-predicts the monthly
maximum.** Drawing 263 times from 36 pooled months over-samples the tail relative to one month's
single shot at its own maximum.

⇒ ⭐ **This is the first evidence for any stratification.** Window length alone cannot fix it — a
short window loses the tail entirely — but conditioning $z$ on the **month's own activity level**
addresses it directly, because it makes the pool's tail match the month being forecast. ⚠️ No other
candidate key (region, duration, trajectory) has evidence behind it yet; this one does.

### ⛔⛔ 48,183 WAS CHECKED, and it disqualifies the all-history pool

Measured 2026-11-04. The top of the all-history pool, and the 3x3 neighbourhood of its largest value:

```
  month 246  priogrid 150919  value 48183        [0,     0, 0]
  month 234  priogrid 150919  value 30000        [0, 48183, 0]
  month 230  priogrid 150916  value 15034        [0,     0, 0]
```

That cell has **336 months, 6 nonzero, median 0**, max 48,183, second largest 30,000 — and it is
**ringed by zeros**. ⇒ **That is the signature of `priogrid 148759, month 504` carrying 113,395
fatalities**, which datafactory confirmed as `date_end_only` applied to annual-source events
(`date_prec == 5`): a VIEWSER-parity rule, **not a genuine per-cell count**.

**And a handful of such values carry the entire tail.** Dropping the largest: 48,183 -> 30,000 ->
15,034 -> 13,300 -> 5,184 -> 3,000, while the pool **mean barely moves** (23.96 -> 21.03). The top
values cluster around row 210, col 436-439 — one region, a few cells.

### ⭐⭐ Consequence: the two windows disagree about whether the tail is heavy at all

Generalized Pareto shape fitted over a threshold sweep, 400-replicate bootstrap intervals:

| pool | $\hat\xi$ across thresholds | mean excess | reading |
|---|---|---|---|
| **all history** | **0.71 → 0.61 → 0.59 → 0.57 → 0.60**; CI excludes 0.5 **from above** at the two best-determined thresholds | 143 → 983, rising, **no stable region** | ⛔ $\xi>1/2$ ⇒ **the variance does not exist** — but the tail is **manufactured** |
| **36 months** | 0.44 → 0.33 → 0.16 → **−0.09** → −0.17 | 107 → 251, **flattening** | $\xi\le0$ ⇒ a **bounded** tail |

⇒ This is not a parameter disagreement. ⛔ **The heavy reading rests on summary-event artifacts**,
which the 36-month window happens to exclude because those months are 124–246.

⇒ Three things follow:

1. ⛔ **The all-history pool is unusable as specified.**
2. ⭐ §2.1's argument for a recent window is now **much stronger than "regime currency"** — the long
   window's tail is a data artifact.
3. ⚠️ **P5's ceiling may be far less of a coverage failure than §2.2 claims.** If the real tail is
   bounded, the pool maximum is roughly the truth's maximum. ⛔ We cannot currently tell, because the
   only evidence for a heavy tail is contaminated.

⇒ ⛔⛔ **New prerequisite: the pool needs a summary-event filter before anything is fitted.** See
§2.9.

**Default.** None offered. ⛔ This is the one decision I will not default, because every option
costs something the programme is explicitly about.

**Who decides.** Simon, after 2.2.

**Deadline.** Before any pool is built.

---

## 2.2 ⬜ Does the tail get its own component, and when?

**Decision.** Whether the body is the empirical pool alone (v1) or a bulk-plus-tail mixture (v3).

**The problem is coverage, not calibration.** Under v1,
`max_i y_it <= max P_z` — a **hard ceiling**. If the decision-relevant question is *"could this month
produce 500+ fatalities in cell X"*, and the pool's maximum is 200, the answer is **structurally no**.
No amount of recalibration changes it.

⛔ **And v2 makes it worse, not better:** a neighbourhood is a subset of the pool, so
`max N_k(e_it) <= max P_z`. v2 buys conditioning; it does nothing for coverage.

⚠️ **It is a measured, dodged limitation, not an accepted trade-off.** `L4` is titled *"the
extrapolation limit, measured"* and its fit max of 1,528 sat below a score max of 2,482. And the
split I proposed for the abandoned S5 had fit max 2,482 above score max 681 — which was choosing a
window where the problem could not appear.

**Options.**

| | |
|---|---|
| **v1 only** | accept the ceiling, report it with every verdict, and state the pool maximum beside every tail claim |
| **v3 · extreme-value mixture** | `MacDonald2011` **C-4462**: kernel bulk below `u`, generalized Pareto above, `u` estimated jointly with the bandwidth and `(sigma, xi)`. ⭐ The only candidate that estimates `xi` — which Epic #230 named as the gap |
| **v3' · error-based members** | `Jeworrek2023` **C-4568**: members from the past analog **error**, so values outside the archive become possible. Cheaper. Measured POD +~30% at the 90th and +~60% at the 95th percentile |

**Default.** ⭐ **v1 first with the ceiling reported, v3 as the immediate successor** — because v1 is
the clean one-variable test of the coupling, and a tail component changes the marginal and so would
confound the ablation in 2.0.

**Deadline.** v1 can start without it. The decision is needed before any claim about big events.

---

## 2.3 ⬜ Which quantity supplies the order

**Decision.** Does `rho_t` come from the regression output, the gate, or a combination?

⚠️ **The specification says the regression output, and the measured evidence is uncomfortable.** The
number I have is the **gate's** ranking quality — AP **0.5667** at h1, ~165x chance. For the
regression head's *magnitude* ordering the record's figure is the amount-ceiling result: **Spearman
0.303 among event cells, against persistence at 0.367.**

⇒ If that still holds for this checkpoint, the ordering v1 depends on is **weak and beaten by a dumb
baseline** — and `L5b`'s measured effective resolution of **~115 levels** is consistent with that.

⚠️ There is also a reason to expect it by construction: the regression head's `mu` field was fitted
under the likelihood whose shape parameter collapsed. The ordering came out of a broken objective.

**Options.** regression output alone (as specified) · the gate alone · a product or a learned
combination · the regression output with the gate as a tie-break.

**Default.** ⭐ **Measure before choosing**, and there is a one-number instrument for it.
`Arnold2024_CRPSDecompositions`, read in the PDF: an *"isotonicity-based decomposition of the mean
crps into interpretable components that quantify miscalibration (MCB), discrimination ability
(DSC)"*, with `DSC = 0` for a static forecast.
⇒ **`DSC_ISO` restricted to event cells says whether any discrimination survives recalibration —
before a single magnitude is assigned.** ⭐ It is better founded than Spearman, whose maximum is
**unattainable on a discrete outcome** (`Walz2026` **C-3605**), and it is the cheapest measurement in
the whole programme.

⇒ Compute `DSC_ISO` on event cells **for each candidate order**, with Spearman reported alongside for
continuity with the record. The permutation ablation in 2.0 then measures the ordering's value
*inside the mechanism that uses it*, which is the complementary question.

**Deadline.** Before the first fit. Cheap: the orders are all available from one forward pass.

---

## 2.4 ⬜ The firing surplus and `n_t`

**Decision.** What to do about the gate firing more often than conflict occurs.

**Measured here, 13 delivered h1 months:** expected firing **0.004063** against an observed rate of
**0.003448** — a ratio of **1.179**, i.e. ~263 cells expected to fire per month where ~224 have
conflict.

⚠️ Under rank coupling this has a specific consequence, not a generic one: **it inflates both the
count and the size of the top-ranked cells**, because the rank-`k` cell receives the `k`-th order
statistic of `n_t` draws and `n_t` is too large.

⭐ Partially self-correcting: the surplus cells land at the **bottom** of the order, where the pool's
mass is, so they receive small values.

⛔ **I previously claimed the gate under-fires at 0.42x and that was wrong** — I could not reconstruct
that figure's denominator and should not have used it. Corrected by direct measurement.

**Options.** accept and report · draw `n_t` from a corrected rate rather than the raw Bernoulli ·
recalibrate the gate.

**Default.** **Accept and report for v1.** ⛔ Recalibrating the gate is explicitly out of scope —
the gate is calibrated on its own terms (median `|gate - observed|` **0.0054**) and the body is the
target.

**Deadline.** Before the first verdict, because the surplus enters the total.

---

## 2.5 ⬜ Where per-cell spread should come from

**Decision.** Which sources of randomness are switched on.

Under v1 there are three, and a fourth is optional:

1. the gate's Bernoulli — whether the cell fires at all;
2. the order-statistic sampling;
3. ⭐ **the rank map, which varies across samples even with the regression output fixed**, because
   each gate draw changes the firing set and therefore every cell's rank;
4. optional: redrawing the regression output per sample, which only exists if the head emits draws.

⇒ ⛔ **See §3.4 P6, corrected 2026-11-04.** Two things were wrong here: (3) supplies the **very top
cell nothing at all**, because it always receives $u_{(n)}$; and the top ranks are not
under-dispersed against the pool — **measured, their sd is up to 3.92x the pool's**, while the bulk
is 0.01x. ⇒ The profile is **non-monotone**, which is the shape §1.1 requires.

⚠️ **What is still open and is what §2.5 must decide:** the widening covers only **5 of 263 ranks
(1.9%)**, against `M90`'s under-dispersed bin at 10% of cells with spread. Source (4) — a head
emitting draws — is the candidate for widening it further, which would promote §2.7 ahead of the
frozen-head route. ⇒ **Unmeasured. My `ranks FIXED` ablation was broken** (it reassigned ranks by
cell index rather than holding them), so source (3)'s contribution has not been quantified at all.

**Default.** 1–3 on, 4 reported as a sensitivity if the head emits draws.

**Deadline.** Before the first fit.

---

## 2.6 ⬜ The scoring instrument, and whether FAO-02 is amended

**Decision.** What the verdict is read on.

⭐ The two scores fail differently, which is why both are needed:

| | sees |
|---|---|
| **per-cell CRPS** | the ranking's per-cell effect — a correctly ranked cell receives tail values where a permuted one receives the field marginal |
| **variogram score** | the **joint** dependence, which a pooled per-cell score cannot see. `Scheuerer2015`: energy score 0.175 random against 0.184 coupled — not separated; the variogram score separates them clearly |

⚠️ **The variogram score is not in FAO-02.** As a **diagnostic** it needs nothing. For a **promotion**
decision it needs Simon's amendment, and `M90`'s caveat column is explicit that using
non-FAO-02 diagnostics for selection *"needs Simon's amendment FIRST"*.

⭐ **And there is a proper instrument for the question the firing-union split was asking improperly.**
`Gneiting2011` gives the **threshold decomposition** of the mean CRPS, Eq. (20) — confirmed in the
PDF — which *"can be plotted to diagnose strengths and deficiencies of density forecasting methods
across the distribution, providing region-specific forecast comparison that the scalar CRPS cannot."*
⇒ It gives the whole curve over magnitude thresholds, is proper by construction, and **subsumes any
event/background split as reads off one plot.**

⚠️ Caveat from the same literature: `Lerch2017_ForecastersDilemmaExtremeEvents`, read in the PDF —
weighted rules' benefits *"in terms of power are rather limited, as compared to us[ing]"* unweighted
ones. ⇒ **Propriety, not power.**

**Default — a four-part battery, one of which gates:**

| | role |
|---|---|
| unweighted CRPS on differences, origin-blocked interval | ⭐ **the gate** |
| the **threshold decomposition** curve | the diagnostic: *where* in the magnitude range the coupling helps |
| the **variogram score** | the joint dependence, which no per-cell score sees |
| the firing-union difference | a **paired contrast** only (§5.1 E1) — never an absolute score |

Plus two cheap proper diagnostics, both verified in `Czado2009_PredictiveAssessmentCountData`'s PDF:
the **non-randomised PIT for counts** (*"nonrandomized PIT histogram, which can be interpreted
diagnostically"* — and the right PIT for a discrete outcome), and the **Dawid–Sebastiani score**
(*"a proper alternative"*, depending only on the first two moments — a dispersion-sensitive gate for
a programme whose own ruler sees dispersion in 4 of 20 columns).

Any promotion claim deferred until the amendment question is answered separately.

**Deadline.** Before a promotion claim, not before the first measurement.

---

## 2.7 ⬜ What a retrained head would optimise

**Decision.** Only live once the frozen-head route has reported.

Simon's note names two directions: keep the NB head — in which case every draw gives a new ordering
and spread source (4) above comes for free — or replace it with *"an MSE (or otherwise) optimised
softplus"*, i.e. a head whose only job is to order.

⚠️ **A ranking loss is a third option and is not the same as MSE on a softplus.** MSE still fits a
level; a ranking loss fits only the order, which is all `rho_t` uses.

**Default.** None. Deferred by the sequencing decision in 2.0.

---

## 2.8 ⬜ Dependencies on work elsewhere

| | |
|---|---|
| **views-baseline `B`, `C`, `E`** | scored ~18:00–19:00 today. All three are positives-only hurdles, so they are the closest prior art that will have **evidence** rather than only structure. ⛔ Simon's instruction: hold until they are scored |
| **the delivered-regime fit window** | `M89`: *"the delivered regime has NO fit window — the delivery run produced no month before 457"*. ⚠️ Any pool drawn from months < 457 is a **teacher-forced-era** pool being used on delivered forecasts. Whether that is acceptable is part of 2.1 |
| **#568** | writing the head's shape parameter into the diagnostic dump. ⭐ **No longer a prerequisite for anything** — `/falsify` established that one CPU forward from the committed checkpoint yields `(mu, theta)` in 0.75 s at 1.06 GB. Convenience only |

---

## 2.9 ⬜⬜ The summary-event filter — a new prerequisite

**Decision.** How to identify and handle summary-event assignments in $\mathcal{P}_z$.

⛔ **This was not an open question before 2026-11-04; the measurement in §2.1 created it.** A handful
of cells carry values of 48,183 / 30,000 / 15,034 ringed by zeros, with 6 nonzero months out of 336
— the same signature as the 113,395 at `priogrid 148759, month 504` that datafactory confirmed is
`date_end_only` applied to annual-source events, **not a per-cell count.**

⇒ **Everything downstream depends on it:** $\hat F_z$'s tail, $\xi$, $q^\ast$, P5's ceiling, E4's
over-prediction of the monthly maximum, and whether v3 is needed at all.

**Options.** drop cells matching the signature (isolated, ringed by zeros, few nonzero months) ·
use `date_prec` from the source if it can be obtained · cap at a quantile · restrict $W$ to windows
that happen to exclude them.

⚠️ ⛔ **The last option is what the 36-month window does by accident, and it must not be mistaken for
a decision.**

**Default.** None. ⛔ Simon has taken the summary-event question into views-datafactory; this
programme must **not** invent a filter of its own. ⇒ Until it is answered, every pool figure is
reported **with and without** the top few values, and no $\xi$ is quoted as a measurement.

**Deadline.** Before any tail is fitted. Not before v1, which does not fit a tail.

## 2.10 ⬜ An alternative mechanism I did not list

**Decision.** Whether `Psi` is rank coupling at all, or a conditional transformation model.

`Hothorn2014_ConditionalTransformationModels`, read in the PDF: the partial-transformation
decomposition relaxes the standard model's assumption *"that higher moments are not affected by the"*
covariates, so variance and skewness may depend on `x`; and the model is fitted by *"regularized
optimization of scoring rules for probabilistic forecasts, e.g. the continuous ranked probability
score."*

⇒ ⭐ A **CRPS-fitted** body whose spread can move at fixed location — which is §1.1's defect stated
exactly, and which v1 addresses only indirectly through the pool.

⛔ **It does not displace v1 as the first test**, because v1 is the one-variable version: TRAM changes
the mechanism **and** the fitting objective together, which is the confound that made Epic #230
unreadable.

**Default.** `Psi_tram` named as the principal alternative to v1, scheduled after the ablation
reports. ⚠️ If v1 fails its permutation test, TRAM is the better next step than a retrained head,
because it does not depend on an ordering.

**Deadline.** After the ablation.

## 2.11 Decisions deliberately NOT taken

- ⛔ **No retraining is specified.** The sequencing puts it after the frozen-head result.
- ⛔ **The gate is not touched.** Not its threshold, not its calibration, not its composition.
- ⛔ **FAO-02 is not amended** by this document.
- ⛔ **v2 is specified but not scheduled.** It is named so the mathematics stays general, and because
  adding a feature such as population would flow into `h` and `c` and therefore into the embedding
  with no change to the body's definition.
- ⛔ **The 2026-11-03 programme document is not repaired.** Its contradictions are noted there.

---

*Part 2 ends. Parts 3–5 follow.*

# 3 · The mathematics

⛔ Every symbol is defined in §3.1 before it is used. Where a symbol would have clashed with an
established one it has been renamed, and the rename is noted.

## 3.1 Glossary of symbols

### Indices and sets

| symbol | meaning |
|---|---|
| $i, j$ | units — PRIO-Grid cells |
| $\mathcal{I}$ | the set of units; $N_{\mathcal{I}}=\lvert\mathcal{I}\rvert$ (here 64,818 global `land` cells) |
| $t, \tau$ | times — months. $t$ is a forecast time, $\tau$ a reference time |
| $T$ | the number of months in the record |
| $W$ | the **reference window**: a set of times, $W\subset\{1,\dots,T\}$, ending strictly before the forecast origin |
| $z$ | a **stratum** label; $\mathcal{Z}$ the set of strata |
| $z(i,t)$ | the stratum of unit $i$ at time $t$ |
| $M_t$ | the **firing set** — the units whose gate fired at $t$ |
| $M^z_t$ | the firing set restricted to stratum $z$; $\;n^z_t=\lvert M^z_t\rvert$ |
| $n_t$ | $\lvert M_t\rvert$ |
| $r$ | a **sample index**, $r=1,\dots,R$ — one simulated field per $r$ |

### Data

| symbol | meaning |
|---|---|
| $x_{it}$ | the input data for unit $i$ at time $t$ |
| $o_{it}$ | the **observed** count, $o_{it}\in\{0,1,2,\dots\}$ |
| $y_{it}$ | a **simulated** count. ⛔ $o$ is always observed, $y$ always simulated |
| $y^g_{it}$ | the simulated gate indicator, $\in\{0,1\}$ |
| $y^b_{it}$ | the simulated body magnitude, $\in\{1,2,3,\dots\}$, defined only when $y^g_{it}=1$ |
| $y^{(r)}_{it}$ | the value of $y_{it}$ in sample $r$ |

### Network

| symbol | meaning |
|---|---|
| $f$ | the network — one function, shared over all units and times |
| $\theta$ | its parameters |
| $\pi_{it}$ | the **gate probability**, $\in[0,1]$ |
| $\hat y_{it}$ | the **regression output**, $\in\mathbb{R}^{d_y}$. ⛔ Used only for its ordering |
| $d_y$ | its dimension. $d_y=1$ for a point head; $d_y>1$ if the head emits draws or parameters |
| $h_{it}, c_{it}$ | the hidden and cell state, each $\in\mathbb{R}^{d_h}$, passed to $t+1$ |
| $d_h$ | the state dimension per block; $d_h=16$ for this architecture |

### Embedding

| symbol | meaning |
|---|---|
| $\phi$ | the **embedding map** |
| $e_{it}$ | the **embedding**, $e_{it}=\phi(\hat y_{it},h_{it},c_{it})\in\mathbb{R}^{d}$ |
| $d$ | the embedding dimension |

### Reference set

| symbol | meaning |
|---|---|
| $\mathcal{P}_z$ | the **reference set** for stratum $z$ — a multiset of *(embedding, observed positive count)* pairs |
| $N_z$ | $\lvert\mathcal{P}_z\rvert$ |
| $\mathcal{O}_z$ | the multiset of observed counts in $\mathcal{P}_z$, i.e. $\{\!\{o:(e,o)\in\mathcal{P}_z\}\!\}$ |
| $\hat F_z$ | the empirical CDF of $\mathcal{O}_z$ |
| $\mu_z,\ \sigma^2_z$ | the mean and variance of $\hat F_z$ |
| $\hat F_z^{-1}$ | its quantile function |

### Mechanism

| symbol | meaning |
|---|---|
| $\Psi$ | the **mechanism** — a map from a stratum's firing embeddings and its reference set to a joint draw of magnitudes |
| $g$ | the **ranker**, $g:\mathbb{R}^{d}\to\mathbb{R}$ |
| $\kappa$ | a **tie-break key** on units — any fixed injective map $\mathcal{I}\to\mathbb{R}$, e.g. the unit index |
| $\prec_t$ | the **strict total order** on $M^z_t$ induced by $g$ and $\kappa$ |
| $\rho_t$ | the **rank map**, a bijection $M^z_t\to\{1,\dots,n^z_t\}$ |
| $u_1,\dots,u_{n}$ | the **draws** from $\hat F_z$ |
| $u_{(k)}$ | their **order statistics**, $u_{(1)}\le\dots\le u_{(n)}$ |
| $a(\cdot)$ | an arbitrary real function — used only in P1. ⚠️ Renamed from $g$, which is the ranker |
| $\tilde\rho_t$ | a **uniformly random** bijection $M^z_t\to\{1,\dots,n^z_t\}$ — the permutation control |

### v2 symbols

| symbol | meaning |
|---|---|
| $\mathrm{sim}(\cdot,\cdot)$ | a similarity on $\mathbb{R}^{d}$. ⚠️ Named `sim` rather than $\sigma$, which is a variance here |
| $\nu$ | the **neighbourhood size**. ⚠️ Named $\nu$ rather than $k$, which indexes order statistics |
| $\mathcal{N}_\nu(e)$ | the $\nu$ pairs in $\mathcal{P}_z$ most similar to $e$ |

### v3 symbols

| symbol | meaning |
|---|---|
| $q^\ast$ | the **bulk/tail threshold**. ⚠️ Named $q^\ast$ rather than $u$, which denotes draws |
| $p^\ast$ | the exceedance probability $P(O>q^\ast)$. ⚠️ Named $p^\ast$ rather than $\phi_u$, since $\phi$ is the embedding map |
| $\beta$ | the kernel bandwidth. ⚠️ Named $\beta$ rather than $\lambda$ for readability |
| $B_\beta$ | the **bulk** CDF — a kernel estimate on $\mathcal{O}_z$ with bandwidth $\beta$ |
| $\sigma_{\mathrm{gpd}},\ \xi$ | the generalized Pareto **scale** and **shape** |
| $G$ | the generalized Pareto CDF |

### Terminology

| term | meaning |
|---|---|
| **hurdle** | a two-part model in which one component decides zero versus positive and a second, defined only on the positives, decides how large |
| **gate** | the component emitting $\pi_{it}$ |
| **body** | the component supplying magnitude. ⛔ Here it is supported on $\{1,2,\dots\}$ and never on $0$ |
| **firing set** | $M_t$ — the units whose gate indicator is 1 in a given sample |
| **rank coupling** | assigning a set of drawn magnitudes to units in the units' own rank order. The Schaake-shuffle / ensemble-copula-coupling device |
| **comonotone** | two sequences ordered identically; here, magnitude order equals score order |
| **field** | the whole set of units at one time; a quantity is *field-level* if defined over $M^z_t$ jointly |
| **cell** | one unit at one time; a quantity is *per-cell* if defined for a single $(i,t)$ |
| **pool** | $\mathcal{O}_z$ — the observed positive counts available to draw from |
| **permutation ablation** | replacing $\rho_t$ with $\tilde\rho_t$, holding everything else fixed |
| **plug-in / fair CRPS** | the two CRPS estimators. Defined in `reports/GLOSSARY.md` §4 and not restated here |

---

## 3.2 The model

**Network.** One $f$, parameters $\theta$, shared across all units and times:

$$(\pi_{it},\;\hat y_{it},\;h_{it},\;c_{it}) \;=\; f(x_{it},\;h_{i,t-1},\;c_{i,t-1}\,;\,\theta), \qquad h_{i0}=c_{i0}=0$$

ℹ️ $f$ is convolutional, so it also receives the spatial neighbourhood of $i$, not only $x_{it}$.

**Embedding.** $\;e_{it}=\phi(\hat y_{it},h_{it},c_{it})\in\mathbb{R}^{d}$.

**Gate.** $\;y^g_{it}\sim\mathrm{Bernoulli}(\pi_{it})$ independently across $(i,t)$, and
$$M_t=\{\,j\in\mathcal{I}: y^g_{jt}=1\,\}, \qquad M^z_t=M_t\cap\{i: z(i,t)=z\}, \qquad n^z_t=\lvert M^z_t\rvert$$

**Reference set.**
$$\mathcal{P}_z \;=\; \big\{\!\big(e_{j\tau},\,o_{j\tau}\big) \;:\; o_{j\tau}\ge 1,\;\; \tau\in W,\;\; z(j,\tau)=z \big\}$$

**Mechanism.** Applied **within each stratum** and **jointly** over that stratum's firing set:
$$\big(y^b_{it}\big)_{i\in M^z_t} \;=\; \Psi\big((e_{it})_{i\in M^z_t},\;\mathcal{P}_z\big)$$

**Conditional independence, stated correctly.** Conditional on the gate realisation $M_t$, the
mechanism $\Psi$ consumes only randomness independent of $\{y^g_{it}\}$.
⛔ $y^b_{it}$ is **not** unconditionally independent of $y^g$: it is undefined when $y^g_{it}=0$, and
its law depends on $n^z_t$, which is a function of the other units' gate draws.

**Outcome.**
$$y_{it}=\begin{cases}0, & y^g_{it}=0\\[2pt] y^b_{it}, & y^g_{it}=1\end{cases}$$

**Implied marginal.**
$$P(y_{it}=0)=1-\pi_{it}, \qquad P(y_{it}=k)=\pi_{it}\cdot P\big(y^b_{it}=k \,\big|\, y^g_{it}=1\big),\quad k\ge1$$

⚠️ The conditional on the right **averages over the other units' gate draws**, through $n^z_t$. It is
an *induced* marginal, not a chosen family, and it equals a stated distribution $D(e_{it})$ only when
$\Psi$ factorises over units — which v1 does not.

---

## 3.3 Instance v1 — rank coupling

**Embedding.** $\;\phi_1(\hat y,h,c)=\hat y$, so $d=d_y$.

**Ranker.** $\;g:\mathbb{R}^{d}\to\mathbb{R}$, with $g=\mathrm{id}$ when $d_y=1$.

**Strict total order on $M^z_t$.**
$$i\prec_t j \iff g(e_{it})<g(e_{jt}) \;\;\text{or}\;\; \big(g(e_{it})=g(e_{jt}) \;\text{and}\; \kappa(i)<\kappa(j)\big)$$

**Rank map.** $\;\rho_t(i)=1+\lvert\{\,j\in M^z_t: j\prec_t i\,\}\rvert$, a bijection onto $\{1,\dots,n^z_t\}$.

**Draw.** $\;u_1,\dots,u_{n^z_t}\overset{\text{iid}}{\sim}\hat F_z$, with order statistics $u_{(1)}\le\dots\le u_{(n^z_t)}$.

**Couple.** $\;y^b_{it}=u_{(\rho_t(i))}$.

**Sampling.** One sample $r$ draws $\{y^{g,(r)}_{it}\}$, forms $M^{z,(r)}_t$, recomputes $\rho^{(r)}_t$
on that set, draws $\{u^{(r)}_k\}$, and emits $y^{(r)}_{it}$. ⛔ **The per-cell predictive
distribution is the empirical distribution of $\{y^{(r)}_{it}\}_{r=1}^{R}$** — it is not available in
closed form and must be obtained by repeating the whole field draw.

## 3.4 Properties of v1

All statements are **per stratum** and conditional on $n^z_t$ unless stated otherwise.

**P1 · Field marginal is exact — and that is a WEAK property, obtained trivially.**
$$\{\!\{\,y_{it}: i\in M^z_t\,\}\!\} \;=\; \{\!\{\,u_1,\dots,u_{n^z_t}\,\}\!\} \quad\Longrightarrow\quad \sum_{i\in M^z_t} a(y_{it}) \;=\; \sum_{k=1}^{n^z_t} a(u_k)\;\;\forall a$$
⇒ The stratum's realised magnitudes **are** an $n^z_t$-sample from $\hat F_z$.

⚠️ **This is *marginal* calibration, and v1 gets it by construction rather than by being right.**
`Gneiting2023_ConditionalCalibration`, **Theorem 2.3** read in the PDF: *"Auto-calibration implies
marginal and probabilistic calibration"*, with auto-calibration *"being the strongest notion"*.
⛔ **It is not the "weakest" notion in a chain**, because the same paper exhibits a forecast that
*"is probabilistically calibrated but fails to be marginally calibrated"* **and** one that *"is
marginally calibrated but fails to be probabilistically calibrated"* — the two are **incomparable**.
⇒ P1 therefore buys one of two incomparable weak notions, for free, and says **nothing** about
auto-calibration, which is the standard that matters for a covariate-indexed forecast.

**P2 · Total is unbiased, not exact.**
$$\mathbb{E}\Big[\sum_{i\in M^z_t} y_{it} \,\Big|\, n^z_t\Big]=n^z_t\mu_z, \qquad \mathrm{Var}\Big[\sum_{i\in M^z_t} y_{it} \,\Big|\, n^z_t\Big]=n^z_t\sigma^2_z$$
⛔ The **realisation** is a random sum of heavy-tailed draws, dominated by $\max_k u_k$. With a long
$W$ a single unit can exceed a typical month's whole total.

**P3 · Per-cell law is an order statistic, and depends on other cells.**
$$y_{it}\,\big|\,\rho_t(i)=k,\ n^z_t \;\sim\; u_{(k)} \;=\; \text{the }k\text{-th order statistic of }n^z_t\text{ draws from }\hat F_z$$
⛔ Not $\hat F_z$. ⚠️ And it depends on $n^z_t$, hence on the gate's realisation at **other** units:
a larger firing set raises the top-ranked unit's magnitude.

**P4 · Comonotone within stratum.** $\;i\prec_t j \Rightarrow y_{it}\le y_{jt}$ for $i,j\in M^z_t$.
The ordering is preserved exactly.

**P5 · Hard ceiling.** $\;\max_{i\in M^z_t} y_{it}\le\max\mathcal{O}_z$.
⛔ Nothing above the pool's maximum is emitable, at any sample size. This is **coverage, not
calibration**: no recalibration can produce a value the pool does not contain.

**P6 · Sources of spread.** Three, with a fourth optional:
1. the gate's Bernoulli — whether $i\in M^z_t$;
2. the order-statistic sampling — $u_{(k)}$ varies across $r$;
3. ⭐ the **rank map**, which varies across $r$ even with $\hat y_{it}$ fixed, because $M^z_t$ and
   hence $\rho_t$ change with the gate draw;
4. optional: redrawing $\hat y_{it}$ per sample, available only if $d_y>1$.

⇒ ⛔⛔ **CORRECTED 2026-11-04, and the correction runs in the programme's favour.** The earlier
claim — *"high-ranked units are somewhat under-dispersed, but less so than a fixed-rank account
implies, because of (3)"* — was **wrong in sign at the top ranks, and wrong about (3) for the very
top cell**, which always receives $u_{(n)}$ and so gets no variation from the rank map at all.

**Measured on the real 36-month pool** (8,208 positives, sd **68.90**), $n^z_t=263$, 20,000
replicates — the sd of each order statistic against the pool's own sd:

| rank $k$ | 26 | 132 | 237 | 258 | 261 | 263 |
|---|---|---|---|---|---|---|
| $\mathrm{sd}(u_{(k)})$ | 0.01 | 0.80 | 10.6 | 60.9 | 135.0 | **269.9** |
| ratio to pool sd | 0.000 | 0.012 | 0.154 | 0.884 | **1.96** | **3.92** |

⇒ ⭐ **v1's per-cell dispersion is NON-MONOTONE against the pool: far sharper in the bulk, wider at
the top** — which is the shape §1.1 says the data requires, and the opposite of a monotone
sharpening.

⚠️ **But the widening is narrow.** The sd exceeds the pool's for ranks **259–263 only — 5 of 263,
1.9%** of the firing set — while `M90`'s under-dispersed bin is 10% of the cells that have spread.
⇒ The mechanism has the **right shape**; whether it has the right **extent** is unmeasured.

⚠️ And "under-dispersed" must always say *relative to what*: against the **pool** v1 widens at the
top; against each cell's **own error** (M90's `sd/RMSE`) the top bin still sits at 0.79, i.e. below
1. Both are true and they are different statements.

**P7 · Relation to quantile matching — MEASURED, and the error changes sign.**
$$\mathbb{E}\big[u_{(k)}\big] \;\approx\; \hat F_z^{-1}\!\big(k/(n^z_t+1)\big)$$
⇒ v1 **is** quantile matching in expectation, with the sampling variability retained rather than
averaged away.

⭐ Measured 2026-11-04 on the 36-month pool (8,208 positives, median 6, p99 314, max 1,528, 325
distinct values, **36.8% of mass at $\le 3$**), $n^z_t=263$, 20,000 replicates:

| rank $k$ | $\mathbb{E}[u_{(k)}]$ | $\hat F_z^{-1}(k/(n{+}1))$ | ratio |
|---|---|---|---|
| 50 | 1.70 | 2.00 | **0.85** |
| 132 | 5.87 | 6.00 | 0.98 |
| 255 | 152.4 | 145.0 | 1.05 |
| 262 | 458.0 | 384.8 | **1.19** |
| 263 | 676.4 | 567.6 | **1.19** |

⇒ **~19% low at the top three ranks** — i.e. the approximation *understates* $\mathbb{E}[u_{(k)}]$
there. ⚠️ **And the error changes sign**: it *overstates* in the lower bulk (0.85 at rank 50), because
36.8% of the pool is tied at $\le 3$. The approximation is therefore **not** uniformly conservative
in either direction, and the Blom form is derived for continuous distributions, which this is not.

ℹ️ Field total: $\mathbb{E}[\sum_k u_{(k)}]=6{,}581$ against 6,134 for the quantile rule, while
$n^z_t\mu_z=6{,}584$ — confirming P2 numerically to 0.05%.

**P8 · Zero capture cannot occur.** $\hat F_z$ is supported on $\{1,2,\dots\}$ by construction, so no
zero observation enters it. ⭐ ADR-073 §0's mechanism is **structurally absent**, not bounded,
masked or regularised away.

## 3.5 The permutation ablation

Replace $\rho_t$ by $\tilde\rho_t$, a uniformly random bijection $M^z_t\to\{1,\dots,n^z_t\}$ drawn
independently of everything else, and change nothing else.

**A1 · The control has the same field marginal and the same total law.** $\tilde\rho_t$ is a
bijection, so P1 and P2 hold verbatim.

**A2 · The control's per-cell law is the pool itself.** Averaging over $\tilde\rho_t$,
$$P\big(y_{it}=k \,\big|\, i\in M^z_t,\ n^z_t\big) \;=\; \frac{1}{n^z_t}\sum_{\ell=1}^{n^z_t} P\big(u_{(\ell)}=k\big) \;=\; \hat F_z(k)-\hat F_z(k^-)$$
⇒ **Each unit receives exactly the pool's marginal.** Comonotonicity (P4) is destroyed.

⭐⭐ **Therefore v1 versus the permutation control differs in the coupling and in nothing else** —
same mechanism, same pool, same gate, same marginals, same totals. It is a one-variable contrast in
the strictest available sense.

---

## 3.6 Instance v2 — embedding neighbourhood (specified, not scheduled)

$$\phi_2(\hat y,h,c)=(\hat y,h,c), \qquad d=d_y+2d_h$$
$$\mathcal{N}_\nu(e_{it})=\text{the }\nu\text{ pairs of }\mathcal{P}_{z(i,t)}\text{ maximising }\mathrm{sim}(e,e_{it}), \qquad y^b_{it}\sim\text{emp}\big\{\!\big\{o:(e,o)\in\mathcal{N}_\nu(e_{it})\big\}\!\big\}$$

$\Psi$ now **factorises over units**.

⚠️ **Loses P1, P2 and P4.** ⛔ **Worsens P5:** $\mathcal{N}_\nu(e_{it})\subseteq\mathcal{P}_z$ so
$\max\mathcal{N}_\nu\le\max\mathcal{O}_z$, strictly for most units.
⚠️ **The stored $e_{j\tau}$ are in-sample** — produced by a network fitted on those same units — so
neighbours appear more informative than they will at forecast time; and $\mathcal{P}_z$ must be
recomputed whenever $\theta$ changes.
⭐⭐ **Storage and search re-measured 2026-11-04, and the earlier figures were wrong by ~1000x.**
$\mathcal{P}_z$ holds **positives only** (§3.2), not all cell-months — and the state splits as
`split_h = channels/8` into `hs_1..hs_4 | hl_1..hl_4`, confirmed at
`HydraBNrecurrentUnet_06_LSTM4.py:574-575`, so $d_h=16$ and $d=d_y+32$.

| | |
|---|---|
| $d$ | **33** with a scalar head ($d_y=1$); **48** with a 16-draw head. ⛔ Not 46 |
| pairs in $\mathcal{P}_z$ | **45,612** all-history · 22,310 at 120 months · 8,208 at 36 months |
| storage at $d=33$ | **6.02 MB** · 2.94 MB · 1.08 MB |
| brute-force search, 842,634 queries | **50.7 s** · 24.8 s · 9.1 s |

⇒ ⛔ The previously stated *"6 GB, ANN search in minutes"* was computed over all 64,818 x 500
cell-months rather than over the positives. ⭐ **v2 is essentially free and can run alongside v1**;
no approximate-nearest-neighbour index is needed at all.

## 3.7 Instance v3 — bulk plus generalized Pareto tail (the remedy for P5)

Replace $\hat F_z$ by
$$\hat F^{\mathrm{v3}}_z(x)=\begin{cases}(1-p^\ast)\,\dfrac{B_\beta(x)}{B_\beta(q^\ast)}, & x\le q^\ast\\[8pt] (1-p^\ast)+p^\ast\,G\big(x\mid q^\ast,\sigma_{\mathrm{gpd}},\xi\big), & x>q^\ast\end{cases}$$

with $q^\ast$ estimated **jointly** with $\beta$ and $(\sigma_{\mathrm{gpd}},\xi)$
(`MacDonald2011` **C-4462**).

⇒ **P5 becomes unbounded above.** P1 and P4 survive; P2's variance grows and must be reported;
P7's upward bias at the top ranks is then a property of the fitted tail rather than of a finite pool.
⭐ $\xi$ is the quantity Epic #230 named as the gap.

⚠️ Fitting $\xi$ needs exceedances, and the count is window-dependent: ~456 above the pool's p99
for all history, ~82 for 36 months, ~13 for 6 months.

### Discretisation — a bulk problem, not a tail problem

$\hat F^{\mathrm{v3}}_z$ is continuous and the outcome must lie in $\{1,2,\dots\}$, so a draw must be
mapped to an integer.

⭐ **The convention is set by the repository, not by argument:** views-baseline's `B` rounds half-up
to an integer $\ge 1$ and `C` reflects at 0.5 and rounds to $\ge 1$, with the bias already
characterised — *reflection moves mass from 1 onto 2, 3, …, so $P(Y=1\mid Y\ge1)$ falls.* Matching
them is what makes a comparison against those baselines meaningful.

⭐⭐ **And $\xi$ does not absorb discretisation bias, because discretisation does not reach the tail.**
$q^\ast$ sits near the pool's p99 = **314**; rounding 412.7 to 413 changes nothing about a GPD fitted
there. The bias lives where the mass is — **36.8% of the pool is at $\le 3$**.

⇒ **Therefore a simpler v3 than MacDonald's is preferable here:**

$$\hat F^{\mathrm{v3'}}_z(x)=\begin{cases}\text{the EMPIRICAL CDF of }\{o\in\mathcal{O}_z: o\le q^\ast\}, & x\le q^\ast\\[4pt] (1-p^\ast)+p^\ast\,G\big(x\mid q^\ast,\sigma_{\mathrm{gpd}},\xi\big), & x>q^\ast\end{cases}$$

⇒ The bulk stays **discrete and is never rounded at all**; rounding applies only above $q^\ast$, where
it is immaterial. ⚠️ Cost: there is no bandwidth to estimate jointly, so `MacDonald2011`'s joint
treatment reduces to estimating $q^\ast$ and $(\sigma_{\mathrm{gpd}},\xi)$ together.

### Q-E · the four discretisation candidates, priced against the three requirements

Requirements: **P8 recovered** (support on $\{1,2,\dots\}$), **P5 unbounded above**, $\xi$
**identifiable**.

| candidate | P8 | P5 | $\xi$ identifiable | verdict |
|---|---|---|---|---|
| **truncate and renormalise to $[1,\infty)$** | ✅ | ✅ | ✅ | ⚠️ works, but renormalising a kernel bulk at 1 piles mass on the first integers — the bias `C` already measured |
| **difference the CDF on integer cells**, $P(k)=\hat F(k+\tfrac12)-\hat F(k-\tfrac12)$ | ✅ with first cell $[1,\tfrac32)$ | ✅ | ✅ | ⭐ principled, and the natural discretisation of a continuous fit |
| **fit the kernel on $\log y$ or on $y-1$** | ✅ | ✅ | ⛔ $\xi$ becomes the shape of the **transformed** tail, and $\log$ makes a heavy tail light | ⛔ changes the quantity estimated |
| **a discrete heavy tail** (zeta / discrete Pareto) instead of the GPD | ✅ | ✅ | ⚠️ a different parameter, not comparable to `MacDonald2011`'s $\xi$ | worth knowing; not the first move |

⭐⭐ **v3′ removes the question for the bulk entirely** — the bulk is never continuous, so there is no
renormalisation, no kernel and no mass below 1. ⇒ Only GPD draws above $q^\ast$ need discretising,
and there the **difference-the-CDF** rule is exact and immaterial in magnitude.

⛔ **MacDonald's model as imported is a live hazard**: a continuous kernel bulk fitted to a pool whose
mass sits at 1–3 places real probability **at and below zero**, reintroducing the zero surface P8
exists to remove. ⇒ **v3′ is not a simplification of taste; it is the version that preserves P8.**

### What rounding does and does not break

| | |
|---|---|
| **P1 · exact** | rounding is applied elementwise to the drawn multiset, so the multiset identity survives verbatim |
| **P4 · exact** | rounding is **monotone**, so $u_{(1)}\le\dots\le u_{(n)}$ implies the rounded sequence is still sorted |
| **P2 · ⛔ loses unbiasedness** | $\mathbb{E}[\sum_k \mathrm{round}(u_k)] = n^z_t\,\mathbb{E}[\mathrm{round}(U)] \ne n^z_t\mu_z$, with bias $n^z_t\big(\mathbb{E}[\mathrm{round}(U)]-\mathbb{E}[U]\big)$ — positive under round-half-up-with-floor-1, and concentrated in the bulk |

ℹ️ Ties created by rounding are **not** a problem for the assignment: $\rho_t$ ranks on the **scores**,
not on the magnitudes, so two units may legitimately receive the same value.

---

## 3.8 Q-F · Property x instance matrix

⛔ §3.6 and §3.7 each audited a *selected* subset — the same bug twice: a property proved for
$\hat F_z$ assumed to transfer through a **replacement** of $\hat F_z$. This is the complete grid.

| | v1 | permutation control | v2 | v3′ |
|---|---|---|---|---|
| **P1** field marginal exact | ✅ proved §3.4 | ✅ proved A1 — $\tilde\rho_t$ is a bijection | ⛔ **false** — each unit draws from its own $\mathcal{N}_\nu$ | ✅ holds — still $n^z_t$ iid values from one law |
| **P2** total unbiased | ✅ $n^z_t\mu_z$ | ✅ identical | ⛔ **false** — no common pool | ⚠️ **conditional.** Mean exists iff $\xi<1$; **variance iff $\xi<1/2$** — which the all-history fit violates ($\hat\xi\approx0.6$, CI excluding $1/2$) |
| **P3** per-cell law is an order statistic | ✅ | ⚠️ **changes** — averaging over $\tilde\rho_t$ gives exactly $\hat F_z$ (A2) | ⛔ **false** — it is the neighbourhood's empirical law | ✅ same form, with $\hat F^{v3'}_z$ |
| **P4** comonotone | ✅ | ⛔ **destroyed by design** — that is the ablation | ⛔ **false** — no ordering exists | ✅ rounding is monotone, so it survives |
| **P5** hard ceiling | ✅ $\max\mathcal{O}_z$ | ✅ identical | ⛔ **WORSE** — $\max\mathcal{N}_\nu\le\max\mathcal{O}_z$ | ⭐ **removed**, unbounded above |
| **P6** three spread sources | ✅ | ⚠️ (3) becomes a *uniform* rank draw, not a gate-induced one | ⚠️ only (1) plus the neighbourhood draw; no rank source | ✅ unchanged, plus the tail's variability |
| **P7** quantile-matching relation | ✅ measured: 1.19 at the top, 0.85 at rank 50 | ⛔ not applicable — no rank-to-quantile correspondence | ⛔ not applicable | ⚠️ **open** — derived for an empirical $\hat F_z$; unmeasured under a fitted tail |
| **P8** zero capture cannot occur | ✅ proved | ✅ identical support | ✅ a subset of the positives | ⛔⛔ **AT RISK, and this is the finding.** A continuous bulk places mass at and below 0. ⭐ **v3′ recovers it**; MacDonald as imported does **not** |

⭐⭐ **Both findings are in the P2 and P8 rows, and both are about v3:**

1. **P2's variance clause is conditional on $\xi<1/2$**, which the all-history fit violates ⇒ §3.4
   must carry the condition, with a trimmed moment reported where it fails.
2. **P8 — the property §1.4 calls the point — is the one §3.7 was silent about**, and it is exactly
   the one a continuous bulk breaks.

⚠️ **Two cells are open rather than proved:** P7 under a fitted tail, and P6's source (3) under the
permutation control. Both are measurable in the simulation that answers Q-B/C/D.

---

*Part 3 ends. Parts 4–5 follow.*

# 4 · Simplified pseudo-code

⛔ Deliberately simplified. No leakage checks, no gates, no strata loop, no I/O, no estimator choice.
Those belong in the instrument, not here. One stratum and one target are assumed throughout.

## A · Build the pool — once, from the reference window

```
POOL = []
for tau in W:                       # W ends strictly before the forecast origin
    for j in UNITS:
        if o[j, tau] >= 1:          # ⛔ POSITIVES ONLY. This is the whole point.
            POOL.append( o[j, tau] )
```

⇒ `POOL` is a flat list of observed positive counts. **No zero ever enters it**, so nothing in it can
pull a shape parameter toward zero.

## B · One sample of one month

```
function sample_month(t, POOL, pi, yhat):

    # 1 · the gate decides WHO fires, and nothing else
    fired = [ i for i in UNITS if coin_flip(pi[i, t]) == 1 ]
    n     = length(fired)

    # 2 · the model decides the ORDER, and nothing else
    order = sort(fired, key = yhat[i, t], tiebreak = unit_index(i))   # ascending

    # 3 · the pool decides the MAGNITUDES, and nothing else
    draws = sample_with_replacement(POOL, n)
    sort(draws)                                                      # ascending

    # 4 · couple them: k-th smallest magnitude to k-th smallest score
    y = zeros(UNITS)
    for k in 1..n:
        y[ order[k] ] = draws[k]

    return y
```

⭐ Read step 2 and step 3 together: **the model never says how large, and the pool never says where.**

## C · Many samples — because the per-cell distribution has no closed form

```
FIELDS = []
for r in 1..R:
    FIELDS.append( sample_month(t, POOL, pi, yhat) )

# the predictive distribution for one cell is its own column across samples
function predictive(i):
    return [ FIELDS[r][i] for r in 1..R ]
```

⚠️ This is not an implementation detail. Because the magnitudes are assigned **jointly** across the
firing set, a cell's distribution cannot be written down — it has to be accumulated over whole-field
draws. Everything downstream is shaped by that.

## D · The ablation — the control differs by ONE line

```
function sample_month_PERMUTED(t, POOL, pi, yhat):
    fired = [ i for i in UNITS if coin_flip(pi[i, t]) == 1 ]
    n     = length(fired)

    order = shuffle(fired)          # ⛔ THE ONLY CHANGE: random order, not the model's
                                    #    (was: sort by yhat)

    draws = sample_with_replacement(POOL, n)
    sort(draws)

    y = zeros(UNITS)
    for k in 1..n:
        y[ order[k] ] = draws[k]
    return y
```

⭐⭐ Same gate draws, same pool, same number of firing cells, same drawn magnitudes, **same field
total.** The only difference is which cell gets which number. ⇒ Any score difference between C and D
is attributable to the coupling and to nothing else.

ℹ️ To make the comparison tighter still, use **common random numbers**: give both arms the same
gate flips and the same `draws` in each sample $r$, so the two differ only in `order`.

## E · Score

```
for each arm in { v1, permuted }:
    for each cell (i, t) in the scoring window:
        crps[arm][i,t] = CRPS( predictive_distribution(i, t, arm), o[i, t] )

    per_cell[arm] = mean over cells of crps[arm]          # sees the ranking's per-cell effect
    vs[arm]       = variogram_score( FIELDS[arm], o )     # sees the JOINT dependence

verdict:
    if per_cell[v1] < per_cell[permuted]  →  the coupling helps, and CRPS can see it
    if only vs[v1]  <  vs[permuted]       →  the coupling helps, and CRPS is the blind instrument
    if neither                            →  ⛔ the ordering carries nothing. Stop.
```

## F · What v3 changes — two lines

```
# A, replaced:
FIT = fit_bulk_and_tail(POOL)        # kernel bulk below q*, generalized Pareto above;
                                     # q*, bandwidth, scale and shape estimated JOINTLY

# B step 3, replaced:
draws = sample_from(FIT, n)          # ⇒ can now exceed max(POOL)
```

Everything else is untouched. ⭐ That is the payoff of keeping the mechanism a slot: the remedy for
the hard ceiling is a change of two lines, not a change of design.

## G · What v2 changes — and why it is a different shape

```
# B steps 2-4, replaced entirely:
y = zeros(UNITS)
for i in fired:
    neighbours = nearest( POOL_WITH_EMBEDDINGS, e[i, t], nu )
    y[i]       = sample_one( outcomes_of(neighbours) )
```

⚠️ There is no `order` and no shared `draws`, so each cell is independent. ⇒ P1, P2 and P4 are gone
and the hard ceiling is **lower**, because a neighbourhood is a subset of the pool.

---

## What this pseudo-code deliberately omits

| omitted | belongs in |
|---|---|
| the leakage check — that no scoring month entered `POOL` | the instrument, as a gate that can fire |
| the stratum loop and within-stratum ranking | §3.2, once `z` is decided |
| plug-in versus fair CRPS | `reports/GLOSSARY.md` §4 |
| how `pi` and `yhat` are obtained | one CPU forward per origin from the committed checkpoint |
| the choice of `R` | a convergence check, not a free parameter |
| reporting `max(POOL)` beside every tail claim | mandatory, per P5 |

---

*Part 4 ends. Part 5 follows.*

# 5 · Expectations, hopes and fears

⛔ Written before anything is implemented. These become numbered `P-` predictions when the
pre-registration is written; until then they are stated here so they cannot be revised silently.

## 5.1 Expectations

**E1 · The coupling beats its own permutation, but the signal is diluted ~250x.**
⚠️ Both arms give **0 to every non-firing cell**, so they are identical on 99.6% of the population by
construction. The expected firing set is ~**263** cells per month, so over 7 score months the
difference lives in ~**1,841 of 453,726** cell-months — **0.41%**.

⇒ A pooled per-cell CRPS still moves, because each firing cell's difference is large, but the pooled
number understates the effect by about the dilution factor. ⛔ **The ablation must therefore be
reported on the union of firing sets as well as pooled**, or the comparison is weaker than the data
supports. Confidence the coupling wins on the firing union: **~65%**.

⚠️⚠️ **But a restricted score is IMPROPER, so the firing-union figure is a paired difference and
nothing else.** `Gneiting2011_ComparingDensityForecasts`, read in the PDF: *"the composite score is
improper, unless the weight"* is constant — and restricting the evaluation set is a nonconstant
weight. The cost is not lost power: in simulation the improperly weighted test *"prefers a
deliberately misspecified density forecast over the nearly ideal one."*
⭐ **What saves it here:** under common random numbers both arms share the same gate draws, so the
firing union is **the same set for both** — a paired contrast on a fixed subpopulation, with a
bootstrap interval on the difference. ⛔ It is **not** a quotable absolute score and **not** a
promotion metric.

**E2 · v1 is better than the delivered emission on event cells and worse on background.**
The emission issues **zero with certainty on 99.46%** of cells, which is *perfect* on a true zero.
v1 instead hands a positive magnitude to every firing cell — and the gate fires ~**263** where only
~**224** have conflict. ⇒ v1 pays on false positives where the emission pays nothing, and gains on
true events where the emission pays the full count.

⚠️ **I will not predict the sign of the pooled difference.** It depends on a false-positive cost I
have not measured, and guessing it is exactly the habit that produced three of this session's
errors. It is cheap to measure first.

**E3 · The ceiling binds on any month containing a large event**, and the pool maximum must be
printed beside every tail claim (P5).

**E4 · ⭐⭐ The top cell is over-predicted — MEASURED, and not for the reason I first gave.**

⛔ My first account blamed P7's approximation error. That was wrong: $\mathbb{E}[u_{(k)}]$ **is** the
correct expectation under the mechanism, and P7's gap is a statement about quantile matching, not
about v1.

**The real mechanism, measured.** Observed monthly maxima in the score window are
**547, 638, 244, 99, 681, 148, 138** — mean **356**, median **244**. v1's expected top cell, drawing
$n^z_t=263$ from the 36-month pool, is **676**; from the 120-month pool, **787**.

⇒ ⛔ **v1's top cell exceeds the month's actual maximum in 89% of months.**

⇒ The cause is the **pool**, not the coupling: drawing 263 times from 36 pooled months over-samples
the tail relative to one month's single shot at its own maximum. ⭐ **It is therefore independent of
the ranking and will not be fixed by a better order** — and it does not appear in the permutation
ablation at all, because both arms share it.

⇒ ⚠️ **This is the sharpest open consequence in the document.** It reframes §2.1 and it is the first
piece of evidence favouring any stratification: conditioning $z$ on the month's own activity level
addresses it directly, where window length alone does not.

**E6 · v1's PIT will concentrate at rational fractions, and that is not miscalibration.**
`Biegert2026_CalibrationIDR`, read in the PDF: isotonic distributional regression's PIT values
*"resemble Thomae's function"*, *"with additional clusters at 0 and 1"*, and the effect traces to
the **pool-block structure** of isotonic regression.

⭐⭐ v1 has the same structure: its per-cell law is the $k$-th order statistic of a **finite pool**
(P3). ⇒ Expect the artifact, do not read it as a defect, and ⛔ **bin the PIT at ≥100 bins** — the
paper's own finding is that a default 10–20-bin histogram **masks** it.

**E5 · The best window is longer than I would like.** The p99 is flat across windows while the
maximum moves by a factor of 56, and fitting a tail needs exceedances. ⇒ Expect the evidence to
favour a long window and a broad stratum, which is the opposite of conditional relevance.

## 5.2 Hopes

**H1 · ⭐⭐ The one that matters is bigger than this experiment.** If a positives-only body works at
all, then **P8 unblocks every parametric remedy the record confounded.** Epic #230's mixture, the
zero-inflation family, a heavy-tail head — all were fitted where the zeros could reach their shape.
A result here does not just produce a forecast; it tells us whether the whole class was mis-tested.

**H2 · A clean negative is a real result.** `RP-3` was closed on a *label* (`W-107`) and then on a
*conditioning variable* (`M89`). ⇒ This would be the first time the direction was settled with the
zeros out of the body, which is the only version of it anyone actually proposed.

**H3 · The ordering gets measured where it is used.** The permutation ablation values the ranking
*inside the mechanism that consumes it*, which is more to the point than a rank correlation against
truth — and it settles question 1's prerequisite as a by-product.

**H4 · The slots make the successors increments.** v3 is two lines (§4F), v2 is a block (§4G), and
adding a feature such as population flows into `h` and `c` and therefore into the embedding with no
change to the body's definition.

**H5 · The division of labour is defensible on its own terms.** Magnitudes from the record, placement
from the model. That matches the repository's own finding — `thin_0.75` keeps **97%** of the oracle
after deleting a quarter of events while scrambling positions costs **81%**.

## 5.3 Fears, ranked by what they would cost

**F1 · ⛔⛔ The ordering is weak, and the programme stops at step one.**
The record's figure for the regression head's magnitude ordering is **Spearman 0.303 among event
cells, against persistence at 0.367** — beaten by a dumb baseline. And `L5b` measured the related
lookup's effective resolution at **~115 levels**. If that holds, v1 ≈ its permutation and there is
nothing to couple.
⭐ **Mitigation:** `DSC_ISO` on event cells (§2.3) answers this with **one number before any
magnitude is assigned**, and it is a *result* rather than a failure — it redirects to a head trained
for ordering, with a measured reason.

**F2 · ⛔⛔ The verdict depends on which score we committed to, and we have not committed.**
`Scheuerer2015` measured that a pooled score cannot separate a correct coupling from a random one
(0.184 against 0.175) while the variogram score can. ⇒ v1 could be right and read as null. ⚠️ And the
variogram score is **not in FAO-02**, so acting on it needs an amendment — which, if raised after the
numbers exist, is exactly the move this repository's devices exist to make expensive.
⭐ **Mitigation:** settle §2.6 before the first fit, not after.

**F3 · ⛔ The false-positive cost swamps the gain.**
The gate's 1.179x surplus means ~39 extra cells per month receive a positive magnitude they should
not have. Under rank coupling they land at the bottom of the order where the pool's mass is, so each
is cheap — but there are many of them and they are scored against exact zeros.

**F4 · ⛔ The three-way tension in §2.1 has no good resolution and we pick one anyway.**
Narrow stratum starves the pool; broad stratum dilutes it; long window mixes regimes; short window
caps the ceiling. ⇒ The honest answer is to split the choice across mechanisms, which requires v3,
which is not scheduled.

**F5 · ⚠️ One fear I can retire, and it corrects §2.8.**
I listed *"the pool is drawn from the teacher-forced era"* as a dependency. ⛔ **That is wrong for
v1.** The pool holds `o[j, tau]` — **observations**, not model output — so the inference regime of the
reference window is irrelevant to v1. Only the **embeddings** in `P_z` are regime-dependent, and v1
does not use them. ⇒ The worry is real for v2 and absent for v1. §2.8 overstated it.

**F6 · ⛔⛔ The real fear: that I get this wrong again, in the direction of the result.**
Errors I made in this session alone, each caught by someone or something other than my own care:

| | caught by |
|---|---|
| slid to the **gate** rather than the body, five times | Simon, each time |
| claimed `mu` *"is not on disk (verified)"* and that recovering it needed a GPU | `/falsify` — one CPU forward, 0.75 s, 1.06 GB |
| quoted a **0.42x** firing deficit whose denominator I could not name | Simon asking me to be careful |
| called the field total **"exact"** when only its expectation is pinned | the `baseline` session |
| claimed ranks are **fixed across samples** | Simon |
| wrote **"six copies"** of an estimator where a grep finds ten | my own re-check, after the fact |
| wrote a **vacuous gate** (permuting the scored truth) | caught before committing |
| a docstring claiming h36 arrays were never written, when they were | `/code-review` |

⇒ The pattern is that **the guards catch me rather than my care catching me**, and the specific
danger is an error that happens to point the way the result is wanted.
⭐ **Mitigation, and it is the only one that has worked:** the permutation ablation is a
*self-referential* control — v1 against v1 — so it cannot be rescued by a convenient comparator, and
a null in it is unambiguous. ⛔ Plus the standing device: if the result is negative and I then want
to argue the test was inadequate, that argument gets logged as a `W-` row against myself **before**
it is made.

## 5.4 What would make me abandon this

| | |
|---|---|
| v1 does not beat its permutation on **either** score, on the firing union | ⛔ the ordering carries nothing ⇒ stop, and say so as *"not demonstrated on this window and this head"* |
| v1 beats its permutation but loses to the emission on every reported split | the coupling works and is not worth having at this operating point ⇒ park, name what would reopen it |
| the gain is carried by fewer than ~100 cells | one event, not a method |
| the result depends on which of the two scores is chosen | ⛔ no verdict until §2.6 is settled — and settling it afterwards is not available |

---

*Part 5 ends. Document complete: 5 parts. Three decisions remain open in §2.*

---

# 6 · Changelog — what was carried in from the 2026-11-03 document

⛔ **All of §6.1–§6.7's findings are now FOLDED IN to parts 2, 3 and 5.** This section is the
changelog, kept so the provenance of each change is traceable.

## 6.0 ⭐⭐ Every claim was verified against the PDF, not the extracted claim file

The 2026-11-03 document cites the library by claim ID, and `9_library/papers/_claims/*.json` is
itself someone's extraction. ⇒ Each item below was checked in the **paper's own text**
(`pdftotext` over `9_library/papers/*.pdf`) on 2026-11-04.

**The check caught one error of mine.** I was going to write that marginal calibration is *"the
weakest notion in the hierarchy"*. `Gneiting2023_ConditionalCalibration` says **Theorem 2.3.
Auto-calibration implies marginal and probabilistic calibration**, with auto-calibration *"being the
strongest notion"* — but it also exhibits a forecast that *"is probabilistically calibrated but fails
to be marginally calibrated"* **and** one that *"is marginally calibrated but fails to be
probabilistically calibrated"*. ⇒ The two weak notions are **incomparable**, not ordered. "Hierarchy"
is a partial order and I was about to read it as a chain.

⭐ One detail the claim file did not carry, found in the PDF: Biegert's PIT concentration has
*"additional clusters at 0 and 1"* beyond the rational fractions.

## 6.1–6.7 · The seven edits, and where each landed

| # | finding, verified in the PDF | folded into |
|---|---|---|
| **1** | `Gneiting2023` Thm 2.3 — auto-calibration implies marginal **and** probabilistic; the latter two are incomparable | **§3.4 P1**, now stated as a weak property obtained by construction |
| **2** | `Gneiting2011` — *"the composite score is improper, unless the weight"* is constant; the improper test *"prefers a deliberately misspecified density forecast"* | **§5.1 E1**, which now makes the firing-union figure a **paired difference only** |
| **3** | `Gneiting2011` Eq. (20) — the **threshold decomposition**, *"region-specific forecast comparison that the scalar CRPS cannot"* provide; `Lerch2017` — weighted rules' power *"rather limited"* | **§2.6**, now a four-part battery with one gate |
| **4** | `Biegert2026` — PIT *"resemble Thomae's function"*, *"additional clusters at 0 and 1"*, from the **pool-block structure** | **§5.1 E6** (new), with a ≥100-bin reporting rule |
| **5** | `Arnold2024` — the isotonicity decomposition into **MCB** and **DSC**, `DSC = 0` for a static forecast | **§2.3** and **§5.3 F1**, as a one-number pre-check |
| **6** | `Hothorn2014` — higher moments may depend on the covariates; fitted by *"regularized optimization of scoring rules … e.g. the continuous ranked probability score"* | **§2.9** (new), as `Psi_tram` |
| **7** | `Czado2009` — the *"nonrandomized PIT histogram"* for counts, and the **Dawid–Sebastiani score**, *"a proper alternative"* | **§2.6**'s diagnostics |

## 6.8 What in the 2026-11-03 document does NOT improve this one

Stated so the question is closed rather than left open.

| | why not |
|---|---|
| its §6–§8 three-experiment structure | this programme is one experiment with a self-referential control; the three-experiment ladder was built around a frozen-backbone / hidden-state / representation split that §3's slot structure already subsumes |
| its H1–H5 hypothesis set | superseded. The mechanism here is specified, not hypothesised |
| SCRPS (`Bolin2023` C-94) | ⛔ **undefined on a point mass**, and v1 emits a point mass at 0 on every non-firing cell — which is most of them |
| IDR as a candidate | `Henzi2021` **C-4394**: it cannot distinguish distributions differing in spread or shape at fixed location, which is the defect. Already noted in §1.5 as a benchmark only |
| CMA / CID as the ranking instrument | **no implementation exists anywhere in the repository.** Already recorded in §2.6's lineage; `DSC_ISO` (6.5) is the available substitute |
| its §11 curriculum and 16-week timeline | built for the three-experiment structure |

---

*Part 6 ends. Document complete: 6 parts. Three decisions remain open in §2, and §6 names seven
edits to fold into parts 3, 4 and 5.*
