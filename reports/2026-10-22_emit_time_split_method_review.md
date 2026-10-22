# Expert method review — the emit-time split, the sampler, and the collapsed dispersion

**2026-10-22.** Seven seats, fanned out as independent agents with no shared context. ⛔ Every panel
point below carries **my own verification verdict** — confirmed by derivation or source, refuted, or
genuinely open. A seat that asserted something checkable and got it wrong is recorded as wrong.

⛔ **Read §6 first if you read nothing else.** The panel did not split on taste. It converged, from
seven unrelated directions, on the same conclusion: **the two decisions we were about to execute are
both downstream of a defect neither of them touches**, and the number justifying them is not stable.

---

## 1. Target & decisions under review

| | decision |
|---|---|
| **D1** | Fix `TruncatedNBFamily.sample()`, whose fast path writes the literal value `1` wherever `P(Y=0) > 0.95` instead of rejecting and redrawing. Proposed fix: the logarithmic-series limit. |
| **D2** | ADR-073, "the emit-time split" — train the body as `nb`, emit zero-truncated. One flag, no retraining. Measured **−5.94%** composed CRPS, emitted positive fraction 0.0009 → 0.0037. |
| **D3** | Whether the collapsed dispersion parameter (grid median `theta` 3.6e-06) is the **primary** defect, rather than third in the queue. |

**The DGP.** 360×720 land grid, cell-month, 36-month horizon, count truth, three targets. Observed
positive rate **0.0034**; 99.66% exact zeros. Shared ConvLSTM/U-Net trunk, 3 classification gate
heads + 3 two-parameter NB body heads, composed at emit as gate × body. **Fitted by NLL, scored by
CRPS.** One GPU, ~$2.25/arm, n=1 at 300 lessons, seed sd ≈0.019.

---

## 2. Panel

| seat | why seated | fault line covered |
|---|---|---|
| **Tilmann Gneiting** | proper scoring, calibration-subject-to-sharpness, EMOS post-processing | is emission surgery legitimate post-processing, or Goodhart? |
| **Christopher Bishop** | distributional heads, mixture density networks | fix the head vs patch the emission |
| **Michael Betancourt** | computational faithfulness, degenerate geometry, pushforward checks | is the fit identified at all? |
| **Frank Harrell** | anti-dichotomization, ordinal/CPM, calibration | is the gate/body split itself the disease? |
| **Richard Sutton** | the bitter lesson; standing contrarian | hand-designed emission vs learned emission |
| **Operational lead** (archetype) | monthly production, provenance, policy audience | will this survive delivery? |
| **Conflict substance** (Buhaug/Gleditsch composite) | conflict dynamics, recurrence, diffusion, reporting bias | does it respect the process, not just the statistics? |

**Not seated, deliberately:** Kendall (the MTL balancer is not under review), Shi (architecture not
under review), Gal (uncertainty representation not under review).

---

## 3. Library grounding

**Held and load-bearing.** Mullahy1986 **C-809** (the hurdle's positive part *is* the zero-truncated
count model), **C-810** (the with-zeros ψ), ⭐ **C-811** (over/under-dispersion is *misspecification
of the parent's relative zero/positive probabilities* — a description of this defect, written in
1986); Cragg1971 **C-791** (the two-part likelihood separates **only when the two parts have
independent parameters** — ours share a trunk); Gneiting2007 **C-28/C-31/C-33/C-34**, **C-3526**,
**C-3529**, **C-3530**; Gneiting2005 EMOS **C-3553–C-3556** (⭐ **C-3554**: minimum-CRPS estimation is
sharper *and* better calibrated than MLE, because **MLE over-disperses**); Jordan2019 **C-84**
(closed-form CRPS for truncated and censored families), **C-87**; Czado2009 **C-3561** (nonrandomized
PIT for counts), **C-3563** (proper rules disagree because a single high count can dominate);
Arnold2024 **C-3574–C-3578** (the isotonic CRPS decomposition — MCB/DSC); Resin2026 **C-3599–C-3603**;
Lerch et al. **C-79/C-80** (conditioning the evaluation set on the realized outcome makes a proper
score **improper**); Machete2012 **C-1419/C-1422**; Malyshkina2009 **C-613/C-614/C-617**; Lambert1992
**C-798**; Xu2023 **C-563/C-565** (the hurdle assumes perfect detection of presence; it is preferable
under *high* reporting rates, zero-inflation under low); Kong2020 **C-472/C-473**; Gasthaus2019
**C-1404–C-1406**; Walz2024 EasyUQ **C-3583–C-3586**; Ferro2014, Hersbach2000 (fair/unbiased CRPS);
Vesco2026 **C-113** (UCDP underreporting **>100%** at single-fatality events, ~30% at 100), **C-116**;
Hegre2019 **C-102**, Hegre2024 **C-1336/C-1339**, Hegre2025 **C-111**; Abebe2025 **C-1299**;
Randahl2024 **C-1288**; Buhaug/Gleditsch **C-1352, C-1363, C-1364, C-1367, C-1265**; Sundberg &
Melander **C-210/C-212/C-213/C-214**.

**Gaps to fetch** (verified absent — `ls | grep -i` returns nothing):
Harrell, *Regression Modeling Strategies* + Liu/Shepherd/Li/Harrell 2017 (ordinal CPMs);
Betancourt, *Towards A Principled Bayesian Workflow* / *Identity Crisis*;
Sutton, *The Bitter Lesson* (2019);
Kemp 1981, *Efficient generation of logarithmically distributed pseudo-random variables* (the exact
sampler D1 needs); Johnson/Kotz/Kemp, *Univariate Discrete Distributions* (the primary source for the
log-series limit); Fisher/Corbet/Williams 1943; Cameron & Trivedi, *Regression Analysis of Count
Data*; Hilbe, *Negative Binomial Regression*; Ridout 1998; GradNorm (Chen 2018); Krüger & Ziegel,
*Generic conditions for forecast dominance*.

---

## 4. What the panel found, with my verdict on each

### ⭐⭐ The finding that reframes everything — `theta` is not a dispersion parameter

**Bishop.** As `theta → 0`, `NB(mu, theta)` is *identically* a zero-inflated log-series:

```
NB(mu, theta)  ≡  (1 − pi)·delta_0  +  pi·LogSeries(mu/(mu+theta)),    pi = theta·log(1 + mu/theta)
```

⇒ **the body is already a hurdle model**, and `theta` is carrying its zero mass. An all-cell,
99.66%-zero target with only two channels spent one of them on occurrence.

> **MY VERDICT: CONFIRMED, by derivation and numerically.** `1 − p0 = −expm1(−theta·log1p(mu/theta))`,
> which is exactly `pi` to first order. At the control's values `pi` = 3.434e-05 against an exact
> `P(Y>0)` = 3.434e-05. The conditional law matches the log-series to **TV ≤ 1.3e-04** for
> `theta ≤ 1e-04` swept over `mu ∈ [0.005, 2]`.

**Consequence, and it kills ADR-073's stated rationale.** §3 prefers truncation because it has "NO
free parameter" and rejects the `theta` rescale for needing "a scale fitted on data". **Both set the
same scalar** — `pi`. Truncation pins it at **1**; the ×22 rescale sets it to **5.1e-04**.

> **MY VERDICT: ADR-073 §3's central rationale is REFUTED.** Truncation is the `theta` lever pinned
> at its boundary by fiat. That is why the 2×2 found the levers to be **substitutes** — recorded in
> the OUTCOME as a surprise, it was a prediction of this identity. Pinning a parameter at a boundary
> without likelihood justification is not parameter-free; it is the most expensive free parameter
> available.

### ⭐⭐ My own headline number was computed on the wrong population

**Betancourt** derived, from the published numbers alone and before touching the cube, that
`1 − p0 ≤ 1 − e^(−mu) ≤ mu` monotonically in `theta`, so the gate-weighted mean of `1 − p0` must be
≈0.26 — which requires `theta ≈ 0.15–0.32`, **five orders of magnitude above the quoted median**.

> **MY VERDICT: CONFIRMED on the frozen cube.** OOS 457–469, 3,370,536 draw-cell-months:
>
> | population | n | median `theta` | median `mu` | median `E[Y|Y>0]` |
> |---|---|---|---|---|
> | all draw-cell-months | 3,370,536 | **3.69e-06** | 0.0007 | 34.69 |
> | gate > p90 | 337,054 | 0.0004 | 0.0525 | 24.06 |
> | gate > p99 | 33,706 | 0.0352 | 2.4079 | 16.01 |
> | gate > p99.9 | 3,371 | **0.189** | 12.5177 | 22.46 |
>
> Gate-weighted mean `theta` = **0.0902** — 24,000× the grid median. Gate-weighted mean `1 − p0` =
> **0.2428** (predicted ≈0.26). And the identity closes end to end: `E[gate·(1−p0)]` = **0.000899**
> against the measured emitted positive fraction **0.000900**.
>
> ⛔ **Therefore my own tail warning to the owner was WRONG.** I reported that a faithful draw implies
> a 29–56% chance of ≥100 fatalities given any violence. That was computed at the grid median, on
> cells that almost never emit. **Gate-weighted `P(Y≥100 | Y>0)` = 0.00054** — about one positive in
> 2,000. This is the same error as my refuted P-2: weighing a population by its count instead of its
> contribution. Twice in one programme.
>
> It also means "99.05% of cells are in the broken branch", "`E[Y|Y>0]` = 35.1" and "54×" are all
> **background statistics**, and the floor-to-1 shortcut is nearly free precisely where it is applied.

### ⭐⭐⭐ The finding that should stop D2 — truncation inverts the magnitude field

**My own measurement**, prompted by the conflict-substance seat's question about the direction of the
dispersion field. The emitted magnitude is `mu` today and `mu/(1−p0)` under truncation:

| emitted magnitude field | Spearman rank corr. with the gate | high-gate vs rest (swept) |
|---|---|---|
| NB (today) | **+0.4925** (sd 0.0013) | 2.1× → 86.8× → 3,569.7× → 18,347.5× |
| zero-truncated | **−0.1605** (sd 0.0016) | 0.78× → 0.67× → 0.46× → 0.65× |

Dividing by `1 − p0` cancels the signal `mu` carried, because `1 − p0` rises faster across the gate
than `mu` does.

> ⛔ **This falsifies ADR-073's core argument in its own terms.** The ADR's case is that retraining on
> the truncated likelihood destroyed the body (separation 5–70× → 1.0–1.1×) and that the emit-time
> route "takes the truncation without the retrain that caused that damage." **It does not.** The
> damage is a property of truncation, not of retraining, and I measured it on a model that was never
> retrained. Pooled CRPS cannot see it because CRPS here is dominated by *whether*, not *where*.
> ⚠️ Scope: measured against the gate, which is the model's own field. The decisive version uses
> observed conflict and is also free on this cube.

### The number itself is not stable

**Gneiting** (reproduced 0.050758 and 0.047741 exactly, so the same instrument): the net −5.94% is
**18.0% of a gross flow** (gains −8317, losses +5775); **7 cell-months of 842,634 carry 50% of the
net**; the top 50 carry 97.3%; the largest single cell-month is 25.0% of it. Month-block bootstrap
(2000 draws, 13 months): **−5.86% ± 1.22**, CI [−8.34, −3.54]. Paired against the `theta` rescale,
**truncation is 3.68% worse**, P(worse) = **0.960**.

**Harrell**: the plug-in `crps_ensemble` is biased by `≈E|X−X′|/(2m)`, which differs **5.5×** between
these two arms because their spreads differ — so equal `m` is necessary, not sufficient. Under the
unbiased (fair) estimator the headline is **−16.27%**, and the levers become **antagonistic** rather
than substitutes.

> **MY VERDICT: CONFIRMED, reproduced independently.** Same cube, same seed, same common random
> numbers, both estimators on all four arms. ⭐ My plug-in column reproduces ADR-073's committed
> numbers exactly, so this is the ADR's own instrument:
>
> | arm | plug-in | fair (unbiased) | bias | Δ plug-in | Δ fair |
> |---|---|---|---|---|---|
> | NB `theta`×1 | 0.050758 | 0.049628 | 0.001129 | — | — |
> | **ZT `theta`×1** | **0.047741** | **0.041556** | **0.006184** | **−5.94%** | **−16.27%** |
> | NB `theta`×22 | 0.046048 | 0.045300 | 0.000748 | −9.28% | −8.72% |
> | ZT `theta`×22 | 0.046023 | 0.044933 | 0.001091 | −9.33% | −9.46% |
>
> The bias differs **5.5×** between the two arms (0.001129 vs 0.006184), exactly as the derivation
> predicts, because the truncated arm's ensemble spread is far larger. ⛔ **The conclusion inverts
> with the estimator.** Plug-in: the levers are substitutes and the `theta` rescale is the better one
> (−9.28% vs −5.94%). Fair: they are **antagonistic** — truncation alone is −16.27%, and adding the
> `theta` rescale to it *destroys* 6.8 points of that gain. ⚠️ Gneiting's concentration and paired
> figures I did **not** reproduce; they are
> plausible and come from a verified instrument, but they are **OPEN** pending my own run.
>
> Either way the governance point stands and is new: **FAO-02 does not say which CRPS estimator is
> primary**, and on this comparison the choice moves the headline from −5.94% to −16.27% and flips
> the sign of the substitutes finding. That is a gap in a LOCKED framework.

### D1 is a fidelity fix, not a score fix — and my proposed route is worst where the score lives

**Gneiting**, on identical cubes with common random numbers: NB 0.050758; exact inverse-CDF 0.047741
(−5.94%); **the known-wrong floor-to-1 path 0.047469 (−6.48%)**; log-series 0.047620 (−6.18%). The
broken sampler scores **best**, and all three sit inside the ±1.22-point bootstrap sd.

And: the log-series limit is accurate to ~1e-05 relative at `theta` = 3.6e-06 but **1.65% at
`theta` = 0.044**, which is the gate-firing population (median `theta` in the gate's top 1% =
0.0349). ⇒ D1 as I proposed it would replace an approximation that is catastrophic where ~1% of the
score lives with one that is worst where 99% of it lives.

> **MY VERDICT: CONFIRMED from my own sweep, and it corrects my recommendation to the owner.** My TV
> table shows exactly this: TV ≤ 1.3e-04 at `theta` ≤ 1e-04, but **0.0116 at `theta` = 0.044,
> `mu` = 0.05** and **0.0282 at `theta` = 0.044, `mu` = 2**. I reported the favourable end of my own
> sweep as if it were the operating point. The repo's own rule covers this
> (`feedback_sweep_before_reporting_a_scalar`) and I broke it.

**Sutton**, separately: the sampler's docstring claims "there is no cheap exact alternative". The
repo's own driver (`emit_time_split_test.py`'s `ensemble()`) does it exactly, in three lines, with cost that
*falls* as `p0 → 1`:

```python
p0 = np.power(p_succ, th)
u  = p0[None] + u * (1.0 - p0[None])       # rescale the uniform above the zero atom
y  = nbinom.ppf(u, th, p_succ)
```

> **MY VERDICT: CONFIRMED, and it corrects me.** I told the owner there was no cheap exact fix; the
> exact fix is in this repository and produced the headline. The docstring describes a limitation of
> `torch`, not of the problem. I also verified a torch-native route exists — the log-series draw has
> a closed form in two uniforms, so it vectorises and takes a `torch.Generator`; against the driver's
> oracle the medians are identical (66/66, 208/208) and the means agree to 0.2%. But given the
> `theta` = 0.044 error above, **inverse-CDF is the right target and the log-series is at best the
> small-`theta` branch of it.**

### The rollout, the provenance, and the comparator

**Operational lead.** (i) ADR-073 §5 calls the feedback family "open", but the code already decides
it: `hydranet_inference.py:1109` resolves the fed-back draw from the **emitted** family, so merging
the ADR feeds truncated draws through all 36 steps by default — the configuration M45 measured at
×8–22 compounding by h18 and **−0.238 AP**. (ii) The trained family travels on the artifact; the
*emitted* family is read from whatever config the scorer passes, and nothing writes it into a
delivered cube. (iii) The comparator is our own previous emission rule, not a field benchmark.

> **MY VERDICT: (i) CONFIRMED** — the code is explicit and marked PROVISIONAL with the conflict on
> record, so "silently" is too strong, but it *is* a live default and the ADR calls it open.
> **(ii) CONFIRMED** — `hydranet_inference.py:266` vs `:278`; no emit-side sidecar writer exists.
> **(iii) PARTLY REFUTED** — the seat recommended building a `conflictology_neighbors12` reference as
> "the only benchmark that beat `exactly_zero` at grid-month". The holdings say the opposite:
> Hegre2025 **C-111** has `exactly_zero` at **CRPS 0.45** being the hard one to beat at pgm, with the
> conflictology benchmark **worse** at 0.51. The seat also attributed "98% coverage on zeros, 52% on
> non-zeros" to NB intervals; those are **conformal prediction's** numbers (Randahl2024 C-1288), with
> NB among the alternatives compared. **Seat recorded as wrong on both.**

### The truth we are calibrating to is undercounted

**Conflict substance**, and it cuts in our favour. UCDP defines an event as organised violence with
**at least one fatality** (C-210) — so "the gate fired" and "y ≥ 1" are the same statement, and a
floor of 1 is not an invention. And Vesco2026 **C-113**: UCDP underreporting exceeds **100%** at
single-fatality events, declining to ~30% at 100. Single-fatality events are exactly our grid-month
positives.

> **MY VERDICT: CONFIRMED verbatim.** ⇒ "emitted 0.0037 against observed 0.0034 = 1.09× of truth" is
> **not** a calibration endpoint; the denominator is biased low. ⛔ It must not become a tuning target.

**And the gate's calibration — D2's entire premise — is already on record as poor where D2 uses it.**
`reports/2026-10-15_loss_balance_method_review.md:217`, written seven days before ADR-073 and never
connected to it: per-bin relative errors **50% / 34% / 3.4% / 4.1% / 2.3%**, with the note *"the
gate's relative calibration in the two lowest bins, which hold ~99% of the mass, is genuinely poor."*

> **MY VERDICT: CONFIRMED in our own file.** "The gate is calibrated, median |gate − p| = 0.0054" is
> a cross-bin median. D2 hands occurrence entirely to a field whose relative error is 34–50% over
> ~99% of the map — which is the onset and diffusion population, i.e. the thing early warning is for.

### The premise of the whole detour may be confounded

**Bishop.** M45 — "retraining on `truncated_nb` destroyed the body", the single result the emit-time
route exists to avoid — is dated **2026-08-24**. `body_nll_denominator` landed **2026-10-15**, and
`truncated_negative_binomial.py:86` records `"positives"` as "the default, **and every pre-2026-10
config**", with `:126` stating that under it the gradient is **"biased and high-variance"** at this
sparsity. ⇒ M45 trained the truncated body on the biased estimator by construction, and no
`all_cells` outcome exists anywhere in `reports/`.

> **MY VERDICT: the construction is CONFIRMED** (dates and defaults verified in the source). Whether
> the biased reduction *caused* the loss of spatial discrimination is **OPEN** — plausible, untested,
> and worth one arm. ⛔ This is the strongest single argument in the review for not spending the next
> $2.25 on anything else.

### The structural dissent

**Harrell**: the architecture is a dichotomization, trained at the single least-informative threshold
in the integral the metric takes over all thresholds; he proposes a ~12-level ordinal head (softmax →
cumsum → monotone CDF, no family, trained on discrete CRPS), and reports a **free pre-test** in which
a family-free conditional CDF over `gate × mu` beats the best parametric emission by ~5.7%, with the
bin count swept and the claim stated as a range rather than a scalar. **Sutton**: the CRPS-consistent
quantile head already exists in-tree (`views_hydranet/utils/quantile_head.py`, verified present, with
`midpoint_levels`, `monotone_quantiles`, `QuantileLoss` and `hurdle_quantiles_to_samples`), was
shelved as "loses to baseline as-built", and that verdict **predates every measurement on this
table**. He also commits a falsifier: if a free isotonic-regression screen does not beat 0.047741, he
withdraws.

> **MY VERDICT: the quantile head's existence is CONFIRMED; the pre-test numbers are OPEN** (not
> reproduced). Both seats independently reached "stop choosing a family, fit the conditional CDF",
> from opposite traditions, and both named a **free** screen before any arm is spent. That
> convergence is the most actionable thing in the review after the three ⭐⭐ findings.

---

## 5. Key disagreements

| fault line | side A | side B | my reading |
|---|---|---|---|
| **Is D1 urgent?** | Betancourt, conflict seat: a prerequisite — ADR-073 §6's reproduce-the-arithmetic gate is unsatisfiable while `sample` floors to 1 | Gneiting, Harrell: demote it — the broken path *scores best*, so it is a fidelity fix worth ~0.5 points, and fixing it faithfully emits a law with a diverging first moment | **B on priority, A on necessity.** It must be fixed before anything ships, and it must not be allowed to *gate* D3. |
| **Is D2 worth having at all?** | Conflict seat: yes, the occurrence/magnitude split is well grounded (C-1299) | Gneiting: nearer a Goodhart patch — 18% of a gross flow, 7 observations, losing to the estimated alternative by 3.7% at P=0.96 | **B on the ADR, A on the capability.** Keep `emit_distribution` as an axis; do not promote it as the route. |
| **Is the headline −5.94% or −16.27%?** | plug-in (FAO-02's current code) | fair/unbiased (Ferro2014, Hersbach2000) | **Unresolved by the lock, which is the finding.** The estimator is not specified, and it moves the headline 10 points and flips the substitutes result. |
| **What is `theta`?** | a nuisance to rescale | Bishop/Betancourt/Harrell: a hidden zero-inflation weight, unidentified in the limit, duplicating the gate | **B, confirmed.** The NB NLL has no interior optimum in `theta` on an empty cell; the optimiser is not at an optimum, it is somewhere it cannot leave. |
| **Replace the family, or stop having one?** | incremental: 3-channel (π, μ, θ) head, or Markov switching (C-617) | Harrell/Sutton: ordinal or quantile head, no family, trained on the score | **Open, and testable for $0 before either.** |

---

## 6. Synthesis — what I now recommend

⛔ **I withdraw the plan I gave the owner.** It was: launch the control run, fix the sampler in
parallel, investigate `theta` third. All three parts are wrong in their ordering, and two of my own
numbers behind them were computed on the wrong population.

**Order of work, cheapest and most decisive first.**

1. **$0, hours — settle the estimator.** Report every FAO-02 arm under both the plug-in and the fair
   CRPS estimator, with a month-block bootstrap interval and the gross-flow ratio. Decide which is
   primary and write it into FAO-02. ⛔ Until this is done, **no number in ADR-073 has a defined
   meaning**, because the headline ranges −5.94% to −16.27% depending on an unstated choice.
2. **$0, hours — the two free falsifiers of D2, both of which I expect to fire.** (a) Re-score the
   existing 2×2 for **average precision**, which is unmeasured for this arm and is the metric that
   caught all six previous interventions; my measured rank-correlation flip (+0.49 → −0.16) predicts
   it degrades. (b) Stratify the −5.94% by conflict history — recurrence / diffusion frontier / quiet
   interior — because a pooled gain concentrated on 7 cell-months tells us nothing about placement.
3. **$0, hours — re-specify D3 against the right population.** Not "is `theta` collapsed" but "is
   `theta` identified as dispersion, given that a gate already owns occurrence". The profile
   likelihood in `theta` on the frozen `mu` field, computed separately on `{truth>0}` and
   `{truth=0}`. Flat ⇒ no emission rule and no retrain can fix it.
4. **$0, hours — the free screens both dissenting seats proposed**, with their committed falsifiers:
   a family-free conditional CDF / isotonic fit on the frozen cube, bin count **swept**, scored
   against 0.047741.
5. **Then one arm, ~$2.25 — and not the one I proposed.** The highest-value arm is
   `truncated_nb` + `body_nll_denominator=all_cells`, because it tests whether **M45** — the result
   the entire emit-time detour exists to avoid — was confounded by a reduction this repo has since
   documented as biased. Primary readout is the within-window separation statistic, not CRPS.
6. **D1 after D3, and as inverse-CDF, not the log-series.** Use the exact method already in the
   driver, with the driver as the oracle and a parity test. Fix the branch **predicate** to the one
   the docstring itself derives (`mu/(1−p0) < 1 + tol`), not `p0 > _MODERATE_P0`.
7. **D2 stays unpromoted.** Keep `emit_distribution` as a capability with its tests; do not move
   ADR-073 to active, and do not set it in a roster config. Before it could ship it needs: a
   cube-side record of the resolved emission rule, and a code refusal for multi-step rollout until
   #437 measures.

**The strongest dissent to keep live.** Gneiting's: that this is not post-processing of the
legitimate kind, because EMOS *estimates* a correction against the proper score and reports its
uncertainty, whereas D2 changes the emitted family to obtain a dispersion correction it declines to
estimate, then defends it on a marginal-rate diagnostic. If the free work above goes against D2,
that is the reading that was right.

**Second dissent, from the opposite direction.** Harrell's and Sutton's shared conclusion — that the
argument over which family to emit from is downstream of having a family at all, and that the
CRPS-consistent alternative is already written in our tree and was dismissed on evidence that
predates everything on this table.

---

## 7. Methodological risks — register-compatible, NOT appended

⛔ Route via `register-risk`. Highest existing ID at time of writing: **C-357**.

**M-1 (proposed Tier 2).** *Trigger:* reporting or comparing any FAO-02 CRPS number across arms whose
ensemble spread differs. *Location:* `reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py:26-33`;
`reports/2026-10-21_emit_time_split_prereg.md` §6. *Narrative:* the plug-in CRPS estimator is biased
by ≈`E|X−X′|/(2m)`, proportional to **each arm's own** mean absolute difference, so equal `m` equalises
the bias's *form*, not its size. The prereg asserts the bias "cancels in every comparison made here";
it does not. On the emit-time 2×2 the bias differs ~5.5× between arms and the headline moves from
−5.94% to −16.27%, with the substitutes/antagonists conclusion inverting. FAO-02 does not state which
estimator is primary. Silent, and it changes a published conclusion's sign.

**M-2 (proposed Tier 2).** *Trigger:* promoting ADR-073, or setting `emit_distribution` in a roster
config. *Location:* `views_hydranet/distributions/truncated_negative_binomial.py:205` (`mean`);
ADR-073 §3/§4. *Narrative:* truncated emission replaces the emitted magnitude `mu` with `mu/(1−p0)`,
which inverts the field's rank correlation with the gate from **+0.49 to −0.16** and collapses
high-gate separation from 18,348× to 0.65× (swept, 4 cuts, frozen control cube, nothing retrained).
The ADR attributes this damage to *retraining* and claims the emit-time route avoids it. AP is
unmeasured for the arm, and AP is the metric that detected all six prior interventions.

**M-3 (proposed Tier 2).** *Trigger:* citing "median `theta` = 3.6e-06", "99.05% of cells", "54×" or
"`E[Y|Y>0]` = 35.1" in support of a decision about emission. *Location:*
`truncated_negative_binomial.py:171-180`; `RESULTS_LEDGER.md:424` (M66). *Narrative:* all four are
grid-median statistics over a population that supplies ~1% of pooled CRPS and almost none of the
emitted positives. Gate-weighted mean `theta` is **0.0902**, 24,000× the median; median `theta` in the
gate's top 1% is 0.0352. The same C-351 defect the register already tracks, now inside a blocking
finding. ⚠️ The chair committed this error twice in one session, in both directions.

**M-4 (proposed Tier 3).** *Trigger:* citing the `y>0` restriction ("CRPS on event cells improves
17.5%") as justification rather than diagnosis. *Location:*
`reports/2026-10-21_emit_time_split_prereg.md` OUTCOME. *Narrative:* restricting the evaluation set on
the realized outcome makes a proper score improper, with the optimum moved to `w(y)g(y)` (Lerch et al.,
**C-79/C-80**). Legitimate as a diagnostic, never as a rationale.

**M-5 (proposed Tier 3).** *Trigger:* treating "emitted positive fraction = 1.09× observed" as a
calibration target. *Location:* ADR-073 §1/§4; prereg P-1. *Narrative:* UCDP fatality counts are
underreported by **>100%** at single-fatality events, falling to ~30% at 100 (Vesco2026 **C-113**),
and single-fatality events are the grid-month positives. The denominator is biased low, so 1.09× is
not evidence of calibration and is a Goodhart target if optimised. Also: it is marginal calibration,
a diagnostic (Gneiting2007 **C-3529**), and marginal + probabilistic calibration does not imply
auto-calibration (Resin2026 **C-3570** Ex. 2.4).

**M-6 (proposed Tier 3).** *Trigger:* merging ADR-073 without resolving #433. *Location:*
`views_hydranet/utils/hydranet_inference.py:1109`; ADR-073 §5. *Narrative:* §5 records the feedback
family as open; the code already defaults it to the emitted family, so merging feeds truncated draws
through 36 rollout steps — the configuration M45 measured at ×8–22 compounding and −0.238 AP. The
code comment is explicit and marked PROVISIONAL, so this is a documentation/default mismatch rather
than a silent one. Cross-ref the existing emit-provenance gap (ADR-073 §4, shared with
`emit_family_core` and `forecast_composition`): no emit-side sidecar is written, so a delivered cube
does not record the rule that produced it.

---

## 8. Panel points recorded as WRONG

| seat | claim | verdict |
|---|---|---|
| Operational | "98% coverage on zeros, 52% on non-zeros" attributed to **NB** intervals | **WRONG.** Those are standard **conformal prediction**'s figures (Randahl2024 **C-1288**); NB is one of the alternatives compared. The sparse-region under-coverage point survives; the attribution does not. |
| Operational | build `conflictology_neighbors12`, "the only benchmark that beat `exactly_zero` at grid-month" | **UNSUPPORTED.** Hegre2025 **C-111**: at pgm `exactly_zero` (CRPS 0.45) is the hard one to beat and the conflictology benchmark is **worse** (0.51). A field-benchmark reference is still worth building; this is not the one the holdings name. |
| *(the chair)* | "a faithful draw implies a 29–56% chance of ≥100 fatalities given any violence" | **WRONG, mine.** Gate-weighted `P(Y≥100 | Y>0)` = **0.00054**. Computed at the grid median on cells that do not emit. |
| *(the chair)* | "there is no cheap exact alternative to the floor-to-1 path" | **WRONG, mine** (inherited from the docstring). The exact inverse-CDF is in `emit_time_split_test.py`'s `ensemble()` and produced the headline. |
| *(the chair)* | "the log-series limit is exact in our regime, so build it" | **HALF WRONG, mine.** Exact to TV ≤ 1.3e-04 at `theta ≤ 1e-04`, but TV **0.0116–0.0282** at `theta` = 0.044 — the gate-firing population. I quoted the favourable end of my own sweep. |
