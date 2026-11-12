# Research Programme: Non-Parametric Conditional Distribution Estimation for Count Prediction

**Date:** 2026-11-03
**Status:** Programme design. No implementation decisions taken.
**Relationship to RESEARCH_PROGRAMME.md:** This is a self-contained programme document for a direction that sits outside the current ranked queue. It does not insert queue entries or re-rank. Whether and where it enters the queue is Simon's call.

---

## 0. Document identity

This document describes a research programme investigating whether the existing recurrent neural network's probabilistic count forecasts can be substantially improved by replacing or supplementing its parametric negative binomial output head with non-parametric or semi-parametric conditional distribution estimators.

The programme was developed through an adversarial dialogue. The original proposal was a broad survey of non-parametric distribution estimation methods organised as a roadmap. That proposal was criticised for three structural defects: it made no hard choices about what to kill and when, it assumed the hidden state contained distributional information without testing the assumption, and it did not include a serious parametric challenger. The revised programme introduces explicit competing hypotheses, pre-specified success criteria, decision gates that can terminate branches, and a strict experimental ordering that runs the cheapest falsification first.

The document is organised as follows:

- **§1** states the distributional problem from existing measurements, not from theory.
- **§2** inventories what has already been tested and what each test actually proves - because several of this programme's proposed experiments have partial precedents in the existing record, and stating what was tested, what was confounded, and what remains open prevents the programme from re-running experiments it already has answers to.
- **§3** identifies the zero-capture mechanism as a confound that gates everything.
- **§4** states three competing hypotheses and a parametric challenger.
- **§5** pre-specifies quantitative success criteria before any results are examined.
- **§6-8** describe three experiments in strict priority order.
- **§9** states the decision tree with kill gates.
- **§10** presents the opponent audit.
- **§11** gives the curriculum - the sequenced work plan.
- **§12** lists open decisions the experiments must resolve.
- **§13** provides the literature.

**Notation.** Measurement references use the existing ledger convention: `M<n>` is a `RESULTS_LEDGER` entry, `W-<n>` is a withdrawal or correction, `C-<n>` is a concern on the technical risk register, `R-<n>` is a recorded result, `D-<n>` is a standing decision. ADR references point to the `docs/decisions/` directory. Report references point to `reports/`.

### Reviewer critique (opponent seat, 2026-11-03)

Two of the conventions declared here do not resolve, and both are one-line fixes.

- `M<n>` is declared to be a `RESULTS_LEDGER` entry. `M6` as used at §3.1 resolves to the ledger's
  M6, which is the Moran's I / gate-smearing finding and says nothing about `theta`. The
  conditionally-honest `0.4323` lives at `reports/2026-10-12_underfire_next_step.md:224` and is
  called `M6` only inside `reports/2026-10-12_composition_arm_prereg.md:88`, which is a
  dossier-local numbering. Either cite the dossier path or declare the second namespace here.
- ADRs are at `docs/ADRs/`, not `docs/decisions/`. And `ADR-073`, which supplies the zero-capture
  mechanism that §1.2 and §3.1 call the most important structural fact in the programme, sits in
  `docs/ADRs/proposed/`, not `active/`. The algebra stands on its own; the citation should say the
  ADR is proposed, because §3 is where a reader goes to check whether the constraint is settled
  repository policy or a live proposal.


---

## 1. The distributional problem - stated from measurements

The model is a spatiotemporal UNet-LSTM (`HydraBNUNet06_LSTM4`) that predicts conflict fatalities on a 180x180 PRIO-Grid. Its body emits predicted counts through a negative binomial distribution head parameterised by a conditional mean `mu` and a shape parameter `theta`. It produces ensemble forecasts via MC-dropout passes and per-draw NB sampling.

The distributional problem has been diagnosed through a sequence of measurements. The following is what is known, stated as measured facts with provenance, not as theory.

### 1.1 The emission is too concentrated in the bulk and too thin in the tail

Among cells that have non-zero ensemble spread, the body is **over-dispersed in 9 of 10 bins** (`sd/RMSE` 1.15-2.14) and under-dispersed only in the extreme top bin (`M90`, `reports/2026-10-29_draw_count_sweep/`). The pooled dispersion ratio of 0.59 is a Simpson artifact: **99.42% of the population is a point mass** - not a narrow distribution, no distribution at all - and the under-dispersed top bin dominates the aggregate because event-cell errors are large.

This is the specific failure that motivates the programme: the parametric NB cannot simultaneously represent the concentration of ordinary observations (where it over-disperses) and the frequency and magnitude of extreme observations (where it under-disperses). A single two-parameter family with a global shape parameter cannot produce the non-monotone correction the data requires.

### 1.2 The shape parameter is captured by the zeros

The NB negative log-likelihood of a zero observation is `theta * log(1 + mu/theta)`, which is monotone increasing in `theta` (ADR-073 §0). On an empty cell the optimum is `theta -> 0` at every `mu`, with no interior optimum. **99.66% of cells are empty.** The grid-median `theta` has collapsed to `3.6e-06` (`M66`, ADR-073 §0). The fitted shape parameter is set by the zeros, not by the events.

This is the most important structural fact for this programme, because it means that any parametric distribution family trained under the same maximum-likelihood objective on the same zero-dominated population will have its shape parameters pulled by the same mechanism. This includes mixture models.

### 1.3 The ranking information is useful but degrades with horizon

On teacher-forced data (the model receives last month's truth at every inference step), the ranking AUC is 0.99777. On the delivered autoregressive forecast (the model consumes its own predictions), the ranking AUC drops to 0.72340 (`reports/2026-10-27_family_free_head/FINDING.md`). Only the autoregressive ranking decays with forecast horizon.

This is important because any conditional distribution estimator that conditions on the network's forecast score inherits this ranking quality. An AUC of 0.72 means the ranking is substantially better than random but far from perfect. The programme should be explicit that it is working with this delivered ranking, not the teacher-forced one, unless stated otherwise.

### 1.4 The family-free CDF lookup beats the parametric head by ~15%

A non-parametric conditional CDF estimator that bins historical outcomes by the network's `gate * mu` index and constructs an empirical distribution within each bin achieves a CRPS improvement of 14.9-15.3% over the best parametric arm on the same cube (`M67`, `reports/2026-10-27_family_free_head/FINDING.md`).

**This result is measured on a teacher-forced cube** (`C-399`, `W-107`). At T=0 (one-step-ahead with truth as input) the index ranks events at AUC 0.998; at the delivered autoregressive horizon it ranks at 0.72. The 15% margin was measured under the favourable ranking. The within-cube emission-family comparison survives at T=0 (`#443` annotated, not retracted), but the 15% number is not a forecast-skill statement.

The fair CRPS estimator improves monotonically to at least 32,000 bins on 13 held-out months, and the apparent optimum at 8,000 bins is an artifact of the plug-in estimator's `1/n` bias (`M67`, `#459`). The ceiling is unlocated.

### 1.5 The metric battery is largely blind to dispersion

14 of 20 scoring columns in the frozen evaluation framework are functions of the mean field alone, confirmed at exactly `0.000e+00` (`M69-M73`). Dispersion is visible to exactly four columns, three of which are one CRPS split. The plug-in CRPS estimator used for all three CRPS columns (`lodestar_score.py:33`) favours under-dispersed ensembles (Ferro 2014); its apparent improvement with draw count is almost entirely bias - 13.9% of the score over 4 to 4,096 members, against 0.65% for the fair estimator (`M90`, `#442`, `#459`).

PIT, the standard dispersion diagnostic, is rejected by FAO-02. The locked evaluation framework therefore contains no unbiased dispersion instrument.

### 1.6 The spatial field is too smooth, and the cause is the decoder architecture

The model's output field has lag-1 spatial autocorrelation of +0.926 against the 336-month truth's +0.638 and the single-month truth's +0.344 (`M93`, `d12b_curve_readout.out`). At one cell the body carries roughly a fifth of the spatial structure it is due; the defect closes by about 32 cells (`d12b_curve_readout.out` §5).

The cause is measured and localised: both upsampling stages are `ConvTranspose2d(kernel=2, stride=2)`, where kernel equals stride means the windows do not overlap and no spatial mixing occurs. The encoder is roughness-neutral (its full-resolution features bracket its own input at rho(1) +0.314/+0.443), and the decoder discards the detail (`M92`, `M93`, `reports/2026-10-30_smoothness_localisation/`).

This is relevant to the distribution programme because the spatial smoothness affects the quality of the score used for conditional distribution estimation. A smoother-than-truth mean field compresses the score distribution and reduces the discriminative information available to a conditional estimator.

### 1.7 The spread is weakly informative beyond the mean

Spearman correlation between predicted standard deviation and absolute error is +0.73/+0.93 pooled, but only **+0.18** conditional on the mean, events only (`M91`, `spread_partial.out`). The pooled correlation is confounded: `sd` is a monotone function of `mu` via `var = mu + mu^2/theta`, and on true-zero cells `|error| = mu` exactly. A recalibration keyed on the mean alone would capture most of what the spread contains (`W-113`, `R-53`, `R-54`).

### 1.8 The Bayes error floor has never been estimated

The target variable `ged_sb_best` is a sum of geocoded point events whose spatial precision is partly an artifact of measurement, reporting and aggregation (`RP-10`). There is an irreducible error floor and it has never been estimated. If the fine-scale target is partly noise, then 1-cell distributional calibration may be measuring the distribution estimation against a target that includes geocoding error, and aggregates may be the only well-posed evaluation level (`RP-7`).

### 1.9 Summary of the starting position

The distributional problem is real, measured, and structural. The negative binomial cannot represent the non-monotone dispersion correction the data requires. But the problem is entangled with at least four other issues: the objective's zero-capture mechanism (§1.2), the decoder's spatial smoothness (§1.6), the evaluation framework's dispersion blindness (§1.5), and the unmeasured Bayes error floor (§1.8). Any programme that changes the distribution head without acknowledging these entanglements risks measuring its own interventions against rulers that cannot see them, on a spatial field that is distorted, against a target whose noise floor is unknown.

### Reviewer critique (opponent seat, 2026-11-03)

Opening a programme from the measurement record rather than from theory is the right move and this
section mostly does it well. Six provenance repairs, in descending order of how much they change
what the programme should do.

**1. §1.1 drops `M90`'s own attribution, and restoring it adds the cheapest candidate in the
programme.** `M90` finding (1) concludes: *"The dominant defect is not width, it is the ABSENCE of a
distribution, **and the COMPOSITION creates it**."* On identical draws, `soft_gate` leaves **79** of
the 2,905 real events in the zero-spread set where `threshold_gate` at tau=0.20 leaves **550**.
`M66` adds that moving mass off the zero atom at emit time *"costs nothing and no retraining"* and
measures composed CRPS **0.050758 to 0.047741 (-5.94%)** on the frozen control cube. §1.1 carries
the 99.42% and not the cause. Constructive consequence: a composition arm belongs in Experiment 1's
candidate list (see the §6 note), because if it moves the dispersion diagnosis then candidates B
through E are being scored against the wrong baseline.

**2. §1.3's AUC pair reproduces a reading the record retracted, and the corrected version names a
lever.** `M89` finding (3) splits `0.99777 -> 0.72340` on one population, holding each term fixed in
turn: **emission -0.1197** (teacher-forced expectation to teacher-forced delivered sampler),
**inference mode -0.0752**, **horizon -0.0795**, and records that the one-variable form was
confounded (`W-51`). Its conclusion: *"the largest single term is the EMISSION, which has nothing to
do with teacher forcing."* §1.3's horizon claim survives in stronger form - `M89`(4) has the
delivered AUC decaying 0.77561 to 0.73483 while the teacher-forced control does not - but the
attribution changes, and it changes toward something the programme can act on. Rewrite as the
three-way split and cite `M89`.

**3. §1.1 and §1.7 quote `M90` without its inference-mode flag, while §1.4 flags `M67` three
times.** `M90`'s row carries a declared **TEACHER-FORCED** marker in its headline and, in its caveat
column, *"says nothing about whether `mu`/`theta` are right, nor about the delivered autoregressive
regime."* Same flag, same discipline, and the asymmetry currently runs one way: the measurement that
motivates the programme reads as settled while the measurement that would make it unnecessary reads
as unproven. This is worth fixing for a positive reason - it promotes `T-1.1` from a formality to a
finding. Nobody has measured the 9-of-10-bins dispersion shape on the delivered cube, so the
baseline task produces a new result rather than a reference number.

**4. §1.1's first factual sentence names a retired grid.** `180x180` is `africa_me_legacy`,
**13,110 cells**, retired. Every measurement cited in §1 is the global `land` model: **64,818 land
cells on a 360x720 raster**, which is what §6.5 and §7.5 already compute with. Two prior
corrections of this exact confusion are on record
(`reports/2026-10-11_conflictology_empty_mode_prereg.md:51`,
`reports/2026-10-12_underfire_diagnosis_prereg.md:211`).

**5. §1.4 names `C-399` and `W-107` but not the selection caveat.** `M77`/`M78`/`C-393`: *"the
published window is the easiest of eight AND the published target is the best of three ⇒ the
headline is the most favourable cell of an 8 x 3 grid."* The `-15.3%` still stands same-window and
same-target on both sides, which is why the caveat belongs beside `C-399` rather than instead of it.

**6. §1.6 mixes three conventions in one sentence.** `+0.926` is **E-W only** from
`d12b_curve_readout.out:92`; `+0.638` is that file's 336-month base rate; `+0.344` is `M93`'s
**single-month pooled** value. `M93`(5) records that d12b's 36-month figure is **not reproducible**
(this script's `+0.601076` against d12b's `+0.499`, *"a gap larger than the defect either number is
used to discuss"*) and that the comparator was corrected to `0.344` in three artifacts. `M93`(4)
states that a rho quoted beside one on a different convention is *"a `W-51` violation of the
plainest kind"*, and pooling is the same kind of slot. Separately, *"roughly a fifth"* is not in the
cited file: d12b §5's own pre-registered answer at d1 is **0.391 E-W / 0.628 N-S**, and §5 is marked
post-hoc with the fit void. Quote 0.391 and name the convention.

Minor: the ledger states the `sd/RMSE` range as **1.15-2.12**; §1.1 writes 1.15-2.14. Reconcile
against `spread_skill.out`.


---

## 2. What has already been tested and what each test actually proves

Several of the experiments proposed in this programme have partial precedents in the existing record. This section states what was tested, what was confounded, and what remains open - because the programme should not re-run experiments it already has answers to, and should not claim novelty where it is extending existing work.

### 2.1 The two-component negative binomial mixture (Epic #230)

**What was tested.** A mixture of two NB components was trained as the emission head, with independently learned means and mixture weights.

**What was found.** Component 2 was alive (median `w|act` 0.71-0.92), with mean separation of 14x-690x between components, but magnitude was still capped. The gap is tail shape (`xi > 0`), not tail location.

**What the test actually proves.** Less than it appears. The mixture was trained under the same NB negative log-likelihood objective on the same zero-dominated population. ADR-073 §0's mechanism applies: both components' `theta` are pulled by the same 99.66% zero mass. The mixture therefore tests "a mixture trained under this objective did not crack the wall", not "a mixture cannot" (`reports/2026-10-30_state_based_direction/RESEARCH_NOTE.md` §2).

**What remains open.** Whether a mixture trained under a different objective - one that does not allow the zero mass to capture the shape parameter - would produce a different result. This is directly relevant to Experiment 1 of this programme and is the reason a re-test is justified despite Epic #230's null.

**Provenance concern.** Epic #230's numerical results (component means, mixture weights, magnitude cap) exist in this repository only as prose at `RESEARCH_PROGRAMME.md:501`. No committed `.out` file contains them. They are relayed, not measured here.

### 2.2 The family-free conditional CDF (M67, #443, #460)

**What was tested.** A non-parametric conditional CDF estimator: historical outcomes binned by the network's `gate * mu` index, with the empirical distribution within each bin serving as the predictive distribution for new forecasts falling in that bin.

**What was found.** The method reproduces (independently written script recovered the headline to 0.36%). At 8,000 equal-count bins on a 36-month fit window, scored on 13 held-out months, CRPS improves by 14.9% (plug-in) / 15.3% (fair) over the best parametric arm. The fair estimator improves monotonically to at least 32,000 bins; the apparent optimum at 8,000 is the plug-in's `1/n` bias (`#459`). The ceiling is unlocated.

**What the test actually proves.** That at T=0 (teacher-forced), a non-parametric conditional distribution keyed on the model's own index substantially outperforms the parametric NB head. The within-cube emission-family comparison is internally consistent and survives at T=0.

**What the test does NOT prove.** Forecast skill. The cube is teacher-forced: the model receives last month's truth at every step. The index ranks events at AUC 0.998 on that cube and 0.723 on the delivered autoregressive forecast. The 15% margin was measured under the favourable ranking. Any reading of these numbers as forecast skill is retracted (`C-399`, `W-107`).

**What remains open.** Whether the method retains its advantage on autoregressive cubes with the delivered ranking. This is the central question of Experiment 1 in this programme. The family-free CDF is not a new idea for this codebase; it is the method that has already demonstrated the ceiling this programme is trying to reach under operational conditions.

### 2.3 The draw-count sweep (M90, RP-4)

**What was tested.** Whether increasing the number of head draws (K) from 4 to 4,096 improves the ensemble's distributional properties on a frozen cube.

**What was found.** The ensemble converges by K=256. The delivered 16-member ensemble is 38% narrower than the model it is drawn from. Over 4 to 4,096 members the fair CRPS moves +0.46% - in the wrong direction. Under-firing cannot be fixed by draw count at all: the per-draw firing rate is flat across the whole sweep at **0.001446..0.001451** (the 4.02e-03 in `draw_sweep.out:48` is the relative spread across K — the FLATNESS — not the rate), sitting at 0.42x the observed positive rate of 0.003448.

**What the test proves.** Draw count is not a lever. The defect is in the distribution's parameters, not in the sampling from it.

### 2.4 The structure penalty (B-1 arm)

**What was tested.** A spatial structure penalty on the loss, intended to encourage spatially sharper output fields.

**What was found.** B-1 went 0.400 to 0.090 - 4.44x worse. The two predictions that passed were exactly the spread-sensitive ones; it was worse on 11 of 12 held-out scores. The penalty moved its own objective and nothing else - the loss-hacking signature, measured (`reports/2026-10-26_structure_penalty_arm_outcome.md`).

**What the test proves.** Penalising the output for spatial smoothness does not work. The defect is architectural (the decoder), not a training signal that a loss penalty can correct.

### 2.5 The ZINB / hurdle mask

**What was tested.** A zero-inflated negative binomial with a hurdle threshold that masks regression loss for all cells where y=0 (`training_engine.py:172-178`).

**What was found.** Gradient starvation on 95% of cells, training non-convergence, autoregressive divergence, and silent corruption (`reports/paths_forward.md`). The ZINB was falsified in the V2 rollout (blooms, loses).

**What the test proves.** One specific parameterisation of zero-inflation - per-cell `pi` from current features, no persistence, with a hard hurdle mask - fails catastrophically. It does not prove that zero-inflation as a modelling concept is wrong. The mechanism of failure (gradient starvation from masking 95% of cells) is specific to the hurdle implementation.

### 2.6 Summary: what the existing record supports and what it leaves open

| question | status | what is settled | what is open |
|---|---|---|---|
| Does a NB mixture help? | CONFOUNDED | Epic #230's null holds under the existing objective | Whether a mixture under a different objective would work |
| Does a non-parametric CDF help? | YES at T=0, UNKNOWN for forecasts | The family-free CDF beats parametric by ~15% at T=0 | Whether it retains the advantage on autoregressive cubes |
| Is draw count the lever? | NO | Settled by M90 | - |
| Can a loss penalty fix smoothness? | NO | Settled by B-1 arm | - |
| Does zero-inflation help? | ONE IMPL FAILED | The hurdle mask fails | Whether a different zero-inflation mechanism would work |
| Does spread carry information beyond the mean? | WEAKLY | +0.18 conditional, stable | Whether that +0.18 justifies a representation-conditioned estimator |
| Is there a Bayes error floor? | UNMEASURED | - | RP-10 |
| Can the evaluation framework see dispersion? | BARELY | 14/20 columns are mean-field; plug-in CRPS is biased | - |

### Reviewer critique (opponent seat, 2026-11-03)

**This is the strongest section in the document and the main reason the programme deserves to run.**
Distinguishing *"a mixture trained under this objective did not crack the wall"* from *"a mixture
cannot"* is exactly the discipline that makes a re-test legitimate rather than repetitive. Two
additions make the section decisive instead of nearly so.

**`M89` is missing, and reading it correctly makes Experiment 1 sharper rather than smaller.**

`M89` (2026-10-29, same epic as the `M90` this section does cite) scored the delivered
autoregressive forecast and ran the family-free arm on it. On the delivered index: the optimum
**collapses from B=8,000 to B=250**, the best achievable is **worse than the delivered ensemble's own
score** (+4.1% / +4.6%) and worse than the no-model floor at every B >= 1,000. The mechanism is
measured, not argued: the delivered index takes **1,321 distinct values** over 842,634 cell-months
and is **exactly 0.0 on 99.6567%** of them, because `forecast_composition='threshold_gate'` zeroes
the NB draw wherever the gate falls below tau=0.20. Tie handling is not the cause (`average` versus
`ordinal` moves the number by `< 0.0023` everywhere). Its verdict: *"there is nothing to recalibrate
on."*

So §2.6's *"UNKNOWN for forecasts"* was true on 2026-10-29 and is now answered - **for one specific
conditioning variable: the composed, post-threshold index.**

**And that is an opening, not a closure.** `M89`'s mechanism is a property of the *composition*, not
of conditional distribution estimation. The variable it falsified is the delivered product's index,
zeroed on 99.66% of cells by a threshold applied after the draw. The variable §6.1 proposes to
condition on - `gate * mu`, the hurdle expectation - is the **un-composed** index, and on the
autoregressive cube it has never been tested. It has full support by construction, because the gate
emits a probability on every cell. That is a different object from the one `M89` closed, the
difference is measured and named, and testing it is still the cheapest experiment in the programme.
Stating it this way converts the document's largest unstated risk into its clearest design decision.

Suggested rows to add to the §2.6 table:

| question | status | what is settled | what is open |
|---|---|---|---|
| Does the **composed** (post-`threshold_gate`) index support recalibration on the delivered cube? | **NO, measured** | `M89`: 1,321 distinct values, exactly 0.0 on 99.6567%, worse than the no-model floor at every B >= 1,000 | nothing - this route closes |
| Does the **un-composed** `gate * mu` index support it on the delivered cube? | **UNTESTED** | the gate emits a probability on every cell, so support is not the obstacle | the whole of Experiment 1 |
| Does the composition itself cause the dispersion defect? | **STRONGLY INDICATED, never run as an arm** | `M90`(1): 550 to 79 affected events under `soft_gate`; `M66`: free at emit time, -5.94% analytically | whether it moves delivered CRPS |

**Second addition: say that this reopens `RP-3`, because that is the stronger claim.**
`RESEARCH_PROGRAMME.md` records **RP-3 as CLOSED BY FALSIFICATION** (`W-107`, `C-399`), with
EasyUQ / IDR named under it at line 400. Candidates C, D and E are RP-3's content. The document's
header currently says it *"sits outside the current ranked queue"*, which is harder for a reader to
act on and gives away the document's best argument. The accurate framing is: this programme
**reopens RP-3 with the falsification that closed it addressed** - it moves off the teacher-forced
cube - and with `M89`'s measured negative on the composed index as the reason the new conditioning
variable is the un-composed one. That is a re-rank argument, and the queue's own note says the next
pick is Simon's between `#491`, `RP-10`, `RP-7` and `RP-5`.


---

## 3. The zero-capture mechanism and why it gates everything

This section isolates the single most important structural constraint on the programme. It is stated separately because it affects every experiment and every decision gate, and because ignoring it was the specific reason Epic #230's mixture null is confounded.

### 3.1 The mechanism

The negative binomial negative log-likelihood for a count observation `y` with predicted mean `mu` and shape parameter `theta` decomposes into a zero term and a positive term. For `y = 0`:

```
NLL(0 | mu, theta) = theta * log(1 + mu/theta)
```

This is monotone increasing in `theta` for all `mu > 0`. On a cell where the observation is zero - and 99.66% of cells are zero in every month - the loss-minimising `theta` is `theta -> 0`, at every `mu`, with no interior optimum (ADR-073 §0).

For `y > 0`, the NLL has a different dependence on `theta` and does generally have an interior optimum. But the 99.66%-to-0.34% ratio means the gradient from the zeros overwhelms the gradient from the events. The fitted `theta` is set by the zero mass.

The conditionally honest `theta` on event cells is `0.432` (`M6`). The actually fitted `theta` is `0.154` at the body level and `3.6e-06` on the grid.

### 3.2 Why this gates the parametric challenger

The programme proposes testing a two-component NB mixture as the primary parametric challenger (Experiment 1). But a mixture of two NB distributions trained under the same log-likelihood objective on the same population inherits the same mechanism: both components' `theta` parameters are pulled toward zero by the zero mass, regardless of how different their means are.

Epic #230 measured this directly: the two components achieved mean separation of 14x-690x but the tail remained capped. The components found different levels; neither found a different shape.

This means that for the mixture NB to be a fair test, it must be fitted under an objective that does not allow the zero mass to capture the shape parameter. Options include:

1. **Event-conditional fitting.** Fit the NB mixture only on cells where `y > 0`, using the gate's predicted zero probability to handle the zero process separately. This is a hurdle-model structure.
2. **CRPS-based fitting.** Minimise the CRPS of the emitted distribution rather than the NB log-likelihood. CRPS weights the entire distribution equally and does not have the same zero-capture property.
3. **Constrained theta.** Impose a lower bound on `theta` during fitting so that the shape cannot collapse. This is a regularisation approach.
4. **Two-stage fitting.** First fit the gate (zero/nonzero) under BCE, then fit the body (magnitude given nonzero) under NB NLL restricted to event cells.

Each has trade-offs. Option 1 is clean but requires the gate to be well-calibrated before the body is fitted. Option 2 changes the objective in a way that affects the entire training dynamics. Option 3 is simple but arbitrary. Option 4 is the structure of the existing hurdle architecture but without the gradient-starvation failure mode of the hard mask.

The choice of fitting objective is not a peripheral implementation detail. It is the single decision that determines whether the parametric challenger gets a fair test. Experiment 1 must either resolve this explicitly or accept that a negative result for the mixture NB is confounded.

### 3.3 Why this also gates the non-parametric estimators

The non-parametric conditional distribution estimators (conditional empirical, conditional KDE, IDR) condition on the network's forecast score. That score is the output of a network whose body parameters were fitted under the zero-captured objective. The score therefore reflects a mean field computed by a body whose shape parameter was set by the zeros.

If the mean field is distorted by the zero-capture mechanism - if the network would have learned a different `mu` field under a different objective - then the conditional distribution estimators are conditioning on a suboptimal score. This does not prevent them from working (the family-free CDF already demonstrates that conditioning on `gate * mu` produces useful distributions at T=0), but it places a ceiling on what they can achieve.

The conditional estimators can work around this ceiling to the extent that the score preserves ranking information despite the objective distortion. The AUC of 0.72 on autoregressive data suggests that the ranking is imperfect but far from random. Whether it is good enough for useful conditional distribution estimation is exactly what Experiment 1 will test.

### 3.4 Consequence for the programme

The zero-capture mechanism does not block the programme. It constrains the interpretation of results:

1. A positive result from a non-parametric estimator using the existing score may underestimate the ceiling of the approach, because the score itself is suboptimal.
2. A negative result from a parametric mixture trained under the same objective as the existing head is confounded and does not prove that a mixture cannot work.
3. A positive result from a parametric mixture trained under a different objective must be attributed partly to the objective change and partly to the distribution family change - the two are confounded in the other direction.

The programme must live with these confounds. The alternative - first solving the zero-capture problem (RP-5) before starting any distributional work - may be the correct sequencing for the standing queue, but it is not a prerequisite for the conditional distribution programme. The conditional estimators bypass the parametric head entirely and therefore bypass its objective's pathology. Their success or failure depends on the quality of the conditioning score, not on the quality of the parametric fit.

---

## 4. Three competing hypotheses and a parametric challenger

The research programme should distinguish between three competing explanations for the distributional failure, plus one important fourth possibility that the original roadmap underemphasised. These are not stages in a progression; they are alternative explanations, and the experiments should discriminate between them.

### H1 - Distribution-head bottleneck

The recurrent representation contains sufficient information for predicting the conditional distribution. The problem is that the output head is forced to express that information through an inadequate distribution family whose shape parameter is additionally captured by the zero mass.

**Prediction.** A better distributional estimator using the existing predictions or hidden representations improves calibration without retraining the backbone. The improvement is visible on autoregressive cubes, not just teacher-forced cubes.

**What existing evidence says.** Supported at T=0: the family-free CDF achieves 15% CRPS improvement on teacher-forced data. Unknown for forecasts. The T=0 result is consistent with H1 but does not distinguish it from H2, because at T=0 the score is nearly perfect (AUC 0.998) and the distinction between score-conditioning and hidden-state-conditioning collapses.

### H2 - Information bottleneck at the scalar prediction

The hidden state contains information about conditional variance, skewness and tail risk, but this information is lost when the network compresses its representation into a single predicted count. Two observations with similar predicted counts may have very different distributional characteristics, and the scalar score cannot distinguish them.

**Prediction.** Conditioning on the hidden state improves distributional forecasts beyond conditioning on the scalar score. The improvement is specifically in distributional characteristics (zero probability, spread, tail exceedance), not just in predicted magnitude.

**What existing evidence says.** Partially supported: the spread carries +0.18 conditional information beyond the mean (`M91`). This is small but non-zero and correctly signed. It is also measured on a teacher-forced cube, so the autoregressive hidden state may carry more or less. The +0.18 is a lower bound on the information available in the full hidden state, since spread is a single scalar derived from the hidden state.

**Connection to the state-based direction.** Simon's state-based hypothesis - that there exist latent states distinguishing cells that can experience extreme events from cells that cannot, even when they currently look identical - is a specific form of H2. If such states exist and the hidden state encodes them, H2 would be supported. If they exist but the hidden state discards them, that is H3. The distinction between a latent-state model (H2/H3) and a mixture-of-experts model (H1) is memory: a latent state carries forward via transitions `P(s_t | s_{t-1})`, while a mixture weight is computed fresh from current features (`reports/2026-10-30_state_based_direction/RESEARCH_NOTE.md` §4).

### H3 - Representation bottleneck

The network does not preserve the information needed to distinguish observations with different conditional distributional characteristics. The relevant information exists in the original inputs but is lost during encoding.

**Prediction.** Neither the scalar prediction nor the hidden representation adequately explains distributional variation, but the original input features or a different representation do.

**What existing evidence says.** No direct evidence for or against. The encoder is roughness-neutral (`M92`) and preserves spatial detail, so the representation preserves at least some input structure. But roughness-neutrality says nothing about whether the representation preserves information relevant to distributional variation as opposed to spatial structure.

### H4 - A simple parametric extension suffices

The distribution family is insufficiently flexible, but a relatively simple parametric extension - a two-component negative binomial mixture, a zero-inflated model with a different fitting objective, or a different count distribution family - is sufficient. The non-parametric apparatus is unnecessary.

**Prediction.** A mixture of two NB distributions, or a similar parametric extension, fitted under an objective that does not suffer from the zero-capture mechanism, achieves calibration improvements comparable to or better than the non-parametric conditional estimators.

**What existing evidence says.** Unclear. Epic #230's mixture null is confounded by the zero-capture mechanism (§3.2). The ZINB with hurdle mask failed catastrophically, but the failure was in the implementation (gradient starvation), not in the concept. H4 has never been fairly tested.

**Why H4 matters.** If H4 is true, the entire non-parametric programme is unnecessary engineering. A mixture NB with event-conditional fitting or CRPS-based fitting might solve the problem in a day of implementation. This is the cheapest falsification of the non-parametric direction, and it should therefore run first.

### 4.1 The relationship between hypotheses

H1 and H4 are both "the representation is fine, the output mechanism is wrong." They differ in the proposed remedy: H1 says the remedy must be non-parametric, H4 says a better parametric family suffices.

H2 says the representation carries information the scalar prediction discards. H3 says the representation itself is the bottleneck.

The experiments should distinguish between them in this order:

1. First test H4 (cheapest). If a parametric mixture works, stop.
2. Then test H1 (next cheapest). If a non-parametric estimator conditioned on the scalar score works, the hidden state is unnecessary.
3. Then test H2 (requires hidden-state extraction). If the hidden state adds distributional information beyond the score, representation-conditioned estimation is justified.
4. H3 is tested as a byproduct of H2: if the information audit (Experiment 2) shows that the original inputs predict distributional targets better than the hidden state, the representation is losing information.

### Reviewer critique (opponent seat, 2026-11-03)

Adding H4 was the right response to the dialogue and it is now the programme's best-protected
branch: **nothing in the measurement record touches it.** Epic #230's null is confounded exactly as
§2.1 and §3.2 argue, and `M89`'s negative is about a conditioning variable, not about a distribution
family, so it does not reach H4 at all. Worth saying so explicitly here, because it is the
justification for the ordering in §4.1 and it is stronger than the cost argument the section
currently gives.

One addition to the hypothesis set. The record supports a fifth explanation that §4 does not state
and none of the three experiments tests:

**H5 - the composition, not the family or the representation.** The emitted distribution is adequate
at the body, and the defect is created downstream by `forecast_composition='threshold_gate'` at
tau=0.20, which replaces a drawn distribution with a point mass on 99.42% of the population
(`M90`(1)) and zeroes the conditioning index on 99.66% of cells (`M89`(2)).

*Prediction.* Swapping `soft_gate` for `threshold_gate` at emit time, with nothing retrained,
materially improves dispersion and restores support to the conditioning index.

*What existing evidence says.* Supported on both counts without ever having been run as an arm:
`M90`(1) measures 550 to 79 affected real events on identical draws; `M66` measures -5.94% composed
CRPS from zero-truncation alone and records that the lever is free at emit time. `M66` also carries
the counter-caution - its engine delivered -11.46% against a committed +/-1.0 pp tolerance because
`TruncatedNBFamily.sample()`'s floor-to-1 fast path is false when `theta -> 0`, so any composition
arm must verify the sampler before trusting its number.

*Why H5 matters here.* It is cheaper than H4, it is upstream of all three hypotheses, and if it is
true then H1 through H4 have all been measured against a baseline that an emit-time flag change
would have moved. It belongs first in §4.1's ordering.


---

## 5. Pre-specified success criteria

The decision gates in §9 use terms like "solves the problem" and "material improvement." This section defines those terms quantitatively before any results are examined. The criteria are set now so that they cannot be adjusted after the fact.

### 5.1 Primary metric: fair CRPS

All distributional comparisons use the fair (unbiased) CRPS estimator, not the plug-in. The plug-in's bias toward under-dispersed ensembles (`#442`, `#459`, `M90`) disqualifies it as a primary evaluation metric for a programme investigating distributional calibration. The plug-in may be reported alongside for comparability with the existing record, but no decision gate is evaluated on plug-in scores.

CRPS is computed on the delivered autoregressive ensemble at all forecast horizons the evaluation framework supports (months 1-36 where available, or at a minimum month 1 and the longest available horizon). Teacher-forced CRPS is reported separately for comparison with the existing T=0 results but does not feed decision gates.

### 5.2 Event-conditional metrics

Because 99.42% of the population is a point mass, the aggregate CRPS is dominated by the background. A method that improves background prediction while degrading event-cell prediction would improve the aggregate and worsen the forecasts that matter. Separate reporting of:

- **CRPS on event cells** (`truth > 0`): the distributional quality where it matters.
- **CRPS on background cells** (`truth = 0`): must not degrade materially.
- **CRPS on event cells above the 90th percentile of the event-cell distribution**: tail calibration.

### 5.3 Ranking preservation

The existing network's ranking ability is valuable and must not be degraded. Any distributional estimator that improves CRPS at the cost of ranking is a net loss for the system.

Measured by:
- Spearman rank correlation between predicted expected value and observed count, restricted to event cells.
- AUC of the binary event indicator against the predicted expected value.

Ranking metrics are computed on the autoregressive cube. A method that degrades ranking by more than 2% relative is rejected regardless of its distributional improvement.

### 5.4 Calibration diagnostics

Randomised PIT histograms for discrete counts, computed on the autoregressive cube. Calibration is evaluated visually (uniform PIT = calibrated) and via the Anderson-Darling test statistic for uniformity. PIT is computed separately for:

- All cells.
- Event cells only.
- Cells above the 90th percentile of the event-cell distribution.
- Each forecast horizon separately (to detect calibration degradation with lead time).

Randomisation is necessary because the distributions are discrete: the randomised PIT is `U = F(y-) + V * p(y)` where `V ~ Uniform(0,1)`.

### 5.5 What "solves the problem" means

A distributional estimator "solves the problem" if it achieves ALL of the following on autoregressive out-of-sample data, with the improvement stable across at least three temporal validation folds:

1. **Fair CRPS on event cells** improves by at least 5% relative to the existing NB head.
2. **Ranking** (Spearman on event cells) degrades by no more than 2% relative.
3. **PIT on event cells** is visibly closer to uniform than the NB head's PIT, and the Anderson-Darling statistic improves.
4. **CRPS on background cells** does not degrade by more than 1% relative.
5. **Tail calibration**: the predicted probability of exceeding the 90th and 95th percentile thresholds is closer to the observed exceedance frequency than the NB head's.

If all five criteria are met on at least three of four temporal folds (or two of three if only three folds are available), the estimator is declared a solution.

### 5.6 What "material improvement" means

For the information audit (Experiment 2), a "material improvement" in distributional target prediction from adding the hidden state means:

1. The cross-validated log-loss (for binary targets) or proper scoring rule (for continuous targets) improves by more than 1% relative when `H` is added to `(S, C)`.
2. The improvement is stable across temporal validation folds.
3. The improvement survives the randomised hidden-state control (§7.3): the improvement from the real `H` is at least 3x the improvement from the permuted `H_tilde`.
4. The improvement is concentrated in a few principal dimensions of `H` (the low-rank bottleneck shows most of the gain at k=8 or fewer), not spread across the full hidden state dimensionality.

### 5.7 What "nothing improves" means

A direction is declared dead if:

1. The best estimator in that direction fails to meet criterion 5.5 on any temporal fold.
2. The confidence interval on the CRPS difference (estimated by block bootstrap respecting temporal dependence) includes zero.
3. No secondary diagnostic (PIT, tail calibration, ranking) shows a convincing improvement.

A negative result on one direction does not kill the programme. It kills that branch and promotes the next hypothesis.

### 5.8 Evaluation infrastructure requirements

All comparisons must use:

- **The same held-out forecast archive** of genuine out-of-sample autoregressive predictions.
- **The same temporal validation splits**, with identical forecast origins.
- **The fair CRPS estimator** as the primary metric.
- **Block bootstrap confidence intervals** (blocks of at least 3 months to account for temporal dependence) on all CRPS differences.
- **Geographic holdouts** where possible, to distinguish temporal calibration from spatial generalisation.

The evaluation must be pre-committed: the validation splits, metric code and threshold definitions are frozen before any estimator is evaluated. Changes to the evaluation after seeing results require a documented justification and re-evaluation of all previously evaluated estimators.

### Reviewer critique (opponent seat, 2026-11-03)

Pre-committing the criteria is the right instinct and §5.2's event/background/tail split is the
right decomposition for a 99.66%-zero population. Four repairs so the commitments actually bind.

**1. §5.5 and §5.7 contradict each other, and the same result satisfies both.** §5.5 declares a
solution when all five criteria hold *"on at least three of four temporal folds"*. §5.7 declares a
direction dead when the best estimator *"fails to meet criterion 5.5 on any temporal fold"*. An
estimator that holds on three of four folds is simultaneously a solution and dead. Fix: make §5.7
read *"fails on every fold"*, and state whether its three conditions are AND or OR - as written it
is ambiguous, and the AND reading is the defensible one.

**2. §5.5 criterion 3 is the only unquantified criterion of the five.** *"Visibly closer to
uniform"* plus *"the Anderson-Darling statistic improves"* cannot be pre-registered. Give the AD
threshold, or drop the visual clause and gate on AD alone.

**3. Criterion 3 gates promotion on an instrument the locked framework rejects, and the record puts
the amendment first.** §1.5 notes that FAO-02 rejects PIT. `M90`'s caveat column goes further: the
spread-skill table and coverage counts *"are DIAGNOSTICS, not FAO-02 columns - FAO-02 rejects PIT
and governs selection; using them for a promotion decision needs Simon's amendment FIRST."* §5.5
criterion 3 is exactly such a use. O-8 treats this as a post-adoption deployment dependency; the
record makes it a precondition. Constructive fix, and it costs nothing: `T-4.3` already drafts the
amendment, at the end. Move that one-page ask into Phase 0 and the gate is legitimate when it fires
rather than retroactively. Keep PIT as a reported diagnostic either way.

**4. The 5% is a placeholder and O-3 is right.** `T-0.2` already fixes it. Say here that the number
is provisional pending `T-0.2`, so the pre-registration is not read as having committed to 5%.


---

### Panel note - scoring and propriety (expert method review, 2026-11-03)

Full argument and the panel's disagreements are in **§14**. Four items bite here, and all four have
library anchors.

**1. §5.5's primary criterion is an improper score, and the fix is in the same paper O-3 cites.**
`Lerch2017` **C-80**: multiplying any proper score by a nonconstant weight `w(y)` *"renders it
improper, with the optimal predictive density proportional to `w(y)g(y)` rather than the true
density `g(y)`"*, and the indicator weight `w(y) = 1{y > r}` *"produces the same ranking of
competing forecasts"* as restricting the evaluation set. §5.2's CRPS-on-event-cells and §5.5
criterion 1 are that restriction with `r = 0`. Constructive fix, and it is cheap: gate on
**twCRPS** at the §5.2 thresholds, which is a *proper weighted* rule built for exactly this
emphasis; keep unweighted fair CRPS as the primary gate; demote event-conditional CRPS to a
reported diagnostic. The caution comes from the same paper - `Lerch2017` **C-83**: weighted rules'
power advantages are limited and *"vanish under weight functions with increasingly extreme threshold
values"*. So twCRPS buys propriety, not power.

**1a. The primary source for both the problem and the fix is also in the library, and it is
stronger than the Lerch route.** `Gneiting2011_ComparingDensityForecasts` (Gneiting & Ranjan 2011)
**C-53** is the theorem itself: *"If `S0` is any proper scoring rule, then the multiplicatively
weighted score `S(g,Y) = w(Y) S0(g,Y)` is improper unless the weight function `w` is constant
(Theorem 1). The score is minimized by issuing the density `g` proportional to `w f` instead of the
forecaster's true belief `f`, providing a concrete hedging strategy."* **C-54** is the fix, with the
mechanism: threshold-weighted CRPS and quantile-weighted CRPS *"are proper scoring rules for any
nonnegative weight functions"* because *"the weighting applies to the integrand (individual
Brier/quantile scores at each threshold or level) **before** integration, preserving propriety -
unlike multiplicative weighting of a composite score."* And **C-56** shows the stakes are not
theoretical: in simulation the improperly weighted test *"prefers a deliberately misspecified
density forecast over the nearly ideal one"* while *"all corresponding threshold- and
quantile-weighted CRPS tests correctly prefer the nearly ideal forecast."* An improper gate does not
merely lose power - it can select the wrong candidate, which is what decision gate 1 is for.

**1b. And it supplies a better instrument than §5.2's three-way split.** `Gneiting2011` **C-55**:
the mean CRPS admits a **threshold decomposition** `CRPS_n = integral BS_n(z) dz` and a **quantile
decomposition** `CRPS_n = integral QS_n(alpha) d(alpha)`, and these *"can be plotted to diagnose
strengths and deficiencies of density forecasting methods across the distribution, providing
region-specific forecast comparison that the scalar CRPS cannot."* §5.2 wants exactly this - where
in the distribution the improvement sits - and reaches for it by restricting the population, which
is the improper route. The decomposition gives the whole curve, is proper by construction, and
subsumes the event / background / tail split as three reads off one plot. `T-0.1` should implement
the threshold decomposition; §5.2's split then becomes a reporting choice rather than a scoring one.

**2. Every criterion in §5.5 is a skill score, and §5.8 already contains the remedy.**
`Bolin2023` **C-97**: *"Skill scores, which normalize a scoring rule by a reference forecast score,
are in general improper even when based on a proper scoring rule."* §5.5's five criteria are all
relative percentages (5%, 2%, 1%, and §8.4's 3%). §5.8 already specifies block-bootstrap confidence
intervals on CRPS **differences** - which is the proper form. The two sections disagree and §5.8 is
the one to keep.

**3. The metric §10.4 proposes as possibly primary is undefined on 99.42% of the population.**
`Bolin2023` **C-94** gives `SCRPS(P,y) = -E_P|X-y| / E_{P,P}|X-Y| - (1/2) log E_{P,P}|X-Y|`. On a
point-mass forecast `E_{P,P}|X-Y| = 0`, so both terms are undefined. `M90`(1): the delivered ensemble
**is a point mass on 99.42% of the population** (837,759 of 842,634) under `threshold_gate` at
tau=0.20. SCRPS therefore cannot be a primary metric on the delivered cube **as the product
currently composes**. This is the most constructive finding in the panel's set, because it inverts
into a reason: SCRPS becomes available exactly when the composition arm restores spread, so
candidate A' (§6 note) is not merely the cheapest candidate - it is the precondition for the better
ruler. And the argument for wanting SCRPS is sound and well-anchored: `Bolin2023` **C-93** shows
CRPS is not locally scale invariant, so *"observations with larger predictive variance contribute
more to the average score, which can lead to unintuitive forecast rankings"* - which is `M90`'s
Simpson artifact restated as a property of the scoring rule. Note also **C-96**: SCRPS is locally
scale invariant but **not robust** (`alpha = 1`), which matters on a field where the top spread
bin's RMSE is 146-308.

**4. §5.4 and §5.5 should name which calibration notion they are testing.** `Resin2026` **C-3599**
places PIT uniformity (probabilistic calibration) at the **weak** end of the hierarchy: it implies
only unconditional quantile calibration, while *auto-calibration* is the strong notion and
*conditional* calibration is the one that matters for a covariate-indexed forecast. A forecast can
be probabilistically calibrated and conditionally miscalibrated at every score level - which is the
failure mode the programme is hunting. Two cheap additions: `Resin2026` **C-3601** extends **double
PIT calibration** (Yang 2024, originally for count data) to real-valued outcomes and establishes
that for outcomes with a discrete component it is *logically independent* of every other notion,
including probabilistic calibration - so it is a second, non-redundant instrument for exactly this
outcome type. And `Czado2009` **C-3564**: the **Dawid-Sebastiani score** is proper and depends on
the predictive distribution only through its first two moments - a cheap dispersion-sensitive gate
for a programme whose own ruler sees dispersion in 4 of 20 columns. Both belong in `T-0.1`.

---

## 6. Experiment 1 - Frozen-backbone distribution challenge

**Priority:** First. This is the cheapest experiment and the one most likely to end the programme early.

**Question:** Can we solve the distributional problem without changing the neural network?

**Cube:** Autoregressive. All estimators condition on the delivered forecast, not on teacher-forced predictions. This is a deliberate departure from the existing family-free CDF work, which was measured on the teacher-forced cube. The question is whether conditional distribution estimation works under operational conditions (AUC 0.72 ranking), not under idealised conditions (AUC 0.998).

**Historical forecast archive.** Construct a common archive of genuine out-of-sample autoregressive forecast-observation pairs. For every forecast in the archive, retain:

| field | description |
|---|---|
| forecast origin | the month at which the forecast was issued |
| valid time | the month being forecast |
| spatial identifier | PRIO-Grid cell ID |
| forecast horizon | months ahead (1 to max available) |
| observed count | `ged_sb_best` realised value |
| scalar prediction | the network's predicted `mu` (or `gate * mu`, see below) |
| target type | `sb_best` (extend to `ns_best`, `os_best` if resources permit) |
| model version | training fold identifier |

The archive must consist of genuine out-of-sample forecasts: no observation may be used both to fit the conditional distribution and to evaluate it. Use rolling-origin temporal validation with at least 12-month fit windows and non-overlapping scoring windows.

### 6.1 The conditioning score

The family-free CDF used `gate * mu` as the conditioning score. This is the expected count under the hurdle model: `P(Y > 0) * E[Y | Y > 0]`. It is a natural choice because it combines the gate's event probability with the body's magnitude prediction.

An alternative is to use the rank-transformed score (percentile) rather than the raw score. Score percentiles make the conditioning less sensitive to changes in the absolute scale of predictions across forecast horizons and model versions.

Both should be tested. The rank-transformed score should be computed within the fit window to avoid look-ahead.

### 6.2 Candidates

Five candidates, not seventeen. Each is chosen for a specific reason.

#### A. Existing negative binomial head (baseline)

The current NB emission, scored on the autoregressive cube with the fair CRPS estimator. This is the reference against which all others are measured. It has never been scored with the fair estimator on the autoregressive cube (all existing scores use the plug-in on the teacher-forced cube, or the plug-in on the autoregressive cube with limited draw count).

**Implementation.** Extract the NB parameters (`mu`, `theta`, `gate`) from the autoregressive cube. Compute the fair CRPS of the emitted NB distribution against the observed count.

#### B. Two-component negative binomial mixture (parametric challenger)

A score-conditioned mixture of two NB distributions:

```
P(Y = y | s) = pi(s) * NB(y; mu_1(s), theta_1) + (1 - pi(s)) * NB(y; mu_2(s), theta_2)
```

where `s` is the conditioning score, `pi(s)` is the mixture weight, `mu_1(s)` and `mu_2(s)` are the component means, and `theta_1`, `theta_2` are global dispersion parameters.

This is the parametric challenger that the original roadmap underemphasised and that the state-based direction research note elevated. It is included because it directly tests H4: whether a simple parametric extension solves the problem.

**Fitting objective.** The mixture must NOT be fitted under the standard NB NLL on the full population (§3.2). Instead, fit using one of:

- **CRPS minimisation**: fit the mixture parameters to minimise the fair CRPS of the mixture distribution. This avoids the zero-capture mechanism entirely because CRPS weights the entire distribution, not just the likelihood at the observed point.
- **Event-conditional NLL + gate BCE**: fit the mixture NLL on event cells only (`y > 0`), using the existing gate's predicted zero probability to handle the zero process. This is a hurdle-style decomposition without the hard mask.

Both fitting objectives should be tested. If CRPS minimisation is computationally expensive (it requires evaluating the mixture CDF), event-conditional NLL is acceptable as the primary fitting method.

**Parameterisation.** Start with a small regularised model: `pi(s)` and `mu_1(s)`, `mu_2(s)` as monotonic functions of the score (isotonic regression or low-degree splines), with `theta_1` and `theta_2` as global parameters. This is a post-processing model, not a neural architecture change.

**Constraint.** `theta_1` and `theta_2` must have a lower bound (e.g., `theta >= 0.01`) to prevent the zero-capture mechanism from collapsing them. The bound should be set from the conditionally-honest `theta` estimate of 0.432 (`M6`) as a prior, not from the data.

#### C. Score-conditioned empirical distribution

The conditional empirical distribution, keyed on the forecast score:

```
F_hat(y | s) = sum_i w_i(s) * I(y_i <= y)
```

where `w_i(s) = K((s - s_i) / h_s) / sum_j K((s - s_j) / h_s)` and `K` is a kernel function on the score dimension.

This is the non-parametric core of the family-free CDF approach, extended to the autoregressive setting. No outcome smoothing is applied: the distribution at each score level is the weighted empirical distribution of historical observations.

**Implementation.** Use equal-count bins (as in the family-free CDF) rather than kernel weights. Sweep the bin count from 250 to 32,000 on the fit window, evaluate on held-out months. Report both plug-in and fair CRPS.

This experiment isolates the value of the neural network's ranking information from the value of density smoothing. If it works, it demonstrates that the ranking alone is sufficient to construct useful conditional distributions.

#### D. Score-conditioned KDE

Extend the conditional empirical distribution by smoothing the outcome values:

```
f_hat(y | s) = sum_i w_i(s) * K_{h_y}(y - y_i)
```

This introduces two independently configurable parameters: the score bandwidth `h_s` (how many historical forecasts contribute) and the outcome bandwidth `h_y` (how much the outcome distribution is smoothed).

For count data with massive zero-inflation, outcome smoothing must be handled carefully:

- **Zero-preserving KDE.** Treat `P(Y = 0)` as a separate probability mass estimated from the score-conditioned zero fraction, and smooth only the positive-count distribution.
- **Log-scale smoothing.** Transform positive counts via `z = log(1 + y)`, estimate the density on the transformed scale, and recover the count distribution through appropriate probability integration.

Both approaches should be tested. Bandwidth selection should optimise the fair CRPS on a validation window within the fit period (not on the scoring window).

#### E. Isotonic Distributional Regression (IDR)

IDR estimates conditional distributions under a stochastic ordering constraint: for scores `s_1 < s_2`, the estimated distributions satisfy `F(y | s_1) >= F(y | s_2)` for every outcome threshold `y`.

This is a specific conditional distribution estimator with a theoretical guarantee (stochastic monotonicity) that neither the empirical binning nor the KDE enforces. It provides a principled comparison: does enforcing stochastic ordering improve performance, or do the actual conditional distributions cross too frequently for the constraint to help?

If implementation cost is low (the `isodistrreg` R package exists, and Python implementations are available), include it. If implementation cost is high, defer to a later stage - candidates C and D are more important because they connect directly to the existing family-free CDF work.

### 6.3 Experimental design

All five candidates are evaluated on the same held-out forecast archive, using the same temporal validation splits and the same fair CRPS estimator.

**Validation structure.** Rolling-origin validation with at least three non-overlapping scoring windows of at least 6 months each. The fit window for each fold uses all available history before the scoring window. Candidates B (mixture NB) and D (KDE) require hyperparameter selection; this is done within the fit window via a further held-out validation split, never on the scoring window.

**Reporting.** For each candidate and each fold:

1. Fair CRPS, aggregate and event-conditional.
2. Fair CRPS by forecast horizon (if multiple horizons available).
3. PIT histogram (all cells, event cells, event cells above 90th percentile).
4. Ranking preservation (Spearman, AUC).
5. Tail exceedance calibration at the 90th and 95th percentile thresholds.

### 6.4 Decision gate 1

The result of Experiment 1 determines the next step.

**Mixture NB wins or ties the non-parametric estimators (H4 supported):**

Adopt the mixture NB. The distributional problem is solved by a parametric extension. The non-parametric programme is unnecessary as an operational intervention. The information audit (Experiment 2) remains scientifically interesting but is no longer urgent.

**Non-parametric estimator wins and meets the success criteria of §5.5 (H1 supported):**

Adopt the best-performing non-parametric estimator. Keep the backbone frozen. The information audit becomes a question about whether the ceiling can be raised further by conditioning on the hidden state, not a prerequisite for an operational improvement.

**Nothing meets the success criteria (H1 and H4 not supported by score conditioning alone):**

Move to the information audit (Experiment 2). The failure of score-conditioned estimation does not kill the non-parametric direction - the scalar score may simply be insufficient, which is H2. But it does kill the "frozen backbone, score-conditioned post-processing" approach.

**Conditional empirical distribution (C) works but KDE (D) does not improve over it:**

Smoothing adds nothing. The outcome distribution's discrete, zero-inflated structure makes kernel smoothing counterproductive. Adopt the empirical binning approach.

**KDE works but only with zero-preserving or log-scale smoothing:**

The zero-inflation structure requires explicit treatment. Any subsequent work must model the zero process separately.

### 6.5 What could go wrong

1. **The autoregressive ranking is too poor.** If the AUC of 0.72 means the score does not separate forecast levels well enough for conditional estimation, all score-conditioned estimators will fail. This is a legitimate outcome that supports moving to H2 (hidden-state conditioning) rather than a flaw in the experiment.

2. **The effective sample size in score bins is too small.** With 99.66% zeros, the number of event-cell observations per score bin may be very small, especially in the upper tail. At 842,634 total cell-months and 0.34% events, there are roughly 2,865 event observations. With 1,000 bins, that is ~3 events per bin on average, with heavy skew toward the lower bins. The experiment must report effective sample sizes per bin and flag any result that depends on bins with fewer than 10 events.

3. **The evaluation framework cannot see the improvement.** If the improvement is in dispersion and 14/20 columns are blind to dispersion, the frozen framework will not register it. The experiment must use its own evaluation (fair CRPS, PIT, tail calibration), not the frozen FAO-02 battery.

4. **The mixture NB gets a fair test but the score-conditioned estimators do not.** The mixture NB conditions on the score through explicit parameterisation. The empirical and KDE estimators condition on the score through binning. If the binning is too coarse, the comparison is unfair. The bin count sweep addresses this, but the result should be reported as a function of bin count so the comparison is transparent.

### Reviewer critique (opponent seat, 2026-11-03)

Five candidates rather than seventeen, each with a stated reason, is the right design. Five
additions, one of which is a new arm.

**1. Add candidate A' - the composition arm, and run it first.** Same checkpoint, same body, same
`theta`, `soft_gate` in place of `threshold_gate` at the emit step, nothing retrained. `M90`(1)
measures 550 to 79 affected real events on identical draws; `M66` measures composed CRPS
**0.050758 to 0.047741 (-5.94%)** from the related zero-truncation lever and records that it
*"costs nothing and no retraining"*. It is cheaper than candidate B, no prior measurement has run it
on the delivered cube, and it changes the baseline every other candidate is scored against. It also
carries its own verification requirement, from `M66`'s blocking finding: the engine delivered
-11.46% against a committed +/-1.0 pp tolerance because `TruncatedNBFamily.sample()`'s floor-to-1
fast path assumes `E[Y|Y>0] -> 1`, which is false when `theta -> 0` (median `theta` 3.6e-06 here;
the analytic value is 35.1 against the sampler's 1.02 on 99.05% of the grid). Any composition arm
verifies the sampler before quoting a number.

**2. §6.1 should say in one sentence which index it means, and why.** As written it is ambiguous
between the composed (post-`threshold_gate`) index and the un-composed `gate * mu`, and the whole
experiment turns on the distinction: `M89` measured the composed one and found 1,321 distinct values,
exactly 0.0 on 99.6567% of cells, and an estimator worse than the no-model floor at every
B >= 1,000. The un-composed index is the programme's actual bet and it has full support by
construction. Naming it converts the largest unstated risk here into the clearest design decision.

**3. Use the measured event count.** §6.5 derives *"roughly 2,865"* from 842,634 x 0.34%. The
measured value on exactly that population is **2,905** (`M89`, `M90`, and `M93`'s `G-19` reproduces
it exactly from an independently written script). O-7's 143 and 287 inherit the estimate.

**4. §6.5 point 2's worry already has an instrument.** *"The effective sample size in score bins is
too small"* is the right concern, and `M89`'s distinct-value count is the measurement of it. Cite it,
and keep the per-bin effective-sample-size reporting - that part is well specified.

**5. §6.4's first branch disagrees with §9's tree.** Here the mixture is adopted when it *"wins or
ties the non-parametric estimators"*, with no §5.5 bar; §9 requires §5.5. Two estimators that both
fail can tie, and under §6.4 that ends the programme with an adoption. Align on §9's version.


---

### Panel note - the candidate set (expert method review, 2026-11-03)

Full argument and the panel's disagreements are in **§14**. The candidate list is where the library
is most directly useful, and it changes the list in four ways.

**1. Candidate E cannot represent the defect the programme exists to fix.** `Henzi2021` **C-4394**
states IDR's two limitations in the authors' own emphasis: it *"cannot extrapolate beyond the range
of training responses: the highest outcome judged feasible equals the largest observation in the
training set"*, and it *"uses only the usual stochastic order on conditional distributions, making
it **unable to distinguish situations where distributions agree in location but differ in spread or
shape**"*. §1.1's diagnosis is a non-monotone **dispersion** correction at fixed location -
over-dispersed in 9 of 10 bins, under-dispersed in the top bin. That is precisely the class of
difference IDR is constructed not to see. And the no-extrapolation limitation is Epic #230's finding
restated: *"the gap is tail shape (`xi > 0`), not tail location"*.

This is not a reason to drop E. `Henzi2021` **C-4381** recommends IDR *"as a generic benchmark
technique"* - competitive, no tuning, no implementation choices beyond the partial order, and a
unified treatment of discrete and mixed outcomes. Keep it as the calibrated benchmark and the
reference distribution it is good for. But §6.4's *"non-parametric estimator wins and meets §5.5"*
branch should exclude E from the **adopt** set on structural grounds, stated in advance, because an
E win would be a win on aggregate CRPS by a method that cannot express the finding.

**2. The library's best-supported candidate for a discrete outcome is missing.** `Henzi2021`
**C-4382**'s four-scenario simulation: in the **discrete** scenario IDR *"is outperformed only by
TRAM"* - conditional transformation models (Hothorn, Kneib & Buhlmann 2014), already listed in
§13.2 and held in the library. TRAM learns a monotone transformation to a fixed reference
distribution, so unlike IDR it can move spread and shape at fixed location. **Add as candidate F.**
It is the single highest-value addition to Experiment 1 and it costs one R or Python package.

The library holds the paper (`Hothorn2014_ConditionalTransformationModels`) and its claims make the
case more strongly than the Henzi simulation alone. **C-4430**: the partial-transformation
decomposition `h(v|x) = sum_j h_j(v|x)` means that *"unlike standard transformation models where
`h = h_Y + h_x` constrains higher moments to be independent of `x`, this formulation allows the
conditional distribution `P(Y <= v | X = x)` to have **variance, skewness, and other higher moments
that depend on the explanatory variables**."* That is the exact capability `Henzi2021` **C-4394**
says IDR lacks, and the exact defect §1.1 diagnoses. **C-4431**: the population minimizer of the
integrated loss is the true conditional transformation function, and *"the squared error version is
the continuous ranked probability score (CRPS)"* - so TRAM is **fitted by CRPS minimisation**, which
is the first of the two objectives §6.2 candidate B asks for in order to escape the zero-capture
mechanism, already solved and already implemented. **C-4434**: in simulation on a heteroscedastic
varying-coefficient model at `N=200` with up to five noise variables, CTM matches parametric GAMLSS
in median MAD *"but show[s] substantially lower variability"*, while *"kernel conditional
distribution estimators show the largest median MAD"* - the library's own simulation ranks TRAM
above candidate D's route at a sample size comparable to this programme's event count.

One caveat to carry into `T-1.x`, from **C-4433**: the boosted estimate *"is not automatically
monotone in `v`"*; three enforcement strategies exist and none was needed in the paper's studies,
and non-monotonicity *"is interpreted as a diagnostic for model misspecification, overfitting, or
low signal-to-noise ratio."* On a 0.34% event rate that diagnostic will earn its keep, so report it.

Taken together: TRAM is CRPS-fitted, moves higher moments with the covariate, beats the KDE route in
the library's own simulation, and is the only method in the set that is second-best in `Henzi2021`'s
**discrete** scenario rather than absent from it. If Experiment 1 can only afford three new arms,
the panel's cut is **A' (composition), F (TRAM), and C-as-analog-ensemble** - one that removes an
error surface, one that can express the defect, one that tests whether the ranking alone suffices.

**3. Candidate D's bandwidth selection is the weakest-specified part of Experiment 1, and the
library holds both the diagnosis and the fix.** `Hyndman1996` **C-4406**: bandwidth selection is
*"explicitly deferred"* in the paper the estimator comes from - the optimal bandwidths depend on the
unknown density, the paper's own examples *"use bandwidths chosen by trial and error"*, and the
authors state practical selection is beyond scope. `MacDonald2011` **C-4464**: likelihood-based
kernel bandwidth estimators are **inconsistent for heavy-tailed distributions** (Schuster & Gregory
1981) - on Cauchy data at n=500 the kernel-only bandwidth is **12.41** against the two-tailed
mixture's **0.41**, a 30x oversmoothing. §6.2 and `T-1.4` specify **Sheather-Jones**, a univariate
*unconditional* selector, applied per score bin to a heavy-tailed count field. Fix: select by
leave-one-out **CRPS** over the grid `T-1.4` already half-specifies (0.5x / 1x / 2x), and treat
`MacDonald2011`'s two-tailed mixture as the principled alternative, since it estimates the bandwidth
and the tail jointly and was built for this failure.

Related and worth one line in §6.2: `Hyndman1996` **C-4403** - the kernel conditional density
estimator's conditional mean *is* the Nadaraya-Watson smoother, so it inherits that smoother's
boundary bias. The score has a hard boundary at zero with 99.66% of the mass against it.

**4. Candidate B's mean constraint already has a non-parametric twin with a proof attached.**
`T-1.2` constrains the mixture mean to match the network's `mu` so the ranking is preserved.
`Hyndman1996` **C-4404** is that idea as an estimator: shift each observation to
`Y_j*(x) = r_hat(x) + e_j - sum w_i(x) e_i`, so the conditional density's mean is *exactly* `r_hat(x)`
for any chosen smoother, and it has *"provably smaller IMSE than the standard kernel estimator under
conditions where the conditional density shape does not change with x (only the location shifts)"*.
**Add as candidate D'** - near-zero marginal cost on top of D, and it isolates the shape question
from the location question, which is the split §1.1 says the NB cannot make.

**5. One further candidate with no distribution family at all.** `Kuleshov2018` **C-1451**: the
isotonic recalibration route applies to a **non-probabilistic** forecaster through any increasing
feature function, for example `F_x(y) = (y - mu(x)) / sigma(x)`. That is cheaper than candidate B,
needs no family and no mixture optimisation, and the programme already has `mu` and `sigma`. Its
predicted cost is on record too - `Kuleshov2018` **C-1454**: recalibration preserves point accuracy
but *"forecasts may become less sharp if the original forecaster was underestimating its
uncertainty"*, and `M90`(6) measures the per-draw firing rate at **0.42x** the observed positive
rate. Expect the sharpness cost; §5.3 is where it will appear.

**6. A correction to §10.4's reading of `Zhang2025`, because it reverses the recommendation.**
§10.4's O-5 note says *"the programme's hurdle decomposition has the same structure and the same
risk."* `Zhang2025` **C-3042** says the opposite: the hurdle *"provides a unified probabilistic
framework that **avoids** the hard-threshold discontinuity of two-stage modeling, which breaks
uncertainty propagation between rain detection and rate estimation stages."* So this citation
**endorses** §12.2's hurdle default. What it warns against is a hard threshold interposed between
detection and rate - which in this system is `threshold_gate` at tau=0.20, applied at emit time
between the gate and the body. `Zhang2025` is therefore library evidence *for* the composition arm,
from a second and independent direction.

---

## 7. Experiment 2 - Representation information audit

**Priority:** Second. Runs only if Experiment 1 does not fully solve the problem, or in parallel if resources permit. Does not require any output from Experiment 1.

**Question:** Does the hidden state contain information about the conditional distribution of outcomes that the scalar prediction discards?

This experiment is the gate for the entire representation-conditioned branch of the programme. If the hidden state carries no distributional information beyond the scalar score, there is no justification for building a hidden-state-conditioned distributional architecture.

### 7.1 What is being tested

Let:

- `X` = original input features (the spatiotemporal feature volume fed to the network).
- `H` = recurrent hidden representation (the LSTM hidden state `h_t`, and optionally the cell state `c_t`).
- `S` = scalar neural network prediction (`gate * mu`, the conditioning score from Experiment 1).
- `Y` = observed fatalities.
- `C` = known context: spatial identifier, forecast horizon, target type, temporal features (month, season).

The central question is:

```
I(H; Z | S, C) > 0 ?
```

where `Z` represents an aspect of the outcome distribution that we care about.

### 7.2 Distributional targets

An individual observation does not have an observed variance. We have one realised count for each forecast. The information audit therefore tests whether the hidden state predicts specific distributional characteristics of the outcome, constructed from the observed counts.

Four targets, chosen because they cover different aspects of the distribution:

| target | construction | what it tests |
|---|---|---|
| **Zero indicator** | `Z = I(Y = 0)` | Does `H` carry information about the probability of zero fatalities beyond what `S` provides? This is the zero-inflation structure. |
| **Bulk residual magnitude** | `Z = |Y - E[Y|S,C]|` where the conditional expectation is estimated non-parametrically from the fit window | Does `H` identify observations with unusually large deviations from their expected magnitude? This is conditional heteroskedasticity. |
| **Upper-tail exceedance** | `Z = I(Y > q_{0.95}(Y|S,C))` where the conditional 95th percentile is estimated from the fit window | Does `H` identify observations likely to exceed high conditional quantiles? This is tail risk. |
| **Conditional PIT** | `Z = F_hat_S(Y^- | S, C) + V * p_hat_S(Y | S, C)` where `F_hat_S` is the score-conditioned CDF from Experiment 1 and `V ~ Uniform(0,1)` | Does `H` explain remaining distributional miscalibration after conditioning on the scalar score? |

The conditional PIT target is the most powerful because it directly tests whether the hidden state predicts systematic deviations in the PIT values from uniformity. If the score-conditioned distribution from Experiment 1 is adequate, the PIT should be approximately uniform and not predictable from `H`. If `H` consistently maps certain regions to PIT values near 0 or 1, it carries information the score has discarded.

### 7.3 Probe design with dimensionality controls

The comparison between a scalar score `S` and a high-dimensional hidden state `H` (typically 64-512 dimensions in an LSTM) is confounded by dimensionality. A more flexible probe on `(S, H, C)` has vastly more capacity than one on `(S, C)`, and even with regularisation and cross-validation, a positive P2-P1 gap could reflect overfitting to the additional dimensions rather than genuine distributional information.

The experiment therefore uses three controls, as proposed in the adversarial dialogue.

#### Control A: Randomised hidden-state control

Compare four probes:

| probe | inputs | what it measures |
|---|---|---|
| P1 | `S, C` | Baseline: what the scalar score and context predict |
| P2 | `S, H, C` | Full: what the hidden state adds |
| P3 | `S, H_tilde, C` | Control: what random noise of the same dimensionality adds |
| P4 | `H, C` | Hidden-only: what the hidden state predicts without the score |

`H_tilde` is a within-stratum permutation of the hidden-state rows. Strata are defined by spatial identifier (to preserve the empirical hidden-state distribution within each location while breaking the correspondence between the hidden state and the specific observation's outcome). This is preferable to iid noise because it preserves the empirical covariance structure of `H`.

The expected result under H2 (the hidden state carries distributional information):

```
Performance(P2) > Performance(P1) ≈ Performance(P3)
```

If the randomised control produces the same improvement as the actual hidden state, the apparent gain is dimensionality, not information.

#### Control B: Low-rank bottleneck

Do not immediately give the probe access to all hidden dimensions. Compare projections:

```
H -> Z_k, k in {1, 2, 4, 8, 16, full}
```

using PCA-compressed hidden states (principal components fitted within the training fold to avoid leakage).

If distributional information is concentrated in a small number of dimensions (e.g., most of the gain appears at k=4 or k=8), this is stronger evidence for genuine information than if the gain requires the full hidden state. Concentrated information is a structural signal; distributed information is more likely to be overfitting.

Also compare:
- PCA-compressed `H` at dimension `k`.
- Random projection of `H` to dimension `k` (as a further control).
- Learned low-rank projection (a small linear bottleneck, fitted within the training fold).

#### Control C: Capacity-controlled probes

All probes use the same architecture, the same training budget, the same regularisation strategy, and the same hyperparameter-selection procedure. Specifically:

- **Architecture:** A shallow MLP (2 hidden layers, 64 units each, ReLU activations, dropout 0.3). This is deliberately small to prevent overfitting to high-dimensional inputs.
- **Training:** Adam with weight decay 1e-4, early stopping on validation loss with patience 10.
- **Regularisation:** L2 penalty on all weights, applied identically across probes.
- **Hyperparameter selection:** The same grid of learning rates and weight-decay values for all probes, selected by validation loss within the fit window.
- **Validation:** Temporal block validation with at least 3 folds. Blocks of at least 3 months. All probes are trained and evaluated on identical splits.

A high-dimensional probe must not get a more generous optimisation budget than the baseline.

### 7.4 Hidden-state extraction

The hidden state must be extracted from the autoregressive inference loop, not from teacher-forced inference. The teacher-forced hidden state reflects a network that has seen the truth at every step; the autoregressive hidden state reflects the network operating under its own predictions. The conditional distribution programme operates on autoregressive forecasts, so the hidden states must come from the same setting.

The existing infrastructure for hidden-state extraction already exists in the codebase: the smoothness localisation work (`M92`) used forward hooks on all 11 named stages of `HydraBNUNet06_LSTM4`. The same approach can extract the LSTM hidden state `h_t` and cell state `c_t` at each inference step.

**Storage.** For 64,818 cells, 36 forecast horizons, and a hidden state of dimension `d` (typically 64-512), the storage per cube is `64818 * 36 * d * 4 bytes`. At d=256, this is ~24 GB per cube. If storage is a constraint, extract hidden states only at selected horizons (month 1, month 6, month 12, month 36) and expand if the audit finds signal.

### 7.5 Sample size

The archive from Experiment 1 provides the forecast-observation pairs. For the probe analysis:

- Total pairs per fold: `64,818 cells * scoring_months` (at 13 months: 842,634).
- Event pairs per fold: approximately 2,865 (0.34% of total).
- For the zero indicator target: all pairs are informative.
- For the tail exceedance target: approximately 143 events above the 95th percentile per fold (5% of 2,865).
- For the bulk residual magnitude: all pairs are informative, but the signal is concentrated in event cells.

The zero indicator and bulk residual targets have adequate sample sizes. The tail exceedance target is marginal - 143 events per fold is enough for a binary classification probe but may be too few for stable cross-validated estimates. If tail exceedance results are unstable across folds, lower the threshold (90th percentile, giving ~287 events per fold).

### 7.6 Decision gate 2

| finding | conclusion | next step |
|---|---|---|
| P2 substantially outperforms P1 AND the improvement survives the randomised control (P3) AND the improvement is concentrated in a few dimensions | **H2 supported.** The hidden state carries distributional information the scalar score discards. | Proceed to Experiment 3: representation-conditioned distribution estimation. |
| P2 outperforms P1 but the randomised control (P3) shows a similar improvement | **Dimensionality confound.** The apparent gain is probe capacity, not hidden-state information. | Do not proceed to Experiment 3. Return to Experiment 1's best estimator. |
| P2 does not outperform P1, regardless of P3 | **H2 not supported.** The hidden state carries no additional distributional information beyond the scalar score. | Do not proceed to Experiment 3. The representation-conditioned branch is dead. Adopt Experiment 1's best estimator. |
| P4 (hidden-only) outperforms P1 (score-only) | The hidden state contains information the score does not, and the score may not be the best conditioning variable. | Worth investigating further, but primarily supports refining the conditioning score rather than building a full representation-conditioned estimator. |
| Original-input probes (if tested) outperform hidden-state probes | **H3 supported.** The representation itself is losing information. | Investigate auxiliary training objectives or architectural changes to the encoder. This is a different programme. |

"Substantially outperforms" means the criteria of §5.6 are met.

### Reviewer critique (opponent seat, 2026-11-03)

§7.3's three controls are the strongest methodological work in the document - Control A in
particular, and the choice of within-stratum permutation over iid noise is correct and well
reasoned. Four repairs, three of them bookkeeping, which matter because §12.5 makes this file the
pre-registration and the labels have to survive into §11.

**1. The probe labels collide with `T-2.2`.** Here `P3` is the permuted-`H_tilde` control and `P4`
is hidden-only. In §11.2, `P3` is the **original-input probe**, `P4` is the H-residual probe, and
the permuted control is renamed `C1`. §5.6's criterion 3 - a pre-specified success criterion - grades
against *"the permuted `H_tilde`"*, which §11 does not label `P3`. Pick one naming and propagate it
through §5.6, §7.3, §7.6 and §11.2.

**2. §7.2's four targets are not the four `T-2.1` builds.** Here: zero indicator / bulk residual
magnitude / upper-tail exceedance / conditional PIT. There: dispersion ratio / upper-tail exceedance
/ event onset / conditional PIT residual. The **zero indicator is dropped** - the one target §7.5
says has adequate sample size, the one that speaks directly to a 99.66%-zero population, and the one
that would test `M66`'s composition finding in the probe framework. "Event onset" arrives with no
§7 specification. Keep the zero indicator; if event onset is wanted, specify it here.

**3. §7.4's storage figure is an order of magnitude high, and the correction is good news.**
`64,818 x 36 x 256 x 4 bytes = 2.4 GB`, not ~24 GB. §8.3's 860 MB for `842,634 x 256 x 4` is
correct, and the two figures differ by 28x where the underlying forecast counts differ by 2.8x. The
horizon subsetting §7.4 proposes as a fallback is therefore unnecessary - the full archive fits on
one machine, and the audit can run at every horizon from the start.

**4. §7.4's alignment check deserves a named gate, not a task bullet.** Item 2 - confirming that the
hidden state at `(i,j,t)` matches the prediction and observation at `(i,j,t)` - is the one failure
that would produce a false positive on H2 and survive every control in §7.3, because a one-step
misalignment preserves the covariance structure that `H_tilde` is designed to control for. The
dossiers this programme cites use declared `G-` gates committed before the run (`D-5`); this check
should be one.


---

## 8. Experiment 3 - Representation-conditioned distribution estimation

**Priority:** Third. Runs only if Experiment 2 identifies meaningful additional information in the hidden state (decision gate 2 supports H2).

**Question:** Does conditioning on the hidden state produce a materially better conditional distribution than conditioning on the scalar score?

This experiment moves from "does the hidden state contain information?" (Experiment 2) to "can that information improve a conditional distribution?" (Experiment 3). The distinction matters because information that is statistically detectable in a probe does not automatically translate into a better distributional estimator. The estimator must contend with the curse of dimensionality, the sparsity of the event population, and the practical constraints of a high-dimensional conditioning variable.

### 8.1 The comparison

Take the existing frozen recurrent network. Extract historical hidden representations and their corresponding observed outcomes:

```
D = {(h_i, s_i, y_i, c_i)}_{i=1}^{n}
```

Compare conditional distributions estimated from three conditioning variables:

| estimator | conditioning | what it tests |
|---|---|---|
| E1 | `P(Y | S, C)` | Score-conditioned (Experiment 1's best estimator) |
| E2 | `P(Y | H, C)` | Hidden-state-conditioned |
| E3 | `P(Y | H, S, C)` | Both |
| E4 | `P(Y | Z_k, C)` where `Z_k = PCA_k(H)` | Low-rank hidden-state-conditioned |

The interesting question is not whether the hidden representation predicts counts better - that is a mean-prediction question, and the existing score already does that. The question is whether it predicts the **distributional characteristics** of counts better:

- Two observations with similar predicted magnitudes but different probabilities of extreme fatalities.
- Two observations with similar expected counts but different probabilities of zero fatalities.
- Two observations with similar rankings but different conditional spread.

### 8.2 Estimation methods

#### 8.2.1 Hidden-state-conditioned empirical distribution

The simplest approach: find historically similar hidden representations and construct an empirical distribution from their outcomes.

For a new representation `h`, construct weights:

```
w_i(h) = K(d(h, h_i) / tau) / sum_j K(d(h, h_j) / tau)
```

where `d` is a distance function on the hidden-state space and `tau` controls the neighbourhood width.

Distance options:

- **Euclidean distance on PCA-projected hidden states.** Use the top `k` principal components (from Experiment 2's bottleneck analysis) to reduce dimensionality before computing distances. If most information is in 4-8 dimensions, the full hidden state is unnecessary.
- **Cosine similarity.** More robust to scale differences across hidden-state dimensions.
- **Mahalanobis distance.** Accounts for the covariance structure of the hidden state. Requires estimating the covariance matrix, which may be unstable if the hidden-state dimensionality is large relative to the sample size.

Construct the predictive distribution:

```
F_hat(y | h) = sum_i w_i(h) * I(y_i <= y)
```

This is a representation-conditioned analog ensemble. Compare it directly against the score-conditioned analog (Experiment 1, candidate C).

#### 8.2.2 Learned similarity function

Rather than selecting a fixed distance metric, investigate learning a similarity function:

```
z_t = phi_theta(h_t)
```

where `phi_theta` is a small projection network (1-2 layers), trained to produce embeddings such that historically similar observations (in terms of outcome distribution) are close.

The projection is trained within the fit window using a contrastive or metric-learning objective:

- Pairs `(h_i, h_j)` with similar outcomes should produce similar `z_i, z_j`.
- Pairs with dissimilar outcomes should produce dissimilar embeddings.
- The definition of "similar outcomes" must respect the distributional targets: not just magnitude similarity, but similarity in the distributional characteristics that Experiment 2 identified as informative.

Regularisation is critical: without it, the learned retrieval mechanism can collapse onto a single historical observation, producing a point mass rather than a distribution. Approaches:

- Temperature scaling on the softmax weights.
- Minimum effective sample size constraint.
- Entropy regularisation on the weight distribution.

#### 8.2.3 Conditional KDE on the hidden state

Extend the empirical approach with outcome smoothing:

```
f_hat(y | h) = sum_i w_i(h) * K_{h_y}(y - y_i)
```

Use the same zero-preserving or log-scale smoothing strategies as Experiment 1, candidate D. The score bandwidth is replaced by a hidden-state bandwidth (the `tau` parameter in the kernel weights).

### 8.3 Practical constraints

**Computational cost.** Computing distances in the hidden-state space for every new forecast against every historical forecast is O(n * d) per prediction. With n = 842,634 historical forecasts and d = 256, this is feasible but not cheap. For operational use, approximate nearest-neighbour methods (FAISS, annoy) would be necessary. For the experiment, exact distances on the evaluation set are acceptable.

**Memory.** The hidden-state archive at d=256 for 842,634 forecasts requires ~860 MB. Manageable on a single machine.

**Temporal restriction.** The retrieval archive must obey the same temporal restrictions as Experiment 1: no observation from after the forecast origin may appear in the retrieval set. This prevents look-ahead and ensures the conditional distribution is genuinely predictive.

### 8.4 Decision gate 3

| finding | conclusion | next step |
|---|---|---|
| Hidden-state conditioning (E2 or E4) meets §5.5 success criteria and outperforms score-conditioning (E1) by at least 3% relative on event-cell CRPS | **H2 confirmed.** The hidden state carries operationally useful distributional information. | Investigate an integrated distributional head (§12, future work). |
| Hidden-state conditioning improves over score-conditioning but the improvement is less than 3% relative | **H2 weakly supported.** The hidden state carries distributional information, but the operational value is marginal. | Adopt the score-conditioned estimator from Experiment 1. The hidden-state direction is parked, not dead, pending a representation that carries more information. |
| Hidden-state conditioning does not improve over score-conditioning | **H2 not supported operationally.** Even though Experiment 2 detected distributional information in the hidden state, it does not translate into a better distribution estimator. | Adopt Experiment 1's best estimator. The information exists but is not usable at the current sample size or with the current estimation methods. |
| Learned similarity outperforms fixed distance | The hidden state's distributional information is encoded in a subspace that Euclidean distance does not align with. | The learned similarity is the operationally preferred method, but it introduces a training loop (the similarity function) that must be evaluated for stability across folds. |
| Low-rank conditioning (E4 at k=4 or k=8) matches full hidden-state conditioning (E2) | The relevant information is low-dimensional. | Use the low-rank projection for operational deployment (cheaper, more stable). |

### Reviewer critique (opponent seat, 2026-11-03)

§8.1's framing - that the interesting question is distributional characteristics rather than
magnitude, with three concrete contrasts - is the right question and well put. Two fixes.

**The estimator labels collide with Phase 3.** §8.1 defines `E4` as the low-rank PCA-conditioned
estimator. `T-3.4` defines `E4` as the conditional KDE, and `T-3.3` gives `E3` to learned similarity
where §8.1 gives it to `P(Y | H, S, C)`. As a result the §8.4 gate row above - *"Low-rank
conditioning (`E4` at k=4 or k=8) matches full hidden-state conditioning (`E2`)"* - cannot be
evaluated against the curriculum as written. One naming, propagated through §8.1, §8.4, §11.4 and
§9's tree.

Minor: §8.1's lead-in says *"three conditioning variables"* and the table lists four.

**§8.3's arithmetic is the correct one** (860 MB at d=256 for 842,634 forecasts) and should replace
§7.4's figure.


---

## 9. The decision tree

This section states the full decision tree in one place, consolidating the per-experiment gates from §6.4, §7.6 and §8.4.

```
START
  |
  v
Experiment 1: Frozen-backbone distribution challenge
  |
  |---> Mixture NB meets §5.5? ----YES----> ADOPT mixture NB. STOP.
  |                                          (H4 supported. Programme complete.)
  |
  |---> Non-parametric estimator    YES----> ADOPT best non-parametric estimator.
  |     meets §5.5?                          (H1 supported.)
  |         |                                Experiment 2 is optional enrichment.
  |         |
  |         NO
  |         |
  |         v
  |     Nothing meets §5.5 from score conditioning alone.
  |     (H1 and H4 not supported by score conditioning.)
  |         |
  |         v
  Experiment 2: Representation information audit
    |
    |---> Hidden state carries       YES----> Experiment 3: Representation-conditioned
    |     distributional info                 distribution estimation
    |     beyond score (§5.6)?                   |
    |         |                                  |---> Meets §5.5? --YES--> ADOPT.
    |         |                                  |                         (H2 supported.)
    |         |                                  |---> NO -----------------> ADOPT Exp 1 best.
    |         |                                                             Hidden state info
    |         |                                                             exists but is not
    |         |                                                             operationally useful.
    |         NO
    |         |
    |         v
    |     Hidden state carries no additional
    |     distributional information.
    |     (H2 not supported.)
    |         |
    |         v
    |     ADOPT Experiment 1's best estimator
    |     (even if it does not meet §5.5).
    |     The representation-conditioned
    |     branch is dead.
    |         |
    |         v
    |     If Experiment 1's best also does not
    |     meet §5.5: the distributional problem
    |     cannot be solved by post-processing.
    |     Investigate:
    |       - Retraining the backbone (RP-5: objective)
    |       - Fixing the decoder (M92: architecture)
    |       - Estimating the Bayes floor (RP-10)
    |       - The state-based direction
    |
    v
END
```

### 9.1 Falsification table

| observation | conclusion |
|---|---|
| Conditional KDE improves substantially using frozen scalar predictions | Strong evidence that the existing distribution head is the bottleneck (H1) |
| Two-component NB performs as well as or better than KDE | Parametric mixture is a viable solution (H4); no reason to insist on non-parametric estimation |
| Hidden state predicts tail/dispersion targets beyond scalar score AND survives randomised controls | The scalar prediction discards useful distributional information (H2) |
| Hidden-state improvement disappears under low-rank and control experiments | Do not invest in a learned retrieval architecture |
| Original inputs predict tail/dispersion targets beyond hidden state | The recurrent representation is losing relevant information (H3) |
| Neither hidden state nor original inputs predict distributional variation beyond score | No evidence that a richer representation will help; investigate aggregate conditional distributions and irreducible uncertainty |
| No estimator improves, even with additional representation information | Investigate training objective (RP-5), decoder architecture (M92/`#491`), target definition (RP-10), and sample support before changing architecture |

### 9.2 What would change the decision tree itself

The tree assumes the delivered autoregressive cube has enough ranking quality (AUC ~0.72) for conditional estimation to work in principle. If this assumption is wrong - if no conditional estimator improves over unconditional climatology - the entire programme is premature and the ranking must be improved first.

The tree also assumes the evaluation infrastructure (fair CRPS, PIT, tail calibration) is trustworthy. If the fair CRPS estimator proves unreliable at the available sample sizes, or if the PIT computation is unstable with the discrete count distribution, the evaluation must be debugged before any decision gate is evaluated.

### Reviewer critique (opponent seat, 2026-11-03)

Consolidating the gates in one place is what makes this a programme rather than a roadmap, and §9.2
asking what would change the tree itself is the right question. Three changes make the tree
executable.

**1. Reconcile the mixture branch with §6.4** (see the §6 note): §6.4 adopts on a tie, §9 requires
§5.5.

**2. §9.2 names an assumption that is weaker than the one actually carrying the tree.** It says the
tree assumes AUC ~0.72 is enough. The load-bearing assumption is narrower, more testable, and
already measured for one variable: **the conditioning variable must have support on the cells that
matter.** `M89` shows the composed index does not - 1,321 distinct values over 842,634 cell-months,
exactly 0.0 on 99.6567% of them, and an estimator conditioned on it scoring worse than the no-model
floor at every B >= 1,000. Restating §9.2 in those terms does three useful things: it makes the
un-composed index the programme's explicit bet, it makes `T-0.3` a genuine gate instead of a
formality, and it gives the tree a root node - *"does the conditioning variable have support on
event cells?"* - whose negative branch is already measured for one variable and open for the other.

**3. Add the terminal branch the record indicates.** §9's bottom-left node lists retraining
(`RP-5`), the decoder (`#491`), the Bayes floor (`RP-10`) and the state-based direction as the next
steps if nothing works. The composition belongs at the top of that list and is missing from it:
`M66` records that fixing firing at emit time costs nothing and needs no retraining, and `M90`(1)
attributes the point mass to it. It is cheaper than every other node on the tree, which is also the
argument for running it as candidate A' at the start rather than reaching it at the end.


---

## 10. Opponent audit

This section presents the adversarial review of the programme. The opponent's job is to find the claims that sound plausible but have not been verified, the experiments that are designed to confirm rather than to falsify, and the decision gates that can be gamed.

### 10.1 Findings

#### O-1. The "ranking is good" claim is weaker than stated

**The claim.** The neural network has useful ranking ability (§1.3, §4 preamble). This is the foundation for the entire conditional distribution programme.

**The challenge.** The AUC of 0.72 on autoregressive data is quoted as evidence of useful ranking. But AUC is a global summary that averages over the entire population, of which 99.66% are zeros. An AUC of 0.72 on a population that is 99.66% negative may mean the model is merely good at separating zeros from non-zeros (which the gate already handles), while being poor at ranking event magnitudes among non-zero observations.

**What the programme should do.** Report ranking quality separately for:
- The binary zero/nonzero classification (already handled by the gate).
- The magnitude ranking among event cells only (Spearman correlation restricted to cells where `truth > 0`).
- The tail ranking among cells above the 50th percentile of event magnitudes.

If the magnitude ranking among event cells is poor, conditional distribution estimation conditioned on the score will not capture distributional variation in the range that matters. The programme's value proposition depends on whether the score carries magnitude-ranking information, not just zero/nonzero separation.

**Severity:** Soft. The programme can proceed but must report the decomposed ranking as part of Experiment 1's baseline. If the event-cell magnitude ranking is below Spearman 0.3, the score-conditioned branch is unlikely to work and the programme should move directly to Experiment 2.

#### O-2. The conditional PIT target in Experiment 2 has a circularity

**The claim.** The conditional PIT (§7.2, target 4) directly tests whether the hidden state predicts systematic deviations from the score-conditioned distribution.

**The challenge.** The conditional PIT is computed using `F_hat_S(y | S, C)`, the score-conditioned distribution from Experiment 1. If that distribution is poorly estimated (too few bins, wrong bandwidth, poor kernel choice), the PIT residuals will contain apparent structure that is actually estimation error, not hidden-state information. The probe on `(S, H, C)` would pick up this spurious structure and falsely support H2.

**What the programme should do.** Use a conservative (deliberately underfit) score-conditioned distribution for the PIT computation. A coarse binning (250-500 equal-count bins) will leave residual score-predictable structure in the PITs, but the P1 probe on `(S, C)` will also capture that structure, so the P2-P1 difference should still isolate the hidden-state contribution. The risk runs the other way: an overfit score-conditioned distribution absorbs real H-predictable structure, leading to a false negative on H2. The underfit direction is conservative.

**Severity:** Observation. The circularity exists but the direction of bias is conservative (toward false negatives on H2, not false positives). Document the choice of PIT-computation distribution and its bin count in the pre-registration.

#### O-3. The 5% CRPS threshold in §5.5 is not grounded

**The claim.** A distributional estimator "solves the problem" if fair CRPS on event cells improves by at least 5% relative (§5.5, criterion 1).

**The challenge.** Where does 5% come from? The family-free CDF achieved 15% on teacher-forced data. The autoregressive degradation is substantial (AUC 0.998 to 0.723). The 5% threshold splits the difference but is not derived from any measurement. It could be too generous (accepting a noisy improvement that disappears on the next temporal fold) or too strict (rejecting a real improvement that the problem's noise floor makes difficult to detect).

**What the programme should do.** Ground the threshold in a power analysis. Given the event-cell sample size (~2,865 per fold), the temporal block bootstrap, and the variance of the CRPS difference estimator, what is the smallest detectable improvement with 80% power? If 5% is below the detection threshold, the criterion is vacuous. If 5% is well above it, the criterion may be wastefully strict for a first-pass experiment. The power analysis should be run during evaluation infrastructure setup (§5.8), not after seeing results.

**Severity:** Soft. The 5% is a placeholder that should be refined by measurement before the experiment runs.

#### O-4. Experiment 1 does not address the decoder smoothness problem

**The claim.** Experiment 1 conditions on the network's existing score, which reflects a spatially over-smooth mean field (§1.6).

**The challenge.** The decoder's spatial smoothness (`M92`) means the score compresses spatial variation. Two cells with different true risks receive similar scores because the mean field cannot distinguish them at fine spatial scales. A conditional distribution estimator inherits this compression: it bins by score, so cells that the decoder conflates end up in the same bin with the same estimated distribution, regardless of their true distributional differences.

**What the programme should do.** Acknowledge this as a ceiling on what score-conditioned estimation can achieve. Report results at different spatial scales: full grid, 8-cell neighbourhoods, and administrative regions. If conditional estimation works at coarser scales but not at fine scales, the decoder smoothness is the binding constraint, and the indicated next step is the architecture fix (`#491`), not further distributional work.

**Severity:** Observation. The decoder smoothness is a known issue documented in the state of knowledge. The programme cannot fix it (that is an architecture change, outside scope). But it should measure its impact rather than ignoring it.

#### O-5. The event-conditional NLL for the mixture NB has never been validated

**The claim.** The mixture NB should be fitted under event-conditional NLL or CRPS minimisation to avoid the zero-capture mechanism (§6.2, candidate B).

**The challenge.** Event-conditional NLL (fitting only on cells where `y > 0`) has a specific failure mode: it optimises the distribution of non-zero counts without any gradient from the zero mass. If the gate's predicted zero probability is miscalibrated - and the gate has a measured dispersion defect (slope b > 1, logits compressed) - then the hurdle decomposition inherits the gate's calibration error. The mixture NB would be well-fitted conditional on an event, but the overall distribution (zero probability times event distribution) could be worse because the zero probability is wrong.

**What the programme should do.** Include a gate-calibration diagnostic alongside the mixture NB evaluation. If the gate is miscalibrated, report the mixture NB results both with the raw gate probabilities and with post-hoc gate calibration (Platt scaling or isotonic regression on the gate logits). This separates the mixture NB's contribution from the gate's calibration error.

**Severity:** Soft. The gate calibration is a known issue (b > 1 in all fits). The mixture NB experiment should control for it.

#### O-6. The programme ignores the temporal structure of the problem

**The claim.** The conditional distribution estimators condition on the score (or hidden state) at a single forecast step.

**The challenge.** Conflict dynamics are temporal. The distribution of fatalities next month depends not just on the current score but on the trajectory: is the score rising or falling? Has the cell been in conflict for 3 months or 30? A conditional distribution that ignores the trajectory may conflate cells that are escalating with cells that are de-escalating, even when they have the same current score.

**What the programme should do.** Include the score's recent trajectory as a conditioning variable: `delta_s = s_t - s_{t-1}` or a short history `(s_{t-2}, s_{t-1}, s_t)`. This is cheap (no architecture change) and directly tests whether temporal context improves the conditional distribution. If it does, it supports moving to hidden-state conditioning (which carries temporal context implicitly via the LSTM state).

**Severity:** Observation. The trajectory is available for free and should be included as an additional conditioning variable in Experiment 1.

#### O-7. The sample size for tail exceedance probes is marginal

**The claim.** The upper-tail exceedance target in Experiment 2 uses approximately 143 events above the 95th percentile per fold (§7.5).

**The challenge.** 143 events is enough for a binary classification probe, but the tail is where the distributional problem is most severe. If the probe is unstable because of sample size, the experiment may produce a false negative on the most important distributional target.

**What the programme should do.** As stated in §7.5, use the 90th percentile (~287 events) if the 95th percentile is unstable. Additionally, pool across spatial strata if the probe design permits: rather than predicting per-cell exceedance, predict whether the cell-month's count exceeds its score-conditional expectation by more than 2 standard deviations. This reframes the target as a residual-based exceedance and increases the effective sample size.

**Severity:** Observation. The programme already flags this (§7.5). The pooling suggestion is an additional mitigation.

#### O-8. The programme does not state what happens to the existing evaluation framework

**The claim.** The programme uses its own evaluation (fair CRPS, PIT, tail calibration) rather than the frozen FAO-02 battery (§6.5, point 3).

**The challenge.** If the programme adopts a new distributional estimator, that estimator must eventually be evaluated within the FAO-02 framework for operational deployment. But 14/20 FAO-02 columns are blind to dispersion (§1.5), and the plug-in CRPS is biased. A distributional improvement that is invisible to the locked framework will not be recognised operationally, regardless of how well it scores on the programme's own metrics.

**What the programme should do.** State explicitly that an operationally successful distributional estimator requires a change to the evaluation framework. This is not within the programme's scope (changing FAO-02 is Simon's call), but the programme should document the dependency: adopting the result requires an evaluation amendment, and without it the improvement is invisible.

**Severity:** Soft. The programme correctly avoids the locked framework for its own evaluation, but the operational deployment path requires addressing the framework gap.

### 10.2 Opponent summary

The programme is structurally sound. The hypothesis structure is testable, the decision gates are explicit, and the experimental ordering is correct (cheapest falsification first). The opponent finds no hard falsification - no structural defect that would invalidate the programme's logic.

The soft findings cluster around three themes:

1. **Measurement gaps in the premises.** The ranking quality is quoted globally but not decomposed (O-1); the CRPS threshold is not grounded in a power analysis (O-3); the tail sample size is marginal (O-7).
2. **Upstream confounds.** The decoder smoothness compresses the score (O-4); the gate calibration affects the hurdle decomposition (O-5); the temporal trajectory is ignored (O-6).
3. **Operational path.** The programme's evaluation framework is independent of the locked FAO-02 battery (O-8), creating a deployment dependency.

None of these kill the programme. They constrain the interpretation of results and should be addressed in the experimental design (O-1, O-3, O-5, O-6, O-7) or documented as dependencies (O-4, O-8).

### Reviewer critique (opponent seat, 2026-11-03)

O-1 through O-8 are real findings, O-2's reasoning about the direction of the bias is the kind that
should appear in more of these documents, and O-5's gate-calibration catch is load-bearing for
`T-1.2`. Two structural notes, offered as additions.

**1. O-1 is right and aimed one statistic short.** Its challenge - that a global AUC on a
99.66%-negative population may reflect only zero/nonzero separation - is correct, and the measurement
that settles it sits in the same ledger row as the AUC it quotes: `M89`(2)'s **1,321 distinct values
and exactly 0.0 on 99.6567% of cells**. `T-0.3`'s fourth item proposes to re-derive it. Citing
`M89` upgrades O-1 from a caution into a result: closed for the composed index, open for the
un-composed one. That is a stronger finding than the one O-1 currently makes, and it is the finding
that sets Experiment 1's conditioning variable.

**2. §10.2's "no hard falsification" was reached by the seat that wrote §1 through §9.** Every
defect in these notes is available from this repository's own ledger and from this document's own
text, and the largest - `M89`'s absence - is the kind an independent read finds first and an
author's read structurally cannot. That is not a reason to drop §10. It is a reason to date it, name
the seat, and run one clean-context pass before `T-1.1`. The programme already pays for independence
everywhere else: Control A in §7.3, the committed-before-run discipline (`D-5`) and the `G-` copy
gates in every dossier it cites. §10 is the one place it does not, and it is the place where the
cost of a miss is a 12-to-16-week commitment rather than a number.


### 10.4 Library-grounded notes on making the opponent findings work (added 2026-11-03)

These notes do not change the opponent's text. They respond to each finding with what the library's literature says about how to address it - where the solutions already exist, where the real gaps sit, and what concrete steps each finding implies.

#### On O-1: The ranking decomposition is off-the-shelf

The opponent asks for decomposed ranking. The tools exist in the library and are stronger than what O-1 requests.

**Walz et al. (2026)** introduce the Coefficient of Monotone Association (CMA), with a mixture representation integrating AUC across all dichotomisation thresholds weighted by `alpha(1-alpha)` (Theorem 2.5). CMA depends much less on the discretisation level than Kendall's correlation (AGC vs AKC, their Figure 1). For the conflict-count case - a highly discrete non-negative integer - CMA is more stable than threshold-specific AUC decompositions. T-0.3 should report CMA alongside the decomposed AUCs.

**Arnold et al. (2024)** provide the CRPS decomposition `CRPS = MCB_ISO - DSC_ISO + UNC_0` via isotonic distributional regression, satisfying all five desiderata (Proposition 3.1). The discrimination component DSC_ISO directly measures what the opponent asks for: how much discriminatory ability survives recalibration. If DSC_ISO is near zero when restricted to event cells, the score carries no useful ranking among events - and we know this before running a single conditional estimator.

**Gneiting & Resin (2023)** extend this to any identifiable functional T via a universal `R* = DSC/UNC` that nests classical R-squared. Computing R* restricted to event cells directly answers O-1's question.

**What this means.** T-0.3 should use CMA (Walz et al.) and the Arnold et al. CRPS decomposition restricted to event cells. These are existing computations, not novel methodology. The reviewer critique's note about M89 is the empirical complement: the measurements already exist, the decomposition tools already exist, and the viability check at Checkpoint 0 has a stronger basis than the opponent assumes.

#### On O-2: Czado et al. (2009) and Biegert et al. (2026) jointly resolve the circularity

The opponent's conservative-direction argument is correct. The library sharpens both the fix and the hazard.

**Czado et al. (2009)** provide the nonrandomised PIT for count data, replacing the randomised PIT with a conditional CDF approach that is continuous on [0,1] (their eq. 2). This removes one source of noise in the PIT residual. Their diagnostic interpretation applies directly: U-shaped PIT histograms indicate underdispersion, inverse-U indicates overdispersion. If the score-conditioned PIT histogram is already approximately uniform at 250 bins, the circularity is moot.

**Biegert et al. (2026)** warn about a specific hazard: IDR's PIT values concentrate at simple rational fractions with frequencies following Thomae's function, due to the pool-block structure. If IDR (candidate E) is the score-conditioned estimator used for PIT computation in the probe target, this artifact will introduce spurious structure in the PITs that the probe detects as "information." The fix: use the KDE estimator (candidate D) for PIT computation, not IDR.

**What this means.** Use Czado et al.'s nonrandomised PIT. Use the KDE estimator (not IDR) as the reference distribution for the conditional PIT target. Check the PIT histogram at 250 bins before building the probe target.

#### On O-3: The threshold depends on more than sample size

The 5% threshold is indeed ungrounded. The library says it depends on the scoring rule's behaviour in the tail, not only on sample size.

**Lerch et al. (2017)** - the Forecaster's Dilemma - show that when evaluation is restricted to a low-probability region, the frequency of correct rejections under proper scoring rules decays to zero with increasing threshold (their Scenario B, Figure 7). Event-conditional CRPS is precisely this: evaluation restricted to cells where `y > 0`, which is a 0.34% event rate. The power analysis (T-0.2) must account for the scoring rule's inherent discrimination decay in the tail, not only sample-size variance.

**Bolin et al. (2023)** propose the Scaled CRPS (SCRPS), which is locally scale-invariant: the penalty is standardised by the forecast's own spread. For the programme, SCRPS may be a better primary metric than fair CRPS because it normalises for the very different scales across score bins. A 5% fair-CRPS improvement could be driven entirely by improved calibration in the largest-count bins while doing nothing for the bulk of events.

**What this means.** T-0.2 should report detection thresholds for both fair CRPS and SCRPS. Two metrics, two thresholds, reported together. If they diverge, the divergence itself is a finding about where the distributional improvement concentrates.

#### On O-4: The smoothness ceiling is quantifiable from Hyndman et al. (1996)

The opponent frames this as an observation. The literature makes it measurable.

**Hyndman et al. (1996)** establish that the kernel conditional density estimator has convergence rate `n^{-2/3}`, slower than unconditional density estimation's `n^{-4/5}`, because conditioning "effectively reduces the number of points used at each x value." If the decoder compresses distinct cells to similar scores, the effective conditioning sample per bin is larger (good for estimation) but the bins are impure (bad for capturing real distributional variation). The within-bin heterogeneity acts as irreducible bias that no bandwidth can remove.

**Goodfellow et al. (2016, Ch. 5)** state the complementary result: non-parametric methods relying on the smoothness prior can distinguish only as many regions as they have training examples. The decoder smoothness means the score's neighbourhood is larger than it should be, so the conditional distribution estimator pools genuinely different distributions.

**What this means.** T-1.6's spatial-scale analysis can be made precise: compute the within-bin variance of the outcome at each spatial resolution (full grid, 4-cell, 8-cell neighbourhoods). If within-bin variance drops substantially at coarser scales, the decoder smoothness is the binding constraint and the programme has measured its ceiling rather than merely noting it.

#### On O-5: The gate recalibration has a known optimal solution

**Zhang et al. (2025)** decompose rainfall imbalance into zero inflation and long tail, handled by a hurdle model and an imbalanced learning framework. Their key contribution: the hurdle model "provides a unified probabilistic framework that avoids the hard-threshold discontinuity of two-stage modelling, which breaks uncertainty propagation between rain detection and rate estimation stages." The programme's hurdle decomposition has the same structure and the same risk.

**Kuleshov et al. (2018)** provide the post-hoc recalibration: train an auxiliary monotone model `R: [0,1] -> [0,1]` via isotonic regression such that `R(p)` estimates the true conditional probability. This is what the programme needs for the gate.

**Henzi et al. (2021)** - IDR - is the strictly stronger version: IDR learns conditional distributions that are simultaneously optimal with respect to comprehensive classes of proper scoring rules, subject to isotonicity constraints. Applying IDR to the gate's predicted probabilities gives nonparametric recalibration that is CRPS-optimal by construction.

**What this means.** T-1.2 should include three gate variants: (a) raw gate, (b) Platt scaling, (c) IDR recalibration. IDR recalibration is strictly more general and adds no computational cost. If the gate calibration materially affects the mixture NB's performance, that is a finding about the architecture, not about the distributional estimator.

#### On O-6: The analog ensemble literature solves the trajectory problem

The opponent suggests score trajectory conditioning. The literature has the complete machinery.

**Delle Monache et al. (2013)** define the analog ensemble (AnEn): find the n best historical analogs using a multivariate weighted metric over a temporal window, then use their verifying observations as ensemble members. Their formulation conditions on a vector of predictor values across time - precisely the `(s_{t-2}, s_{t-1}, s_t)` trajectory the opponent suggests. The programme's empirical estimator (candidate C) is structurally an analog ensemble with the score as the sole predictor.

**Jeworrek et al. (2023)** extend the AnEn so that members are formed from the forecast error rather than the raw observation, enabling the ensemble to take values outside the historical range. This matters: if the score trajectory leads to a situation with no close historical analog (rapidly escalating conflict in a previously peaceful cell), the standard empirical distribution has no good matches. The error-based formulation allows extrapolation.

**What this means.** Candidate C (T-1.3) should be reframed as an analog-ensemble-style estimator conditioning on the temporal window, with optional error-based member construction. This is not a new estimator - it is a reframing using proven methodology. The temporal window makes O-6 concrete and testable.

#### On O-7: The tail probe is subject to the Forecaster's Dilemma

**Lerch et al. (2017)** show that evaluating forecasts conditional on observed extremes biases the comparison. The programme's tail exceedance probe (Experiment 2) evaluates a probe conditional on whether the count exceeds a high quantile. This is exactly the conditioning-on-extremes setup Lerch et al. warn about. A probe that predicts tail exceedance may succeed because the hidden state correlates with count magnitude, not because it predicts genuine distributional tail behaviour.

**MacDonald et al. (2011)** offer a different frame: the extreme value mixture model, where the threshold between bulk and tail is a parameter estimated jointly with the distribution. For the programme, this suggests a complementary probe target: rather than binarising at the 95th percentile, test whether the hidden state predicts variation in the bulk-tail transition threshold. This avoids the Forecaster's Dilemma because it tests structural tail behaviour, not exceedance probability.

**What this means.** The programme should acknowledge that the tail exceedance probe is subject to the dilemma. The 90th-percentile fallback helps with sample size but not with the conditioning bias. A complementary probe target based on MacDonald et al.'s bulk-tail threshold would be a stronger test of genuine tail information.

#### On O-8: The evaluation framework gap is quantifiable

**Arnold et al. (2024)** show that the Candille-Talagrand CRPS decomposition - the type most frameworks use - is degenerate: `MCB_CT = CRPS` whenever forecast CDFs are pairwise distinct (Proposition 2.1). The Hersbach decomposition can yield negative discrimination components. Only the isotonicity-based decomposition satisfies all five desiderata. If FAO-02 uses either of the first two, its CRPS scores are not decomposable into calibration and discrimination - it cannot tell whether a new estimator is better-calibrated or better-discriminating.

**Gneiting & Resin (2023)** establish that conditional calibration is the correct target, not marginal calibration. A forecast can be marginally calibrated while being conditionally miscalibrated at every covariate value. If FAO-02 checks only marginal calibration, a distributional improvement that fixes conditional calibration at event cells will be invisible.

**What this means.** T-4.1 should include a quantitative demonstration: compute the programme's metrics and FAO-02's metrics on the same estimator, and show where they diverge. This makes the evaluation amendment proposal (T-4.3) concrete rather than advisory. The strongest version: show that an estimator the programme adopts as an improvement is scored as equivalent or worse by FAO-02.


---

### Panel note - §10.4's citations, checked against the library (expert method review, 2026-11-03)

**All fifteen works cited in §10.4 are held in `9_library` with extracted claims.** That is unusual
for a reply of this length and it is the reason the panel in §14 could run at all. Claim IDs below
are the library's, so every line here is checkable without re-reading a PDF.

**Verified as stated.** Walz's mixture representation, Theorem 2.5, the `alpha(1-alpha)` weighting
and the AGC-vs-AKC stability result in Figure 1 (**C-3607**). Arnold's `CRPS = MCB_ISO - DSC_ISO +
UNC_0`, the five desiderata, and `DSC_ISO = 0` iff the IDR-calibrated forecasts are constant
(**C-3574**), plus the Candille-Talagrand degeneracy `MCB_CT = CRPS` and the Hersbach negative-DSC
failure (**C-3576**). Czado's nonrandomised PIT, its continuity on `[0,1]`, and the
U-shape/inverse-U reading (**C-3561**). Biegert's Thomae-function PIT concentration and its
pool-block mechanism (**C-4491**). Bolin's SCRPS and local scale invariance (**C-93**, **C-94**).
Hyndman's `n^{-2/3}` versus `n^{-4/5}` rate and the quoted explanation (**C-4402**). Zhang's hurdle
sentence, verbatim (**C-3042**). Kuleshov's auxiliary monotone `R: [0,1] -> [0,1]` by isotonic
regression (**C-1450**). Henzi's simultaneous optimality over comprehensive classes of proper scoring
rules (**C-4379**). AnEn's multivariate weighted metric over a temporal window (**C-4554**).
AnErrEn's out-of-archive members (**C-4568**). MacDonald's jointly estimated threshold (**C-4462**).

**Three readings to adjust, all in the same direction - the library supports the point more
strongly than §10.4 does.**

- *On O-3, the Lerch mechanism.* §10.4 says *"the frequency of correct rejections under proper
  scoring rules decays to zero with increasing threshold"*. `Lerch2017` **C-82** attributes that
  decay to *"all proper **weighted** scoring rules (twCRPS, CL, and CSL)"* in Scenario B, and
  **C-81**/**C-83** say the *unweighted* LogS is near-optimal by Neyman-Pearson while weighted rules'
  power gains are limited. So the decay is not a property of the unweighted event-conditional CRPS
  the programme uses. The conclusion - the dilemma applies - is right, and the mechanism that
  delivers it is **C-80**, which is a stronger finding: restricting to `truth > 0` is a nonconstant
  weight, and that makes the score *improper*, not merely low-powered. See the §5 panel note.
- *On O-5, the Zhang direction.* See the §6 panel note: **C-3042** makes the hurdle the fix, not the
  hazard.
- *On O-1, why CMA.* §10.4 argues CMA *"is more stable than threshold-specific AUC decompositions"*.
  **C-3607** establishes AGC's stability against **AKC / Kendall**, not against AUC, and **C-3604**
  has CMA *reduce to* AUC when the outcome is dichotomous - so that particular comparison is an
  extrapolation. More importantly, **C-3607**'s mixture representation is stated *"For continuous
  outcomes `Y_c`"*, which is the one case this programme is not in. The argument §10.4 wants is
  **C-3605**, and it is considerably stronger: CMA and CID have the *perfect monotone predictor
  property*, whereas *"symmetric rank correlation coefficients such as Spearman's Rho and Kendall's
  Tau **cannot attain their maximum value under perfect monotone dependence for non-continuous
  distributions**."* That is a direct hit on §5.3 and §12.4, which set degradation tolerances on
  event-cell **Spearman** - an instrument whose ceiling is below 1, unknown, and moves with the
  outcome's discreteness. See the §11 and §12 panel notes.

**One citation is not in the library and is relayed.** *"Gneiting & Resin (2023)"* and the universal
`R* = DSC/UNC`. The library holds `Resin2026_HierarchiesCalibration`, which *cites* Gneiting and
Resin (2023, Assumption 2.15) at **C-3599** but is a different paper, about calibration hierarchies.
The `R*` instrument §10.4 proposes for `T-0.3` therefore rests on a work we do not hold. It is the
right instrument and it should be fetched - see §14's gaps list.

> **CORRECTED, same sitting, before this note was reported.** The paragraph above is **wrong and is
> left standing deliberately**, in the house convention for a withdrawn reading. `9_library` holds
> **`Gneiting2023_ConditionalCalibration`** - *"Regression diagnostics meets forecast evaluation:
> Conditional calibration, reliability diagrams, and coefficient of determination"*, Gneiting and
> Resin, 2023 - and its **C-3572** carries the cited result verbatim: *"A score decomposition
> `S-bar = MCB - DSC + UNC` ... applies to any identifiable functional T with an associated
> consistent scoring function. A universal coefficient of determination `R* = DSC/UNC` nests the
> classical R-squared in least squares regression and its analog `R^1` in quantile regression, with
> `R*` in `[0,1]` subject to modest conditions."* **§10.4's citation is accurate and held, and
> nothing needs fetching for it.** The error was mine: I searched the library by first author and
> `Resin2026_HierarchiesCalibration` matched, so I stopped at the paper that *cites* Gneiting and
> Resin (2023) instead of searching for the paper itself, which is filed under Gneiting. An absence
> claim built on one spelling of one author's name is not an absence test - the same failure mode as
> a corpus-negative tightened until it excludes the real case.
>
> Two further things that paper holds and §10.4 did not use, both load-bearing here:
> **C-3570** - auto-calibration is the strongest notion and *"Example 2.4 resolves the conjecture of
> Gneiting and Ranjan (2013) by constructing continuous forecast distributions that are both
> marginally **and** probabilistically calibrated but fail to be auto-calibrated"* ⇒ §5.5 criterion
> 3's PIT gate can pass on a forecast that is not auto-calibrated, which is the failure the
> programme is hunting. **C-3571** - the **CORP** reliability diagram (isotonic / pool-adjacent-
> violators) gives *"non-decreasing reliability diagrams with guaranteed nonnegative MCB and DSC
> components"* for real-valued outcomes and general functionals. That is the diagnostic plot `T-1.1`
> and `T-1.6` should produce, and it is the same machinery as `Arnold2024`'s decomposition.
> **C-3573** names the intermediate notions (CEP, threshold, quantile calibration) and, crucially,
> that *"Under minimal conditions this equivalence breaks down"* - so the three are distinct on a
> discrete count and §5.4 must say which one it is testing.

**A provenance note, offered because it would otherwise cost someone a re-derivation.**
`DelleMonache2013` and `Jeworrek2023` are two of the four papers in the 2026-11-02 library ingestion
confirmed to carry **sign-glyph font corruption** - an AMS-journal text-layer encoding that maps
`=` to `5`, `+` to `1`, minus to `2`, `.` to `:`, `+/-` to `6`, the degree sign to `8` and the
multiplication sign to `3`. That ingestion's audit verified that **no corrupted numeral reached any
claim** in any of the four papers, so every figure quoted from those two here is safe. Any *new*
number pulled from those two PDFs needs a rendered-page read first, not a text-layer read.

**And one note on what §10.4 covers.** It answers O-1 through O-8 - the document's own eight
findings - thoroughly and with real literature. It does not engage the four findings the reviewer
critique added: the composition attribution and `M89`, the retired grid, the §5.5/§5.7 and
label contradictions, and the teacher-forced asymmetry. That is worth saying only because §10.4's
*method* is the right one and applying it to those four is what §14 does - and in two cases
(`Zhang2025` on the hurdle, `Bolin2023` on scale invariance) the library turns out to support the
programme's own design more strongly than §10.4 claimed.

---

## 11. Curriculum

This section sequences the programme's work into ordered tasks with explicit dependencies, estimated effort, and checkpoints. The ordering principle is: infrastructure first, cheapest falsification second, and each experiment's gate must be evaluated before the next experiment begins.

### 11.1 Phase 0: Infrastructure (weeks 1-3)

The infrastructure phase builds the evaluation and data extraction machinery that all three experiments require. Nothing in this phase tests a hypothesis. Its purpose is to ensure that when results arrive, the evaluation is trustworthy.

#### T-0.1. Evaluation infrastructure

**What.** Implement the programme's evaluation metrics as a reusable module:

1. **Fair CRPS estimator.** The fair (energy-score) CRPS estimator that is unbiased with respect to ensemble size. Implementation follows the formula from Zamo & Naveau (2018): `CRPS_fair = (1/M) * sum |X_m - y| - (1/(2*M*(M-1))) * sum |X_m - X_m'|`. Validate against the plug-in estimator at M=4096 (they should converge). Verify that the estimator is monotonically improving in M from 4 to 4096 (known from M90, `#459`).

2. **PIT computation.** Probability Integral Transform for discrete count distributions. For count data, the randomised PIT: `u = F(y-1) + v * p(y)` where `v ~ Uniform(0,1)`. Validate by computing PITs for a known well-calibrated distribution (e.g., Poisson data with known lambda) and confirming uniformity via the Kolmogorov-Smirnov test.

3. **Event-conditional metrics.** Wrapper that restricts any metric to the subset of observations where `y > 0`, and to the subset where `y > q_{0.95}(y | y > 0)`. Report sample sizes alongside all event-conditional metrics.

4. **Tail calibration diagnostic.** For a given conditional distribution, compute the coverage of the 90th, 95th, and 99th quantiles restricted to event cells. Report as a calibration table: nominal coverage vs. empirical coverage.

5. **Temporal block bootstrap.** For confidence intervals on all metrics. Block length = 6 months (the temporal autocorrelation scale of the conflict process). Number of bootstrap replicates = 1000. Validate by checking that the bootstrap CI for the existing NB baseline's CRPS contains the point estimate.

**Dependencies.** None. This is the first task.

**Effort.** 3-5 days. Most of the statistical machinery exists in standard libraries. The main work is the event-conditional wrapper and the temporal bootstrap.

**Deliverable.** A tested evaluation module with a validation notebook showing that each metric behaves as expected on synthetic data.

**Addresses.** O-3 (grounding the 5% threshold via power analysis), O-7 (tail sample size measurement).

#### T-0.2. Power analysis for the CRPS threshold

**What.** Using the evaluation infrastructure from T-0.1, estimate the minimum detectable CRPS improvement.

1. Compute fair CRPS for the existing NB baseline on each temporal fold of the autoregressive cube.
2. Compute the variance of the CRPS difference estimator between two hypothetical estimators using the temporal block bootstrap.
3. Determine the minimum detectable improvement at 80% power and alpha=0.05 for: (a) full population, (b) event cells only, (c) tail cells only (above 95th percentile of event magnitudes).
4. If the 5% threshold from §5.5 is below the detection limit for event cells, revise the threshold upward. If 5% is far above the detection limit, consider whether a stricter threshold (3%, 2%) is feasible.

**Dependencies.** T-0.1.

**Effort.** 1-2 days.

**Deliverable.** Revised quantitative success criteria (§5.5) grounded in measurement, not assumption.

**Addresses.** O-3 directly.

#### T-0.3. Ranking quality decomposition

**What.** Decompose the global AUC=0.72 from the autoregressive cube into components that matter for conditional estimation.

1. **Binary AUC.** AUC for the zero/nonzero classification task. Expected to be high (the gate handles this).
2. **Event-cell Spearman.** Spearman correlation between predicted score and observed count, restricted to cells where `truth > 0`. This is what score-conditioned estimation depends on.
3. **Tail-cell Spearman.** Spearman correlation restricted to cells above the 50th percentile of event magnitudes.
4. **Score histogram among event cells.** How many distinct score values exist? If the score is effectively binary (zero vs. nonzero) among event cells, conditional estimation has nothing to condition on.

**Dependencies.** Access to the autoregressive cube (already exists).

**Effort.** 1 day.

**Deliverable.** A table of ranking metrics with the decomposition. If event-cell Spearman < 0.3, flag that the score-conditioned branch of the programme is unlikely to work.

**Addresses.** O-1 directly. This is a programme viability check.

**Checkpoint 0.** Review T-0.1 through T-0.3 results before proceeding. If the ranking decomposition reveals that event-cell magnitude ranking is poor (Spearman < 0.3), skip Experiment 1's score-conditioned estimators and move directly to Experiment 2 (the hidden state may carry information the scalar score discards). Document the decision.

#### T-0.4. Hidden-state extraction infrastructure

**What.** Build the machinery for extracting hidden states from the trained network. This is needed for Experiments 2 and 3 but should be built early to amortise the cost.

1. **Hook registration.** Register forward hooks on the LSTM's final hidden state layer to capture `h_t` for every cell-month during autoregressive rollout. Store as a memory-mapped array: shape `(T, H_lat, W_lat, d_h)` where `d_h` is the hidden dimension.

2. **Alignment verification.** Confirm that the extracted hidden state at cell `(i, j, t)` corresponds to the prediction for cell `(i, j, t)` and the observation at `(i, j, t)`. This is critical: a one-step misalignment between features and targets would invalidate Experiment 2.

3. **Storage format.** Memory-mapped NumPy array with a metadata sidecar (JSON) recording: model checkpoint, rollout configuration, extraction date, shape, dtype, and the mapping from array indices to PRIO-Grid cell-month identifiers.

4. **Dimensionality characterisation.** Compute the eigenspectrum of the hidden-state covariance matrix (across all cell-months with `y > 0`). Report the effective dimensionality (number of components explaining 95% of variance). This informs the low-rank bottleneck design in Experiment 2.

**Dependencies.** Access to the trained model checkpoint and the autoregressive rollout code. Independent of T-0.1 through T-0.3.

**Effort.** 3-5 days. The hook registration is straightforward but the alignment verification requires careful bookkeeping of the autoregressive unrolling.

**Deliverable.** A hidden-state extraction script with alignment tests, plus a dimensionality report.

#### T-0.5. Score trajectory conditioning variables

**What.** Compute the score trajectory features suggested by opponent finding O-6.

1. **Delta score.** `delta_s_{i,j,t} = s_{i,j,t} - s_{i,j,t-1}` for each cell-month.
2. **Short history.** The triple `(s_{i,j,t-2}, s_{i,j,t-1}, s_{i,j,t})` for each cell-month.
3. **Conflict duration.** Number of consecutive prior months with `y > 0` at cell `(i, j)` as of month `t`. Capped at 36 (3 years).

Store alongside the score cube. These are cheap conditioning variables for Experiment 1.

**Dependencies.** Access to the autoregressive cube. Independent of other T-0 tasks.

**Effort.** 0.5 days.

**Deliverable.** Trajectory features aligned with the score cube.

**Addresses.** O-6.

### 11.2 Phase 1: Experiment 1 - Frozen-backbone distribution challenge (weeks 3-7)

This phase implements and evaluates the five candidate estimators from §6. The backbone (trained network) is frozen. All conditioning is on the scalar score (and trajectory features from T-0.5).

#### T-1.1. Baseline measurement (candidate A)

**What.** Compute all §5 evaluation metrics for the existing NB distribution head on the autoregressive cube.

1. Fair CRPS (full, event-conditional, tail-conditional).
2. PIT uniformity (KS test, full and event-conditional).
3. Tail calibration (90th, 95th, 99th coverage on event cells).
4. Event-conditional NLL.

This establishes the baseline against which all candidates are compared.

**Dependencies.** T-0.1, T-0.2 (for the finalised thresholds).

**Effort.** 1 day.

**Deliverable.** Baseline evaluation table.

#### T-1.2. Two-component NB mixture (candidate B)

**What.** Fit a two-component NB mixture to the score-conditioned data, using the event-conditional CRPS objective to avoid the zero-capture mechanism.

1. **Hurdle decomposition.** Use the existing gate probability `pi_{i,j,t}` from the network. Fit the mixture only on cells where `y > 0`.
2. **Parameterisation.** For each score bin (or via a smooth function of score): two NB components with parameters `(mu_1, theta_1, w_1)` and `(mu_2, theta_2, w_2)` where `w_1 + w_2 = 1`. The mixture mean is constrained to match the network's predicted `mu` (so the ranking is preserved).
3. **Fitting.** Minimise the event-conditional CRPS over the mixture parameters, using the fair CRPS estimator with M=1024 samples from the mixture. Optimise via L-BFGS-B with multiple random initialisations (10 per score bin).
4. **Gate calibration diagnostic.** Fit Platt scaling on the gate logits. Report mixture results with both raw and calibrated gate probabilities.
5. **Score binning.** Equal-count bins of the score among event cells. Start with 100 bins; check sensitivity to 50 and 200.

**Dependencies.** T-0.1, T-1.1 (baseline for comparison).

**Effort.** 5-7 days. The main challenge is the optimisation: the mixture likelihood surface has local optima, and the CRPS objective is more expensive to evaluate than NLL.

**Deliverable.** Mixture NB evaluation table, comparison to baseline (A), gate calibration report.

**Addresses.** O-5 (gate calibration diagnostic).

#### T-1.3. Score-conditioned empirical distribution (candidate C)

**What.** For each score value (or bin), collect the empirical distribution of observed counts from the training data.

1. **Bin construction.** Equal-count bins of the score among all cell-months (not just events). Start with 500 bins; check sensitivity from 250 to 1000.
2. **Leave-one-fold-out.** Build the empirical distribution using training folds only. Evaluate on the held-out fold.
3. **Zero-handling.** The empirical distribution naturally includes the zero mass. No hurdle decomposition needed.
4. **CRPS computation.** The CRPS of a discrete empirical distribution is computed exactly (no sampling needed). This avoids the plug-in/fair distinction entirely.

**Dependencies.** T-0.1, T-1.1.

**Effort.** 2-3 days. Conceptually simple but requires careful cross-validation bookkeeping.

**Deliverable.** Empirical distribution evaluation table, comparison to A and B.

#### T-1.4. Score-conditioned KDE (candidate D)

**What.** Kernel density estimation conditioned on score, with two variants: zero-preserving and log-scale.

1. **Zero-preserving KDE.** Hurdle decomposition: estimate `P(y=0 | score)` from the training data, then estimate the density of `y | y > 0, score` using a KDE on the positive counts. Bandwidth selection via Sheather-Jones (leave-one-out cross-validation within each score bin).

2. **Log-scale KDE.** Transform `y > 0` to `log(y + 1)`, fit KDE in log space, transform back. This handles the heavy tail without needing a specialised kernel.

3. **Score conditioning.** Use the score trajectory features from T-0.5 as additional conditioning variables. Start with score alone; add `delta_s` and `conflict_duration` to test whether trajectory improves the conditional distribution.

4. **Bandwidth sensitivity.** Report results for the Sheather-Jones optimal bandwidth and for 0.5x and 2x that bandwidth.

**Dependencies.** T-0.1, T-0.5, T-1.1.

**Effort.** 3-5 days. The KDE itself is standard but the conditioning and bandwidth selection require care.

**Deliverable.** KDE evaluation table (both variants, with and without trajectory features), comparison to A, B, C.

**Addresses.** O-6 (trajectory conditioning).

#### T-1.5. Isotonic distributional regression (candidate E)

**What.** IDR using the score as the single ordering variable.

1. **Implementation.** Use the `isodistrreg` R package (or reimplement in Python). IDR estimates the conditional CDF `F(y | score)` under the constraint that the CDF is stochastically ordered in the score.
2. **Calibration.** IDR is inherently calibrated by construction (isotonic regression). Report PIT uniformity as a sanity check.
3. **Score trajectory.** If the multivariate IDR (partial-order version) is computationally feasible, include `delta_s` as a second ordering variable.

**Dependencies.** T-0.1, T-1.1.

**Effort.** 2-3 days. The R package is mature; the main work is the Python/R interface and the evaluation integration.

**Deliverable.** IDR evaluation table, comparison to A, B, C, D.

#### T-1.6. Experiment 1 synthesis and decision gate 1

**What.** Compile results from T-1.1 through T-1.5. Evaluate decision gate 1 (§6.4).

1. **Comparison table.** All five candidates on all §5 metrics, with bootstrap confidence intervals.
2. **Spatial scale analysis.** Report the best candidate's improvement at three spatial scales: full grid, 8-cell neighbourhood averages, and administrative regions (per O-4).
3. **Decision gate evaluation.** Does any candidate meet §5.5? If mixture NB meets §5.5, adopt it and stop (H4 supported). If a non-parametric estimator meets §5.5, adopt it (H1 supported). If nothing meets §5.5, proceed to Experiment 2.
4. **Trajectory effect.** For candidates C, D, and E, does adding trajectory features improve results? If yes, this suggests temporal context matters and supports moving to hidden-state conditioning (which carries temporal information implicitly).

**Dependencies.** T-1.1 through T-1.5.

**Effort.** 2-3 days for synthesis.

**Deliverable.** Experiment 1 report with decision gate evaluation. Updated RESEARCH_PROGRAMME.md if a candidate is adopted.

**Checkpoint 1.** This is the first major decision point. If the programme stops here (H4 or H1 supported from score conditioning alone), the remaining phases are optional enrichment. If the programme continues, all T-1 results carry forward as the "best score-conditioned estimator" baseline for Experiments 2 and 3.

### 11.3 Phase 2: Experiment 2 - Representation information audit (weeks 7-11)

This phase tests whether the hidden state carries distributional information beyond the scalar score. It requires the hidden-state extraction from T-0.4 and the evaluation infrastructure from T-0.1.

#### T-2.1. Probe targets preparation

**What.** Construct the four probe targets from §7.2.

1. **Dispersion ratio.** For each cell-month with `y > 0`: compute `r_{i,j,t} = (y_{i,j,t} - E[y | score])^2 / Var[y | score]`, where expectation and variance are estimated from the training data's score-conditional distribution. Binarise at threshold 2 (residual exceeds twice the expected variance).

2. **Upper-tail exceedance.** Binary: `y_{i,j,t} > q_{0.95}(y | y > 0)`. If the 95th percentile yields fewer than 143 events per fold, use the 90th percentile (per O-7 and §7.5).

3. **Event onset.** Binary: `y_{i,j,t} > 0` and `y_{i,j,t-1} = 0`. This targets the first month of a new conflict episode.

4. **Conditional PIT residual.** Using the best score-conditioned distribution from Experiment 1 (call it `F_hat`), compute the randomised PIT `u_{i,j,t}`. Binarise at `|u - 0.5| > 0.3` (departure from uniformity). Use a deliberately underfit score-conditioned distribution (250 equal-count bins) to avoid the circularity identified in O-2.

Report sample sizes for each target at each temporal fold.

**Dependencies.** T-0.1, T-0.4 (hidden states), T-1.6 (best score-conditioned estimator for target 4).

**Effort.** 2-3 days.

**Deliverable.** Probe target arrays aligned with the hidden-state array. Sample size report.

#### T-2.2. Probe implementation (P1-P4 with controls)

**What.** Implement the four probe configurations from §7.3-7.4.

1. **P1: Score-only probe.** `f(S, C) -> target`. Logistic regression with L2 regularisation. Features: score, score^2, log(score+1), conflict duration. This is the baseline: it measures how much the scalar score already predicts each distributional target.

2. **P2: Full probe.** `f(S, H, C) -> target`. Same as P1 but with the full hidden-state vector appended. Features: all P1 features plus the full `h_t` vector.

3. **P3: Input probe.** `f(S, X_original, C) -> target`. Same as P1 but with the original input features (the raw grid-cell features that entered the encoder) appended. This tests H3: whether the inputs contain distributional information that the representation lost.

4. **P4: Score-conditioned H probe.** `f(H_residual, C) -> target`. Regress `H` on `S` (linear), take residuals. Probe targets from `H_residual`. This isolates the hidden-state contribution beyond what the score already captures.

**Control experiments** (§7.4):

5. **C1: Randomised H.** Replace `H` with a random permutation across cell-months (preserving marginal distribution but destroying the cell-month alignment). The P2-equivalent probe on randomised H should not improve over P1. If it does, the probe is picking up distributional properties of H rather than cell-month-specific information.

6. **C2: Low-rank bottleneck.** Replace `H` with its projection onto the top-k principal components (k=4, 8, 16, 32). If the P2-equivalent probe with k=8 matches the full-H probe, the relevant information is low-dimensional.

7. **C3: Capacity-controlled probes.** For each probe (P1 through P4), report results at three capacity levels: (a) logistic regression (linear), (b) one-hidden-layer MLP with 32 units, (c) one-hidden-layer MLP with 128 units. If the MLP substantially outperforms logistic regression, the information is there but is nonlinearly encoded.

**Dependencies.** T-0.4 (hidden states), T-2.1 (probe targets).

**Effort.** 5-7 days. The number of probe configurations is large (4 probes x 4 targets x 3 capacities + 3 controls x 4 targets x 3 capacities = 84 runs). Each individual run is fast (logistic regression on ~2,865 events), but the bookkeeping is substantial.

**Deliverable.** A results table with 84+ cells: probe configuration x target x capacity. Plus the control results.

#### T-2.3. Experiment 2 synthesis and decision gate 2

**What.** Evaluate decision gate 2 (§7.6).

1. **H-information test.** For each target: is `AUC(P2) - AUC(P1) > 0.05` (or the revised threshold from T-0.2's power analysis)?
2. **Control validation.** Does `C1` (randomised H) improve over P1? If yes, the probe design is flawed. Stop and debug.
3. **Dimensionality.** At what rank k does C2 match P2? This determines the practical dimensionality of the useful hidden-state information.
4. **Capacity.** Does the MLP substantially outperform logistic regression? If yes, the information is nonlinearly encoded and the conditioning mechanism in Experiment 3 must be nonlinear.
5. **H3 assessment.** Does P3 (input probe) outperform P2 (full probe) on any target? If yes, the recurrent representation is discarding relevant information.

**Decision gate evaluation:**

- If P2 does not improve over P1 on any target (after controls), the hidden state carries no additional distributional information. Adopt Experiment 1's best estimator. The programme's H2 branch is dead.
- If P2 improves over P1 on at least one target and survives controls, proceed to Experiment 3.
- Record which targets showed improvement (this determines the objectives for Experiment 3).

**Dependencies.** T-2.2.

**Effort.** 2 days for synthesis.

**Deliverable.** Experiment 2 report with decision gate evaluation. The report explicitly states whether H2 is supported and, if so, what the hidden state predicts that the score does not.

**Checkpoint 2.** If H2 is not supported, the programme enters its final phase: adopt Experiment 1's best and document the finding that the distributional problem cannot be solved by post-processing. If H2 is supported, proceed to Experiment 3 with a clear specification of what the hidden state adds.

### 11.4 Phase 3: Experiment 3 - Representation-conditioned estimation (weeks 11-15)

This phase builds distributional estimators that condition on the hidden state, not just the scalar score. It is the most ambitious and most expensive phase, and runs only if Experiment 2 confirmed that the hidden state carries useful distributional information.

#### T-3.1. Conditioning variable selection

**What.** Using Experiment 2's results, determine the conditioning variables for Experiment 3.

1. **If low-rank k=8 matches full H (from C2):** Use the top-8 PCA projection of H as the conditioning variable. This is cheaper and more stable than the full hidden state.
2. **If full H is needed:** Use the full hidden-state vector with a learned dimensionality reduction (autoencoder or random projection as a regulariser).
3. **Trajectory features.** If trajectory features improved Experiment 1's estimators (from T-1.6), include them.
4. **Score.** Always include the scalar score as a conditioning variable.

**Dependencies.** T-2.3 (decision gate 2 must support H2).

**Effort.** 1 day.

**Deliverable.** Specification of the conditioning variable set.

#### T-3.2. Empirical conditioning (estimator E1 and E2)

**What.** Build score-conditioned and hidden-state-conditioned empirical distributions.

1. **E1: Score-conditioned empirical (baseline from Experiment 1).** Carried forward from T-1.3.
2. **E2: Hidden-state-conditioned empirical.** Bin by the conditioning variable from T-3.1. For the low-rank case (k=8), use a multidimensional binning (e.g., 2^8 = 256 bins via recursive partitioning, or k-means clustering of the projected hidden states). Collect the empirical distribution of counts within each bin.

The key question: does conditioning on H in addition to S improve the conditional distribution?

**Dependencies.** T-3.1, T-1.3 (score-conditioned empirical baseline).

**Effort.** 3-5 days. The multidimensional binning is the hard part: too many bins yields empty bins and noisy estimates, too few loses the information that H carries.

**Deliverable.** E1 and E2 evaluation tables with all §5 metrics.

#### T-3.3. Learned similarity conditioning (estimator E3)

**What.** Instead of binning H, use a learned similarity function to weight training observations.

1. **Architecture.** A small network (2-layer MLP) that maps `(S, H, C)` to a similarity score between two cell-months. Train to predict whether two cell-months have similar count distributions (defined by whether their counts fall in the same quantile of the score-conditional distribution).
2. **Conditional distribution.** For a test cell-month, compute similarity to all training cell-months. Weight the training counts by similarity to form a weighted empirical distribution.
3. **Regularisation.** L2 on the network weights. Dropout. Early stopping on a validation fold.

**Dependencies.** T-3.1, T-0.4 (hidden states).

**Effort.** 5-7 days. This is the most complex estimator. The training loop, the similarity computation, and the weighted distribution all require careful implementation.

**Deliverable.** E3 evaluation table, comparison to E1 and E2.

#### T-3.4. Conditional KDE (estimator E4)

**What.** KDE conditioned on the selected hidden-state projection, extending T-1.4.

1. **For low-rank case (k=8).** Product kernel in the 8-dimensional projected space. Bandwidth selection via leave-one-out cross-validation.
2. **For full H.** This is likely infeasible (curse of dimensionality). Report the failure if it occurs; this is evidence that the low-rank projection is necessary for practical estimation.
3. **Zero-preserving variant.** Same hurdle decomposition as T-1.4.

**Dependencies.** T-3.1, T-1.4 (score-conditioned KDE baseline).

**Effort.** 3-5 days.

**Deliverable.** E4 evaluation table, comparison to E1, E2, E3.

#### T-3.5. Experiment 3 synthesis and decision gate 3

**What.** Evaluate decision gate 3 (§8.4).

1. **Comparison table.** E1 through E4 on all §5 metrics, with bootstrap confidence intervals.
2. **Does hidden-state conditioning improve over score conditioning?** Compare E2/E3/E4 to E1 on event-conditional CRPS.
3. **Practical assessment.** If the improvement exists but is small (<2% relative CRPS), is it worth the operational complexity of extracting and conditioning on hidden states?
4. **Ranking preservation.** Does the hidden-state-conditioned estimator change the ranking? Report the ranking metrics from T-0.3 for the new estimator's predictive mean vs. the original score.

**Decision gate evaluation:**

- If no hidden-state-conditioned estimator meets §5.5: adopt Experiment 1's best estimator. The hidden-state information exists (Experiment 2 confirmed it) but cannot be operationally exploited.
- If a hidden-state-conditioned estimator meets §5.5 and ranking degradation is ≤2%: adopt it (H2 supported operationally).
- If ranking degradation exceeds 2%: the distributional improvement comes at the cost of point prediction quality. Document the trade-off. Consider whether a constrained estimator (ranking-preserving) can be constructed.

**Dependencies.** T-3.2 through T-3.4.

**Effort.** 2-3 days for synthesis.

**Deliverable.** Experiment 3 report with decision gate evaluation. Final programme recommendation.

**Checkpoint 3.** This is the programme's terminal checkpoint. The result is one of:
- Adopted estimator (mixture NB, score-conditioned, or hidden-state-conditioned).
- No estimator improves: the distributional problem cannot be solved by post-processing. The indicated next steps are: training objective (RP-5), decoder architecture (`#491`), Bayes floor (RP-10), or the state-based direction.

### 11.5 Phase 4: Documentation and integration (weeks 15-16)

#### T-4.1. Programme report

**What.** Write the final programme report as a FINDING.md in the views-hydranet reports directory. The report documents:

1. Which hypothesis was supported (H1, H2, H3, H4, or none).
2. What was adopted and why.
3. What was tested and did not work.
4. What the results mean for the RESEARCH_PROGRAMME.md queue (which RPs are advanced, which are blocked, which are resolved).
5. Updated state of knowledge for the distributional problem.

**Dependencies.** All prior phases.

**Effort.** 2-3 days.

#### T-4.2. Research programme queue update

**What.** Update RESEARCH_PROGRAMME.md:

1. Move RP entries that this programme addresses (re-rank if resolved or partially resolved).
2. Add any new RPs that emerged from the programme's findings.
3. Log the re-rank with rationale.

**Dependencies.** T-4.1.

**Effort.** 0.5 days.

#### T-4.3. Evaluation framework proposal (if applicable)

**What.** If the programme adopted an estimator, draft a proposal for amending the FAO-02 evaluation framework to include distributional metrics. This is not within the programme's scope to implement, but the proposal should exist so the operational path is clear.

**Dependencies.** T-4.1. Only if the programme adopts an estimator.

**Effort.** 1-2 days.

**Addresses.** O-8.

### 11.6 Summary timeline

| Week | Phase | Tasks | Key output |
|------|-------|-------|------------|
| 1-2 | Infrastructure | T-0.1, T-0.3, T-0.4, T-0.5 (parallel) | Eval module, ranking decomposition, hidden-state extraction, trajectory features |
| 3 | Infrastructure | T-0.2 | Power analysis, revised thresholds |
| 3 | Checkpoint 0 | Review | Programme viability check |
| 3-5 | Experiment 1 | T-1.1, T-1.2, T-1.3 (T-1.1 first, then parallel) | Baseline, mixture NB, empirical |
| 5-6 | Experiment 1 | T-1.4, T-1.5 (parallel) | KDE, IDR |
| 7 | Experiment 1 | T-1.6 | Synthesis, decision gate 1 |
| 7 | Checkpoint 1 | Review | Proceed / adopt / stop |
| 7-9 | Experiment 2 | T-2.1, T-2.2 | Probe targets, probe runs |
| 10 | Experiment 2 | T-2.3 | Synthesis, decision gate 2 |
| 10 | Checkpoint 2 | Review | Proceed / adopt / stop |
| 11 | Experiment 3 | T-3.1 | Conditioning variable selection |
| 11-13 | Experiment 3 | T-3.2, T-3.3, T-3.4 (partially parallel) | Four estimators |
| 14 | Experiment 3 | T-3.5 | Synthesis, decision gate 3 |
| 14 | Checkpoint 3 | Review | Final adoption decision |
| 15-16 | Documentation | T-4.1, T-4.2, T-4.3 | Report, queue update, eval proposal |

**Total estimated effort.** 12-16 weeks elapsed, approximately 50-70 person-days of focused work.

**Early termination paths.** The programme can terminate honourably at three points:
- Checkpoint 1 (week 7): if mixture NB or a score-conditioned estimator solves the problem.
- Checkpoint 2 (week 10): if the hidden state carries no additional distributional information.
- Checkpoint 3 (week 14): regardless of outcome - the programme is complete.

### Reviewer critique (opponent seat, 2026-11-03)

The sequencing principle - infrastructure first, cheapest falsification second, each gate evaluated
before the next experiment - is right, and the per-task dependency and deliverable fields are what
make this plan runnable. Six items, all of the same kind: the curriculum drops something §1 through
§8 specified. Each is cheap to restore.

**1. `T-1.2` drops the `theta` lower bound.** §6.2 requires `theta_1, theta_2 >= 0.01`, set from the
conditionally-honest `0.4323` as a prior, as the specific guard against the mechanism §3 calls the
gate on everything. `T-1.2`'s parameterisation does not mention it. Event-conditional fitting
partly neutralises the mechanism, which is presumably the reasoning - state that, or restore the
bound.

**2. `T-1.2` drops one of the two fitting objectives.** §6.2 asks for both CRPS minimisation and
event-conditional NLL; `T-1.2` runs one hybrid. Running both is what disentangles the objective
change from the family change, which §3.4 point 3 identifies as the confound in the *favourable*
direction - the one that would make a positive mixture result unattributable. It is also the
cheapest arm in the phase.

**3. `T-1.3` drops the bin sweep's top end.** §6.2 asks for 250 to 32,000 because `M67` and `#459`
show the fair estimator still improving monotonically at 32,000 and *"the ceiling is unlocated"*.
`T-1.3` sweeps 250 to 1,000. The unlocated ceiling is the single open quantity in the prior record
and the sweep that locates it costs nothing extra.

**4. `T-0.3` item 4 is already measured for one of the two indices.** *"How many distinct score
values exist?"* on the delivered index: **1,321, exactly 0.0 on 99.6567%** (`M89`). Keep the task
for the **un-composed** index, where it is genuinely open and is the programme's viability check,
and cite `M89` for the composed one. As written, Checkpoint 0 risks spending week 1 re-deriving a
ledger row - and Checkpoint 0 is the gate that would otherwise catch this.

**5. `T-2.1` does not build §7.2's targets** (see the §7 note). The pre-registration and the plan
have to name the same four, and the zero indicator is the one to keep.

**6. `T-2.2`'s run count understates the sweep.** `4 x 4 x 3 + 3 x 4 x 3 = 84` is right only if `C2`
is one configuration; `C2` sweeps `k` over four values, so it is 48 runs on its own. §12.6's
sequential-reduction default already handles the cost, but the number in the deliverable should
match.

**And one addition: put the composition arm in Phase 0 as `T-0.6`.** Candidate A' from the §6 note
needs no fitting, no hyperparameter selection and no infrastructure beyond `T-0.1`, and its result
determines the baseline for all of Phase 1. Half a day, and it has to run before `T-1.1` rather than
beside it, because `T-1.1` is the baseline.

Minor: the phase boundaries in the section headers (Phase 2 *"weeks 7-11"*, Phase 3 *"11-15"*) do
not match the §11.6 table (7-10, 11-14), and the per-task efforts sum to 50-74 person-days against
the stated 50-70. The timeline is what a resourcing decision reads, so it is worth one pass.


---

### Panel note - the archive Phase 1 needs may not exist yet (expert method review, 2026-11-03)

Full argument in **§14**, operational seat. This is the panel's answer to *"if we cannot make this
work, why"* - and the answer is not a methods failure.

**1. The analog framing has a data prerequisite the plan does not provision.** §10.4's O-6 note
reframes candidate C as an analog ensemble, which is correct and well grounded. The same literature
prices it: `DelleMonache2013` **C-4559** - *"AnEn benefits more from increased training data length
than logistic regression does. Reducing the training period from 12-15 months to 6-9 months degrades
AnEn performance more than it degrades LR."* §6's archive spec asks for *"at least 12-month fit
windows"*. But `M89`'s own caveat column records: ***"The delivered regime has NO fit window - the
delivery run produced no month before 457 - so the family-free arm borrows the teacher-forced
window's bins by rank; a delivered-regime fit would need the pipeline re-run over 36 in-sample
rolling origins (GPU, not done, stated rather than worked around)."***

So Phase 1 on the delivered cube needs a GPU pipeline re-run that is not in Phase 0, not in the
50-70 person-day estimate, and not in §12. It is the single thing most likely to stop this
programme, it is **fixable**, and it is a resourcing decision rather than a scientific one. It
belongs in Phase 0 as `T-0.0`, with its GPU cost stated, and in §12 as a decision with a default.
The honest fallback - run Experiment 1 on a rank-borrowed window and say so in the report - is also
available and should be written down as the alternative rather than discovered in week 4.

**2. A pre-registrable prediction for `T-1.3`, from the analog literature's own failure mode.**
`DelleMonache2013` **C-4555**: with a finite record, imperfect analogs *"introduce extra spread and
decreased PDF resolution while preserving reliability"*. The measured dominant defect here is
**over-dispersion in 9 of 10 spread bins** (`M90`(2)). So AnEn's known cost compounds the problem
rather than offsetting it, and the expected gain is in **reliability**, not sharpness. That is a
risky prediction `T-1.3` can commit to before it runs, which is the discipline `D-5` already
requires of every script in this record.

**3. A second pre-registrable prediction, for `T-1.6`'s spatial-scale sweep.** `Jeworrek2023`
**C-4570**: analog performance is *"more sensitive to the grid spacing of the NWP model than to its
physics configuration"*, and coarser **27 km** grids *"yielded better AnEn performance across bias,
MAE, and CRPS than finer (9 km and 3 km) grids, contradicting earlier findings by Delle Monache et
al. (2013)"*. `T-1.6` already plans a three-scale analysis for O-4. The literature predicts the
coarse end wins, which - if it holds here - is independent evidence for the decoder-smoothness
ceiling rather than a disappointment.

**4. The strongest positive argument for the analog reframing is in the library and §10.4 did not
make it.** `DelleMonache2013` **C-4557**: AnEn *"captures flow-dependent forecast error growth: its
ensemble spread dependably reflects the forecast error variance, as shown by an excellent
spread-skill relationship"*. The programme's measured problem (`M91`, `W-113`) is that spread adds
only **+0.18** beyond the mean. A method whose headline property is a working spread-skill
relationship is aimed precisely at the defect. That belongs in §2 and §6 as the reason candidate C
is worth running, independent of the 15%.

**5. Two instruments for `T-0.3` that are better than what it currently specifies.**
`Walz2026` **C-3606**: central limit theorems for the sample `CMA_n` / `CID_n` and **DeLong-type
tests for `CMA(X_1, Y) = CMA(X_2, Y)`**, computable in `O(n log n)`. That is the exact instrument for
two live comparisons - the composed index against the un-composed one, and §5.3's
ranking-preservation gate with a p-value instead of a 2% rule of thumb. And `Arnold2024` **C-3578**:
the **MCB-DSC plot** with parallel equal-CRPS contours is the right single figure for `T-1.6`'s
five-candidate (now seven-candidate) comparison, generalising the binary-outcome plots of
Dimitriadis et al. (2024). Its precondition is named too - `Arnold2024` **C-3577**: when few
forecast pairs are comparable under the stochastic order, truncate the CDFs to the
observation-containing interval `[a,b]` with a tolerance `epsilon` on discarded tail mass, which
*"raises the comparable fraction from near zero to about 0.90 or higher"*.

---

## 12. Open decisions

These are decisions that must be made before or during the programme but are not yet resolved. Each has a default and a deadline.

### 12.1. Score binning strategy

**Decision.** How to bin the scalar score for conditional estimation (affects T-1.2 through T-1.5).

**Options.**
- Equal-count bins (each bin has the same number of observations). Default.
- Equal-width bins (each bin spans the same score range). May leave tail bins empty.
- Adaptive bins (tree-based partitioning that maximises distributional heterogeneity across bins).

**Default.** Equal-count bins. This ensures every bin has enough observations for reliable distribution estimation.

**Deadline.** Before T-1.2 begins.

### 12.2. Hurdle vs. full-support estimation

**Decision.** Whether all estimators should use the hurdle decomposition (separate gate probability + conditional distribution) or whether some should model the full distribution including zeros.

**Trade-off.** The hurdle decomposition isolates the distributional problem from the zero-mass problem, which is cleaner. But it inherits the gate's calibration error (O-5). Full-support estimation avoids the gate dependency but must handle the 99.66% zero mass, which dominates any density estimator.

**Default.** Hurdle decomposition for parametric estimators (B, D), full-support for empirical/isotonic estimators (C, E) which naturally include the zero mass.

**Deadline.** Before T-1.2 begins.

### 12.3. Autoregressive cube selection

**Decision.** Which autoregressive cube to use as the programme's data.

**Options.**
- The existing autoregressive cube (from the operational pipeline).
- A freshly generated cube with a specific temporal partition for the programme's cross-validation.

**Trade-off.** Using the existing cube is cheaper and ensures results are operationally relevant. A fresh cube with a programme-specific partition gives cleaner cross-validation but may not reflect operational conditions.

**Default.** The existing autoregressive cube. Generate a programme-specific partition only if the existing cube's temporal folds are too coarse for the bootstrap (T-0.1).

**Deadline.** Before T-0.1 begins.

### 12.4. What counts as "ranking preservation"

**Decision.** The §5.5 criterion requires ranking degradation ≤2%. But ranking of what? Global AUC? Event-cell Spearman? Tail-cell Spearman?

**Trade-off.** Global AUC is the operational metric but is dominated by the easy zero/nonzero distinction. Event-cell Spearman is more relevant to the distributional problem but is not operationally tracked.

**Default.** Both must be within tolerance: global AUC degradation ≤1% AND event-cell Spearman degradation ≤5%. This is stricter on the operational metric and more lenient on the research metric.

**Deadline.** Before T-1.6 (Experiment 1 synthesis).

### 12.5. Whether to pre-register

**Decision.** Whether to pre-register the programme's hypotheses, experiments, and decision gates before seeing results.

**Trade-off.** Pre-registration prevents post-hoc rationalisation but adds administrative overhead. The programme is internal research, not a clinical trial. The decision gates are already explicit.

**Default.** Soft pre-registration: commit the programme document (this file) to the repository before running any experiments. Date-stamp the commit. Any changes to decision gates after results are available must be documented as amendments with rationale.

**Deadline.** Before T-1.1 begins.

### 12.6. Compute budget for Experiment 2

**Decision.** The number of probe configurations (84+) is large. Some may not be worth running.

**Options.**
- Run all 84+ configurations (comprehensive but expensive in researcher time).
- Run a reduced set: P1 and P2 at logistic regression capacity first. If P2 does not improve, skip MLP and controls. If P2 improves, run controls (C1-C3) and capacity sweep.

**Default.** Sequential reduction. Start with the cheapest probes, expand only if the initial results warrant.

**Deadline.** After T-2.1 (probe targets are ready), before T-2.2.

### Reviewer critique (opponent seat, 2026-11-03)

Giving every open decision a default and a deadline is the right pattern and §12.2's hurdle
trade-off is correctly stated. Three decisions belong here that are currently implicit, and one note
on §12.5.

### 12.7. Which conditioning index

**Decision.** Whether the programme conditions on the **composed** index (post-`threshold_gate`,
tau=0.20) or the **un-composed** `gate * mu`.

**Trade-off.** The composed index is what ships, so results on it are operationally direct - but
`M89` measured it and closed it: 1,321 distinct values, exactly 0.0 on 99.6567% of cells, worse than
the no-model floor at every B >= 1,000. The un-composed index has full support by construction and
has never been tested on the delivered cube, but a result on it is a statement about a product that
does not currently exist without a composition change.

**Default.** Un-composed, with candidate A' (§6) measuring what the composition change costs or
gains. This is the decision the whole of Experiment 1 rests on and it is currently unstated.

**Deadline.** Before `T-0.1`, because it determines what the archive stores.

### 12.8. Whether the composition arm runs

**Decision.** Whether candidate A' - `soft_gate` for `threshold_gate` at emit time, nothing
retrained - is in scope.

**Trade-off.** It is the cheapest arm in the programme and `M66`/`M90`(1) indicate it moves both the
dispersion defect and the index degeneracy. Against: it changes the delivered product rather than
post-processing it, so adopting it is a different kind of decision from adopting an estimator, and
`M66`'s blocking finding means the sampler needs verifying first.

**Default.** Yes, in Phase 0 as `T-0.6`, run as a diagnostic rather than a promotion proposal.

**Deadline.** Before `T-1.1`, since it sets Phase 1's baseline.

### 12.9. When the FAO-02 PIT amendment is raised

**Decision.** Phase 0 or Phase 4.

**Trade-off.** §5.5 criterion 3 gates on PIT; §1.5 records that FAO-02 rejects it and `M90`'s caveat
column says a promotion decision on those diagnostics *"needs Simon's amendment FIRST"*. `T-4.3`
currently drafts the amendment last, which means every gate that fires before it is provisional.

**Default.** Raise it in Phase 0 as a one-page ask. It costs nothing to move and it is the
difference between a gate that is legitimate when it fires and one that is legitimated afterwards.

**Deadline.** Before `T-1.6`.

**On §12.5.** Soft pre-registration is the right call for internal research. It binds only if §5
through §8 and §11 agree on labels, targets and gates - and at present they disagree on the probe
labels, the estimator labels, the four probe targets, the mixture-adoption rule and the
solution/dead boundary. The notes on §5, §7, §8 and §11 are that list. Fixing them before the commit
is what makes the date-stamp mean something.


---

### Panel note - three further open decisions (expert method review, 2026-11-03)

Same pattern as §12.1 through §12.9: decision, trade-off, default, deadline. Argument in **§14**.

#### 12.10. The primary scoring rule

**Decision.** Which score the decision gates are evaluated on.

**Trade-off.** Three candidates and each has a disqualifying property on this population.
Event-conditional fair CRPS (§5.2, §5.5) is **improper** - `Lerch2017` **C-80**. SCRPS (§10.4's O-3
proposal) is **undefined on a point mass**, and `M90`(1) has the delivered ensemble as a point mass
on 99.42% of the population. Unweighted aggregate fair CRPS is proper and defined, but §5.2's own
argument is right that it is dominated by the background.

**Default.** A three-part battery, with only one part gating: **unweighted fair CRPS on score
differences with block-bootstrap CIs** (as §5.8 already specifies) is the gate; **twCRPS** at the
§5.2 thresholds carries the tail claim, because it is proper under exactly this emphasis; and
event-conditional CRPS plus the **Dawid-Sebastiani score** (`Czado2009` **C-3564**) are reported
diagnostics. SCRPS enters only if the composition arm restores spread.

**Deadline.** Before `T-0.1`, because it determines what the evaluation module computes.

#### 12.11. Whether the delivered-regime fit window is provisioned

**Decision.** Whether to re-run the pipeline over 36 in-sample rolling origins to give the delivered
cube a genuine fit window, or to run Experiment 1 on a rank-borrowed window.

**Trade-off.** `M89` records the gap and that the re-run is GPU work *"not done, stated rather than
worked around"*. `DelleMonache2013` **C-4559** prices a short window specifically for the analog
family. Against: GPU time, and it is the only item in this programme that needs any.

**Default.** Provision it in Phase 0 as `T-0.0`. If that is refused, Experiment 1 runs on the
borrowed window and every result says so on its face - which is the honest version, not a
workaround.

**Deadline.** Before `T-1.3`.

#### 12.12. The ranking instrument

**Decision.** Whether §5.3 and §12.4's ranking-preservation tolerances are set on Spearman or on
CMA / CID.

**Trade-off.** Spearman is familiar and already in the record. But `Walz2026` **C-3605** establishes
that Spearman's Rho and Kendall's Tau *cannot attain their maximum under perfect monotone dependence
for non-continuous distributions*, and `ged_sb_best` is a highly discrete count. A tolerance of
*"degrades by no more than 5% relative"* is then set against a ceiling that is below 1, unknown, and
varies with the outcome's discreteness. CMA and CID attain it, reduce to AUC for dichotomous
outcomes and to Spearman for continuous ones, and come with a comparison test (**C-3606**).

**Default.** CMA as the gate with the DeLong-type test; Spearman reported alongside for continuity
with the existing record.

**Deadline.** Before `T-0.3`.

---

## 13. Literature

This section lists the papers that ground the programme's design, organised by role in the argument. Papers marked with `*` are in the 9_library.

### 13.1 Distribution estimation

- Silverman (1986). *Density Estimation for Statistics and Data Analysis*. The standard KDE reference. Bandwidth selection, boundary effects, multivariate extension. `*`
- Sheather & Jones (1991). A reliable data-based bandwidth selection method. The Sheather-Jones plug-in selector for univariate KDE. `*`
- Chacón & Duong (2018). *Multivariate Kernel Smoothing and Its Applications*. Modern multivariate KDE with full-matrix bandwidth selectors. `*`
- Abramson (1982). On bandwidth variation in kernel estimates - a square root law. Adaptive KDE with local bandwidth proportional to `f(x)^{-1/2}`. `*`
- Escobar & West (1995). Bayesian density estimation and inference using mixtures. The Dirichlet process mixture model for nonparametric density estimation. `*`
- Bishop (1994). Mixture density networks. Neural network outputs parameterising a Gaussian mixture. `*`

### 13.2 Conditional distribution estimation

- Hyndman, Bashtannyk & Grunwald (1996). Estimating and visualizing conditional densities. Conditional KDE with local bandwidth adaptation. `*`
- Hall, Wolff & Yao (1999). Methods for estimating a conditional distribution function. Theoretical foundations for conditional CDF estimation. `*`
- Hothorn, Kneib & Bühlmann (2014). Conditional transformation models for distributional regression. Distribution regression via transformation functions. `*`
- Klein, Kneib, Lang, & Sohn (2024). Distributional regression. Comprehensive review of the distributional regression framework. `*`

### 13.3 Quantile regression and isotonic methods

- Meinshausen (2006). Quantile regression forests. Random-forest-based conditional quantile estimation. `*`
- Romano, Patterson & Candès (2019). Conformalized quantile regression. Conformal prediction wrappers for quantile regression. `*`
- Henzi, Ziegel & Gneiting (2021). Isotonic distributional regression. IDR: nonparametric distributional regression with stochastic ordering constraint. (Lipiecki et al. 2024 `*` and Biegert et al. 2026 `*` are applications.)

### 13.4 Scoring rules and evaluation

- Gneiting & Raftery (2007). Strictly proper scoring rules, prediction, and estimation. The foundational reference for proper scoring rules, including CRPS. `*`
- Zamo & Naveau (2018). Estimation of the continuous ranked probability score with limited information and applications to ensemble weather forecasts. The fair CRPS estimator.
- Gneiting, Balabdaoui & Raftery (2007). Probabilistic forecasts, calibration and sharpness. PIT-based calibration assessment.
- Bröcker (2012). Evaluating raw ensembles with the continuous ranked probability score. The plug-in bias problem.
- Bröcker & Smith (2008). From ensemble to predictive distribution: explicit link between ensemble interpretation and scoring rule. `*`

### 13.5 Forecast combination and post-processing

- Gneiting & Ranjan (2013). Combining predictive distributions. Optimal combination of probabilistic forecasts. `*`
- Delle Monache et al. (2013). Probabilistic weather prediction with an analog ensemble. The analog ensemble method for conditional distribution estimation from similar historical cases. `*`

### 13.6 Zero-inflated and count distributions

- Lambert (1992). Zero-inflated Poisson regression with an application to defects in manufacturing. The original ZIP model.
- Hilbe (2011). *Negative Binomial Regression*. Comprehensive treatment of NB models, including NB1/NB2 parameterisations and zero inflation.

### 13.7 Representation learning and probing

- Alain & Bengio (2017). Understanding intermediate layers using linear classifier probes. The linear probing methodology for assessing what information neural network representations contain.
- Belinkov (2022). Probing classifiers: promises, shortcomings, and advances. Critical assessment of probing methodology, including the information-theoretic concerns about probe capacity.

### 13.8 Domain-specific

- Hegre et al. (2022). ViEWS: a political violence early-warning system. The VIEWS system architecture. `*`
- The views-hydranet measurement record: ADR-073 (zero-capture mechanism), M67 (family-free CDF), M69-M73 (evaluation blindness), M90 (plug-in bias), M92-M93 (decoder smoothness), W-107 (teacher-forced vs. autoregressive), C-399 (AUC degradation), `#442` (draw count sweep), `#443` (family-free finding), `#459` (fair CRPS), `#491` (decoder architecture fix).

### Reviewer critique (opponent seat, 2026-11-03)

Organising the literature by role in the argument rather than by topic is the right choice and makes
§13 usable. Three corrections, two of which save implementation time.

**Three library markers are missing**, and one of them matters for Phase 0. Checked against
`9_library/papers/_claims/`:

- **Zamo & Naveau (2018)** is in the library (`Zamo2018_EstimationCRPSLimitedInformation`, with
  claims and extracted text). `T-0.1` implements the fair CRPS estimator from a formula relayed in
  this document; the paper itself is on disk and should be read rather than the formula transcribed.
- **Henzi, Ziegel & Gneiting (2021)** is in the library
  (`Henzi2021_IsotonicDistributionalRegression`), though §13.3 marks only its applications. Relevant
  to candidate E's build-or-defer decision in §6.2.
- **Lambert (1992)** is in the library (`Lambert1992_ZeroInflatedPoisson`).

Not in the library, correctly unmarked: Hilbe (2011), Alain & Bengio (2017), Belinkov (2022),
Gneiting/Balabdaoui/Raftery (2007), Broecker (2012).

**Four entries are missing from the measurement record in §13.8, and three of them are load-bearing
for §2, §6 and §9:**

- **`M89`** - the delivered autoregressive forecast scored for the first time: the composed index's
  1,321 distinct values and 99.6567% exact zeros, the family-free arm's failure to transfer, and the
  three-way decomposition of the `0.99777 -> 0.72340` AUC pair.
- **`M66`** - the emitted zero mass as a composition defect, free at emit time, with the
  `TruncatedNBFamily.sample()` blocking finding any composition arm must clear.
- **`M70`** - the *"fourteen of the battery's twenty columns"* figure that §1.5 attributes to
  `M69-M73` collectively.
- **`M77`/`M78`/`C-393`** - the 8 x 3 favourable-selection caveat on the 15% headline.

**Two reference-convention fixes** (also noted at §0): ADRs live at `docs/ADRs/`, not
`docs/decisions/`, and `ADR-073` is in `proposed/` rather than `active/`.


---

*Document ends. Last updated: 2026-11-03.*


---

## 14. Expert method review - the panel (added 2026-11-03)

This section adds a design-and-methodology review of the programme and of §10.4's replies. It
removes nothing. It is run under two constraints set by Simon: the question is **how to make this
work**, and where it cannot work, **why, and what could work instead**. Personas are seated from a
fixed casting pool and constructed from documented positions; where a figure's real view on a
specific point is not documented, the note says so rather than inventing one. **The chair is Simon.
No persona of the chair is seated** - a self-seat only reflects known priors.

### 14.1 Decisions under review, and the data-generating process

The programme's §12 lists six open decisions. The panel reads nine more as live, and they are the
ones that determine whether the programme produces a usable answer:

| # | decision | where it currently sits |
|---|---|---|
| D1 | the conditioning variable: composed (post-`threshold_gate`) or un-composed `gate * mu` | unstated; §6.1 is ambiguous |
| D2 | the primary scoring rule, given propriety | §5.1, §5.5 - and both are improper as written |
| D3 | which calibration notion the PIT gate tests | §5.4 - unnamed |
| D4 | which distribution families can express a location-preserving spread change | §6.2 - three of five cannot |
| D5 | whether anything in the candidate set can exceed the historical maximum | §6.2 - nothing can |
| D6 | whether the delivered regime has a fit window | §6 assumes one; `M89` says there is none |
| D7 | the ranking instrument and its attainable ceiling | §5.3, §12.4 - Spearman |
| D8 | whether the hidden state is read per cell or per neighbourhood | §7.4 stores a tensor, §7.3 probes a vector |
| D9 | whether conflict cell-months have historical analogs at all | §10.4's O-6 reframing assumes yes |

**The DGP the design must respect.** A monthly count field on 64,818 land cells, `ged_sb_best`,
99.66% zero in any month, 2,905 positives in the 13-month scoring window, a heavy right tail whose
shape the record says is the gap (`xi > 0`, Epic #230), spatial dependence the decoder destroys
(`M92`/`M93`), temporal dependence with escalation and de-escalation regimes, an irreducible
geocoding floor that has never been estimated (`RP-10`), and a delivered product whose emission
replaces a drawn distribution with a point mass on 99.42% of cells (`M90`(1)). Non-stationary,
non-exchangeable across space, and measured with error. Every panel note below is judged against
that, not against a weather benchmark.

### 14.2 Panel, and why these seats

Seven seats. Each either authored a method the programme proposes, or opposes one on a documented
principle. Every live fault line has at least two sides.

| seat | why seated | fault line it generates |
|---|---|---|
| **Tilmann Gneiting** | the entire §10.4 reply is built on his school's instruments, and the propriety problems are his theorems | assume vs learn the distribution; post-process vs retrain |
| **Frank Harrell** | the design dichotomises at six separate points; he is the documented opponent of that | fancy vs simple; dichotomisation |
| **Anthony Davison (EVT seat)** | the record says the gap is tail *shape*; nobody else owns the tail | tail / escalation |
| **Xingjian Shi** | authored ConvLSTM, which is this backbone's family; owns the representation question | recurrence / architecture; scale vs structure |
| **Rich Sutton** | the standing contrarian against 16 weeks of hand-designed post-processing on a frozen model | scale vs structure; post-process vs retrain |
| **Operational-forecasting archetype** (de-personalised VIEWS-ethos house view) | owns the archive, the GPU cost, FAO-02 and the queue | deliverability |
| **Conflict-science substance seat** (Buhaug / Gleditsch composite) | the analog reframing makes an untested claim about the DGP | does the design respect the process |

**Not seated, and why.** Bishop (MDN) - his point, that candidate B is a two-component mixture
density network with the trunk's gradient cut, is carried by Shi. Kendall / Gal / Lakshminarayanan -
the uncertainty-religion line is not live here, because the programme does not propose to change the
posterior. Kohl / Kingma - the integrated-head route is explicitly deferred to future work by §12,
so seating them would critique a decision the document has not taken.

### 14.3 Library grounding

**Held, and cited below by claim ID.** Every one of the fifteen works §10.4 cites is in `9_library`
with extracted claims; the per-citation verification is in the §10 panel note. The panel adds five
further holdings that bear directly on the decisions above and are **not** currently used by the
programme or by §10.4:

| holding | claim | what it decides |
|---|---|---|
| `Gneiting2011_ComparingDensityForecasts` (Gneiting & Ranjan 2011) | **C-53** improper weighting theorem; **C-54** twCRPS/qwCRPS are proper; **C-55** threshold and quantile decompositions of CRPS; **C-56** the improper route selects a misspecified forecast | **D2** |
| `Gneiting2023_ConditionalCalibration` (Gneiting & Resin 2023) | **C-3570** auto-calibration hierarchy and Example 2.4; **C-3571** CORP reliability diagrams; **C-3572** `R* = DSC/UNC`; **C-3573** CEP / threshold / quantile calibration are distinct under minimal conditions | **D3** |
| `Hothorn2014_ConditionalTransformationModels` | **C-4430** higher moments vary with the covariate; **C-4431** the CRPS is the loss; **C-4433** monotonicity caveat; **C-4434** beats kernel estimators in simulation | **D4** |
| `Dimitriadis2024_ProbabilisticClassifiersTriptych` | the triptych (reliability / ROC / Murphy diagram) that `Arnold2024` **C-3578** generalises | **D7** |
| `Klein2024_DistributionalRegression` | the review §13.2 already lists but the candidate set does not draw on | **D4** |

**Gaps to fetch - short, and none of them blocking.**

1. **Yang (2024), double PIT calibration for count data.** Cited by `Resin2026` **C-3601** as the
   origin of the one calibration notion that is *logically independent* of all others when the
   outcome has a discrete component. We hold the extension, not the original. Relevant to D3.
2. **Schuster & Gregory (1981)**, the inconsistency of likelihood-based kernel bandwidth estimators
   for heavy tails. The result is fully stated in `MacDonald2011` **C-4464**, so fetching is optional
   - it would only be needed to cite the proof rather than the finding.
3. **A conflict-domain analog-forecasting precedent.** The panel could not find one in the library.
   `Yang2025_AnalogEnsembleSierraNevada` and `Jeworrek2023` are both meteorological. The question in
   D9 - whether a social-conflict process has usable analogs - has no in-library answer, which is
   itself the finding. `Radford2022` is the nearest in-domain forecasting precedent and is worth
   checking for an analog or nearest-neighbour baseline.

### 14.4 Independent critiques

#### Gneiting seat - "you cannot gate on a score whose optimum is not the truth"

**Endorses.** §5.1's rejection of the plug-in estimator, which is correct and well evidenced
(`M90`(5): the plug-in moves 13.9% of the score against the fair estimator's 0.65% over the same
sweep). §10.4's reach for `Arnold2024` is the right instrument. §5.8's block bootstrap on
differences is the right inferential form.

**Challenges.** Three, all theorems rather than preferences. (i) §5.2 and §5.5 criterion 1 restrict
the score to `truth > 0`, which is a nonconstant weight, which makes the score improper -
`Gneiting2011` **C-53**, and **C-56** shows the consequence is selecting a misspecified forecast,
not merely losing power. (ii) §5.5's criteria are all skill scores, improper even on a proper rule
(`Bolin2023` **C-97**), while §5.8 already specifies the proper form. (iii) §5.4 gates on PIT
uniformity, which is probabilistic calibration - and `Gneiting2023` **C-3570**'s Example 2.4
constructs forecasts that are *both marginally and probabilistically calibrated and still not
auto-calibrated*. The gate can pass on exactly the failure being hunted.

**What to build.** Implement the **threshold decomposition** of the mean CRPS (**C-55**) in `T-0.1`;
it gives §5.2's region-specific comparison properly and as a curve. Gate on **twCRPS** at the §5.2
thresholds (**C-54**). Produce **CORP reliability diagrams** (**C-3571**) and the **MCB-DSC plot**
(`Arnold2024` **C-3578**) as `T-1.6`'s two figures. And the cheapest Checkpoint 0 number in the whole
programme: **`DSC_ISO` restricted to event cells**, for both candidate conditioning variables.
`Arnold2024` **C-3574** gives `DSC_ISO = 0` iff the IDR-calibrated forecasts are constant - so one
number says whether there is any discrimination left to recalibrate, before a single estimator is
built. On the composed index it should reproduce `M89` in one line; on the un-composed index it is a
genuinely new measurement. Precondition stated in **C-3577** (truncate to `[a,b]` with tolerance
`epsilon` when stochastic comparability is low).

#### Harrell seat - "you have dichotomised six times and counted none of them"

**Challenges.** The design converts a continuous quantity to a binary one at six places:
`threshold_gate` at `tau = 0.20` (emit), `truth > 0` (§5.2, §5.5), `q0.90` and `q0.95` (§5.2, §5.5
criterion 5, §7.2), `|u - 0.5| > 0.3` (`T-2.1` item 4), residual-exceeds-twice-variance at
threshold 2 (`T-2.1` item 1), and `y_t > 0 AND y_{t-1} = 0` (`T-2.1` item 3). Each discards
information and each introduces a discontinuity. The one that matters most is `T-2.1` item 4:
binarising the conditional PIT at `|u - 0.5| > 0.3` maps **both** failure modes to the same label,
because `Czado2009` **C-3561** reads a U-shaped PIT as underdispersion and an inverse-U as
overdispersion - and the programme's measured defect is *over*-dispersion in 9 of 10 bins and
*under*-dispersion in the top one (`M90`(2)). A binary target cannot see the sign of the thing the
programme exists to fix.

**Second challenge.** §5.5 is a conjunction of five thresholded criteria evaluated on four folds
with no multiplicity accounting, and §5.7's "dead" rule contradicts it (see the §5 reviewer note).

**What to build.** Keep the outcome continuous wherever a probe can take it: regress the **continuous**
PIT on `(S, H, C)` as a distributional-regression probe, or test uniformity of the `H`-conditional
PIT directly. For the gate, `Kuleshov2018` **C-1451** is the Harrell-shaped candidate - isotonic
recalibration through a feature function such as `(y - mu(x)) / sigma(x)`, monotone, continuous, no
family, no threshold, and cheaper than candidate B. Its predicted cost is on record: **C-1454**,
recalibration *"may become less sharp if the original forecaster was underestimating its
uncertainty"*, and `M90`(6) measures the firing rate at **0.42x** the observed rate.

**Faithfulness note.** Harrell's documented position is against dichotomising continuous variables
*for analysis*. `tau = 0.20` is a product requirement, not an analysis choice, so his objection
there is specifically to **using the thresholded product as the analysis population** - which is
also precisely what `M89` measured breaking the index. The two arguments converge from different
directions.

#### Davison seat (EVT) - "five of five candidates are capped at the historical maximum"

**The gap.** The programme's own record says the gap is **tail shape**, not tail location (Epic
#230, §2.1). Nothing in §6.2 estimates a shape. Candidate A has one global `theta`; B has two global
`theta`; C is an empirical distribution and therefore cannot exceed the largest historical
observation by construction; D's tail is whatever the kernel puts there, and its bandwidth selector
is inconsistent in exactly this regime (`MacDonald2011` **C-4464**); E cannot extrapolate past the
training maximum at all - `Henzi2021` **C-4394**, in the authors' own emphasis. For a programme whose
stated defect is under-dispersion in the extreme bin, that is the central design gap and it is not
named anywhere in §6.

**What to build.** `MacDonald2011` **C-4462**: an extreme-value mixture - nonparametric kernel for
the bulk below a threshold `u`, generalized Pareto above it, with `u` estimated **jointly** with the
bandwidth and the tail parameters, *"avoiding subjective threshold pre-selection and fully
accounting for threshold uncertainty"*. It is the one candidate that can exceed the historical
maximum, the one that estimates a shape, and it fixes candidate D's bandwidth problem as a
by-product - **C-4464**'s two-tailed mixture takes the Cauchy bandwidth from 12.41 to **0.41** while
recovering both tail shapes near their true value. Add as **candidate G**. The cheaper alternative,
if G is refused: `Jeworrek2023` **C-4568**'s error-based analog members, which *"enable ensemble
members to take values outside the range of the finite archived observational dataset, making them
theoretically capable of forecasting record events that traditional analog methods cannot"* - with a
measured effect, **C-4569**: probability of detection improved by roughly **30%** for 90th-percentile
and **60%** for 95th-percentile precipitation events.

**Challenge to §10.4's O-7 proposal.** It suggests probing the bulk-tail transition threshold
instead of exceedance. `MacDonald2011` **C-4463** is the reason not to: threshold uncertainty has a
*local* effect, and coverage at the threshold-level quantile (90th percentile) is only **50-60%**,
rising to near-nominal **95%** only at the 99th and 99.9th. §10.4 targets the threshold and §7.5/O-7
fall back to the 90th percentile under sample-size pressure - both land where the instrument is
weakest. Probe **`xi`** instead, with its interval, which is also the quantity Epic #230 named as
the gap. At ~2,905 events and ~143 above `q0.95`, `xi` is a diagnostic with a wide interval, not a
gate - and saying so in advance is the honest form.

#### Shi seat - "you are storing a spatial tensor and probing it as a per-cell vector"

**Endorses.** §7.4's forward-hook extraction and its alignment check, which is the right worry and
the right place for it.

**Challenges.** (i) §7.4 stores the state as `(T, H_lat, W_lat, d_h)` - a spatial tensor, because
this is a convolutional recurrent backbone - and §7.3 then hands the probes a single cell's `d_h`
vector. That discards the latent neighbourhood, which is the architecture's entire inductive bias,
and it is the most likely place for the information H2 is looking for to live: two cells with the
same score and different conditional distributions most plausibly differ in their *surroundings*,
which is exactly what §1.6 says the decoder destroys. (ii) The programme freezes the backbone for
12-16 weeks while the measured cause of the spatial defect is two lines -
`ConvTranspose2d(kernel=2, stride=2)` in both upsampling stages, kernel equal to stride, no window
overlap, no spatial mixing (`M92`, `M93`, `#491`). (iii) The integrated distributional head that
§12 defers to future work is what this architecture family would normally do, and candidate B is
that head with the trunk's gradient cut.

**What to build.** A fifth probe at near-zero cost: give `P2` a **local patch** of the latent state
(3x3 or 5x5 around the cell) rather than the cell's vector alone. It is the one probe that tests
the architecture's own prior, it fits inside §7.3's existing controls, and §7.4's corrected storage
arithmetic (2.4 GB, not 24 GB - see the §7 reviewer note) means the full tensor is affordable.
`Goodfellow2016_Ch05` **C-4251** is the general argument and it cuts this way: nonparametric methods
on the smoothness prior *"can distinguish only as many regions as they have training examples"*,
while compositional structure gets `O(2^k)` regions from `O(k)` examples - an argument for using the
representation over post-processing a scalar.

#### Sutton seat - "the hand-designed stage gets superseded"

**The dissent.** 50-70 person-days of hand-built post-processing, on a model whose training
objective is known to be wrong (`ADR-073`'s zero capture) and whose decoder is known to be wrong
(`M92`/`M93`), to produce a layer that must then be maintained against a backbone that will be
retrained. The programme's own §9 lists the objective (`RP-5`) and the decoder (`#491`) as the
fallback *after* fifteen weeks, which inverts the order the bitter lesson would argue for.

**The concrete form of the objection.** §12 has no decision for what happens to an adopted
estimator when the backbone is retrained, and every candidate in §6.2 is fitted to **this
checkpoint's** score distribution. A score-conditioned lookup is a function of a specific `mu` field.
Ask the question now, because the answer changes which candidates are worth having: a monotone
recalibration (`Kuleshov2018` **C-1450**) transfers across checkpoints by refitting one isotonic
curve; an 8,000-bin empirical lookup does not.

**Where this seat concedes.** Candidate A' (the composition arm) is not a hand-designed prior - it
is the removal of one, and it is the cheapest item in the programme. This seat would run A' first and
`RP-5` second, and treat Experiments 2 and 3 as contingent on both.

**Faithfulness note.** Sutton has no documented position on this system. The stance above is the
bitter lesson applied to this case, flagged as an application rather than a quotation.

#### Operational seat - "the archive Phase 1 needs does not exist"

**Three deliverability findings, in order of how much they cost.** (i) `M89`: *"The delivered regime
has NO fit window - the delivery run produced no month before 457 ... a delivered-regime fit would
need the pipeline re-run over 36 in-sample rolling origins (GPU, not done, stated rather than worked
around)."* §6 specifies at least 12-month fit windows, and `DelleMonache2013` **C-4559** prices a
short window specifically for the analog family. This is the single most likely cause of the
programme stalling, it is **not** a methods failure, and it is the only item here needing GPU.
(ii) §5.5 gates on PIT, which §1.5 records FAO-02 as rejecting, and `M90`'s caveat column states
that using those diagnostics *"for a promotion decision needs Simon's amendment FIRST"* - while
`T-4.3` drafts the amendment in week 15. Sixteen weeks of work whose adoption path requires a
governance change nobody has requested. (iii) `RESEARCH_PROGRAMME.md`: *"The next pick is Simon's
between `#491`, `RP-10`, `RP-7`, and `RP-5`."* This programme is a 16-week commitment that is not on
that list, and the register's stated discipline is instruments before experiments.

**One scoping distinction §12.8 should make.** The composition change is a **views-models config
entry**, and `pod_run_model.sh:172` is never-modify. So A' as a *diagnostic* on an existing cube is
cheap and inside this seat's gift; A' as a *product change* is a different approval. Separate them in
§12.8 or the cheap version will be blocked by the expensive one's review.

**What to run first.** Three measurements, none needing the archive, each able to redirect the
programme, all inside one week: `DSC_ISO` on event cells for both indices (Gneiting seat); the
composition arm as a diagnostic, with the sampler verified per `M66`'s blocking finding; and the
ranking decomposition on **CMA** with the DeLong-type test (`Walz2026` **C-3605**, **C-3606**) rather
than Spearman.

#### Conflict-science substance seat - "does this process have analogs at all?"

**The untested premise.** §10.4's O-6 reframing of candidate C as an analog ensemble is methodologically
clean and it imports an assumption from meteorology without testing it. `DelleMonache2013` **C-4555**
grounds AnEn in the model *attractor*: *"With an infinite training record, perfect-match analogs
would yield members drawn from the true PDF. With a finite record, imperfect analogs map from nearby
points on the model attractor."* A weather attractor is revisited. Armed conflict has actor entry
and exit, mobilisation, external intervention and regime change; the process is not stationary over
a 36-month window, and the analog of an escalating cell in 2023 may not exist in 2019. If it does
not, candidate C is not under-tuned - it is inapplicable, and that is a different finding with a
different remedy.

**It is cheap to test, and the test is missing from Phase 0.** For each event cell-month, compute the
distance to its `k` nearest analogs in the archive and regress that distance on temporal separation
from the target month. If analog quality degrades with time separation, the method has a horizon,
and that horizon is the programme's real scope limit rather than a hyperparameter. Half a day, no
GPU, and it belongs beside `T-0.3` as the second viability check.

**Second point - the target the curriculum added without specifying is the best one in the set.**
`T-2.1` item 3, **event onset** (`y_t > 0` and `y_{t-1} = 0`), appears with no §7.2 specification and
the §7 reviewer note flags that. From this seat it is the most substantively motivated target in the
list: onset versus continuation is a first-order distinction in the conflict literature, the two have
different covariates, and the duration feature `T-0.5` already computes is onset's natural partner.
Keep it, specify it in §7.2, and say why.

**Third - score-conditioning assumes spatial exchangeability.** The same score in the Sahel and in
eastern Ukraine is not the same conditional distribution; actor structure, reporting intensity and
escalation dynamics all differ. §5.8 mentions geographic holdouts *"where possible"*. This seat would
make stratified reporting mandatory rather than aspirational, because pooling across regions is the
assumption that makes a single score-conditioned lookup coherent at all, and it is testable with the
bins the programme is already building.

### 14.5 Key disagreements - the product of the review

**1. Post-process the frozen model, or fix the model.** *Gneiting / Harrell / Operational:* the
ruler has to be fixed regardless, post-processing is auditable, reversible and cheap, and a
calibration layer is a legitimate deliverable. *Sutton / Shi:* the objective and the decoder both
have measured mechanisms, and a layer fitted to this checkpoint's score field is thrown away at the
next retrain. **Both are right about different time horizons.** The one item every seat endorses is
candidate A' - the composition arm - because it is neither post-processing nor retraining: it removes
an error surface at emit time for free. That is where the programme should start, and it is the
panel's single strongest recommendation.

**2. Can a nonparametric conditional distribution reach the tail?** *Davison:* no - every candidate
is bounded at the training maximum, the record says the gap is shape, so you need a GPD tail or
error-based members. *Gneiting:* a fitted `xi` on ~143 observations above `q0.95` is not estimable
with a usable interval, and the tail claim should be made with a proper threshold-weighted score
rather than by extrapolating a shape you cannot validate. **Resolution the panel can agree on:**
estimate `xi` as a reported diagnostic with its interval, gate on twCRPS. Davison's candidate G
earns a place; it does not earn the gate.

**3. Dichotomise or not.** *Harrell:* six thresholds, all costing information, one of them
(`|u-0.5| > 0.3`) destroying the sign of the defect. *Operational / Gneiting:* the product has a
`tau`, and event-conditional numbers are what a policy audience reads. **Resolution:** thresholds in
the report, never in the gate. The threshold decomposition of CRPS (**C-55**) gives the operational
seat its regional read without the propriety cost.

**4. Is the hidden state the right conditioner?** *Shi:* yes, and you are flattening it wrong -
probe a patch. *Gneiting / Harrell:* a 256-dimensional conditioner at 2,905 events is a variance
problem that no amount of control design fixes. *Conflict seat:* the right conditioner is neither the
score nor the state but onset, duration and actor structure. **All three have merit and all three are
cheap to test** - the patch probe is one more row in `T-2.2`, the low-rank bottleneck already tests
the variance worry, and `T-0.5` already computes two of the conflict seat's three conditioners.

**5. Do analogs exist in this DGP?** *Conflict seat:* untested, possibly not, and the test is half a
day. *Operational:* the family-free lookup demonstrably read *something* at T=0 (`M67`, `M78` on
three targets), so the index carries information. **Resolution:** the T=0 result is evidence the
index carries information; it is not evidence that *analogy* is the mechanism. The analog-quality
measurement separates the two and should run in Phase 0.

### 14.6 Synthesis - what to build, in what order

**Answering (a), how to make this work.** The programme is salvageable and most of the repair is
substitution rather than addition.

**Week 1 - four measurements, no archive required, each able to redirect the programme.**
1. `DSC_ISO` and `MCB_ISO` on event cells, for the composed and the un-composed index
   (`Arnold2024` **C-3574**, **C-3577**). One number for "is there discrimination to recalibrate".
2. Candidate **A'**, the composition arm, as a diagnostic - `soft_gate` for `threshold_gate`,
   nothing retrained, sampler verified first per `M66`'s blocking finding. It sets Phase 1's baseline
   and it is the precondition for SCRPS being defined at all.
3. The ranking decomposition on **CMA** with the DeLong-type test (`Walz2026` **C-3605**, **C-3606**),
   not Spearman.
4. **Analog quality against temporal separation** (conflict seat) - the one test of the premise
   §10.4's reframing introduces.

**Then settle D1-D3 and D6-D7** using §12.7 through §12.12 (the five decisions the reviewer and
panel notes add). D1 and D6 determine what the archive stores and whether it can be built at all.

**Then Experiment 1, with a substituted candidate set.** Keep A, B, C, D, E. Add **A'**
(composition), **D'** (`Hyndman1996` **C-4404**'s mean-matched estimator), **F** (TRAM -
`Hothorn2014` **C-4430**, CRPS-fitted and able to move higher moments with the covariate), **G**
(`MacDonald2011` **C-4462**'s extreme-value mixture), and the family-free recalibration of
`Kuleshov2018` **C-1451**. Re-label **E** as a calibrated benchmark that is excluded from the adopt
set, on the structural ground in **C-4394**. **If only three new arms can be afforded: A', F, and
C-as-analog-ensemble** - one removes an error surface, one can express the defect, one tests whether
the ranking alone suffices.

**Evaluation, substituted throughout.** Threshold decomposition of CRPS and twCRPS (**C-54**,
**C-55**) in place of population restriction; score differences with bootstrap CIs in place of
relative percentages; CORP reliability diagrams and the MCB-DSC plot (**C-3571**, **C-3578**) as the
two figures; Dawid-Sebastiani (**C-3564**) and double PIT (**C-3601**) as the cheap dispersion and
discrete-outcome diagnostics; and the FAO-02 amendment raised in Phase 0 rather than week 15.

**Answering (b), where it cannot work and what could instead.** Two honest limits, and neither is a
methods failure.

- **The delivered-regime fit window does not exist** and building it is GPU work nobody has costed
  (`M89`). Until it exists, Experiment 1 on the delivered cube can only borrow the teacher-forced
  window's bins by rank - which is what `M89` did, with a measured negative. **What could work
  instead, today:** candidate A', which needs no fit window; the `DSC_ISO` measurement, which needs
  none; and `T-0.3`, which needs none. Three of the four week-1 items are unblocked.
- **No candidate in §6.2 can exceed the historical maximum, and the record says the gap is tail
  shape.** A post-processing layer built only from historical outcomes cannot produce a record event
  by construction. **What could work instead:** `Jeworrek2023`'s error-based members (**C-4568**,
  cheap) or `MacDonald2011`'s GPD tail (**C-4462**, principled). Either changes the answer to "can
  this programme fix the tail" from *no, structurally* to *maybe, and here is the arm*.

**The strongest dissent to keep live.** Sutton and Shi: the objective (`RP-5`) and the decoder
(`#491`) have measured mechanisms, this programme does not touch either, and an estimator fitted to
this checkpoint's score field does not survive the next retrain. §12 should carry a decision for
what happens to an adopted estimator when the backbone changes - and the answer selects candidates,
because a monotone recalibration transfers by refitting one curve and an 8,000-bin lookup does not.

### 14.7 Methodological risks (register-compatible - for `register-risk`, not appended here)

| ID | tier | title | trigger | location | narrative |
|---|---|---|---|---|---|
| **M-01** | 1 | §5.5's primary criterion is an improper score | evaluating decision gate 1 on event-cell CRPS | §5.2, §5.5 crit. 1, `T-1.1` | Restriction to `truth > 0` is a nonconstant weight; `Gneiting2011` **C-53** makes the score improper and **C-56** shows the improper route can select a misspecified forecast. Fix: twCRPS (**C-54**) + threshold decomposition (**C-55**). |
| **M-02** | 1 | SCRPS is undefined on 99.42% of the population | adopting SCRPS as primary per §10.4's O-3 note | §10.4 *On O-3*, `T-0.1` | `Bolin2023` **C-94**'s denominator `E_{P,P}|X-Y|` is 0 for a point mass; `M90`(1) has the delivered ensemble as a point mass on 837,759 of 842,634 cells. Available only after the composition arm restores spread. |
| **M-03** | 2 | Candidate E cannot represent the defect the programme exists to fix | §6.4's "non-parametric estimator wins" branch selecting E | §6.2 E, `T-1.5`, §8.4 | `Henzi2021` **C-4394**: IDR uses only the stochastic order and is *"unable to distinguish situations where distributions agree in location but differ in spread or shape"*; §1.1's defect is exactly that. Keep E as benchmark, exclude from adopt set. |
| **M-04** | 2 | §5.5's criteria are skill scores and therefore improper | any gate evaluated on a relative percentage | §5.5 (contradicts §5.8) | `Bolin2023` **C-97**. §5.8 already specifies differences with bootstrap CIs; the two sections disagree and §5.8 is right. |
| **M-05** | 2 | The delivered regime has no fit window; Phase 1 has an uncosted GPU prerequisite | starting `T-1.3` on the delivered cube | §6 archive spec, §11.1, §11.6 | `M89`: no month before 457 in the delivery run; a delivered-regime fit needs 36 in-sample rolling origins re-run. `DelleMonache2013` **C-4559** prices short windows for the analog family. |
| **M-06** | 2 | Ranking tolerances are set on an instrument that cannot attain its maximum | evaluating §5.5 crit. 2 or §12.4 | §5.3, §12.4, `T-0.3` | `Walz2026` **C-3605**: Spearman and Kendall cannot attain their maximum under perfect monotone dependence for non-continuous outcomes. A "5% degradation" tolerance has an unknown, discreteness-dependent ceiling. CMA/CID attain it. |
| **M-07** | 2 | No candidate can exceed the historical maximum, and the record says the gap is tail shape | §6.4 concluding that no estimator fixes the tail | §6.2 A-E, §2.1 | Epic #230: *"the gap is tail shape (`xi > 0`), not tail location."* A, B share global `theta`; C and E are bounded at the training max (`Henzi2021` **C-4394**); D's tail is the kernel's. Remedy: **C-4568** or **C-4462**. |
| **M-08** | 3 | Candidate D's bandwidth selector is unsupported by its own source and inconsistent for heavy tails | `T-1.4` | §6.2 D, `T-1.4` | `Hyndman1996` **C-4406** defers selection entirely; `MacDonald2011` **C-4464** shows likelihood-based selectors are inconsistent for heavy tails (12.41 vs 0.41 on Cauchy). Fix: leave-one-out CRPS, or the two-tailed mixture. |
| **M-09** | 3 | PIT diagnostics at default binning mask the IDR artifact | §5.4 or `T-2.1` item 4 with 10-20 bins | §5.4, `T-0.1`, `T-2.1` | `Biegert2026` **C-4491**: *"Standard PIT histogram binning (10-20 bins) masks the pattern; fine binning (100 bins) ... reveals it."* **C-4492** adds that smoothing introduces its own deviations and that calibrated continuous predictions from IDR *"remains an open problem"*. |
| **M-10** | 3 | Binarising the PIT probe target destroys the sign of the defect | `T-2.1` item 4 | `T-2.1`, §7.2 | `Czado2009` **C-3561**: U-shape is underdispersion, inverse-U is overdispersion. `|u - 0.5| > 0.3` maps both to one label, and §1.1's defect is non-monotone in exactly that dimension. |
| **M-11** | 3 | The analog reframing imports an untested stationarity assumption | `T-1.3` run as an analog ensemble | §10.4 *On O-6*, `T-1.3` | `DelleMonache2013` **C-4555** grounds AnEn in a revisited model attractor; conflict is non-stationary over 36 months. Also **C-4555**'s imperfect-analog cost is *"extra spread"*, which compounds the measured over-dispersion in 9 of 10 bins. Test: analog distance against temporal separation. |
| **M-12** | 3 | The hidden state is stored as a spatial tensor and probed as a per-cell vector | `T-2.2` | §7.4, §7.3, `T-2.2` | Flattening a convolutional recurrent state discards the latent neighbourhood, which is the architecture's inductive bias and the most likely location of the information H2 seeks. Fix: one patch probe. |

---

*§14 ends. Panel run 2026-11-03 under the expert-method-review protocol; read-only, additive, nothing removed.*
