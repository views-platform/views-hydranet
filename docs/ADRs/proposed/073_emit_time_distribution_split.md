# ADR-073: Separate the EMITTED distribution family from the TRAINED one

**Status:** Proposed — ⛔ **NOT ADOPTED.** The capability ships default-off; the decision below
is **not taken**, and `emit_distribution` must not be set in a roster config.
**Date:** 2026-10-21 · **Verdict added:** 2026-10-22
**Deciders:** Simon Polichinel von der Maase
**Informed:** views-hydranet contributors; views-models roster owners

---

## 0. ⛔ VERDICT (2026-10-22) — read this before §1

> ⛔⛔⛔ **CORRECTION, 2026-10-29 — the label on this document's decisive measurement is wrong.**
> The §0 claim that the family-free fit scores *"−17.6% against the best parametric cell and −22.5%
> against what ships today"* compares **two numbers from the same teacher-forced cube** to a phrase
> naming the delivered 36-step autoregressive rollout. `reports/2026-10-21_emit_time_split_prereg.md:102`
> confirms every arm in the 2×2 — 0.050758, 0.047741, 0.046023 — was scored on the frozen `nb`
> control cube, which `vehicle.py:301-307` fills from observed truth at every step. ⇒ The **margin is
> like-for-like and survives as an emission-family comparison at T=0**; the words *"what ships today"*
> are false, because what ships is the autoregressive rollout and it has never been scored. On the
> deliverable index the family-free method scores **0.067752** against **0.060323** for no model at
> all. **The arithmetic was never wrong; the label was.** See **M85**, **C-399**, **W-107**.


A seven-seat `expert-method-review` and the measurements it prompted rejected this ADR's
**decision**. The code it describes is merged and tested, default-off and byte-identical when
unset, and it stays that way: a capability, not a route. What follows is kept as the record of
why, so this is not rediscovered in three months.

**Four of this document's own load-bearing claims are refuted below**, each in the section that
makes it. In summary:

| § | the claim | verdict |
|---|---|---|
| §3 | truncation is preferred because it "has **no free parameter**" | ⛔ **REFUTED.** As `theta -> 0` the NB is *identically* `(1-pi)·delta_0 + pi·LogSeries`, with `pi = theta·log(1 + mu/theta)`. Truncation and the theta-rescale set the **same scalar**: truncation pins `pi = 1`, the x22 rescale sets `pi = 5.1e-04`. Pinning a parameter at its boundary by fiat is not parameter-free. ⇒ It is also why §4.3's "substitutes, not additive" was a *prediction* of the identity rather than a surprise. |
| §1, §4 | the emit-time route "takes the truncation **without the retrain** that caused that damage" | ⛔ **REFUTED.** Truncation replaces the emitted magnitude `mu` with `mu/(1-p0)`, which **inverts** the field's rank correlation with the gate from **+0.4925 to −0.1605** and collapses high-gate separation from **18,348x to 0.65x** (swept over four cuts, frozen `nb` control cube, out of sample, **nothing retrained**). The damage is a property of truncation, not of retraining. |
| §1, §4 | the headline **−5.94%** composed CRPS | ⚠️ **ESTIMATOR-DEPENDENT, and FAO-02 does not say which estimator is primary.** The plug-in `crps_ensemble` is biased by `~E|X−X'|/(2m)` *per arm*; the two arms' spreads differ, so the bias differs **5.5x** (0.001129 vs 0.006184) and does **not** cancel at equal `m`. Under the unbiased (fair) estimator the same cube gives **−16.27%**, and the levers become **antagonistic** rather than substitutes. Ferro2014 **C-1384** proves the plug-in systematically favours **under-dispersed** ensembles; Zamo2018 **C-1457** says use the fair estimator for the infinite-member question, which is the one a design decision asks. |
| §5 | the feedback family is "**open**, settled by measurement in #433" | ⛔ **ANSWERED BY THE CODE'S OWN DEFAULT**, which §5 did not state. `hydranet_inference.py` resolves the fed-back draw from the **emitted** family, so adopting this ADR would have fed truncated draws through all 36 rollout steps — the configuration M45 measured at **x8–22 compounding by h18** and **−0.238 AP**. The comment at the draw site is marked PROVISIONAL and cites the conflict, so this was documented rather than silent; it was still a live default presented as an open question. |

**And the decisive measurement this ADR could not have known.** A **family-free** conditional CDF,
conditioned on nothing but the model's own `gate x mu` index and fitted on months 421–456, scores
**0.039327** out of sample on the same 842,634 cell-months with the same m=16 plug-in estimator —
**−17.6%** against the best parametric cell in §4.3's 2x2 and **−22.5%** against what ships today.
Bin count **swept**: it loses below 250 bins, crosses around 1,000, peaks at 8,000 and turns back
down, so the optimum is out-of-sample and not an overfit. **~99.7% of that gain comes from the
1.36% of cells where it forecasts anything at all** — it matches the delivered model on the quiet
background (0.000260 vs 0.000266) and beats it on the conditional law where events occur
(**10.831 vs 14.589**, −25.7%).

⇒ **Truncation is strictly dominated.** It buys its event-cell gain (14.589 -> 12.034) by *paying*
13.5x on the background (0.000463 -> 0.006274). The family-free fit takes a larger event-cell gain
at **no background cost**. The live question is therefore not which family to emit from — it is
that a two-parameter negative binomial cannot represent the conditional law the trunk already ranks
well, and no choice of emitting family fixes that.

**Where the real defect is.** The NB negative log-likelihood of a zero observation is
`theta·log(1 + mu/theta)`, which is monotonically **increasing** in `theta`. On an empty cell the
loss is therefore minimised by driving `theta -> 0`, at every `mu`, with **no interior optimum** —
and 99.66% of observations are empty cells. `theta = 3.6e-06` is what the objective asks for, not a
convergence failure. ⇒ No emission rule, no sampler fix and no rescale repairs it. See #396.

⚠️ **One statistic in this document is reported on the wrong population** and is corrected here:
§4's "`TruncatedNBFamily.sample` floors to 1 across the `P(Y=0) > _MODERATE_P0` background, which is
most of this grid" is true of the **grid** and misleading about the **emission**. Gate-weighted mean
`theta` on the same cube is **0.0902**, 24,000x the grid median, and gate-weighted
`P(Y >= 100 | Y > 0)` is **0.00054**. The cells the median describes are not the cells that emit.

**Superseding work:** `reports/2026-10-22_emit_time_split_method_review.md` (the full review, the
seven seats, and every verification verdict); #396 (the live diagnosis); and the two issues the
review opened — the CRPS-estimator gap in FAO-02, and reviving the family-free head.

---

## 1. Context

**Problem.** The body decides both *whether* an event happens and *how large* it is, and it says
"nothing" far too often. On the out-of-sample partition the delivered `nb` model emits a positive in
**0.0009** of cell-months against an observed **0.0034** — about a quarter of reality. The gate, by
contrast, is calibrated: median `|gate − p| = 0.0054` over 842,634 cell-months. The composition work
measured that the body draws zero in **~57%** of draws, so the miscalibrated half is deciding
occurrence while the calibrated half is discarded.

**Assumption no longer true.** `output_distribution` has always been treated as one choice. It is
two: it selects the family for the **training loss** (`utils/utils.py:186-199`) *and* for the
**emit-time draw** (`utils/hydranet_inference.py:270`). Nothing requires them to be the same, and
the combination that the evidence favours — train as `nb`, emit zero-truncated — is unreachable
because they share one key.

**Urgency.** Measured on the frozen `nb` control cube, out of sample, **nothing retrained**
(`reports/2026-10-21_emit_time_split_prereg.md`, OUTCOME):

| emission | composed CRPS | emitted positive fraction |
|---|---|---|
| NB (today) | 0.050758 | 0.0009 |
| **zero-truncated** | **0.047741 (−5.94%)** | **0.0037** |

Observed positive fraction **0.0034**, so the emitted rate lands at **1.09× truth**. ⛔ This was
pre-registered at 65% confidence to be *worse* by 5–30%; falsifier `F2` fired and the prediction is
logged wrong as `W-55`.

And it is the question #396 says nobody has asked: *"the untested alternative is that every attempt
overshot and the rollout amplified it, and **nobody has tried getting firing right**."* Retraining
on `truncated_nb` reached 1.16–1.75× and destroyed the body — measured the same day, the
`truncated_nb`-trained body separates conflict cells from their empty neighbours by **1.0–1.1×**
where the `nb`-trained body separates by **5–70×**. The emit-time route takes the truncation without
the retrain that caused that damage.

---

## 2. Decision

**Statement.** *We will add `emit_distribution` so a model can be trained on one distribution family
and emitted from another, to let the gate own occurrence and the body own magnitude without
retraining.*

- **In-Scope:** the family used for the emitted draw, the emitted magnitude (`family.mean`), and —
  subject to §5's open question — the autoregressive feedback draw. Default `None`, meaning *emit as
  trained*, which is byte-identical to today for every existing configuration.
- **Out-of-Scope:** any change to training; to `forecast_composition`'s three values; to FAO-02; or
  to which family a roster model is trained on. Retiring `truncated_nb` as a *trained* family is a
  separate, already-taken decision (views-models#590).

---

## 3. Rationale & Integrity Impact

⛔ **THE PARAGRAPH BELOW IS REFUTED — see §0.** Truncation and the `θ` rescale set the same
scalar `π = θ·log(1 + μ/θ)`; truncation pins it at 1. "No free parameter" was the stated reason to
prefer this route and it is not true.

**Logic — why this rather than rescaling `θ`.** A single multiplicative rescale of the body's
dispersion recovers a comparable **9.28%** held out. It is rejected as the primary route because it
needs a **scale fitted on data**, re-fitted per model, and tuned against the metric it is scored on.
Zero-truncated emission has **no free parameter**: it is a change of rule. The 2×2 measured them
together and found they are **substitutes, not additive** — each alone recovers 6–9%, both together
9.33% — so the parameter-free one is preferred at near-equal benefit. Truncation also lands the
emitted rate at 0.0037 against 0.0034 observed, where the `θ` route reaches only 0.0022.

**Fortress State.** The surface is kept narrow by a **structural** constraint rather than by a
convention: the head's channels were trained to mean the *trained* family's parameters, so an
emitting family must declare the same `n_params`. Today that admits exactly `nb ↔ truncated_nb`
(2 parameters each) and refuses `zinb` (3) automatically. The admissible set therefore cannot widen
by accident; it widens only if a family with a matching parameter layout is added deliberately.

**Fail-Loud.** The field is **declared in the schema**, not read only through `config.get`.
`HydraNetConfig` sets `extra="allow"`, so an undeclared key on a package that predates it is
*silently accepted and ignored* — the W-22 trap, which has already cost one confounded ~$2.25 run.
Declaring it makes an older package raise instead. Three rules raise at config time: unregistered
name, mismatched parameter count, and `emit_distribution == output_distribution` (which reads as an
intervention and is none — C-324's inert-knob signature). Every rule is re-checked inside
`HydraNetInference.__init__` for callers that construct from a raw dict, the same pattern
`choose_loss` uses for `body_nll_denominator`.

---

## 4. Consequences

### ✅ Positive

⛔ **EVERY ITEM BELOW IS QUALIFIED OR REFUTED — see §0.** The rate claim is measured against an
**undercounted** truth (UCDP underreporting exceeds **100%** at single-fatality events, which are
exactly the grid-month positives — Vesco2026 **C-113**), so 1.09× is not a calibration endpoint and
must not become a tuning target. The CRPS claim is estimator-dependent. And the gain on event cells
is reported on an evaluation set **restricted by the realized outcome**, which makes a proper score
improper (Lerch et al. **C-79/C-80**) — legitimate as a diagnostic, never as a rationale.

- [ ] The emitted positive rate moves from **0.26×** to **1.09×** of observed, with nothing retrained.
- [ ] Composed CRPS improves **5.94%** out of sample, concentrated in the 0.34% of cells that carry
      ~99% of the score.
- [ ] A capability the roster has wanted since 2026-10-13 becomes expressible at all.
- [ ] No fitted parameter is introduced, so nothing must be re-fitted per model or per vintage.

### ⚠️ Negative

- [ ] **A second axis in an area ADR-069 deliberately kept narrow.** Mitigated by the `n_params`
      constraint, which bounds the admissible set structurally.
- [ ] **`forecast_composition`'s `self_zeroed` rules must now read the emitting family**, because
      composition happens at emit. Unreachable today (the two are always equal) and live the moment
      this lands.
- [ ] **An artifact can be scored under a different emit rule than its sidecar records**, because
      the flag is read from the *scoring* config rather than carried on the model. ⚠️ Pre-existing
      and shared with `emit_family_core`; recorded here, not fixed here.
- [ ] Per-cell truncated draws are more work than plain NB draws, which touches #402. ⚠️ Bounded:
      `TruncatedNBFamily.sample`'s two-regime path floors to 1 without rejection across the
      `P(Y=0) > _MODERATE_P0` background, which is most of this grid.

### Relationship to ADR-068 and ADR-069

**ADR-068 (`emit_family_core`) is not subsumed and must not be folded in.** It answers a different
question: *which part of one family's own distribution to emit* (the π-stripped bulk of a
self-zeroed family), and it applies only where there is a structural π to strip. This ADR answers
*which family to emit from at all*. They compose — `emit_family_core` continues to select
`mean_core`/`sample_core` on whichever family `emit_distribution` resolves to.

**ADR-069 is amended in one respect only.** Its composition rules are statements about the draws the
gate multiplies, which is the **emitting** family's business. They are re-pointed accordingly, with
behaviour provably unchanged when `emit_distribution is None`. The three composition values and
their meanings are untouched.

---

## 5. ⛔ The open question this ADR does NOT answer

⛔ **IT WAS NOT OPEN — see §0.** The code already defaults the fed-back draw to the **emitted**
family, so adopting this ADR would have fed truncated draws through all 36 rollout steps without a
further decision being taken. Calling it "open" while a default existed is the defect, not the
wiring; the comment at the draw site is explicit and marked PROVISIONAL, so it was documented
rather than hidden. #433 is closed as moot — nothing will set the flag — and the live default is
registered as a risk instead.

**Whether the autoregressive feedback follows the emitted family or the trained one.**

`hydranet_inference.py:1052` resolves the fed-back draw from the same `self._family`, so a naive
implementation changes the rollout as a side effect of changing emission. Two recorded positions
conflict:

- **Coherence (C-234, `hydranet_inference.py:1054-1058`):** *"Feeding back the self-zeroed body
  while emitting the core makes the rollout incoherent (h≥2 on a history the model never emitted)."*
  By this principle the feedback **must** follow the emitted family.
- **The bloom record (M45, `RESULTS_LEDGER.md:403`, 4 seeds):** `truncated_nb` moved step-1
  `act_ratio` to 1.16–1.75× truth, which compounded **×8–22 at h18** and **×21–69 at h36**, costing
  **0.238 AP** with the permutation test at its exact floor.

⇒ **Recorded as open and settled by measurement in #433**, not by whichever line is edited. The
decision and the rejected option's numbers are written back into this ADR before it moves to
`active`.

⚠️ **Everything measured so far is at T=0.** #437 tests whether the split compounds in the 36-month
rollout and may conclude that this is a T=0-only capability, in which case that qualifier attaches
to every number above.

---

## 6. Validation

**Invariants.**
- ⛔ With `emit_distribution is None`, the emitted cube is **byte-identical** to today's for every
  roster configuration. Every existing result depends on this.
- `FAMILY_N_PARAMS` equals `{n: get_family(n).n_params for n in family_names()}` — a parity test, as
  `SELF_ZEROED_FAMILIES` already has, so the torch-free mirror cannot drift from the classes.

**Tests (ADR-005).**
- The flag **reaches the emitted cube**, asserted through the truncation mechanism rather than array
  inequality — proving a config validates is not proving it arrives (views-models#578).
- Illegal pairs refused at config time *and* on the raw-dict path.
- Mutation-audited against committed code, each mutation asserting the **code line** changed.
- The engine reproduces the arithmetic (−5.94%, 0.0037) within Monte-Carlo agreement at m=16; the
  engine's sampler is rejection-based where the measurement used inverse-CDF, so agreement is
  distributional, never bit-exact.

⛔ **The "reproduces the arithmetic" invariant above was ILL-POSED TWICE OVER — see §0.** It ran,
it failed (engine **−11.46%** against a measured −5.94%, outside the ±1.0 pp tolerance committed
before the run), and it correctly stopped the feature — that is the gate working. But the target
number it compared against is itself estimator-dependent to **10 points**, so even a perfect
sampler could not have satisfied a tolerance stated that way. A reproduce-the-number gate must name
its estimator. Recorded against #436.

**Failure mode that would trigger reconsideration.**
- The rollout blooms under the chosen feedback wiring (#437) ⇒ T=0-only, or withdrawn.
- Any FAO-02 guardrail degrades beyond its margin (#439) ⇒ a negative, not a trade to be argued.
- The engine fails to reproduce the arithmetic ⇒ what ships is not what was measured; stop.

---

## 7. Implementation Notes

- **Location.** `distributions/registry.py` (`FAMILY_N_PARAMS`), `utils/config_initializer.py`
  (field + validators, including the ADR-069 re-point), `utils/hydranet_inference.py:270` (the
  single seam — every other `self._family` use is already emit-side).
- **References.** Epic #428; tracking #440; stories #429–#439.
  `reports/2026-10-21_emit_time_split_prereg.md` (prediction, falsifiers, outcome);
  `reports/2026-10-13_composition_arm_outcome/OUTCOME.md:77-84` (where this was first named);
  #396 (the under-firing diagnosis this answers a piece of); views-models#590 (the decision to stop
  *training* on `truncated_nb`).
