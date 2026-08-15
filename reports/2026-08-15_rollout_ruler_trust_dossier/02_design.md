# 02 — Design: why the T>0 ruler is untrustworthy, and the three additions that fix it

**Date:** 2026-08-15 · **Epic:** #263 · **Status:** S0

---

## 1. The design problem

The ruler can report a **zero-driven artifact as a win**, and has already done so once.

`crps_all` on a 99.3–99.7% zero field is dominated by the zeros. The V2 scoreboard's headline —
`gated_NB` h36 `crps_all` 0.877 vs climatology 0.960 — decomposes exactly (S1):

```
Δcrps_all = (1 − pₑ)·Δcrps_none  +  pₑ·Δcrps_events
          = 0.98956 × (−0.0639)  +  0.010438 × (−1.9268)
          =        −0.06323      +        −0.02011        = −0.08334   ✔ (observed −0.0833)
```

**75.9% of the "win" is confident zeros.** Meanwhile AP is *worse* (0.162 vs 0.195) and `size_ratio = 0.0000`.
A number that means "the model has learned to be confidently empty" was read as skill.

## 2. The three additions

Everything else is imports. These three are what the ruler lacks:

| # | Addition | Closes | Story |
|---|---|---|---|
| **A1** | **A reference forecast the scorer can build itself.** FAO-02's baseline **is** implemented — as `ConflictologyModel` in views-baseline (`white_ranger` / `light_strider`) — but scoring against it needs its prediction cubes, which are deleted after scoring. The scorer's only in-process baseline was a 1-sample persistence whose CRPS is just MAE, so `crps_all` had no usable denominator *inside the ruler*. `climatology_resample` is a stand-in matched to the canonical parameters (0.9591 vs its archived 0.9601). Duplication risk: **C-279**. | C-219 | S3 |
| **A2** | **The zero-share decomposition, emitted by default** — every headline row carries `zero_share_of_gap`, and the assembler *raises* on a row without the all/events/none split + AP. Makes "never headline a bare `crps_all`" a code invariant rather than a norm. | C-219, C-231 | S1, S3 |
| **A3** | **Provenance asserted, not assumed** — partition non-leak, `rollout_feedback == 'sample'`, cube-not-mean, truth hash, and one-artifact-per-arm (Giacomini's fixed-scheme requirement) all become raises. | C-217, C-218, C-220 | S2 |

Plus one bounded diagnostic (**S5**, C-224) and the driver that joins them (**S6**).

## 3. The ask we are NOT making yet

The absorbed catalog below makes a strong case that FAO-02's blessed metrics (CRPS, MCR) were blind to the
#258 failure, and that some of its rejected ones deserve re-examination. **This dossier does not make that
ask.** It produces the evidence — a decomposition showing where a proper-score win actually came from, and a
tail diagnostic that does not violate FAO-02's twCRPS rejection — and stops there.

Reopening FAO-02 is a governance conversation with its owner, out of scope (`SCOPE.md` #7). What this dossier
contributes to it is: *here is a case where the sanctioned set said "win" and the decomposition said
"confident zeros", measured, on the sanctioned substrate.*

---

# Absorbed from the 2026-08-13 DRAFT

> Absorbed **verbatim** from `reports/2026-08-13_evaluation_pitfalls_and_metric_battery_DRAFT.md`
> (§1 DGP, §2 catalog A–I, §3 meta-patterns), which is superseded by this dossier. The one bad citation
> (`C-1830`) is corrected to **C-231** below. The remaining citations are unverified — the DRAFT's own header
> says several are from memory; verifying them is a `/verify-sources` job, parked as `SCOPE.md` P4.
>
> **Do not re-litigate individual entries.** They are the design rationale, not open questions. Where an entry
> is directly instantiated by a cluster-16 register concern, it is annotated `[→ C-xxx]`.

## 1. Why this task is a metric minefield (the data-generating process)

Everything below follows from the DGP. The task is adversarial to naive evaluation:

- **Extreme sparsity.** Monthly PRIO-GRID conflict fatalities: **~99.3–99.7% exact zeros**, ~0.3–0.7%
  positive. Any metric averaged over all cells is dominated by the zeros.
- **Heavy tails.** Positives are heavy-tailed, EVT **ξ ≈ 0.8**; single cells reach 10⁴–10⁵. Proper scores
  lose power exactly on the tail that matters (Lerch2017 Scenario B).
- **Spatiotemporal structure.** Grid cells × months; spatial persistence dominates (recent realized
  intensity → current, spearman ≈ 0.37).
- **Autoregressive free-running rollout.** 36 steps, prediction→input, trained one-step-ahead only. Errors
  compound; the object at h=36 is not the object at h=1.
- **Composed (hurdle/gated) output.** occurrence gate × magnitude body — two heads that can fail
  independently and mask each other.
- **Stochastic.** posterior samples (D×K MC-dropout × head draws); the emitted *mean* and the drawn
  *sample* are different random variables.
- **The amount-ceiling.** *How big* a jump is is close to unpredictable (spearman 0.303 < persistence
  0.367, confound-clean; `2026-07-14_amount_ceiling_dossier`). *Whether/where* and *how volatile* IS
  predictable (jump-risk spearman ≈ 0.79). ⇒ the right thing to score is occurrence/rank/volatility, not
  point magnitude — but most scalar metrics score magnitude.

**One-line thesis:** on a 99.5%-zero, heavy-tailed, autoregressive, gated, stochastic target, *almost every
convenient scalar metric is dominated by the zeros, blind to placement, blind to frequency, low-power on the
tail, and computed on a different object than the one that rolls out.* You have to design the evaluation as
deliberately as the model.

---

## 2. The catalog of failure modes

Grouped. Each: **what it is → the instance we hit → the lesson.**

### A. First-moment & aggregate blindness (the metric can't see the defect)

- **A1 — MCR is blind to spatial over-concentration.** `MCR = mean(y_pred)/mean(y_true)` is a first-moment
  ratio. A forecast can score MCR≈1 while placing all mass on 1/30th of the correct cells with huge values
  (few cells × 99,465 ≈ right total). *Instance: issue #258* — MCR≈1.0–1.5 at m1 read as "calibrated," was
  actually 4–115× under-activation compensated by over-magnitude. **Lesson: never trust a first-moment
  aggregate as a calibration statement.**
- **A2 — MCR/CRPS are blind to activation frequency.** Nothing in the headline set measured
  `P(y_pred>0)` vs `P(y_true>0)` per step — the one quantity that was 4–115× wrong. **Lesson: activation
  frequency (occurrence rate) must be a first-class metric.**
- **A3 — CRPS goes flat once the field is sparse.** From ~step 6 onward all 8 roster models had CRPS  `[→ C-219]`
  identical to **3 decimals**; pooled CRPS spanned **0.9%** while MCR spanned **10×**. Once every model
  predicts ≈0 on a mostly-zero target, CRPS measures the *actuals*, not the model. *Instance: #258 comment
  2.* **Lesson: a proper score can be simultaneously "proper" and non-discriminating; check its dynamic
  range across the model set before ranking on it.**
- **A4 — "confident zeros, not event skill."** crps-all (all-cell CRPS) rewards predicting zero on a  `[→ C-231]`
  99.5%-zero field; the long-horizon "win" of gated_NB was driven by confident zeros, not by getting events
  right (register C-231 / similar). **Lesson: on sparse targets, an all-cell proper score is an occurrence-
  free lunch.**
- **A5 — Brier (occurrence) also flattens.** `Brier_cls_sample` spanned 0.0052–0.0057 across the roster —
  as uninformative as CRPS once sparse. **Lesson: even the "occurrence" metric can be dominated by the
  easy true-negatives; use rank/PR metrics (AP/AUPRC) that condition on the positives.**
- **A6 — evaluation-window artifacts masquerading as model properties.** A **6.6× CRPS discontinuity at
  s36** (0.133→0.875), *identical for all 8 models* — a property of the eval window, not the models.
  **Lesson: any horizon-varying number must be sanity-checked against a "same for all models ⇒ it's the
  window, not the model" control before it is quoted externally.**
- **A7 — a metric that flagged the problem but whose *severity* wasn't legible.** `size_ratio ≈ 0.02`
  (timid body) was known for months and called "the next lever" — the number was right, but it was read as
  "magnitude to improve later," not "rollout-fatal miscalibration." **Lesson: a metric being *present and
  correct* does not mean its *consequence* is understood; tie each diagnostic to a decision threshold, not
  a vibe.**

### B. The mean/sample split & the horizon (you evaluate a different object than the one that acts)

- **B1 — scored-emit ≠ fed-back-input.** The scored/emitted product is the **mean composition**  `[→ C-220]`
  `compose_mean = gate·μ` (dense, well-behaved first moment). The autoregressive rollout feeds back the
  **sample composition** `compose_samples = Bernoulli(gate)·NB_draw` (sparse, degenerate). Two different
  random variables (`distributions/composition.py`). A thorough T=0 read of the mean can be spotless while
  the fed-back sample is garbage. *Instance: #258.* **Lesson: score the object you deploy (the sample /
  the rollout), not the convenient proxy (the mean).**
- **B2 — T=0-only evaluation.** After the bloom made rollout scores untrustworthy, we scored **T=0 only**  `[→ C-218]`
  (the frozen lodestar ruler). The horizon was rarely the object of evaluation; the activation deficit is
  present at T=0 too, but MCR≈1 hid it there. **Lesson: a defect can be present at T=0 and invisible at
  T=0 simultaneously — you must measure the *right statistic* at T=0, not just "evaluate at T=0."**
- **B3 — a mitigation validated on the one horizon where it cannot differ.** The `sample`-feedback bloom
  fix is documented "T=0-neutral so the scored T=0 product is byte-unchanged" — and all three feedback
  modes are *identical at m1* (0.0301). The acceptance criterion (T=0) is by construction unable to
  distinguish the thing being changed (rollout feedback). **Lesson: an acceptance test must be able to
  *see* the axis it is accepting; a byte-identical-at-T=0 criterion cannot validate a rollout change.**

### C. Goodhart / metric-gaming (optimizing the metric ≠ optimizing the thing)

- **C1 — Forecaster's Dilemma / subset-on-outcome.** Selecting or scoring on the positive subset  `[→ C-219]`
  (`crps_events`) is improper (Gneiting2011 Eq 2.10; Lerch2017). *Instance:* nb "beats climatology" on
  `crps_events` ≈14 with `size_ratio` 0.0 — the timid-zero Goodhart trap; the all/events/none split caught
  it. **Lesson: NEVER select on an outcome-conditioned score. Pre-commit the split.**
- **C2 — proper-score low power on the tail.** twCRPS / threshold-weighted scores lose power to **zero**  `[→ C-224, C-254]`
  at extreme thresholds in exactly our regime (Lerch2017 Scenario B: heavy-tailed truth, forecasts differ
  on the positive half-axis). A null on twCRPS can mean "no difference" OR "no power." **Lesson: report the
  *power* / minimum detectable effect, not just the p-value; a stratified conditional-predictive-ability
  test (Giacomini–White) on an ex-ante high-risk stratum is the escape, and even it must state its MDE.**
- **C3 — rescaling ≠ calibrating.** A winsorized τ-pinball dial lifted magnitude (games `size_ratio`) but
  **exploded CRPS** (`body_knob_quest`). **Lesson: a knob that moves the target metric while destroying a
  proper score is gaming, not improvement.**
- **C4 — magnitude XOR calibration.** `body_mask=pos_cells` lifted `size_ratio` ×11–60 but blew crps-all
  (ns 0.083→24.6; `2026-07-18_body_mask_sweep_dossier`). Two metrics that move in opposition mean the model
  is trading, not improving. **Lesson: watch metric *pairs*; a lift you can only buy by detonating its
  partner is the wall, not a win.**
- **C5 — in-sample winners evaporate.** `count_mean` loss lifted in-sample, collapsed OOS. Baseline
  horse-race: in-sample family winners evaporated on validation. **Lesson: validation partition + ≥3 seeds
  is the floor; in-sample rankings are worthless here.**
- **C6 — a "win" that was a bug.** dense-mse "beats baseline" was a **mismatched-months + dead-body
  artifact** (corrected). **Lesson: a surprising win is a bug until proven otherwise; check the harness
  before believing the number.**
- **C7 — the different-months bug.** Comparing models on different month sets inflated a comparison; the  `[→ C-217]`
  **frozen ruler** (identical months/cells/truth) killed it. **Lesson: freeze the comparison substrate
  (months, cells, truth) into one immutable ruler; never compare across substrates.**

### D. Stability / reproducibility (the number isn't even stable)

- **D1 — seed bimodality.** The floor was seed-bimodal (~40% bad basins); a single-seed read could land in
  a bad basin and be reported as "the model." Root = **recurrent BatchNorm** (C-184); fix = post-train BN
  recalibration. **Lesson: multi-seed is not optional; a single run is a sample from a bimodal
  distribution.**
- **D2 — non-determinism.** Non-deterministic ops made "byte-identical" claims fragile until a determinism
  gate was built. **Lesson: a determinism gate is a prerequisite for any parity/regression claim.**
- **D3 — stale-cache / all-equal-but-wrong.** The ensemble silently reloaded stale cached predictions at
  the *same wrong* sample count, so a config change was silently discarded with no error (C-85). **Lesson:
  cache identity must be fingerprinted on the config that determines the output; "the progress bar finished
  fast" is the only tell otherwise.**
- **D4 — fragile single-read claims.** Repeatedly, a single read was over-interpreted and later retracted
  ("no fragile single-read claims" became a standing rule). **Lesson: no load-bearing claim from one
  read.**
- **D5 — never pool the rollout (MCR trap).** Pooling predictions across the rollout inflates apparent
  calibration (quantile-head dossier). **Lesson: evaluate per-step; pooling hides horizon collapse.**

### E. Composition / decomposition (parts fail independently and hide each other)

- **E1 — gate and body fail independently and compound (double zero-inflation).** `Bernoulli(gate)·NB`
  applies zeros twice: the gate models P(y>0), and a plain NB with tiny unconditional mean already has
  P(0)→1. Composed activation 12–69× below what the gate implies (`implied P(body>0)` 0.015–0.154 where it
  should be ≈1). The composed metric cannot see which part failed. *Instance: #258.* **Lesson: score the
  gate and the body *separately* against truth (occurrence recall/precision for the gate; conditional
  magnitude + P(body>0) for the body) — a composed scalar is a lossy sum of two independent defects.**
- **E2 — score-time vs emit-time composition changes the verdict.** The lodestar crps-all is
  gate-*independent*, so an early "score-time gated_NB" was scored **ungated** and undersold; properly
  composed, gated_NB ≈ th_gated_NB (Epic #183). **Lesson: the composition must be applied *in the model at
  emit time*, identically to deployment; scoring a differently-composed object silently re-ranks arms.**
- **E3 — a whole channel silently dropped.** The ensemble `concat` pool dropped the `by_*` occurrence gate
  channel entirely; AP was understated with **no error signal** (C-132/C-286). **Lesson: assert channel
  presence in the pooled output; a missing channel is invisible to any downstream metric.**
- **E4 — you can't decouple a jointly-fit decomposition.** `gated_ZINBcore` (strip ZINB's π, gate the NB
  core externally) destroyed the property that made ZINB win — π and the core are jointly fit
  (`postmortem_gated_zinbcore`). **Lesson: a decomposition that scores well as a unit may not survive being
  split; the parts are not independently transplantable.**

### F. Distribution / representation traps

- **F1 — "a plain NB is plain" is false in practice.** An NB with near-zero unconditional mean is
  *effectively zero-inflated* (P(0)→1) and, with small learned θ, simultaneously fat-tailed ("mostly zero,
  occasionally 99,465"). The design assumed the gate supplies zeros and the NB supplies magnitude; the NB
  supplies zeros too. *Instance: #258.* **Lesson: characterize the *realized* zero mass of the body,
  not its nominal family label.**
- **F2 — numeric blow-ups masquerade as model behavior.** `log1p` emit was catastrophic (baseline);
  lognormal inverse `exp(µ)` overflowed float32 → **63% Inf** (C-72). **Lesson: a "bad" result can be a
  numeric overflow; check finiteness before interpreting.**
- **F3 — continuous families hit a structural sparse-cell wall** (tweedie/lognormal). **Lesson: the
  container (family) can be wrong for the DGP independent of fit.**
- **F4 — the tail SHAPE, not mean-decoupling, is the gap.** A 2-NB mixture-density head did NOT crack the
  amount wall (GW sig but sub-5%, h=1-only); component-2 was alive but magnitude stayed capped ⇒ tail shape
  (ξ>0), not mean-decoupling (`2026-08-01_tail_decoupled_head_dossier`). **Lesson: a "richer" head that
  doesn't move the target statistic is a null, not a win; measure the specific thing (tail shape).**

### G. The autoregressive rollout

- **G1 — the bloom is feedback, not loss/gate** (C-113). The 36-month explosion is autoregressive, dissolvable
  by the gate at T=0 but not free-running. **Lesson: rollout pathologies are a *dynamical-systems* property;
  T=0 diagnostics cannot see them.**
- **G2 — exposure bias.** Trained one-step-ahead (`prev_pred.detach()`), run 36 steps free-running; the model
  never sees its own output distribution as input (Axis B, parked). **Lesson: train/deploy input-distribution
  mismatch is a first-order effect at long horizon.**
- **G3 — a mitigation that trades one failure for another.** `sample` feedback (ADR-070) killed the *bloom* but
  the sparse fed-back field *collapses* the rollout — ablation: `mean`→bloom (8.13 @ m36), `sample`→collapse,
  `teacher_forced`→flat. **Lesson: a rollout mitigation must be evaluated for the failure it *introduces*, not
  only the one it removes — and across the full horizon.**
- **G4 — feedback/training-input mismatch unifies bloom & collapse.** What you feed back (dense mean → compounds
  → bloom; sparse sample → off-distribution → collapse) matches neither the real training inputs (sparse *but
  persistent*, 0.46% active, moderate values). **Lesson: the fed-back quantity should be as close as possible to
  the training-input distribution; both current options are off-distribution.**
- **G5 — the oracle-input rollout is the decisive diagnostic; it *localises* the failure.** (2026-08-14, africa
  truncated_nb.) Re-emit the SAME trained model with `rollout_feedback='teacher_forced'` (feed the REAL month-t
  field each step) and compare gate AP-by-horizon to the free rollout. Result: **oracle AP holds 0.30→0.27 across
  h1→h36** (activation stays calibrated ~1.2), while free-`sample` collapses (0.30→0.008) and free-`mean` blooms
  (act_ratio →96×), both AP-blind (~0.01). **Interpretation:** given in-distribution inputs the model is skilful
  at every horizon ⇒ the collapse is NOT the predictability ceiling (I1), NOT hidden-state drift, and NOT
  input-exposure *quantity* — it is the **distributional gap**: the model can't emit a fed-back field that *looks
  like real conflict history*. The two naive feedbacks bracket the target oppositely; only the oracle works.
  **Lesson: before building a rollout fix, run the oracle-input probe — it separates "the model can't predict far"
  (ceiling) from "the model can't feed itself" (distribution gap), which demand completely different fixes.**
- **G6 — a popular exposure-bias fix can be a pre-registered null.** Scheduled sampling (ADR-056; the *partial* (c)
  — feed back own prediction during training, but still score each step vs the true target) was **falsified** as
  the fix here: dose-response ε∈{0,0.1,0.25,0.5} gave flat rollout AP and *harmed* T=0 at higher ε. **Lesson:
  "expose the model to its own inputs" (SS) attacks exposure *quantity*; when the failure is feedback *quality*
  (the generated field isn't realistic), SS trains on garbage and doesn't help. Pre-register the falsifier and
  judge on activation-aware metrics across the horizon, not `crps_all`.**

### H. Baseline & comparability

- **H1 — which baseline horizon you report changes the story.** Models beat datafactory climatology on short-
  horizon occurrence CRPS but climatology overtakes by ~h18 (magnitude regime). **Lesson: report the full
  horizon vs baseline; a single-horizon "beats climatology" is cherry-picking.**
- **H2 — v1/v2 non-comparability.** viewser (v1) vs datafactory (v2) results are a clean-cut, *not* comparable;
  the frozen scoring functions are reused but the truth is re-anchored (ADR-071). **Lesson: never compare across
  data foundations; state the substrate.**
- **H3 — the locked eval framework exists for a reason** (FAO-02): CRPS primary + QS99/Brier/MCR guardrails, vs
  empirical baseline, **validation partition**, 5%/1% — twCRPS/LogScore/PIT were rejected. **Open tension: the
  metrics FAO-02 blessed (CRPS/MCR) are exactly the ones #258 showed blind; the rejected ones (PIT, activation-
  aware) may need re-examination.** This document is partly a case for reopening FAO-02.

### I. Epistemics — distinguishing "model is bad" from "task is unpredictable"

- **I1 — the amount-ceiling is irreducible for magnitude** (spearman 0.303 < persistence 0.367, confound-clean).
  A metric punishing magnitude error may be punishing the *task*, not the model. **Lesson: measure predictability
  (info-theoretic / rank-correlation vs a persistence confound) before blaming the model.**
- **I2 — but volatility/jump-risk IS predictable** (spearman 0.79; quantiles 48% sharper; tail ξ≈0.8). **Lesson:
  score the predictable thing (occurrence, rank, volatility, intervals), not the unpredictable one (point
  magnitude).**
- **I3 — calibration must be on active cells.** All-cell PIT is dominated by the zeros; the eval bar needs
  active-cell PIT / conditional calibration. **Lesson: condition the calibration diagnostic on occurrence.**
- **I4 — the oracle-input rollout separates "unpredictable" from "self-poisoning."** A *rollout* collapse has two
  very different causes that look identical in the free-running score: (a) the far horizon is genuinely
  unpredictable (I1 ceiling), or (b) the model would be skilful on real inputs but poisons itself with an
  off-distribution fed-back field (G5). **The teacher-forced (oracle-input) rollout is the clean discriminator:**
  if oracle AP stays high at h36, the signal exists and the failure is self-poisoning (a *fixable* engineering
  problem); if oracle AP also collapses, you are at the ceiling. **Lesson: never diagnose a rollout collapse as a
  ceiling (or as exposure bias) without the oracle-input probe — it is cheap (emit-only) and decisive.**

---

## 3. Meta-patterns (the recurring shapes across the catalog)

1. **The zero-domination pattern.** Any all-cell aggregate on a 99.5%-zero field is a report about the zeros.
   (A3, A4, A5, C1, I3.)
2. **The wrong-object pattern.** We repeatedly evaluated a different object than the one deployed: mean vs sample
   (B1), score-time vs emit-time composition (E2), T=0 vs rollout (B2/G1), pooled vs per-step (D5). **The single
   most dangerous class.**
3. **The compensating-error pattern.** Two errors cancel in the aggregate (under-activation × over-magnitude →
   MCR≈1; A1, F1). The aggregate looks calibrated because the errors hide each other.
4. **The blind-acceptance-test pattern.** The acceptance criterion cannot see the axis it accepts (B3).
5. **The trade-not-improve pattern.** A metric lift bought by detonating its pair (C3, C4, G3).
6. **The half-known pattern.** The register/dossiers *contained the breadcrumb* (size_ratio≈0, C-231 "confident
   zeros"), but the severity/decision-relevance wasn't wired to a gate (A7).
7. **The power-vs-null pattern.** A null result conflated "no difference" with "no power" (C2).
8. **The invalid-knowledge-from-a-bug pattern.** A wrong *implementation* silently produces a confident *verdict*.
   (2026-08-14: a buggy zero-truncated sampler — 128 full-grid rejection rounds instead of scatter — made the
   SAME model+config read as a **bloom** (2 origins, slow sampler) and then, once fixed + scored on all 13
   origins, as a **collapse** — the opposite conclusion. Cf. the emit_family_core half-wiring that overturned an
   F2 verdict, and the "different-months" bug.) **The countermeasure is not a metric — it is process:** verify the
   implementation *before* trusting the experiment (code review + a train/inference parity test + enough
   origins), because no metric can distinguish "the model did X" from "the code that computed X was wrong." This
   is the strongest argument for the verify-correctness-first discipline that precedes any of the metrics above.

---

---

## PARKED — DRAFT §6 (research article)

The DRAFT's §6 research-article outline (working title *"When the ruler lies: failure modes in evaluating
sparse, heavy-tailed, autoregressive spatiotemporal count forecasts — and a metric battery that doesn't"*) is
**parked verbatim in the original DRAFT file** and is out of scope for this dossier (`SCOPE.md` #22).

It is not deleted, not rewritten, and not worked on. When the programme has a validated ruler and a body of
results, it is a separate decision.
