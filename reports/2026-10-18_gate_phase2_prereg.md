# Pre-analysis plan — does the gate's under-dispersion exist on the models we actually serve?

**Date:** 2026-10-18 · **Phase 2** of the gate under-dispersion programme · **Written before any Phase-2 measurement.**
**Owner:** views-hydranet seat. **Executing seat:** views-models, which **attacks this plan first and does not decide which of its own objections survive.**
**Predecessors:** `2026-10-18_gate_underdispersion_prereg/RESULTS.md` (the complete Phase-1 record) · `RND_PLAN.md` (dependencies) · `2026-10-16_pos_weight_arm_outcome/OUTCOME.md` rev 3
**Gate this satisfies:** ⛔ Simon, 2026-10-18 — nothing runs until the results record, the R&D plan and a strong pre-registration exist. This is the third.

---

## 0. In plain words

We measured two models and found the gate's probabilities are squashed. One of them we can fix with a single number; the other we cannot. **We have never measured any of the eight models actually being served.**

This plan measures them. **It is a measurement, not an intervention** — nothing is changed, nothing is tuned, no model is touched. The question is only: *does this defect exist out there, and on which models?*

⛔ **The one thing that would make this phase worthless is deciding afterwards what we expected.** So the readings are committed below, exhaustively, including the outcomes we think unlikely.

---

## 1. The question, and the one thing being measured

**For each served model: the gate's calibration slope `b` and intercept `a`, from `logit(observed rate) = a + b·logit(gate)`, held out, bin-free and binned.** Honest is `a = 0, b = 1`.

**Population:** the eight `rusty_bucket` members. **Nothing is varied.** There is no arm, no control, no intervention — this is a survey.

---

## 2. ⛔ THE PREREQUISITE THAT DECIDES THE COST, AND IT IS UNMEASURED

Phase 1 could not read the gate from delivered artifacts: `p2_gate_calibration.py` had to **recompute it from the weights with a local forward pass**, because the gate posterior does not come home (#558 / C-161).

⛔ **So "measure the eight served models" may mean eight local forward passes, not eight array reads — and nobody has checked which.** Step 0 of this plan, before any `b` is fitted:

**For each of the eight models, establish and record:** does a gate posterior exist on disk at matchable months? If not, is the artifact + sidecar + `sha256` present so a local sweep is possible? **Report the answer per model before fitting anything.**

⚠️ **The other seat's estimate is ~1.5 h local CPU, $0, no GPU. This seat has NOT measured it, and nine of nine cost estimates across both seats were optimistic.** ⛔ **Measure the first model end to end and report its wall time before running the remaining seven.** That is a pre-registered stop, not advice.

---

## 3. ⛔ Skepticism ledger

### 3.1 The honest prior, with numbers

**P(the defect appears on a majority of the eight) ≈ 0.6.** Reasoning: it is present in both Phase-1 models and its mechanism (BatchNorm statistics from activity-seeking crops applied to a global grid) is a property of the **training and recalibration machinery shared by all of them**, not of either arm's config.

**P(a single remedy works on all affected models) ≈ 0.15.** ⛔ **Reasoning: it already failed this test at n=2.** R1 works on amber and fails on copper, which differ in **one config key**. That is the single most informative fact in Phase 1 about generalisation, and it is discouraging.

**P(at least one model is over-dispersed, `b < 1`) ≈ 0.2**, and it is pre-registered as a real outcome rather than an anomaly (§5, reading 3).

### 3.2 This phase cannot establish a cause, and must not be read as doing so

Every Phase-1 candidate (H2 weight decay, H3 resolution limit, H4 geometry) remains **live and untested**. ⛔ **A survey of slopes across models is consistent with all of them and discriminates none.** Any sentence in the outcome that reads as causal is a defect in the write-up.

### 3.3 The three threshold-gate models are already excluded, and measuring them is a control, not a lead

`RESULTS.md` §8 derives and verifies exactly (200,000/200,000) that a threshold gate is **invariant to a monotone distortion up to the choice of τ**, and that τ was tuned empirically, so **the tuning already absorbed it.**

⛔ **So their `b` is informative but NOT actionable.** Measuring them tests whether thresholding correlates with a different slope — which is interesting — but **no treatment follows from any value.** F5 fires if this plan's own outcome document reasons from their `b` to a treatment.

### 3.4 ⛔ The remedy-selection rule must be fixed BEFORE the readout, and it is fixed here

Phase 1's `implied_w` and the recal-distribution Goodhart ban are the same error: a parameter fitted to the endpoint it is judged on. **Choosing between R1 and R3 after seeing which wins per model would be the third instance.**

**Pre-committed now:**

> For each affected `soft_gate` model, fit **both** R1 (temperature, 1 param) and R3 (Platt, 2 params) under **13-fold origin cross-validation**. Select **R1** unless R3's held-out `|gate−obs|` beats it by more than **one seed sd (0.019)**. ⛔ **Additionally pre-commit the diagnostic that predicts which will win, so it can be scored:** bin-free `|a| > 0.5` predicts R3, `|a| < 0.5` predicts R1. **Amber (`a = +0.16`) → R1; copper (`a = +1.30`) → R3**, which is what Phase 1 measured — so the rule is **falsifiable on the new models and already consistent with the old.**

### 3.5 Every number this phase produces is subject to the sweep rule

⛔ **Before any scalar is reported, sweep the free parameter it depends on and show the family does not cross the value being tested.** Six instances in Phase 1 died to this, two of them inside the amendment that created the rule.

### 3.6 The record search, as a numbered step rather than advice

⛔ **Enumeration is not closed until, for each model and each candidate explanation, the following four places have been searched and the search terms and hits recorded in the outcome:** `reports/`, `scripts/`, **metric definitions inside scoring functions**, and **anything marked "superseded"** — which retires a conclusion, never a tool, an instrument or a falsifier. **Five pieces of prior art were missed in Phase 1, including a committed instrument and a 142× confound measurement.**

---

## 4. Pre-registered predictions — and ⛔ what is NOT a prediction

| # | | status |
|---|---|---|
| **P-1** | `b > 1.05` on a majority of the eight | **TEST** — could fail; prior 0.6 |
| **P-2** | the `a`/`b` diagnostic in §3.4 predicts the R1/R3 winner on every affected model | ⛔ **TEST, and the riskiest thing here.** It is a rule derived from n=2 applied blind to new models |
| **P-3** | `b` and `a` per model | ⛔ **MEASUREMENT, not a test.** No prior constrains any individual value |
| **P-4** | the three `threshold_gate` models' `b` | ⛔ **MEASUREMENT and a control.** No treatment follows (§3.3) |

⚠️ **P-3 and P-4 are labelled measurements because labelling them tests would be Phase 1's E1 error** — bands derived by inverting numbers already in hand, then reported as confirmation.

---

## 5. ⛔ Readings — EXHAUSTIVE over the real line, including the undefined case

Phase 1's `F1⁗` had three branches and the outcome that occurred was in none of them. **This table covers every possible value of `b`.**

| reading | condition | consequence |
|---|---|---|
| **1** | `b > 1.05` on ≥ 5 of 8 | **the defect is platform-wide.** Proceed to Phase 3 per-model remedy selection, `soft_gate` models only |
| **2** | `b > 1.05` on 1–4 of 8 | **model-specific.** Treat only those; the rest need nothing. ⚠️ Report *why they differ* as an open question, not as a cause |
| **3** | ⛔ any model with `b < 0.95` | ⛔ **OVER-dispersed — a NEW finding, not an anomaly.** R1 would make it **worse**. That model leaves the treatment set and gets its own write-up. The defect is then **not directional** and `OUTCOME.md §2` needs qualifying |
| **4** | all eight within `0.95 ≤ b ≤ 1.05` | ⛔ **the defect does NOT generalise beyond amber/copper.** Phase 3 is cancelled; the finding becomes specific to the two kestrels and the programme's scope collapses to them |
| **5** | `b` not fittable for a model | **VOID for that model** (F1), not negative. Report which and why |
| **6** | mixed, with no majority and no clean split | **INCONCLUSIVE** (C-307). ⛔ **No treatment is selected.** Report the eight numbers and stop |

---

## 6. ⛔ Falsifiers — pre-committed, with consequences

| | fires if | then |
|---|---|---|
| **F1** | a model's positives fall below **300**, or its gate saturates (any bin at 0 or 1 beyond the clip) | ⛔ **VOID for that model.** 300 is the floor Phase 1's E6c failed at (314 positives, `sd` 0.047 against a 0.027 gate) |
| **F2** | bootstrap `sd(b)` for a model **≥ 0.027** | ⛔ **that model is UNMEASURED, not honest.** Its `b` may not be quoted or used in a reading. **Measured per model BEFORE its `b` is read** |
| **F3** | the harness fails to reproduce amber's bin-free `b = 1.1875` **to three decimals** | ⛔ **VOID — everything.** Stop and fix the harness. This is the paired-baseline precondition that E6f met exactly |
| **F4** | the **sign** of any `b − 1` changes across the three bin resolutions, or bin-free and binned disagree in sign | ⛔ **VOID for that model** by §3.5's sweep rule |
| **F5** | the outcome document reasons from a `threshold_gate` model's `b` to a treatment | ⛔ **a REASONING error, not a result.** §3.3 derives why; retract and re-write |
| **F6** | per-model positive counts differ by more than **10×** and any `b` is pooled across models | ⛔ **VOID the pooled number.** Report per model, never pooled — this is the #258/MCR blindness in another guise |
| **F7** | the first model's measured wall time exceeds **3× the ~1.5 h estimate** | ⛔ **STOP and report** before running the remaining seven (§2) |
| **F8** | any cost, power or sufficiency figure appears in the outcome without a measurement behind it | ⛔ **VOID that figure.** Nine for nine optimistic across both seats |

---

## 7. Method — LOCKED

- **Seeds** `np_seed = torch_seed = 42`. **CPU.** `model.eval()` then `set_locked_dropout(True)`, `reset_locked_dropout()` once per pass, **`D = 4`**.
- **Months:** matched out-of-sample, the protocol `p2_gate_calibration.py` established — `BASE_ORIGIN = 456`, `N_ORIGINS = 13`, `LEADIN_START = 421`. ⚠️ **Per model, verify the artifact's training partition actually ends before 457**, and VOID any model where it does not.
- **Target:** `sb` (`by_sb_best` against `lr_sb_best`), as Phase 1. ⚠️ `ns`/`os` are a separate question and **not** in this phase.
- ⛔ **The estimator is BIN-FREE maximum likelihood** — `logit P(y=1) = a + b·logit(gate)` over every cell, which removes the bin-threshold free parameter entirely. Binned fits at the locked P2 edges are reported **alongside** for continuity, never instead.
- **Every artifact verified** by `sha256` against its `.pt.sha256` and its sidecar's `output_distribution` before use. ⚠️ `vehicle.py` hardcodes `"nb"` and **VOIDs on anything else** — the override is explicit and asserted against the committed file's hash, never silenced (`OUTCOME.md §9` issue 3).
- **Signed error reported alongside absolute, always.** **Common bin support across any compared pair, always.** Positive fraction pooled, never a median over cells.
- ⛔ **Record, per model:** `n`, positives, `sd(b)` bootstrapped, bin-free `a` and `b`, binned `b` under all three weightings, the composition mode and `gate_threshold`, and the wall time.
- ⛔ **No value is written into a stored artifact under a name that is not true of it under the weakest reading a later reader could take** (`OUTCOME.md §9` issue 10: `pos_weight = 2.0` is in copper's `.npz` and copper ran at 1.0). **A field whose value depends on a branch must encode the branch or be split.**

---

## 8. Decision rules

1. **Step 0 (§2) first.** Report gate-posterior availability per model before fitting anything.
2. **Model 1 end to end, then STOP and report wall time** (F7).
3. **Per model: F1 → F2 → F3 → F4 before `b` is read.** A model failing any is VOID and does not enter a reading.
4. **Read §5's table.** It is exhaustive; if no row matches, that is itself a finding and the table was wrong.
5. **Phase 3 proceeds only on readings 1 or 2**, and only for `soft_gate` models, with §3.4's pre-committed selection rule.
6. ⛔ **No GPU, no re-run, no training arm, and no change to any model is authorised by this plan.** Phase 4 (F11, the firing-rate check) **needs a re-run and is Simon's** (`RND_PLAN.md` §5).
7. ⛔ **Nothing is presented as a product improvement.** `RND_PLAN.md` §6 — a gate correction lowers the delivered totals on a product already ~10× low — and §7 — the eligibility criteria are satisfiable by the artefact this programme found.

---

## 9. Out of scope

- Any remedy **application**. This phase measures; Phase 3 selects; nothing ships.
- The body (#530) — Simon's sequencing. ⚠️ But see `RND_PLAN.md` §6 for the coupling.
- `h ≥ 2` — the gate posterior is not retained (#558). **Everything here is h=1 and must not be generalised to the 36 delivered horizons.**
- The root cause (train/deploy geometry) — training-side, GPU, never scoped.
- H2, H3, H4 — live, untested, and **not discriminated by this phase** (§3.2).
- ⛔ Tuning the recalibration distribution — **banned** (`RESULTS.md` §6).
- The 13 issues in `OUTCOME.md §9` and the 3 proposed in `RND_PLAN.md` §7 — **Simon's repos, Simon's call.**

---

## 10. Definition of done, and the stop rules

**Done** when all eight models have either a recorded `(a, b, sd, n, positives, mode, τ, wall time)` row or a recorded VOID with its reason; every falsifier F1–F8 has been evaluated; §5's reading is stated; and an outcome document exists that **does not claim a cause** (§3.2).

**Stop rules.**
- ⛔ **No new questions mid-phase.** A candidate not in this plan is a note for Phase 3.
- ⛔ **A null is INCONCLUSIVE, not negative** (C-307) — reading 4 is a real finding, reading 6 is not.
- ⛔ **No scalar leaves this phase without its parameter sweep** (§3.5).
- ⛔ **No cost or power figure without a measurement** (F8).
- ⛔ **The executing seat attacks this plan before running it, and does not decide which of its own objections survive.** Neither does this seat.

---

## 11. Glossary

Every term checked against `reports/GLOSSARY.md`. **§5b** locks `the calibration level` (`a`), `the calibration slope` (`b`), `under-dispersed`, `over-dispersed` and `temperature scaling`, with the banned aliases — including that ⛔ **`timid` (§5) is about the BODY and must not be used for the gate**, and that *"the gate is miscalibrated"* names neither axis and is drift. `soft_gate`, `threshold_gate`, `self_zeroed`, `AP`, `act_ratio`, `MCR` and `calibrated` are unchanged. ⚠️ **`act_ratio` and `MCR` are BIAS metrics, not calibration metrics** — `OUTCOME.md §3.5`.

---

# AMENDMENT 1 — §2's premise is false, the reading table is not mutually exclusive, and the horizon limitation lifts. 2026-10-18, BEFORE any Phase-2 measurement.

**Source:** six attacks from the `views-models` seat. **All six upheld. Two of them invert things this plan asserted, one is a contribution, and one of my own risk assessments was backwards.**

---

## A1 — ⛔ UPHELD. The gate posterior IS on disk, for all eight, at all 36 horizons.

**Verified independently here:**

```
purple_alien     78 gate files      heavy_freighter   78
pink_pirate      39                 bright_starship   39
blue_stranger    39                 violet_visitor   117
bold_comet       78                 blazing_meteor    39
                 (39 = 13 origins x 3 targets)
```

Shape confirmed on `by_sb_best/y_pred.npy`: **`(471960, 16)` float32 = 13,110 cells × 36 horizons**, unsaturated, with `identifiers.npz` carrying `time`/`unit`.

⛔ **So §2's premise — "may mean eight local forward passes" — is FALSE. Phase 2 is array reads.** My Step 0 pre-registered a question whose answer was already on disk, and assumed the expensive branch.

⚠️ **And the other seat's ~1.5 h estimate was for sweeps, so it is also wrong — too HIGH.** ⛔ **The tenth unmeasured figure across both seats, and the first optimistic in the cheap direction.** F8 applies in both directions and both of us earned it.

### ⛔ And it lifts the horizon limitation this plan inherited

#558 / C-161 records *"the gate posterior does not come home."* **True of `pod_run_model.sh`'s calibration archive, which globs `'*/lr_*'` only. NOT true of these `predictions_validation_*` runs, which retain `by_*` at all 36 horizons.**

**So the horizon question Phase 1 declared unmeasurable is measurable for these eight.** ⛔ **A limitation carried forward verbatim from Phase 1 was stale**, and neither seat checked it before writing it into two documents.

**Bounded scope addition, pre-registered rather than allowed to balloon:** measure `b` at **h=1** (method-comparable to Phase 1) **and h=36** (the delivered horizon). **Not all 36** — the whole-curve question is a separate phase. **Pre-registered prediction: `b` at h=36 ≥ `b` at h=1 on every model**, since the rollout injects noise at every step. ⛔ **This is a genuine blind test** — no horizon-resolved gate calibration exists anywhere.

## A2 — ⛔ UPHELD AND SHARPENED. All eight are AFRICA. None is on Phase 1's grid.

They flagged `violet_visitor` as 13,110 cells against Phase 1's 64,818. **Measured here: all eight are 13,110 — `africa_me_legacy`. Every one.**

That cuts two ways and the plan must say both:

- ✅ **Their confound largely DISSOLVES for the eight-model comparison** — it is **within-grid**, same domain, same base rate, so counting models with `b > threshold` is legitimate.
- ⛔ **But the comparison to Phase 1 is CROSS-GRID for all eight.** `amber`/`copper` are global land (64,818); every served model is africa (13,110). **So `b = 1.1875` is not a comparable reference for any served model**, and A1's measured lesson applies exactly: **restricting the same model's data to a different support moved `b` by up to 0.099, against a 0.10 decision threshold.**

⛔ **Consequence for the generalisation question:** "does the defect generalise to the served models" is answerable **within africa**. ⛔ **Attributing any DIFFERENCE between the served models and amber/copper to anything other than the grid is NOT available from this data**, and no reading may do so.

**F12′ extended:** common support **across arms, across populations, and across MODELS** — and ⛔ **a model's grid and cell count are recorded before any `b` is fitted**, with the Phase-1 reference marked cross-grid wherever it appears.

## A3 — ⛔ UPHELD. The table was exhaustive but NOT MUTUALLY EXCLUSIVE.

They are right and the defect is the mirror of the one I fixed:

- **Reading 2** (`b > 1.05` on 1–4 of 8 → *treat only those*) and **reading 6** (*"mixed, no majority and no clean split"* → ⛔ *no treatment*) **overlap and prescribe opposite actions.** 1–4 of 8 **is not a majority**, and *"clean split"* is undefined — so reading 6 is **not a condition at all**, it is a catch-all that reverses reading 2.
- **Readings 1 and 3 co-fire** (five at 1.2, one at 0.9) and then disagree on whether the defect is "platform-wide" or "not directional", **with no precedence given.**

⛔ **I fixed non-exhaustiveness in Phase 1 and introduced non-exclusivity in Phase 2. A table that is exhaustive but overlapping has the identical defect: the outcome gets chosen after the fact.**

### Readings, REPLACED — disjoint, ordered, with precedence

Let `A` = the count of **`soft_gate`** models with `b − 1 > max(0.05, 2·sd_model)`, and `V` = models VOID under F1–F4. **Evaluate in order; the first matching row is the reading.**

| # | condition | consequence |
|---|---|---|
| **R-0** | `V ≥ 3` | ⛔ **VOID the phase.** Too little survives to read |
| **R-1** | any model with `1 − b > max(0.05, 2·sd_model)` | ⛔ **OVER-dispersion exists. Takes PRECEDENCE over all below.** A new finding; R1 would make that model worse; it leaves the treatment set and gets its own write-up; `OUTCOME.md §2` is qualified |
| **R-2** | `A ≥ 4` of the 7 `soft_gate` | **majority — the defect is platform-wide within africa.** Phase 3 proceeds for those models |
| **R-3** | `1 ≤ A ≤ 3` | **minority — model-specific.** Treat only those. ⚠️ Report *why they differ* as an open question, never as a cause |
| **R-4** | `A = 0` | ⛔ **the defect does NOT generalise beyond the two kestrels.** Phase 3 is cancelled; the programme's scope collapses to amber/copper |

**Disjoint by construction** (`A` partitions at 0, 1–3, ≥4), **ordered**, and R-1 is explicitly a precedence rule rather than a competing row. ⛔ **Reading 6 is DELETED** — "inconclusive" is now `R-0` with a numeric condition, not a catch-all.

## A4 — ⛔ UPHELD. The fixed band contradicted Phase 1's own lesson.

Phase 1 ended with `F1⁗` as an **equivalence test against measured `sd`**, precisely because a movement test would have named H1 on `b = 1.06`. **§5 then used an absolute `0.95 ≤ b ≤ 1.05` for all eight.**

`sd(b)` was **0.0107** on 842,634 cells. On **13,110 cells × 13 origins** it scales to roughly **0.024**, so `1.05` is only ~2 sd — and ⛔ **for any model with fewer usable cells the band sits inside the noise, making `b > 1.05` unfalsifiable while reading as a finding.**

⛔ **Band replaced everywhere by `max(0.05, 2·sd_model)` with `sd` bootstrapped PER MODEL**, already applied in the table above. **F2's floor stands** at `sd ≥ 0.027` → that model is UNMEASURED.

## A5 — ⛔ UPHELD. My risk assessment was BACKWARDS. One addition restores the real risk.

I called §3.4's diagnostic the riskiest item. **It is close to definitional, and their argument is correct:** R3's only advantage over R1 is that it fits an intercept; **`a` IS that intercept**; so *"large `|a|` predicts R3 wins"* is nearly *"the two-parameter family beats its restriction when the parameter it adds is large"* — the **definition** of when a nested family wins, not a claim about the world. In the bias limit it is an identity.

**And their second point is decisive:** amber `a = +0.16`, copper `a = +1.30`, so ⛔ **any threshold in (0.16, 1.30) separates the two training cases perfectly — an eightfold interval. The 0.5 carries no information from n=2.** By §3.5's own sweep rule the admissible family is wide and all its members agree on the training cases, so ⛔ **the rule can be FALSIFIED by a miss but barely CONFIRMED by a hit.** That asymmetry is now stated, and the scoring will respect it.

**Their refinement adopted:** the residual content is whether the bias reduction beats the **variance cost** of the extra parameter on held-out folds — **a sample-size question, not an `|a|` question** — so the threshold scales with per-fold `n` rather than sitting at a constant.

### ⛔ My addition: the genuinely risky and UNMEASURED question is whether `a` is STABLE

The diagnostic is measured on the **raw** gate's fit; the selection happens on **held-out folds**. ⛔ **If `a` is itself unstable across folds, it cannot be a selector at all — even in the bias limit where the rule is definitional.**

**Tier 2 measured `T`'s fold stability (0.8%) and never measured `a`'s.** So:

> ⛔ **Pre-registered prerequisite: report `a`'s spread across the 13 origin folds per model, BEFORE using it to select a remedy. If `a`'s fold spread exceeds its own distance from the threshold, the diagnostic is inadmissible for that model and the selection falls back to held-out `|gate−obs|` alone.**

That is the part of the rule that is **not** definitional, it is measurable, and it has never been measured.

## A6 — ✅ CONTRIBUTION, verified. F11 is open for the eight, blocked only for the kestrels.

They answered the question I flagged and could not cost: ⛔ **`draws/` does NOT retain the uncomposed magnitude cube**, and there is a README saying so — `amber_kestrel/.../draws/README_GATE_POSTERIOR_MISSING.txt`: the `lr_*` archive is complete, the `by_*` archive is **9% truncated and marked DO_NOT_USE** because `pod_run_model.sh` globs `'*/lr_*'` only. **So F11 on amber does need a re-run, as `RND_PLAN.md` §5 said.**

✅ **But for the eight served models the gate posterior IS intact — and their factorisation argument is correct.** Under `soft_gate` the composed sample is `body_draw × Bernoulli(gate)` with the two draws independent by construction, so:

```
P(composed > 0) = P(body > 0) · gate        =>   P(body > 0) = P(composed > 0) / gate
under a corrected gate g':  P'(composed > 0) = P(composed > 0) · g' / gate
```

⛔ **So `act_ratio` under a corrected gate is computable ANALYTICALLY, per cell, at every horizon, with no re-run.** ⚠️ With the caveat that `P(composed > 0)` is estimated from 16 draws and so is quantised at 1/16 per cell — **fine pooled, not per-cell**, and the estimator must be pooled accordingly.

⛔ **`RND_PLAN.md` §5 is therefore wrong as written: Phase 4 is GPU-blocked for the kestrels and OPEN for the models that actually ship.** Corrected there.

---

## What this amendment changes

| | |
|---|---|
| §2 | premise false — **array reads, not sweeps**; Step 0 becomes a recorded fact, not an open question; both cost estimates void |
| §2 (new) | ⛔ **h=1 AND h=36**, with a blind prediction that `b` worsens with horizon. The #558 limitation was stale |
| §5 | ⛔ readings **replaced** with a disjoint, ordered table; reading 6 deleted; R-1 given precedence |
| §5, §4 | the band is `max(0.05, 2·sd_model)`, bootstrapped per model |
| §3.4 | the diagnostic is **near-definitional**, the threshold is undetermined by n=2, falsifiable-not-confirmable — **plus a new prerequisite that `a`'s fold stability be measured first** |
| F12′ | extended to **across models**; grid and cell count recorded before fitting; Phase 1's reference marked **cross-grid** |
| `RND_PLAN.md` §5 | F11 is **open for the eight**, blocked for the kestrels |

⛔ **Four of the six attacks landed on things this plan asserted rather than measured** — a premise, a stale limitation, a table's logic and a band inherited from the wrong phase. **Phase 2 still needs Simon's word and nothing has run.**

---

# AMENDMENT 2 — A1 and A6 WITHDRAWN. The gate posteriors are for the wrong runs, and the served weights cannot be identified from this machine. 2026-10-18.

**Source:** the `views-models` seat withdrawing both of its own Amendment-1 contributions after reading the run type and region it had not read the first time. ⛔ **Both withdrawals are correct. This seat's A2 extension inherits the defect and is withdrawn with them.**

---

## 1 — ⛔ WHAT WAS WRONG: the files answer a different question

The on-disk `by_*` gate posteriors are **`predictions_validation_*` runs dated 2026-09-07 to 2026-09-17**, shape `(471960, 16)` = **13,110 africa cells × 36 horizons**. FAO is served from **`rusty_bucket_forecasting_20261001_012905`** — a **global forecast**, 64,742 cells.

**Different region, different run type, different weights, two weeks earlier, pre-switch configs.** ⛔ **The filename says `predictions_validation_` and the date is in it, and neither seat read either.**

**Verified here, independently:**

- The region switch is on the record in this repo: **`2026-10-12_underfire_diagnosis_prereg.md:207` — "Sep 19 ⛔ `REGION` switched `africa_me_legacy` → `land` (runbook #499)"**, with the note that the newest local artifact *"is two days older than the region switch, so it was trained on `africa_me_legacy` (13,110 cells) — not the global `land` model (64,818 cells) that produced the delivery."*
- ⛔ **Zero post-switch artifacts exist for any of the eight.** 22 `.pt` files across the eight models, **all dated before 2026-09-19**: purple_alien 0/3, pink_pirate 0/2, blue_stranger 0/2, bold_comet 0/3, heavy_freighter 0/3, bright_starship 0/3, violet_visitor 0/4, blazing_meteor 0/2.
- The `forecasting_20261001` output directory is **not on this machine**.

## 2 — What is withdrawn

| | status |
|---|---|
| **A1** *"Phase 2 is array reads"* | ⛔ **WITHDRAWN.** True only if we accept measuring September africa **validation** runs in place of the served October **global forecast**. Their 1.5 h sweep estimate was **not too high — it was the estimate for the right question**, and they withdrew it in favour of files that answer a different one |
| **A6** *"F11 is open for the eight"* | ⛔ **WITHDRAWN.** The factorisation `P(composed>0) = P(body>0)·gate` **is still correct**, but it needs the gate for the **served** run, which is not retained (`pod_run_model.sh` globs `'*/lr_*'`, #558). **F11 is blocked for the eight too.** `RND_PLAN.md` §5 reverts to blocked |
| **this seat's A2 extension** | ⛔ **WITHDRAWN.** *"All eight are africa"* is established **of these September validation runs** and **not** of the served October run. The dissolution of the cross-grid confound rested on it, so that dissolution is withdrawn as well |
| the h=1-and-h=36 addition | ⛔ **WITHDRAWN as scoped.** 36 horizons exist **for the September africa validation runs only.** The #558 limitation is **not** stale for the served run |

⛔ **Eleventh instance of provenance-mismatch — a value used in a role other than the one that produced it — and the worst-placed: both seats celebrated an unblock in the same round, built on files whose run type and region neither had read.** Amendment 6's own control (*"the `file:line` that SETS it, and the question its setter was answering"*) applies to a **filename**, and neither of us applied it.

### ⛔ And the hazard was already guarded in this programme's own tooling

`reports/2026-10-12_underfire_diagnosis/vehicle.py:148-151` **denies four known `africa_me_legacy` artifacts by hash**, with the reason stated:

> *"The config check cannot catch this — the sidecar records no region and the net is **fully convolutional** — so it is denied by hash."*

and `:187-191`:

> *"…the region is recorded as what `pod_run_model.sh:279-282` **refuses**, NOT a read-back of the grid the weights saw. **Nothing in a `.pt` or its sidecar records the region**, and the net is fully convolutional, so **this remains weaker than a measurement.**"*

⛔ **So this repository already knew an africa artifact could be used where a global one was needed, built a hash-denial guard against it, and documented that region is NOT recoverable from an artifact — while both seats read eight directories without checking either.** Fourth instance of prior art in our own tooling, and the first where the guard was written by this very programme.

## 3 — ⛔ WHAT IS ESTABLISHED, AND WHAT IS NOT

The other seat raises two possibilities for the served run: **(a)** it ran the September africa-trained weights on the global grid, or **(b)** new weights were trained and never came home. **Neither seat can evidence either.**

⛔ **This seat will not state (a).** It is the alarming reading, it is not excluded, and it is not established — and a programme that has withdrawn sixteen claims between two seats does not get to publish an inference of that size on an absence of files.

**What IS established, and it is enough to act on:**

> ⛔ **The weights behind the forecast FAO is being served cannot be identified from this machine.** No artifact for any of the eight postdates the region switch, and the serving run's own output directory is absent.

**That holds under BOTH (a) and (b), and it is a finding in its own right:**

- **Under (b)** it is a **reproducibility and provenance defect** — the served forecast is produced by weights that exist nowhere locally, independent of any calibration question.
- **Under (a)** it would additionally be **H4 in a far more extreme form than anything measured** — trained on 13,110 africa cells, deployed on 64,742 global cells including continents absent from training. ⛔ **Not asserted.**

⚠️ **And it decides whether Phase 2 is executable at all**, which is why it is now Step 0.

## 4 — Step 0, REWRITTEN. A provenance gate, not a cost question.

**Before any `b` is fitted, and in this order:**

1. ⛔ **Identify which artifact produced `rusty_bucket_forecasting_20261001_012905`, per member, by `sha256` against the run's own provenance record — NOT by date, NOT by filename.**
2. **Establish whether that artifact is on this machine.** ⛔ **If it is not, Phase 2 cannot measure the served models, and must say so rather than substituting September africa runs.**
3. **Read back what region each member's weights were trained on.** ⚠️ Per `vehicle.py:187-191` this may be **unanswerable** — the sidecar records no region and the net is fully convolutional. **If so, state it as unanswerable; do not infer it from a date.**
4. **Only then** the array-reads-or-sweeps question, and the cost.

### ⛔ F-NEW — pre-registered refusal

> **If the served weights cannot be identified, Phase 2 may measure the September africa validation runs ONLY if every statement about them is scoped to "these models, in September, on africa."** ⛔ **F-NEW fires on any sentence that crosses that boundary into a claim about what FAO receives** — and a fired F-NEW voids the sentence, not the measurement.

**This is the control that was missing when both seats celebrated an unblock.** The measurement would have been fine; the sentence about it would not.

## 5 — What survives of Amendment 1

| | |
|---|---|
| ✅ **A2's core** (the cross-grid confound) | **stands and is now stronger** — if the served run is global and the measurable runs are africa, the mismatch is the whole problem rather than a nuisance |
| ✅ **A3** disjoint ordered reading table | stands |
| ✅ **A4** `max(0.05, 2·sd_model)` band | stands |
| ✅ **A5** the near-definitional diagnostic, its undetermined threshold, and the falsifiable-not-confirmable asymmetry | stands |
| ✅ **the `a`-fold-stability prerequisite** | stands — `T`'s spread was measured at 0.8% and `a`'s never was, and an unstable `a` cannot select even where the rule is definitional |
| ✅ **the 1/16 quantisation caveat** on the factorisation | stands, and applies whenever the factorisation is used on a run whose gate IS retained |

⛔ **Phase 2 is a weaker proposition than when Simon gated it, and the plan says so rather than preserving its own scope.** The next step is not a measurement — **it is a provenance question, and it may return "not answerable from here."**
