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
