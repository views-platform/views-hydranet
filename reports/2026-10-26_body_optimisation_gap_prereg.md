# Pre-registration — is the body's deficit an OPTIMISATION gap or a REPRESENTATION gap?

**Date:** 2026-10-26 · **Branch:** `prereg/body-optimisation-gap` (⚠️ cut off
`prereg/control-vehicle`, **not** off `development`, because this plan depends on
`2026-10-26_structure_penalty_arm_outcome.md` §4c which lives there and is unmerged. The standing
rule is "cut off `development`"; the deviation is named rather than hidden.)
**Status:** PRE-REGISTERED, not run. ⛔ Nothing below has been measured.
**Cost:** CPU only. No pod, no GPU, no data pull beyond one freshness check, no release, no money.

---

## 1. Why this, and why now

Seven interventions have failed, five of them on the same lever — how much the model fires or
spreads. **Every one of them assumed we are optimising the wrong thing. None has tested whether we
are reaching the thing we already optimise.**

That second possibility is already measured, once, and was explicitly parked.
`reports/2026-10-12_underfire_next_step.md:219-235`:

> | the model emits | **θ = 0.1544** |
> | **NLL-optimal** at that μ | **θ = 0.2327** (NLL **4.1377** vs the model's **4.2121**) |
>
> **The model sits BELOW its own training loss's optimum.**

and `reports/2026-10-12_underfire_diagnosis_prereg.md:729-733`:

> The gate sits below its own loss's optimum — the same shape M6 found on the body. … **Two
> independent halves of the model, each falling short of its own objective in the same direction.**
> … *Named, not chased.*

⭐ This pre-registration chases it. It is a **body** plan — the standing note that 3 of the last 7
pre-registrations were gate-named and **0** body-named is the reason it is framed on the body.

⭐ It also has a second parent. `2026-10-26_structure_penalty_arm_outcome.md` **§4c** measured, on a
matched window with byte-identical truth, that the arm's body **preserved ranks and destroyed
magnitudes** (`body vs gate` fell 3.5× on Pearson, rose on Spearman). That reading makes a
prediction about this experiment, pre-registered below as **P-4**, and this is the first chance to
score it.

---

## 2. The hypothesis — ONE variable, and it is not a knob

**H.** The body's deficit against **its own training objective** is dominated by a
**rank-preserving mapping error** — the ordering of cells is substantially right and the map from
that ordering to a magnitude is wrong — rather than by **placement**.

⛔ Nothing is retrained. Nothing is re-emitted. No loss term is changed. The "intervention" is a
**post-hoc fit applied to fields already on disk**, and the quantity that varies is **how
expressive that fit is allowed to be**.

⇒ If H holds, the deficit is something gradient descent was free to remove and did not: an
**optimisation** limit. If H fails, the deficit is in which cells carry the mass: a
**representation** limit, and a whole class of interventions — every loss re-weighting, every
calibration, every composition change — is dead for free.

---

## 3. Method — a ladder of nested post-hoc fits, all minimising the SAME objective

Each rung minimises the **body NLL of the family the cube records** (`nb`), by the repo's own
`views_hydranet/distributions/nb_core.py:62 weighted_nll_mean` reduction, on the **same**
population, starting from the model's own emitted fields. Each rung strictly contains the one above
it, so its NLL can only fall — see falsifier **F2**.

| rung | what is free to move | the question it answers |
|---|---|---|
| **R0** | nothing — the model as trained | the reference NLL |
| **R1** | one global `θ` | replicates the 2026-10-12 measurement ⇒ a **replication gate**, not a result |
| **R2** | a global affine on `log μ` (scale, offset) + global `θ` | *did SGD fail to set two numbers?* |
| **R3** | R2 per target (3 × 3 parameters) | is the failure shared across targets or per-target? |
| **R4** | ⭐ a **monotone (isotonic)** recalibration of `μ` against truth + global `θ` | **the rank-preserving ceiling** — everything a fix that never reorders a cell could recover |
| **R5** | `μ_i = y_i` per cell (oracle) + global `θ` | the **NLL floor** ⇒ the denominator |

**READOUT.** For each rung, the share of the recoverable gap closed:

> **S_k = (NLL_R0 − NLL_Rk) / (NLL_R0 − NLL_R5)**

⭐ **R4 is the decisive rung.** R4 recovers exactly what a rank-preserving correction can recover
and nothing else. What R4 leaves on the table is, by construction, **placement**.

### 3.1 Population, transform, estimator — named on the line, per C-351

- **POPULATION:** `d4_result*.npz`, 64,818 land cells × months **421–456** (the **in-sample**
  window pinned by `body_scoreboard.py:35`), 4 MC-dropout draws. ⛔ Months **457–469** are reported
  **beside** the headline as the out-of-sample view and are never averaged into it.
- **ARMS:** control `d4_result_hf0125.npz` (sha `0ca3f31f…`) and arm `d4_result_ARM_hf0126.npz`
  (sha `aab48beb…`). Both `family = nb`. Both carry `mu`, `theta`, `gate`.
- **TRANSFORM:** none on the truth — raw counts, the space the NLL is defined in. `μ` is used as
  emitted via `vehicle.py:65 emitted_magnitude`.
- **ESTIMATOR:** `weighted_nll_mean` with `weight=None` ⇒ the mean over **all** cells, which is the
  `nb` family's own training reduction. ⛔ Not the positive-cell denominator; that belongs to
  `truncated_nb` and neither arm is one.
- **TRUTH:** the cached land pull `pull_land_all3_121_504.parquet` via `vehicle.py:271 pull`,
  ⛔ **gated on a freshness check — see F4.**

### 3.2 ⛔ Pre-flight gates, all of which must pass before any rung is read

1. **F4's freshness check passes** — the cached truth pull matches a fresh pull.
2. **R1 reproduces θ\* ≈ 0.2327** on this population (P-1).
3. **Monotonicity holds** across R0 → R5 (F2).
4. The oracle rung R5 is the minimum of all six (F3).
5. Every rung is run on **both** arms through **one** script in **one** session — ⛔ never one arm's
   number taken from a prior write-up.

---

## 4. Pre-registered predictions — committed before anything is run

| # | prediction | what it is risking |
|---|---|---|
| **P-1** | R1's θ\* lands within **±0.02** of **0.2327** on the control | a **replication gate**. If it misses, this instrument disagrees with the measurement it generalises and everything below is VOID, not interesting. |
| **P-2** | ⭐ **S_4 ≥ 0.50** on the control — a monotone recalibration closes **at least half** the recoverable NLL gap | the headline. This is H. |
| **P-3** | **S_2 ≤ 0.25** — two global numbers do **not** close most of it | that the gap is distributed, not one missed scalar. ⛔ If S_2 is large that is a **different and more damning** finding and I commit now to reporting it as such rather than folding it into P-2. |
| **P-4** | ⭐ **S_4(arm) ≥ S_4(control)** | §4c's "ranks preserved, magnitudes destroyed" reading, scored out of sample on a different instrument. If P-4 fails, **today's reading of the matched window is wrong** and §4c needs amending. |
| **P-5** | the ordering **S_1 ≤ S_2 ≤ S_3 ≤ S_4** holds on both arms | the ladder is nested and the fitter works. Overlaps F2 deliberately: F2 is the void condition, P-5 is the prediction. |

⚠️ **Direction is committed, not just magnitude.** I expect the body's *ordering* to be
substantially right and its *mapping* wrong. The opposite outcome is fully possible and is what
P-2 risks.

---

## 5. Falsifiers — pre-committed, and VOID is not the same as NEGATIVE

| | condition | consequence |
|---|---|---|
| **F1** | R1 does not reproduce θ\* = 0.2327 ± 0.02 | ⛔ **VOID.** The instrument disagrees with the prior measurement. Fix the instrument; report nothing. |
| **F2** | any rung's NLL **rises** relative to a rung it contains | ⛔ **VOID.** The inner optimiser failed; this is a numerical defect, never a finding. |
| **F3** | R5 is not the minimum, or (NLL_R0 − NLL_R5) ≤ 0 | ⛔ **VOID.** The denominator is not a denominator. |
| **F4** | the cached truth pull disagrees with a fresh pull | ⛔ **VOID until re-pulled.** Standing rule: never validate on cached data. The check is cheap and runs first. |
| **F5** | **S_4 < 0.25 on BOTH arms** | ⭐ **H IS FALSIFIED.** The deficit is in placement. This is a *result*, not a void. |

---

## 6. Decision rules — written before the number exists

| S_4 (control) | reading | what happens next |
|---|---|---|
| **≥ 0.50** | the limit is **optimisation** | ⛔ **no new loss term.** The next arm is training-side. Which one is a separate question this plan does not pre-judge. |
| **< 0.25** | the limit is **representation** | kills the loss-reweighting / calibration / composition class in one afternoon. The quantile head (`views_hydranet/utils/quantile_head.py`) and the representation become the live axis. |
| **0.25 – 0.50** | ⚠️ **INCONCLUSIVE on one vehicle** | reported as inconclusive and **not rounded** to either side. C-307: a cheap screen's NO is recorded with its false-negative mode and a reopen trigger, or it closes nothing. |

---

## 7. ⚠️ LIMITS — binding, and stated before the result

1. ⛔ **The NLL here is a PROXY for the training objective, not that objective.** Training minimised
   a per-window, per-step, dropout-on, single-draw loss under a frozen balancer that weights the
   body **0.5**. This reads a 4-draw-averaged μ on a pinned month range. ⇒ **the comparison between
   arms and between rungs is the finding; the absolute NLL is not quotable as "the training loss".**
2. ⛔ **97.8% of land cells have a window truth of exactly 0** (`body_scoreboard.py:16`). This
   population is dominated by zeros by construction. Every rung sees the same population, so the
   *shares* are comparable; ⛔ no absolute NLL from it describes anything.
3. **n = 1 vehicle per arm, single seed.** Per C-307 a null is **INCONCLUSIVE**, not negative.
   Against ~0.019 seed sd on AP-type quantities — ⚠️ and **no seed sd for S_k exists**, which is
   limit 5.
4. **4 MC-dropout draws**, the standing cap. ⛔ Not to be raised.
5. ⚠️ **There is no noise floor for S_k.** The split-half device used for B-1 does not transfer —
   S_k is a ratio of two differences of NLLs and its sampling behaviour is unmeasured. ⇒ the 0.25 /
   0.50 thresholds are **pre-committed conventions, not significance boundaries**, and are named as
   such wherever the result is quoted.
6. ⛔ **This cannot say WHY the optimiser stopped short** — not the balancer, not the learning rate,
   not the trunk capacity, not BPTT. It can only say whether anything is left on the table that a
   rank-preserving map could take. Attributing the cause is a separate plan.

---

## 8. What this is not

Not a retrain. Not an arm. Not a new loss term. Not a new metric — every instrument it uses is
already committed (`nb_core.weighted_nll_mean`, `vehicle.emitted_magnitude`, `vehicle.pull`), which
is `2026-09-30_evaluation_learning_report.md` §8.1's standing preference: **wire an existing metric
to a decision before adding one.**

It also deliberately obeys the two gate rows added to that report today:
**§8 row 16** — the quantity being read is the body NLL, which **is** in the training objective, so
it is explicitly a **diagnostic and never a promotion criterion**; nothing is promoted on this.
**§8 row 17** — the five predictions span three sensitivity classes: **spread** (P-1, θ),
**level** (P-3, the affine), **rank/placement** (P-2, P-4, the isotonic rung) — which is the thing
the structure-penalty prereg failed to do.

---

## Provenance

- the parked measurement: `reports/2026-10-12_underfire_next_step.md:205-235`,
  `reports/2026-10-12_underfire_diagnosis_prereg.md:725-733`
- the §4c reading P-4 scores: `reports/2026-10-26_structure_penalty_arm_outcome.md` §4c,
  re-derived by `reports/2026-10-26_matched_window/matched_forensic_window.py`
- the cubes: `reports/2026-10-12_underfire_diagnosis/d4_result_hf0125.npz` (control, `0ca3f31f…`)
  and `d4_result_ARM_hf0126.npz` (arm, `aab48beb…`)
- the population pins and the zero-share caveat: `reports/2026-10-12_underfire_diagnosis/body_scoreboard.py:16,35`
