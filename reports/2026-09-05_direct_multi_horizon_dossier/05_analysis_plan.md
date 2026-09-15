# 05 — Pre-analysis plan (S3, #328)

**Written 2026-09-06, BEFORE any direct-head arm is trained.** Nothing in this file may be edited
after the first training run; amendments are appended, dated, and justified.

## Hypothesis

The 36-month degradation is caused by **what the model is fed after step 1**. A head that never
feeds itself will recover a substantial share of the gap between the current rollout and the
real-data ceiling.

## The one variable

The forecast head: recursive (feeds its own prediction, state evolves) → direct (every horizon
decoded from the origin state, horizon as an input covariate, nothing fed back).

**Everything else is frozen at the incumbent's values, including the loss.** The panel's F5
(per-intensity weighting) is **excluded from the headline arm** — a build that changes four things at
once cannot attribute its own result, and the loss change is separately rejected as improper.

## The measured ceiling and control — 4 seeds, already on disk, zero GPU

`AP@h18`, `sb`, 13 origins, free-running:

| seed | ceiling (fed real data) | clamped control (production) | gap |
|---|---|---|---|
| 42 | 0.4974 | 0.3622 | 0.1353 |
| 43 | 0.5014 | 0.3709 | 0.1305 |
| 44 | 0.4910 | 0.3518 | 0.1392 |
| 45 | 0.4932 | 0.3644 | 0.1288 |
| **mean** | **0.4958** | **0.3623** | **0.1334** (sd **0.0047**) |

**The gap is precisely known.** Its seed sd (0.0047) is a third of the sd of AP itself (0.0134),
because ceiling and control move together.

## ⛔ A rule this programme has been using is FALSIFIED by its own control

**M45 has been applied as "AP loss scales with how much the model FIRES", and that reading is
wrong.** The ceiling arm fires **×1.92–2.46 more** than the clamped control **on all four seeds** and
scores **+0.13 higher**:

| seed | oracle act_ratio | control act_ratio | ratio | ΔAP |
|---|---|---|---|---|
| 42 | 0.339 | 0.168 | ×2.02 | **+0.1353** |
| 43 | 0.373 | 0.174 | ×2.14 | **+0.1305** |
| 44 | 0.396 | 0.207 | ×1.92 | **+0.1392** |
| 45 | 0.319 | 0.130 | ×2.46 | **+0.1288** |

**Firing more is not the failure. Firing more in the WRONG PLACES is.** Six interventions raised
firing *without* improving placement and lost AP; the oracle raises firing *and* places it correctly
and gains. The programme has been treating a correlate as a cause.

**Consequence, and this is why pre-registration matters:** an earlier draft of `02_design`
pre-registered *"if it works, `act_ratio` should NOT rise."* **That criterion would have rejected the
right answer.** It is withdrawn here, before any run, and replaced below.

## Pre-registered decision rule

Primary: **ΔAP@h18 vs the clamped control**, `sb`, 13 origins, free-running, paired on origins.

| outcome | rule | disposition |
|---|---|---|
| **STRONG** | ΔAP ≥ **+0.067** (≥50% of the gap) on ≥2 of 3 seeds | promote; propose an ADR; plan the fleet path |
| **SUCCESS** | ΔAP ≥ **+0.033** (≥25% of the gap, ≈ the cell clamp's own effect) on ≥2 of 3 seeds | build out; 4-seed confirmation |
| **NULL** | \|ΔAP\| < 0.033 | the recursion was **not** the binding constraint. Record it. **Do not** propose a successor without new evidence. |
| **HARM** | ΔAP ≤ −0.033 | close the direction |
| **VOID** | any gate below fails | not a result; re-run or abandon, **never recorded as a null** |

**Why 0.033 is the floor:** MDE at 2 seeds/arm is 0.0333 (α=0.05 one-sided, 80% power, σ=0.0134);
at 3 seeds 0.0272. **3 seeds per arm** is pre-registered so the threshold sits above the MDE.

## Secondary criteria — all pre-committed

1. **Placement, not firing.** `act_ratio` at h18 must move **toward the ceiling's ~0.36**, not away.
   A rise is **expected and permitted**. What is *not* permitted is a rise with **no** AP gain —
   that is the six-failure signature and triggers the HARM branch regardless of the point estimate.
2. **Non-inferiority floor at h1**: ΔAP@h1 ≥ **−0.0134** (one seed sd). h1 involves no feedback, so a
   loss there is the architecture damaging what already works. **Breach = failure, not a trade-off
   argued after the numbers land.**
3. **Horizon shape**: the gain must be ≥ 0 at every horizon and **non-decreasing in h** — the gap
   grows with horizon (0.133 at h18, 0.184 at h36), so a fix for the stated mechanism must too. A
   flat-in-horizon gain falsifies the mechanism even if h18 is positive.
4. **Fair CRPS co-primary**, with the reliability–resolution decomposition, equal `S` in both arms.
5. `crps_events` is **descriptive only** — conditioning on `y>0` makes it improper as a comparator.

## VOID gates — checked before any score is read

- weight hashes differ between arms; `arm_identity_check` passes
- `floor_gate` FG-A and FG-C on the **control**, before the treatment arm runs
- potency on the arm's own config **and at a trained checkpoint** (C-324/C-325)
- **the ruler proven to score a direct cube identically to a rollout cube** (S5) — including the
  **C-218** question: a direct arm has no `rollout_feedback`, and `partition_audit` refuses to score
  without it. Settle before GPU, not after.
- `freeze_multitask_balancer: True` pinned (C-312)
- no BatchNorm writes from any added forward (C-328, four instances)

## What this design CANNOT answer

- Whether a direct head preserves **joint** behaviour across horizons. It does not, and neither does
  the encoder-forecaster variant — both are marginal. **No score in this repo can see it**, and the
  library holds no multivariate proper score. Recorded as a known, unmeasured cost.
- Whether the result transfers to the other roster members, to `ns`/`os`, or to global resolution.
- Anything about magnitude — `size_ratio` is 0 throughout and that is a separate epic.

## Skepticism ledger

- The ceiling is an **oracle** — it is fed truth. No model can reach it; it bounds the prize, it is
  not a target.
- The gap may be *irreducible* — knowing the future input may be worth 0.13 no matter how the head is
  built. **This experiment cannot distinguish "the recursion is the problem" from "not knowing the
  future is the problem."** That is the single largest threat to the inference and it is stated here,
  in advance, rather than discovered in the disposition.
- 4 seeds, one vehicle, one target, one region.
