# 05 — Analysis plan (pre-registered)

# **LOCKED 2026-08-15**

> Locked **before any arm runs**. Nothing below may be changed afterwards. If something here turns out to be
> wrong, that is a logged finding in `07_experiment_log.md` and a *new* pre-registration — not an edit to this
> file.

---

## The question

**Does the rollout gate collapse because the recurrent state is corrupted by the model's own output?**

Free-running, gate AP goes **0.30 → 0.01** by h18. Under the oracle (`rollout_feedback='teacher_forced'`) it
holds **0.30 → 0.27** to h36.

`#262` read the oracle result as ruling the state out: *"NOT hidden-state / recurrent drift (overturns the
prior C-222-based bet)"*. **That inference does not follow.** The oracle varies the *input* while the state
evolves normally, so it shows the state is fine when it is never polluted. It cannot show what happens when
the state *is* polluted — which is the free-running condition. C-222 states exactly this confound.

## Hypothesis

**H1.** Holding the recurrent state at its last real-data value preserves gate skill across the horizon —
i.e. the corruption accumulates *in the state*, not only in the instantaneous input.

**H0.** The state path is inert for the gate, as it was for the C-113 bloom. Freezing changes nothing.

## The one variable

`freeze_recurrent` ∈ {`none`, `hidden`, `cell`, `all`}. Nothing else moves: same artifact, same data, same
seed, same `rollout_feedback='sample'`, no retraining.

---

## Method

### Why this is not already answered

`reports/results_freezeh_ablation.md` (2026-06-04) ran these four arms and concluded freezing was inert.
Its **endpoint was regression CRPS**, on a pre-ADR-070 artifact that exploded at ~1e17 in *every* arm.
Classification was never measured, and activation-aware metrics did not exist until 2026-08. The prior
result is about the bloom; this question is about the gate.

### Arms

The ConvLSTM packs its state into one tensor of 8 equal channel groups — `hs_1..hs_4` (short-term) then
`hl_1..hl_4` (long-term). Arm names match 2026-06 so the two runs are comparable.

| arm | short-term | long-term |
|---|---|---|
| `none` | evolves | evolves |
| `hidden` | **held** | evolves |
| `cell` | evolves | **held** |
| `all` | **held** | **held** |

The hold starts at `t > origin` — after history digestion **and** the seed step. The anchor is the state at
the end of the seed step: everything learned from real observations.

### Parameters — locked

| Parameter | Value |
|---|---|
| **Vehicle 1** | `truncated_smoke`, artifact `calibration_model_20260814_003058.pt` (the EXP-SS-2 artifact) |
| **Vehicle 2** | `violet_visitor`, artifact `calibration_model_20260812_191742.pt` |
| **Horizons** | 1, 6, 12, 18, 24, 30, 36 |
| **Target** | `sb` (headline); `ns`/`os` reported where available |
| **Feedback** | `rollout_feedback='sample'` — unchanged from the arm being explained |
| **Primary metric** | **gate AP** per horizon |
| **Secondary** | `act_ratio`, precision@k, and the composed forecast's `crps_all` split |

### Metrics

Reused unchanged from `reports/2026-07-29_v2_scoreboard_dossier/tools/`: `score_v2_horizons.py` and
`activation_metrics.py` (`activation_frequency`, `topk_occurrence`). **No new metric code**, so the numbers
are directly comparable to EXP-SS-1/2 and to the v2 board.

---

## Pre-registered predictions

| # | Prediction |
|---|---|
| **P1** | `none` reproduces EXP-SS-2's free-`sample` row: h1 AP 0.298 / act 1.41; h18 0.007 / 0.29; h36 0.008 / 0.27. |
| **P2** | At least one held arm keeps gate AP **materially above** `none` at h18 **and** h36. *(the maintainer's recollection, stated falsifiably)* |
| **P3** | No held arm restores magnitude calibration — `act_ratio` stays off. Occurrence and amount are separate ceilings. |

## Falsifiers (pre-committed)

| # | Fires if… | Consequence |
|---|---|---|
| **F1** | h=1 is **not identical across all four arms** | the hold is leaking into history digestion ⇒ the arms no longer share a history, **all results void**, stop |
| **F2** | `none` does not reproduce EXP-SS-2 | the harness differs from the probe it is compared against ⇒ stop and reconcile before reading anything |
| **F3** | all four arms track each other within seed noise at every horizon | **P2 refuted.** The state path is inert for the gate as it was for the bloom; C-222 settles negative and the state is crossed off the list. **A result, not a failed run.** |

**F1 is guarded in CI, not only at read time** — `tests/test_recurrent_state_freeze.py::
test_h1_is_byte_identical_across_every_mode`, checked against a deliberate sabotage.

---

## Decision rule — pre-committed

Evaluated at **`sb`, h18 and h36**, on `truncated_smoke` first:

```
STATE-IMPLICATED   iff  max(AP_held) - AP_none >= 0.05  at h18 AND h36
STATE-INERT        iff  |AP_held - AP_none| < 0.01      for every held arm at every horizon
INCONCLUSIVE       otherwise
```

The 0.05 threshold is FAO-02's superiority margin, reused rather than invented. A `STATE-IMPLICATED` verdict
must additionally survive vehicle 2 before it is reported as anything but indicative.

---

## Skepticism ledger

1. **`truncated_smoke` is 40 lessons / seed 42 / one origin set.** Indicative only, per #262's own caveat.
   No ranking claim on one seed. `violet_visitor` is the confirmation vehicle.
2. **A positive result does not mean "reinstate freezing."** ADR-027's retirement stands and this is not a
   config key. It would mean the *state* is a legitimate place to intervene, making a **soft** prior
   (decayed or confidence-weighted update while free-running) the next object of study — not a hard switch.
3. **Freezing creates a train/inference mismatch.** That was one of the two 2026-06 reasons for retirement
   and it remains true. It is acceptable in a diagnostic and would not be acceptable in a deployment.
4. **`none` is not free of the composition defect.** Even on `truncated_smoke` the fed-back field is only
   calibrated at h1; a held state does not fix what is fed back. The arms isolate the *state* channel, not
   the whole problem.
5. **Meta-pattern 8 (invalid knowledge from a bug).** Three sabotage checks were run against the guard tests
   before any arm, because this programme has twice produced a confident verdict from a wrong implementation.

## What this plan does NOT decide

Whether to build a soft state prior. If H1 holds, that is the next pre-registration, not this one.
