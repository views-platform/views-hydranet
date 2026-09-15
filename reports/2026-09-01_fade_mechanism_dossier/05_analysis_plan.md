# 05 — Pre-analysis plan: does clamping the cell state stop the fade? (M48's missing mechanism)

**Status: LOCKED 2026-09-01, before any run.** `results/` is empty.

## 1. Question

**M48** is the programme's only positive result: clamping the ConvLSTM long-term ("cell") state
during free-running improves gate skill on 4/4 seeds, gains **exactly 0.0000 at h1** rising to
**+0.0591 at h36**, and needs no retraining. **Why it works is unknown.** M43 refuted the
out-of-distribution-state explanation; M49 refuted the blur explanation.

A third explanation is assembled from measurements that already exist in three separate dossiers:

* the recurrent state's magnitude **collapses ~40×** during free-running — `max|h|` 65.6 → 1.6
  (falsifier-checks CHECK D, **seed 43**)
* the fed-back field collapses on both axes — occurrence 0.000612 → 0.000001 (**612×**), magnitude
  18.4 → 0.79 — while the same model fed *real* data stays flat (M49, **seed 42**)
* clamping the **cell** specifically recovers ~22% of the oracle gap; clamping the **hidden** half
  does nothing (M38/M39)

**H (the fade): the state drains → the emitted field weakens → the weaker field is fed back → the
state drains further. Clamping breaks the loop by holding the one thing that carries magnitude
across steps.**

If true, it also explains why every *firing* intervention failed (M42, M45, M47): they changed the
firing rule while the signal driving it decayed to nothing.

## 2. The one variable

`freeze_recurrent`: `None` → `'cell'`. Emit only, no training, same artifact.

**A blocker and its lever, recorded because it shapes the arm.** Feedback-field statistics are only
recorded when a feedback arm is set — `_record_feedback_stats` sits inside `if self._feedback_arm:`
(`hydranet_inference.py:1000`) — so a plain freeze run records **nothing**.
`feedback_transform="identity"` turns recording on while being **byte-identical to `None`**, asserted
by `tests/test_feedback_transform_seam.py::test_F3_none_is_byte_identical_to_identity`. The arm is
therefore `identity` **+** `freeze='cell'`, and the baseline is `identity` alone.

## 3. Two levels, each WITHIN its own seed

The state numbers are **seed 43**; the field numbers are **seed 42**. Comparing across them would be
a cross-seed comparison dressed as a within-vehicle one, so each level stays in its own vehicle.

### 3a — the field (seed 42, `fullzero_fortytwo`)

Baseline already in hand:
`reports/2026-08-31_sharpness_diagnostic_dossier/results/fedfield_fullzero_fortytwo_identity.csv`.
Verified equivalent to the archived control *and* to the freeze runner's `none` arm — all three
`AP@h18 = 0.3298395823400329`, exactly.

### 3b — the state (seed 43, `fullzero_fortythree`)

Baseline is the existing autoregressive capture in
`reports/2026-08-23_falsifier_checks/results/states_ar/` (`max|h|` 65.6 → 1.6). Re-run with
`--freeze cell --skip 335 --stride 1`; `origin = 335` and period 371 were **measured**, not assumed.

## 4. Predictions and falsifiers — registered before running

| | prediction | falsifier |
|---|---|---|
| **3a** | clamped `active_fraction` collapses **far less** than the 612× baseline | **F1:** if it collapses comparably, **H is dead** — M48 works some other way |
| **3b** | clamped `max|h|` holds near its anchor instead of draining 40× | **F2:** if it drains anyway, the clamp is not doing what its name says, and 3a's result needs re-reading |

* **F3 identity check.** 3a's arm must score `AP@h18 = 0.3621885544392029` — the archived seed-42
  `cell` value. If not, the arm is not what it claims and nothing here counts. (Same shape as
  EXP-04's G1, which caught nothing but proved the vehicle.)
* **F4 default-off.** A run without `--freeze` must reproduce the existing `identity` numbers
  exactly. The flag must not change the control.
* **F5 hidden half.** 3b's **cell** half is near-analytic — clamping holds it by construction — so
  a "cell held" result is **not** evidence for anything. The informative measurement is the
  **hidden** half, which evolves freely. C-292 notes `hs = o ⊙ tanh(hl)`, so holding the cell may
  bound hidden automatically; whether it does is the actual finding.

**Stop rule:** if F1 fires, 3b is not run and H is reported dead.

## 5. What this can and cannot establish

It can show whether the clamp **stops the fade**. It **cannot** show the fade *causes* the AP gain —
both could follow from a third thing. Establishing causation would need an intervention that stops
the fade *without* clamping, which is not in scope. The write-up must say so.

## 6. Prediction, on the record

I expect both to hold: the clamped field collapses much less, and the clamped state holds. I hold it
**weakly** — three predictions in this programme have now been wrong (`conc1pct` direction, the
`fss_ratio` discriminator, and M49's Moran's-I direction), and the last two "obvious" mechanisms for
M48 were both refuted.

**Budget: ~30 minutes GPU, emit only. Hard stop.**
