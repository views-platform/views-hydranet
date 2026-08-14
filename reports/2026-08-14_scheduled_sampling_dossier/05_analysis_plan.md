# 05 — Analysis plan (pre-registration) — 🔒 LOCKED 2026-08-14

**STATUS: LOCKED.** Predictions + falsifiers committed *before* any arm is scored.
**Preconditions met (hard gates):**
- SS piping proven == inference exposure — `tests/train/test_feedback_parity.py` GREEN (train feedback
  byte-equals inference feedback for `truncated_nb`+`soft_gate`); the coupling/order validators
  (C-259/C-260) are live and fail-loud. Committed `c07a352`.
- `truncated_nb` sampler perf-fixed (699 ms/call, was 112 s) so SS training (samples every step) is
  feasible; all family correctness tests green.
- Fresh africa datafactory pull (never stale).

Do not amend below this line except to record that execution has begun.

**Builds on:** the #258 truncated_nb result (T=0 collapse FIXED — activation 1.62× the truth at h=1 vs
old NB 0.03× — but the free-running rollout BLOOMS: act_ratio 1.6×→18×→44× across h1/18/36). Root
(agreed): **exposure bias** — the model trains pure teacher-forced (`ss_epsilon_max=0` everywhere) and
never conditions on its own gated forecast, yet rolls out on exactly that. C-259..C-262, C-246.

## 1. Hypothesis
Closing the train/rollout gap with scheduled sampling (ADR-056) **flattens the activation bloom**, and
does so **monotonically in the self-exposure dose ε** (`ss_epsilon_max`): more exposure to its own
gated forecast during training ⇒ flatter horizon curve (act_ratio → ~1 across h1/18/36), **without
collapsing T=0 activation**.

## 2. Intervention (the ONE variable)
`ss_epsilon_max` swept over **{0.0, 0.1, 0.25, 0.5}** — a dose-response. Every other knob is held at the
`truncated_smoke` floor: `output_distribution=truncated_nb`, `forecast_composition=soft_gate`,
`body_supervision=active`, seed 42, 40 lessons, D×K=4×4, `ss_schedule=linear`, `ss_warmup_lessons=15`,
**`ss_feedback='sample'`** and **`rollout_feedback='sample'`** (validator-forced equal for ε>0). ε=0.0 is
the teacher-forced anchor (re-emitted from the existing trained artifact with the fast sampler).

## 3. Skepticism ledger
1. **Undertraining confound.** 40 lessons is short; some of the bloom is likely undertraining, and SS
   fed a not-yet-good model's own predictions can *destabilise* (why `ss_warmup_lessons=15` exists).
   A null at 40L does NOT rule out SS helping at 300L.
2. **Gate-precision wall.** The bloom is the *gate* over-firing on its own drift (AP→0.01 by h36). SS
   trains the model to be robust to self-inputs but does not directly make the gate more precise, so
   the bloom may be only partly exposure-bias.
3. **Partial-(c).** SS still scores each step against the TRUE target; it teaches robustness to
   self-generated inputs, it does not make generated fields provably match the DGP. A win here is
   "the gap matters," not "exposure bias solved."
4. **Single seed.** Seed 42 only — a monotone trend across 4 ε values on one seed is suggestive, not
   seed-robust; magnitude is not to be trusted, only the direction/shape.
5. **Goodhart.** Judge on activation shape across horizons, not a single scalar; never on crps_all.

## 4. Pre-registered predictions (primary first)
| # | Endpoint | Prediction | Pass threshold |
|---|---|---|---|
| P1 | `act_ratio@h36` vs ε | **monotone decrease** toward 1 | act_ratio@h36 strictly ↓ over ε=0→0.5, and the ε=0.5 value < ½ the ε=0 value (44×) |
| P2 | `act_ratio@h1` vs ε | stays ~1 (T=0 not collapsed) | act_ratio@h1 ∈ [0.5, 3] for all ε |
| P3 | `mag_on_false_pos@h36` vs ε | falls with ε | mag_on_false_pos@h36 ↓ monotone |
**Fixed method constants (locked):** horizons h∈{1,18,36}; targets sb (ns/os reported, not gating);
score with `score_v2_horizons.py` + `activation_metrics.py` on the fresh africa truth; support
intersected across all 4 arms (G4). **NEVER judge on crps_all** (blind when sparse, #258).

## 5. Falsifiers (pre-committed — any one fires ⇒ hypothesis rejected, not rescued)
- **F1 (no lever):** the bloom persists across all ε — act_ratio@h36 stays ≫1 with **no monotone
  trend** ⇒ exposure bias isn't the (dominant) driver at this scale; the wall is gate precision.
- **F2 (traded failure / non-monotone):** SS collapses or destabilises T=0 (act_ratio@h1 leaves
  [0.5,3]), OR higher ε makes h36 **worse** (non-monotone) ⇒ SS trades one failure for another.
- **F-DEGEN:** training NaN / diverges at some ε (bloomy-undertrained ramp instability) ⇒ that ε is
  uninterpretable; report and do not rescue.

## 6. Method
4 arms (seed 42, 40L), one variable = `ss_epsilon_max` ∈ {0.0, 0.1, 0.25, 0.5}, `ss_feedback='sample'`.
Fresh datafactory pull; `diagnostic_visualizations=False`; setsid-detached hardened driver
(config floor-md5 trap-restore, clear-predictions, TRAIN proc1 → EMIT fresh proc2 `-e -sa`,
inline-score, manifest+sentinel). ε=0 arm re-emits the existing trained artifact (no retrain). Score
each arm's D×K cube; assemble the horizon curve per arm. **Never score on stale data.**

## 7. Decision rules & ⚠️ honest null-scoping
- **P1+P2+P3 hold** (monotone flattening, T=0 intact) ⇒ exposure bias is a real, dose-responsive
  driver of the bloom → promote scheduled sampling to the roster program (retrain at 300L, multi-seed).
- **F1 fires** ⇒ the wall is gate precision, not exposure → pivot the program to gate precision; SS is
  not the (whole) answer.
- **F2/F-DEGEN** ⇒ SS at this ε/length is not viable as-is → scope (warmup/ε schedule, longer train).
- ⚠️ **A null means "SS at ε≤0.5 / 40 lessons did not flatten the bloom," NOT "exposure bias is
  irrelevant"** (skepticism-ledger items 1–3). The truncated_nb T=0 fix stands regardless.
