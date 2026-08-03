# Pre-Analysis Plan — In-Domain Feedback-Input Clamp (C-113)

**Date:** 2026-06-04 (pre-registered *before* running)
**Branch (views-hydranet):** `fix/variational-dropout-autoregressive-stability`
**Builds on:** `reports/results_freezeh_ablation.md` (driver = prediction→input loop, not recurrent state), `reports/results_io_gain_diagnostic.md` (violet's free-running map settles at an out-of-range log-space attractor ~40 → `expm1` ≈ 1e17; pink stays in-range)
**Risk:** C-113

---

## 1. Hypothesis

**H:** The runaway is the fed-back prediction *ratcheting past the training data range over the 36 autoregressive steps* (step-1 in-range ~3 → climbs to log ~40). If we bound the **fed-back copy** of the prediction to the per-target training range each step, the ratchet breaks at the source and violet's evaluation metrics return to in-range — **without** capping any emitted prediction.

## 2. The intervention (precise — read this before judging it)

In `predict()`, the autoregressive feedback is `t0_autoreg = t1_pred.detach()`, fed as the next input. The change clamps **only that fed-back copy**, per target channel, to the log1p training-data max:

```
t0_autoreg = t1_pred.detach()
t0_autoreg = torch.minimum(t0_autoreg, ceiling)   # ceiling = [sb, ns, os] per channel
```

Ceiling (log1p space, from the scaler's observed per-target max on this data): **`[lr_sb=10.78, lr_ns=7.60, lr_os=12.09]`**. ReLU already gives a ≥0 floor; only `max` is bounded.

**What is and is NOT clamped:**
- **NOT clamped:** every *emitted* prediction (`acc_magnitudes.append(t1_pred)`) — the values that flow to scoring. The emission path is byte-unchanged. So this is **not** an output clamp (ADR-028 §3, rejected).
- **Clamped:** only the *input copy* handed to the next step. Downstream emitted predictions are *indirectly* affected (the next step sees an in-range input), which is the intended ratchet-break.

Off by default (`feedback_clamp_log1p = None`) → zero behavior change for every existing model and test.

## 3. Skepticism ledger (the reservations we hold about clamping — explicit, because clamping is the thing we have most deliberately avoided)

1. **Clamping is what ADR-028 §3 explicitly REJECTED.** Output clamping caps the upper tail and worsens MCR (we already under-predict, MCR ≪ 1). The defence here is structural: we clamp the *feedback input*, never the *emitted output*. But this defence is only partial — see #2.
2. **Indirect magnitude/MCR effect is real.** Although no emitted value is directly capped, clamping the feedback lowers *all subsequent* predictions (the next step ingests a smaller input). So downstream emitted magnitudes drop. This could *worsen* MCR (more under-prediction) — or it could be neutral/irrelevant because the values it suppresses were garbage (log 40), not legitimate signal. **Must measure MCR, not assume.**
3. **It treats the symptom, not the cause.** The trained map still *wants* to run to log 40; we slap it back to ~12 every step. The fix does not repair the pathological attractor — it truncates the trajectory. Durable fixes (spectral-norm on the input→output path, pushforward/GTF, count-likelihood head) remain necessary; this must not be sold as the root fix or used to declare victory.
4. **Bounded ≠ correct (the degenerate-trajectory risk).** A clamped rollout could be a meaningless sawtooth (predict 40 → clamp 12 → predict 40 …) or pin every cell at the ceiling. "CRPS no longer 1e17" is necessary but **not sufficient**; the predictions must also be *sensibly varied below the ceiling*, not a uniform wall at 12.09. This is a pre-registered falsifier (§5).
5. **Escalation truncation — domain cost.** This system exists to forecast conflict *escalation*. Clamping at the historical per-target max forbids predicting above-historical events — exactly the tail we care about. Aggressive but defensible as "in-domain"; the cost is real and must be named. (A looser/global ceiling is the more escalation-permissive alternative if per-target proves too tight.)
6. **The ceiling is partition-specific.** `[10.78, 7.60, 12.09]` are this data partition's observed log1p maxima, hardcoded for the experiment. Production should source them from the *training* scaler's locked stats, not hardcode. Using eval-partition maxima is a mild self-reference we accept for a falsification test.
7. **Train/inference mismatch.** Training is teacher-forced on real (in-range) inputs; it never saw clamping. Adding a clamp at inference is a mismatch — though it arguably makes inference inputs *more* like training (in-range) than the unclamped log-40 runaway. Double-edged.
8. **Not guaranteed benign on healthy models.** In the synthetic diagnostic, pink transiently hit log ~16 mid-rollout — above the 12.09 ceiling. On real data pink predicts ~in-range, so the clamp *should* be ~no-op, but this is not guaranteed. Pink is run as a **control**; if its metrics degrade, the clamp is not safe as a universal default.
9. **Numerical edges.** `minimum(inf, c)=c`, `minimum(nan, c)=nan`. If a pre-clamp value is already NaN/Inf the clamp won't launder it; rely on existing NaN guards. Acceptable (we expect finite log-space values pre-clamp).

## 4. Pre-registered predictions

- **Violet (test):** with the clamp, step-wise CRPS on all three regression heads returns **in-range** — primary `lr_sb_best/CRPS < 10` (was 2.13e17; healthy ≈ 0.1), `lr_ns_best/CRPS < 10` (was 2.78e9). Raw predictions ≤ ~`expm1(ceiling)` (within data range).
- **Pink (control):** CRPS changes by **< ~10%** on all heads (clamp ≈ no-op on a healthy model). 
- **MCR:** reported for both; expected to change for violet. Stabilization is the endpoint; a finite, non-collapsed MCR is acceptable.

## 5. Falsifiers (pre-committed)

- **F1 — ineffective:** violet still explodes (`lr_sb_best/CRPS > 1e3`) ⇒ feedback ratcheting is not the (sole) mechanism, or ceiling too high ⇒ rethink (within-step amplification? per-channel mismatch?).
- **F2 — bounded-but-degenerate:** violet's CRPS drops but predictions are pinned at / pile up against the ceiling (check the prediction distribution: fraction of cells at the ceiling; spatial variety) ⇒ "bounded garbage," not a fix.
- **F3 — not benign:** pink's CRPS shifts > ~10% ⇒ clamp not safe as a universal setting; would need per-model / adaptive ceiling.
- **F4 — MCR collapse:** violet/pink MCR collapses toward 0 ⇒ the magnitude-neutrality defence fails in practice.

## 6. Method

- **Models:** violet_visitor (test), pink_pirate (control). Same artifacts, `n_posterior_samples=16`, LockedDropout active (constant). One model on GPU at a time.
- **Only variable:** `feedback_clamp_log1p` = `[10.7828, 7.6014, 12.0921]` (None for the established baselines, already on record).
- **Invocation:** `bash models/<m>/run.sh --evaluate --run_type calibration --saved --report` (canonical).
- **Detection:** `wandb:` CRPS + MCR summary block per run (authoritative; the drift guard is blind here). For F2, inspect the emitted-prediction distribution if CRPS bounds.
- **Safety:** config backed up; `trap … EXIT` restores both configs even on crash. No `set -e`.
- **Implementation gating:** clamp behind `feedback_clamp_log1p` (default None); training path and all existing tests byte-unchanged; TDD with ADR-005 taxonomy; full suite + ruff green before any eval.

## 7. Disposition rules

- **Clamp works (violet in-range, pink benign, predictions non-degenerate, MCR not collapsed):** record as a **validated stopgap** (inference-only, retrain-free) — explicitly *not* the root fix; promote spectral-norm/pushforward/count-head as the durable program; consider an ADR for the clamp as a guard rail with the escalation caveat documented.
- **Any falsifier fires:** document honestly (as with the dropout postmortem); do not rescue ad hoc; fold the finding back into the options catalogue.
