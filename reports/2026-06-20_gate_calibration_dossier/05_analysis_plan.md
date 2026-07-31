# 05 — Pre-Analysis Plan: C-147 gate reliability (PRE-REGISTERED)

**Date:** 2026-06-20 · **Status:** pre-registered **before** the rigorous run and **before** reading the
`loss_class_pos_weight` value. Append-only below this line once results land in `07`.

## One variable
This is **diagnostic, not comparative**: measure the calibration of `π = sigmoid(cls)` against binary
onset truth `1[lr_*>0]` on existing artifacts. Two runs are read (R4 no-coords `…_162127`, R5 coords
`…_165915`) but the **per-run** verdict is what matters; the coords-vs-no-coords gap is secondary.

## What "calibrated" means here
A gate is calibrated iff, among cells where it predicts probability `p`, the empirical onset frequency is
`≈ p` (reliability curve on the diagonal). Onset is **rare** (base rate ~0.1–0.5%/target), so a calibrated
gate must output **mostly small** probabilities. The proper reference is the **base-rate-constant predictor**
(predict the prevalence everywhere): its Brier ≈ prevalence·(1−prevalence) ≈ the prevalence (~0.001–0.005).

## Hypotheses (what I expect to see)
- **H1 — gate miscalibrated-HIGH (primary).** The reliability curve lies **below the diagonal** (predicted
  π ≫ empirical onset frequency) across most occupied bins; **mean π ≫ onset base rate**; **ECE > 0.3**;
  **Brier on π is WORSE than the base-rate-constant predictor** (Brier skill score **< 0**). I.e., the gate
  is not just imperfect — it is *worse than predicting the prevalence everywhere*.
- **H2 — the class weight drives it.** `loss_class_pos_weight` is set to a value **≫ 1** (predicted order
  ~inverse base rate, **O(100–700)**). Mechanism: heavy false-negative penalty ⇒ the gate floods positives
  ⇒ saturates toward π→1. *Conditional:* if `loss_class_pos_weight` is **unset/None or ≈1**, H2 is
  **falsified** and the cause shifts to optimization/feedback/capacity — H1 can still hold.

## Pre-committed falsifiers
- **F-calibrated (would surprise me):** reliability curve **on the diagonal** (within bootstrap CI) **and**
  Brier ≈ base-rate predictor (skill score ≈ 0 or > 0). ⇒ gate is calibrated ⇒ **the bloom is NOT the
  gate** ⇒ redirect to body/feedback. **Prediction: this does NOT fire.**
- **F-underconfident:** reliability curve **above** the diagonal / mean π **<** base rate (gate too timid,
  would *suppress* extremes). Prediction: does not fire (the plots show saturation, not timidity).
- **F-alignment-artifact (guard, not about the model):** if the C-136 truth-alignment guards fail (non-unique
  index, unmatched cells) or `by_*` ∉ [0,1], **STOP** — the readout is untrustworthy; fix pairing before any
  verdict. A miscalibration claim is void if pairing is wrong.

## Metrics (per target sb/ns/os × per forecast step k∈[0,7], and pooled)
- **Brier** on π; **Brier skill score** vs the base-rate-constant predictor (the decisive number).
- **Reliability diagram** — binned (deciles) mean-π vs empirical onset frequency, with per-bin counts.
- **ECE** (expected calibration error, count-weighted).
- **mean π** vs **onset prevalence** (base rate) — the one-line summary of direction/magnitude.
- **Bootstrap 95% CI** (resample cells) on Brier / mean-π (reuse `mcr_readout.py::_bootstrap_mcr_ci`).
- Internal-consistency checks: pooled Brier ≈ mean of per-step; base-rate Brier ≈ prevalence; bins' weighted
  mean-π ≈ overall mean π.

## Decision bar
- **H1 holds AND H2 holds** ⇒ gate is miscalibrated-high *because* of the class weight ⇒ the principled
  hypothesis (**pure Bernoulli NLL gate + explicit base-rate prior**, no class-weight hack) is supported →
  becomes the next **separately pre-registered** experiment. *(No fix in this dossier.)*
- **H1 holds, H2 falsified** ⇒ gate miscalibrated but not via the weight ⇒ pre-register an optimization/
  feedback investigation instead.
- **F-calibrated fires** ⇒ reject the gate hypothesis; redirect to the regression body / feedback loop.

## Exploratory peek (disclosed, NOT the pre-registered result)
During planning, an approximate single-step pairing on R5/`by_ns_best` gave mean π≈0.905, onset≈0.30%,
Brier≈0.895. This informed H1's direction but is **not** the rigorous result; it used coarse alignment, one
step, one target, no CI, no reliability curve. The `07` entry reports the rigorous run only.
