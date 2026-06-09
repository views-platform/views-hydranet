# 05 — Analysis Plan: the first ZITD experiment (P3 MVP)

**Date:** 2026-06-05 (pre-registered *before* the experiment — and before P1/P2 are even built) · **Status:** seeded
**Dossier:** [00_README](00_README.md) · **Gated on:** [04_roadmap](04_roadmap.md) M1 (loss+sampler) + M2 (head behind flag, baseline parity).

Pre-registers the **smallest unambiguous test** of the distributional head, so the result can't be rationalized after the fact. Structure mirrors this session's other pre-analysis plans (freeze_h, feedback-clamp, balancer-bisect).

---

## 0. ⏭ HURDLE-NB MVP pre-registration — the ESCALATION run (2026-06-09; only if hurdle + rollout underdelivers — see `00 §0`)

The Tweedie/ZITD MVP (§1–§8 below) is preserved as the **tail-escalation**. **NOTE (2026-06-09): this is the *escalation* pre-registration — the live next step is hurdle + rollout training (`00 §0` / `../2026-06-08_magnitude_calibration_dossier/`); run this only if that underdelivers.** The escalation run (if triggered) is a **hurdle-NB** (a 2nd review caught that reusing the focal classifier as a ZINB *structural* π mis-specifies the likelihood — see `02 §0.0`).

- **Hypothesis (dual):** a **hurdle-NB** head (gate = the existing classifier `P(y>0)`; positives = a **zero-truncated NB** with softplus μ + learnable θ; `E[y]=P(y>0)·E[NB⁺]`) trained on **violet / seed-42 / 40 lessons** (matched to the Tobit baseline `calibration_model_20260608_165326`) (a) keeps the 36-step free-running `E[y]` **bounded**, and (b) **un-collapses magnitude without exploding** — `MCR_pos` rises toward O(1) (vs Arm-1, where un-collapsing → explosion).
- **Intervention (a coherent head-swap — NOT one variable; vs the Tobit baseline):** `output_distribution="hurdle_nb"` (gate-as-classifier + the zero-truncated-NB softplus-μ positive part + learnable θ, the hurdle-NB loss, the contained raw-target provider). Everything else matched (40 lessons, seeds, dropout, onset_bias, SS off, log1p **inputs**). Skill attribution needs **ablations** (global-vs-varying θ; a dedicated-π ZINB if the hurdle underfits) — pre-registered as follow-ups.
- **GATE FIRST (read-only go/no-go):** before *any* scored eval, run `diagnose_io_gain` on `E[y]` over 36 steps. **Explodes → STOP** (Arm-1 proved an un-collapsed head can run away); also read the **operator gain** (Part A); log + escalate, do NOT score. **Bounded → proceed** to eval.
- **Falsifiers:** reuse **F1 (stability)** + **F5 (zero-rate trap)** below verbatim; add **F-bridge** — the raw-target provider must pass its exact-round-trip + never-touch-predictions tests or the run is invalid. Judge skill on **positive-subset proper scores (twCRPS/CRPS on y>0) + PIT/coverage** + a **posterior-predictive zero-rate/tail check**; **MCR diagnostic only**; **multi-seed** before any adoption claim (shrinkage-volatility lesson); **benchmark vs DynAttn** grid-level numbers.
- **Decision rules:** gate explodes → rollout-training (Axis B) / direct multi-horizon. Bounded + calibrated, no falsifier → multi-seed → proposed ADR. Bounded but tail-underfit → Tweedie/ZITD (below) or DEMM tail; structural-zeros evidence → dedicated-π ZINB. **No ad-hoc rescue.**

---

## 1. Hypothesis

**H:** A ZITD output head (`E[y]=(1−π)μ`, sub-exponential **softplus** link, **fixed `ρ≈1.5`**, **mean rollout**) trained on **violet** (the clean exploder) will (a) keep the free-running forecast **in-range** across the 36-step rollout *by construction* (no `expm1` cliff — `02 §0.1`), and simultaneously (b) **improve calibrated magnitude** — CRPS no worse than the healthy baseline and **MCR moving toward 1** from the chronic ≪ 1.

This is the dual claim: ZITD fixes the **acute** runaway *and* the **chronic** under-prediction in one move, where the in-domain clamp only did a degenerate version of the former (`results_feedback_clamp.md`).

## 2. Intervention & configuration (one variable from baseline: the output head)

- Model **violet_visitor**, full retrain; everything else per current config except `output_distribution="zitd"`.
- **Fixed `ρ=1.5`** (avoid the `ρ∈(1,2)` saturation risk on the first cut, `02 §7.2`); **softplus** `μ`; **mean** autoregressive feedback (`log1p(E[y])`); `n_posterior_samples=16`; one model on GPU.
- Readout order (`03/04`): **`diagnose_io_gain` (adapted to `E[y]`)** first (~30 s) → then a full `--evaluate --saved` for the scored metrics.

## 3. Pre-registered predictions

Baseline references: `reports/s0_baseline_metrics.md` (the healthy pre-explosion log1p baseline) and pink's healthy metrics (lr_sb CRPS ≈ 0.13). Violet-active is *not* a skill baseline (it explodes) — it is the **stability** baseline.

| Endpoint | Prediction |
|----------|-----------|
| **Stability** (free-running `E[y]`, all 36 steps) | stays within the data range (`≲ expm1(12.1)≈1.8e5` counts); no ratchet. |
| **CRPS** (step-wise, lr_sb/ns/os) | **in the healthy range** (O(0.1)), ≤ the best log1p/Tobit baseline. |
| **MCR** | **moves toward 1** (from ≪ 1); not collapsed toward 0 or blown up. |
| **Calibration** | PICP near nominal; PIT roughly uniform. |
| **Zero-rate** | predicted zero-rate ≈ empirical (~95%); **not** 100% (see F5). |

## 4. Falsifiers (pre-committed — any one fires ⇒ the MVP config is rejected, not rescued)

- **F1 — stability fails:** free-running `E[y]` still leaves the data range ⇒ the link/parameterization doesn't bound the rollout ⇒ rethink the head before scaling.
- **F2 — no skill:** CRPS materially worse than the log1p/Tobit baseline ⇒ ZITD doesn't help magnitude/skill.
- **F3 — magnitude unfixed:** MCR still ≪ 1 ⇒ the chronic under-prediction is not addressed.
- **F4 — training pathology:** NaN/Inf NLL, `ρ`/`φ` saturation, or non-convergence ⇒ numerical/parameterization issue (ties to the Tweedie-density blocker, `03 §3.1`).
- **F5 — the zero-rate trap (the dangerous one):** with ~95% zeros, NLL can be minimized by predicting **always zero** (π→1 everywhere, μ→0). If the predicted zero-rate is ~100% / `E[y]≈0` on known-active cells / derived `P(Y>0)` AUC collapses ⇒ the model is exploiting zero-inflation, "calibrated" but useless. **This is the count-likelihood analog of F2 in the clamp experiment (bounded-but-degenerate)** and must be checked explicitly, not inferred from CRPS.

## 5. Metrics & instrumentation

Beyond CRPS (proper, already our metric — `01 §5`), the zero-inflation structure demands metrics that the 95% zeros can't dominate:

- **zRMSE** (zero-weighted RMSE, Kong 2020) — separates zero vs positive error.
- **Positive-cell skill** — CRPS/MAE on the `y>0` subset; `E[y]` on known-active cells.
- **Derived `P(Y>0)` AUC** — guards F5 (is the zero/non-zero discrimination preserved?).
- **PICP / MPIW / PIT** — interval coverage, sharpness, calibration (`01 §5`; observation-error-aware per Bessac/Weijs).
- **Stability trajectory** — per-step `max E[y]` over the rollout (the F1 curve).

## 6. Controls

- **pink ZITD (regression control):** a *healthy* model under ZITD must stay healthy (CRPS not degraded). If pink degrades, ZITD has a cost even where there was no problem.
- **Baselines for skill:** log1p-point (`s0`) and Tobit/ADR-054 — ZITD must beat **both** to justify adoption (`04 §5`).
- **Single seed caveat (C-112):** one seed conflates signal with variance; treat the MVP as directional, confirm on ≥1 more seed in P4 before any adoption claim.

## 7. Skepticism (carried from `02 §7` + this session)

1. **Density approximation bias** — the Jiang lower bound (or chosen density route) may bias `μ,φ` estimates; validate against a reference impl in P1.
2. **Fixed `ρ` is a simplification** — regions differ in tail heaviness; a single `ρ` may underfit. That's deliberate for the MVP; learned `ρ` is a P4 ablation.
3. **Evaluation comparability** — ZITD changes the output interface; CRPS/MCR must be computed *identically* to the baseline or the comparison is void (harness eval-comparability gate).
4. **Balancer meaning change** (`02 §7.5`) — with one loss per target, the MTL balancer now weights target-vs-target; interacts with the C-111 bisect; keep `freeze_multitask_balancer` available.
5. **Not the acute fix's substitute** — if the C-111 bisect shows a cheap acute fix, ZITD is still justified for the chronic problem (`02 §0.4`), but don't conflate the two wins.

## 8. Decision rules

- **All predictions hold, no falsifier fires** ⇒ MVP succeeds → proceed to P4 ablations (learned `ρ`, sampled rollout, per-target `φ,ρ`), then a multi-seed confirmation, then P5 ADR.
- **F1/F4 fire** ⇒ head/numerics problem → fix in P1/P2 before re-running.
- **F2/F3 fire (stable but no skill/magnitude gain)** ⇒ ZITD bounds but doesn't improve → reconsider vs Tobit; consider Path C tail.
- **F5 fires (zero-rate trap)** ⇒ add a zero-rate/positive-cell penalty or revisit the `π` parameterization before continuing.
- Any outcome → `07_experiment_log` entry; negative results documented (no ad hoc rescue), per the session's discipline.
