# Pre-Analysis Plan — Bulk-magnitude dial (winsorize + τ-pinball)

**Date:** 2026-07-16 (pre-registered **before** execution).
**Dossier:** `reports/2026-07-16_bulk_calibration_dossier/` · **Builds on:** `02_design`, `03_harness §D`
(the LOCKED metric), [[project_densemse_beats_baseline]], [[project_body_knob_quest]].

## 1. Hypothesis
**H:** On the dense-mse+wBCE(pw2) body, an **outlier-robust per-cell winsorized target** (stabilizer) plus a
**moderate-τ log-space pinball** magnitude dial (lifter, τ the knob) lifts the **bulk** expected value from
TIMID (`ratio_med` 0.05–0.11) toward CALIBRATED (`ratio_med ∈ [0.7, 1.3]`) at **T=0**, **without** regressing
the frozen gate's Brier or overall CRPS/QS99 — where plain reweighting and uncapped lifters could not
(they went timid or exploded).

## 2. Intervention (the ONE variable — the body target+loss; gate & everything else FROZEN)
Replace plain MSE-on-log1p body loss with **winsorized-target + τ-pinball**, behind a default-off
`loss_reg` flag. Held constant: backbone `HydraBNUNet06_LSTM4`, gate (wBCE `pos_weight=2`), data, seed,
40 lessons, the (N,S) pipeline, and the **T=0 bulk-calibration metric (03 §D)**. Arms (built-in controls to
attribute the effect):
- **A0** baseline — dense-mse pw2, plain MSE (the P0 anchor).
- **A1** control — **winsorize + plain MSE** (stabilizer ALONE). Predicted to stay timid → proves the
  *dial*, not the cap, is the lever, and proves the cap doesn't itself explode.
- **A2** hypothesis — **winsorize + τ-pinball**, τ swept {0.5, 0.6, 0.7, 0.8} (τ=0.5 ≡ A1's minimiser).

## 3. Skepticism ledger
1. **"Works" by over-firing** (bulk mass leaks onto zeros / tail) → guardrails Brier/CRPS/QS99 gate it; the
   lab's τ=0.99 pinball did exactly this. **F2.**
2. **The cap alone lifts** (confound) → A1 control must stay timid; if it lifts, attribution is wrong. **F3.**
3. **Global rescale, not genuine calibration** → require `within2x_rescaled` to improve, not just `ratio_med`.
4. **Regime overfit** (count_mean lifted on calibration then collapsed OOS) → validation graduation. **F5.**
5. **Instability survives the cap** (exp-gradient/fan still explodes) → A2 non-finite or `ratio_med`≫1. **F4.**
6. **Metric self-deception** → the metric is LOCKED in `03 §D` (T=0-only, positives-only, bulk-only, per-cell
   `ratio_med` not MCR, cut reported at 97/98/99, same-seed A/B, bootstrap CI, ≥3 seeds) BEFORE any run.
7. **Winsorizing biases the point low vs the true (uncapped) mean** — intentional: the body should predict
   the body; the capped-off tail is the tail's job (parked). Judge on the capped-truth bulk, not raw truth.

## 4. Pre-registered predictions
| Endpoint (primary first) | Prediction | Threshold (pass / fail) |
|---|---|---|
| **A2 bulk `ratio_med` @ T=0 (best τ)** | lifts into calibrated band | **PASS ≥ 0.7 (≤1.3); FAIL if < 0.5 at every τ** |
| Δ`ratio_med` A2 − A0 (month-block bootstrap CI) | positive, real | CI excludes 0 |
| A1 (winsorize+MSE) bulk `ratio_med` | stays timid | **< 0.3** (else confound, F3) |
| `within2x_rescaled` A2 vs A0 | up (genuine) | strictly higher |
| Guardrails Brier / CRPS / QS99 @ T=0 (A2 vs A0) | held | not worse than A0 beyond noise |
| Ratio spread p10/p50/p90 (bulk) | tightens toward 1 | reported (even vs lopsided) |
| Robustness | reproducible | ≥3 seeds; validation partition holds |

## 5. Falsifiers (pre-committed — any one fires ⇒ hypothesis rejected, not rescued)
- **F1 (ineffective):** A2 `ratio_med` < 0.5 at every τ ⇒ the dial does not lift the bulk ⇒ winsorize+pinball
  is not the body knob.
- **F2 (works-but-degenerate / over-firing):** A2 hits the band but Brier/CRPS/QS99 regresses beyond noise ⇒
  it's over-firing, not calibration.
- **F3 (confound):** A1 (cap alone) also lifts `ratio_med` ≥ 0.3 ⇒ the cap, not the dial, moved it (or the
  metric/cut is confounded) ⇒ re-attribute.
- **F4 (unstable):** A2 non-finite or `ratio_med` ≫ 1 uncontrolled even WITH the cap ⇒ the stabilizer failed.
- **F5 (OOS collapse):** A2 lifts on calibration but collapses on validation ⇒ regime overfit; the cap did
  not prevent it.

## 6. Method
Backbone/gate/data frozen. Winsorize = per-cell cap at `k ×` a robust running statistic of the cell's own
recent positives (k / stat fixed in P1, logged). τ-pinball = log-space asymmetric loss, `LOSS_REG_REGISTRY`
(OCP, default-off), NaN/Inf-guarded, unit-tested (minimiser=τ-quantile, finite grad, off⇒baseline byte-
identical). Metric = `03 §D` via extended `t0_score.py` (retrain-free). **Readout order (cheap→expensive):**
P0 anchor A0 on existing preds → 2-lesson smoke → 40L A0/A1/A2(τ-sweep) seed 42 → ≥3 seeds → validation.
`conda run -n views-hydranet-env`; views-models config stealth (trap-restore); one heavy job at a time.

## 7. Decision rules
- **PASS (H holds):** A2 clears `ratio_med`∈[0.7,1.3] with CI>0, A1 stays timid, guardrails held,
  `within2x_rescaled` up → confirm ≥3 seeds → validation → `promote` (proposed ADR: body-magnitude loss).
- **F1 →** switch lifter to `count_mean`-on-capped (the proven-lifter fallback); if that also fails, the bulk
  bias is not loss-movable at capacity → re-scope toward the mixture (body + borrowed tail).
- **F2 →** back off τ / tighten the cap; the calibrated point may sit below [0.7,1.3] without breaking
  guardrails — record the best guardrail-safe `ratio_med`.
- **F3 →** the metric/cut is confounded → fix the metric before any more runs (measurement is sacred).
- **F4 →** lower the cap / clamp the loss; if still unstable, capping doesn't fix the exp-gradient family.
- **F5 →** banked negative (postmortem): winsorize does not prevent the regime overfit → the calibration
  win was illusory; redirect to the mixture path.

## Changelog — 2026-07-16 (amendment after the P0 anchor; user-approved)
The P0 anchor found the baseline body is **DEAD, not timid** (`ratio_med` 0.000; 97% of positive cells
emit *exactly* 0). Root cause is definitive: `output_distribution='standard'` defaults to **ReLU** (the
dead-ReLU, C-178), and our run never set `reg_activation` → we ran the ReLU default (saved config confirms
`reg_activation=relu`). The T=0-screen's alive dense-mse explicitly set `reg_activation='softplus'`. Two
amendments:
1. **Revive-first.** A dead body has no gradient to dial. Add a revival step (a 1-line flag, no new code):
   - **A0 (anchor)** = dead ReLU dense-mse (`ratio_med` 0.000) — recorded.
   - **A0′ (revive)** = dense-mse + `reg_activation='softplus'` — expected alive-but-timid (`ratio_med`
     ~0.05–0.11, the screen's number). This is the *real* baseline the winsorize+dial rides on.
   - **A1 (control)** = A0′ + winsorize (cap only) — predicted to stay timid.
   - **A2 (hypothesis)** = A0′ + winsorize + τ-dial — lifts to `ratio_med` ∈ [0.7, 1.3].
   - *Fallback (only if A2's lift is fought by the all-cell zero-pull):* train the body on positive/active
     cells (hurdle-compose with the frozen gate).
2. **Honest CRPS bar = `white_ranger` (0.276), NOT the dead dense-mse (0.140).** The dead body's low CRPS
   is itself the artifact; a revived/honest body will predict real magnitudes → higher CRPS. Guardrail
   restated: **A2 CRPS must still beat white_ranger's T=0 CRPS (sb 0.276 / ns 0.108 / os 0.039)** and hold
   Brier — NOT "not worse than the dead dense-mse." (Primary `ratio_med` bar unchanged: [0.7, 1.3].)

## Changelog — 2026-07-16 (P3 arm realization, recorded BEFORE the run)
The P2 smoke passed (mechanism trains, finite, dial-active). Realizing the amended A0′/A1/A2 as the actual
P3 runs surfaced two decisions, recorded here to keep the pre-registration↔outcome linkage honest:

1. **Positives-only compose from the start (05's sanctioned fallback, promoted to base).** The revive-first
   amendment listed "train the body on positive cells" as a *fallback* "only if the all-cell zero-pull
   fights the lift." But P0 **already proved** the all-cell zero-pull kills the body (`ratio_med` 0.000; the
   99.7%-zero mean drags MSE to 0). Running all-cell A0′/A1/A2 first would only reproduce that understood
   dead/timid result and burn ~half the batch before the inevitable fallback. So P3 runs the fallback config
   directly: `output_distribution='hurdle_shrinkage'` + `hurdle_threshold=0` (per-step positives mask) — the
   frozen gate (`weighted_bce`, pw2) owns the zeros, the softplus body owns the positive magnitude. This is
   where the dial *can* work. A0 (all-cell dead-ReLU, `ratio_med` 0.000) stays the banked anchor.

2. **Realized as a clean single-variable PINBALL ladder** (so F1/F3 attribution is unconfounded):
   - **A0′** = `pinball` τ=0.5, **no cap** — revived median body, uncapped (the timid baseline; expect
     `ratio_med` low, ~0.05–0.2).
   - **A1** = `pinball` τ=0.5, **cap=5.7** — adds ONLY the winsorize (F3 control: must stay timid, <0.3).
   - **A2** = `pinball` τ∈{0.6,0.7,0.8}, **cap=5.7** — adds ONLY the dial (raise τ; H: lifts to [0.7,1.3]).
   This supersedes 05 §4's "A1 = winsorize+MSE": using pinball τ=0.5 as A1's base makes A0′→A1 a pure
   add-cap step and A1→A2 a pure raise-τ step (MSE→pinball would confound the loss family with the cap).
   `cap=5.7` = log1p(300) ≈ the sb 98th-pct positive count (single cap across targets; per-target caps a
   later refinement). All arms: seed 42, 40 lessons, `priogrid_id` (#144 flip), train+eval, scored by the
   locked metric (`ratio_med` primary + Brier/CRPS/QS99 guardrails vs white_ranger).
