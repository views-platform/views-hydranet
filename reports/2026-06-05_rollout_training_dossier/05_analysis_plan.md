# 05 — Analysis Plan: the first rollout-training experiment (B1 pushforward MVP)

**Date:** 2026-06-06 (pre-registered *before* B1 is built) · **Status:** seeded
**Dossier:** [00_README](00_README.md) · **Design:** [02_design](02_design.md) (R1–R7 folded) · **Review:** [02b_method_review](02b_method_review.md)
**Gated on:** the C-113 regression guard (#76, **DONE**) + `03_harness_and_invariants` (the rollout-loss test harness, TODO) + GPU.

Pre-registers the **smallest unambiguous test** of training-through-the-rollout, and **resolves the two open planning gaps** the `/falsify` audit found (RT-P2 balancer/R6 → C-125; RT-P4 rollout×ZITD → C-129). Structure mirrors the dossier convention (freeze_h, feedback-clamp, balancer-bisect, ZITD `05`).

---

## 0. Two decisions this plan settles (falsify RT-P2 / RT-P4)

### 0.1 P2 — balancer config + the R6 gate (resolves the C-125 RT-P2 note)
- **R6 is satisfied.** The method review's R6 ("sequence Axis B after the C-111 balancer verdict closes") is met: the balancer×seed sweep concluded (`reports/preanalysis_balancer_sweep.md` §RESULT) — active explodes 3/3, frozen is seed-fragile (F2 fired) → **freezing is not the robust fix; exposure bias is the root → do rollout training.** We are not waiting on further balancer work. **Proceed.**
- **B1 runs `freeze_multitask_balancer=False` (ACTIVE) as the PRIMARY arm.** Active is both the production-intended setting *and* the exploder (baseline `…233938.pt`, ~log 16). The real test is therefore: *does training-through-the-rollout bound the model the active balancer currently blows up?* **Success resolves C-124** — the active (learnable) balancer earns its place once training is honest about the rollout.
- **Secondary control:** `freeze_multitask_balancer=True` (FROZEN, `…051634.pt`, healthy) — B1 must not break an already-healthy model (also the calibration reference).

### 0.2 P4 — rollout × ZITD coordination (resolves C-129)
- **Sequence, not concurrent. Axis B goes first.** B1 is a *training-loop-only* change (output stays `log1p`+`expm1` → composable with a later ZITD head); its guard (#76) is in place; ZITD is a larger architectural change; and both edit `training_engine` + the autoregressive feedback, so concurrent development would collide (merge + confounding).
- **Layering rule:** if B1 bounds the rollout **but** calibration stays wrong (F2 / C-126), the ZITD softplus head is the next layer (it targets calibration directly). If B1 fixes both, ZITD may be deprioritised. A reciprocal one-liner is to be recorded in the ZITD dossier's `02_design`.

## 1. Hypothesis
**H:** B1 pushforward — train through K autoregressive steps feeding back the model's own one-step-prior prediction (detached across steps; backprop the last step only), adding an **annealed** stability term — makes the **active-balancer** violet model (which currently explodes) keep its **free-running forecast in-range across all 36 steps**, **without degrading calibration** (CRPS / MCR / coverage no worse than the healthy frozen-seed-42 reference).

Dual claim, mirroring the runaway's two faces: B1 fixes the **point/mean** runaway (the C-113 signature) **without** trading it for the **calibration** pathology the panel warned of (mean-hedging / blurring — C-126).

## 2. Intervention & configuration (one variable: the rollout objective)
- **violet_visitor**, seed 42, **active** balancer (`freeze_multitask_balancer=False`, the exploder); full retrain; everything else per current config except the new `rollout_horizon`.
- **`rollout_horizon = 12`** (reaches the observed step-12 blow-up onset); pushforward stability term with an **annealed weight** (R1 — → small, CRPS reported uncontaminated); **gradient clipping** (R7).
- Readout order: **`diagnose_io_gain` free-running attractor (full 36 steps)** (~30 s, retrain-free on the produced artifact) → `--evaluate --saved` for scored CRPS / MCR / calibration.
- The C-121 guard (`tests/test_rollout_stability_guard.py`) must be green before the retrain (it is).

## 3. Pre-registered predictions
Baselines: **active-seed42 `…233938`** (stability baseline — the explosion, ~log 16) and **frozen-seed42 `…051634`** (skill/calibration reference — healthy: lr_sb CRPS ≈ 0.197 / ns 0.043 / os 0.052).

| Endpoint | Prediction |
|---|---|
| **Stability** (free-running, all 36 steps) | in-range (≲ log 13; vs the active baseline ~log 16); no ratchet |
| **CRPS** (step-wise, lr_sb/ns/os) | healthy O(0.1), ≤ the frozen reference; **uncontaminated** by the stability term |
| **MCR** | not worse than the frozen reference; ideally toward 1 |
| **Calibration / sharpness** | PIT ~uniform, coverage ~nominal; **no mean-hedging collapse** (sharpness/zero-rate not deflated) |

## 4. Falsifiers (pre-committed — any one fires ⇒ MVP rejected, not rescued)
- **F1 — stability fails:** free-running forecast still leaves the data range with B1 on ⇒ pushforward (one-step-back gradient) is insufficient ⇒ escalate to **B2 GTF** (`02 §7.4`) before scaling.
- **F2 — calibration traded away (the dangerous one, C-126):** B1 bounds the rollout but MCR/coverage/zero-rate degrade vs the frozen reference (mean-hedging / blurring) ⇒ point-stability bought at the cost of the distribution ⇒ the **ZITD head (P4 layering)** is needed, not more B1.
- **F3 — truncation blindness (M-RT2 / C-125):** bounded at K=12 but diverges in steps 13–36 ⇒ K too short ⇒ raise K (checkpointed) or re-pre-register.
- **F4 — proper-score corruption (M-RT1 / C-125):** the stability-term weight isn't annealed and the headline CRPS is contaminated ⇒ the optimum is biased off the true predictive distribution ⇒ fix the objective before trusting CRPS.
- **F5 — balancer-conditional:** B1+active only bounds when the balancer is *also* frozen ⇒ B1 doesn't fix the exposure bias under the production setting ⇒ Q4 unresolved, **C-124 stays open**, reconsider.

## 5. Metrics & instrumentation
- **Stability trajectory** — per-step `max|reg|` over 36 steps (the F1/F3 curve), via `free_running_attractor` (the C-121 helper).
- **CRPS** (proper) step-wise, all 36 steps, per head; **uncontaminated** (R1) — report the predictive CRPS separately from the stability regulariser.
- **MCR + PIT + coverage (PICP)** — calibration (F2 guard).
- **Sharpness / zero-rate** — mean-hedging/blurring guard (Shi caution).
- **Gradient norms** (post-clip, R7) + the stability-weight annealing schedule — training-health record.

## 6. Controls
- **`rollout_horizon = 1` parity** — byte-identical to today's training path (`02 §8` invariant); confirms B1 is opt-in and inert when off.
- **Frozen-seed42 arm** (`…051634`) — B1 must not break the already-healthy model; the calibration reference.
- **pink** — a healthy model under B1 stays healthy (no cost where there was no problem).
- **Single-seed caveat (C-112)** — MVP is directional (violet/42); confirm on ≥1 more seed before any adoption claim.

## 7. Decision rules
- **All predictions hold, no falsifier:** B1 succeeds → the C-113 durable fix on the active balancer → **resolves C-124** → multi-seed confirm → promote **ADR-058**. ZITD layering revisited only if calibration needs more.
- **F1 fires:** escalate to **B2 GTF** (un-detach, soft-mix `(1−α)·pred + α·GT`, α-bounded, gradient clipping).
- **F2 fires:** B1 stabilises but the distribution suffers → **P4 layering**: bring in the ZITD softplus head.
- **F3/F4 fire:** config/objective fix (K, annealing) before re-running.
- **F5 fires:** B1 doesn't fix the exposure bias under the production balancer → C-124 stays open; reconsider.
- Any outcome → `07_experiment_log` entry; negatives documented, no ad hoc rescue (session discipline).
