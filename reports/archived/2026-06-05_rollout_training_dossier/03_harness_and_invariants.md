# 03 — Harness & Invariants (what must be in place before B1 runs)

**Date:** 2026-06-06 · **Status:** seeded · **Dossier:** [00_README](00_README.md) · **Design:** [02_design](02_design.md) · **Plan:** [05_analysis_plan](05_analysis_plan.md)

The safety scaffolding for the rollout-training program: what must **never** break, what B1 **intends** to change (and how to change it safely), the harness we run experiments through (most already built this session), and the pre-flight gate before any B1 training run. This is the **last doc gate** before implementing B1 (#78).

---

## 1. Invariants — three kinds

### 1a. HARD invariants — never break (any experiment that violates one is invalid)

| Invariant | Source | Why it holds for B1 |
|-----------|--------|----------------------|
| **`rollout_horizon=1` parity** | this program (`02 §8`) | B1 ships behind `rollout_horizon` (default 1). At K=1 the training path is **byte-identical** to today's one-step path — zero-regression proof, like `feedback_clamp`/`freeze_multitask_balancer`. |
| **Proper score stays the headline** | Gneiting (R1) | CRPS is the predictive metric; the pushforward/GTF **stability term is a weighted, annealed regulariser**, never a replacement for the score. Headline CRPS is reported **uncontaminated**. |
| **No output capping** | ADR-003 / ADR-028 §3 | Stability comes from *training dynamics*, not magnitude clamps. A bounded rollout that only looks bounded because something was clamped doesn't count. |
| **Train-only forcing** | `02 §8` | Pushforward / α affect **training only**; inference stays unchanged free-running — so `diagnose_io_gain` remains a valid before/after probe. |
| **C-121 guard stays green** | #76 | The runaway-detection guard (`test_rollout_stability_guard.py`) must pass; no B1 retrain may regress the detector. |
| **Fail-loud + reproducibility + full suite green** | ADR-003 / ReproducibilityGate (C-42) / ADR-005 | Seed-locked, deterministic, TDD; the suite + register-integrity stay green before any run. |

### 1b. Invariants B1 DELIBERATELY changes (carefully, behind the flag)

| Current behavior | Source | What B1 changes it to | Care required |
|------------------|--------|------------------------|---------------|
| **Prediction→input feedback gradient is SEVERED** | `training_engine.py:200` (`prev_pred = t1_pred.detach()`) | **Bounded gradient flows** through the fed-back operator — pushforward backprops the *last* unroll step (flat memory); GTF (fallback) soft-mixes + α-bounds. | Gradient **clipping** (R7); detach across steps for B1 (flat memory); annealed stability weight (R1). This is the whole point — the operator the runaway rides finally gets training signal. |
| **One-step-ahead objective** | `training_engine._process_sequence` | **+ a K-step rollout stability term** (`L_stability`, pushforward) | Annealed/small weight; CRPS uncontaminated (R1). |
| **Trained horizon = 1 step** | the loss loop | **K-step rollout** (`rollout_horizon=12` candidate) | K must reach the blow-up onset (~step 12); the **readout certifies the full 36 steps**, not just ≤K (R2 / M-RT2). |

### 1c. Constraints to respect *while* changing (don't break in passing)

- **ConvLSTM hidden-state BPTT already flows** (gradient through `h` across the window). B1 adds the *feedback-path* gradient **on top** — do not break the existing `h` BPTT.
- **`ModelOutput` contract** (`reg`/`cls`/`h_next`) — unchanged.
- **36-step free-running inference** — unchanged (train-only forcing).
- **The C-111 active balancer** — B1's primary arm trains *under* it (`05 §0.1`); the balancer interaction is the experiment's subject (Q4), not a thing to silence.
- **`_df`/`_pf` parity, curriculum sampling (ADR-011/012), eval-comparability** — unchanged.

## 2. The standing harness (already built — reuse, don't reinvent)

| Mechanism | Where | Use for B1 |
|-----------|-------|------------|
| **C-121 runaway guard** | `views_hydranet/utils/rollout_diagnostics.py` (`free_running_attractor`, `is_out_of_range`) + `tests/test_rollout_stability_guard.py` | The boundedness regression guard; the shared helper is the readout primitive. |
| **Fast retrain-free readout** | `scripts/diagnose_io_gain.py` (now consumes the helper) | Probe a fresh B1 artifact's free-running attractor (~30 s) before a ~40-min eval. |
| **Default-off feature flags** | `config_initializer.py` (`feedback_clamp_log1p`, `freeze_multitask_balancer`) | Add `rollout_horizon` (default 1 = parity) the same way. |
| **Reproducibility & run discipline** | ReproducibilityGate; conda `views-hydranet-env`; one model/GPU; n ≤ 64; background+notify; timestamped artifacts | All B1 runs. |
| **GPU-enforced driver + ledger** | `views-models/scripts/run_*.sh` (CUDA pre-flight + on-GPU PID verify, `trap`-restore, no `set -e`) + `logs/*_RESULTS.txt` | B1 retrain gets a driver + an `07_experiment_log` entry. |
| **Evaluation comparability** | baselines `…233938` (active exploder, stability baseline) + `…051634` (frozen healthy, calibration ref) + `s0_baseline_metrics.md`; CRPS is *proper* | Score B1 against these on identical metrics. |
| **Negative-result discipline** | `reports/postmortem_*.md` | Falsifications recorded honestly; no ad hoc rescue. |

## 3. New harness B1 REQUIRES (gaps to build first — TDD, before any retrain)

1. **The pushforward training path behind `rollout_horizon`** + a **parity test**: with `rollout_horizon=1`, `_process_sequence` is byte-identical to today (zero-regression proof). **Blocker.**
2. **Feedback-gradient-liveness test (the crown-jewel test):** under B1 (K>1), assert the gradient through the fed-back operator is **non-zero and finite** — i.e. B1 actually trains the operator that is currently severed (`training_engine.py:200`). This is the test that proves B1 does what it claims.
3. **Annealed stability-weight + uncontaminated CRPS (R1):** the `L_stability` weight anneals (→ small); a test that the **headline CRPS is computed without the stability term** so the proper score isn't biased.
4. **Gradient clipping (R7):** applied on the B1 path; a test it's active.
5. **Full-36-step boundedness readout (R2 / M-RT2):** extend the guard / `diagnose_io_gain` to certify **all 36** steps (not just ≤K); ideally a **real-artifact** boundedness check (the remaining C-121 layer) on the produced B1 `.pt`.
6. **Calibration readout (F2 / C-126):** PIT / coverage (PICP) / MCR / **zero-rate + sharpness** wired into the B1 eval — to catch "bounded but mean-hedged/blurred", not infer it from CRPS.
7. **K=12 peak-memory measurement (R4):** measure activation memory at K=12, real batch, 32×32 before committing; `torch.utils.checkpoint` is the plan if it OOMs (only relevant if B2/BPTT is reached — B1 is flat in K).

## 4. Experiment protocol & decision gates

- **One variable:** `rollout_horizon` (the pushforward term) on the active-balancer baseline — never B1 + another change at once.
- **Pre-register, then run:** `05_analysis_plan` is committed *before* execution (done). Acceptance + kill conditions are its falsifiers.
- **Cheap readout before expensive:** `diagnose_io_gain` (attractor in-range across 36?) → only then a full `--evaluate`.
- **Falsifier honesty:** a pre-registered falsifier that fires kills the hypothesis; postmortem it, don't rescue (esp. F2 — don't rationalise a calibration regression).
- **GPU discipline:** one model; n ≤ 64; background+notify; `trap`-restore configs; preserve timestamped artifacts.
- **Magnitude-neutral by construction:** stability must come from the *training objective*, not a clamp.

## 5. Pre-flight checklist (must be green before the FIRST B1 training run)

- [ ] **B1 pushforward path behind `rollout_horizon`; parity at K=1 byte-identical (§3.1)** — *blocker*.
- [ ] Feedback-gradient-liveness test green (§3.2) — the severed gradient is live+finite under K>1.
- [ ] Annealed stability weight; headline CRPS uncontaminated (§3.3 / R1).
- [ ] Gradient clipping applied (§3.4 / R7).
- [ ] C-121 guard green (it is); full-36-step boundedness readout wired (§3.5 / R2).
- [ ] Calibration readout (PIT/coverage/MCR/zero-rate) wired into the B1 eval (§3.6 / F2 / C-126).
- [ ] K=12 peak memory measured (§3.7 / R4).
- [ ] Full suite green; ruff clean; register integrity green.
- [ ] `05_analysis_plan` pre-registered (done); baselines present (`…233938`, `…051634`).
- [ ] GPU up (CUDA pre-flight, C-115); GPU-enforced driver ready.
- [ ] Risk entries current: C-125 (premises), C-126 (calibration), C-129 (ZITD coordination).

> Reassurance: §2 already exists (the C-121 guard, `diagnose_io_gain`, the flag pattern, the GPU driver, the baselines). The genuinely new build is §3.1–3.3 (the pushforward path + the parity & gradient-liveness tests + the annealed-weight/uncontaminated-CRPS discipline) and §3.6 (the calibration readout). That is the real "before you experiment" work — to be sequenced in `04_roadmap`.
