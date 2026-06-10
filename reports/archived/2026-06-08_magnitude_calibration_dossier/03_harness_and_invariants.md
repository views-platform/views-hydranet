# 03 — Harness & Invariants

**Date:** 2026-06-08 · **Status:** seeded · **Dossier:** [00_README](00_README.md)

Much of the harness already exists (the ZITD/rollout dossiers built it). This file maps what to
**reuse**, the invariants candidate #1 must respect, and the pre-flight gate. Detailed harness
philosophy → `../2026-06-05_distributional_head_dossier/03_harness_and_invariants.md` (by pointer).

## 1. Invariants

### 1a. HARD — never break
| Invariant | Source | Why |
|-----------|--------|-----|
| **`gate_emitted_by_prob=False` parity** | this program | Flag off ⇒ emitted stack byte-identical to baseline (zero-regression proof). |
| **Feedback isolation** | `02 §1b` | The gate touches the **emitted** copy only; the autoregressive **feedback stays ungated** (`hydranet_inference.py:241`). Gating feedback could reinforce collapse. |
| **No output capping / no clamp** | ADR-003 / ADR-028 §3 | Calibration comes from the objective + the probability gate, never a magnitude cap. |
| **Proper score is the headline; MCR is a diagnostic** | Gneiting (`02b`) | Judge on twCRPS + Coverage. MCR reported, never optimized. |
| **One variable per arm** | `02b` M-MC4 | Arm ladder; `loss_reg_sigma` and `total_lessons` pinned. |
| **Fail-loud + reproducibility + full suite green** | ADR-003 / ReproducibilityGate / ADR-005 | Seed-locked, deterministic, TDD; suite + register-integrity green before any run. |

### 1b. DELIBERATELY changed (behind the flag)
| Current | Changes to | Care |
|---------|-----------|------|
| Regression loss on ALL cells (Tobit) | **positive-cell-only** (lognormal, hurdle mask) | config-only; classifier head untouched. |
| Magnitude emitted ungated | **P(positive) × magnitude** (emitted only) | new `_gate_emitted`; feedback untouched. |

### 1c. Respect while changing
`_df`/`_pf` parity; `ModelOutput` contract; 36-step free-running inference (gate is emit-time, not a
training/feedback change); curriculum sampling; classifier head + its focal loss.

## 2. Standing harness to reuse (already built)
- **Default-off flag pattern** — `config_initializer.py` (`feedback_clamp_log1p` :130, `freeze_multitask_balancer` :141, `rollout_horizon` :150). New `gate_emitted_by_prob` follows verbatim.
- **Parity tests** — `tests/test_feedback_clamp.py` (flag-off byte-identity), `tests/test_balancer_freeze.py` (toggle), `tests/test_cluster_e.py` (the positive-only hurdle branch is already tested).
- **Proper-score metrics** — `views-evaluation/.../metric_catalog.py`: **twCRPS** (threshold=0.0) and **Coverage** (alpha) already implemented. No new metric code for the judge metrics.
- **MCR diagnostic** — `native_metric_calculators.py:107-122` (read-only readout).
- **Saved-prediction loader** — `views-pipeline-core/.../managers/prediction/io.py:47-109` (PF `y_pred (N,S)`) for Phase 0 probes (ii)/(iii)/(iv), no retrain.
- **wandb training logs** — `mtl_log_var/*`, `sigma/*` (`training_engine.py:654-671`) for Phase 0 probe (i), no retrain.
- **Cheap rollout readout** — `scripts/diagnose_io_gain.py` + the C-121 guard for `F-rollout-interaction`.

## 3. New harness candidate #1 requires (TDD before any retrain)
1. `_gate_emitted` + its **flag-off parity test** (byte-identity) — *blocker*.
2. **Feedback-isolation test** — assert the feedback tensor is the **ungated** magnitude (the likely bug).
3. Extend `test_cluster_e.py` to assert the positive-only mask path for `lognormal_nll + hurdle_threshold=0`.

## 4. Gaps (deferred)
- **PIT** — does not exist in views-evaluation; would be new. **Not a candidate-#1 blocker** (twCRPS +
  Coverage cover the calibration judgment); nice-to-have for the ZITD escalation.

## 5. Pre-flight checklist (green before the FIRST candidate-#1 retrain)
- [ ] Phase 0 diagnosis run (probes i–iv); decision table filled (`05`/`04`).
- [ ] `_gate_emitted` + flag-off parity test green (§3.1).
- [ ] Feedback-isolation test green (§3.2).
- [ ] `test_cluster_e.py` extended green (§3.3).
- [ ] Judge metrics pinned: twCRPS(threshold=0.0) + Coverage(alpha=baseline's); MCR diagnostic only.
- [ ] Baseline pinned: today's 40-lesson calibration model (`calibration_model_20260608_165326.pt`, eval `ffldgbxf`); `loss_reg_sigma=0.9` and `total_lessons` matched.
- [ ] Full suite green; ruff clean; register integrity green.
- [ ] GPU healthy (CUDA pre-flight; the 2026-06-08 Xid fault is cleared).
