# Pre-Analysis Plan — `freeze_h` Channel-Isolation Ablation (C-113)

**Date:** 2026-06-04 (pre-registered *before* running — Popperian discipline, as with the dropout postmortem)
**Author:** autonomous overnight run (Claude), executing on user's explicit instruction
**Branch (views-hydranet):** `fix/variational-dropout-autoregressive-stability` (LockedDropout wiring live; held constant across all arms)
**Companion to:** `reports/options_catalogue_autoregressive_stability.md`, `reports/postmortem_locked_dropout_negative_result.md`, ADR-027, ADR-028
**Risk:** C-113 (autoregressive recurrent runaway)

---

## 1. Question

Where does the autoregressive divergence in `violet_visitor` enter — through the **recurrent hidden/cell state**, or through the **prediction→input feedback loop**?

## 2. Motivating observation (the reason this ablation exists)

All three golden_hour configs run `freeze_h="hl"`. Under `"hl"`, the long-term cell state `hl` is **frozen** at its post-digest value for all 36 autoregressive steps, and the short-term state `hs = o ⊙ tanh(hl_internal)` is **tanh-bounded** (`|hs| < 1`). Therefore, under `"hl"`, the *entire recurrent hidden state `h_tt` is provably bounded across the rollout*.

Yet `violet_visitor` still exploded under `"hl"` (I2: step-wise `lr_sb_best/CRPS = 2.13e17`, n=16, LockedDropout active). If the recurrent state is bounded and the output still diverges, the divergence cannot be flowing through the recurrent state. The only remaining unbounded feedback path is **prediction → next-step input `x`** (the post-ReLU prediction, unbounded above, fed back). 

> Note on the in-code guard: the `Autoregressive drift` warning (log-space `|pred| > 100`) fired **0 times** in the exploding I2 baseline, because the divergence is `expm1`-amplified — log-space ≈ 40 → raw ≈ 2e17 sails under a log-space threshold of 100. **The guard log is therefore not a usable signal here.** The authoritative signal is the `wandb:` run-summary CRPS block.

## 3. Hypothesis

**H1:** The divergence is driven by the prediction→input feedback loop (C1), *not* by either recurrent state channel.

Decomposition of the across-step feedback channels:
- **C1** — input loop: `prediction → x_next`. Always live (disabling it would change the model's input; out of scope).
- **C2** — short-term hidden `hs` update/carry.
- **C3** — long-term cell `hl` update/carry.

`freeze_h` toggles C2 and C3; C1 is always on. Full 2×2 over (C2, C3):

| Arm | C2 (`hs`) | C3 (`hl`) | Prediction (H1) | Prediction (¬H1, recurrent-driven) |
|------|:-:|:-:|---|---|
| `none` | ✓ | ✓ | EXPLODE | EXPLODE |
| `hl` (baseline) | ✓ | ✗ | EXPLODE *(known: 2.13e17)* | EXPLODE |
| `hs` | ✗ | ✓ | EXPLODE | EXPLODE (via C3) |
| `all` | ✗ | ✗ | **EXPLODE** *(risky)* | **BOUNDED** |

The **risky prediction** that separates the hypotheses is the `all` arm: H1 predicts it *still explodes* despite a fully frozen recurrent state; the recurrent-driven hypothesis predicts it *becomes bounded*.

## 4. Pre-registered decision rules

Primary endpoint: `step-wise/lr_sb_best/CRPS` (the head that exploded to 2.13e17). Healthy reference (pink_pirate): ≈ 0.13.

- **BOUNDED:** `< 1.0`  (≈ 1 order above healthy at most)
- **EXPLODED:** `> 1e3` (≥ 4 orders above healthy)
- **AMBIGUOUS:** `[1, 1e3]` → report as ambiguous, do **not** over-claim; flag for follow-up.

Secondary endpoints recorded for all arms: `lr_ns_best`, `lr_os_best` (step-wise *and* time-series-wise CRPS), for completeness and cross-head consistency.

**Interpretation:**
1. **`all` EXPLODED** → C1 (input loop) alone diverges with the recurrent state fully frozen → **H1 corroborated**. The recurrent state is not the driver. `freeze_h` is confirmed useless against this failure. Fix must target the **input→output map**: spectral-norm/Lipschitz on the input-to-hidden convs `Wx*` *and* the U-Net encoder/decoder, and/or pushforward training, and/or an in-domain feedback-input clamp (magnitude-neutral on the emitted output).
2. **`all` BOUNDED while `hl` EXPLODED** → the *difference is C2* (`hs` update) → divergence rides the short-term hidden channel → **H1 refined/partly falsified**; recurrent-path spectral control matters after all.
3. **`none` ≫ `hl`** → C3 (`hl` update) amplifies the runaway → cell-state damping (ADR-028 §2) has a role.
4. **`hs` vs `all`** isolates C2's contribution when C3 is live.

**Reproduction control:** `hl` arm must reproduce the I2 baseline (`lr_sb` ≈ 2e17, within ~1 order) or the harness is suspect and conclusions are void.

## 5. Method (single-variable, config-only — NO code change)

- **Model:** `violet_visitor` only (the cleanest exploder; pink stays healthy, blue not run per prior user instruction). Artifact: `calibration_model_20260603_180015.pt`.
- **Only variable:** `freeze_h ∈ {hl, all, hs, none}` in `models/violet_visitor/configs/config_hyperparameters.py:44`. Everything else (weights, LockedDropout, `n_posterior_samples=16`, detach law, seed) held constant.
- **Invocation per arm:** `bash models/violet_visitor/run.sh --evaluate --run_type calibration --saved --report` (canonical eval path; matches `run_golden_hour_eval.sh`).
- **Arm order:** `hl` (validate harness + reproduce baseline) → `all` (KEY) → `hs` → `none`. If interrupted after 2 arms, `hl`+`all` already answer the core question.
- **One model at a time** on the 8 GB GPU (sequential; satisfies the GPU constraint). `n=16` ≤ 64.
- **Detection (headless):** parse the `wandb:` CRPS summary block from each arm's tee'd log. Drift-guard log is NOT used (see §2 note).
- **Safety:** config backed up; `trap … EXIT` restores `freeze_h="hl"` even on crash/kill. No `set -e` (one arm crashing must not abort the others or the restore).

## 6. Threats to validity (pre-registered)

- **Single model.** Localizes the channel for *violet*; blue not run, so the *universal* claim is not tested. Stated as a limitation.
- **`all` also freezes `hs`** → if `all` explodes, it shows C1 *alone* (with static state) suffices; it does not by itself quantify C2/C3 contributions — that is what `hs`/`none` add.
- **LockedDropout active in all arms** — a constant, not a confound for the freeze_h contrast; but the absolute magnitudes are specific to the locked-mask code path (continuous with I2, not with the June-3 per-step n=4 numbers).
- **Aggregate CRPS, not per-step trajectory.** Per-step max|pred| is not reliably logged (see §2). The channel-localization conclusion rests on final aggregate CRPS per arm, which is binary-robust (bounded vs exploded by ≥4 orders). The per-step growth shape is already established (June-3: step-1 normal → step-12 catastrophic).

## 7. Outputs

- Per-arm logs: `views-models/logs/ablation_freezeh_<arm>_<ts>.log`
- Consolidated results: `views-models/logs/ablation_freezeh_RESULTS.txt`
- Results write-up + catalogue update authored by the run on completion: `reports/results_freezeh_ablation.md` (verdict against §4) and a correction to `reports/options_catalogue_autoregressive_stability.md` (Axis-0 quantity + Axis-A target).
