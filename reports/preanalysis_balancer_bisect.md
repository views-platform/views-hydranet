# Pre-Analysis Plan — C-111 Balancer-Freeze Bisect (C-113)

**Date:** 2026-06-04 (pre-registered *before* running)
**Branch (views-hydranet):** `fix/variational-dropout-autoregressive-stability`
**Builds on:** the reframe that C-113 is a **regression**, not a flaw — the model ran stably for years on log1p and the explosion first appeared on the **post-C-111 retrain (2026-06-03)** (`memory: project-explosion-is-regression`); the diagnostic (`reports/results_io_gain_diagnostic.md`) showing violet's free-running input→output map settles at log ~40 (pink ~10).
**Risk:** C-113

---

## 1. Hypothesis

**H:** The C-113 explosion is caused by the **C-111 change** — adding the `MultiTaskLoss` balancer's `log_vars` to the optimizer (`training_engine.make()` lines 101–104). For years the `log_vars` were frozen at their zero init (Kendall-Gal homoscedastic weighting ≡ equal weights), and the model was stable. Un-freezing the balancer this session may have been **load-bearing**: an active balancer can drive a task's effective weight and destabilise training for some seeds (the explosion is seed-dependent: pink-4 fine, violet-42 / blue diverge). If so, **freezing the balancer again removes the explosion** — and the cause is the training objective, not the recurrence, the transform, or the architecture.

## 2. Design — 2-arm bisect on violet_visitor (same seed, data, code; only the balancer freeze differs)

| Arm | `freeze_multitask_balancer` | balancer `log_vars` | role | prediction |
|-----|:--:|----|------|-----------|
| **CONTROL** | `False` (C-111 default) | trainable | reproduce the bug under current code | free-running attractor **out-of-range (~log 40)** |
| **TEST** | `True` (pre-C-111) | frozen at 0 (equal weights) | the fix candidate | free-running attractor **in-range (~log 10)** if H holds |

Both arms: full retrain from scratch, identical config except the flag, `ReproducibilityGate` seed locked. The CONTROL retrain rules out "a fresh retrain / CUDA non-determinism alone changes the outcome" — without it, a non-exploding TEST arm could not be attributed to the freeze.

**Only the balancer is frozen.** The ADR-055 Tobit `sigma` params (added separately at `make()` lines 83–93) stay trainable — this isolates the C-111 change and does *not* confound with the Tobit likelihood.

## 3. Readout — retrain-free diagnostic (fast), eval as confirmation (slow)

Primary readout per fresh artifact: **`scripts/diagnose_io_gain.py`** — the free-running rollout attractor level vs the data range. It was validated against the real eval (violet→log 40 ⇄ CRPS 1e17; pink→log 10 ⇄ healthy), runs in ~30 s, and needs no eval. This replaces two ~40-min evals.

- **In-range** if the rollout settles `≲ log 13` (within the data range; `expm1` → counts in-distribution).
- **Out-of-range / pathological** if it settles `≳ log 20` (→ `expm1` astronomical).

**Confirmation:** if the TEST arm reads in-range, run one full `--evaluate --saved` on it to confirm `lr_sb_best/CRPS` is genuinely in-range (≈ pink's ~0.1), not just the synthetic-seed proxy.

## 4. Pre-registered predictions

- **CONTROL (active):** attractor out-of-range (~log 40), reproducing the explosion under current code.
- **TEST (frozen):** attractor **in-range (~log 10)** ⇒ H corroborated; C-111 is the cause.

## 5. Falsifiers (pre-committed)

- **F1 — balancer is NOT the cause:** TEST arm *still* settles out-of-range (~log 40) despite the frozen balancer ⇒ C-111 is not the (sole) driver. Next bisect targets: Tobit loss (ADR-054), scheduled sampling (ADR-056), per-target sigma (ADR-055), or seed.
- **F2 — control fails to reproduce:** CONTROL arm comes out in-range (no explosion) ⇒ the bug is not deterministically reproducible from a clean retrain (run-to-run variance / the June-3 explosion was partly stochastic) ⇒ the whole bisect logic is undermined; re-think (multiple seeds needed).
- **F3 — diagnostic/eval disagree:** TEST reads in-range on the diagnostic but the confirmatory eval still explodes ⇒ the synthetic-seed proxy is not faithful for this artifact ⇒ trust the eval, re-examine the diagnostic.

## 6. Skepticism

1. **Co-suspects.** C-111 is the *lead* suspect (it specifically changed training dynamics and the explosion appeared on its retrain), but Tobit / scheduled sampling / per-target sigma also landed recently. A clean TEST result implicates the balancer; a null result (F1) does not exonerate the others — it redirects the bisect.
2. **"The bug was load-bearing" is a hypothesis, not a fix.** If freezing the balancer works, C-111's *intent* (a learnable balancer) is still valid — the real fix may be to **regularise/constrain** the balancer (bound `log_vars`, lower its LR, or weight-decay it), not to abandon it. Freezing is the diagnostic, not necessarily the shipped fix.
3. **CUDA non-determinism.** Training is not bit-reproducible even with locked seeds; hence the CONTROL arm (F2 guards this). Single seed only — a null bisect would need multiple seeds before concluding.
4. **Diagnostic is a synthetic-seed proxy** (mitigated: it matched the real eval for both pink and violet; F3 + confirmatory eval guard this).

## 7. Method

- **Model:** violet_visitor (the clean exploder). Same artifacts/data; `ReproducibilityGate` locked.
- **Only variable:** `freeze_multitask_balancer ∈ {False (control), True (test)}`.
- **Train invocation:** `bash models/violet_visitor/run.sh --train --run_type calibration` (one model on GPU at a time; ~2h20 each).
- **Preserve** the June-3 artifact (new trains are timestamped; do not delete).
- **Readout:** `diagnose_io_gain.py <new_artifact>` after each train; confirmatory eval on TEST if in-range.
- **Safety:** flag inserted via config (trap-restore of the config); default `False` ⇒ zero behavior change for all other models/tests; training path otherwise byte-unchanged; TDD with full suite + ruff green before any retrain.

## 8. Disposition rules

- **TEST in-range + CONTROL out-of-range (H corroborated):** C-111 balancer is the cause. Record as the root cause of C-113. Next: design the *real* fix (regularise the balancer — bound/decay `log_vars` or lower its LR) rather than freeze-and-forget; re-evaluate whether the spectral-norm/pushforward/clamp program is even needed.
- **F1 (TEST still explodes):** balancer exonerated; redirect bisect to Tobit / SS / sigma (one flag at a time, same protocol).
- **F2 (CONTROL doesn't reproduce):** escalate to multi-seed; the explosion may be stochastic-onset.
- Any outcome documented honestly (as with the dropout postmortem); no ad hoc rescue.
