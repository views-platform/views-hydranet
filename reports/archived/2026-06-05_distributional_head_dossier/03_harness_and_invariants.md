# 03 — Harness & Invariants (what must be in place before experimenting)

**Date:** 2026-06-05 · **Status:** seeded · **Dossier:** [00_README](00_README.md)

This is the safety scaffolding for the distributional-head program: what must **never** break, what this program **intends** to change (and how to change it safely), the harness we run experiments through (most of it already built this session), and the pre-flight gate before any training run. The constraints baseline is ported from `reports/paths_forward.md §2`, then re-classified for *this* program.

---

## 1. Invariants — two kinds

The mistake to avoid is treating every current behavior as sacred. Some are hard invariants; some are exactly what we are here to replace. Be explicit about which is which.

### 1a. HARD invariants — never break (any experiment that violates one is invalid)

| Invariant | Source | Why it holds for this program |
|-----------|--------|-------------------------------|
| **Fail-loud, no silent clamping** | ADR-003 | We replace the explosion with a *structural* bound (softplus link), **not** an output clamp. A clamp masks; a link is the model's geometry. If something diverges, detect & fail — don't cap. |
| **Pandas-free output path** | ADR-047 | The head changes what's emitted, not how it's delivered. PredictionFrames stay; the distribution's mean/quantiles flow through the existing `_pf` path. |
| **`_df`/`_pf` parity** | ADR-047 | The diagnostic `_df` path must keep matching `_pf`. New head must not silently diverge them. |
| **Reproducibility** | ReproducibilityGate (C-42) | Seed-locked runs; deterministic algorithms; parameter manifest. Every experiment reproducible. |
| **Stable baseline stays runnable** | this program | The current `log1p`+`expm1`+BCE path must remain the default and pass unchanged — the new head ships behind a default-off flag. No experiment may regress it. |
| **Full suite green + CIC contracts synced** | ADR-005 | TDD; taxonomy; the field-count / contract tests stay green before any run. |

### 1b. Invariants this program DELIBERATELY changes (carefully, behind a flag)

| Current behavior | Source | What we change it to | Care required |
|------------------|--------|----------------------|---------------|
| `expm1` inverse of a point output | `config_initializer.py` (`log1p`/`expm1`) | **No `expm1` of a free output.** Output parameterizes a distribution; forecast = mean/quantiles. `log1p` stays on the **input** side only. | The autoregressive feedback must re-encode `log1p(mean or sample)` to keep input space consistent. |
| ReLU point output in log-space | `HydraBNrecurrentUnet_06_LSTM4.py` (head convs) | Distribution **parameters** (`π, μ, φ, ρ`) via appropriate links (**softplus** for positive μ/φ; sigmoid for π; constrained ρ/p∈(1,2)). | Link choice is the safety lever (softplus ⇒ linear-in-counts, not exponential). Document link per parameter. |
| Separate BCE classifier + shrinkage regressor, combined post-hoc | ADR-020 multi-task | A **single likelihood** whose zero-inflation gate `π` *is* the P(conflict); `E[y]=(1−π)μ` recovers the decomposition. | Removes one head pair; revisit what the MTL balancer now balances (see below). |
| MC-dropout is the *only* uncertainty source | current config | Likelihood head = **aleatoric**; MC-dropout (optional) = **epistemic** (Kendall 2017). | **Not a prerequisite to remove dropout** — they coexist; sampling from the distribution can later replace K dropout passes for the aleatoric part. |

### 1c. Constraints to respect *while* changing (don't break these in passing)

- **36-step autoregressive stability** — the new feedback (`log1p(mean/sample)`) must not reintroduce a runaway; verify with the fast readout (§2) before any full eval.
- **Curriculum / conflict-biased sampling** (ADR-011/012) — unchanged; the loss must behave under non-uniform windows.
- **Multi-task weighting** (`mtloss.py`, Kendall 2018) — with fewer heads, re-examine balancer behavior (ties to the C-111 bisect). Keep `freeze_multitask_balancer` available as a control.
- **Zero-inflation reality** (~95% zeros) — the loss must give *all* cells signal (the hurdle-mask gradient-starvation failure in `paths_forward §1` is the cautionary tale; the ZITD gate avoids it).

## 2. The standing harness (already built — reuse, don't reinvent)

| Mechanism | Where | Use in this program |
|-----------|-------|---------------------|
| **Default-off feature flags** | `config_initializer.py` (e.g. `feedback_clamp_log1p`, `freeze_multitask_balancer`) | Add the head as a new loss option, off by default. |
| **Loss registry (OCP seam)** | `utils.py` `LOSS_REG_REGISTRY` / `choose_loss` | Register the Tweedie/ZITD loss here — **extend, don't modify** `choose_loss`. |
| **Fast retrain-free readout** | `scripts/diagnose_io_gain.py` | Probe a fresh artifact's free-running attractor (~30 s) before spending a ~40-min eval. |
| **Pre-registration** | `reports/preanalysis_*.md` pattern | A hypotheses+falsifiers plan before each run (→ `05_analysis_plan`). |
| **Reproducibility & run discipline** | ReproducibilityGate; conda `views-hydranet-env`; one model/GPU; n ≤ 64; background+notify; timestamped artifacts preserved | All experiments. |
| **Driver + results ledger** | `scripts/run_*.sh` + `logs/*_RESULTS.txt` pattern (config backup + `trap` restore, no `set -e`) | Each experiment gets a driver + an entry in `07_experiment_log`. |
| **Evaluation comparability** | `reports/s0_baseline_metrics.md` (locked baseline); CRPS is *proper* | Score the distribution head against the locked log1p baseline on the same metrics. |
| **Negative-result discipline** | `reports/postmortem_*.md` (e.g. locked-dropout) | Record falsifications honestly; no ad hoc rescue. |

## 3. New harness this program REQUIRES (gaps to build first)

1. **A validated Tweedie/ZITD NLL implementation + its own test suite.** *Implementation-critical* (lit gap #2: Dunn & Smyth density). Tests (ADR-005 taxonomy): known-value checks vs a reference (e.g. `tweedie` R/py package or statsmodels); finite gradients for `p∈(1,2)`; exact-zero handling (the `π`/compound-Poisson zero mass); NaN/Inf guards on `μ,φ`; behavior at `φ→0`, `μ→0`, large `μ`. **No training run until these pass.**
2. **A sampling path for the posterior.** Ancestral sampling from the predicted distribution (for forecasts and for autoregressive feedback `log1p(sample)`), with tests that samples are non-negative, finite, and recover the analytic mean in expectation.
3. **A calibration harness.** With a real predictive distribution we can (must) check **coverage vs nominal** and **PIT/rank histograms**, not just CRPS — to catch "bounded but mis-calibrated." (Pairs with the lit §5 scoring refs; observation-error-aware per Bessac/Weijs.)
4. **Link-not-clamp discipline encoded.** The positivity/boundedness comes from the *link* (softplus), and that is explicitly **not** the ADR-003-forbidden clamp. State this in the design + a test that the emitted forecast is never hard-capped (only link-shaped).
5. **NLL numerical guards.** Detect NaN/Inf in the loss (mirror the input-side guards in `feature_scaler.py`) and fail loud — a divergent NLL must surface, not poison weights silently.
6. **Parity guard for the new path** — a test that with the head **off** (default), outputs are byte-identical to the current baseline (zero-regression proof, as done for `feedback_clamp`/`freeze_multitask_balancer`).

## 4. Experiment protocol & decision gates

- **One variable at a time.** Change the loss/head OR the link OR the sampling — never several at once (the bisect lesson). Each behind its own flag.
- **Pre-register, then run.** Hypothesis + risky prediction + falsifiers in `05_analysis_plan` *before* execution. State the acceptance metric and the kill condition.
- **Cheap readout before expensive.** `diagnose_io_gain` (attractor in-range?) → only then a full eval. Don't burn a 40-min eval to learn what 30 s tells you.
- **Falsifier honesty.** A pre-registered falsifier that fires kills the hypothesis; document it (`postmortem`/`07_experiment_log`), don't rescue.
- **GPU discipline.** One model at a time; n ≤ 64; background + notify; preserve prior artifacts (timestamped); restore configs via `trap`.
- **Magnitude-neutral by construction.** Improvements come from better *representation/likelihood*, not from capping outputs. If a result only looks good because something got clamped, it doesn't count.

## 5. Pre-flight checklist (must be green before the FIRST distributional-head training run)

- [ ] Tweedie/ZITD NLL implemented + unit tests green (§3.1) — **blocker**.
- [ ] Loss registered via `LOSS_REG_REGISTRY` (OCP); `choose_loss` unmodified.
- [ ] New head/loss behind a default-off flag; baseline byte-identical with it off (§3.6).
- [ ] Sampling path + tests (§3.2); autoregressive feedback uses `log1p(sample/mean)`.
- [ ] NLL NaN/Inf guards (§3.5); fail-loud (ADR-003) preserved — no clamp.
- [ ] Full suite green; CIC contracts synced; ruff clean.
- [ ] `05_analysis_plan` pre-registered for the run (hypothesis, falsifiers, metrics vs `s0` baseline).
- [ ] `diagnose_io_gain` adapted to read the distribution's mean (so the fast readout still works).
- [ ] Risk-register entries for the new failure modes (Tweedie density instability; sampling overflow; calibration drift).

> Reassurance: items in §2 already exist; the genuinely new build is §3.1 (the Tweedie loss + tests) and §3.2–3.3 (sampling + calibration). That is the real "before you experiment" work — sequenced in [`04_roadmap`](04_roadmap.md).
