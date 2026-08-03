# Results — Axis-0 Diagnostic: input→output gain / free-running attractor (C-113)

**Date:** 2026-06-04
**Script:** `scripts/diagnose_io_gain.py` (retrain-free; loads saved artifacts; CPU)
**Companion to:** `reports/results_freezeh_ablation.md`, `reports/options_catalogue_autoregressive_stability.md`
**Verdict:** Mechanism confirmed and **sharpened**. Violet's free-running input→output map drives the log-space prediction to an **out-of-range attractor (~log 40)** that `expm1` turns into ~1e17 — pink's settles in-range (~log 10 → ~2e4). It is **state-independent** (frozen state reaches the same level), and it is **not** a simple `‖J‖₂>1` instability.

---

## 1. What was measured (pink_pirate vs violet_visitor, 32×32, h fixed/evolving)

- **Part A — local operator gain** `‖∂reg/∂x‖₂` (top singular value of the one-step Jacobian) via power iteration, across a sweep of in-range inputs.
- **Part B — free-running rollout** `x_{t+1} = model(x_t, h).reg` from synthetic in-range seeds, recording the **log-space level** the prediction settles at vs the data range (log1p max ≈ 12.1), under `freeze_h='none'` (state evolves) and `'all'` (state frozen).

## 2. Results

**Part A — local operator norm does NOT discriminate.** Both models have `‖J‖₂ > 1` across the input range; pink is sometimes *higher* (at constant-fill 12: pink 2.0, violet 1.5). A ReLU map with local gain >1 can still have a bounded attractor. **So the single-point Jacobian gain is the wrong diagnostic — correcting my own earlier Axis-0 proposal.**

**Part B — the free-running attractor level discriminates cleanly** (max-cell log-space level; `expm1` = raw-count equivalent):

| Model | freeze_h | step1 | step12 | step24 | step48 | expm1(final) | verdict |
|-------|----------|------:|-------:|-------:|-------:|-------------:|---------|
| pink | none | ~3 | ~16 | ~13 | **~10** | ~2.4e4 | healthy (in-range) |
| pink | all  | ~3 | ~16 | ~9  | **~8**  | ~3e3   | healthy (in-range) |
| **violet** | none | ~3 | ~28 | ~47 | **~40** | **~1e17** | **PATHOLOGICAL** |
| **violet** | all  | ~3 | ~32 | ~48 | **~37** | **~1e15–1e17** | **PATHOLOGICAL** |

(3 synthetic seeds each `U[0,s]`, s∈{0.5,1,2}; values are representative.)

## 3. Reading

1. **Reproduces the real explosion from synthetic seeds.** Violet's free-running map settles at log-space ~40 → `expm1(40) ≈ 2.4e17`, matching the observed eval `lr_sb_best/CRPS = 2.13e17` in order of magnitude. Pink settles at ~10 → `expm1 ≈ 2e4`, matching pink's healthy metrics. **The divergence is a property of the trained weights' input→output map, robust to the seed** (all violet seeds diverge; no pink seed does) — which substantially de-risks the "synthetic input" caveat.
2. **State-independent.** Violet's `all` rollout (entire recurrent state frozen) reaches the same ~log 40. This independently reproduces the `freeze_h` ablation conclusion in a controlled standalone setting: **the recurrent state is not the driver; the prediction→input loop is.**
3. **Not a spectral-radius / `‖J‖₂>1` instability.** Both maps are locally expansive (>1); the difference is *where the free-running dynamics settle*. Pink's attractor is **inside** the data range (~log 10–12); violet's is **far outside** it (~log 40), i.e. 3–4× the data range in log space → astronomically above it after `expm1`. The disease is an **out-of-range fixed point of the feedback map**, not raw local gain.

## 4. Corrected diagnostic + implications for the fix

- **Axis-0 — corrected quantity (again).** The cheap retrain-free discriminator is the **free-running rollout attractor level vs the data range** (run `diagnose_io_gain.py` Part B), *not* the recurrent spectral radius and *not* the local input→output operator norm. Both of those fail to separate pink from violet.
- **★ In-domain feedback clamp is now the lead near-term fix — and may be retrain-free.** The pathology is the prediction *ratcheting* past the data range over steps (step1 in-range ~3 → ratchets to ~40). Clamping the **fed-back copy** of the prediction to the training input domain (`t0_autoreg = t1_pred.detach().clamp(max≈log1p(data_max))`) breaks the ratchet at the source. It is:
  - **magnitude-neutral on the emitted output** (the clamp applies only to the input copy fed to the next step; `acc_magnitudes` still stores the unclamped prediction),
  - **inference-only / testable with zero retrain** (one line in `predict()`),
  - arguably *reduces* train/inference mismatch (training never saw inputs at log 40 either).
  This contradicts the catalogue's "every fix needs a retrain" — there is a surgical inference-only candidate worth testing first. **Cheap next experiment:** re-evaluate violet with a clamped feedback input and check whether `lr_sb` returns in-range.
- **Spectral-norm / pushforward (Axis A/B)** remain the durable retraining options (lower the attractor by constraining the input→output path / training the map to stay in-range), but they are now *second* to the clamp for a first attempt, since pink demonstrates that local gain >1 is tolerable — the target is the attractor *level*, not the gain.
- **ADR-028 §2 cell-state clamp** stays pre-falsified (state-independent, per §3.2).

## 5. Honesty notes

- **Synthetic operating points** (no data pipeline). Mitigated by: divergence reproduced for all violet seeds and no pink seed, at the *correct magnitude* and with the *correct state-independence* matching the real-data eval — strong evidence the synthetic rollout captures the real mechanism.
- **Part A used `h` fixed at init**; Part B used both evolving and frozen `h` with identical conclusions, consistent with the ablation.
- **Spatial size 32×32** (training `window_dim`); the conclusion is a property of the conv weights and is not expected to be size-sensitive for the dominant mode, but this was not swept.
- BN in eval (running stats), dropout off — the deterministic map, which is the correct object for a gain/attractor analysis.
