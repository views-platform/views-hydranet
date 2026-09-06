# 02b — S2 design review: rulings

**Date:** 2026-09-06 · **Issue:** #327 · **Panel:** 6 independent seats (Shi, Hochreiter, Gneiting,
Murphy, Hyndman, operational), run in isolation. **Verdict: BUILD, but not yet, and not as designed.**

## What the panel corrected in this dossier

| # | claim | status |
|---|---|---|
| 1 | *(C) keeps horizon coherence; it removes the real trade* | **FALSE.** Shi's forecaster is a **deterministic** state recursion ⇒ `p(y₁..y₃₆|x) = Π p(yₖ|θₖ(x))`, identical to (B). **Both are marginal.** Corrected in `02`. |
| 2 | *M41's dial was measured on the 40-lesson vehicle and is open* | **FALSE, and it was in an accepted ADR + CIC.** M41 **is** at L=300; it **closes** the dial (no interior optimum; the hard clamp is significantly better at h6). **C-303's 13th occurrence.** Corrected. |
| 3 | *TFT is absent from the library* | **FALSE.** `Lim2021_TemporalFusionTransformers` + `Lim2020` (the canonical direct-vs-iterative taxonomy) are held with claims. |
| 4 | *"direct multi-horizon is the cell clamp taken to its limit"* | **Mechanically wrong.** The clamp holds the cell **while the hidden half keeps evolving** (M50: 0.98→0.93 clamped vs 0.98→0.56 free). (B) stops both. |
| 5 | *deleting exposure bias is the mechanism* | **Undercut by M60**: rolling the fed **input** moved the field **0/26**; rolling the cell **26/26**. The defensible claim is narrower — **the 36-step autonomous dynamic is deployed and never trained**. |
| 6 | *the 8 HydraNets are …* | Two were wrong: `light_strider`/`white_ranger` are climatology baselines. Roster is `ROSTER` in `test_roster_conformance.py`. Corrected on views-models #440. |

## Architecture facts the design assumed wrongly — all verified in code

* **The four ConvLSTM cells are PARALLEL, not stacked** (`:569-600`; each reads the same `x` and its own `hs_i`), at **full 180×180**, with `pool0/pool1` **outside** the recurrence ⇒ **state propagates one grid cell per month.** Shi's §3.1 has a 3-layer *resolution hierarchy* precisely so one step moves information far. **That property does not exist here, and it is the main thing his paper offers.**
* **Training crops are 32×32** (`window_dim`), deployment 180×180. A K=36 full-res forecaster reaches **36 cells — wider than the training crop**; long-horizon dynamics would be learned from boundary.
* **`total_hidden_channels: 32` ⇒ 4 channels per cell.** Shi's carried 64/192/192.
* **Line 604** gives the U-Net only `hs`; the cell reaches the decoder solely through `o_t ⊙ tanh(hl)`. Under (B) the state is frozen ⇒ **one `hs`, read 36 times, identical** — and M54/M60 say the **cell** carries the map. **(B) would be strictly less informed than its own control**, since under the clamp `hs` still varies because `x` still varies.

## Rulings

**F0 — build (C) behind a flag that degrades to (B).** Not because (C) buys coherence (it does not) but because it is an **ablation rather than a bet**, and because the honest hypothesis is *training the deployment regime*. Put the forecaster's recurrence at **/2 resolution**, not full-res: 36 steps × 3×3 at /2 reaches 72 cells and costs 4× less. Feed it **zeros** (Shi Fig. 1). **K curriculum 1→36.**

**F1 — FiLM-style conditioning, never a new input channel.** Forced: `input_channels == 3*output_channels + len(static_channels)` (`config_initializer.py:427`) plus the Checksum Law; a horizon channel violates both, and `static_channels` requires `f(GridGeometry)->[H,W]` so a per-forward index cannot ride that seam. Condition on a smooth basis of `k` (not a free `nn.Embedding`, which would *manufacture* horizon incoherence), **zero-initialised**, applied in **link space** (pre-`softplus`/`sigmoid`) with **separate coefficients for gate and body** — the horizon effect belongs in the occurrence logit and the dispersion, not the magnitude.

**F2 — give the decoder both halves, and TEST IT FIRST on the current recursive model.** ~10-line subclass, paired seeds, ~8 GPU-h. It is S2's "settle it with a measurement", it de-risks the forecaster's readout, and **if it moves AP@h36 it is a cheaper result than this entire epic.**

**F3 — gradient reaches `h_origin`, with `mean` (never `sum`) over horizons.** (B) is a **sum of K paths, not a product** ⇒ worst case linear in K; `O(e^{λT})` does not apply. (C) is a real K-step chain but **shorter than the ~383-step undetached encoder chain this repo already trains** (`training_engine.py:501`; measured reach 1.6e-02 at 118 steps). **(C)'s risk is vanishing, not exploding** — M50's 41× forward drain attenuates backward on the same path.

**F4 — uniform horizon subsampling with a mean reduction is unbiased.** Biased under: non-uniform π without 1/π, contiguous non-wrapping blocks (h1 gets 1/31, h18 gets 6/31), or weights not matched to sampling. **Under (C), supervise every step** (deep supervision) or the only route from h36 to the origin is the vanishing product. **Never add horizons as `MultiTaskLoss` tasks** — that would be 216 learned `log_vars`, and C-312 already bit us.

**F5 — REJECT per-intensity weighting. Per-horizon weighting approved at equal weights.** Multiplying a proper score by an **outcome-keyed** weight is improper by theorem (`Gneiting2011` C-53); the minimiser is the tilted density `g ∝ w·f`. Shi needs it because **MSE is improper to begin with**; we have a likelihood. **And this repo has already run it twice and lost**: `body_supervision='active'` is exactly an outcome-keyed likelihood weight (blew crps-all on `ns` 0.08 → **24.6**), and `truncated_nb` cost **−0.2376 AP@h18**, 4/4 seeds. The propriety-preserving route is **importance-corrected sampling on covariates, never on realised `y`**.

**F6 — the ruler is structurally blind to what is being surrendered.** Every score is a per-horizon marginal; `assert_sample_cube` checks only `(N,S)`, `S≥2`; **nothing anywhere asserts horizon k came from step k**, and the ensemble combiner contracts on `PredictionFrame`, so a direct member is invisible through to the pooled ensemble. S5 needs: support identity; width identity; a **contents-held-fixed byte-identity test**; and a **multivariate proper score** (variogram/energy) — **which the library does not contain at all.**

## Blockers the epic did not carry — all verified

1. **C-218:** `partition_audit.py:143` **refuses to score** any arm whose config lacks `rollout_feedback: 'sample'`. A direct arm has no feedback ⇒ it must lie in its config or be scored `diagnostic_only=True`, which means *"not deployed skill"*. **S5 must settle this before GPU.**
2. **ADR-027 §1 mandates recursion** (*"For t>0, the model input Xₜ is the prediction ŷₜ₋₁"*). **An amendment is required, not optional.**
3. **`HydraNetConfig` sets `extra = "allow"`** ⇒ a misspelled key is silently accepted and unvalidated. A reject-if-unwired validator is **mandatory** (C-324).
4. **`freeze_recurrent` is in ZERO production configs.** The +0.0591 still has not shipped.

## Power — the pre-registered MDE does not hold

M65's paired sd (0.0075) came from an **emit-only** flag on **one** artifact. A new architecture is a training-time treatment whose arms cannot share weights, so the pairing that bought that number is gone (M65's own caveat says so). Honest figure ≈ the unpaired **0.0334**. And the calculation is **circular**: it is sized against +0.0367, the clamp gain the control now *contains*. State a minimum interesting effect **against the clamped arm** (≈ +0.02) — which needs **~6 seeds ≈ 48 GPU-h** — or pre-declare UNDERPOWERED.

**Single-variable discipline:** the build changes four things at once (architecture, horizon covariate, forking sampler, loss weights). **Freeze the loss to the incumbent's for the headline arm.**

## Revised sequence

0. **views-models #440** (corrected roster) — the clamp still has not shipped.
1. **The no-self-feedback emit probe, ~1 GPU-h**: `freeze_recurrent='cell'` + a feedback transform holding a constant field for all 36 steps — **direct multi-horizon's inference semantics exactly**, on existing artifacts. The seam exists (`FEEDBACK_TRANSFORMS`). **If it does not beat the clamped control, the remaining claim is training-time alignment only, and the epic must be re-argued on that narrower ground.**
2. **PROBE-1, zero training**: the horizon reach curve `‖∂L_k/∂h_origin‖` for k=1..36. Pre-registered: ratio(36/1) > 1 ⇒ (C) needs a bound before a single lesson; < 1e-3 ⇒ (C)'s advantage is nominal and F0 dies without GPU.
3. **F2 test on the current model** (~8 GPU-h) — possibly a cheaper result than the epic.
4. Then S3 (re-powered), S4, S5 (with the multivariate score), S6–S10.
