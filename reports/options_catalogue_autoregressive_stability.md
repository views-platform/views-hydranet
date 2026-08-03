# Options Catalogue — Stabilizing the Autoregressive ConvLSTM (post-falsification)

**Date:** 2026-06-04
**Companion to:** `postmortem_locked_dropout_negative_result.md`, `preanalysis_autoregressive_stability.md`, ADR-057
**Status:** Drawing-board. All 7 newly-fetched papers + Hess/Bengio read end-to-end. This is a *qualified, commented* menu — not a decision.

---

## 0. The problem, restated precisely (so every option is judged against the same target)

A ConvLSTM U-Net (`HydraBNUNet06_LSTM4`: 4 ConvLSTM cells, conv gates, hidden state `h` carried across steps) forecasts **36 steps free-running** (its post-ReLU prediction feeds back as next input). For *some trained-weight configurations* the recurrence diverges over the horizon — **seed-dependent**: pink_pirate (seed 4) bounded, violet_visitor (seed 42) blows to log-space ~13–51, which `expm1` (log1p inverse) amplifies to ~1e17 raw counts.

**Three facts that constrain every option below:**
1. **It's the deterministic recurrence, not noise.** Identical locked-mask code, identical n; pink bounded, violet diverged. → effective spectral radius ≥ 1 for some learned weights. (Dropout falsified; that whole axis is closed.)
2. **Magnitude-neutrality is mandatory.** We are *under*-predicting (MCR ≪ 1). Any fix that caps the emitted count (output clamp, bounded head activation) fights MCR and is rejected as a primary lever. Fixes must bound the *dynamics*, not the *output values*.
3. **`expm1` is an orthogonal amplifier.** Even a perfectly stable recurrence emitting the data-max (log 12.1) gives 1.8e5; the catastrophe is the recurrence pushing log-space *above* the data range. Two ways out: stop the recurrence exceeding the range (Axes A/B), or stop using `expm1` at all (Axis C — count likelihood).

**The strategic realization:** every genuine fix below **requires retraining** — none is inference-only (the only inference-only options were the falsified dropout and the rejected output-clamp). So we are necessarily in retrain territory, which argues for **batching** the chosen interventions into a single training campaign and **re-measuring the spectral diagnostic** (Axis 0) after each, not just CRPS.

> **⚠ UPDATE (2026-06-04) — `freeze_h` ablation sharpens fact #1.** A pre-registered 2×2 ablation (`reports/results_freezeh_ablation.md`) re-evaluated violet under `freeze_h ∈ {none, hl, hs, all}`. **All four arms exploded within ~half an order of magnitude on `lr_sb` (2.1–7.0 ×10¹⁷), including `all` — the entire recurrent hidden/cell state frozen for all 36 steps (5.13e17).** Since freezing the whole recurrent state does not change the explosion, the divergence is **not** carried by the recurrent hidden-to-hidden dynamics; it rides the only channel left live in every arm — the **prediction→input feedback loop**. Two corrections follow, applied inline below:
> - **Axis 0 — measure the wrong operator no longer.** The diagnostic quantity is the **input→output Jacobian gain `‖∂pred/∂x_input‖`** over one autoregressive step (pink < 1, violet > 1 expected), *not* the recurrent hidden-to-hidden spectral radius (which `freeze_h` already neutralizes with no effect).
> - **Axis A — retarget the constraint.** Spectral-norm / Lipschitz belongs on the **input-to-hidden convs `Wx*` AND the U-Net encoder/decoder convs** (the input→output path), not (only) the recurrent `Wh*` gates. This also *promotes* Axis B (pushforward/GTF attack the fed-back-error channel directly) and the in-domain feedback-input clamp; and it **pre-falsifies the ADR-028 §2 cell-state clamp** (`freeze_h="all"` is its extreme and didn't help).
>
> **⚠ UPDATE 2 (2026-06-04) — Axis-0 diagnostic run (`reports/results_io_gain_diagnostic.md`); my own `‖∂pred/∂x‖` proposal above is itself partly wrong.** Standalone retrain-free diagnostic on pink vs violet:
> - **The local operator norm `‖∂reg/∂x‖₂` does NOT discriminate** — both models are locally expansive (>1); pink is sometimes higher. Drop it as the diagnostic.
> - **The discriminator is the free-running attractor level vs the data range.** Violet's rollout settles at **log-space ~40** → `expm1` ≈ **1e17** (matches the observed CRPS); pink's settles **in-range at ~log 10** → ~2e4. State-independent (frozen-state rollout reaches the same ~40), independently reproducing the ablation. So the disease is an **out-of-range fixed point of the feedback map**, not raw local gain.
> - **The in-domain feedback clamp was TESTED — it is a safety rail, not a fix (`reports/results_feedback_clamp.md`).** Clamping the fed-back copy per-target to the log1p data max (retrain-free, one line in `predict()`) **averts the catastrophe** (violet `lr_sb` CRPS 2.13e17 → 798; `lr_ns`/`lr_os` fully recover to healthy; **benign on pink**). But it **triggers falsifier F2** for `lr_sb`: the head *ramps to the ceiling over the horizon and pins there* (MCR ~56,000 — gross over-prediction), because the clamp caps *where* the runaway lands without stopping the map from running there. So: keep it as an **optional finite-output guard rail** (default off), but the **durable fix is still required** — lower the input→output attractor (spectral-norm/Lipschitz on the input→output path, pushforward/GTF, or a count-likelihood head). "Bounded" was necessary, not sufficient — exactly the pre-registered skepticism. (The "may be retrain-free fix" hope is thus only half-true: catastrophe-prevention is retrain-free; calibrated `lr_sb` is not.)

---

## Axis 0 — DIAGNOSE FIRST (cheap, do before any fix)

### Pascanu et al. 2013 — *On the difficulty of training RNNs* — **DIAGNOSTIC, not a fix**
- **What it gives us:** the canonical analysis that BPTT gradients (and the forward map) scale as `λ_max(W_rec)^t`; `λ_max>1` ⇒ exponential blow-up. Plus gradient clipping (Algorithm 1).
- **My commentary:** the agent is right that as a *fix* this is a **poor fit** — clipping stabilizes *training*, not *free-running inference*, and our training curves were healthy (violet's HEALTH AUDIT passed). **But its real value is as the diagnostic that confirms our whole theory.** Before spending any retrain budget, measure, on the existing pink vs violet artifacts: (a) **[CORRECTED per the freeze_h ablation — see §0 UPDATE]** the **input→output Jacobian gain `‖∂pred/∂x_input‖`** over one autoregressive step (the gain on the *fed-back prediction*), and (b) the step-over-step growth of `‖prediction‖` across the 36-step rollout. The recurrent hidden-to-hidden gate σ is now *secondary* — the ablation showed freezing the entire recurrent state leaves the explosion intact, so the divergence is in the input→output map, not the recurrence. **Prediction:** violet's input→output gain > 1 and `‖prediction‖` grows geometrically; pink's gain < 1. If that holds, the spectral-radius diagnosis is confirmed and Axis A is pointed-at directly. If it *doesn't*, we've been wrong and must rethink. **This is the single highest-value next action — it's cheap (no retrain) and it's falsifiable.**
- **Cost:** none (analysis on existing weights). **Verdict:** **do this first.**

---

## Axis A — Constrain the recurrence (architectural / weight-level)

*These bound the hidden-state dynamics by construction. All require retrain; all are magnitude-neutral on the output (the decoder head is a separate projection from the bounded state). The shared caveat is the **stability↔memory tradeoff** — see note at end of axis.*

### A1. Miyato et al. 2018 — *Spectral Normalization* — **★ LEAD (cheapest root fix)**
- **Mechanism:** rescale each weight matrix so its largest singular value σ(W)=1 (power iteration, ~10–20% train overhead). A layer with σ≤1 is 1-Lipschitz; with tanh/sigmoid gates (Lipschitz ≤1), the per-step recurrent amplification is bounded ≤1.
- **Fit to us:** **high and cheap.** Applies directly to the ConvLSTM hidden-to-hidden conv gates (Miyato themselves normalize conv layers). **No architecture change** — just normalize the gate weights during training; **inference-free** (weights already normalized). **Magnitude-neutral**: bounds `‖W h‖ ≤ ‖h‖`, never caps the decoder output.
- **My commentary:** this is the **principled, modern version of ADR-028 §1 (weight damping)** — and it's the minimal-change root fix. It's my pick for the **first retrain experiment** because it's the lowest-effort intervention that directly attacks the confirmed mechanism. **[CORRECTED per the freeze_h ablation — see §0 UPDATE]** Apply the constraint to the **input-to-hidden convs `Wx*` and the U-Net encoder/decoder convs** (the input→output path the ablation implicated), not only the recurrent `Wh*` gates — freezing the recurrent state didn't stop the explosion, so bounding only `Wh*` would miss the driver. Pair with the Axis-0 diagnostic: normalize the input→output path, retrain, re-measure `‖∂pred/∂x_input‖` and `‖prediction‖`.
- **Caveat:** σ≤1 per gate bounds *per-matrix* amplification but not the *sum/composition* across the 4 gates + cell-state accumulation — so it constrains but doesn't *prove* global stability. Marginal case (σ=1) preserves more memory than strict contraction. **Verdict: lead; first to try.**

### A2. Chang et al. 2019 — *AntisymmetricRNN* — **★ ELEGANT (best memory-preserving, heavier)**
- **Mechanism:** parameterize the recurrent matrix as antisymmetric `W=(M−Mᵀ)` ⇒ eigenvalues on the imaginary axis ⇒ ODE dynamics that neither grow nor decay (norm-preserving), discretized by forward Euler with step ε.
- **Fit to us:** **theoretically the sweet spot** — bounded dynamics *without* memory loss (unlike contractive σ<1). Retrofit is "parameterize each recurrent gate as antisymmetric," matrix-agnostic so conv-compatible in principle.
- **My commentary:** this is the most *intellectually* attractive — it bounds the recurrence while *preserving* the long-horizon memory a 36-step forecast needs. **But it changes the cell's identity** (it's no longer the LSTM recurrence; it's an antisymmetric-ODE recurrence), which is a real architectural commitment, invalidates artifacts, halves recurrent parameter count (n(n−1)/2), and adds a step-size ε to tune. **Verdict: strong candidate if A1 (spectral norm) bounds the dynamics but costs too much forecast skill (the memory tradeoff) — antisymmetric is the memory-preserving escalation.**

### A3. Erichson et al. 2021 — *Lipschitz RNN* — **viable but heavy**
- **Mechanism:** RNN-as-ODE with a skew-symmetric+damping parameterization giving *global exponential stability* (`‖h(t)−h*‖ ≤ Ce^{−κt}…`), tunable stability↔expressivity via β, γ.
- **My commentary:** rigorous and magnitude-neutral, and the tunable β/γ is nice (don't have to choose full contraction). But it's a dense-RNN formulation; transfer to conv gates is "moderate difficulty, no implementation given," needs β/γ tuning inside a stability region, and is a bigger rebuild than A1. **Verdict: the principled fallback if both A1 and A2 disappoint — most theory, most engineering.**

### A4. Arjovsky et al. 2016 — *Unitary RNN* — **poor fit**
- **Mechanism:** unitary recurrent matrix (|λ|=1) via Fourier/diagonal/reflection composition; complex hidden states; modReLU.
- **My commentary:** elegant for vanishing gradients on toy tasks, but **conv-incompatible** (the parameterization is for dense matrices), needs complex states + custom nonlinearity + full cell rebuild, and underperformed LSTM on real tasks (91% vs 98% pMNIST). **Verdict: poor fit; do not pursue.**

> **Axis-A shared caveat — the stability↔memory tradeoff (Miller & Hardt 2019).** *Stable Recurrent Models* proves the uncomfortable theorem: a **contractive** recurrence (λ<1) **cannot retain long-term memory** — "either the task doesn't need it, or the unstable model doesn't have it." Our forecast spans 36 steps and plausibly *needs* memory. Implication: **strict contraction (A-via-λ<1) risks trading the explosion for a skill loss.** The ranking A1(σ=1, marginal) → A2(antisymmetric, norm-preserving) → A3(tunable) reflects *increasing* memory-preservation. Miller & Hardt also supply the LSTM row-wise spectral-norm recipe (a concrete A1 variant for gated cells).

---

## Axis B — Train against rollout divergence (training-procedure)

*These keep the model from learning a divergence-prone map, by exposing it to its own error during training. All require retrain; magnitude-neutral; no architecture change. We ALREADY have one member (scheduled sampling) and it did **not** prevent the divergence — a caveat that colours the whole axis.*

### B1. Brandstetter et al. 2022 — *pushforward trick* (Message-Passing PDE Solvers) — **★ STRONG (closest domain analog)**
- **Mechanism:** add a stability loss that penalizes sensitivity to a perturbed input `A(uᵏ+ε)` (adversarial robustness ⇒ implicit Lipschitz map), plus *temporal bundling* (predict K steps, backprop only the last). No Jacobians, no ground-truth at inference.
- **Fit to us:** **excellent and the closest analog we have** — it's literally for stabilizing *autoregressive rollouts of spatiotemporal field predictors* (PDE solvers ≈ our ConvLSTM-over-a-grid), validated to 1000+ steps. Simpler than Hess (no Lyapunov estimation), magnitude-neutral.
- **My commentary:** my pick for the **training-side complement to A1**. It makes the learned map robust *at the points it actually visits during rollout* — directly attacking the "the model learns a sharp manifold that diverges off-distribution" failure. Crucially it is **not** a teacher-forcing/feedback-curriculum method, so it doesn't share DNA with our failed scheduled sampling. **Caveat:** it's a regularizer (implicit Lipschitz), not a guaranteed Lyapunov bound; perturbation scale σ needs tuning (~1% of input range). **Verdict: strong; pair with A1 in the first retrain.**

### B2. Hess et al. 2023 — *Generalized Teacher Forcing* — **most-targeted, heaviest, family-risk**
- **Mechanism:** interpolate model and ground-truth states `z̃=(1−α)ẑ+αz̄`, with **α set adaptively from the estimated Lyapunov exponent / Jacobian spectral norm** to keep the Jacobian-product σ̄_max ≤ 1. Built for *chaotic* dynamics (Lorenz, EEG).
- **Fit to us:** **the most direct theoretical match to "exploding Jacobian product."** If Axis-0 shows violet is genuinely chaotic (σ̄_max>1), GTF is the tool that explicitly clamps it.
- **My commentary (skeptical):** two reasons I rank it *below* B1 for a first attempt: (i) it requires **Jacobian/Lyapunov estimation each (few) step(s)** — expensive and numerically delicate at our state dimension; (ii) it is **the same family as our already-failed scheduled sampling** (manipulate the teacher-forcing/feedback signal). It's far more sophisticated than binary SS (adaptive, dynamics-aware) so it *could* succeed where SS failed — but I'd want the Axis-0 diagnostic to confirm "chaotic, σ̄_max>1" before paying its implementation cost. **Verdict: lead-tier *if* the diagnostic shows true chaotic divergence; otherwise B1 first.**

### B3. Bengio et al. 2015 — *Scheduled Sampling* — **we have it; insufficient alone**
- **Status:** implemented as ADR-056 (binary, ε schedule). It did **not** prevent the divergence.
- **My commentary:** the agent flags two things worth checking before we write it off: (i) confirm we flip **per-step, not per-sequence** (per-sequence "failed dramatically" in Bengio); (ii) our schedule may decay too fast — try a shallower inverse-sigmoid. **But the honest read:** SS is a *curriculum*, not a stability constraint; it "does not penalize gradient/Jacobian growth." At a 36-step horizon with σ≥1 weights, it's expected to be insufficient — consistent with what we saw. **Verdict: floor; tune only as a cheap add-on to B1/A1, not a standalone fix.**

---

## Axis C — Bound the output channel (likelihood / head) — *also fixes MCR, dissolves `expm1`*

*Parallel strategic track. Already on the roadmap as research issues. These don't constrain the recurrence; they remove the `expm1` amplifier and model counts directly — and are the only axis that also attacks the standing MCR problem.*

- **Tweedie likelihood** (Damato 2025 GP-Tweedie; Jiang 2023 spatiotemporal Tweedie) — compound-Poisson, native zero-inflation + heavy tail, no `expm1`. → **issue #60.**
- **Zero-Inflated Negative Binomial** (Iacus 2025 on PRIO-Grid; Lambert 1992 ZIP foundation) — explicit zero mass + count tail; validated on *our* data (R²=0.955). → **issue #63.**
- **DeepAR-style** (Salinas 2020) — autoregressive probabilistic forecasting with a bounded conditional likelihood and *sample-based* (not expm1-of-point) rollout.
- **My commentary:** **highest ceiling, biggest lift.** A count likelihood (NB/ZINB/Tweedie) predicts counts *directly* — there is no log-space-then-`expm1` step to amplify, and a well-specified count tail *is* calibrated magnitude (MCR). This is the only axis that could resolve **explosion AND MCR together**. It's a new loss + head + sampling path (overlaps the VAE/learned-posterior arc, I10). **Verdict: the strategic parallel track — pursue alongside, not instead of, Axis A.** If we're retraining anyway, this is the moment to consider it.

---

## Cross-cutting commentary (my synthesis, beyond the per-paper reads)

1. **Diagnose before you retrain.** Axis 0 (measure σ of violet's gates + `‖h_t‖` growth, pink vs violet) is cheap, falsifiable, and decides everything. If σ≥1 confirmed → Axis A is bullseye. If not → stop and rethink.
2. **The axes are complementary, not competing.** Axis A constrains the dynamics *structurally*; Axis B teaches stability *procedurally*; Axis C removes the *amplifier*. A defensible campaign is **A1 (spectral norm) + B1 (pushforward) in one retrain**, with **C (count head)** as a parallel higher-ceiling track.
3. **The agents each over-promoted their own cluster's lead.** My cross-cluster call: **A1 spectral-norm is the cheapest highest-confidence first fix**; **B1 pushforward is the best training complement** (and our closest domain analog); **GTF is lead-tier only if the diagnostic confirms chaos**; **Tweedie/ZINB is the deepest fix** but a bigger commitment.
4. **Mind the memory tradeoff (Miller & Hardt).** Don't over-contract. Prefer σ=1 (marginal) or antisymmetric (norm-preserving) over strict λ<1, because the 36-step forecast needs memory.
5. **Everything here costs a retrain.** That reframes cost: batch the cheap-and-complementary pair (A1+B1) into one campaign; reserve the heavier rebuilds (A2 antisymmetric, A3 Lipschitz, B2 GTF, C count-head) for if the cheap combo underperforms — and re-run Axis-0 after each so we measure *dynamics*, not just CRPS.

## Recommended sequencing (proposal, not a decision)

1. **Axis 0 diagnostic** (no retrain) — confirm σ≥1 / Jacobian growth on violet vs pink.
2. **A1 spectral norm** on the ConvLSTM recurrent gates → retrain violet → re-measure σ + CRPS + MCR.
3. **+ B1 pushforward** if A1 alone is partial (same retrain campaign).
4. **C count-likelihood (Tweedie/ZINB)** as a parallel, higher-ceiling track that *also* targets MCR (issues #60/#63, I10).
5. **Escalate to A2 antisymmetric / A3 Lipschitz / B2 GTF** only if 2–3 trade too much skill or fail to bound.

## Library status

In library now: all 7 fetched (`incoming/recurrent_stability/`), Hess GTF (`incoming/deep_consored/hess23a.pdf`), Bengio SS, Tweedie (Damato/Jiang), ZINB (Iacus), Lambert 1992 ZIP, DeepAR (Salinas). **No outstanding fetches** for these axes.

---

*Read end-to-end 2026-06-04: Miyato 2018, Miller&Hardt 2019, Pascanu 2013, Chang 2019, Erichson 2021, Arjovsky 2016, Brandstetter 2022, Hess 2023, Bengio 2015. Axis C papers were read at abstract/intro depth in the 2026-06-02 classification-loss/CRPS exploration.*
