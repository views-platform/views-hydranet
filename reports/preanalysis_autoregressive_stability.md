# Pre-Analysis Plan — Autoregressive Stability of the HydraNet MC-Dropout Posterior

**Date:** 2026-06-03
**Branch:** `fix/variational-dropout-autoregressive-stability`
**Status:** Pre-registration. Hypotheses and expected outcomes are stated *before* implementation and runs, so that results are judged against commitments made in advance (Popperian discipline — no post-hoc rescue).
**Scope:** Why the June-3 retrain produced astronomically exploding evaluation metrics on specific regression heads, what we will change, why, how, and what we expect — grounded in three papers read end-to-end and cited with verbatim text.

---

## 0. The observation we are explaining

The post-C-111 clean retrain (June 3) produced finite, healthy metrics on `pink_pirate` (all targets) and on the `lr_os_best` head everywhere, but **astronomical** raw-space CRPS/MCR on exactly two heads:

| Model | Exploding target | Step-1 CRPS | Step-12 CRPS | Raw pred. max |
|-------|------------------|-------------|--------------|---------------|
| blue_stranger | `lr_ns_best` | 0.10 | 6.1e15 | ~1.66e22 |
| violet_visitor | `lr_sb_best` | 0.29 | 9.2e5 | — |
| pink_pirate | (none) | normal | normal | normal |

The decisive datum is the **step-1 → step-12 growth**: at forecast step 1 the prediction is normal; the divergence appears and compounds across the 36-step autoregressive roll-forward. A static, output-side amplification would already be visible at step 1. **It is not. The error is generated and amplified *by the recursion itself*.**

This reframes our earlier (incomplete) diagnosis. The expm1 inverse-transform (ADR-028 §3) is the *final amplifier* that turns a log-space value of ~36–51 into 1e15–1e22 raw — but the *root* is recurrent runaway: ADR-028 §1 (spectral-radius / weight-multiplied-36-times) and §2 (additive cell-state "interest rate" explosion). Both §1 and §2 mitigations were marked **Deferred** in ADR-028's own status notes (2026-03-13). The IntegrityGuardian ceiling (1000, log-space) never fires because the damage is done at log-space ≈ 36–51, below threshold.

A second, independent suspect — and the subject of this plan — is the **stochasticity mechanism**: our posterior is Monte-Carlo Dropout, and our inference loop **resamples a fresh dropout mask at every one of the 36 autoregressive steps**. The literature is explicit that this is the wrong thing to do in a recurrent model. That is what we will fix first.

---

## 1. The three papers and what each contributes

### 1.1 Gal & Ghahramani (2016), *A Theoretically Grounded Application of Dropout in Recurrent Neural Networks* (NeurIPS 2016; arXiv:1512.05287)
`papers/Gal2016_RecurrentDropout.pdf`

This is the **direct prescription** for our failure. The paper opens by naming our exact symptom:

> *"Empirical results have led many to believe that noise added to recurrent layers (connections between RNN units) will be amplified for long sequences, and drown the signal. Consequently, existing research has concluded that the technique should be used with the inputs and outputs of the RNN alone."*

And it identifies the mechanism as **resampling the mask per step**:

> *"Current techniques (naive dropout, left) use different masks at different time steps … The proposed technique (Variational RNN, right) uses the same dropout mask at each time step, including the recurrent layers."* (Figure 1)

Its remedy — grounded in variational inference, not heuristics:

> *"In the new dropout variant, we repeat the same dropout mask at each time step for both inputs, outputs, and recurrent layers (drop the same network units at each time step). This is in contrast to the existing ad hoc techniques where different dropout masks are sampled at each time step."*

The theoretical justification (§4): treat the weights `ω` as random variables with approximate posterior `q(ω)`; MC-integrate the sequence likelihood with a **single** weight sample per sequence:

> *"Note that for each sequence xᵢ we sample a new realisation ω̂ᵢ … and that each symbol in the sequence xᵢ = [xᵢ,₁, …, xᵢ,T] is passed through the function fₕ with the same weight realisations … used at every time step t ≤ T."*

Concretely (eq. 7), the mask `zₓ, z_h` is *"repeated at all time steps."* At test time, prediction is MC dropout (eq. 4): draw `ω̂ₖ ∼ q(ω)`, forward, average over K — **one mask per trajectory, held fixed across the unroll.**

**Relevance:** our autoregressive forecast horizon (36 steps) is exactly the "long sequence" over which a per-step-resampled mask injects compounding noise into a recurrent loop. The paper says: hold the mask fixed across the horizon, vary it across posterior samples.

### 1.2 Gal & Ghahramani (2016), *Dropout as a Bayesian Approximation* (ICML 2016)
`papers/Gal2016_DropoutBayesian.pdf`

This paper is *why we are allowed to call MC dropout a posterior at all* — and simultaneously a statement of how crude that posterior is. It casts dropout as approximate variational inference in a deep Gaussian process; the predictive distribution is obtained by moment-matching over T stochastic forward passes:

> *"We sample T sets of vectors of realisations from the Bernoulli distribution … We refer to this Monte Carlo estimate as MC dropout. In practice this is equivalent to performing T stochastic forward passes through the network and averaging the results."* (eq. 6)

with predictive variance `≈ τ⁻¹ I_D + (1/T)Σ ŷᵀŷ − E[y]ᵀE[y]` — i.e. an inherent-noise term plus the sample variance of the forward passes.

Crucially, the **variational family is fixed and minimal**: a *"mixture of two Gaussians with small variances, with the mean of one Gaussian fixed at zero"* (per the RNN paper's summary of this one). The posterior is not learned; its shape is dictated by the dropout rate. This is the formal sense in which MC dropout is "the poorest of choices but a choice": it is a *zero-extra-parameter* approximate Bayesian posterior whose expressiveness we do not control.

**Relevance:** it justifies our current uncertainty mechanism and gives the correct predictive-moment estimator — but also tells us the posterior we are sampling is a rigid Bernoulli-induced family, not something tuned to conflict-count tails. It is the bridge to paper 1.3.

### 1.3 Kingma & Welling (2019), *An Introduction to Variational Autoencoders* (Foundations & Trends in ML; arXiv:1906.02691v3)
`incoming/vea/1906.02691v3.pdf`

This is the **principled destination**, not the immediate fix. It frames the general problem we are crudely solving with dropout: learn a model `p_θ(x,z)` with latent variables and an *amortized* approximate posterior `q_φ(z|x)` (the encoder/inference model), optimized jointly by maximizing the ELBO:

> ELBO `L_{θ,φ}(x) = E_{q_φ(z|x)}[log p_θ(x,z) − log q_φ(z|x)]` (eq. 2.10), with `L = log p_θ(x) − D_KL(q_φ(z|x) ‖ p_θ(z|x))` (eq. 2.11) ≤ log p_θ(x).

Two pieces matter for us:

1. **The reparameterization trick (§2.4)** — the variance-reduction device. Externalize the randomness: `z = g(ε, φ, x)`, e.g. the factorized-Gaussian case `z = μ + σ ⊙ ε`, `ε ∼ N(0, I)` (eq. 2.40). The monograph is explicit that this is the cure for exactly the pathology we have:

   > *"this sampling induces sampling noise in the gradients required for learning. Perhaps the greatest contribution of the VAE framework is the realization that we can counteract this variance by using what is now known as the 'reparameterization trick', a simple procedure to reorganize our gradient computation that reduces variance in the gradients."*

2. **Flexible posteriors (§3) and autoregressive decoders (§4.3)** — `q_φ(z|x)` can be made arbitrarily expressive via auxiliary variables or normalizing flows / inverse autoregressive flow (IAF), and the generative model itself can be autoregressive (eq. 4.8). This is the menu for a *learned, tail-aware, calibrated* posterior — the thing that would actually address our standing MCR problem rather than just stabilizing the current sampler.

**Relevance:** it tells us where to go *after* stabilization — replace the rigid dropout-induced posterior with a learned latent-variable posterior (reparameterized sampling, ELBO training, optionally IAF for non-Gaussian tails). It also gives the correct vocabulary (amortized VI, ELBO, reparameterization) for the eventual ADR.

---

## 2. What we are doing, why, and how

### 2.1 What
Implement **variational (consistent-mask) dropout** in the HydraNet inference path: within a single posterior sample's 36-step autoregressive roll-forward, **hold every dropout mask fixed across all steps**; draw a fresh mask only when starting the next posterior sample.

### 2.2 Why
Because Gal & Ghahramani (2016, RNN) state precisely that per-step mask resampling *"will be amplified for long sequences, and drown the signal,"* and that the grounded alternative is *"the same dropout mask at each time step."* Our inference loop currently does the naive thing: `nn.Dropout` in `train()` mode, called once per step, draws a new mask every step. In a recurrent feedback loop whose spectral radius is ≥1 for some heads (ADR-028 §1/§2), this white, time-uncorrelated noise is the seed that the recursion amplifies into the step-12 blow-up. Holding the mask fixed converts the 36 forward passes into a *single coherent weight realization* `ω̂ ∼ q(ω)` — exactly the estimator the theory prescribes (eq. 4).

This fix is **magnitude-neutral**: it changes the *temporal correlation* of the dropout noise, not the values the heads are permitted to emit. That matters because we are simultaneously fighting **under**-prediction (MCR ≪ 1 on `sb`/`ns`). Any output clamp or bounded head activation would cap the upper tail and push MCR the wrong way; the consistent-mask fix does not.

### 2.3 How
A custom locked-mask dropout module replacing bare `nn.Dropout` in the stochastic path: it caches its Bernoulli mask on first forward and reuses it until explicitly reset; the inference loop calls `reset_mask()` once per posterior sample (not per step). (Implementation deferred — no code in this plan.)

---

## 3. What we expect that to do, why, and how we will know

### 3.1 Primary pre-registered prediction
With consistent-mask dropout and **no magnitude capping of any kind**, the step-wise CRPS trajectory for `blue_stranger/lr_ns_best` and `violet_visitor/lr_sb_best` will **remain bounded across all 36 steps** (no step-12 explosion), and raw predictions will stay within physically plausible range (≲ the data max, expm1(12.1) ≈ 1.8e5, not 1e22).

- **Why:** if the runaway is seeded by per-step mask noise (Gal's "amplified for long sequences"), removing the per-step resampling removes the seed. The single-`ω̂`-per-trajectory estimator is dynamically coherent and cannot random-walk off the manifold the way fresh-mask-per-step can.
- **How measured:** re-run evaluation on the existing June-3 artifacts with the new inference path; compare per-step CRPS (step 1 … 36) before/after on the two exploding heads, plus the healthy controls (`pink_pirate`, all `os`).

### 3.2 Secondary predictions
- **MCR-neutral on healthy heads:** `pink_pirate` and all `os` metrics change only within run-to-run/sample-count noise (we are not touching their magnitudes). If healthy heads shift materially, the mechanism is not what we think.
- **MCR not worsened on the formerly-exploding heads:** once finite, their MCR should land in a sane range (≪ the 1e8 artifact, and ideally not *below* the pre-existing conservative baseline). We are explicitly *not* expecting consistent-mask dropout to *fix* MCR — only to stop the explosion. MCR remains the open problem for the later (VAE-class) mechanism.

### 3.3 The falsifier
If, with consistent masks and no clamping, the step-12 explosion **persists**, then the driver is *not* the dropout noise but the **deterministic recurrence** (spectral radius >1 / unbounded cell-state accumulation, ADR-028 §1/§2). In that case the next intervention is cell-state clamping (§2) and/or bounding the *autoregressive feedback input* to the training domain (in-domain feedback) — neither of which is an output cap, so both remain MCR-neutral. This branch is pre-committed so we do not rationalize a null result.

---

## 4. Design-decision record (RESOLVED 2026-06-03)

All three decisions are settled. The **chosen** option is marked ✓; the **rejected/deferred** alternatives are preserved deliberately — they are not discarded, and each carries the condition under which we would revisit it. (Mechanical grounding for these choices is in Appendix A.)

### Decision 1 — Mask granularity

The mapping to Gal (2016, RNN) is exact: our "sequence" = one `predict()` trajectory (digest + seed + 36 autoregressive steps); our "K MC samples" = the `for sample_idx in range(n_posterior_samples)` loop. Gal eq. (4): one weight realization `ω̂` per trajectory, fresh per sample.

- ✓ **1a — Per-posterior-sample.** One locked mask fixed across the whole trajectory; reset at the top of `predict()` (per `sample_idx`). This is the literal Gal MC-dropout estimator and matches our loop structure.
- ✗ **1b — Per-step (status quo).** Fresh mask every forward. This *is* the bug (noise "amplified for long sequences, and drown the signal").
- ✗ **1c — One mask across many samples/origins.** Too few independent draws → posterior collapses (under-dispersed). Rejected. *Revisit:* never as the posterior; only conceivable as a debugging aid.

### Decision 2 — Train vs inference-only

- ✓ **2a — Inference-only, first.** Lock masks only in the posterior-sampling path. **Testable with zero retrain** by re-evaluating the existing June-3 artifacts — immediate falsification of §3.1. Keeps trained weights identical to the C-111 baseline, isolating the variable. Accepts a mild train/test variational mismatch (widely done in practice).
- ◻ **2b — Both train and inference (DEFERRED, not rejected).** Strictly faithful to Gal's Bayesian story (optimize `q(ω)` under the same scheme used to sample it). Requires a full retrain to test and couples with scheduled sampling (ADR-056, which already feeds `prev_pred` back during training). 2a is a strict, reversible subset of 2b. *Revisit:* if (i) 2a stabilizes inference but we want the Bayesian interpretation to hold rigorously for a production train, or (ii) we keep dropout as the long-term posterior (we likely will not — see §5).

### Decision 3 — Scope of the locked mask

Two sub-questions, both grounded in Appendix A (Fact A: one shared `nn.Dropout`, 16 emission-path applications, **zero recurrent dropout**).

**(i) Add recurrent-connection dropout (full Gal)?**
- ✓ **3a — No.** Lock only the *existing* emission-path dropout. Adding dropout to the 4 ConvLSTM cells is a **new regularization surface = a modeling change**, not a stabilization fix, with its own retraining and risk.
- ◻ **3b — Add consistent-mask recurrent dropout (DEFERRED).** More faithful to full Gal. *Revisit:* only as a deliberate regularization experiment, and **only after** resolving the open provenance question below.

> **OPEN QUESTION (provenance unknown):** *Why does HydraNet have no dropout on the recurrent (ConvLSTM) connections?* The original rationale is not remembered and not recorded anywhere we can find. Before any 3b experiment we must establish whether this was (a) a deliberate stability choice (recurrent dropout was tried and destabilized — consistent with the pre-Gal literature Gal cites), (b) an oversight, or (c) inherited from a predecessor architecture. Until resolved, "no recurrent dropout" is treated as an *undocumented assumption*, not a justified decision. → candidate risk-register entry.

**(ii) One shared mask or per-call-site?**
- ✓ **Per-call-site, temporally locked.** The 16 applications operate on tensors of **different shapes** (encoder vs bottleneck vs decoder dims), so a single mask tensor cannot be shared across sites. Each call-site therefore gets its *own* mask, each held fixed across the 36 steps. Interface consequence: the `LockedDropout` module must cache masks keyed by tensor shape (or be instantiated per site) and expose a `reset()` that clears all cached masks, called once per `sample_idx`.

### Resolved interface (falls out of 1a + 2a + 3a)
A `LockedDropout` module (caches Bernoulli masks by shape, `reset()` clears them) replaces the single shared `nn.Dropout`; `reset()` is called at the top of `predict()`; behavior is gated so training is unchanged (inference-only). No recurrent dropout. No output/activation clamping (see §5 on why clamping is rejected here).

---

## 5. Calibration, the "right-for-the-wrong-reason" trap, and the VAE arc

### 5.1 Why we must not over-clamp (rejected alternative, recorded)
The most obvious "fix" — and the one ADR-028 §3 actually prescribes (`torch.clamp(out_reg, max≈15)` before `expm1`) — is **rejected here**, and the rejection is itself a decision worth recording. We are *simultaneously* fighting under-prediction (MCR ≪ 1 on `sb`/`ns`): the model predicts too low in magnitude. Any output clamp or bounded head activation (hardtanh / ReLU6 / clipped-ReLU) caps the *upper* tail and pushes MCR further the wrong way. So the ADR-028 §3 clamp, while it would make the metric finite, fights the calibration goal. The consistent-mask fix (and, if needed, the §3.3 fallback of cell-state clamping / in-domain feedback bounding) stabilizes the **dynamics** without censoring the **magnitudes** — that is the whole point. *Revisit the §3 clamp* only as a last-resort safety net beneath a dynamics fix, never as the primary lever.

### 5.2 The posterior-spread trap (important, easy to misread)
Locking the dropout mask **will reduce the posterior spread**, because per-step resampling was injecting fresh white noise at every one of the 36 steps. We must **not** read a post-fix narrower posterior as a regression. The pre-fix spread was *right for the wrong reason*: it looked like uncertainty but was largely an **artifact of per-step mask noise compounding through the recursion** — the same artifact that, on two heads, ran away to 1e22. A wide posterior produced by an unstable mechanism is not calibrated uncertainty; it is noise that happened to have non-trivial variance. The correct test post-fix is not "is the spread as wide as before" but "is the spread *calibrated*" (does coverage match nominal; does MCR sit near 1). If the fixed-mask posterior is too narrow to be calibrated, that is a true finding about MC dropout's inadequacy — not a reason to revert to the unstable mechanism.

### 5.3 The VAE arc — promoted from "deferred" to near-term candidate
Stabilization ≠ a good uncertainty mechanism. The principled replacement for the dropout-induced posterior is a **learned latent-variable posterior** in the Kingma–Welling sense: an inference model `q_φ(z|x)` with the **reparameterization trick** (eq. 2.40, `z = μ + σ⊙ε`) for low-variance, magnitude-honest sampling, trained against an **ELBO** (eq. 2.10), and — if the conflict-count tail demands it — a **flexible / autoregressive** posterior (IAF §3.4; autoregressive decoders §4.3) rather than the fixed two-Gaussian Bernoulli-induced family that MC dropout is stuck with (§1.2). This is the only route on the table that addresses **explosion and MCR simultaneously**: a calibrated, learned spread instead of an incidental one. Per the 2026-06-03 discussion this is now a **near-term candidate, not a someday-item** — once the train is stable, it is the next mechanism to prototype. It connects to the registered research directions (Tweedie #60, weighted-CRPS #61, Hurdle-IMDL #62, ZINB #63) and warrants its own ADR when we commit.

---

## 6. Validation protocol & success criteria

- **Reproduce** the explosion deterministically in a unit test (tiny model, forced high-variance dropout, 36-step loop): per-step mask → blow-up; locked mask → bounded. This is the TDD anchor and the mechanism proof.
- **Re-evaluate** the June-3 artifacts with the new inference path; compare per-step CRPS/MCR on the two exploding heads + healthy controls.
- **Success:** §3.1 holds (bounded, no clamp) and §3.2 holds (healthy heads unchanged) → consistent-mask dropout is the stabilizer; proceed to a full clean retrain+eval for the genuine post-C-111 comparison.
- **Failure:** §3.3 — pivot to ADR-028 §2 cell-state clamp / in-domain feedback bounding.

---

## Appendix A — Mechanical findings (code-grounded, 2026-06-03)

These are the concrete facts about the current implementation that ground every decision in §4. Verified by reading `architectures/HydraBNrecurrentUnet_06_LSTM4.py`, `utils/hydranet_inference.py`, and `train/training_engine.py`.

**Fact A — where dropout lives.** There is exactly **one** `nn.Dropout(p)` instance (`self.dropout`, defined once at `__init__`), applied **16 times** in the U-Net *emission* path: encoder (`e0s`, `e1s`), bottleneck (`b`), and all 6 decoder heads (`H{1,2,3}_{reg,class}`, each `_d0` and `_d1`). The **4 ConvLSTM cells carry no dropout** — the gate convolutions (`Wxi/Whi/...`) and the state updates (`hl = f*hl + i*hl_tilde`; `hs = o*tanh(hl)`) are dropout-free. Dropout rates in the golden_hour members: pink 0.125, violet 0.15, blue 0.1.

**Fact B — why "safe" dropout still explodes.** The pre-Gal literature (Zaremba et al., quoted in Gal 2016) holds that dropout on the *non-recurrent / emission* path is the safe placement, because in a **teacher-forced** RNN the output does not feed back. But HydraNet inference is **free-running autoregressive**: `forward(x, h)` produces a prediction that is detached and fed back as the next input (`t0_autoreg = t1_pred.detach()` at `hydranet_inference.py:294`). So emission-path dropout noise — normally harmless — re-enters the recurrence through the autoregressive feedback and compounds. This resolves the apparent paradox "we only have dropout on the U-Net, so why does it run away."

**Two cross-step feedback channels.** Across the 36 steps, state propagates through (1) the **prediction** `t1_pred → x_{t+1}` (autoregressive feedback; carries the dropout noise; this is the channel C-99 and ADR-054 govern — it uses post-ReLU `output.reg`), and (2) the **hidden state** `h = h_next` (deterministic given inputs, but indirectly perturbed because it depends on the noisy fed-back `x`). Locking the mask removes the *injection* of fresh noise into channel (1) at every step.

**Fact C — the loops.**
- *Inference* (`generate_posterior_samples`, `hydranet_inference.py:469`): `for sample_idx in range(n_posterior_samples): predict(...)`. Each `predict()` runs one full trajectory (digest + seed + 36 autoregressive steps). Dropout is forced to `train()` mode (`_apply_dropout`, line 85) → standard `nn.Dropout` resamples a **fresh mask on every forward = every step**. → The natural reset point for `LockedDropout` is the top of `predict()`.
- *Training* (`_process_sequence` via `training_loop`, `model.train()`): per-step forward; scheduled sampling (ADR-056) replaces ground truth with `prev_pred` with probability `ss_epsilon` (`t0_input = where(rand < ss_epsilon, prev_pred, t0_gt)`). So training is **partially** free-running (more teacher-forced early, when `ss_epsilon` is small), which is why per-step masks are far less dangerous at train time than at inference — and why inference-only (Decision 2a) is the surgical fix.

**Guard blindness (recap, confirmed).** The autoregressive magnitude guard warns at log-space |pred|>100, severe>500, and `IntegrityGuardian.PREDICTION_MAGNITUDE_CEILING = 1000` (code) halts. The observed runaway reached raw ≈ 1.66e22 ≈ log-space ≈ 51 — **under** the 100 warning threshold. The guard is calibrated for log-space explosions and is structurally blind to a log-space value that is "only" ~50 but catastrophic after `expm1`.

---

## 7. References

1. Y. Gal and Z. Ghahramani (2016). *A Theoretically Grounded Application of Dropout in Recurrent Neural Networks.* NeurIPS 2016. arXiv:1512.05287. → `papers/Gal2016_RecurrentDropout.pdf`. **Use:** consistent-mask (variational) dropout; "amplified for long sequences, and drown the signal"; same mask at every time step (Fig. 1, eq. 7); MC dropout at test (eq. 4).
2. Y. Gal and Z. Ghahramani (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML 2016. → `papers/Gal2016_DropoutBayesian.pdf`. **Use:** justification of MC dropout as approximate Bayesian inference; predictive moments (eq. 6 + variance τ⁻¹I + sample variance); the fixed two-Gaussian variational family (why the posterior is crude).
3. D. P. Kingma and M. Welling (2019). *An Introduction to Variational Autoencoders.* Foundations and Trends in ML. arXiv:1906.02691v3. → `papers/Gal2016_*` sibling at `incoming/vea/1906.02691v3.pdf`. **Use:** amortized VI and the ELBO (eq. 2.10–2.12); the reparameterization trick as variance control (§2.4, eq. 2.40); flexible posteriors / IAF (§3) and autoregressive models (§4.3) as the principled future uncertainty mechanism.

---

*No code has been written. This document is the commitment we will be held to when results arrive.*
