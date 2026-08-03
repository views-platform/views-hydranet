# 01 — Literature (Axis B / rollout training)

**Drafted:** 2026-06-05. Read in full this session: the three primary papers below.
Grouped by role. Each entry: *what it is* → *what we take from it*. Gaps-to-fetch at
the end.

---

## A. The three primary methods (read in full)

### A1. Brandstetter, Worrall & Welling (2022) — *Message Passing Neural PDE Solvers* (ICLR). The **pushforward trick** + **temporal bundling**.
`incoming/recurrent_stability/Brandstetter2022_MessagePassingPDE_pushforward.pdf`

- **Names our problem exactly.** Autoregressive solvers map `uᵏ → uᵏ⁺¹`. One-step
  training minimises `L₁ₛₜₑₚ = E[L(A(uᵏ), uᵏ⁺¹)]`. At test, *"small errors in A
  accumulate over rollouts of length > 1 (the vast majority of rollouts), and lead
  to divergence."* Interpreted as **overfitting to the one-step training
  distribution** — the **distribution-shift problem**: after one step the solver
  sees samples from `A_# pₖ` (the pushforward of the model's own output
  distribution) ≠ `pₖ₊₁`, because *"errors always survive training."*
- **The pushforward trick.** Add a stability loss
  `L_stability = E[L(A(uᵏ + ε), uᵏ⁺¹)]` where `ε|uᵏ` is an **adversarial-style
  perturbation chosen so `uᵏ + ε ~ A_# pₖ`** — achieved cheaply by setting
  `uᵏ + ε = A(uᵏ⁻¹)`, i.e. **the model's own one-step-prior prediction**. Total loss
  `= L₁ₛₜₑₚ + L_stability`.
- **The cheap implementation (the part we take).** *"We implement this by unrolling
  the solver for 2 steps but only backpropagating errors on the last unroll step…
  We found it important not to backpropagate through the first unroll step. This is
  not only faster, it also seems to be more stable."* (Fig. 2: one-step = grad 1
  step; unrolled = grad through all N steps; **pushforward = grad through last step
  only**.) ⇒ **flat activation memory in K**, ~K× forward cost.
- **Adversarial-by-construction ≫ random noise.** Ablation (Fig. 5b): pushforward
  beats Gaussian-noise injection (Sanchez-Gonzalez 2020); *"injecting Gaussian
  perturbations appears worse than using none… they lead to lower accuracy, by
  nature of injecting noise into the system."* → **a direct strike against any "just
  add input noise" patch.**
- **Temporal bundling.** Predict K steps per call (`u⁰ → u¹:ᴷ`) → fewer solver calls
  → fewer distribution shifts → slower error propagation.
- **Zero-stability link (Hairer).** `‖A(u⁰+ε) − u¹‖ < κ‖ε‖`; pushforward minimises κ
  directly. Theoretical hook for "bounded amplification."
- **TAKE:** the *cheapest* Axis-B implementation. We already **detach** the fed-back
  prediction (`prev_pred.detach()`) — so the missing pieces are (i) compute & weight
  the loss on the *prediction-from-fed-back-input* always (not Bernoulli@ε), (ii)
  ensure K reaches the blow-up onset (~step 12). Memory-cheap; no deep BPTT graph.

### A2. Hess, Monfared, Brenner & Durstewitz (2023) — *Generalized Teacher Forcing for Learning Chaotic Dynamics* (ICML). **GTF**.
`incoming/deep_consored/hess23a.pdf`

- **Exploding gradients are a *principle* problem, not an architecture problem.**
  Via Mikhaeil et al. (2022): if the target system is chaotic (max Lyapunov exponent
  `λ_max > 0`), the BPTT Jacobian product `∂z_t/∂z_r = ∏ Jₖ` (Eq. 5) **necessarily
  explodes** as horizon T→∞. Architectures that bound it (LSTM, antisymmetric RNN,
  Lipschitz RNN, unitary RNN) *"either limit expressiveness such that chaotic
  dynamics cannot be learned, or still struggle with exploding gradients."* ⇒ fix at
  the **training-algorithm** level.
- **GTF mechanism.** Linearly interpolate RNN-generated and target states before
  applying the map: `z̃_t = (1−α) z_t + α z̄_t`, `0 ≤ α ≤ 1`. This scales **every**
  Jacobian: `J_t = (1−α) J̃_t` (Eq. 7), so the product becomes
  `∂z_t/∂z_r = (1−α)^{t−r} ∏ J̃_{t−k}` (Eq. 8). `α=0` → vanilla BPTT (explodes);
  `α=1` → no gradient flow; in between, α **directly controls the gradient-product
  norm.**
- **Provably bounded.** Choosing `α* = 1 − 1/σ̃_max` bounds the product from above
  (Prop. 2). A cheap upper bound `⌈σ̃_max⌉ = ‖A‖ + ‖W₁‖‖W₂‖` (Eq. 18), or an adaptive
  data-Jacobian estimate `aGTF` (Eq. 23). **Annealing** strategy: start at `α=1`
  (strong forcing) and decay toward the bounded-from-below `α*`.
- **Train-only.** α is used *only in training*; at test the model evolves
  autonomously. Forcing interval τ ≈ predictability time.
- **Result.** On real-world chaotic data (ECG 5-d, EEG 64-d): shPLRNN+GTF beats
  LSTM-TBPTT, reservoir computing, N-ODE, SINDy on attractor geometry (`D_stsp`) and
  power-spectrum (`D_H`) by **up to ~an order of magnitude**; sparse-TF (Mikhaeil)
  is worse than GTF ⇒ the *soft* interpolation matters.
- **TAKE:** the *principled* Axis-B implementation, and the cleanest reframing of our
  ADR-056 scheduled sampling: **SS is a hard-Bernoulli, detached, biased special case
  of GTF.** GTF says — *soft-mix* the fed-back signal `(1−α)·pred + α·GT`, **keep the
  gradient flowing** through the feedback loop (the operator that is currently
  severed), and **bound** it via α. Caveat to carry into review: GTF's *theory* is
  for chaotic systems; whether conflict-count dynamics are chaotic (λ_max>0) is
  unestablished — see the Sutton/Gneiting fault line in `02`.

### A3. Lamb, Goyal, Zhang, Zhang, Courville & Bengio (2016) — *Professor Forcing* (NIPS).
`incoming/rollout_training/1610.09038v1.pdf`

- **The mechanism we already half-use, and its critique.** Teacher forcing feeds
  ground-truth `yₜ` back during training; at test the model feeds its own samples →
  *"small prediction errors compound… the RNN's conditioning context diverges from
  sequences seen during training."*
- **Scheduled sampling is a biased estimator.** Bengio et al. (2015) mixes GT and
  self-generated inputs, but (Huszár 2015) *"scheduled sampling yields a biased
  estimator, in that even as examples and capacity → ∞, the procedure may not
  converge to the correct model."* ⇒ a recorded limitation of our current ADR-056.
- **Professor Forcing.** Train a **discriminator** (bidirectional RNN) to tell apart
  the *behaviour sequences* (hidden states + outputs) produced in teacher-forcing
  mode vs free-running mode; train the generator to (a) minimise NLL and (b) **fool
  the discriminator** — making free-running dynamics indistinguishable from
  teacher-forced. Matches the **distribution of trajectories**, not just point error.
- **Designed for "train short, generate long."** *"In some domains the sequences
  available at training time are shorter than the sequences we want to generate at
  test time. This is usually the case in long-term forecasting tasks (climate
  modeling, econometrics)… Professor Forcing can be used to improve performance in
  this setting. Note that scheduled sampling cannot be used for this task, because it
  still uses the observed sequence as targets."* — **our exact situation** (train
  window ≪ 36-step inference rollout).
- **Costs/limits.** ~3× teacher-forcing training time (sampling phase + feeding
  hidden-state distributions to D); adversarial-training instability (they only
  backprop D→generator when D accuracy ∈ [75%, 99%]). **No benefit** on word-PTB or
  on sequences < 100 steps ⇒ pays off only when long-horizon dependencies dominate.
- **TAKE:** the *maximal* option, catalogued not recommended-first. The "train short
  / generate long" passage is the strongest external statement of why our problem is
  real and why SS specifically cannot solve it.

---

## B. Recurrent-stability neighbours (held; support the "architecture won't save you" claim)
`incoming/recurrent_stability/`

- **Mikhaeil, Monfared & Durstewitz (2022)** — *On the difficulty of learning
  chaotic dynamics* (the proof Hess builds on; **gap — not held, fetch**).
- **Pascanu, Mikolov & Bengio (2013)** — *On the difficulty of training RNNs*
  (`Pascanu2013_*.pdf`): the original exploding/vanishing-gradient analysis +
  gradient clipping. Baseline lens for the DL-engineer seat.
- **Miller & Hardt (2019)** — *Stable Recurrent Models*
  (`MillerHardt2019_*.pdf`): stable RNNs ≈ feed-forward; the expressiveness cost of
  enforcing stability — the counter-pressure to "just make it contractive."
- **Arjovsky, Shah & Bengio (2016)** unitary RNN; **Chang et al. (2019)**
  antisymmetric RNN; **Erichson et al. (2021)** Lipschitz RNN; **Miyato et al.
  (2018)** spectral normalisation — the *architectural* stability family Hess argues
  is insufficient for chaotic targets. The Hochreiter seat will weigh these vs the
  training-level fix.

## C. Already in the repo / corpus

- **Bengio et al. (2015)** scheduled sampling
  (`incoming/deep_consored/NIPS-2015-scheduled-sampling-*.pdf`) — our ADR-056 basis.
- **Gal & Ghahramani (2016)** variational RNN dropout — the *posterior* mask fix
  (ADR-057), orthogonal; cited here only to keep the two tracks distinct.

## D. Gaps to fetch

1. **Mikhaeil, Monfared & Durstewitz (2022)** — the chaotic-DS ill-posedness proof
   (load-bearing for whether GTF's premise applies to us).
2. **Huszár (2015)** — *How (not) to train your generative model* (the
   scheduled-sampling bias argument, cited by both A1's lineage and A3).
3. **Sanchez-Gonzalez et al. (2020)** — learned simulators / noise injection (the
   baseline pushforward beats; lets us state the noise-injection alternative fairly).
4. **Oreshkin et al. (2019)** — *N-BEATS* (the direct-multi-horizon alternative the
   `02` design explicitly chooses *against*; cite it to make that choice honest).
5. *(optional)* **Vlachas et al. (2023)** — backprop-through-time for forecasting
   chaotic systems / annealing protocols (Hess cites for the annealing precedent).
