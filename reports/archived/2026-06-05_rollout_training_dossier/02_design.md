# 02 — Design: Axis B, "train the model the way it is used"

**Drafted:** 2026-06-05 · **Revised:** 2026-06-05 (folded in the `02b` method-review
findings — see the **[R1–R6] tags** below and §10) · **Status:** reviewed; revisions
folded in; pre-analysis plan (`05`) next · **Graduates to:** ADR-058 candidate.

---

## 1. The problem in one sentence

`HydraBNUNet06_LSTM4` is optimised on a one-step-ahead objective but executed as a
**36-step free-running autoregression** at inference; the prediction→input feedback
operator that carries the runaway gets **no gradient** during training, so the model
is never asked to be stable along the trajectory it will actually walk.

## 2. What the code does today (grounded, not assumed)

`views_hydranet/train/training_engine._process_sequence` (lines 178–231):

```python
prev_pred = None
for i in range(seq_len - 1):                       # window steps
    t0_gt = t0[:, idx.feat, :, :]
    if ss_epsilon > 0.0 and prev_pred is not None: # ADR-056 scheduled sampling
        mask = torch.rand(...) < ss_epsilon         #   hard Bernoulli @ ss_epsilon
        t0_input = torch.where(mask, prev_pred, t0_gt)
    else:
        t0_input = t0_gt
    output = model(t0_input, h)                     # h carried across steps
    t1_pred, t1_pred_class, h = output.reg, output.cls, output.h_next
    prev_pred = t1_pred.detach()                    # <-- feedback gradient SEVERED
    ...                                             # per-step loss summed
```

Two facts that sharpen every design choice below:

1. **The recurrent state `h` is *not* detached** → gradient already flows through the
   ConvLSTM across the window (standard BPTT). So we are *not* fully teacher-forced in
   the hidden-state sense.
2. **The prediction feedback *is* detached** (`prev_pred = t1_pred.detach()`) and only
   engaged with probability `ss_epsilon`. The io-gain diagnostic
   (`scripts/diagnose_io_gain.py`; `reports/results_freezeh_ablation.md`) found the
   runaway **rides this feedback loop** (operator gain ‖d reg/d x‖₂ > 1 at the visited
   operating points), *not* the recurrent state. So the one operator empirically
   responsible for the blow-up is the one operator that receives **zero training
   signal** about its own multi-step behaviour.

This is the "train/inference lie," stated precisely. It is *not* "no gradient
anywhere through time" — it is "no gradient through the specific feedback path that
diverges."

## 3. Why this is the right layer to fix (and freeze_h is not)

All three papers converge on the same claim from different fields:

- **Hess/Mikhaeil:** for chaotic targets the gradient blow-up is a *principle*
  problem — *architectural* constraints (LSTM, Lipschitz/antisymmetric/unitary RNN,
  spectral norm) either can't fix it or kill expressiveness. Fix the **training
  algorithm**.
- **Brandstetter:** instability is *overfitting to the one-step distribution* — fix
  the **training objective** (add the stability term).
- **Lamb:** the conditioning context diverges — fix the **training objective**
  (match free-running to teacher-forced dynamics).

**Corollary — Element 1, kill `freeze_h`.** Freezing the LSTM cell/hidden state at
inference (`execute_freeze_h_option` in `hydranet_inference.py`) is an
inference-time *hard prior* that fights a symptom on the wrong operator (the state,
which the diagnostic exonerated) and contradicts the literature's "fix it in
training" consensus. The freeze_h ablation (`reports/results_freezeh_ablation.md`)
already showed it is **inert** for the runaway. Decision: set `freeze_h="none"`,
remove the inference-time state-freeze. This dossier supplies the real fix that lets
us retire the hack rather than lean on it.

## 4. The design space

Two orthogonal axes. We are choosing on the **training** axis and explicitly *keeping*
the recurrence.

### 4.1 Recursive vs direct-multi-horizon (the axis we are *not* taking)
Brandstetter's framing: **neural-operator/direct** methods map `initial → u(t)` in one
shot (no recursion; N-BEATS, Oreshkin 2019 is the forecasting archetype) vs
**autoregressive** methods iterate `A(Δt, u(t))`. Dropping recursion would sidestep
the feedback loop entirely — but it discards the Markov/temporal inductive bias that
matches conflict dynamics (this period's state conditions next period's), forces a
fixed output horizon, and throws away the ConvLSTM we have. **We keep the recurrence
and fix its training.** (N-BEATS is catalogued as the honest alternative; not chosen.)

### 4.2 The three training-axis implementations (graded by cost/ambition)

| | **B1 — Pushforward** (Brandstetter) | **B2 — GTF** (Hess) | **B3 — Professor Forcing** (Lamb) |
|---|---|---|---|
| **Idea** | add `L_stability` on the prediction made *from the model's own one-step-prior prediction* | soft-mix the fed-back signal `(1−α)·pred + α·GT`, keep gradient, **bound** via α | adversarial discriminator forces free-running dynamics ≈ teacher-forced |
| **Feedback gradient** | **detached** across steps; grad only through the *last* step | **flows** through the rollout, scaled by `(1−α)^{t−r}` (provably bounded) | flows through the free-running rollout (to fool D) |
| **Relation to our code** | we already detach; add the always-on stability term + reach K | the principled upgrade of ADR-056: SS = hard-Bernoulli, detached, *biased* GTF | a new discriminator network + adversarial loop |
| **Activation memory** | **O(1) in K** (no deep graph) | **O(K)** (true BPTT through K) | O(K) + discriminator |
| **Compute / step** | ~K forward + K one-step backprops | K forward + 1 BPTT-through-K | sampling pass + D forward/back; ~3× TF |
| **Bias** | unbiased on the stability term | α-bounded; annealed to remove bias | matches trajectory distribution (strongest) |
| **New machinery** | minimal (loss term + horizon loop) | α schedule + (optional) σ̃_max bound | discriminator arch + adversarial scheduling |
| **Risk** | may under-train very-deep-horizon stability (grad only 1 step) | BPTT memory; α mis-set → kills learning or still explodes | adversarial instability; 3× cost; may not help < 100 steps |

**Reading.** B1 is the cheapest and closest to the current code (we already do the
hard part — detaching). B2 is the most *on-target*: the diagnostic says the runaway is
the feedback loop, and GTF is precisely "put **bounded** gradient back into the forced
feedback." B3 is the maximal, distribution-matching option — kept in the catalogue,
not proposed first (cost + the "no benefit < 100 steps" caveat, and our window may be
short).

### 4.3 B1 relative to ADR-056 scheduled sampling — what actually changes, and the gradient contract

*(Added 2026-06-06, pre-increment-2 design clarification — surfaced when implementation began.)*

**The existing scheduled sampling is already a proto-pushforward.** `_process_sequence`
(lines 191–200) with `ss_epsilon > 0` feeds `prev_pred = t1_pred.detach()` back as the
input (Bernoulli@`ss_epsilon`) and computes the per-step loss on the prediction made
*from that fed-back input*. Because only the **cross-step** link is detached — not the
current step's `model(t0_input, h)` call — **the model's weights already receive gradient
at the perturbed (own-prediction) operating point.** That is exactly Brandstetter's
recipe ("unroll, but backprop only the last step"). So we are **not** adding gradient
where there is none; we are **refining a term-less, Bernoulli-masked proto-pushforward.**

**Therefore "feedback gradient SEVERED" (§2) is precise but easily mis-read.** What is
severed is the **cross-step / through-time** feedback gradient (prev_pred → next input).
The **within-step** operator gradient (fed-back input → output → loss → weights) is live
whenever `ss_epsilon>0`. B1 does **not** restore the cross-step gradient — **B2 GTF does**
(it un-detaches and α-bounds it). This distinction sets the test contract below.

**What B1 (pushforward) changes over ADR-056:**
1. **Always-feed (not Bernoulli):** the stability evaluation uses the model's own
   one-step-prior prediction every step (or on an annealed schedule), not with prob
   `ss_epsilon` — so the operator is trained at the operating point it will actually
   occupy at inference.
2. **An explicit, annealed `L_stability` term** (weighted, → small; CRPS uncontaminated,
   R1) — vs SS's implicit "sometimes the input is a prediction."
3. **K-step coverage** via `rollout_horizon` (reach the step-12 onset).
4. Gradient stays **last-step-only** (detach across steps) — flat memory in K.

**The gradient contract increment-2 tests must assert (corrected):**
- ✅ **B1:** under `rollout_horizon>1`, `∂L_stability/∂(model params)` is **non-zero and
  finite** — i.e. the stability term trains the weights *at the fed-back operating point*.
  (NOT "the cross-step feedback gradient is live" — that would be testing B2.)
- The cross-step feedback **remains detached** under B1 (assert it, to keep memory flat
  and to keep B1 distinct from B2).
- `rollout_horizon=1` ⇒ byte-identical to today (parity).

**Decision (for the method review to contest):** B1 is the *minimal, honest refinement*
of machinery we already ship (ADR-056), so it is the right first rung — **unless** the
panel judges that the cross-step gradient (B2 GTF) is necessary from the outset because
the runaway rides a *through-time* dependency that last-step-only training cannot reach.
That is the live question for the `expert-method-review` below.

## 5. The `rollout_horizon` hyperparameter (K)

A single config knob, the "look-ahead depth" the user described (n-beats-like):

- `rollout_horizon: int = 1` → **byte-identical to today's one-step path** (parity
  guard in `03`). The feature is *off* by default until validated.
- `K ∈ {5, 10, 12, 36}` → unroll/penalise K autoregressive steps. **Candidate default
  K=12**, chosen to reach the empirically observed blow-up onset (~step 12 in the
  step-wise eval), not the full 36 — bounded GPU cost, covers the regime that breaks.
- **[R5] Check `seq_len` vs 36 *first* (Sutton's cheap lever).** If the training
  window is shorter than the 36-step inference rollout, the most general, least-clever
  "rollout training" is simply to **train on 36-step windows** — verify the current
  `seq_len` before implementing any loss term; it may dominate the fancy options.
- K interacts with the implementation:
  - **B1:** K = number of pushforward unroll steps (memory flat in K).
  - **B2:** K = the BPTT truncation window (memory linear in K → the cost driver).
  - **Temporal bundling (Brandstetter):** an optional second knob — emit a bundle of
    `b` steps per call so K rollout-steps cost `K/b` model calls. Deferred; logged.
- **[R2] The readout always runs the full 36 steps even when K<36** — see §8; a
  K-step training horizon must be *certified* against the full inference horizon.
- **α (B2 only):** train-only; annealed `1 → α* = 1 − 1/σ̃_max`. For a ConvLSTM, the
  exact σ̃_max is intractable; use the annealing heuristic (start strongly forced,
  decay) and monitor the per-step-loss curvature rather than computing the bound.

## 6. GPU-cost analysis

The crux of the user's concern. Per training example, window length `L`, horizon `K`:

| Strategy | Fwd passes | Backward | Activation memory | Notes |
|---|---|---|---|---|
| One-step (today) | L | 1-step (through `h`) | O(L) for `h` graph | baseline |
| **B1 pushforward** | L·(1 + 1) | K × one-step | **O(1) in K** | ~2× forward; **no extra graph** — this is why Brandstetter "don't backprop the first step" matters |
| **B2 GTF / TBPTT** | L | BPTT through K | **O(K)** feature maps | the expensive one; the ConvLSTM U-Net stores `[B,C,H,W]` per step × K |

**[R4] Concrete envelope — measure, don't assume.** The model runs at `window_dim=32`
(`diagnose_io_gain --hw 32`). A ConvLSTM U-Net stores `[B, C, H, W]` feature maps *per
step per skip-level*, so the K=12 BPTT footprint must be **measured** at the real batch
size before committing to B2 — the earlier "very likely feasible" was a hand-wave the
review (DL-engineer) rejected. **Plan if it OOMs:** `torch.utils.checkpoint` on the
per-step U-Net forward (recompute activations in the backward pass; ~30% compute for
O(√K) memory). Report the measured peak before any full retrain.

**Mitigations (in order of preference):**
1. **Pick B1 first** — flat memory in K sidesteps the problem entirely; only ~2×
   forward cost. Validate whether one-step-back stability training is *enough* before
   paying for BPTT.
2. **Truncated BPTT** — K=12 not 36 (TBPTT is exactly what the LSTM-TBPTT baselines in
   Hess Table 1 use).
3. **Gradient checkpointing** (`torch.utils.checkpoint`) on the per-step U-Net forward
   → recompute activations in the backward pass; trades ~1.3–2× compute for ~O(√K) or
   O(1) memory. The standard lever if K must be large.
4. **Temporal bundling** — fewer model calls per K rollout-steps.
5. **GTF's α-bounding is itself a stability mitigation** — it prevents the
   gradient-magnitude blow-up that would otherwise make deep-K BPTT diverge (the
   reason naive unrolled training "is hard to train," per Brandstetter §2.3).

**Honest cost statement:** B1 ≈ 2× current train time, no memory risk. B2 ≈
(1 + checkpoint_overhead)× compute and O(K) (or O(√K) checkpointed) memory — the only
option with a real OOM risk, mitigated by K and checkpointing.

## 7. Proposed path (updated 2026-06-05 per the `02b` method review)

0. **[R6] Sequence the whole program *after* the C-111 balancer verdict closes**
   (Operational seat) — don't tune rollout HPs atop the still-unattributed acute
   trigger. *(The in-flight sweep already shows freezing is seed-fragile —
   `seed4_frozen → inf` — which strengthens the case that exposure bias, not the
   balancer, is the root.)*
1. **Element 1:** kill `freeze_h` — **gated** behind the `rollout_horizon=1` parity
   guard **+ a golden_hour re-eval** (Operational), not flipped globally blind. Backed
   by the inert-freeze_h ablation. (Plan: see the freeze_h-kill plan, separate doc.)
2. **[R5] Verify `seq_len` vs 36** before writing any loss term — lengthening the
   training window may be the cheapest rollout training of all.
3. **Implement B1 (pushforward) behind `rollout_horizon`,** `K=1` parity-default —
   smallest diff to the current detached-feedback code, no memory risk, beats noise
   injection (Brandstetter). With the review's guardrails:
   - **[R2] readout certifies the full 36-step horizon** (in-range attractor + step-wise
     CRPS through *all* 36 steps), not just steps ≤K;
   - **[R1+R3] calibration & sharpness in the readout** — PIT/coverage + MCR/zero-rate,
     **not** attractor magnitude alone (the runaway is a *point* pathology; MCR is a
     *calibration* one — fixing one need not fix the other, and mean-hedging could
     worsen it);
   - **[R1] proper-score quarantine:** the `L_stability` weight is **annealed / kept
     small**; **CRPS is reported uncontaminated** as the headline (stability terms are
     regularisers, not proper scores);
   - **a cheap dose-response** (K∈{1,5,12} × stability-weight on the `diagnose_io_gain`
     proxy) *before* a full golden_hour retrain.
4. **If B1 under-delivers on deep-horizon stability,** escalate to **B2 (GTF)** —
   un-detach, soft-mix `(1−α)·pred + α·GT`, anneal α, TBPTT at K with checkpointing.
   With: **[R7] gradient clipping** (Pascanu/Hochreiter — standard companion to any
   BPTT); **α treated as heuristic gradient control, not chaos theory** ([R3], see §9
   Q1); **[R3-space] interpolation in the model's input space** (the gain was measured
   in log1p space — get the space wrong and α is meaningless); **[R4] measured K=12
   memory + checkpoint**. GTF also cleanly **subsumes and de-biases ADR-056 scheduled
   sampling** (soft α-mix replaces the hard Bernoulli mask).
5. **B3 (Professor Forcing)** stays catalogued; revisit only if both B1 and B2 fix
   *point* stability while the *trajectory distribution* (calibration) stays wrong.

## 8. Invariants this must respect (full set → `03`)

- **Parity:** `rollout_horizon=1` ⇒ training path byte-identical to today.
- **[R1] Proper score stays the headline:** the per-step predictive loss remains a
  proper scoring rule (CRPS); pushforward/GTF stability terms are **regularisers**,
  **annealed / small-weighted**, never a replacement for the score; CRPS is reported
  uncontaminated (the Gneiting constraint — see fault line).
- **[R1] Calibration is a first-class readout:** every rollout-training experiment
  reports PIT/coverage + MCR + zero-rate alongside the point/attractor metric.
  "C-113 solved" may **not** be declared on attractor magnitude alone.
- **No output capping:** stability comes from training dynamics, not magnitude clamps
  (ADR-028 §2 clamps stay a separate, last-resort fallback).
- **Feedback-gradient liveness:** a test asserting the fed-back operator's gradient is
  non-zero and finite under B1/B2 (the thing that is currently zero).
- **[R3-space] Feedback-space is explicit:** the fed-back quantity, the pushforward
  perturbation, and any GTF interpolation all live in the **same space the model
  consumes** (log1p space — where the io-gain `‖J‖₂>1` was measured), specified in the
  pre-analysis plan.
- **[R7] Gradient clipping** accompanies any path that backprops through the rollout
  (B2): standard BPTT hygiene (Pascanu 2013).
- **Train-only forcing:** α / pushforward affect training only; inference is unchanged
  free-running (so the io-gain diagnostic remains a valid before/after probe).

## 9. Open questions (seeded for the panel)

1. **[R3 — promoted to a B2 BLOCKER] Is conflict-count dynamics actually chaotic
   (`λ_max>0`)?** GTF's *theory* (the `α*=1−1/σ̃_max` bound) is only valid for chaotic
   systems. If our explosion is the milder Brandstetter distribution-shift problem (not
   Lyapunov divergence), B1 may suffice and B2's machinery is over-imported. **Must be
   resolved before adopting B2** — and even then α is used as heuristic gradient
   control, not as a correctness guarantee. *(Sutton / Gneiting; fetch Mikhaeil 2022.)*
2. **Does training-the-rollout fix point stability but *not* calibration?** The
   posterior-spread question is ADR-057's; if B1/B2 narrow the rollout but leave MCR
   wrong, that argues for B3 or the distributional head. *(Gneiting.)*
3. **K=12 vs 36:** does covering only the blow-up onset generalise to the full 36-step
   horizon, or does a new instability appear past K? *(Hochreiter / DL-engineer.)*
4. **Interaction with the C-111 balancer:** does rollout training make the active
   (learnable) balancer safe again — i.e. is exposure-bias the *root* and the balancer
   merely the *trigger*? *(ties to the balancer sweep — the in-flight sweep's
   `seed4_frozen → inf` already hints "yes, exposure bias is the root.")*

---

## 10. Review-driven revisions (folded in 2026-06-05 from `02b`)

The method-review verdict was *"design sound, layer right, ordering right — but not yet
experiment-ready."* These binding changes (tagged `[R…]` above) close that gap:

| Tag | Fix | From whom | Where |
|-----|-----|-----------|-------|
| **R1** | Calibration/sharpness (PIT, MCR, zero-rate) is a first-class readout; `L_stability` weight annealed/small; CRPS reported uncontaminated | Gneiting | §7.3, §8 |
| **R2** | Readout certifies the **full 36-step** horizon even when K<36 | Hochreiter | §5, §7.3 |
| **R3** | Chaos premise promoted to a **B2 blocker**; α = heuristic gradient control, not theory; interpolate in the model's input space | Sutton, Gneiting, Shi | §4.2, §7.4, §9-Q1 |
| **R4** | K=12 memory **measured, not assumed**; checkpointing as the explicit plan | DL-engineer | §6 |
| **R5** | Check `seq_len` vs 36 first — lengthening the window may be the cheapest fix | Sutton | §5, §7.2 |
| **R6** | Sequence the program **after** the C-111 balancer verdict closes | Operational | §7.0 |
| **R7** | Gradient clipping accompanies any rollout-BPTT path (B2) | Hochreiter | §8 |

**Strongest live dissent carried forward (D5):** fixing the *point* runaway may not fix
— and mean-hedging/blurring could worsen — the *chronic* MCR/calibration problem
(ADR-057). This is a falsifier in the pre-analysis plan (`05`), not a footnote.

The six methodological risks (M-RT1…M-RT6) are filed via `register-risk` (see the
technical risk register).
