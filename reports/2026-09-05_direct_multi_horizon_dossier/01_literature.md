# 01 — Literature

**Written:** 2026-09-05. Claim IDs are the library's (`~/brain/9_library`).

## The blueprint

### `Wen2017_MQRNN` — Multi-Horizon Quantile Recurrent Forecaster

**C-525** — three components: an **LSTM encoder** producing a fixed-length context vector from
history, plus a decoder. **C-527** — the **global decoder** `f_θ(c_t, x_{t+k})` takes the encoder
context *and horizon-specific covariates* and outputs estimates for **all desired horizons**.

**This is the design.** Mapped onto HydraNet: `c_t` is the ConvLSTM state at the origin, `x_{t+k}`
is the static channels plus a horizon encoding, and the "global decoder" is the existing U-Net.

**C-526 — forking sequences.** Every time step in the history is a fork point; at each fork the
decoder predicts the next K horizons. This is what stops a direct head from throwing away training
positions, and it is the answer to *"won't you have far fewer training examples?"*

**C-528** — MQ-RNN is evaluated by ablation against its own variants, including **"MQ RNN cut (no
forking)"**. That ablation is worth copying: it isolates the training scheme from the architecture.

**C-529 — the one thing that does NOT transfer.** MQ-RNN's quantile predictions are not constrained
monotone, so **quantile crossing is possible**. Our head is distributional (NB/ZINB) and emits
*parameters*, not quantiles, so the crossing problem does not arise. Recorded so nobody imports a
fix for a problem we do not have.

## The platform precedent — and the direct answer to the chair's founding worry

### `VonDerMaase2025_ViEWSPipelineHandbook`

**C-118** — ViEWS 2020 **hardcoded** a Direct Multi-Step Forecasting architecture via the
`stepshifter` wrapper, **requiring 36 separate submodels per model specification**, which constrained
the system to a single strategy.

**This is exactly the failure mode the chair named**: *"if we train 36 individual models (like
views-stepshifter) then we lose the joint and every step is completely marginal."* The concern is
correct **about that implementation** and does not apply here: MQRNN's contribution is **one**
encoder and **one** decoder with horizon as an input, so all horizons share every weight. What is
lost is joint *coherence* across horizons, not shared learning — a real cost, stated in `02`.

**C-119** — the updated pipeline is deliberately **forecasting-model-agnostic and supports both
direct and recursive strategies**. Nothing downstream blocks this.

**C-120/C-121** — the pipeline has drift detection and a gated production/development merge. A
direct head changes the shape of what is delivered, so the drift baselines move; that is an
operational cost to schedule, not a blocker.

## The theory

### `Aceituno2025_TemporalHorizons`

**C-459** — minima found training on **long** horizons generalise well to short-term forecasts;
minima found on **short** horizons **do not** generalise long. We currently train on one-step
likelihood and grade on 36-step AP, which is the wrong side of that asymmetry.

**C-458** — the gradient scales `O(e^{λT})` with the horizon trained over. This is why one cannot
simply backprop through a 36-step rollout, and it is the measured cause of #308's death (**M61**:
pre-clip norm 133,465 → 9.4e9, lesson 48). **A direct head sidesteps it entirely**: there is no
product of 36 Jacobians, because there is no chain — each horizon is one forward pass from the
origin state.

⚠️ **C-461 — the honest caveat, and it must not be dropped.** The theory is derived for
**autoregressive feedforward networks**; extension to RNNs "requires additional mathematical
treatment." Aceituno is **directional support, not proof**, for a ConvLSTM.

**C-460** — the loss landscape roughens as the training horizon grows. A direct head trains at all
horizons simultaneously, so this is a cost we adopt knowingly, not one we escape.

## Supporting

| source | what we take |
|---|---|
| `Salinas2019_DeepAR` | **C-2339** — an LSTM emitting *likelihood parameters*, trained jointly across many related series: the lineage our NB/ZINB head already sits in. **C-2340** — velocity-dependent scaling for power-law magnitudes, relevant to a 99.94%-zero field. *(#310's note that this paper has "zero claims extracted" is **stale** — C-2339..C-2342 exist.)* |
| `Hafner2020_Dreamer` | **C-2525** — RSSM produces coherent **45-step open-loop** video prediction from 5 context frames, i.e. **longer than our 36 with nothing fed back**. Adjacent evidence that a learned state can carry a long horizon open-loop. |
| `Hegre2019_ViEWS` | dynamic simulation is one of three estimation strategies, not the mandated one. |
| `Hess2023_GeneralizedTeacherForcing` | **C-499/C-502** — the fallback if a direct head disappoints and a *training-time* state intervention is still wanted. Filed as #294. |

## FETCHED 2026-09-05 — and it changes the design

### `Shi2017`, *Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model*

Read in full (17pp, `~/brain/9_library/incoming/`, not yet ingested). Three things transfer, and the
first reopens the central design fork.

**1. §3.1, the encoding-forecasting structure — a THIRD option, distinct from both the incumbent and
the MQRNN design in `02`.** Verbatim:

> *"Our encoding-forecasting network first encodes the observations into n layers of RNN states:
> `H_t^1, ..., H_t^n = h(I_{t-J+1}, ..., I_t)`, and then uses another n layers of RNNs to generate
> the predictions based on these encoded states: `Î_{t+1}, ..., Î_{t+K} = g(H_t^1, ..., H_t^n)`."*

A **separate forecasting RNN stack** produces all K predictions from the encoded states. **No input
feedback anywhere** — the forecaster's own recurrence carries it from horizon k to k+1, and a
prediction is never fed back as an input.

This matters because it **removes the cost `02` identified as the real trade**. The MQRNN design
decodes each horizon independently from the origin state, so horizons are *marginal* — nothing links
month 7 to month 8. The encoder-forecaster deletes exposure bias **and keeps horizon-to-horizon
coherence**, because the state still evolves; it is simply never contaminated by a prediction.

The cost it reintroduces: a K-step chain for gradients, i.e. Aceituno **C-458**'s `O(e^{λT})` and
#308's ghost. Mitigating evidence — Shi trains exactly this at **K = 20** without special measures,
and the chain carries *state*, not fed-back predictions, so the error-compounding is different in
kind from the incumbent's.

**A non-obvious architectural detail worth having:** the forecasting network's layer order is
**reversed** relative to the encoder, *"because the high-level states, which have captured the global
spatiotemporal representation, could guide the update of the low-level states"* — and it removes the
need for skip connections to aggregate low-level information.

**2. B-MSE / B-MAE — the nearest published treatment of a heavily-imbalanced field, and the finding
is stronger than "it helps".** Weights are assigned per pixel by intensity band —
`w(x) = 1, 2, 5, 10, 30` for `x < 2, [2,5), [5,10), [10,30), ≥ 30` — with masked pixels at 0.

> *"training with the balanced loss functions is **essential** for deep learning models to achieve
> good performance at higher rain-rate thresholds."*

And the sharp version: **ConvGRU trained without the balanced loss — the configuration that "best
represents" the original ConvLSTM paper — scores *worse than the optical-flow baselines* at the
10 mm/h and 30 mm/h thresholds.** On an imbalanced field, an unweighted loss makes a deep model lose
to a simple baseline **precisely on the rare, high-impact events**.

That is our situation (99.94% zeros; the rare events are the entire point), and it bears directly on
`02`'s **F5** (per-horizon loss weighting) — which should now be read as *per-horizon **and**
per-intensity* weighting.

**3. Seed discipline, comparable to M65.** Shi trains each model at **3 random seeds** and treats a
difference as significant only when it exceeds **three times the standard deviation**. Independent
support for this dossier's paired 2-seed design over the repo's habitual n=1.

*Noted, not adopted:* **TrajGRU** learns *location-variant* recurrent connection structure, on the
argument that convolutional recurrence is location-invariant while real motion is not. Interesting
against **M54** (our state holds a spatial map), but a larger architectural change than this epic
scopes.

### `Shi2015_ConvolutionalLSTMNetwork` — also fetched

The backbone's original paper, fetched to close the same gap. Not load-bearing for this design;
ingested for the record.

## Remaining gaps

### Still missing


1. **`Shi2017`, *Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model*** —
   **ABSENT, and so is every other primary ConvLSTM/nowcasting source** (no Shi 2015, no DGMR, no
   PredRNN, no Earthformer) across 567 held papers. Its **encoder-forecaster** protocol is the
   closest published precedent to this exact design — a forecasting network unrolled with **no input
   feedback**, only state carried forward — and its B-MSE/B-MAE rare-event weighting is the nearest
   published treatment of a heavily-zero field. **We would be designing our backbone's successor
   with our backbone's own paper missing from the evidence base.** Blocks the design freeze.
2. **Lim et al. 2021, Temporal Fusion Transformer** — absent. The modern standard for direct
   multi-horizon with known-future covariates, and the natural comparison to MQRNN. Not a blocker.
3. **`Wen2017_MQRNN` has claims but no key passages.** Extract the global-decoder and
   forking-sequences formulations verbatim before `02` freezes.
