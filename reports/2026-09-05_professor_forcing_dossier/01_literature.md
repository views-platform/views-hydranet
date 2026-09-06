# 01 — Literature

**Written:** 2026-09-05. Sources are in `~/brain/9_library`; claim IDs are the library's.

## Load-bearing

### `Lamb2016_ProfessorForcing` — the method

An adversarial method for improving long-term generation from RNNs. A **discriminator** takes a
*behaviour sequence* `b` — chosen hidden states and output values — and classifies whether it was
produced in **teacher-forcing** mode (inputs clamped to a training sequence) or **free-running**
mode (inputs self-generated). The generative RNN is trained, in addition to its own likelihood, to
**fool** that discriminator.

**What we take from it:**

* **The target is the hidden-state distribution**, not the emitted field. This is the whole reason
  the paper is here: it is the only candidate that attacks the quantity M50/M54/M60 measured.
* **The discriminator is bidirectional** in their experiments, "so that it can combine evidence at
  each time step *t* from the past of the behaviour sequence as well as from the future."
  *Adoptable but expensive for us — see the design fork in `02`.*
* **Two generator objectives**, and the choice is ours: `(a)` the negative log-likelihood of the
  discriminator being wrong, and `(b)` a term that **only changes the free-running behaviour** to
  match the teacher-forced. They train on `NLL + C_f` or `NLL + C_f + C_t`. The `C_f`-only variant
  matters to us: it leaves the teacher-forced pass — our entire current training signal — untouched.
* **Reported result:** generalises from 50-step training segments to coherent 1000-step generation,
  and is rated better than teacher forcing in **76.9%** of comparisons.
* **Explicitly distinguished from GANs:** the classifier discriminates hidden states from the two
  *modes*, not real samples from generated ones. A practical consequence they name — it avoids
  backpropagating through discrete sampling — is **also why it suits us**: our feedback field is a
  draw from a discrete count distribution, and #308 (M63) established that our straight-through
  approximation of that draw's gradient, while not backwards, was the wrong thing to be optimising.

**What it does not tell us:** anything about a 180×180 spatial field, a 99.94%-zero target, or a
36-step horizon. The paper's sequences are text and speech.

## Read and DISQUALIFIED — do not re-propose

### `Zhuang2025_HorizonForcing` (ACM TIST) — ETT + HF

Claim **C-464** records that ETT+HF outperforms teacher forcing, scheduled sampling **and Professor
Forcing** on three chaotic systems and six real-world tasks. #309 flagged it as a caveat that might
redirect this build. **It was read in full on 2026-09-05 and does not transfer.** Three obstructions,
in severity order:

1. **The objective degenerates on our data.** HF optimises a *Lyapunov Horizon Loss* whose term is
   `log(‖d_t^k‖ / ‖d_t‖)` — the ratio of k-step-ahead error to one-step error, a finite-time
   Lyapunov exponent estimate. It presumes a chaotic system with a defined exponent (the paper
   quotes Lorenz's, ≈0.906). Our field is **99.94% exactly zero** and mostly *correctly predicted*
   zero, so `d_t → 0` across the modal case and the ratio is undefined. This is not a tuning
   problem; the denominator is our data.
2. **It is an architecture, not a training regime.** HF requires the **ETT "tower"** — a parallel
   error-tracing pathway with shared weights, unrolled to the horizon, plus a modified BPTT
   (their §3.5.2). Algorithm 1 builds towers layer by layer. Adopting it replaces the head *and*
   the training loop.
3. **Scale.** Validated on GRU cells over Lorenz / Rössler / Lotka-Volterra (3-D) and six
   univariate series, at horizons `n × k` with `n ∈ [1,4]`, `k = 5` — about 20 steps. We need 36 on
   a 180×180 ConvLSTM.

**C-464 is real evidence about those benchmarks and is not evidence about ours.** Recorded here so
the caveat is answered once rather than re-raised.

*Worth keeping from it anyway:* claim **C-465** — sharing weights between the forecasting and
error-tracing pathways means minimising long-term error also improves single-step error, "because in
chaotic systems controlling long-term error necessitates controlling short-term error." That is a
useful prior against the fear that a rollout-targeted objective must cost us h1.

## Context we already hold

| source | role here |
|---|---|
| `Aceituno2025_TemporalHorizons` | **proven**: long-horizon minima generalise short, not vice versa; gradient scales `O(e^{λT})`. Explains #308's explosion and prices PF's free-running unroll. |
| `Bengio2015_ScheduledSampling` | the baseline PF is defined against; Lamb notes it "is not a consistent estimation strategy". Our own M30–M33 closed it independently (ε=0.5 at L=300, −0.0426 AP@h18). |
| `SanchezGonzalez2020_GraphNetworkSimulators` | #311's source. Its "no target adjustment" advice is what M64's postmortem showed does not survive the change from symmetric jitter to deletion. |
| `Brandstetter2022` (pushforward) | implemented, audited, **never run**. Its code branch is the seam PF reuses. |

## Gaps to fetch or extract

- [ ] **`Salinas2019_DeepAR`** — PDF in the library, **zero claims extracted**. Carries the standard
      justification for single-sample feedback, which is load-bearing here via ADR-070.
- [ ] **Lim et al. 2021, Temporal Fusion Transformer** — **absent**. Needed only if #310 is
      un-parked; not a blocker for this dossier.
- [ ] `Lamb2016_ProfessorForcing` has claims in the library but **no key passages**. Worth extracting
      the two generator objectives verbatim before `02_design` fixes a choice between them.
