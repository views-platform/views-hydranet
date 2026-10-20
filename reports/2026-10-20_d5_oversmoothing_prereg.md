# D5 — pre-registration: is the over-smoothing ARCHITECTURAL or LEARNED?

**2026-10-20, written and committed BEFORE either measurement runs.** Free, CPU, no pod, no
training, no fetch. Scripts: `2026-10-12_underfire_diagnosis/d5_why_smooth.py`.

## The question D4 opened and did not answer

`2026-10-20_d3_static_channel_ceiling.md` §9 measured both heads emitting fields far smoother than
the process: body `log mu` lag-1 autocorrelation **0.8594 / 0.7636** in sample and **0.9266 / 0.9227**
out of sample, gate logit **0.9598 / 0.9574** and **0.9498 / 0.9464**, against the truth's
noise-controlled reference of **0.6419 / 0.6261**. §9 states explicitly that **why** is untested, and
lists four live candidates: the decoder's receptive field, the recurrent BatchNorm recalibration
(C-184, fitted on 32×32 windows and applied to 360×720), `weight_decay: 0.1` (global —
`utils.py:276-281` applies it to `unet.parameters()`, and the three class heads are **not** in the
`weight_decay: 0.0` exclusion groups at `training_engine.py:113,136`), and the objective's own
preference.

⛔ **Three of those four are training-side and one is not.** One measurement separates them.

## D5-A — the same architecture, untrained

Build `choose_model(cfg)` from the **same sidecar config** and do **not** load the state dict. Run
the identical 49-step sweep over the identical land mask. Measure lag-1 spatial autocorrelation of
`log mu` and the gate logit.

### Pre-committed readings — one of these three, chosen now

| init autocorrelation | reading |
|---|---|
| **≥ 0.85** (inside the trained range 0.86–0.96) | **The architecture imposes the smoothness floor.** The trained value is then **not** evidence of a learned pathology, and all three training-side candidates are **demoted** — they would be explaining something that was already there. |
| **≤ 0.65** (at or below the truth's noise-controlled 0.64) | **The architecture does not impose it ⇒ the smoothness is LEARNED.** BN recal, global weight decay and the objective stay live, and the next experiment is E6, which separates the first two for free on CPU. |
| **0.65 – 0.85** | ⛔ **INCONCLUSIVE.** Reported as inconclusive with no mechanism claimed. No post-hoc reinterpretation of the band. |

### ⛔ VOID conditions, committed now

1. `sd(log mu) < 0.01` or `sd(gate logit) < 0.01` at init. An almost-constant field makes lag-1
   autocorrelation numerically meaningless, and a random-init network through `softplus` is a real
   candidate for exactly that. A degenerate field is **VOID, not smooth.**
2. Any non-finite value in either field.

### ⚠️ The limit, stated before the number exists

**C-325** records that two of three earlier mechanism tests were void because they measured a network
**at initialisation**, where an instability that *training builds* cannot appear. D5-A is **not** that
shape — it measures a **floor the architecture imposes before training**, which is precisely what init
can measure — but the asymmetry must be respected in both directions:

- A **smooth** init shows the architecture **permits and defaults to** smoothness. It does **not**
  show training added nothing; training may have made it smoother still.
- A **rough** init shows the architecture does **not force** smoothness. It does **not** prove any
  particular training-side cause; it only keeps all three alive.

Neither outcome identifies a mechanism. D5-A **narrows the candidate set**; it does not close the
question, and under **C-307** an n=1 screen with no falsifier on the outcome closes nothing anyway.

## D5-B — how far does one input cell reach in one step? (trained weights, descriptive)

With the **trained** vehicle, run one step on the real volume, then re-run with a single land cell's
`sb` input raised by +1 in `log1p` space, and measure `|Δ log mu|` against distance from that cell.
Repeated over a handful of well-separated cells and pooled.

⛔ **No falsifier and no threshold is pre-committed for D5-B, because I have no prior for what the
number should be.** It is reported as a description — the radius at which the response falls to 10%
of its peak — and **may not be converted into a pass/fail after the fact.** It avoids C-325 entirely
by using the trained weights, which is why it is here at all.

## What D5 cannot do

It cannot tell us whether the over-smoothing *costs* anything. That needs the composed forecast
scored, which is `crps`/AP on a rollout, not a forward pass. Nothing here touches the C-152 class
statement: a proposal that sharpens the field still inherits that null unless it changes the joint or
the sampler.
