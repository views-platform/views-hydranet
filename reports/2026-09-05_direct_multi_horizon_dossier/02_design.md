# 02 — Design

**Written:** 2026-09-05. **Not yet method-reviewed** — `expert-method-review` precedes
pre-registration, and the open forks below are what it should rule on.

## The shape

**Today:** the encoder digests history to a state `h`, then a loop runs 36 times, each step feeding
its own prediction back in and evolving `h`.

**Proposed:** the encoder digests history to a state `h` at the origin. Then **every horizon is
decoded from that same state**, with the horizon supplied as an input covariate. Nothing is fed
back. `h` never evolves.

```
history ──► ConvLSTM encoder ──► h (origin)
                                   │
              ┌────────────────────┼────────────────────┐
            h, k=1               h, k=18              h, k=36     ← the SAME h
              │                    │                    │
            U-Net                U-Net                U-Net       ← the SAME weights
              │                    │                    │
             ŷ₁                   ŷ₁₈                  ŷ₃₆
```

## Why `forward(x_k, h_origin)` × K, and not "emit 36 horizons at once"

`views_hydranet/architectures/registry.py` pins the contract every architecture must satisfy:
`forward(x, h) -> ModelOutput` with **`reg` of width `n_targets * n_params`**, `cls` of width
`n_targets`, `h_next` the same shape as `h`, `total_hidden_channels` divisible by 8, and a `base`
attribute.

A head emitting all K horizons **breaks that contract**, and with it the loss loop, the inference
path and the scoring path — a far larger change than "a head plus a loss", and one that would touch
every existing architecture.

**MQRNN's own formulation avoids it (C-527):** the global decoder is `f(c_t, x_{t+k})` — context
plus *horizon-specific covariates*. Calling `forward(x_k, h_origin)` K times with the **same** state
and a horizon covariate in `x` **preserves the registry contract exactly**, while still deleting the
recursion: nothing is fed back, the state does not evolve, and horizon identity arrives as an input.

**This is the single most important design decision in the dossier**, because it converts a
cross-cutting rewrite into a change with a seam.

## The seam

`FiLMSkip(HydraBNUNet06_LSTM4)` in `views_hydranet/architectures/dynamic_skip.py` establishes the
pattern: **architecture variants are subclasses registered in `registry.py`**, never edits to the
base class. A direct-horizon variant follows it, so the incumbent stays byte-identical and
unselected configs are untouched.

The FiLM primitive itself is a candidate for the horizon conditioning (`self.film` produces per-channel
`gamma, beta`, zero-initialised so it starts as a no-op) — **F3 below**.

## What is kept, what changes, what is lost

**Kept:** the ConvLSTM encoder and the spatial map it builds (M54/M55); the U-Net decoders; the
distributional heads and `n_params` layout; gate/body composition (ADR-069); the entire scoring path.

**Changed:** (1) the inference loop stops feeding back and stops evolving the state; (2) training
adds the horizon dimension — every origin supplies K targets, not one (**forking sequences**,
C-526); (3) one horizon encoding enters the input.

**Lost, and stated plainly:** each horizon is predicted **marginally**. Month 7 and month 8 share
every weight and the same context, but nothing enforces that they are *jointly coherent*. The
current rollout gets that coherence for free from the recursion — and pays for it with the collapse
this programme exists to fix. **This is the trade, and it is not free.**

## The correction this dossier makes to #310

#310 was filed saying a direct head "throws away" the recurrent state, and listed that as the reason
it needed a chair decision. **That is overstated.** The encoder is unchanged; the map it builds at
the origin is unchanged. What is deleted is the state's *evolution across the forecast* — which
**M50** measures draining 41× and which **M48/M56** shows is better held still.

**Direct multi-horizon is the cell clamp taken to its limit.** ADR-027 §2.1 shipped "stop the state
drifting" today; this is "do not evolve it at all."

## The baseline moved today

The control is **no longer** the unclamped rollout. ADR-027 §2.1 shipped the clamp, so the honest
comparison is against the **clamped** arm:

| | AP@h18 | AP@h36 |
|---|---|---|
| unclamped (the old control) | 0.3257 | 0.2250 |
| **clamped — the control now** | **0.3624** | **0.2841** |

A direct head that merely matches the clamp has bought nothing but simplicity. **Pre-register against
the clamped arm.**

## Open forks for the method review

| # | fork | why it is live |
|---|---|---|
| **F0** | **⚠️ THE PRIMARY FORK, opened 2026-09-05 by reading `Shi2017`. Marginal decode (B) or encoder-forecaster (C)?** See the table below. **This supersedes "The shape" above, which described (B) as settled. It is not.** |
| **F1** | **How is the horizon encoded?** A scalar broadcast channel, a learned per-horizon embedding, sinusoidal, or FiLM modulation off a horizon index. FiLM already exists here and is zero-initialised. |
| **F2** | **Which state is the context — the full `h`, the cell half, or the hidden half?** The clamp evidence says the **cell** carries placement (M54/M60); the encoder feeds the U-Net only `hs` (`:604`). This is the same disagreement the PF panel split on and it recurs here. |
| **F3** | **Does `h_origin` get gradient from all K horizons?** It must, or the encoder never learns to build a state that serves horizon 36. But that is K paths into one tensor — a gradient-magnitude question with #308's ghost on it, even though there is no *chain*. |
| **F4** | **Which horizons are trained?** All 36 every step is K× the decoder cost. Sampling a subset per step is cheaper and is closer to forking sequences. Cost model required before this is settled. |
| **F5** | **Is the loss per-horizon-weighted?** Equal weight lets 36 easy long horizons swamp h1. Aceituno C-459 argues long horizons carry the short ones; that is a testable prediction, not an assumption. |
| **F6** | **What replaces the rollout in evaluation?** The scoring path assumes a 36-step cube from a rollout. A direct head produces the same cube by a different route — the ruler must be proven to treat them identically before any arm is scored. |

### F0 in full — the three architectures

| | how the 36 horizons are produced | exposure bias | horizon coherence | gradient chain |
|---|---|---|---|---|
| **(A) recursive** — today | loop 36×, each step fed its own prediction, state evolves | **present, and it is the failure** | yes | 36 steps + feedback |
| **(B) marginal decode** — MQRNN, this doc as first written | K decodes from the **same** origin state, horizon as covariate | **absent** | **no — each horizon is marginal** | 1 step |
| **(C) encoder-forecaster** — `Shi2017` §3.1 | a **separate forecasting RNN stack** unrolls K steps from the encoded states, **no input feedback** | **absent** | **yes — the forecaster's own recurrence carries k→k+1** | K steps, but of *state*, not of fed-back predictions |

**(C) removes the cost this document identified as the real trade** — marginal horizons — while
still deleting exposure bias. It is what the ConvLSTM seat meant by *"the nowcasting-native answer is
the parked one"*, and it is a published, benchmarked protocol rather than an adaptation of one.

**What (C) costs:** a K-step gradient chain — Aceituno **C-458**'s `O(e^{λT})` and #308's ghost; a
second RNN stack rather than a horizon covariate, so a larger build; and it does **not** preserve the
registry contract as cleanly as (B).

**What argues (C) is nonetheless safe:** Shi trains it at **K = 20** with no special measures, and
the chain carries *state* only — a prediction is never an input, so the compounding is different in
kind from (A). Our K is 36.

**A cheap discriminator, available before committing:** **(B) is a strict special case of (C)** with
the forecaster's recurrence disabled. Building (C) behind a flag that zeroes the forecaster's state
transition yields (B) for free, making **F0 an ablation rather than a bet** — and `Shi2017`'s own
`ConvGRU-nobal` ablation is the precedent for isolating one design choice exactly this way.

**F5 is upgraded by the same paper.** Shi's B-MSE/B-MAE weight each pixel by intensity band
(`w = 1, 2, 5, 10, 30`), and he reports that *without* it a ConvGRU scores **worse than optical flow**
at the rare high-intensity thresholds. On a 99.94%-zero field, F5 must be read as **per-horizon *and*
per-intensity** weighting, not horizon weighting alone.

## What would falsify this before it is built

Pre-registered now, so it can be attacked:

* **If a direct head's `act_ratio` rises ×3 and AP falls**, that is the seventh instance of M45 — and
  because a direct head **never feeds itself**, it would exonerate the feedback loop entirely and
  relocate the firing lever to the *data* (99.94% zeros). **That would be a more important finding
  than the architecture**, and it is the reason this is worth running even if it loses.
* **If a direct head matches the clamped control at every horizon**, the recursion was never the
  binding constraint and this programme should stop attacking it.

## Out of scope

Professor Forcing (#309, closed — its own paper reports no benefit below 100 steps); Horizon
Forcing (disqualified); GTF (#294, the fallback if this disappoints); the magnitude ceiling (#241).
