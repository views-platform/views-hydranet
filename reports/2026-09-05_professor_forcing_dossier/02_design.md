# 02 — Design

**Status: STUB, deliberately.** This document is the blocker for everything downstream, and it
should not be written by the same pass that scaffolded the dossier. The forks below are the ones an
`expert-method-review` panel needs to rule on **before** `05_analysis_plan` commits to anything.

## The one variable

A flag — provisionally `professor_forcing_weight`, default `0.0` — that adds an adversarial term to
the generator loss, computed from a free-running forward's hidden states against the teacher-forced
ones.

## Open forks, none of them settled

| # | fork | why it is live |
|---|---|---|
| **F1** | **Which half of the state to discriminate: the cell (`hl_*`), the hidden (`hs_*`), or both?** | Every measurement implicates the **cell** (M50/M54/M55/M60), and #309 argues for constraining it. But the cell is also what M55 shows *working*; constraining the thing that works is a real risk, not a formality. |
| **F2** | **How long a free-running segment?** | PF's paper uses full sequences. Our pushforward seam uses **one** step. One step is cheap and stable; 36 steps is what exploded in #308. The segment length is the exposure–stability dial and **must be a config field** (C-85). |
| **F3** | **`C_f` only, or `C_f + C_t`?** | `C_f` changes only the free-running behaviour and leaves the teacher-forced pass — our entire current training signal — untouched. `C_t` also pulls the teacher-forced states toward the free-running ones, which on this vehicle could degrade h1. The paper offers both. |
| **F4** | **Discriminator architecture.** | The paper uses a bidirectional RNN over the behaviour sequence. On a 180×180×C state that is a second recurrent net over spatial fields — expensive. A per-timestep convolutional discriminator is far cheaper but forfeits the "combine evidence over time" property the paper credits. |
| **F5** | **Does the adversarial gradient reach the recurrence, or only the emission?** | The pushforward seam has exactly this fork already (`pushforward_detach_state`) and its comment records that the paper cannot settle it. Here it is sharper: the whole point is to train the recurrence. |
| **F6** | **What makes a run VOID.** | Discriminator accuracy has two degenerate ends — 0.5 (learned nothing) and 1.0 (won; no generator gradient). The band, and the response when it is breached, must be pre-registered, not discovered. |

## The prediction that would make this worth running

Stated now so it can be attacked: **if PF works, `act_ratio` should NOT rise.** Six interventions
have failed by making the model fire more (M45, M62, M63, M64 and two others). PF's objective
contains no term rewarding occurrence. If a PF arm loses AP *and* `act_ratio` rises ×3 or more,
that is the seventh instance of the same failure and the mechanism story is that **any** perturbation
of this training loop lands on the firing lever — which would be a more important finding than PF
itself.

## Explicitly out of scope

Direct multi-horizon (#310 — a chair decision, parked); Horizon Forcing (disqualified, `01`);
unbiased reparameterised feedback (#308's open follow-up); any 4-seed confirmation, which is
conditional on the screen.
