# 02 — What to build next: the training/deployment gap

**2026-09-03.** Prompted by the chair: *"the elephant in the room is still that we never train beyond
first step, so it's barely machine learning out here. How do other autoregressive models handle
this?"*

Grounded in `~/brain/9_library` via `/library search`. Claims below are the library's **extracted**
claims, not independently verified by me — verify before any of them becomes a design commitment.

## The gap, stated plainly

The model is trained one step ahead and deployed 36 steps ahead on its own output. Everything this
programme has measured is damage from that mismatch: the field collapses on both axes (M51), the
cell state drains (M50), and the one intervention that works (M48/M56) is an **inference-time patch**
that hands back a state the training never taught the model to maintain.

## What this programme has already tried, and how it failed

| approach | source | outcome here |
|---|---|---|
| **Scheduled sampling** | `Bengio2015_ScheduledSampling` | **M26–M33: HURT AP.** ε=0.5 at L=300, 4v4, AP@h18 −0.0426. It largely *fixed* the zero collapse and still lost skill. |
| **Increasing teacher forcing (ITF)** | `Teutsch2022_FlippedClassroom` | **M45: −0.019.** Also records that scheduled sampling can cause *premature termination* on chaotic series. |
| **Pushforward trick** | `Brandstetter2022_MessagePassingPDE` | **M47: UNDERPOWERED-negative.** |
| **Generalized teacher forcing (α)** | `Hess2023_GeneralizedTeacherForcing` | **#294: withdrawn** — σ_max was measured on a model never GTF-trained, so α was not derivable from it. |

Four attacks on the same gap, four failures. That is the context for anything proposed below.

## Three candidates not yet tried

### 1. Professor Forcing — the closest fit to what we measured

`Lamb2016_ProfessorForcing`. An adversarial discriminator forces the **distribution of hidden states**
under free-running to match the teacher-forced distribution.

*Why it fits:* M51/M50/M60 localised the failure to the **recurrent state drifting when self-fed** —
which is exactly the quantity Professor Forcing constrains. Every failed attempt above operated on
the *fed-back field*; none operated on the state. The library records it generalising from 50-step
training segments to coherent **1000-step** generation, rated better than teacher forcing 76.9% of
the time — the horizon-extension property this programme needs.

*Cost:* a discriminator plus adversarial training. The heaviest of the three.
*Caveat:* `Zhuang2025_HorizonForcing` claims to **beat** Professor Forcing on chaotic benchmarks.

### 2. BPTT-SA — and it explains why scheduled sampling failed here

`Vlachas2023_LearningFromPredictions`. Scheduled sampling feeds back model outputs but treats them as
**constants** — no gradient flows through them. BPTT-SA changes the computational graph so gradient
flows **through the predicted outputs**.

*Why it fits:* this is a mechanical explanation for M26–M33 rather than a new hope. The library
records BPTT-SA reconciling short- and long-term accuracy *more effectively than Bengio (2015)*,
**without extra training cost**, and being **more pronounced on high-dimensional spatiotemporal
systems** than low-dimensional chaotic ones — ours is 180×180×3.

*Feasibility is already established here:* the 2026-08-26 training-loop audit measured gradient
reaching back **118 steps** (`d‖h_final‖²/dx_i` = 1.6e-02 trained vs 2.8e-17 at init), with no BPTT
truncation anywhere. The graph BPTT-SA needs already exists.

*Cost:* the cheapest of the three — a change to what the rollout graph retains, not a new component.

### 3. Delete the problem: direct multi-horizon

`Wen2017_MQRNN`. An encoder produces a context vector; a global decoder MLP emits **all horizons
directly** from that context plus horizon-specific covariates. No recursion, therefore **no exposure
bias at all**.

*Why it deserves to be on the list before more mitigation:* `VonDerMaase2025_ViEWSPipelineHandbook`
records that the VIEWS pipeline is forecasting-model-agnostic and **already supports both direct and
recursive strategies**. Everything this programme has measured is damage from a recursion the
platform may not require.

*Cost:* a different head and training target — the largest architectural change, and it forfeits the
recurrent state that M54/M60 show is carrying a genuinely useful spatial map.

## Recommendation

**BPTT-SA first.** Cheapest, its enabling condition (gradient reach) is already *measured* here rather
than assumed, and it comes with a mechanical account of why the four previous attempts failed — so a
negative result would also be informative. Professor Forcing second, because it targets the state,
which is where every measurement in this dossier points. Direct multi-horizon is the one to put in
front of the chair as a **strategy** question rather than an experiment.

⚠️ All three are **training-time** changes. Every result in this dossier is inference-time and
retrain-free; none of it transfers automatically, and C-112 applies — changing training dynamics
makes pre/post metrics incomparable.
