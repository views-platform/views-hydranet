# Professor Forcing — constrain the free-running STATE, not the fed-back field

**Issue:** [#309](https://github.com/views-platform/views-hydranet/issues/309) · **Opened:** 2026-09-05 · **Status:** **METHOD-REVIEWED — the panel ruled AGAINST building PF first.** Nothing implemented, nothing run. See `02_design.md` for the rulings and the sequence the panel converged on.

## Purpose

Six interventions have now attacked the training/deployment gap. Every one of them operated on the
**fed-back field** or on the gradient flowing through it, and four of them failed *on the same
lever*: **AP loss scales with how much the model fires** (M45, and M62/M63/M64 since).

Professor Forcing (`Lamb2016_ProfessorForcing`) operates somewhere else. It leaves the feed and the
target untouched and adds an adversarial discriminator that classifies whether a sequence of
**hidden states** came from teacher-forced or free-running mode; the generator is trained to fool
it. The training signal is *"make your free-running state distribution look like your teacher-forced
one"* — which carries **no incentive to invent occurrence**. That matters because M64's failure was
precisely such an incentive: deleting events from the input while the target kept them made
inventing occurrence the optimal response.

It is also the **training-time version of this programme's only positive intervention**: freezing
the cell state during free-running (M48/M56, +0.039 AP). Freezing stops the state drifting by force;
PF trains it not to drift.

## Why this and not the alternatives

| candidate | disposition |
|---|---|
| **Professor Forcing** (#309) | **DEMOTED to last, 2026-09-05, by a 7-seat method review.** The target (the state) is right; the instrument is wrong. PF's own paper reports no benefit below ~100 steps (**C-265**) and our horizon is 36, in the *inverted* regime (we train on ~383 steps and generate 36; PF's case is train-short/generate-long). |
| **Horizon Forcing** (`Zhuang2025`) | **DISQUALIFIED 2026-09-05**, not deferred — see `01_literature`. Its objective degenerates on our data. |
| **GTF** (#294) | **PROMOTED — proposed independently by three seats, never considered by this dossier.** Same target as PF, one interpolation line, and a theorem (**C-499**) bounding the Jacobian product series that killed #308. |
| **Direct multi-horizon** (#310) | **PARKED — a chair decision, not an experiment.** Removes exposure bias entirely but forfeits the recurrent state, the one mechanism this programme has established as working (M55: 2.97× fair persistence at h36, gap widening). |

## What our own measurements say to constrain

**The cell (`hl_*`), not the hidden half (`hs_*`).**

| finding | evidence |
|---|---|
| the cell **drains** during free-running | **M50**: `max|h|` 65.6 → 1.6 |
| it holds a **spatial map** | **M54**: roll it 90 cells and the forecast moves 90 cells intact, r ≈ 0.90, while skill collapses 48× |
| it **drives placement; the fed input does not** | **M60**: cell 26/26 origin-seed pairs followed, input **0/26 at every horizon** |
| holding it is the programme's one win | **M48/M56** +0.039; **M55** 2.97× fair persistence at h36 |

In `HydraBNUNet06_LSTM4.forward` the state is one concatenated tensor
`[hs_1..hs_4, hl_1..hl_4]` split `% 8`; the cell half is the **last four**.

## Document index

| # | file | status |
|---|---|---|
| 00 | `00_README.md` | **living** |
| 01 | `01_literature.md` | seeded — PF read, HF read and disqualified; gaps listed |
| 02 | `02_design.md` | **STUB — blocks everything.** Needs `expert-method-review` before pre-registration |
| 03 | `03_harness_and_invariants.md` | seeded from a real audit — the standing harness, the gaps, the pre-flight checklist |
| 04 | `04_roadmap.md` | seeded — phases and gates |
| 05 | `05_analysis_plan.md` | **NOT WRITTEN.** Pre-registration comes after `02` is reviewed |
| 06 | `06_glossary.md` | seeded |
| 07 | `07_experiment_log.md` | empty, append-only |

## Harness at a glance

**~75% already exists** and is reusable; see `03`. The honest summary:

* The **gate scripts all exist** (`floor_gate.py`, `arm_postflight.py`, `arm_identity_check.py`,
  `potency_check.py`, `screen_verdict.py`) — but **no gate in this repo is repo-wide**. Each is
  opted into by a dossier launcher, and a dossier that forgets one gets no warning. #311 was the
  first dossier since August to invoke `floor_gate` at all.
* **The seam already exists.** The pushforward branch (`training_engine.py:780-833`) already builds
  a self-fed input, forwards the model on it, suppresses BatchNorm writes, and offers a
  state-detach fork. PF is architecturally its **sibling**, not a new mechanism.
* **The stabiliser already exists** and must be wired from day one: `ss_feedback_grad_clip`
  (9/9 mutations caught, 80 lessons clean). PF backprops through a free-running unroll, which is
  exactly what diverged in #308.

**The gaps are three:** the discriminator itself, a stability gate that can call a run VOID, and a
free-running segment length that is a config field rather than a constant.

## Current state & next actions

- [x] Library pass: PF read in full; HF read in full and disqualified
- [x] Harness audit against the real repo
- [x] `02_design` written
- [x] `expert-method-review` — 7 seats, **ruled against building PF first**
- [ ] **The C-319 blindness probe** — offline, no training. Could kill the whole distribution-matching family for minutes of GPU. **Do this first.**
- [ ] **Recompute the seed σ** from artifacts already on disk; re-derive the MDE. Costs nothing.
- [ ] The emit-only state-restoration sweep — answers M50's own open question at zero training cost
- [ ] Productionise `freeze_recurrent='cell'` (never used in a delivered forecast), disclosing M58's dispersion cost
- [ ] GTF (#294) if a training-time state intervention is still wanted
- [ ] PF last, and on C-265 arguably not at all at 36 steps

## Conventions

Numbered docs, `00` living and the rest revised in place with dated headers. Git-tracked via
`git add -f` (`reports/` is gitignored). Archived to `reports/archived/` on close.

**Honest scope, pre-registered rather than discovered:** this is a **SCREEN at n=1** against ~20%
training variance (C-119/C-184). Only a large effect is visible, and **a null is INCONCLUSIVE, not
negative** (C-307).
