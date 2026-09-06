# Professor Forcing — constrain the free-running STATE, not the fed-back field

**Issue:** [#309](https://github.com/views-platform/views-hydranet/issues/309) · **Opened:** 2026-09-05 · **Status:** SCAFFOLDED — harness audited, nothing implemented, nothing run

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
| **Professor Forcing** (#309) | **PURSUED — this dossier.** Targets the state, which every measurement implicates. |
| **Horizon Forcing** (`Zhuang2025`) | **DISQUALIFIED 2026-09-05**, not deferred — see `01_literature`. Its objective degenerates on our data. |
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
- [ ] **Write `02_design`** — the open design forks are listed there; this blocks everything
- [ ] `expert-method-review` on `02_design` — **the panel should be seated before pre-registration**
- [ ] `05_analysis_plan` pre-registration, including the **stability gate** and a VOID branch
- [ ] Implement behind a default-off flag; byte-identical when off
- [ ] Adversarial audit in a clean context (a non-author) + mutation testing to exhaustion
- [ ] Smoke + potency on the arm's own config **and at a trained checkpoint** (C-324/C-325)
- [ ] Screen: control vs PF, 300 lessons, n=1

## Conventions

Numbered docs, `00` living and the rest revised in place with dated headers. Git-tracked via
`git add -f` (`reports/` is gitignored). Archived to `reports/archived/` on close.

**Honest scope, pre-registered rather than discovered:** this is a **SCREEN at n=1** against ~20%
training variance (C-119/C-184). Only a large effect is visible, and **a null is INCONCLUSIVE, not
negative** (C-307).
