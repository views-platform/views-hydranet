# Experiment log — is the deployment state outside the range the model was trained on?

Pre-registration `05_analysis_plan.md`: **LOCKED `cbb1e5b`** with `tools/` empty, then **AMENDMENT 1**
`8d1f1d9`, **2** `7f060a3`, **3** `347b118`, **4** `dca4b5d`, **5** `4494556` — every one committed
**before the step it governs ran**. `git log` proves the ordering.

---

### EXP-01 · 2026-08-23 · **IN-RANGE — the hypothesis is dead**

| seed | half | `f` | verdict | \|R2\|max | \|R1\|max |
|---|---|--:|---|--:|--:|
| 42 | **cell** | **0.0030** | IN-RANGE | 29.50 | 33.60 |
| 42 | hidden | 0.0033 | IN-RANGE | 0.991 | 0.991 |
| 43 | **cell** | **0.0032** | IN-RANGE | 66.08 | 66.28 |
| 43 | hidden | 0.0036 | IN-RANGE | 0.972 | 0.972 |

**A 98% interval puts `f = 0.02` outside by construction. The measured `f ≈ 0.003` is roughly SEVEN
TIMES BELOW chance.** The deployment state is not merely inside the training-like range — it is *more*
contained than random draws from that range would be. §4's IN-RANGE branch (`f ≤ 0.05`) fires on both
seeds and both halves; no seed split, no half split.

**H is refuted. Free-running does not begin from an out-of-distribution state.** The zero collapse is a
**dynamical** property of the rollout, not distribution shift at the origin.

## The state starts healthy and then decays — recorded as a finding, not just as F2

The refutation above is about the **starting** state. It says nothing about what happens next, and what
happens next is not subtle. Free-running trajectory, seed 43, production path
(`results/state_decay_fortythree.csv`):

| free step | 0 | 5 | 10 | 15 | 20 | 25 | 30 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `max\|state\|` | **65.62** | 56.62 | 19.74 | 8.19 | 5.13 | 3.02 | **1.83** |

**A 35.8× collapse, and most of it lands between steps 5 and 15** — 56.6 → 8.2, a 7× drop in ten steps,
after barely moving for the first five. This is measured on the production rollout with dropout on; it
appears here as F2's vehicle check, but the shape is a result in its own right and is easy to lose in a
falsifier table.

Together with the verdict: **the state is in-distribution when free-running begins, and decays toward
zero as it proceeds.** The failure is in the dynamics, not the starting point. Note the ordering — the
AP damage measured under **#290** is *front-loaded* (steps 1–6 carry 3.15–4.22× the per-step damage of
later blocks), whereas this collapse is *slowest* over the first five steps. **The two curves do not line
up**, so "the state collapses" is not yet an explanation for "skill drops early", and this log does not
claim it is.

## Falsifiers

| | result |
|---|---|
| **F1** zero init | **PASS** — enforced in `_digest`, which raises if `init_hTtime` returns non-zero |
| **F2** same vehicle | **PASS** — origin `max\|h\|` = **65.622**, collapsing to **1.832** over 30 free steps = **35.8×**, against the registered ≥10× (AMENDMENT 2). Independently, the dropout-OFF R2 measured **66.075** vs this dropout-ON **65.622** — **0.7% apart**, so AMENDMENT 2's simplification is benign for this quantity |
| **F3** geometry (HARD STOP) | **PASS** — interior `rel_abs_diff` **0.0006** (s42) / **0.0059** (s43) ≤ 0.02 |
| **F4** curriculum bites | **WEAK PASS** — 4.548 vs 3.966 densities, +15% and **non-monotone**; see below |
| **F5** finite | **PASS** — enforced in `_digest` |

## Two predictions I made and got wrong

**AMENDMENT 4 predicted F3 would fail. It passed, comfortably.** The reasoning was that a receptive
field of ~12–13 cells compounding over 384 recurrent steps leaves no boundary-free interior in a 32×32
patch. **The measurement says otherwise, and by a wide margin**: pooled `rel_abs_diff` ≈ 0.10 but
interior **0.0006** — a **142× gap** on seed 42, 13× on seed 43. **The theoretical receptive field is not
the effective one.** The recurrence is *contracting* (F2 measures a 35.8× collapse), so influence decays
faster than it propagates. AMENDMENT 3's boundary diagnosis was right; AMENDMENT 4's escalation of it was
wrong. Recorded because the prediction was registered before the run and must be scored.

**And `origin` was nearly wrong in a way that would have gone unnoticed.** `predict()` falls back to
`seq_len - 1 = 383` when no origin is passed, but production rolls over `ctx.origins` and the real
free-running phase begins at **335** (measured: sample period 371, `time_steps` 36). The first pass used
the fallback and reported seed 43's `|R2|max` as **21.59**; at the true origin it is **66.08**, against
the independently published **65.6**. **A 3× error in the headline quantity, in a run where every
falsifier still passed.** Same shape as C-308 — the number was plausible and everything downstream
agreed with it.

## What `f` cannot see — a limitation of the metric, not of the result

`f` counts values falling **outside** an interval. **A state that has collapsed toward zero is trivially
"inside" an interval that spans zero.** So `f` detects *excursions*, not *degeneracy* — and the collapse
F2 measured (65.6 → 1.8) is degeneracy. **A free-running state would very likely also read IN-RANGE
while being obviously pathological.** The IN-RANGE verdict is therefore correctly scoped to *"the origin
state is not an out-of-distribution excursion"* and says nothing about whether the collapsed state is
*representative*. Any follow-up must use a distributional distance, not a containment fraction.

## F4: the curriculum separates the diet far less than its range implies

Mean event density by threshold: **4.548** (thr=143) / **5.985** (thr=75) / **3.966** (thr=10) —
identical across both seeds, since anchor selection depends on the data and the tool's RNG, not on the
model. F4's registered comparison passes (4.548 > 3.966) **by only 15%, and the middle ratio is the
densest**. The production `sigmoid` anchor strategy plus up-to-`dim` spatial jitter evidently blunts the
threshold. **A curriculum whose ratio sweeps 0.665 → 0.05 moves the realised training diet by ~15%
non-monotonically** — carried forward independently of this experiment's verdict.

## Scope

2 seeds, one architecture, `sb`, states only. **No AP was measured and none may be inferred.** Registered
false-negative mode (§7) stands: R1 uses **final trained weights** on training-like input, so IN-RANGE
closes *"the deployed model is being run out of range"* and **not** *"training never saw a different
state regime"*. Reopen if anyone instruments a real training run's states (~5 GPU-h/seed).

## Consequence

**M38 still has no mechanism.** Freezing the cell state is worth +0.039 AP@h18 and the
out-of-range explanation is now excluded. AMENDMENT 4's successor design — comparing selected vs
unselected cells *within* the single full-grid run — is **also answered by this result**: R1 (selected
patches) and R2 (whole map) produce the same state distribution, so there is nothing there to find.
**The next hypothesis has to be about the rollout's dynamics, not its starting point.**
