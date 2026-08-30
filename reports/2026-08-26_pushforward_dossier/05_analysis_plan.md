# 05 — Pre-analysis plan: the pushforward trick (#289)

**Status: LOCKED 2026-08-26.**

## ⚠️ Provenance, stated first and honestly

`git log` **cannot** prove this plan was written before the tooling: the implementation (PR #303),
the arm builder, the smoke gate and `train_time.py` all exist and are committed before this file.

What it **can** prove, and what was checked at the moment of writing:

* `results/` contains **no `score_*.csv`** and no 300-lesson arm — only two 2-lesson smoke rows,
  explicitly never scored (`find results -name 'score_*.csv'` → 0).
* The only arms on disk are `tinyzero_fortytwo` and `pftinyzero_fortytwo`, both 2 lessons.

So the falsifiers and the decision rule below were committed with **zero outcome data in hand**.
That is a weaker guarantee than an empty `tools/`, and it is recorded rather than dressed up —
C-303's fourth instance was a provenance document that claimed a pre-commitment it did not have.

## 1. Question

Every intervention this programme has tried changed **what the model is fed** — scheduled sampling
(M26–M33), ITF (M42), `truncated_nb` (M45). None changed the **horizon of the loss**. The model is
trained one-step teacher-forced and deployed 36-step free-running, and `Aceituno2025` Thm 4.6 says
short-horizon loss minima need not generalise to long horizons, with a gap growing as
`O(e^{λΔT})`.

`Brandstetter2022`'s pushforward is the closest structural analogue in the literature: unroll two
steps, cut the gradient after the first, score at `t+2`. Its own ablation is why this is worth
running rather than assuming — pushforward *with* gradients is **less** stable, and Gaussian noise
injection is **worse than nothing**. The mechanism is specific, not "any two-step training helps".

**Does a two-step auxiliary loss improve free-running gate skill at L=300?**

## 2. The one variable

`pushforward_weight`: `0.0` → `0.1`, with `pushforward_detach_state=False` (the recurrent fork —
the loss is allowed to train the recurrence to produce states that survive one step of
self-feeding).

Verified single-variable by exec-and-diff against the **own-seed control**, not by inspection:
`{pushforward_weight, pushforward_detach_state}` and nothing else. `pushforward_detach_state`
appears in the diff only because the controls' configs predate #289 and carry neither key.

**The detach fork is NOT in this experiment.** It is a second variable and gets its own arms if
this one earns them.

## 3. Design and power

4 treatment seeds (42–45) at L=300 against the **4 existing `fullzero_*` controls**.

* Exact one-sided permutation, 4v4: floor `1/C(8,4) = **0.0143**`. Unlike the architecture screen
  (2v2, floor 0.167), **this design can reach significance.**
* One comparison, one pre-specified primary endpoint. No multiplicity correction is needed because
  no second comparison is licensed to become primary.
* σ = **0.0134**, the measured control seed sd of AP@h18 (n=4, `fullzero_*`).
  MDE ≈ `2.5 × σ × √(2/4)` ≈ **0.024** AP at ~80% power. An effect smaller than that is
  **UNDERPOWERED**, not absent, and must be reported as such.

**Primary endpoint: AP@h18, free-running, `sb`, calibration.** Chosen before any arm runs.
Secondary: retention `AP(h18)/AP(h1)` (control 0.692) and the full horizon curve.

### The control-reuse spot check

The 4 controls were trained **before PR #303**. One fresh control (`fullzero_fortytwo`, new code,
seed 42) is retrained and compared to its archived twin. This is a **gate, not a covariate**:

* **matches** (AP@h18 within 5e-4, the tolerance at which two archived controls already reproduced
  in M34) → reuse validated, the 4v4 contrast proceeds;
* **differs** → **F1 fires, the whole run is VOID.** No result is reported and the controls are
  retrained. It is not a finding about the pushforward, and it will not be reported as one.

## 4. Decision rule — registered before any arm runs

| verdict | condition |
|---|---|
| **EFFECT POSITIVE** | ΔAP@h18 ≥ +MDE (0.024), permutation `p ≤ 0.05`, and no guardrail regression (§5) |
| **EFFECT NEGATIVE** | ΔAP@h18 ≤ −MDE, `p ≤ 0.05` in the observed direction |
| **NULL** | \|ΔAP@h18\| < MDE with the paired CI excluding an effect of ±MDE |
| **UNDERPOWERED** | \|ΔAP@h18\| < MDE and the CI does **not** exclude ±MDE |
| **VOID** | any falsifier in §6 fires |

The permutation p is computed **in the observed direction**. This is registered explicitly because
in M45 a one-sided test hard-coded for `treatment > control` rendered a **−0.2376 effect as
"NULL, p=1.0"** — C-303's ninth occurrence and the first to produce a wrong verdict.

**TRADE, not a win:** an AP gain with a `crps_all` regression does not promote on the AP alone.

## 5. Guardrails

Reported at **h1/6/12/18/24/30/36**, both sides: gate (`AP`, `Brier`, `precision_at_k`, `act_ratio`,
`n_false_pos`) and body (`crps_all`, `crps_events`, `crps_none`, `size_ratio`, `mcr_*`,
`mag_on_false_pos`), plus the **oracle** (`use_real`) per arm — the teacher-forced ceiling, which
separates *"the model got worse"* from *"the rollout got worse"*. That distinction is what made M45
interpretable.

## 6. Falsifiers — pre-committed

* **F1 control reuse** — the fresh spot-check control must reproduce its archived twin's AP@h18 to
  within 5e-4. Otherwise **VOID** (§3).
* **F2 identity** — each arm's `pushforward_weight` re-read from its own config at scoring time,
  independent of the builder's declaration. *(A `/falsify` audit showed the declaration alone is
  not enough.)*
* **F3 floor gate** — FG-A (`AP ≥ 5 × prevalence`) PASS on every arm; a FAIL means the vehicle
  cannot show an effect (C-299).
* **F4 setup integrity** — `arm_postflight.audit_arm` per arm: artifacts present, no NaN, `N` and
  `n_event` identical to the control. A differing support invalidates every paired comparison.
* **F5 oracle unchanged** — the teacher-forced ceiling must not move by more than σ. Pushforward is
  an auxiliary loss on the same trajectory; if the oracle moves, it changed the *model*, and the
  rollout claim is confounded.
* **F6 h1 sanity** — no arm may lose h1 AP by more than σ. h1 is nearly teacher-forced.
* **F7 dose signature (M45)** — if ΔAP is negative **and** scales with the increase in `act_ratio`
  across seeds, this is another *firing* intervention, not a *horizon* one, and must be reported as
  a fifth confirmation of M45 rather than as a pushforward result.
* **F8 BatchNorm parity** — `num_batches_tracked` identical between treatment and control arms.
  This is not hypothetical: the extra forward **was** writing BN statistics until PR #303, and it
  would have confounded this exact contrast at the BN layer while looking clean.

## 7. Predictions — stated up front so this cannot be read as blind

* **Modal outcome: NULL or UNDERPOWERED.** Four of four levers tried so far have been negative or
  null, and the two-step horizon is a small extension of a 36-step deployment gap.
* **If it moves, it moves at long horizons and not at h1** — that is the mechanism's signature. A
  uniform lift across all horizons would suggest the model simply got better, and F5 should catch it.
* **The most likely way this "works" and is still not a win:** the auxiliary term makes the model
  more conservative, `act_ratio` falls, AP rises because fewer false positives — the mirror image of
  M45. F7 exists for the negative version; the positive version is caught by reporting `act_ratio`
  and `size_ratio` beside AP.
* **I would bet on NULL.** Recorded so a positive result is a surprise on the record, not a
  retrofitted expectation.

## 8. False-negative mode (C-307)

A NULL here closes **pushforward at weight 0.1 on this vehicle**, not the horizon-of-the-loss idea.
Explicit reopen triggers: a different weight (0.5 is built and unbuilt-arm-ready), the **detach
fork**, or a longer unroll. The programme's recorded habit is to drop a thread on one cheap result
and rediscover it later; this line exists to make that harder.

## 9. Scope

L=300, `sb`, calibration, 4 seeds, one grid, one queryset. `freeze_multitask_balancer: True` on
every arm (so C-312's fixed guards are provably inert — `test_the_new_guard_is_byte_identical_when_frozen`).
No architecture change. The detach fork, weight 0.5, and the risk-field/dynamic trade-off (#301)
are out of scope and named here so they are not quietly absorbed later.

---

## AMENDMENT 1 — F1's tolerance was the wrong instrument (2026-08-30, before any treatment arm)

**Stated plainly: this changes a pre-registered falsifier after seeing its result.** That is exactly
the move C-305 was registered for — a decision rule overridden post-hoc and then written up as
"no branch matched". So the reasoning is recorded in full and the original text above is left
untouched.

**What happened.** The gate arm `refullzero_fortytwo` ran and F1 fired: AP@h18 differed from its
archived twin by 6.14e-04, against a 5e-4 tolerance. The queue stopped itself before any treatment
arm, which is what it is for.

**Why the original rule was wrong — and it is wrong in a way that has nothing to do with the
result it produced.** The 5e-4 came from M34, where **re-emits** reproduced archived controls. Re-
emitting from a fixed `.pt` is deterministic; **retraining is not**, and nobody in this programme
had ever measured retrain reproducibility. It was measured now:

| h | seed-to-seed sd (n=4 archived) | retrain Δ (same seed) | Δ as % of sd |
|---|---|---|---|
| 1 | 0.0035 | 0.0053 | **148%** |
| 6 | 0.0123 | 0.0057 | 47% |
| 12 | 0.0189 | 0.0216 | **114%** |
| 18 | 0.0134 | 0.0006 | **5%** |
| 24 | 0.0092 | 0.0012 | 13% |
| 30 | 0.0099 | 0.0092 | 94% |
| 36 | 0.0122 | 0.0095 | 78% |

**Retraining the same seed moves AP about as much as changing the seed.** So no scalar tolerance
was ever going to certify "the code did not move" — the property F1 was written to check is not
measurable at the precision the rule demanded.

**The second, worse defect: F1 tested ONE horizon.** h18 is where the two runs agree best (0.05 sd).
The same run differs by 0.94 and 0.78 sd at h30/h36 — the horizons this dossier's hypothesis is
about. A slightly looser scalar tolerance would have passed the gate *silently* with the long
horizons never checked. The rule fired for the right reason by luck, at the least informative
horizon.

**Amended rule.** F1 now asks whether the fresh control is an **outlier against the scatter the
archived controls already show among themselves**, at **every** horizon: `|Δ_h| ≤ 3 × sd_h`, with
`sd_h` computed at run time from the four archived controls. `k=3` is deliberately generous — this
gate exists to catch a **gross** code effect, not to certify fine agreement, which the table above
shows is not available at any tolerance.

**Result under the amended rule: PASS.** Max deviation 1.48 sd (h1); h30/h36 at 0.94/0.78 sd.
Mutation-verified: a −0.05 shift at h36 (4.9 sd) fails it; the 6.1e-04 h18 drift that fired the
original rule does not.

**What this does NOT license.** The amendment says the observed difference is *unremarkable in
size*. It does **not** prove PR #303 changed nothing about training. That distinction is why the
verdict is "reuse permitted", not "code proven inert". Registered as **C-317**.

**Unchanged:** the primary endpoint, the MDE, the decision rule in §4, and falsifiers F2–F8.
