# Pushforward (#289) — dossier

**Status: CLOSED. M47 in the ledger; #289 closed. See `07_experiment_log.md` for the science and
`08_postmortem.md` for how the programme was run — the latter written because it took four days
and 22 hours to reach a result that took thirteen hours once an endpoint existed.**

## The question

Brandstetter et al. 2022's pushforward trick: unroll two steps, cut the gradient after the first,
score the loss at `t+2`. It is the closest structural analogue in the literature to our rollout
problem — the model is trained teacher-forced and deployed free-running, and every intervention we
have tried so far changed what the model is *fed* (SS, ITF, truncated_nb) rather than the
*horizon of the loss*.

The paper's own ablation is why this is worth doing rather than assuming: pushforward *with*
gradients is less stable than without, and Gaussian noise injection is worse than nothing. So the
mechanism is specific, not "any two-step training helps".

## What exists

| | |
|---|---|
| Implementation | `views_hydranet/train/training_engine.py`, `pushforward_weight` (default `0.0`) and `pushforward_detach_state`. Merged in PR #303. |
| Mechanism tests | `tests/train/test_pushforward.py` — 13 tests, each one because a plausible implementation could pass without doing the thing. |
| Arm builder | `tools/make_pf_arm.py` |
| Smoke gate | `tools/smoke_pf.sh` → `results/SMOKE_PF_OK` |

## Why the audit is part of this dossier

The smoke was gated on a training-loop audit (PR #303) rather than run directly, because the
standing instruction was that cheap tests here keep failing for reasons unrelated to the
hypothesis. That was the right call: **the audit found a critical defect in the pushforward
itself.** The extra forward ran in `train()` mode and wrote BatchNorm running statistics —
`num_batches_tracked` 5 → 9 on a T=6 window. Those buffers go into the artifact and are recomputed
by the C-184 recalibration with `momentum=None` (a cumulative average, so the pushforward forwards
would have carried *equal* weight). An arm with `pushforward_weight > 0` would have differed from
its control **at the BatchNorm layer**, for reasons having nothing to do with the auxiliary loss.
The A/B would have been confounded and the run would have looked clean.

Thirteen mechanism tests did not catch it. None of them asked whether the extra forward had side
effects on model *state*.

The audit also corrected a prior that would have shaped this programme: **gradient does reach far
back** through the untruncated BPTT graph (1.6e-02 at 118 steps in the trained model, versus
2.8e-17 at random init), so the recurrence is trained and gradient-carrying, and M46's `WideMemory`
null is not a vanishing-gradient story.

## Guard rails, and what each one is for

Every one exists because something specific went wrong before:

* **`arm_label` is table-driven**, not a slugifier — `validate_model_name` is `^[a-z]+_[a-z]+$` and
  a three-part name is rejected *after* the queue has accepted the arm. That killed the ITF pilot's
  first launch. All six labels were checked against the real validator before anything was built.
* **`_verify` execs both configs** and requires the symmetric difference to be exactly the intended
  key set — against the FLOOR *and* against the CONTROL. It fired on the first build attempt.
* **The treatment must change the loss.** `make_pf_arm` runs the real `_process_sequence` at the
  arm's weight and refuses to build if the loss is unchanged. Verified to bite: forcing weight to
  0.0 makes it refuse by name.
* **The control must NOT change the loss.** The `0.0` arm proves default-off is byte-identical to
  having no flag at all — otherwise the control is itself a weak treatment and every contrast is
  confounded.
* **The smoke runs the control too**, because the number that matters is the *ratio*. A cost
  multiplier measured against a different day's timing is not a measurement.

## Next

1. `tools/smoke_pf.sh` — the gate. Measures the real cost multiplier on the full pipeline.
2. Pre-registration (`05_analysis_plan.md`) once the smoke gives a GPU budget. **No scored arm
   before that file is committed** — C-303's fourth instance was a provenance document that
   claimed pre-commitment it did not have.
3. Then: which architectures are pushforward candidates, and the risk-field vs dynamic-forecast
   trade-off (#301).


---

## Outcome

**The pushforward does not help. It makes the 36-month forecasts worse.**

Four seeds at L=300, weight 0.01: the ability to pick conflict cells 18 months out fell ~7%
(0.3257 → 0.3037), all four seeds worse, p=0.0429. Formally UNDERPOWERED — 0.0220 against a
pre-registered MDE of 0.024 — so the direction is established and the magnitude is not.

**The mechanism is the finding: the model itself is undamaged.** Teacher-forced skill is unchanged.
It is only worse when running on its own predictions — the exact situation the method exists to fix.
A stronger weight (0.1) damaged the model outright and was VOID.

**Not closed:** the horizon-of-the-loss idea. Reopen triggers registered in `05` §8 — the detach
fork and a longer unroll.
