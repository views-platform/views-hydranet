# Do the feedback-realism findings survive a vehicle that has skill?

**Status: COMPLETE (2026-08-17).** Six arms on `violet_visitor`, 59 minutes of GPU, all falsifiers pass.

## Result

**Yes — and they are sharper on the model that has skill.** Three of the four predictions were
falsified, and the conclusion is the opposite of the hypothesis that motivated the run.

| h18, target `sb` | `violet_visitor` (this dossier) | `truncated_smoke` (#278) |
|---|--:|--:|
| occurrence share of the gap | **95.3%** | 88.6% |
| magnitude share | **1.4%** | 7.9% |
| `spatial_scramble` | **−94%** (below the control) | +0.9% |
| `thin:0.75` | 95.5% | (32× the control's AP) |

Three things follow, each measured rather than inferred:

**1. The oracle does not degrade.** Fed the real field, gate AP holds 0.4745 → 0.4793 → 0.4577 across
36 steps. *All* of the free-running decay is attributable to what is fed back — not to the recurrence,
the horizon, or accumulation inside the architecture.

**2. Occurrence is ~95% of the gap; magnitude is ~0%.** Hand the model the correct *occurrence* and let
it keep its own magnitudes — which are **71% inflated** — and it recovers 95.3% of the gap. The mirror
recovers 1.4%, and is *negative* at four of six horizons: true magnitudes with the model's own
occurrence make it slightly **worse**.

**3. Wrong placement is worse than the model's own errors.** `spatial_scramble` (perfect marginals,
perfect magnitudes, permuted locations) scores 0.0486 at h18 against the control's 0.2569 — 5× worse
than feeding the model its own flawed output. Meanwhile `thin:0.75` throws away **three quarters of the
true events** and still recovers 95.5%. Discarding events costs almost nothing; moving them costs
everything.

## Why this dossier exists

Every arm of #277 and #278 ran on `truncated_smoke`, a **40-lesson** vehicle whose config comment says
`# SMOKE (not a scored result)`. Its h1 gate AP (0.29792) **equals climatology's** (0.29798), so it has
no occurrence skill at any horizon, and it collapses 42× by h18. The production model does not behave
that way:

| sb gate AP, free-running | h1 | h6 | h18 | h36 |
|---|--:|--:|--:|--:|
| `violet_visitor` (160 lessons, `nb`) | 0.4745 | 0.3924 | 0.2569 | 0.1370 |
| `climatology` | 0.2980 | 0.2620 | 0.2251 | 0.1667 |
| `truncated_smoke` (40 lessons, `truncated_nb`) | 0.2979 | 0.0284 | 0.0070 | 0.0083 |

`violet_visitor` is **REAL** against climatology through h18 by the audited `verdict_token` (Epic #263).
So the question was whether the mechanism findings described *the rollout* or merely *an undertrained
model failing*. They describe the rollout.

## Two corrections this run forces

**The smoke measurement of placement was floor-limited.** On `truncated_smoke`, `spatial_scramble`'s
"+0.9% of the gap" was the distance between two numbers both pinned near zero (0.0097 vs a control of
0.0070) — the control had no room to fall further. It was never a measurement of placement's
importance. On a vehicle with real skill, scrambling destroys 81% of the control's remaining AP.

**The share statistic does not apply to every arm.** `(arm − control)/(oracle − control)` assumes arms
lie *between* control and oracle. `spatial_scramble` does not, on either vehicle. Any decomposition must
state whether an arm falls outside the interval before quoting a share.

## What makes the numbers trustworthy

* **F1 — the shipped board reproduces bit-for-bit.** The 2026-08-12 production cubes survived on disk;
  re-scoring them reproduced Epic #263's `rescore.csv` with worst |ΔAP| = **0.00e+00** across all seven
  horizons. That is an independently useful result: the board is reproducible from preserved artifacts.
* **F4 — h=1 is identical across all six arms** (0.474461375, worst |ΔAP| = 0.00e+00). Step 1 has no
  feedback, so every arm must agree there; any spread would mean something other than the feedback path
  had moved.
* **F5 — `N` = 170430 in every scored row**, so no two arms were compared on different supports.
* **F6 — five separation relations pass on the real field**, proving each transform bit on *this*
  vehicle and not merely on a fixture (`af(scramble) ≡ af(use_real)` to 0.00e+00; clustering ratio
  0.025; `thin` within 1.0% of the predicted 0.25).
* **A smoke gate ran first**, proving the whole chain end to end in 12 minutes before committing hours.

## The control, and a byproduct

Three commits touching the inference path landed after the preserved cubes were written — notably
`a2eabeb` (per-site LockedDropout, independent MC-dropout masks), on a vehicle that evaluates with
dropout live. Comparing today's arms against a pre-change control would have confounded each
transform's effect with a dropout change, so **AMENDMENT 1** replaced the control with an `identity`
arm run tonight.

Measured: `identity` (today) vs the preserved cubes = **0.00e+00 at every horizon**. Those commits are a
**no-op on this vehicle's free-running path**. The amendment was still right — the equivalence could not
be known without measuring — and now it is on the record.

## Layout

| path | what |
|---|---|
| `05_analysis_plan.md` | the LOCKED pre-registration, P1–P4, F1–F6, + AMENDMENT 1 |
| `07_experiment_log.md` | falsifier verdicts recorded **before** predictions were read |
| `SCOPE.md` | eight exclusions, stated before the run |
| `tools/overnight_run.sh` | the detached, resumable driver |
| `tools/replication_smoke_entry.py` | the tested entry, origins truncated, for the smoke gate |
| `tools/overnight_verify.py` | falsifier checks + `MORNING_REPORT.md` |
| `results/overnight/MORNING_REPORT.md` | **GREEN** |

## Scope

One seed (42), one vehicle for the treatment arms, one target (`sb`), 13 origins, S=16, calibration
partition. `truncated_smoke` differs from `violet_visitor` on **three** axes at once — 40 vs 160 lessons,
`truncated_nb` vs `nb`, *and* `body_supervision` `active` vs `all` — and this cannot separate them.
(Recorded as two axes until 2026-08-17; the third was missed. See `postmortem_floor_limited_vehicle.md`.) Per the standing rule adopted 2026-08-17,
a replication is an **escalation trigger** (second seed, third vehicle), **not** a conclusion.
