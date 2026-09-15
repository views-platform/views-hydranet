# 04 — Config changes and run plan

**Written 2026-09-07.** Target: nine validation runs (8 candidates + the comparator), 300 lessons,
one run each, judged by `06_acceptance_criteria.md`. Deployment-driven: the question is **"is each
model fit to be pooled"**, not "which is best".

## Part 1 — Defects to fix in the CURRENT configs, before any spec change

These are wrong regardless of which eight we end up with.

| # | defect | affected | fix |
|---|---|---|---|
| **D1** | **`freeze_multitask_balancer` unset** → defaults **False**. The MultiTaskLoss balancer's `log(std)` turns negative once a task fits, and both ADR-014 guards key on that sign, so **updates stop and nothing is logged** (C-312). Only `violet_visitor` pins it — a plausible part of why it behaves unlike its siblings. | **7 of 8** | pin `True` on all eight |
| **D2** | scheduled sampling active with `ss_feedback` unset → defaults `'mean'`, contradicts `rollout_feedback: 'sample'`, config **cannot construct** | bold_comet, heavy_freighter | **fixed 2026-09-07**; guarded by `test_roster_configs_load.py` |
| **D3** | `freeze_recurrent` absent everywhere → the cell clamp is off in production despite being measured at **+0.094 / +0.260 AP@h18** | **8 of 8** | pin `'cell'` |
| **D4** | `ss_epsilon_max` differs (0.0 on violet_visitor, 0.5 on six, unset on purple_alien) with **nothing declaring the difference** | 8 | make it explicit per model below |

**D1 is the one to notice.** If the balancer has been frozen on one model and free on seven, then
*every* cross-model comparison in this dossier — including all of yesterday's ranking — is
confounded by it. It does not invalidate the kill criteria (those are absolute, not comparative),
but it does mean the ranking should not be trusted until all eight are on the same footing.

## Part 2 — The eight specifications

Common to all eight, and **changed from today**: `freeze_recurrent: 'cell'`,
`freeze_multitask_balancer: True`, `ss_feedback: 'sample'` where scheduled sampling is on.
Unchanged: `HydraBNUNet06_LSTM4`, `rollout_feedback: 'sample'`, `bn_recalibrate: True`,
4×4 draws, all three targets.

| slot | model dir | family | composition | ss ε | seed | rationale |
|---|---|---|---|---|---|---|
| 1 | `purple_alien` | mixture_nb | soft_gate | **0.0** | 44 | measured best on **both** axes — top AP *and* 2nd-highest mass on real events |
| 2 | `pink_pirate` | mixture_nb | soft_gate | **0.0** | 42 | slot 1's configuration, different seed |
| 3 | `blue_stranger` | mixture_nb | soft_gate | **0.0** | 43 | slot 1's configuration, different seed |
| 4 | `bold_comet` | mixture_nb | **threshold_gate** τ=0.5 | **0.0** | 45 | hard gate ranks better and predicts less — a *measured* trade, so carry both |
| 5 | `blazing_meteor` | mixture_nb | **threshold_gate** τ=0.5 | **0.0** | 46 | as slot 4, different seed |
| 6 | `heavy_freighter` | **nb** | soft_gate | **0.5** | 47 | the only shape predicting mass at a realistic scale (`recall_mass` 0.35 at h18 vs 0.05) — **kept deliberately, ss ON, because that is the configuration that produced it** |
| 7 | `bright_starship` | **nb** | soft_gate | 0.0 | 43 | nb without scheduled sampling — the second-best ranker, and the control for slot 6's ss |
| 8 | `violet_visitor` | **nb** | **threshold_gate** τ=0.5 | 0.0 | 42 | nb + hard gate; the extreme-precision end of the spread |

**Family split 5 mixture / 3 nb** — inverted from today's 3/5, because mixture holds both the best
joint performer (`purple_alien`) and the strongest of the three seeds; nb is kept for the mass shape
and for diversity.

**Slots 6 and 7 are a deliberate one-variable pair**: same family, same composition, ss on vs off.
Whatever else this run answers, it answers *that* cleanly — the one confound this dossier could not
break with existing artifacts.

## Part 3 — Honest labelling

**Evidence-backed:** the clamp (D3, large and measured); mixture+soft+ss-off as the best joint
configuration (measured on 18 target×horizon cells); carrying both compositions (each measurably
better at a different thing); keeping an ss-on nb arm for mass.

**Assumed:** the 5/3 split; the specific seeds; that eight distinct trainings beat fewer with more
compositions (measured, but the measurement was selection-biased and I do not trust it).

**Confounded and acknowledged:** every current "model" differs in family, gate, ss *and* seed at
once, and now also in D1. Slots 6/7 are the only clean contrast in the design.

## Part 4 — Run plan

**Order matters** — the comparator first, so K3 has something to compare against, and a smoke before
committing nine × 300 lessons.

1. **Apply D1–D4** to the eight configs; `test_roster_configs_load.py` green; commit **nothing** in
   views-models (the chair takes that repo to main himself).
2. **Smoke: 2 lessons on one model.** Confirms the changed keys load, the clamp is live, and the
   validation partition resolves. ~10 min.
3. **`light_strider` on validation** — the K3 comparator. Without it K3 is VOID.
4. **The eight, sequentially**, 300 lessons, validation. One at a time: a second concurrent job
   would contend for the GPU and change the timings we would then be comparing.
5. **Score + apply `06`.** Report per model: PASS / BROKEN (naming which criterion and the number)
   / VOID.

**Cost:** ~3.2 h per training × 9 ≈ **29 h**, plus emit/score. Sequential, unattended, `setsid`-detached.

**Guards wired into the launcher** — none of these is repo-wide, each must be opted in:
weight-hash distinctness; `arm_postflight`; refuse-on-leftover-prediction-dir (this cost an hour
yesterday); the recursive kill-tree fix; and a config-construct check per model before its run
starts, so a broken config fails in seconds rather than after 3 hours.
