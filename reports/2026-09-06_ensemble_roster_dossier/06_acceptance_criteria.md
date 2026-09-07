# 06 — Acceptance criteria: is a candidate model BROKEN?

**Written 2026-09-07, before the validation runs.** These are **kill criteria**, not a ranking. A
model that passes all three is *acceptable for the ensemble*; passing says nothing about being good.
Ranking is a separate question and is not decided here.

**Partition:** `validation` (not calibration — calibration was used to select these configurations,
so re-using it would score the choice on the data that made it).
**Vehicle:** 300 lessons, one run per model. **Comparator:** `light_strider`.

> **Comparator note.** The chair asked whether the conflictology baseline is `white_ranger`. It is
> **`light_strider`** for this purpose. Both are `ConflictologyModel` climatology; `white_ranger` is
> the **viewser** build and `light_strider` the **datafactory** build, and the roster is datafactory.
> The v2 scoreboard scored against `light_strider`. ⚠️ `reports/GLOSSARY.md:181` still calls
> `white_ranger` "the baseline… the thing to beat" — **stale since the datafactory migration** and
> to be corrected. `light_strider` must be **re-run on validation too**; a comparator scored on a
> different partition is not a comparator.

---

## K1 — STARVED at t=36

**Definition.** At h=36, on target `sb`, the model's total predicted fatalities landing on cells that
actually had events is below **1% of the true total on those cells**.

```
recall_mass(h=36, sb) = Σ_{cells with truth>0} predicted_mean  /  Σ_{cells with truth>0} truth
K1 FIRES if recall_mass < 0.01
```

**Why this quantity and not `size_ratio`.** `size_ratio` reads **0.0000 for every model at every
horizon** — it is saturated and cannot discriminate. `recall_mass` is the same question asked in a
way that still has resolution: *of the deaths that occurred, what share did the model actually put on
the right cells?*

**Why 1%.** Measured on calibration, the 8 current models span **0.008 → 0.067** at h36. A 1%
threshold fails the worst (violet_visitor at 0.008) and passes the rest. It is deliberately a
**floor, not a bar** — this is a broken-detector, and "predicts essentially nothing" is the failure.

**Also reported, not a criterion:** `waste_frac` = share of predicted mass landing on zero-truth
cells. Currently **54–78%**. Too high to be healthy and too uniform to discriminate, so it informs
rather than kills.

---

## K2 — BLOOM

**Definition.** The free-running rollout explodes rather than decays.

```
K2 FIRES if, on ANY target:
    crps_none(h=36) > 10 × crps_none(h=1)      # mass leaking onto true zeros
 OR total_predicted(h=36) > 5 × total_predicted(h=1)
```

**Why these two.** The established bloom signature is **field-wide `crps_none` + mean magnitude**,
not `crps_all` and not a max — the max false-flagged "blooms everywhere" when this was last studied.
Sample-feedback (ADR-070) is the standing mitigation and every roster member has
`rollout_feedback: 'sample'`, so a bloom here means the mitigation failed for that configuration.

**Calibration reference:** current models run `crps_none` h1→h36 of roughly 0.0013→0.0008 (falling),
so a 10× rise is far outside the healthy range and will not fire on noise.

---

## K3 — WORSE THAN CLIMATOLOGY

**Definition, as specified by the chair.** For each target and each forecast month:

```
month m is a LOSS on target t  if  crps_events(model, t, m) > crps_events(light_strider, t, m)
                                OR mcr_events(model, t, m) > mcr_events(light_strider, t, m)
K3 FIRES if, for ANY target, losses > half the months
```

Both metrics are **event-conditional by construction** — that is the chair's intent: the comparison
is about the cells where something happened, not about correctly predicting the 99.94% of quiet
cells, where climatology is nearly unbeatable and the comparison is uninformative.

⚠️ **Recorded limitation, not a reason to change the rule.** Conditioning the evaluation set on the
realised outcome makes a score **improper** as a general comparator: the score-optimal report becomes
`P(Y | Y>0)` rather than `P(Y)`, which rewards inflation. That is a real hazard *for ranking*. It is
acceptable **here** because K3 is a **kill criterion against a fixed external baseline**, not a
ranking between our own arms — nothing in this design lets a model win by inflating, since inflation
does not move it past a threshold. Do not reuse K3 to rank the roster.

---

## Verdicts

| outcome | meaning |
|---|---|
| **PASS** | none of K1/K2/K3 fired — acceptable for the ensemble |
| **BROKEN** | one or more fired — named, with the number |
| **VOID** | the run did not complete, or a gate below failed — **not** a pass and **not** a fail |

## VOID conditions — checked before any criterion is read

- the run did not reach 300 lessons, or produced no artifact
- weight hashes not distinct across arms (arms not actually different)
- `light_strider` not re-run on **validation** (K3 has no comparator)
- any config that fails to construct (the `bold_comet` / `heavy_freighter` failure of 2026-09-07)
- NaN/Inf in the artifact

## What these criteria deliberately do NOT do

They do not rank, do not choose the roster, and do not test the ensemble. They answer one question:
**is this candidate fit to be pooled at all.** A model can pass all three and still be the worst of
the eight.
