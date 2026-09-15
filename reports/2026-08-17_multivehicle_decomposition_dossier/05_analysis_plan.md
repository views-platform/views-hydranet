# Pre-Analysis Plan — does the occurrence/magnitude decomposition hold across vehicles?

**Date:** 2026-08-17 · pre-registered **before** any arm runs
**Companion to:** `reports/2026-08-17_vehicle_replication_dossier/` (the n=1 result being escalated),
`reports/postmortem_floor_limited_vehicle.md`, `scripts/floor_gate.py`, risk register C-299
**Status:** LOCKED.

## 1. Why

The strongest finding in the programme — **~95% of the oracle→free-running gap is occurrence
*placement*, ~0% is magnitude** — rests on **one vehicle**. Everything downstream leans on it.

Four of my mechanism claims have died in three days, and each died the same way: it held on one vehicle
and dissolved on the next. This is the finding most exposed to that failure mode and the one we would
most regret building on. The standing rule adopted 2026-08-17 says a replication is an **escalation
trigger**, not a conclusion; this is that escalation.

## 2. Vehicles — chosen by the gate, not by hand

`scripts/floor_gate.py` FG-A applied to all six roster models on data already on disk:

| vehicle | AP@h18 ÷ prevalence | eligible |
|---|--:|---|
| `violet_visitor` | 28.30× | ✅ already run |
| `purple_alien` | 21.62× | ✅ |
| `blue_stranger` | 13.85× | ✅ |
| `blazing_meteor` | 9.62× | ✅ |
| `bright_starship` | 2.21× | ❌ floor-limited |
| `pink_pirate` | 0.89× | ❌ **below random ranking** |

The two exclusions are the point: on a floor-limited vehicle a degradation arm cannot fall, so
`spatial_scramble` would read as harmless exactly as it did on `truncated_smoke` (+0.9% vs the true
−93.7%). Excluding them **before** running is the judgement nobody made on 2026-08-14.

## 3. Arms

Per vehicle, inference-only on the existing artifact (no training), ~10 min each:

| arm | what it measures |
|---|---|
| `use_real` | the oracle ceiling — denominator of every share |
| `spatial_scramble` | placement: perfect marginals and magnitudes, permuted locations |
| `occurrence_real_magnitude_model` | E4a — the occurrence share |
| `occurrence_model_magnitude_real` | E4b — the magnitude share |
| `thin:0.75` | whether sparsity alone is survivable |

**Control:** each vehicle's preserved production cubes, already scored. Not re-run — on
`violet_visitor` an `identity` arm was measured bit-identical to its preserved cubes (0.00e+00 at every
horizon), so the preserved score is a valid control.

## 4. Predictions

| # | Prediction | Threshold |
|---|---|---|
| **P1** | The occurrence share (E4a) exceeds the magnitude share (E4b) on **every** eligible vehicle | ordering, not magnitude |
| **P2** | The occurrence share is ≥ 70% at h18 on every eligible vehicle | violet's was 95.3%, smoke's 88.6% |
| **P3** | `spatial_scramble` falls **below the control** on every eligible vehicle | i.e. its "share" is negative |
| **P4** | `thin:0.75` recovers ≥ 60% of the gap on every eligible vehicle | violet's was 95.5% |

**P1 is the claim that matters.** P2–P4 are magnitudes and I expect them to vary; the programme rests
on the *ordering*, which is what a decomposition licenses.

## 5. Falsifiers — checked and recorded before predictions

- **F1** h=1 AP not identical across arms within a vehicle to 1e-6 ⇒ something other than the feedback
  path moved ⇒ that vehicle VOID (step 1 has no feedback).
- **F2** `N` ≠ 170430 or origins ≠ 13 in any row ⇒ arms on different supports ⇒ VOID.
- **F3** an arm's fed-field statistics do not move its own axis on that vehicle's real field
  (`af(scramble) ≡ af(use_real)`; `af(thin) ≈ 0.25 × af(use_real)`; clustering destroyed) ⇒ that arm is
  a silent no-op, its score is **void, not evidence that the axis does not matter**.
- **F4** oracle − control gap at h18 < 0.05 AP on a vehicle ⇒ nothing to decompose there ⇒ report
  undecidable and **do not quote shares** for it.

## 6. Decision rule — pre-committed

* **P1 holds on all three new vehicles** ⇒ the ordering is a family-level property, not violet's. The
  claim graduates from "one vehicle, INDICATIVE" to "four vehicles" and the ledger's I-A is re-scoped
  upward.
* **P1 fails on any eligible vehicle** ⇒ the decomposition is vehicle-specific. Every inference row
  resting on it (I-A, I-B, I-E) is downgraded, and the placement-based reasoning that has driven the
  last three days is retired.
* **A vehicle voids on F1–F4** ⇒ excluded and reported, not quietly dropped.

## 7. Stated confounds

1. **These are different vehicles, not different seeds.** `purple_alien` and `blue_stranger` are
   `mixture_nb`; `blazing_meteor` is `threshold_gate` rather than `soft_gate`. So a difference could be
   family or composition rather than "vehicle". This buys **generalisation across configurations**,
   which is what the claim needs, but it is not a seed study.
2. **`blue_stranger` and `blazing_meteor` load only with an uncommitted config fix** (views-models#404
   adds `ss_feedback: 'sample'`). That key governs *training*; these are inference-only runs on frozen
   artifacts, so it cannot affect the numbers — but the runs depend on a working-tree edit that is
   still awaiting review, and that is recorded here rather than discovered later.
3. **`spatial_scramble` inherits C-291's confound** — destroying clustering also breaks alignment with
   the statics. Unchanged from #278.
4. **One seed per vehicle**, one target (`sb`), 13 origins, S=16.
