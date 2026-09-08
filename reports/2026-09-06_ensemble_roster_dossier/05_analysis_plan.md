# 05 — Pre-analysis plan

**Written 2026-09-06 BEFORE the first emit.** Amendments are appended and dated, never edited in.

## Design

**8 trained artifacts × 2 compositions × clamp ON = 16 emits.** No retraining. Calibration
partition, 13 origins, all three targets (`sb`, `ns`, `os`), all 36 horizons.

A **clamp-OFF** arm is emitted for **2 models only** (one `nb`, one `mixture_nb`) as the control
that tests transfer — see Q1. Full clamp-off across all eight is not bought: 16 more emits to
re-measure an effect already established at 4/4 seeds.

## The three questions, each with its rule written first

### Q1 — Does the clamp transfer to these models?
Measured on 2 models × 2 compositions, clamp on vs off.
- **TRANSFERS** if ΔAP@h36 ≥ +0.03 on both models (half the 4-seed effect of +0.0591).
- **DOES NOT TRANSFER** if ΔAP@h36 ≤ 0 on either. **Then the clamp does not go on the roster**, and
  ADR-027 §2.1's production permission is re-opened. This is the branch that can overturn yesterday's
  decision, and it is written down before the number exists.

### Q2 — Which composition, per model, with the clamp on?
Per model, `soft_gate` vs `threshold_gate`, on the **full metric set** (below). No single-metric
verdict: `crps_all` alone is the instrument that produced the "tied" error this dossier corrects.
- A composition wins for a model if it is better on **AP at h18 and h36** *and* not worse on
  `crps_events` by more than one seed sd.
- If they split, **keep both** — a split is evidence of complementary behaviour, which is the point.

### Q3 — Which eight, pooled?
**The decisive question, and it is answered on the POOLED forecast, not on member scores.**
- Compute the pairwise **error-correlation matrix** across the 16 candidate arms, per target, at
  h1/h18/h36, on the per-cell error of the composed forecast.
- Enumerate candidate 8-subsets, pool by concat (equal weight, matching `rusty_bucket`), and score.
- **Chosen roster = the 8-subset with the best pooled AP@h18 on `sb`**, subject to: it must not be
  worse than the incumbent eight on `ns` or `os`, and must contain **≥2 distinct families**.
- Report the incumbent eight's pooled score in the same table. **If the incumbent wins, say so.**

## Evaluation — both halves, no shortcuts

The chair's requirement is a thorough read on **gate and body**. The scorer already emits 22 columns;
all are reported, grouped:

| half | metrics |
|---|---|
| **gate (occurrence)** | `AP`, `Brier`, `act_pred`, `act_true`, `act_ratio`, `precision_at_k`, `n_false_pos`, `pos_mcr` |
| **body (magnitude)** | `crps_events`, `size_ratio`, `mag_on_true_pos`, `mag_on_false_pos`, `mcr_all` |
| **joint / leak** | `crps_all`, `crps_none`, `mcr_none` |

**Added for this run**, because the panel showed the standing set is insufficient:
- **Fair CRPS** — `CRPS − ½E|X−X′|` (Ferro). The unadjusted estimator is biased **toward
  underdispersed** ensembles, and pooling changes `S` from 16 to 128, so an uncorrected comparison
  between a member and a pool is not like-for-like. **Without this the ensemble comparison is invalid.**
- **Reliability–resolution–uncertainty decomposition** of the CRPS, so a movement is attributed
  rather than asserted.
- **Error correlation** between arms — the quantity Q3 turns on, and which nothing here has measured.

## Pre-committed falsifiers

- **F1 — the clamp does not transfer** (Q1's branch). Fires ⇒ the roster ships unclamped and ADR-027
  §2.1 is re-opened.
- **F2 — the pooled ensemble is not better than its best single member** at h18 on `sb`. Fires ⇒ the
  ensemble is not earning its cost and that is the finding, whatever the roster.
- **F3 — every candidate 8-subset scores within one seed sd of every other.** Fires ⇒ roster choice
  is unmeasurable at this resolution; keep the incumbent and say the selection was not decidable.
- **F4 — a composition change moves AP by more than it moves `crps_events` in the opposite
  direction**, i.e. the gain is pure firing-rate. Read against M45: firing more is only good if
  placement improves.

## VOID conditions

- any arm's weight hash equal to another's (arms not distinct)
- a cube of the wrong shape, or `S` unequal across pooled members (the `2/(m·m)` bias does not cancel)
- the h1 identity check failing: with the clamp on, **h1 must be byte-identical to clamp-off**, since
  the clamp acts only from step 2. A difference at h1 means the arm is not what it claims.

## Scope

Calibration partition, one region, 13 origins, single artifact per model (no seed replication within
a configuration). **This ranks the configurations we have; it cannot invent a better one.** A roster
chosen here is the best of 16 candidates, not the best possible.
