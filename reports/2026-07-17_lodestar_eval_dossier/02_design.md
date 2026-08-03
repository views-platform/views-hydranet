# 02 — Design: the foundation grid

Terminology LOCKED to `reports/GLOSSARY.md`. This is a **measurement build** — the deliverable is one
trustworthy table that answers three questions, not a new model.

## The model under test
**forecast = gate × body** (a gated forecast). Held fixed across the grid:
- **body: all-cell** (trained on every cell), loss **MSE**, softplus head.
- **BatchNorm fix on** (default).
- Config: `output_distribution='hurdle_shrinkage'` (gated compose) + no `hurdle_threshold` (all-cell body,
  verified: `training_engine.py:371-372` applies the MSE loss on all cells when `hurdle_threshold` is unset)
  + `loss_reg='mse'` + `reg_activation='softplus'`.
- Reminder (glossary): the **body = bulk + tail**. This grid measures the whole body; bulk/tail split is a
  later refinement.

## The grid
- **pos_weight (gate eagerness): 2, 3, 4, 5** — the user suspects 2 is too low; this walks it up.
- **seeds: 42, 43, 44** — to see the body's run-to-run wobble.
- = **12 runs**, ~7 h. train + eval → the gated forecast evaluates cleanly (the gate suppresses the bloom).

## The three questions → how the ruler answers each
1. **Is the gate calibrated, at which pos_weight?** → **AP** (higher better) and **Brier** (lower better) at
   each pos_weight, averaged over seeds. Directly shows whether pos_weight 2 is too low and where it peaks.
2. **How much does the body wobble across seeds?** → the body scores (**crps-all / crps-events / crps-none**,
   **size-ratio**, **pos-mcr**) reported **per seed**; the min–max spread is the answer. Also tests whether
   the BatchNorm fix actually settled the wobble.
3. **Gated forecast vs baseline?** → all 12 grid cells sit next to **white_ranger** in one table, on
   identical months (457–469) and identical cells.

## Fairness (unchanged — the ruler enforces it)
Same months (all re-run under partition 457–504 → T=0 457–469) · same cells (intersection, reported N) ·
one truth parquet (grid-name-agnostic) · one frozen script.

## Out of scope
Validation partition (reserved for final graduation) · the bloom / 36-month rollout (T=0 only) · any model
*improvement* — the ideas come after this foundation, judged on this same ruler.
