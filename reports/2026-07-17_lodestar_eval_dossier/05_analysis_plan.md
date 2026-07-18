# 05 — Pre-Analysis Plan (LOCKED before running) · 2026-07-17 (v2, all-cell grid)

Committed BEFORE the smoke or the grid. Terminology LOCKED to `reports/GLOSSARY.md`. Supersedes the v1
positives-only plan (that grid was killed at the user's instruction; see the changelog at the bottom).

## Questions (foundation — not a hypothesis hunt)
Q1 gate calibration vs pos_weight · Q2 body wobble across seeds · Q3 gated forecast vs baseline.

## The model + grid (the one thing built)
gated forecast, **all-cell body, MSE, softplus, BatchNorm fix on**; swept **pos_weight {2,3,4,5} × seed
{42,43,44} = 12 runs**. Scored T=0 on the frozen ruler vs **white_ranger**, identical months/cells.

## Pre-registered predictions (so the table can falsify our beliefs)
- **P1:** the gate's AP is roughly flat across pos_weight (AP is rank-based) while Brier changes — i.e. the
  pos_weight knob moves calibration (Brier), not ranking (AP). If 2 is "too low", a higher pos_weight
  improves Brier without hurting AP.
- **P2:** the body wobble across seeds is **small** now (BatchNorm fix on) — crps-events within ±20% across
  seeds. (If it's large, the fix didn't settle it.)
- **P3 (sobering, honest):** on sb, the **baseline beats or ties** the gated forecast on crps-all and
  crps-events (the aligned A0p read already showed this; all-cell + MSE may or may not change it).
- **P4:** the body stays **timid** (size-ratio well below 1) even all-cell + MSE — the drag persists.

## Falsifiers (pre-committed)
- **F1 (ruler invalid):** self-test fails / empty common support / non-finite metric ⇒ STOP, no table.
- **F2 (bloom):** the all-cell gated forecast blooms in eval (infinity crash) ⇒ the smoke catches it;
  STOP and score T=0 directly instead (NOT the feedback-clamp).
- **F3 (months misaligned):** any grid cell's T=0 months ≠ 457–469 ⇒ STOP.
- **F4 (coverage):** a model missing >20% of common support ⇒ flag; row provisional.

## Method (locked)
1. 2-lesson smoke of the all-cell gated forecast + MSE → trains, finite, evaluates without the bloom.
2. If clean: run the 12-cell grid (train+eval).
3. Score all 12 + white_ranger on the frozen ruler.

## Decision rules
- **Report the full grid** — all 12 cells, every metric, per seed. Nothing hidden.
- **"Best pos_weight"** = the one with the best Brier at flat AP, averaged over seeds (pre-committed;
  no cherry-picking).
- **The table is the foundation/lodestar.** Every future idea is judged on this same frozen ruler, on these
  same cells, against these rows. A change "wins" only by moving a ruler number here.
- **No tags / no conclusions** until the table exists and the smoke is clean.

## Changelog
- **2026-07-17 v1 → v2:** the v1 grid (positives-only body, MAE, pos_weight {1,2,4}) was **killed** at the
  user's instruction after the positives-only body's raw output looked broken. The user reset the plan to a
  **gated forecast with an all-cell body, MSE**, pos_weight {2,3,4,5} × 3 seeds. Also: vocabulary was locked
  (`reports/GLOSSARY.md`) — "gate" (not switch), one "gated forecast" (no "dense forecast"), body = bulk +
  tail. This v2 plan reflects all of that.
