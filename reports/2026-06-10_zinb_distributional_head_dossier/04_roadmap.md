# 04 — Roadmap

**Date:** 2026-06-10 · **Master tracker (single source of truth):** checklist #104.

## Linear roadmap (frozen — one box at a time, findings don't re-order it)
| # | Issue | Step | DoD (one line) |
|---|-------|------|----------------|
| 1 | #98 | raw-count target provider (contained, TDD) | provider tested (round-trip, NaN/Inf, targets-only); suite+ruff green |
| 2 | #99 | `ZINBLoss` (NB NLL on raw counts) | known-value NLL + finite grads; registry-selectable |
| 3 | #100 | head: softplus μ + π-reuse + θ scalar | flag-off byte-identical (parity); flag-on wires μ/π/θ |
| 4 | #101 | inference: emit `E[y]=(1−π)·μ` + `log1p` feedback | emits/feeds under flag; parity holds off |
| 5 | #102 | train + explosion-check gate + eval | RESULTS_LOG row (Bounded? + FAO metrics if bounded); multi-seed if promising |
| 6 | #103 | decide: ship or escalate | a decision recorded with RESULTS_LOG evidence |

**Prerequisite:** #95 (green suite) before the build is trusted.

## Decision points
- After #102: **explosion-check** is the go/no-go. Bounded → eval → #103. Explodes → escalate per `02 §7`.

## Two exits (only)
- **Ship** the ZINB head → proposed ADR.
- **Fallback** → revert to commit `e029e63` (months-old stable) and ship that.
