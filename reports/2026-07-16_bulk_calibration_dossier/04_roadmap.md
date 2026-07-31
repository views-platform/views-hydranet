# 04 — Roadmap (phased, gated)

**One variable at a time. Gate frozen, tail parked throughout. Each phase gates the next.**

## P0 — Metric harness + baseline anchor (retrain-free) — BLOCKER
Extend `t0_score.py` with the LOCKED bulk-calibration metric (`03 §D`): T=0 · positives · bulk (truth ≤
98th-pct, +97/99) · `ratio_med` + within-2× + ratio spread + guardrails (Brier/CRPS/QS99). Anchor it on the
**existing dense-mse pw2 predictions** → record the baseline `ratio_med` (expect ~0.05–0.11). *No GPU.*
**Gate:** metric implemented + unit-checked + baseline numbers recorded, before any new training.

## P1 — Implement the mechanism (TDD, default-off) — BLOCKER
Winsorize target helper (per-cell robust cap) + τ-pinball log-space loss in `LOSS_REG_REGISTRY`; unit tests
(minimiser = τ-quantile, finite grad, NaN/Inf guard, baseline byte-identical when off). *No GPU.*
**Gate:** full suite + lint green; pre-flight checklist (`03 §E`) green.

## P2 — Pre-register (`05`) + 2-lesson smoke — GATE
`preregister` the first experiment (metric + falsifiers + thresholds LOCKED). Then a 2-lesson smoke:
trains, finite, winsorize+dial active, baseline byte-identical with flag off. *GPU (short).*
**Gate:** smoke clean → proceed; else fix.

## P3 — Same-seed A/B (the kill gate) — GATE
dense-mse pw2 (baseline) vs +winsorize+dial, seed 42, 40L, T=0 bulk-calibration metric. Sweep τ if needed
(τ is the dial). **Falsifiers per `05`.**
**Gate:** `ratio_med`→[0.7,1.3] with CI>0 AND guardrails held → P4; timid/explode/guardrail-break → falsified
(postmortem) → fallback lifter (count_mean-on-capped) or re-scope.

## P4 — Robustness (≥3 seeds) — GATE
Seeds {42, +2}. Run-to-run variance is a disqualifier. **Gate:** reproducible → P5.

## P5 — Validation graduation — VERDICT
Same metric on the **validation partition** (the count_mean-OOS-collapse guard). **Gate:** holds OOS →
`promote` to a proposed ADR; collapses → banked negative (the winsorize didn't prevent the regime-overfit).

## Parked (explicitly out of scope now)
Tail modeling (top 1–3%); gate changes; new covariates; the distributional/quantile head; rollout (>T=0).
