# 05 — Analysis Plan (pre-registration)

**Date:** 2026-06-10 · Pre-registered **before** the first ZINB train (#102). Append-only.

## One variable
`output_distribution="hurdle_nb"` (the ZINB head) **vs** the current baseline (the recorded
`RESULTS_LOG` row 1 + a clean same-config comparator). Nothing else changes in the same run.

## Hypothesis
The ZINB head (a) **un-collapses magnitude** on positive cells (the softplus μ learns real counts where
conflict occurred) **and** (b) **stays bounded under the 36-step rollout** (the explosion-check passes) —
because softplus `E[y]` feedback is sub-exponential.

## Falsifiers (pre-committed)
- **F-explode:** `diagnose_io_gain` on `E[y]` over 36 steps goes out-of-range ⇒ the link does **not** tame the runaway ⇒ STOP, escalate to rollout training / direct multi-horizon (`02 §7`).
- **F-tail:** QS99 / high-quantile coverage shows systematic tail under-prediction ⇒ NB tail too light ⇒ escalate to Tweedie / GPD tail.
- **F-volatility:** run-to-run variance across **≥2 seeds** is large ⇒ volatility disqualifies (the shrinkage lesson), regardless of mean performance.
- **F-zero-rate:** apparent gains are a ~95%-zero artifact ⇒ measure on the **positive-cell subset**.

## Metrics (FAO PRN-05 — logged to `../RESULTS_LOG.md`)
- **CRPS** = primary ranking (superiority = ≥5% better than baseline).
- **QS99 / Brier / MCR** = guardrails (non-inferiority; MCR closest-to-1, **diagnostic only**, never the target).
- Reported per target (sb/ns/os), full grid **and** positive-cell subset. **Bounded?** = the hard pre-gate.
- **Multi-seed** (≥2; ≥3 if promising).

## Decision bar
Bounded (F-explode not fired) **and** magnitude un-collapsed **and** CRPS non-inferior, multi-seed stable
→ proceed toward proposed ADR (#103). Any falsifier fires → log, escalate per `02 §7` — do **not** re-tune
in place.
