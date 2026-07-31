# 05 — Analysis Plan (pre-registration)

**Date:** 2026-06-10 · Pre-registered **before** the first hurdle-NB train (#102). Append-only.

## One variable
`output_distribution="hurdle_nb"` (the hurdle-NB head) **vs** the current baseline (the recorded
`RESULTS_LOG` row 1 + a clean same-config comparator). Nothing else changes in the same run.

## Hypothesis
The hurdle-NB head (a) **un-collapses magnitude** on positive cells (the softplus μ learns real counts where
conflict occurred) **and** (b) **stays bounded under the 36-step rollout** (the explosion-check passes) —
because softplus `E[y]` feedback is sub-exponential.

## Falsifiers (pre-committed)
- **F-explode:** `diagnose_io_gain` on `E[y]` over 36 steps goes out-of-range ⇒ the link does **not** tame the runaway ⇒ STOP, escalate to rollout training / direct multi-horizon (`02 §7`).
- **F-tail:** QS99 / high-quantile coverage shows systematic tail under-prediction ⇒ NB tail too light ⇒ escalate to Tweedie / GPD tail. **Pre-registered (C-149): QS99 is the *most likely* binding guardrail** given the geometric-ish NB tail vs the heavy conflict tail — a QS99 miss triggers the Tweedie/GPD escalation, **NOT** a whole-model rejection.
- **F-volatility:** run-to-run variance across **≥2 seeds** is large ⇒ volatility disqualifies (the shrinkage lesson), regardless of mean performance.
- **F-zero-rate:** apparent gains are a ~95%-zero artifact ⇒ measure on the **positive-cell subset**.

## Metrics (FAO PRN-05 — logged to `../RESULTS_LOG.md`)
- **CRPS** = primary ranking (superiority = ≥5% better than baseline).
- **QS99 / Brier / MCR** = guardrails (non-inferiority; MCR closest-to-1, **diagnostic only**, never the target).
- Reported per target (sb/ns/os), full grid **and** positive-cell subset. **Bounded?** = the hard pre-gate.
- **Multi-seed** (≥2; ≥3 if promising).
- **Calibration (C-150): PIT histogram** on the predictive distribution **+ a positive-count posterior-predictive check** — does the fitted head reproduce the ~95% zero-rate *and* the positive tail? (A hurdle-NB can match the zero-rate while mis-fitting the body.)
- **π reliability pre-check (C-147):** before the gate is trusted, a **reliability diagram / Brier on `sigmoid(cls)`** (zero vs positive cells) — a class-weighted NLL can be minimized while π is miscalibrated, which would bias every `E[y]=(1−π)·μ`.

## Decision bar
Bounded (F-explode not fired) **and** magnitude un-collapsed **and** CRPS non-inferior, multi-seed stable
→ proceed toward proposed ADR (#103). Any falsifier fires → log, escalate per `02 §7` — do **not** re-tune
in place.
