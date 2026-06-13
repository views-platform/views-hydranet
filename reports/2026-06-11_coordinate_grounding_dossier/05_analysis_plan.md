# 05 — Analysis Plan (pre-registration)

**Date:** 2026-06-11 · Pre-registered **before** the coordinate train (roadmap box 3). Append-only.

## One variable
The bounded hurdle-NB **S1** config (θ=1.0, `pos_weight`=10, frozen balancer, scheduled sampling off,
40 lessons) **+ coordinate channels** (ADR-061) **vs** the same config without them (the 6-run baseline of
record, `07` "before"). Nothing else changes.

> **Comparator prerequisites (falsify P5/C-155 + C-151).** Before this comparison is valid: pin the
> baseline to `config_hyperparameters.py` (hurdle_nb) + its recorded per-arm env + seed + the C-42
> reproducibility lock; **quarantine the stale `config_sweep.py` (tobit)** so coords are not benchmarked
> against a Tobit baseline; confirm `feedback_clamp` was **off** in the baseline (C-151) so "bounded" is
> intrinsic; and demonstrate I5 (toggle-off **bit-identical**) against a re-run, not just the recorded row.

## Hypothesis
HydraNet's spatial over-firing is a symptom of **position-blindness**: a translation-invariant CNN cannot
represent that most cells are structural zeros. Injecting absolute coordinates lets the model learn a
**spatial base-rate prior**, so (a) the onset gate stops flooding structural-zero regions and (b) the
rollout stops blooming blobs there — moving full-horizon MCR toward 1 **without** any loss change.

## Pre-registered prediction (if coordinates are the lever)
- **Gate forensic:** the "Detection Bias Pulse" event-ratio **stops climbing** to 4–16× and flattens toward ≈1.
- **Rollout biopsy:** blobs **stop blooming in structural-zero regions** specifically (not merely shrink uniformly).
- **MCR:** **FULL MCR moves toward 1** (step-1 was already ~0.4–0.7; the drift is the target).

## Falsifiers (pre-committed)
- **F-persist:** blobs persist or merely **relocate**, and/or the gate still floods (event-ratio still
  climbs) ⇒ coordinates are **not** the lever ⇒ escalate to **static covariates** (ADR-060 enables) — **not**
  a return to loss-level tinkering, and **not** re-tuning coords in place.
- **F-smooth-proxy:** any gain is weak/ambiguous and concentrated where geography is smooth, failing on
  sharp settlement structure ⇒ the smooth-coordinate proxy is too weak (Tancik 2020) ⇒ escalate to
  Fourier features or covariates. *Distinguish from F-persist by where the residual blobs sit.*
- **F-volatility:** run-to-run variance across **≥2 seeds** is large ⇒ volatility disqualifies (the
  shrinkage lesson), regardless of mean.
- **F-offpath:** the toggle-off run is **not** bit-identical to baseline (ADR-060 I5) ⇒ the seam is wrong;
  the comparison is invalid ⇒ stop and fix the seam.

## Metrics (FAO PRN-05 — logged to `../RESULTS_LOG.md`)
- **CRPS** = primary ranking (superiority = ≥5% better than baseline).
- **QS99 / Brier / MCR** = guardrails (non-inferiority; MCR closest-to-1, **diagnostic only**).
- Per target (sb/ns/os), full grid **and** positive-cell subset.
- **Multi-seed** (≥2; ≥3 if promising).
- The three diagnostic instruments (`03`): gate forensic, rollout biopsy, MCR readout — read **side-by-side**
  against the baseline plots, since the diagnosis was visual.

## Decision bar
Prediction holds (gate flattens **and** blobs vacate structural-zero regions **and** FULL MCR toward 1),
CRPS non-inferior, multi-seed stable → **ship** (ADR-061 → Accepted). Any falsifier fires → log, escalate
per `04` box 4 — do **not** re-tune in place.
