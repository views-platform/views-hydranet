# 03 — Harness & Invariants

**Date:** 2026-06-11 · The standing checks the build must satisfy and the readout the experiment is judged by.

## Standing invariants (from ADR-060 — the static-channel contract)
- **I1 — Never a target.** Coordinates never appear in `regression_/classification_targets`;
  `output_channels` stays 3. *Test (Red):* a config with a coord in a target list → validator raises.
- **I2 — No inversion, not in the frame.** Coordinates never pass through `transformations` / inverse-
  transform and never appear in a `PredictionFrame`. *Test (Red):* a coord surfacing in any frame → fail.
- **I3 — Static across the rollout.** Coordinates are re-injected with true values at every step; never
  overwritten by model output. *Test (Beige):* assert the coord channels are identical at every rollout
  step and equal to their source.
- **I4 — Alignment by construction.** Coordinates are derived over the full grid and sliced with the same
  window indices as the conflict channels. *Test (Beige):* the coord slice for a window matches the
  dynamic-channel window indices (a known-corner probe).
- **I5 — Off-path bit-identity.** With the toggle off, the full pipeline output is **byte-identical** to
  the pre-coord baseline. *Test (Green):* run the baseline config with the flag off → identical artifact.
- **I6 — Augmentation sync.** Any spatial transform applied to the conflict channels (training-time
  flips/rotations, the North-Up orientation flip) is applied identically to the coordinate channels.
  *Test (Beige):* a flipped/rotated window's coord channel matches the transformed grid position.
  (Carried from ADR-029 — coords that don't flip with the data encode the wrong position.)

## Build-specific checks
- **Range check.** Coordinate channels lie in `[-1, 1]`; the four grid corners map to the expected
  extremes (a 2×2 corner probe).
- **Shape contract.** Input tensor has 5 channels (3 dynamic + 2 static) at the first conv; the output
  head emits 3; the top-skip tensor gains exactly 2 channels.
- **No FeatureScaler contact.** Coordinates are not `log1p`'d and do not appear in the scaler's
  `transformations` (Q4) — they are produced pre-normalized in-model.

## Pre-run prerequisites (folded in from the 2026-06-13 `/falsify` audit)
Before the coordinate run is launched or trusted:
- **Explosion-check validated for count-space (C-142/P4) — ✅ DONE (#106, 2026-06-13).** The "Bounded?"
  pre-gate now composes `log1p(E[y])` exactly as inference does, via `free_running_attractor(emit_fn=…)`
  backed by the shared `views_hydranet.utils.hurdle_nb.hurdle_nb_expected_log1p` (single source of truth
  with `_emit_magnitude`). It measures what the hurdle-NB rollout actually feeds back — not count-space `mu`
  against the log-space bound. Validated by `tests/test_rollout_stability_guard.py` (an in-range count is no
  longer mis-flagged; a composed-E[y] runaway is flagged). **C-142 closed.**
- **Disk headroom (C-154/P3).** ~2.5 GB/prediction-dir; ≥2 seeds + diagnostics + baseline re-run ≈ 10–15+ GB;
  the dev volume is ~97% full. Pre-run **free-space check + cleanup**; abort if free < budget.
- **Baseline provenance pinned (C-155/P5).** Comparator = `config_hyperparameters.py` (hurdle_nb) + recorded
  per-arm env + seed + the C-42 reproducibility lock; **the stale `config_sweep.py` (tobit) quarantined**;
  `feedback_clamp` confirmed **off** (C-151).
- **Cross-cutting seam landed (C-153/P1).** The static-channel seam touches inference + SS-training feedback
  + arch + scaler (`04` box 1) — all coordinated, not a localized tweak.

## Readout protocol (how the experiment is judged — same instruments as the diagnosis)
The coordinate run is read against the bounded hurdle-NB baseline with the same instruments that produced
tonight's diagnosis:
0. **Bounded? (pre-gate).** `diagnose_io_gain` 36-step rollout stays in-range — now **count-space-valid**
   for the hurdle-NB head (composes `log1p(E[y])` like inference; C-142 closed by #106). Trustworthy.
1. **Gate forensic** — the classification "Detection Bias Pulse" (event-ratio ŷ_events/y_events over
   lessons). *Looking for:* the climb to 4–16× **flattens** toward ≈1.
2. **Rollout biopsy** — the autoregressive forensic (ground-truth / prediction / |Δ| over rollout steps,
   per origin). *Looking for:* blobs **stop blooming in structural-zero regions** specifically.
3. **MCR readout** (`scripts/mcr_readout.py`) — step-1 + full, per target (sb/ns/os), on the full grid and
   the positive-cell subset. *Looking for:* **FULL MCR moves toward 1** (diagnostic, not the target).
4. **FAO metrics** (CRPS primary; QS99 / Brier guardrails) → `../RESULTS_LOG.md`.

## Hard-stops (pause for the chair)
- Any I1–I6 invariant fails (contract violated) ⇒ stop, fix the seam — *not* the experiment.
- Off-path **not** bit-identical (I5) ⇒ the seam is wrong; stop.
- Multi-seed volatility large (the shrinkage lesson) ⇒ volatility disqualifies regardless of mean.
