# 06 — Glossary

**Date:** 2026-06-11 · Shared vocabulary for the coordinate-grounding work.

- **Dynamic (endogenous) channel** — a channel the head **predicts** and feeds back autoregressively; a
  prediction target; inverse-transformed and present in the prediction frame. The conflict histories
  (`lr_sb_best`, `lr_ns_best`, `lr_os_best`). (ADR-060 Axis A.)
- **Static (exogenous) channel** — an **input-only** channel: injected, never predicted, never a target,
  never inverted, re-applied as true values every rollout step. (ADR-060 Axis A.)
- **Architectural / derived channel** — a static channel sourced from the **tensor geometry** (grid
  index); always available, deterministic. Coordinates. (ADR-060 Axis B.)
- **Measured / exogenous-data channel** — a static channel sourced from an **external static raster**
  (population, terrain, ocean); fetched, grid-aligned, per-channel scaled. The covariate escalation.
  (ADR-060 Axis B.)
- **CoordConv** — giving convolution its own coordinate channels so it can condition on absolute position
  (Liu et al. 2018), curing the translation-invariance "intriguing failing."
- **Derive-then-slice** — build the coordinate channels over the full grid, then slice with the **same**
  window indices as the dynamic channels ⇒ global alignment by construction (ADR-060 I4).
- **Structural zero** — a grid cell whose conflict base rate is ~0 for domain reasons (no people / roads
  / cities; ocean, desert), not by chance. The bulk of the grid.
- **Spatial over-firing** — the onset gate predicting events (and the head predicting magnitude) in
  structural-zero regions; the diagnosed failure coordinates are meant to cure.
- **Blob-bloom** — the bounded, localized growth of predicted-conflict "blobs" over the autoregressive
  rollout, in places ground truth never has conflict. The spatial face of the bounded-but-drifting result.
- **Smooth-proxy risk** — raw `(row,col)` channels carry a spectral bias toward low-frequency functions
  (Tancik et al. 2020), so they may under-capture the sharp geography of settlement; the named risk of
  coordinates-as-proxy.
- **Bit-identity (off-path)** — with the coordinate toggle off, the pipeline is byte-identical to the
  pre-coord baseline (ADR-060 I5); the precondition for a clean one-variable comparison.
- **MCR** — Magnitude Calibration Ratio (ŷ/y), a **diagnostic** ratio (target = 1), never the optimization
  objective. CRPS is the primary proper score.
