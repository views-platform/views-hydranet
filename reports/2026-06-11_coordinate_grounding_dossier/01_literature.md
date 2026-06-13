# 01 — Literature

**Date:** 2026-06-11 · Five cites + the domain argument. All registered in the library (2026-06-13);
cited by `papers/<filename>.pdf`. Full annotation lives in ADR-061's Literature table.

## Anchors for the chosen design
- **Liu et al. 2018 — CoordConv** (`papers/Liu2018_CoordConv.pdf`). *Foundational.* A standard CNN is
  translation-invariant by construction and **provably cannot solve the coordinate transform**; adding
  hard-coded (x,y) channels fixes it at the cost of a couple of lines, and **reduces mode collapse** in
  generative settings. Our blob-bloom is a spatial mode-collapse analog. This is the mechanism we adopt.
- **El Jurdi et al. 2021 — CoordConv-Unet** (`papers/ElJurdi2021_CoordConvMedSeg.pdf`). *Closest analog.*
  CoordConv in a **U-Net** stabilizes training and **evades local minima specifically under prior-based
  losses** — and is **inert** with a plain baseline loss. ⚠ **Partial analogy:** their "prior" is an
  *added spatial shape-regularizer* (two-term interchange CoordConv stabilizes); our hurdle-NB is a
  *distributional likelihood* — no equivalent interchange. So this is a *plausibly analogous*, not
  *identical*, regime; the analogy could fail *because* our prior is distributional, not spatial. ⚠ Also
  single-pass medical segmentation — does **not** validate autoregressive/rollout stability. Both gaps
  make `05`'s experiment the real test, not the literature.

- **Islam et al. 2020 — Position Information** (`papers/Islam2020_PositionEncoding.pdf`). *Nuance
  (carried from ADR-029).* CNNs **do** encode some absolute position via zero-padding border effects —
  just not enough; explicit positional channels significantly help spatially precise tasks. Qualifies
  Liu's "cannot represent position" to "leaks limited position implicitly, insufficient" and motivates
  explicit injection.

## Risk + escalation references
- **Tancik et al. 2020 — Fourier Features** (`papers/Tancik2020_FourierFeatures.pdf`). Raw coordinate
  inputs have a **spectral bias toward low frequencies** — a smooth `(row,col)` function may under-capture
  the sharp geography of where people live. Names our pre-registered **smooth-proxy risk** and the
  principled escalation (Fourier-encode the coordinates).
- **Ding & Gao 2025 — GCA-ResUNet** (`papers/Ding2025_GCAResUNet.pdf`). Direction-aware coordinate
  **attention** as a plug-and-play U-Net module addressing CNN locality — the escalation path if
  input + top-skip coordinates underdeliver.

## The domain argument (not in the held literature)
Conflict is a **spatially near-degenerate** process: the vast majority of grid cells have a
**structural-zero** base rate for domain reasons (no people, no roads, no cities; ocean, desert). A
position-aware model can learn that prior; a translation-invariant one cannot, and instead fires on local
patterns wherever they occur — the diagnosed spatial over-firing.

## Gaps to fetch
- **None blocking.** No held paper studies CoordConv under autoregressive rollout or for conflict
  forecasting — a real gap, substituted **empirically** by the rollout-biopsy + gate-forensic readout
  (`03`/`05`) rather than waiting on literature. (Parallels the ZINB dossier's autoregressive-stability gap.)
