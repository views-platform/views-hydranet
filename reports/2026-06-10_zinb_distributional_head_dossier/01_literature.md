# 01 — Literature

**Date:** 2026-06-10 · Full prior bibliography: `../archived/2026-06-05_distributional_head_dossier/01_literature.md` (pointer; not duplicated).

## Anchors for the chosen design
- **Iacus et al. 2025 — DynAttn** (`~/brain/9_library/incoming/deep_consored/2512.21435v1.pdf`). A **ZINB softplus head on the exact VIEWS/PRIO conflict data**, beating the VIEWS baseline. Strongest prior that a ZINB count head fits this DGP. ⚠ It forecasts **direct multi-horizon (no autoregressive feedback)**, so it does **not** validate that a ZINB head stays bounded under our 36-step rollout — that claim is tested by the explosion-check gate (`03`), not assumed.
- **Lambert 1992 — Zero-Inflated Poisson**; **Cragg 1971 / Mullahy 1986 — hurdle models.** Theory for the π-gate × positive-body factorization and the two-independent-gradient-paths argument.
- **Negative Binomial** (standard) — over-dispersed count likelihood; θ dispersion. Closed-form NLL (no Tweedie-density blocker).

## Parked-fallback references (only if ZINB fails — see archived rollout dossier)
- **Hess et al. 2023 — GTF**; **Brandstetter — pushforward** (rollout training, parked #77/#78).

## Gaps to fetch
- **None blocking.** No held paper studies the *autoregressive stability* of a distributional count head — this is a real gap, and the **explosion-check gate substitutes for it empirically** rather than waiting on literature.
