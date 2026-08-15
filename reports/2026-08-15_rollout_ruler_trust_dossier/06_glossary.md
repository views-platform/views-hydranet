# 06 — Glossary

**Date:** 2026-08-15 · **Epic:** #263

The repo's locked vocabulary is `reports/GLOSSARY.md` — **use it, do not invent synonyms.** Terms already
locked there and load-bearing here: **gate**, **body**, **gated forecast** (= gate × body), **crps-all**,
**crps-events**, **crps-none**, **size-ratio**, **pos-mcr**, **AP**, **Brier**, **sample cube**, **T=0**,
**baseline** (white_ranger = climatology).

Note the glossary's `pos-mcr` (mean of per-cell ratios on conflict cells) and FAO-02's `MCR = ȳ_pred/ȳ_true`
(ratio of means, all cells) are **different estimators sharing three letters**. The v2 scoreboard resolves this
by emitting `mcr_all` / `mcr_events` / `mcr_none` separately; this dossier follows that.

---

## Terms this programme introduces

| Term | Definition |
|---|---|
| **zero_share_of_gap** | The fraction of a `Δcrps_all` between two arms attributable to true-zero cells: `(1−pₑ)·Δcrps_none / Δcrps_all`. Emitted on every headline row. `> 0.5` means the "win" is confident zeros. |
| **crps_gap_decomposition** | The exact split `Δcrps_all = (1−pₑ)·Δcrps_none + pₑ·Δcrps_events`. Its `residual` is ~0 by construction and is asserted `<1e-9`. |
| **FAO-02 climatology** | The reference FAO-02 mandates. **Canonically implemented as `ConflictologyModel` in views-baseline**, deployed as `white_ranger` / `light_strider`. `climatology_resample` in this repo is a **scorer-side stand-in** for it (needed because the deployed model's cubes are deleted after scoring), matched to its parameters and validated at 0.9591 vs its archived 0.9601. Duplication tracked as **C-279**; the fixed-vs-sliding window question as **views-baseline #82**. Distinct from `_persistence_gathered` (1-sample, degenerate). |
| **crpss_vs_clim** | `1 − crps_model / crps_climatology`. FAO-02 superiority ≥ 0.05. Raises on a 1-sample reference. |
| **diag_Tu** | The Taillardat CRPS-distribution index `T_u(F,G) = 1 − Ω_G/Ω_F`. **DIAGNOSTIC ONLY** — reported, never selected on, and structurally unable to yield a standalone per-model number. |
| **MDE** | Minimum detectable effect: the smallest `Δcrps_all` a 13-origin block bootstrap separates from 0 at 90%. Reported so a null is not read as "no difference". |
| **the substrate** | The frozen (truth, months, cells) triple a comparison is made on. Comparing across substrates is the C-7 "different-months bug". |
