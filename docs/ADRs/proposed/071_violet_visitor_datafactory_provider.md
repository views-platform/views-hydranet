# ADR-071: violet_visitor's data provider is views-datafactory (`africa_me_legacy`)

**Status:** Proposed
**Date:** 2026-07-28
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers
**Epic:** #203 · **Builds on:** ADR-060 (static_channels), ADR-050 (datafactory consumer contract, pipeline-core), ADR-047 (views-frames PF path) · **Retires to v2:** the viewser-tied lodestar foundation (`reports/2026-07-17_lodestar_eval_dossier`)

## Summary (read this first — self-contained)

violet_visitor pulls its conflict targets from **viewser** (PRIO PostgreSQL, VIEWSER 6). We are moving
its data layer to **views-datafactory** (`region="africa_me_legacy"`, 13,110 PRIO-GRID cells) for three
reasons: (1) the maintainer does not trust viewser beyond the conflict targets, and **population** — the
highest-information untried lever after the magnitude wall — lives in the datafactory; (2) the eventual
**legacy→global** scale-up requires the datafactory; (3) doing it now avoids sinking more work into a
viewser-era foundation we would abandon anyway.

This is a **clean-cut**: results on datafactory data (**v2**) are intentionally **not comparable** with
viewser-era results (**v1**). The frozen lodestar scoring *functions* are reused byte-identical; only the
**truth** they score against is re-anchored (v1 viewser → v2 datafactory).

The platform already supports this: `views-pipeline-core` dispatches on the data source, so a model
migrates by returning the **datafactory dict descriptor** from `config_queryset.generate()`. No
pipeline-core work is required.

## 1. Context

- **The swap seam is one file:** `models/violet_visitor/configs/config_queryset.py::generate()`. Today it
  returns a viewser `Queryset`; the datafactory descriptor (template: `light_strider`) returns a dict with
  `source:"views-datafactory"`, `region:"africa_me_legacy"`, `loa:"priogrid_month"`, `zarr_url`, and a
  `FEATURE_RENAME` mapping `ged_sb/ns/os_best → lr_*_best` and `gaul0_code → c_id`.
- **The platform is already built** (do not rebuild): `ViewsDataLoader.get_data()` inspects
  `get_queryset()` and dispatches — `Queryset`→viewser (untouched); dict `source:"views-datafactory"`→
  `_fetch_data_from_datafactory` (pandas) **or** the frame-native `feature_frame_path` (FeatureFrame,
  S=1 contract). Requires `views-datafactory>=1.9.0` (ADR-050 consumer contract). `bright_starship` /
  `light_strider` run on it.
- **Tier-A parity already PASSED on a FRESH pull** (dossier `07` E1): vs viewser truth on months 121–504,
  cell set **identical** (13,110), exact-match 99.98/99.99/99.99%, identical extreme maxima, and **no**
  pre-registered falsifier (F-A1..F-A4) fired. The remote reaches `last_valid_month_id=559`.

## 2. Decision

violet_visitor's data provider is **views-datafactory**, `region="africa_me_legacy"`, via the datafactory
dict descriptor returned by `config_queryset.generate()`. The migration is executed as Epic #203 (S1–S10).

### The difference ledger (accepted, intentional)

| # | Difference | Disposition |
|---|---|---|
| **L1+L3** | **KGI (Known Geographical Imprecision) handling.** viewser's `ged_*_best_sum_nokgi` = KGI *unhandled* (a legacy stub a departed researcher never finished); the datafactory handles it. Observable as events placed on a different (usually neighbor) gid than viewser's default cell (~0.024% of cells; small-count-dominated; per-month totals largely conserved). | **Accepted — this is the intended improvement.** No attempt to reproduce viewser's unhandled `_sum_nokgi`. |
| **L2** | **GED vintage revisions** — small non-conserving count changes between GED vintages (net total drift sb −0.35% / ns +1.25% / os +0.06% on the fresh pull). | Accepted — benign version drift. |
| **L4** | **Coverage** — the fresh remote reaches month 559 (a stale on-disk cache reached only 492). | Resolved by pulling FRESH (never validate on cached data). |
| **L5** | **`c_id` recode** — viewser `country_id` vs datafactory `gaul0_code`, `−1` sentinel for unassigned. | Accepted — feature semantics; not a target. |

### Locked decisions (Epic #203 S1) — *recommended defaults, pending maintainer confirmation*

1. **Fetch path:** use the **pandas `_fetch_data_from_datafactory`** path now (matches violet's current
   ingestion). Switch to the **frame-native `feature_frame_path`** in **S9** as a separate,
   byte-identical-verified step. Rationale: do not entangle the data-provider swap with the pandas-removal
   refactor — a parity failure must be unambiguously the *data*, not the refactor.
2. **nokgi:** accept the datafactory's KGI handling as-is (L1+L3); do not reproduce viewser's unhandled variant.
3. **Population (S6):** target the datafactory population feature; validate **externally** (no viewser
   reference); encode as `ln_pop = log1p(population)` in `static_channels` (ADR-060 seam), input-only.

### Clean-cut / v1↔v2

- v1 = viewser-era foundation (`reports/2026-07-17_lodestar_eval_dossier`, ARCHIVED as reference).
- v2 = datafactory-era foundation (this epic): the frozen lodestar **truth** is re-anchored to the fresh
  datafactory calibration parquet; the scoring **functions** are reused byte-identical. v1 and v2 numbers
  are **not** comparable; Tier-B checks that the *conclusions* survive (foundation beats `white_ranger`;
  body still timid), not the values.

## 3. Consequences

**Positive:** population + trusted covariates become available; global scale-up is unblocked; violet stops
depending on viewser; the KGI handling is a genuine data-quality improvement.

**Negative / accepted:** v1 results are retired (clean-cut); the v2 foundation must be re-established
(GPU cost — S5); population has no viewser reference so its validation is external-only (S6).

**Invariants preserved:** floor config md5 `6c28bdb1390fc413d43b2d74d87251f8`; the model architecture,
family head/D×K sampler (ADR-067), composition (ADR-069), and sample-feedback (ADR-070) are unchanged —
the only behavioral change is *where the data comes from* (+ the `ln_pop` input channel); FAO-02 selection
metrics only (CRPS + QS99/Brier/MCR; no twCRPS/PIT/LogScore for selection).

## 4. Alternatives considered

- **Stay on viewser + graft population from another source** — rejected: the maintainer distrusts viewser's
  non-conflict layer, and global scale-up needs the datafactory regardless.
- **Branch off a datafactory model (blazing_meteor/bright_starship)** instead of migrating violet — rejected:
  loses violet's experimental history/config lineage; the swap seam is a single file, so migrating in place
  is cheaper.
- **Validate on the cached datafactory parquet** — rejected (maintainer directive): cached pulls go stale
  (the cache was 12 months short); we pull FRESH or we have found a problem.
- **Byte-parity as the acceptance bar** — rejected: intentional preprocessing differences (KGI, vintage)
  make byte-parity impossible and undesirable; the bar is a *decomposed, explained* residual (Tier-A).

## 5. Validation

Tier-A parity harness (Epic #203 S2, `reports/2026-07-28_datafactory_migration_dossier/tools/`) re-runnable
on a fresh pull; v2 baseline + Tier-B (S5); population external validation (S6); `ln_pop` OFF==baseline
parity (S7) + pre-registered A/B (S8). Evidence trail: the migration dossier (00–07).
