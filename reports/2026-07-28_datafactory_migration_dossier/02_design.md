# 02 — Design: the viewser → views-datafactory swap

**Drafted:** 2026-07-28 · graduates to a proposed ADR on `promote`.

## The swap seam (minimal, one boundary)
violet_visitor's data ingress is `configs/config_queryset.py::generate()`. Today it returns a viewser
`Queryset` (VIEWSER 6). The migration replaces it with the **datafactory descriptor** already proven
in `light_strider`:

```python
return {
    "name": model_name,
    "source": "views-datafactory",
    "zarr_url": DEFAULT_REMOTE.zarr_url,
    "region": "africa_me_legacy",   # 13,110 cells = viewser Africa+ME
    "loa": "priogrid_month",
    "features": FEATURE_RENAME,      # ged_*_best → lr_*_best ; gaul0_code → c_id
}
```
Downstream model code is unchanged because `FEATURE_RENAME` maps datafactory names back to the
`lr_*_best`/`c_id` the model already consumes. The grid-naming canonicalization (already shipped)
absorbs the `priogrid_gid`↔`priogrid_id` index-name difference at the data-load boundary.

## The empirical difference ledger (derived — no doc exists)
Compared a datafactory pull vs the viewser truth on shared support. **⚠️ the numbers below are
INDICATIVE — they ran on a STALE cached pull; they are re-derived on a FRESH pull as the real Tier-A
gate (05).** What they tell us about *what to expect*:

| # | Difference | Evidence (FRESH pull, 07 E1) | Class |
|---|---|---|---|
| L1+L3 | **KGI handling** — viewser `ged_sb_best_sum_nokgi` = "**no KGI**" = Known Geographical Imprecision **unhandled** (a legacy column a departed researcher stubbed and never finished); the datafactory **handles KGI properly**. The observable effect is L3: imprecisely-located events placed on a different (usually neighbor) gid than viewser's default cell. | ~0.024% of cells reassigned; small-count (38% ±1, 65% ≤3); neighbor-gid pairs; per-month totals largely conserved by the reassignment part | **intentional improvement** (the reason to migrate) |
| L2 | **GED vintage revisions** | net total drift sb −0.35% / ns +1.25% / os +0.06% (non-conserving part) ⇒ events added/revised between GED vintages; temporally diffuse | benign version drift |
| L4 | **Month coverage** | fresh remote reaches **559** (stale cache was 492); pull covers full 121–504 | resolved by fresh pull |
| L5 | **`c_id` recode** | viewser `country_id` vs datafactory `gaul0_code`, `-1` sentinel for unassigned | feature semantics |
| — | **Cell set + maxima** | **identical** (13,110 cells; sb 113395 / ns 2000 / os 178459) | PARITY ✓ |

**Interpretation:** the conflict backbone is the same GED. The entire residual decomposes into (a) the
datafactory's **KGI handling** (L1+L3 — a desirable, intentional improvement viewser lacked) and (b)
benign **vintage revisions** (L2). Nothing is unexplained. Confirmed on a FRESH pull (07 E1) — this is
no longer indicative.

## Population + covariates (the actual motivation)
- **No viewser counterpart** → cannot be validated by diff. Validated **externally** (vs known
  country/region populations, monotonic-in-time sanity, no-negative, coverage) — pre-registered as its
  own section with falsifiers (05 covariate plan).
- **Encoding:** `ln_pop = log1p(population)` appended as a per-cell **static_channel** (ADR-060 seam,
  currently empty in violet). Input-only; does not touch the target/emit path.

## Sequencing (LOCKED — user agreed; don't entangle)
1. **Data swap + parity FIRST** (P0–P2): provider swap, conflict-target byte parity on fresh data,
   v2 re-foundation. Downstream code, incl. the pandas paths, **untouched**.
2. **THEN pandas-removal / views-frames-native propagation** (P4) as a separate step with its own
   byte-identical proof against the v2 baseline. Rationale: a parity failure must be unambiguously the
   *data*, not a refactor. views-frames is adopted only at the ingestion boundary in P1 (since the
   datafactory returns it natively) and propagated deeper later.

## What this design deliberately does NOT change
Model architecture (`HydraBNUNet06_LSTM4`), the family head / D×K sampler (ADR-067), the composition
axis (ADR-069), sample-feedback (ADR-070), the floor hyperparameters (md5 `6c28bdb…`). The only
intended behavioral change is **where the data comes from** (+ the population input channel).
