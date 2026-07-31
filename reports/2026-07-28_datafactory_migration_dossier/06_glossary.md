# 06 — Glossary (migration-local)

Shared vocabulary is `reports/GLOSSARY.md` (LOCKED — use it; edit there, never invent synonyms). Terms
this dossier introduces:

- **Fresh pull** — data fetched live from the datafactory now. The ONLY valid parity/validation source.
  A cached on-disk parquet is NOT a fresh pull and is never a source of truth.
- **Tier-A (data parity)** — does the datafactory serve the same INPUT conflict data as viewser (modulo
  the ledger)? The go/no-go gate. Validates DATA, not predictions.
- **Tier-B (result correspondence)** — do the foundation CONCLUSIONS survive on datafactory truth?
  Expected to differ numerically (clean-cut); checks direction, not byte-identity.
- **Difference ledger (L1–L5)** — the enumerated, explained set of intentional/known differences
  between the two providers. Any residual NOT in the ledger is a bug.
- **Clean-cut** — the accepted break in comparability: viewser-era (v1) results are not comparable with
  datafactory-era (v2) results; the frozen ruler is re-anchored to v2 truth.
- **v1 / v2 foundation** — v1 = viewser-tied lodestar (2026-07-17). v2 = datafactory-tied re-anchored
  ruler + baseline (this dossier).
- **`ln_pop`** — `log1p(population)`, the population covariate encoded as an input-only static_channel.
- **KGI / nokgi** — **KGI = Known Geographical Imprecision**: GED events whose location is only
  approximately known. `nokgi` ("no KGI") = the viewser column `ged_sb_best_sum_nokgi` where KGI is
  **not handled** (a legacy stub a departed researcher never finished). The **datafactory handles KGI**
  properly. Ledger L1; its observable effect is L3 (imprecise events placed on a different gid than
  viewser's default) — an intentional improvement, not drift.
