# v2 truth — provenance (Epic #203 S4)

The frozen datafactory truth the v2 ruler scores against. The clean-cut re-anchors only the TRUTH;
the lodestar scoring functions are reused byte-identical (see `v2_ruler.py`).

| field | value |
|---|---|
| artifact | `tools/v2_truth/calibration_datafactory_df.parquet` (gitignored binary; identity tracked here) |
| **sha256** | `620f4aa3722fec9fb3965ada5ed3f041ae45a72a5937b5ac748453847d2ce84b` |
| source | views-datafactory `region="africa_me_legacy"`, `loa="priogrid_month"` |
| provider | violet_visitor queryset (datafactory descriptor, ADR-071) via pipeline-core `get_data` |
| pull | FRESH platform fetch 2026-07-28 (remote `last_valid_month_id=559`) |
| support | months 121–504 (384) × 13,110 cells = 5,034,240 rows (calibration partition) |
| columns | `lr_sb_best, lr_ns_best, lr_os_best, c_id, row, col` (schema-identical to viewser) |
| Tier-A | PASS vs viewser (dossier 07 E1); residual = KGI handling + vintage (ledger L1–L3) |

**Regeneration:** a fresh calibration fetch reproduces this up to GED vintage drift (not byte-stable
across vintages — that is why we FREEZE this artifact rather than re-pull). To re-freeze deliberately
(e.g. a data refresh), re-run the S3 fetch and update the sha256 above + `v2_ruler.V2_TRUTH_SHA256`.

**v1 (viewser) truth** — `reports/2026-07-17_lodestar_eval_dossier` — is ARCHIVED as reference; v1 and
v2 numbers are intentionally incomparable (clean-cut).
