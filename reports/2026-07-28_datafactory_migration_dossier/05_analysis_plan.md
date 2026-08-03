# 05 — Analysis plan (Tier-A parity)

**Status: LOCKED 2026-07-28 — pre-registered BEFORE the fresh pull is fetched/diffed.**

Predictions + falsifiers below are committed before any FRESH datafactory data is seen:

- **Hypothesis:** a FRESH `africa_me_legacy` pull reproduces the viewser conflict targets up to the
  derived ledger (L1 nokgi, L2 vintage drift, L3 geocoding, L4 coverage, L5 c_id) and NOTHING more.
- **One variable:** data provider (viewser → datafactory), conflict targets only.
- **Pre-registered predictions (to commit before looking at fresh data):** cell set identical (13,110);
  per-target exact-match ≥ ~99.9%; per-month total conservation within the L2 band (sb ~−0.75%);
  identical extreme maxima; every residual traces to a ledger item.
- **Falsifiers (pre-committed):** (F-A1) cell set differs beyond L4 coverage ⇒ wrong region/grid.
  (F-A2) an unexplained target difference not attributable to L1–L5 ⇒ preprocessing bug — BLOCK.
  (F-A3) fresh pull does not reach ≥month 504 ⇒ pull/coverage problem — BLOCK P1.
  (F-A4) extreme maxima differ ⇒ the CRPS-driving tail is not preserved — BLOCK.
- **Metrics:** the parity scorecard (cell-set identity, exact%, per-month total ratio, maxima,
  ledger reconciliation). **Data validation, not prediction-correlation** (the locked discipline).

Covariate (population) validation gets its **own** pre-registration section when P3 is reached
(external sanity vs known populations; falsifiers: out-of-range, non-monotone, coverage gaps).
