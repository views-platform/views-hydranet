# 01 — Literature (2026-08-08)

This ensemble reuses methodology already surveyed + held; **no new fetch** required for S0. Pointers:

## Held / applied (from prior dossiers)
- **Giacomini & White (2006)** — conditional predictive-ability test; the primary decision statistic
  (ensemble-vs-member, ensemble-vs-climatology), nested-valid. Surveyed in `2026-07-29_v2_scoreboard_dossier`
  and `2026-08-01_tail_decoupled_head_dossier/01_literature`.
- **Gneiting (2011), Lerch et al. (2017)** — proper scores + the Forecaster's Dilemma (never select on
  `crps_events`); why the stratified-proper + GW readout is the honest one. *(held)*
- **Hersbach (2000)** — CRPS ensemble decomposition (the `crps_ensemble` primitive the ruler uses). *(held)*

## Internal evidence base (the real prior art — see 00_README)
- v2 scoreboard: gated_NB ≡ th_gated_NB ship candidate; ZINB falsified + blooms.
- tail_decoupled_head: mixture NULL on magnitude (F4-clean) → mixture is a structural hedge, not a magnitude fix.
- datafactory_migration: Tier-A parity protocol (fresh-pull).

## Gaps to fetch (only if the design widens)
- Ensemble-combination theory for *proper-score-optimal* pooling (if we ever move off equal-weight `concat` to
  weighted pooling — currently OUT of scope).
- None blocking S0.
