# 04 — Roadmap (phased, gated)

**Drafted:** 2026-07-28. Each phase has a GATE that must pass before the next. Data-swap + parity
FIRST; pandas-removal LAST (sequencing lock).

## P0 — Fresh-pull capability  *(GATE: we can pull fresh, and it reaches ≥month 504)*
- Confirm `views-datafactory` installed in `views-hydranet-env` (install if missing).
- User runs an authenticated FRESH `africa_me_legacy` pull (calibration + validation partitions).
- **Gate:** pull succeeds, credentials reach the host, coverage reaches ≥504. If it can't pull fresh,
  STOP — that is the problem to solve (per the discipline), not a thing to work around with cache.

## P1 — Provider swap + Tier-A parity  *(GATE: conflict targets reproduce the ledger on FRESH data)*
- Pre-register Tier-A (05). Promote parity scripts to `tools/`, re-point at the fresh pull.
- Swap violet_visitor's queryset to the datafactory descriptor (resolve/accept L1 `nokgi`).
- Run Tier-A on FRESH vs viewser: cell-set identity, per-target exact-match%, per-month total
  conservation, extreme maxima, ledger reconciliation (L1–L5).
- Downstream code (incl. pandas) untouched; views-frames adopted at ingestion boundary only.
- **Gate:** cell set identical; residual matches L1–L5 (any *new* unexplained difference blocks).

## P2 — v2 re-foundation  *(GATE: a trustworthy datafactory-truth baseline exists)*
- Re-anchor the lodestar TRUTH to the fresh datafactory calibration parquet (v2). Same scoring fns.
- Re-run the floor foundation (3 seeds, 40 lessons) on datafactory data → v2 baseline.
- **Tier-B** correspondence: conclusions survive (foundation still beats white_ranger on crps-all;
  body still timid) — NOT byte-identical CRPS (clean-cut). Record the v2 numbers as the new reference.
- **Gate:** v2 baseline recorded + Tier-B conclusions hold ⇒ future experiments extend from v2.

## P3 — Population covariate  *(GATE: population validated externally + earns its channel)*
- External validation of population (vs known populations; monotone; non-negative; full coverage).
- Wire `ln_pop` into static_channels (input-only) + parity test (OFF == P2 baseline byte-identical).
- A/B: floor vs floor+ln_pop on the v2 foundation (multi-seed, frozen v2 ruler).
- **Gate:** population passes external sanity AND the A/B is pre-registered before scoring.

## P4 — views-frames-native / pandas removal  *(GATE: byte-identical vs P2/P3 baseline)*
- Propagate views-frames deeper; remove pandas from the downstream. Separate, verified step.
- **Gate:** end-to-end output byte-identical to the pre-refactor baseline (pure refactor, no behavior).

## P5 — Global scale-up prep  *(out of this dossier's first arc; note only)*
- Swap `region: africa_me_legacy` → global (`heavy_strider`'s superset). Re-run the parity + coverage
  discipline at global scale. Deferred; recorded so the design doesn't paint us into a legacy corner.

## Dependency graph
P0 → P1 → P2 → {P3, P4 (independent, both need P2)} → P5.
P3 and P4 both depend on the v2 baseline (P2) but not on each other; P4 kept last to avoid entangling
the refactor with the covariate result.
