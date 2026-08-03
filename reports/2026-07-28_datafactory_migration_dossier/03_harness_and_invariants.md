# 03 — Harness & Invariants (the crown jewel)

**Drafted:** 2026-07-28. The guardrails that make this migration safe. `status` refuses "ready to run"
until the pre-flight checklist (§D) is green.

## A. Invariant taxonomy

### 1. Hard invariants — never break (a violation = invalid experiment)
- **Fresh data or bust.** NEVER validate against cached/on-disk pulls (they go stale — the cache used
  for the indicative diff was 12 months short). Parity gates run on a FRESH pull. *(user correction)*
- **Floor config md5 `6c28bdb1390fc413d43b2d74d87251f8`** unchanged; config-mutating runs use the
  trap-restore pattern.
- **Never commit/push `views-models`.** (violet_visitor lives there.)
- **Fail-loud, no silent clamp/fallback** (DataSniffer, device, contract) — no silent CPU fallback.
- **Full suite + ruff green** before any ship.
- **Frozen lodestar SCORING FUNCTIONS reused byte-identical** (`crps_ensemble`, AP/Brier) — the
  *code* is invariant; only the TRUTH it scores against is re-anchored (v1 viewser → v2 datafactory).

### 2. Deliberately changed by this program
- **Data provider** viewser → views-datafactory (the queryset descriptor). *Replaces* the viewser pull.
- **The truth parquet** v1 (viewser) → v2 (datafactory `africa_me_legacy` fresh pull) — the lodestar
  ruler is re-anchored; v1/v2 numbers are **intentionally incomparable** (clean-cut).
- **Adds a `ln_pop` static_channel** (input only; ADR-060 seam).
- **`c_id` semantics** (country_id → gaul0_code, −1 sentinel) — accept the recode.

### 3. Respect while changing (not targeted, breakable in passing)
- Model arch / family head / D×K sampler / composition (ADR-069) / sample-feedback (ADR-070) — all
  must stay byte-identical in behavior (only the input data changes).
- The views-frames PF emit path (tasks #67–71) — P1 touches ingestion only; deeper propagation is P4.
- The pandas downstream — left intact through P1–P3 on purpose (sequencing lock).

## B. Standing harness — ALREADY EXISTS (reuse, don't reinvent)
| Mechanism | Present? | Where |
|---|---|---|
| Grid-naming canonicalization (priogrid_id/gid) | ✅ | `utils/grid_naming.py`, `manager/hydranet_manager.py` |
| Config validators / fail-loud schema | ✅ | `utils/config_initializer.py` (HydraNetConfig) |
| Floor-md5 trap-restore around config-mutating runs | ✅ | driver pattern (mag_sup_sweep.sh etc.) |
| Frozen eval ruler (scoring fns + protocol) | ✅ | `reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py` |
| Parity scripts (cell-set, exact%, invariant) | ✅ (in job tmp) | `parity_targets.py`, `parity_invariant.py` — **promote to tools/** |
| views-frames PF ingestion (native) | ✅ | ADR-047 path; datafactory `load_dataset()` returns frames |
| Reproducibility (seed lock, determinism flag) | ✅ | existing seed/determinism harness |
| Static_channels seam (population rides here) | ✅ (empty) | ADR-060 |
| Credentials for the pull | ✅ | `~/.netrc` (host 204.168.219.108, 2 entries) |

**Honest finding: ~75% of the harness is already in place.** The new build is §C.

## C. New harness this program needs (gaps — build BEFORE the first gate)
1. **Fresh-pull capability, verified in-env** — confirm `views-datafactory` importable in
   `views-hydranet-env`; a FRESH authenticated pull that reaches month ≥504. *(env install unconfirmed)*
2. **violet queryset swap** — datafactory descriptor (from light_strider), with `nokgi` (L1) resolved
   or explicitly accepted.
3. **Tier-A parity harness re-pointed at FRESH pulls** — promote the two parity scripts to `tools/`,
   parameterize the pull path, add the ledger-reconciliation report (does the fresh diff match L1–L5?).
4. **v2 truth re-anchor** — freeze the datafactory calibration parquet as the v2 lodestar truth;
   a thin adapter so `lodestar_score.py` scores against v2 (same functions, new truth/months/cells).
5. **Covariate external-validation harness** — population sanity (vs known populations, monotonic,
   non-negative, full coverage) — NO viewser reference.
6. **`ln_pop` static_channel wiring + tests** — input-only channel; parity test proving OFF == current.

## D. Pre-flight checklist (must be green before the FIRST experiment = Tier-A parity on fresh data)
- [ ] `views-datafactory` importable in `views-hydranet-env` — **blocker**
- [ ] FRESH authenticated pull succeeds and reaches month ≥504 (user runs via `!`) — **blocker**
- [ ] parity scripts promoted to `tools/`, re-pointed at the fresh pull
- [ ] **Tier-A parity plan pre-registered** (05) with predictions + falsifiers BEFORE looking at the fresh diff
- [ ] full suite + ruff green on `views-hydranet` (queryset swap lives in `views-models`, but any hydranet-side load changes gated)
- [ ] new failure modes (nokgi mismatch, coverage short, population out-of-range) noted for the register

## E. Rules of engagement
One variable at a time; pre-register then run; **fresh readout before trusting** (parity on fresh data
gates everything); falsifier honesty; the migration is **data-provenance-neutral by construction** —
any target change must trace to a ledger item (L1–L5), never to a silent transform.
