# Datafactory Migration Dossier — violet_visitor: viewser → views-datafactory

**Status:** Epic #203 · **S1–S5 DONE** (2026-07-28) · v2 baseline established (Tier-B PASS on reproduction+structure) · next = population track (S6–S8) or frame-native (S9)
**Owner program:** migrate violet_visitor's data layer off viewser onto views-datafactory
(`africa_me_legacy`), to (a) add **population** + trusted covariates and (b) prep the
legacy→**global** scale-up. Accepted as a **clean-cut** (results incomparable with the
viewser-era foundation; the frozen lodestar ruler is re-established on datafactory truth → v2).

## 1. Purpose
violet_visitor currently pulls conflict-only targets from **viewser** (PRIO PostgreSQL). The user
does not trust viewser beyond the conflict targets and wants population + trusted covariates, which
live in **views-datafactory**. This program swaps the data layer, **validates the datafactory data is
right before trusting it** (parity on a FRESH pull — never cached), re-establishes the foundation/
baseline on datafactory truth, and makes the ingestion views-frames-native. The magnitude wall
([[amount-ceiling]]) motivates the covariate: **population is the highest-information untried lever**.

## 2. Relationship to prior work / ADRs
- **Prior art absorbed:** memory `project_datafactory_migration.md` (the indicative cached-parquet
  parity + the LOCKED disciplines). Template: `models/light_strider/configs/config_queryset.py`
  (datafactory descriptor). Arch twins on datafactory: `blazing_meteor`, `bright_starship`
  (same `HydraBNUNet06_LSTM4`).
- **Complements:** ADR-060 (static_channels — the population seam), the views-frames migration
  (tasks #67–71, ADR-047 PF path). **Retires to v2:** the viewser-tied frozen lodestar ruler
  (`reports/2026-07-17_lodestar_eval_dossier`).
- **Exits to:** a proposed ADR for "violet_visitor data provider = views-datafactory" on `promote`.

## 3. Document index
| # | File | Status |
|---|------|--------|
| 00 | README (this) | living |
| 01 | literature | stub (migration ≈ engineering; thin lit) |
| 02 | design | DRAFT (swap seam + difference ledger + sequencing) |
| 03 | harness_and_invariants | DRAFT (crown jewel — invariants + pre-flight) |
| 04 | roadmap | DRAFT (P0…P5 gated) |
| 05 | analysis_plan (Tier-A parity) | **TODO — `/rnd-dossier preregister`** before the fresh-pull diff |
| 06 | glossary | stub (points to reports/GLOSSARY.md) |
| 07 | experiment_log | empty ledger |

## 4. Harness at a glance
Most of the standing harness EXISTS (grid-naming canonicalization, config validators, floor-md5
trap-restore, frozen lodestar scoring functions, the parity scripts, the views-frames PF path). The
**gaps to build** are program-specific: verify the **fresh-pull mechanism** in-env, swap violet's
queryset, re-point the parity harness at FRESH pulls, re-anchor the lodestar TRUTH to datafactory,
build the **covariate external-validation** harness, and wire `ln_pop` into static_channels. See `03`.

## 5. Current state & next actions (living)
- [x] init: scaffold + harness audit
- [x] **P0 GATE — fresh-pull capability**: `views-datafactory` imports in-env; FRESH remote reaches month **559** (≥504) ✓
- [x] Tier-A pre-registered (05 LOCKED); **Tier-A on FRESH pull = PASS** (07 E1) ✓
- [x] **S1 (#204)** — ADR-071 proposed (`docs/ADRs/proposed/071_…`) + ledger + 3 locked decisions ✓
- [x] **S2 (#205)** — `tools/tier_a_parity.py` re-runnable harness + 6 unit tests; reproduces E1 on fresh pull ✓
- [x] **S3 (#206)** — violet queryset swapped to datafactory descriptor; platform fetch + 2-lesson smoke green; Tier-A on platform-fetched data = PASS (07 E2) ✓
- [x] **S4 (#207)** — v2 truth frozen (sha256 `620f4aa…`) + `tools/v2_ruler.py` adapter + provenance + 4 tests ✓
- [ ] **HELD → S5 (#208)** — v2 floor 3-seed baseline (GPU) + Tier-B — **needs explicit go** (long batch)
- [ ] then population track (S6–S8) + frame-native (S9) + close-out (S10)

**Uncommitted (per instruction — no staging/commit without explicit go):** ADR-071 + README; dossier
docs + `tools/`; **views-models** `violet_visitor/configs/config_queryset.py` (viewser→datafactory —
NEVER commit views-models). viewser queryset backed up at job tmp `violet_config_queryset.viewser.bak.py`.

## 6. Conventions
Numbered dated docs; `00_README` living, rest point-in-time. git-tracked via `git add -f` (reports/
is gitignored). Vocabulary = `reports/GLOSSARY.md` (locked). On close → `reports/archived/`.
