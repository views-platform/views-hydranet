# HydraNet 8-member Africa `concat` ensemble — dossier (2026-08-08)

## Purpose
Operationalize the validated **`gated_NB` posterior** into a production **8-member, datafactory-consistent,
posterior-informed `concat` ensemble on `africa_me_legacy`** (13,110 cells), scored on the locked
Giacomini–White readout vs its members and vs `light_strider` climatology. This is the execution dossier for
**Epic #242** (stories #243–251); this doc set is its **S0 pre-registration** (#243).

**Honest scope (non-negotiable):** every current head shares the **ξ=0 magnitude ceiling** (v2 scoreboard;
Epic #230). This ensemble buys **variance reduction + occurrence calibration + a short/long-horizon hedge + a
small structural hedge** — **not** a magnitude fix, and it will not beat climatology past ~h18. The mixture
members are a **placeholder** the heavy-tail head (#241) will eventually upgrade. **ZINB is excluded** (blooms
in the free-running rollout).

## Relationship to prior work / ADRs
- **v2 scoreboard** (`reports/2026-07-29_v2_scoreboard_dossier`) — established **gated_NB ≡ th_gated_NB** as the
  ship candidate (beat datafactory climatology on short-horizon occurrence CRPS); **ZINB falsified + blooms**.
  This ensemble consolidates that win. *(prior art — linked, not duplicated)*
- **tail_decoupled_head** (`reports/2026-08-01_tail_decoupled_head_dossier`) — mixture-NB is **NULL on
  magnitude** (F4-clean); gated_NB ships, mixture is a structural hedge only. *(prior art)*
- **datafactory_migration** (`reports/2026-07-28_datafactory_migration_dossier`) — the **Tier-A parity
  protocol** (fresh-pull discipline) S2 reuses. *(prior art)*
- **2026-08-04 plumbing smoke** (this session) — proved the whole pipeline end-to-end at toy scale: 14 trains ×
  D×K=4×4=16 across gated_NB / th_gated / mixture families + datafactory + viewser, pooled 7→112 draws, all
  clean, nothing committed. De-risks S1→S5 mechanics.
- **Governs / becomes:** complements ADR-051 (ensemble architecture), ADR-067/069/070 (family/composition/
  feedback), ADR-071 (datafactory). A validated result graduates to a proposed ADR on `promote`.

## Document index
| # | Doc | Status |
|---|-----|--------|
| 00 | README (this spine) | living |
| 01 | literature | linked to prior dossiers (no new fetch) |
| 02 | design (the ensemble + why) | draft |
| 03 | harness_and_invariants | **audited** (crown jewel; ~80% pre-exists) |
| 04 | roadmap (S0–S7) | draft |
| 05 | analysis_plan (pre-registration) | **🔒 LOCKED (2026-08-08)** |
| 06 | glossary | pointer to `reports/GLOSSARY.md` |
| 07 | experiment_log | empty (append-only) |

## Harness at a glance (see 03)
**~80% already exists** (v2 GW scorer + frozen truth, PredictionFrameEnsembleManager concat + contract guards,
the transient smoke driver, reproducibility gate, fresh-pull discipline). **To build (C):** (a) the committed
`gated_NB` foundation config (S1), (b) the 3 viewser→datafactory migrations + Tier-A (S2), (c) the D×K-vs-
`n_posterior_samples` contract reconciliation (S4), (d) an 8-member ensemble that actually points at the real
dirs (S3/S4). **Pre-flight checklist in 03 must be green before S5 (the real 300-lesson run).**

## Current state & next actions
- [x] Dossier scaffolded (2026-08-08) · plumbing smoke passed (2026-08-04).
- [x] **NEEDS-DECISION resolved (2026-08-08):** roster = **3 gated / 2 th_gated / 3 mixture**; **`S` = 16 (4×4)**.
- [x] **05 LOCKED** (2026-08-08) — roster + region + combiner + `S` + scoring plan + F1–F4 + honest ξ=0 scope.
  **S0 acceptance criteria met** (pending git-tracking + epic link).
- [x] **git-tracked** (`git add -f`, commit `6203fd9`) + linked from epic #243.
- [x] **S1 #244 DONE** — `gated_NB` foundation config reconstructed + banked at
  `tools/foundation_gated_nb.py` (validated; already trained+emitted in EXP-00, so no new smoke). *Committing it
  onto `violet_visitor` + propagating at S3 is the user's action.*
- [x] **S2 #245 VALIDATED (2026-08-08):** fresh datafactory pull = **STABLE** (Tier-A PASS, 100% exact vs v2
  truth, EXP-01) + transitivity from violet's identical-queryset Tier-A. Migration recipe = copy violet's
  datafactory `config_queryset.py` verbatim to the 3 + add `views-datafactory>=1.9.0,<2.0.0` to each
  `requirements.txt`. *Applying + committing to views-models is user-gated (their fix/336 branch is mid-work).*
- [ ] **S3 #246 — NEXT (after S2 applied):** reconfigure the 8 members to the roster on the v2 foundation.
- [ ] S3/S4 → S5 (300-lesson run) → S6 (score) → S7 (disposition + promote).

## Conventions
Numbered dated docs; `00_README` living, rest point-in-time. git-tracked via `git add -f` (reports/ gitignored).
Follow `reports/GLOSSARY.md` (locked vocab). Scope-lock: ONLY the ADR-067 NB/ZINB/mixture family lineage; **no
population** (deferred). On close → `reports/archived/`.
