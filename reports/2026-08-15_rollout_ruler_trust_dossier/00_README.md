# 00 — Rollout-ruler trust dossier (living spine)

**Created:** 2026-08-15 · **Epic:** [#263](https://github.com/views-platform/views-hydranet/issues/263) ·
**Tracking:** [#272](https://github.com/views-platform/views-hydranet/issues/272) · **Status:** COMPLETE (2026-08-15)

---

## 1. Purpose

Make the **T>0 (multi-horizon rollout) skill ruler trustworthy**, then use it to settle one question:
*was `gated_NB`'s h36 "win" over climatology real, or a zero-driven artifact?*

This is risk-register **cluster 16** (11 entries: C-217, C-218, C-219, C-220, C-221, C-224, C-231, C-248,
C-252, C-253, C-254). Each names a distinct way a T>0 skill number can be false. The programme is
**consolidation + proof**, not a build — the ruler already exists across three dossiers.

## 2. Relationship to prior work

| Artifact | Relationship |
|---|---|
| `reports/2026-08-13_evaluation_pitfalls_and_metric_battery_DRAFT.md` | **Absorbed** into `02_design.md` (§1–3, §6) and `03_harness_and_invariants.md` (§4–5); §7 → `SCOPE.md` PARKED. Original retained with a supersede banner. |
| `reports/2026-07-17_lodestar_eval_dossier/` | The **frozen T=0 ruler**. `tools/lodestar_score.py` supplies every metric primitive. Never modified. |
| `reports/2026-07-25_t0_rollout_skill_dossier/` | The T>0 horizon ruler (`gather_all_horizons`, origin-block bootstrap). Where C-217 was first cleared by hand. |
| `reports/2026-07-29_v2_scoreboard_dossier/` | `_metric_row` (21 cols), `gw_stratified.py` (from closed #233), `activation_metrics.py` (from merged PR #261). The archived `results/score_*.csv` are S1's entire input. |
| `reports/2026-07-28_datafactory_migration_dossier/` | `v2_ruler.py` — the pinned truth (`V2_TRUTH_SHA256`). |
| **#262** | **Blocked by this dossier.** Its distribution-matching verdicts cannot be judged on the current ruler. |
| **#249** | **Blocked by S3** — needs the FAO-02 climatology baseline. |
| **#258** | The parent investigation. *"Every headline metric we had trusted for months was blind to it."* |
| ADR-048 | Governs the risk register this dossier corrects in S7. |

**This dossier does not become an ADR in this phase** (epic out-of-scope #8). Promote-vs-park is decided in S7.

## 3. Document index

| Doc | Role | Status |
|---|---|---|
| `00_README.md` | this spine | living |
| `01_literature.md` | the five method papers + what we take from each | S0 |
| `02_design.md` | why the ruler is untrustworthy + the three additions (absorbs DRAFT §1–3, §6) | S0 |
| `03_harness_and_invariants.md` | the 11 invariants → guard map; DRAFT §4 battery with a scope column | S0 |
| `04_roadmap.md` | the 8 stories, their gates and dependencies | S0 |
| `05_analysis_plan.md` | **PRE-REGISTERED, LOCKED before any number is read** | S0 |
| `06_glossary.md` | pointer to `reports/GLOSSARY.md` + the terms this programme introduces | S0 |
| `07_experiment_log.md` | append-only; the verdict lands here | S1 onward |
| `SCOPE.md` | **the railguard** — 21 numbered exclusions + the PARKED table | S0 |
| `tools/` | drivers (gitignored; skip-guarded tests) | S1+ |
| `results/` | outputs | S1+ |

## 4. Harness at a glance

The standing harness is unusually strong and is **reused, not rebuilt**:

| Guardrail | Where | Status |
|---|---|---|
| Frozen metric primitives + a freeze self-test | `lodestar_score.py::_selftest` | exists |
| h=1 byte-reproduces the frozen T=0 ruler | `tests/test_score_v2_horizons.py::test_h1_matches_frozen_lodestar` | exists |
| Identical cross-arm support at every horizon | `_support_keys` + intersection | exists |
| Ex-ante stratum, leakage-tested by outcome permutation | `gw_stratified.exante_stratum` | **C-248 CLOSED** |
| Origin-block bootstrap, asserted wider than iid | `_bootstrap_mean_ci(resample="origin")` | **C-253 CLOSED** |
| OOM guard (per-cell vectors, never cubes) | `del g` ×3 sites | **C-252 closed by construction** |
| Degenerate-forecast red-team | `tests/test_activation_metrics.py` (8 tests) | **DRAFT §5 already implemented** |
| Pinned truth hash | `v2_ruler.V2_TRUTH_SHA256` | exists |

**To build (the six genuine gaps):** a reference forecast · a skill score · the zero-share decomposition ·
a partition auditor · the Taillardat index · a driver. Everything else is imports.

**Known harness gap this dossier fixes:** `reports/` is gitignored, so both existing guard files
`pytest.skip(allow_module_level=True)` and CI has **zero coverage of the ruler**. Resolved by putting the pure
functions in `scripts/rollout_ruler_core.py` (tracked, CI-visible, never skips) and only the path-bound drivers
in `tools/`.

## 5. Current state & next actions

**Status: COMPLETE (2026-08-15). All 8 stories done; epic acceptance D1–D5 green.**

- [x] **S0 #264** — dossier, DRAFT absorbed, `05` LOCKED, `SCOPE.md` (22 exclusions + PARKED)
- [x] **S1 #265** — CRPS-gap decomposition; identity holds to 8.9e-16 on all 198 archived rows
- [x] **S2 #266** — provenance audit; 6/6 arms clean (**D3**)
- [x] **S3 #267** — a scorer-side FAO-02 climatology reference + skill score; **unblocks #249**
- [x] **S4 #268** — MDE 0.00596 at P=13; C-252 explicit; **C-277** registered and pinned (not fixed)
- [x] **S5 #269** — C-224 Taillardat diagnostic, 9 numbers, 3 railguards, inside the 109/118-of-120 cap
- [x] **S6 #270** — re-score (**D1**) + **the verdict: ARTIFACT** (**D5**)
- [x] **S7 #271** — register corrections, blind-metric table, disposition

### The verdict
**`gated_NB`'s h36 "win" over climatology is an ARTIFACT** — unanimous across 12/12 arm×target rows at
h=36, and it clears FAO-02's own 5% superiority bar on a statistically real gap while ranking conflict
*worse*. Short-horizon skill is genuine (7/12 REAL at h=1, 0 ARTIFACT). Full reasoning in `07`.

### Disposition — PARK, do not promote (decided 2026-08-15)
`02_design.md` is **not** promoted to a proposed ADR in this phase. Promotion normally follows a *validated*
design, and this ruler has been exercised on **one** substrate. The natural trigger is a second independent
use — **#249** is the obvious one, and it needs S3's climatology anyway. Recorded per S7's needs-decision.

### Next actions (outside this dossier)
- **#262** (rollout-aware / distribution-matching training) is now judgeable — it has a ruler.
- **#249** can proceed — the climatology baseline exists.
- **C-277** (`block_bootstrap_crps` support), **C-278** (FAO-02 outside the repo) and **C-279** (the
  climatology duplicates views-baseline's `ConflictologyModel`) are registered and open.
- **views-baseline #82** asks which climatology window convention is correct (fixed at `train_end` vs
  sliding per origin). The ruler defaults to the canonical fixed convention and offers both.
- ⚠️ **Nothing in this dossier is committed.** `reports/` is gitignored; the dossier and its `tools/` need
  `git add -f` when the maintainer chooses.

## 6. Conventions

Numbered docs, dated in-header; `00` is living, `07` is append-only, the rest revise. `git add -f` (the
`reports/` tree is gitignored). Negatives are first-class in `07`. On close the directory moves to
`reports/archived/`.

**Programme-specific:** every mid-flight idea goes to `SCOPE.md`'s PARKED table and is **not worked**. If a
story exceeds 1.5× its estimate: stop, log the partial in `07`, escalate. **If the epic's D1–D5 are green,
stop — fix nothing the ruler finds.**
