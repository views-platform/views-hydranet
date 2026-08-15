# SCOPE — the railguard

**Epic [#263](https://github.com/views-platform/views-hydranet/issues/263) · locked 2026-08-15.**

This file exists because this programme has an unusually high number of adjacent, genuinely interesting
problems, and touching any of them converts a 19-hour effort into an open-ended one. **Anything on the
out-of-scope list that gets touched is a scope breach, regardless of how good the reason is.**

---

## Out of scope

### The ruler's findings
1. **Fixing anything the ruler finds.** The deliverable is a *number and a verdict token*, not a remedy. If the
   h36 win is an artifact, the deliverable is the sentence "it is an artifact".
2. **Retraining or re-emitting anything. Zero GPU.** If an arm's cube is missing or unusable, that arm is
   **dropped from the registry**, not regenerated.
3. **Re-running `light_strider`** to reproduce the 0.960. Refused explicitly: it needs an authenticated fresh
   datafactory pull, which re-opens C-275 (data vintage) for zero inferential gain.
4. **Chasing byte-parity** with any 2026-07-30 number. The substrate changed (cubes deleted); state the caveat
   and move on.
5. **The #258 rollout collapse itself.** Diagnosed, has its own thread (#262). The ruler *measures* it; it does
   not fix it.
6. **Anything in `reports/2026-08-08_hydranet_ensemble_dossier`'s agenda.** We are borrowing its cubes, not its
   programme.

### Governance
7. **The FAO-02 amendment.** The DRAFT argues for reopening it; this dossier produces *evidence*. Deliverable
   ceiling: one paragraph in `02_design.md` titled **"The ask we are NOT making yet"** + one register entry.
   Do not open, amend, negotiate, or draft an amendment.
8. **Vendoring FAO-02 into the repo.** Its definition lives outside git (a memory file + a PDF in `~/brain`)
   and every in-repo citation — e.g. `scripts/proper_score_audit.py:31` — points at a path that does not exist
   in the tree. **Real problem. Register it in S7; do not fix it here.**
9. **Promotion to `views_hydranet/`, and any ADR** (proposed or otherwise). Deferred by decision; `scripts/` is
   the tracked staging area. A later promote is `git mv` + one import line.
10. **Scoring the validation partition.** C-217's clearance depends on the artifacts being calibration-trained;
    validation-partition arms (train 121–504) **would leak** across 457–504. A different, currently-blocked
    question.
11. **Data-vintage / fresh-pull verification (C-275).** The pinned `V2_TRUTH_SHA256` is sufficient here.

### The metric zoo
12. **Upgrading `gw_stratified` to a Newey–West / Driscoll–Kraay GW regression.** C-253 is closed by the
    origin-block bootstrap + its guard test. At P = 13 an HAC regression would be *less* honest, not more.
13. **Spatial verification** (FSS, SAL, MODE, Gini/entropy) — DRAFT §4.3. Not in cluster 16.
14. **PIT / active-cell calibration** — DRAFT §4.1, I3. FAO-02-rejected for selection; adding it now re-opens
    item 7 through the back door.
15. **twCRPS, SCRPS, LogScore, QS99** — anywhere, for any reason, **including "just as a diagnostic".**
16. **The full DRAFT §5 degenerate red-team as new code.** `tests/test_activation_metrics.py` already covers
    4 of the 6 degenerates (8 tests). Build **only** the two the new functions need (all-zero, climatology).
    Do not build the spiky / frequency-right / magnitude-right cases.
17. **All 36 horizons.** Seven are pre-registered in `05`. If you want 36, that is a different, pre-registered run.

### Engineering
18. **Refactoring the four scorer tools into one.** Tempting, wrong: the `h=1 → frozen lodestar` provenance
    chain **is** the ruler's credibility. That is chain-of-custody, not tidiness.
19. **Editing `lodestar_score.py` or `rollout_skill_score.py`** — one sanctioned exception below.
20. **C-222 / C-223** (oracle-gap interpretation, direct multi-horizon). Adjacent, not in the cluster.
21. **Cluster 17** (C-115/116/163/164/192/245/272/275/276 — deploy readiness). A separate, parallel epic.

### And
22. **The research article** (DRAFT §6). Parked verbatim in `02_design.md`, worked on never.

---

## The one sanctioned edit to pre-existing code

`reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py:34` — hardcoded absolute `_HN` →
`Path(__file__).resolve().parents[3]`.

Justified because: it matches the proven `gw_stratified.py:128` pattern (already pinned green by
`test_score_gw_v2_repo_root_resolves_frozen_primitives`), it **cannot change any number**, and
`test_h1_matches_frozen_lodestar` re-proves the frozen anchor immediately after.

> **The discipline is the count, not the diff size. The count is one.**

The second known defect — `rollout_skill_score.block_bootstrap_crps:216` computing single-arm support where
`score_horizons`/`score_horizons_v2` use the cross-arm intersection — is **pinned by
`xfail(strict=True)` + a register entry in S4, and NOT fixed.** Rationale: S6 uses `score_gw_v2` (which
computes support correctly), so the function is unused here. Fixing an unused function is creep; leaving it
silently broken is a trap; `xfail` + a register entry is the honest middle.

---

## Stopping conditions

**S5 / C-224 is DONE for this dossier when all four hold:**
1. `taillardat_index` returns finite `T_u` for gated_NB vs the FAO-02 climatology, target `sb`,
   h ∈ {1, 18, 36}, q ∈ {0.99, 0.995, 0.999} — **9 numbers, in one table.**
2. Tests #17–#20 green.
3. `07_experiment_log.md` contains **one paragraph**: the 9 numbers, the calibration caveat, and the literal
   sentence *"`diag_Tu` is not used in any decision rule in this dossier."*
4. C-224's register entry gains an update recording that a diagnostic now exists **and that the Tier-1
   governance ask (FAO-02 owner sign-off before magnitude/tail GPU spend) is UNCHANGED and still open.**

**Size cap: ≤ 120 lines implementation, ≤ 120 lines test. Exceeding the cap is a STOP condition, not a budget
overrun** — log the partial in `07` and escalate.

**Explicitly NOT part of C-224 here:** fitting a GPD to the *truth's* tail · estimating the DGP's ξ ·
threshold-selection methodology (Beirlant / Papastathopoulos-Tawn / Naveau) · the `extremeIndex` R package ·
bootstrap CIs on `T_u` · `T_u` for every arm × target × horizon · Murphy diagrams · Brehmer & Strokorb ·
reconciling with `scripts/tail_scorecard.py` · twCRPS of any kind.

**Epic-level:** if D1–D5 (see #263) are green, **stop. Fix nothing you found.**

**Per-story:** if a story exceeds **1.5× its estimate**, stop, log the partial in `07_experiment_log.md`,
and escalate. Do not push through.

---

## PARKED

Every idea that arrives mid-flight lands here and is **not worked**. If it is a real risk, it also goes to
`/register-risk`. Seeded from DRAFT §7.

| # | Idea | Why it is tempting | Which phase it would have eaten |
|---|---|---|---|
| P1 | Formal FAO-02 amendment reconciling the catalog with its locked choices (DRAFT §7) | The catalog is a strong case, and C-224 needs it | S5 → open-ended governance work |
| P2 | Run the full degenerate red-team as code and tabulate which metric separates which degenerate ("the paper's Figure 1", DRAFT §7) | It is the DRAFT's best idea, and `test_activation_metrics.py` is 2/3 of the way there | S3/S5 → a metric-suite project |
| P3 | Fold in the gate-recall vs body-zero probe (#258) once it lands (DRAFT §7) | It empirically instantiates catalog E1 | S6 → waiting on another thread |
| P4 | Pull real numbers/tables from the seven named dossiers (DRAFT §7) | The DRAFT's citations are partly from memory | S0 → a `/verify-sources` job |
| P5 | Decide venue framing, methods vs applied (DRAFT §7) | The material is nearly there | never — out-of-scope #22 |
| P6 | "Honest scars" section — every retracted register claim as failure-mode data (DRAFT §7) | Genuinely the strongest part of the story | out-of-scope #22 |
| P7 | Test `scripts/tail_scorecard.py` (tracked, C-224-labelled, **zero tests**) | S5 gives it a tested neighbour, making the gap obvious | S5 → out-of-scope #16 territory |
| P8 | Pin the inventory of `allow_module_level=True` skip guards so a new silent skip cannot appear unnoticed | Directly serves epic D2; belongs in `tests/test_falsify_zero_surprises.py` | **stretch — drop first if time is tight** |

*(Append as they arrive. An entry here is a decision to defer, not a promise to do.)*

### Disposition at close-out (S7, 2026-08-15)

Nothing was worked from this table during the epic — the railguard held. Homes assigned:

| # | Home |
|---|---|
| P1 (FAO-02 amendment) | **Register C-278** (FAO-02 lives outside the repo) + C-224's standing Tier-1 governance ask. Both open. |
| P2 (full degenerate red-team as code) | Partially delivered: `tests/test_activation_metrics.py` (4/6) + S3's all-zero case. The remaining two stay parked. |
| P3 (gate-recall vs body-zero probe) | Belongs to **#258**'s thread, not this dossier. |
| P4 (verify the DRAFT's 40 citations) | A `/verify-sources` job. One citation was corrected (`C-1830` → C-231); the rest stay unverified and are flagged as such in `02_design.md`. |
| P5, P6 (venue framing, honest-scars section) | Out-of-scope #22 (the research article). Parked verbatim in `02_design.md`. |
| P7 (test `scripts/tail_scorecard.py`) | Still untested. Worth a small follow-up issue; not opened here (out-of-scope #16 territory). |
| P8 (pin the `allow_module_level` skip inventory) | **Dropped as planned** — it was marked stretch, and the CI-visibility goal was met another way (pure functions in tracked `scripts/`). |
