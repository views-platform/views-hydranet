# 08 — S3 execution plan (roster reconfiguration + violet migration) — 2026-08-08

Ready-to-run plan for **S3 #246**: put the 8 dirs onto the LOCKED roster (05) on the v2 `gated_NB` foundation
(S1 `tools/foundation_gated_nb.py`), including the outstanding **violet datafactory migration**. Executes as one
views-models PR after your go.

## Member → dir mapping (proposed; the one decision to confirm)
| dir | family | output_distribution | forecast_composition | seed | notes |
|---|---|---|---|---|---|
| violet_visitor | gated_NB | nb | soft_gate | 42 | + **datafactory queryset** (still viewser); drop `EXPERIMENT_IN_PROGRESS`, re-pin |
| bright_starship | gated_NB | nb | soft_gate | 43 | africa datafactory ✓ |
| bold_comet | gated_NB | nb | soft_gate | 44 | africa datafactory ✓ |
| blazing_meteor | th_gated_NB | nb | threshold_gate (τ=0.5) | 45 | africa datafactory ✓ |
| heavy_freighter | th_gated_NB | nb | threshold_gate (τ=0.5) | 46 | **global `land` → `africa_me_legacy`** (queryset region + grid 360×720→180×180 off/87/310) |
| pink_pirate | mixture_NB | mixture_nb | soft_gate | 42 | africa datafactory ✓ (S2) |
| blue_stranger | mixture_NB | mixture_nb | soft_gate | 43 | africa datafactory ✓ (S2) |
| purple_alien | mixture_NB | mixture_nb | soft_gate | 44 | africa datafactory ✓ (S2) |

## Per-member config transform (mechanical, proven by the smoke)
Each `config_hyperparameters.py` ← the S1 foundation block (`tools/foundation_gated_nb.py`), overriding only
`(output_distribution, forecast_composition[, gate_threshold=0.5], torch_seed, np_seed)` per the row above, at
`total_lessons=300`, `n_head_samples=4`, `n_posterior_samples=4` (S=16). Generated via the proven
`scratchpad/smoke_mutate.py` transform (validated in EXP-00 for all 3 families; roundtrip-clean). Preserves each
dir's grid/region/data (except heavy_freighter's grid, changed deliberately).

## Two special cases
1. **violet_visitor** — the blocker. (a) queryset viewser→datafactory (`bright_starship` template, per-model
   docstring); (b) config → gated_NB foundation (seed 42); (c) since it becomes a *defined* roster member,
   remove `EXPERIMENT_IN_PROGRESS` and **re-pin** `test_datafactory_parity` to its settled value (resolves the
   C-71/C-87 deferral). This is the "re-pin when the roster lands" noted in the hygiene (S1.5).
2. **heavy_freighter** — global→africa **(TEMPORARY experiment-scoping)**: queryset `REGION land→africa_me_legacy`;
   grid to the africa block (row_offset 87, col_offset 310, 180×180). ⚠️ **PRESERVE its global config** — it is
   the **global proof-of-concept / "how to run global with the datafactory" template** (global+datafactory
   proven on the server; OOMs locally). Bank the global config (e.g. `tools/heavy_freighter_global.py`) before
   overwriting, so the post-S6 **global flip** (all 8 → `region="land"`, 360×720, run on the SERVER, SERVE) can
   restore it fleet-wide. This africa ensemble is the **last local experiment before global-on-server**
   (see 00 honest scope + memory `project_global_server_endgame`).

## Validation (before the 300-lesson run)
- Per-member `config_initializer` validation (K=4⇒family, gate_threshold iff threshold_gate, log1p) — the
  dry-validate already passed for all families.
- Per-member 2-lesson smoke (the `scratchpad/smoke_run.sh` harness) — OR rely on EXP-00 (which already trained
  all 3 families at 40 lessons); a fresh smoke on heavy_freighter-on-africa + violet-on-datafactory is the only
  genuinely new path.
- ADR-015 contract: set `expected_models`/`expected_samples_per_model` on the ensemble (S4) — the D×K-vs-
  `n_posterior_samples` wrinkle is reconciled at S4, not here.
- Full suite + ruff green; the parity-test premise refactor (VIEWSER/DATAFACTORY trios → the gated_NB roster)
  lands here since all 8 leave `tobit`.

## Execution
One views-models PR off `development` (isolated worktree): 8 `config_hyperparameters.py` + violet
`config_queryset.py` + violet `requirements.txt` + `test_datafactory_parity.py` re-pin. Full suite + smoke green
→ hold merge for the user. Then **S4** wires the 8-member `concat` ensemble + reconciles the sample-count
contract; **S5** is the 300-lesson GPU run.

## Open decision
Confirm the **member→dir mapping + seed assignment** above (or adjust). Everything else is mechanical.
