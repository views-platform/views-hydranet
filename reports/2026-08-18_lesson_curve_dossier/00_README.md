# Lesson curve — is 160 lessons on the plateau or still on the slope?

**Opened 2026-08-18.** Status: **pre-registered, harness built, not yet run.**

## The question

`RESULTS_LEDGER.md` §TRAINING LENGTH: *"Whether 160 is on the plateau or still on the slope is unknown — and if it
is still climbing, every experiment at 160 (including the parked SS sweep) measures a partially-trained
model, and a null there may only mean 'this does not help a model that has not finished learning.'"*

Every HydraNet in the tree trains for 160 lessons or 40. 600 has never been run here. The one measured
step of the ladder moved retention 0.068 → 0.541, and the oracle says 1.010 is achievable. So: does the
next step keep paying?

## Documents

| file | what it is |
|---|---|
| `05_analysis_plan.md` | **LOCKED** pre-registration — endpoints, θ = 0.14, the prediction bound, four decision states, F1–F7 |
| `SCOPE.md` | what this does **not** establish, written before the run |
| `07_experiment_log.md` | outcomes, falsifier verdicts recorded before predictions |
| `tools/run_lesson_arm.sh` | one arm: clone → train → emit → score → bootstrap → gate → oracle → delete cube |
| `tools/run_curve.sh` | the staged driver; every prefix of it is a complete result |
| `tools/verify_curve.py` | reads the arms, renders `results/VERDICT.md` — thin by design |
| `LAUNCH.md` | the one command, what runs, and what to have in hand when it lands |

The **decision rule itself** lives in `scripts/lesson_curve_gate.py` — tracked, unit-tested,
with a pinned `rule_md5`. A tracked test may not load the gitignored `reports/` tree, so a rule
that lived only in a dossier would be a rule with no test in CI.


## Relationship to other dossiers

* **Gates** `reports/2026-08-17_ss_retention_dossier/` — that sweep runs at 160 lessons, and its own
  `LAUNCH.md` carries the caveat this dossier exists to resolve. Its three ε=0 seed arms are run here.
* **Uses unchanged**: `scripts/floor_gate.py`, `scripts/ap_block_bootstrap.py`,
  `2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py`,
  `2026-08-16_feedback_realism_dossier/tools/run_realism_arms.py`.
* **Extends**: `2026-08-17_ss_retention_dossier/tools/make_ss_arm.py` (`_LESSON_WORD` += 600/900, additive
  only — the SS sweep uses 160 and is unaffected).

## Status

- [x] Pre-registration LOCKED
- [x] `_LESSON_WORD` extended, labels verified against `ModelPathManager.validate_model_name`
- [x] `ap_ratio_origin_block_ci` added — paired origin bootstrap for the co-primary endpoint;
      refactor verified byte-identical to the archived `stage_a_ap_ci.json` on the real 40L cube
- [x] Decision rule extracted to `scripts/lesson_curve_gate.py`, 21 tests, sabotage-verified,
      `rule_md5 5d6a256bb2b41485220d033cd0bfbc87` pinned in the pre-registration
- [x] `verify_curve.py` reproduces the anchor from live data (R = 0.5415, FG-A PASS, UNDERPOWERED)
- [x] Harness dry-run on a 2-lesson arm — full seam PASS; F1 byte-exact (0.000e+00);
      floor gate correctly FAILED it at 1.17x chance; arm deleted afterwards
- [ ] ⛔ **BLOCKED: 24 GB free, the driver needs 25.** See `LAUNCH.md` §Disk
- [ ] Stage 1 — σ_seed at L=160 (gate G1)
- [ ] Stage 2 — L=300
- [ ] Stage 3 — L=600
- [ ] Stage 4 — branch on stage 3
