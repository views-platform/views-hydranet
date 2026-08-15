# 07 — Experiment log (append-only)

Every run and outcome, **including negatives and postmortems**. Each entry links its pre-registration
(`05_analysis_plan.md`) and states its verdict against the pre-committed falsifiers. No success-only drift.

---

## EXP-00 — build + guard verification (no arms run yet) · 2026-08-15

**Story:** the mechanism, before any experiment. **Code:**
`views_hydranet/utils/hydranet_inference.py::blend_recurrent_state` +
`HydraNetInference(freeze_recurrent=...)`; driver `tools/{run_freeze_arms,freeze_arm_entry}.py`.

### What was built

`freeze_recurrent ∈ {None, "hidden", "cell", "all"}`, an explicit argument on `HydraNetInference`, forwarded
by `InferenceOrchestrator`. Default `None` = today's behaviour, untouched. **Not a config key**, so no model
config can enable it and ADR-027's retirement of `freeze_h` stands;
`tests/test_inference_logic.py::test_freeze_h_option_retired` is green.

The hold is applied at `t > origin` only, against an anchor captured at the end of the seed step
(`t == origin`) — the last state built from real observations.

### Guards verified by sabotage, not by assertion

A guard that cannot fail is not a guard (EXP-09 of the ruler dossier). Three distinct leaks were introduced
deliberately and each was caught:

| sabotage | caught by |
|---|---|
| anchor captured one step late (first free-running step, not the seed) | `test_freezing_all_makes_the_state_stop_contributing_new_information` |
| blend applied at the seed step's forward | same test — h=1 is structurally immune, see below |
| blend leaking into the history-digestion branch (`t < origin`) | **`test_h1_is_byte_identical_across_every_mode`** |

**A correction to the plan's own wording.** It called the h=1 identity check "the load-bearing self-test" and
implied it catches any leak. It does not: h=1 is produced by the seed step's forward pass, which reads the
state built during digestion, so a blend placed at or after that forward cannot move it. What h=1 actually
guards is leakage into **digestion**. Narrower than advertised, still worth having, and now documented
accurately in the test itself.

### A mock that would have made the experiment vacuous

The first `_StateSensitiveModel` gave the two memory halves equal effective rates (short `+1.0/step`, long
`+0.1/step` behind a `×10` coefficient). Under it, `hidden` and `cell` produce **byte-identical** rollouts —
two of the four arms would have silently been the same arm. Caught by
`test_each_mode_produces_a_distinct_rollout_beyond_h1`, which exists for exactly this. Rates are now `+1.0`
vs `+0.5`.

### One pre-registration risk closed in the driver

`truncated_smoke` carries **two** calibration artifacts, and the more recent
(`calibration_model_20260814_061215.pt`) is the ε=0.1 scheduled-sampling arm — **not** the EXP-SS-2 artifact
F2 requires. Letting the pipeline default to the latest would have scored a different model and made the
reproduction control meaningless while looking fine. `--artifact` is therefore **required**, not defaulted,
in both driver scripts, and the file's existence is checked before the run starts.

### Status

**No arm has been run.** 1421 tests pass, ruff clean. Next: the `none` arm on `truncated_smoke`, timed, to
establish the per-arm cost before committing to the remaining seven runs.

---
