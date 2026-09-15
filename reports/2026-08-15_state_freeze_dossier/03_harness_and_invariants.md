# 03 — Harness and invariants

Every falsifier in `05_analysis_plan.md` is backed by a **test that was checked against a deliberate
sabotage**. A guard that has never been seen to fail is not a guard — this programme has twice produced a
confident verdict from a wrong implementation (meta-pattern 8), so the sabotage check is the discipline that
replaces trust.

## Falsifier → guard map

| # | Falsifier | Guard | Sabotage it was checked against |
|---|---|---|---|
| **F1** | h=1 differs across arms | `test_h1_is_byte_identical_across_every_mode` | blend patched into the **history-digestion** branch → red |
| **F2** | `none` does not reproduce EXP-SS-2 | read at run time (EXP-01) | — reproduced to 4 s.f. |
| **F3** | all arms track each other | read at run time (EXP-02) | — did not fire |
| (impl) | anchor captured a step late | `test_freezing_all_makes_the_state_stop_contributing_new_information` | anchor moved to the first free-running step → red |
| (impl) | blend applied at the seed step | same test | blend inserted at the seed forward → red |
| (impl) | two arms silently identical | `test_each_mode_produces_a_distinct_rollout_beyond_h1` | equal-rate mock made `hidden` ≡ `cell` → red |

**What F1 actually guards, established by sabotage.** h=1 is produced by the seed step's forward pass, which
reads the state built during digestion — so it is *immune* to a blend placed at or after that forward, and a
mis-placed blend shows up at h≥2 instead. F1 catches leakage into **digestion** specifically. Narrower than
"the freeze starts after the seed step" implies, and documented in the test rather than assumed.

## Structural invariants

| invariant | how it is held |
|---|---|
| the freeze is **not** a config key | an explicit `HydraNetInference` argument; `test_freeze_h_option_retired` stays green, ADR-027's retirement untouched |
| production is byte-identical when off | `freeze_recurrent=None` takes the pre-existing path unchanged |
| a typo cannot run the control | validated at construction against `FREEZE_RECURRENT_MODES`; an unknown mode raises |
| the state split is honest | `blend_recurrent_state` raises on a channel count not divisible by 8 — an uneven split would silently hold the wrong memory type |
| arms share one history | F1, above |

## Operational invariants

| risk | control |
|---|---|
| two arms' cubes mixed (the pipeline names the dir after the **artifact**, so every arm writes the same path) | the driver refuses to start when a prediction dir already exists |
| the wrong artifact scored (`truncated_smoke` carries two; the newer is the ε=0.1 SS arm) | `--artifact` is **required**, never defaulted, and existence-checked |
| disk exhaustion (the 37 GB scar) | ~2.5 GB/arm, score-then-delete, 25 GB preflight |
| a scoring failure costing a regeneration | score **then** delete — this fired once and saved a 26-minute re-run |
