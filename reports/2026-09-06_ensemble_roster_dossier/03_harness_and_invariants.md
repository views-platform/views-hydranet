# 03 — Harness and invariants

What actually guarded this program, and — the part that matters more — **what did not**. Written
after the fact from the run logs and tool sources, so every "ran" below is evidenced, not intended.
The failure this section exists to prevent is a governance artifact asserting a guard the run never
executed (C-303, 13 occurrences).

## Gates that RAN, with their evidence

| gate | where | evidence it fired |
|---|---|---|
| **Override potency** — refuses to emit if the composition/clamp override did not reach inference | `tools/roster_arm_entry.py:58-82` | `POTENCY OK: composition=… threshold=… freeze=…` in all 20 `results/log_*.txt` |
| **Orientation (C-322)** — cube gate vs dump gate, refuses below 0.9 | `tools/plot_forecast_maps.py:resolve_orientation` | `corr=+1.0000` per target; wrong orientation `+0.0186` / `+0.0241` / `+0.1376` |
| **Land mask north-up** — centre of mass must sit in the expected quadrant | same file | `centre-of-mass row = 64.5 of 180` |
| **Dump instrument check** — refuses a dump with no `n_passes` key (pre-2026-09-03 pass-0 dumps are a different instrument) | `read_dump` | all eight dumps carry `n_passes = 4` |
| **Pooling alignment** — every member's `(time, unit)` must equal the first member's before `concat` | `pooled_draws` | passed for all 8; pooling is positional, so a mismatch would have drawn a map of nowhere |
| **Equal S across members** — the pool assumes equal weight | `render_target` | 8 × 16 = 128 |
| **Re-emit identity** — the body dump must not change the forecast | ad-hoc, M74 | `purple_alien` re-scored byte-identical to the 2026-09-07 CSV, and the cube was demonstrably rewritten first |
| **h1 invariance under the clamp** (pre-registered VOID condition) | `05_analysis_plan.md` | ΔAP@h1 = **exactly 0.0000** on all four clamp-on/off arm pairs (M71) |

## Gates that were named but did NOT run

Stated plainly because the dossier is otherwise the kind of document that implies they did.

- **`scripts/floor_gate.py`** (C-299, FG-A/FG-C on a control arm) — not invoked. No `floor_gate`
  reference exists in `tools/`.
- **Weight-hash distinctness** — listed as a VOID condition in `05_analysis_plan.md` ("any arm's
  weight hash equal to another's"). **Never computed.** The arms are distinct by construction
  (different model directories, emit-only, no retraining), so the risk this guards is low here —
  but it was pre-registered and it did not run.
- **`scripts/arm_identity_check.py` / `scripts/arm_postflight.py`** — not invoked.
- **K3's comparator** — `light_strider` produced no scores, so the third acceptance criterion was
  never evaluated (M73, views-models#445).

## Invariants this program deliberately changed

- `forecast_composition` and `freeze_recurrent` are overridden **in memory on the orchestrator's
  config, never on disk** (`roster_arm_entry.py:41-47`), because both are read in two places
  (`_emit_magnitude` and `_sample_feedback`) and the compositions differ *because* they feed back
  different fields.
- The body-mean dump is set as an orchestrator attribute, not a config key, by design
  (`hydranet_inference.py:314-318`): default `None` is the byte-identical production path.

## Invariants respected while changing them

- **No retraining.** All 20 sweep arms and all 8 re-emits are emit-only from existing artifacts.
- **Dump directory derived from the model name inside the driver**, never from the caller, so two
  arms cannot overwrite each other's `bodymean_origin*.npz` and leave a complete-looking result
  that is half one arm and half another.
- **Scales shared within a target, never across** (M74).

## Known weaknesses of this harness

1. **No seed replication anywhere in this dossier.** One artifact per configuration. C-119 puts
   training variance near 20%, so any comparison between adjacent arms is uninterpretable. Only
   effects that are multiples of that (M68, M70, M71) are read as findings.
2. **One origin** (`origin_12`, months 517–552) and **one region**. `origin_rank` is a parameter
   throughout, so this is cheap to widen and has not been.
3. **`size_ratio` reads 0.0 on every arm** — saturated, and therefore carrying no information in
   any table it appears in. Magnitude questions were answered with `crps_events`, `act_ratio` and
   the peak-cell reads instead.
4. **`results/` is 52 GB**, dominated by 21 retained prediction cubes at ~2.5 GB each. They are
   reproducible from the artifacts by re-emit and should not be kept. `results/` is gitignored and
   was deliberately **not** `git add -f`-ed, after a prior incident in which tracking live
   experiment output caused a branch switch to delete score CSVs and a running script.
5. **`results/score_SMOKE_violet_threshold_cell.csv`** is a leftover from a smoke run and is
   excluded by name in every analysis script — an exclusion-by-name is a weak guard.
