# SS sweep — ready to launch, ~24 h unattended

**Parked 2026-08-17 for a ~26 h window.** Everything below is built, tested and gated. This file exists
so the sweep is a command, not a reconstruction.

## One command

```bash
cd /home/simon/Documents/scripts/views_platform/views-hydranet
setsid nohup bash reports/2026-08-17_ss_retention_dossier/tools/run_sweep.sh \
  > reports/2026-08-17_ss_retention_dossier/results/sweep_launcher.log 2>&1 < /dev/null &
```

Then confirm it started — the one check that catches a silent non-launch:

```bash
sleep 30 && tail -2 reports/2026-08-17_ss_retention_dossier/results/run.log
```

Read `results/VERDICT.md` when it finishes. Its first line is one of **EFFECT / NULL / UNDERPOWERED /
VOID**; read the state before the numbers.

## What runs

**7 arms.** 2 ε × 4 seeds = 8, minus the seed-42 ε=0 control, which is already trained, scored,
bootstrapped and gated (`longzero_fortytwo`, retention 0.54, FLOORGATE **PASS**).

| arm | ε | seed | ~h |
|---|--:|--:|--:|
| `longzero_fortythree` / `_fortyfour` / `_fortyfive` | 0.0 | 43/44/45 | 1.6 each |
| `longhalf_fortytwo` / `_fortythree` / `_fortyfour` / `_fortyfive` | 0.5 | 42–45 | 4.7 each |

≈ **24 h**. SS arms cost 3.32× because the sampler fires every step regardless of dose.

Resumable per arm: an arm whose `score_<label>.csv` exists is skipped, so a relaunch continues rather
than restarts. Cubes are deleted after scoring, so peak disk is ~2.5 GB.

## Preconditions (the driver asserts these; listed so a failure is legible)

- `FLOORGATE_longzero_fortytwo_PASS` exists **and** its threshold md5 is
  `6d5714d5ceda147ed16f53143abe7e37` — the value pinned in `05_analysis_plan.md`. A mismatch means
  someone relaxed a threshold after seeing a control, and the run must not proceed.
- No leftover `predictions_*` in any arm dir.
- ≥ 20 GB free.
- `diagnostic_visualizations: False` in every arm — violet's is `True`, and the per-origin biopsy costs
  ~28 min/origin, i.e. **~6 h per emit**. Asserted at arm-build time, before any GPU work.

## Facts worth having in hand when the numbers arrive

- **Training is bit-reproducible at fixed seed here.** Retraining violet at HEAD on 2026-08-17 gave
  **190 weight tensors with an identical sha256** and predictions matching to 15 decimal places. So the
  ε=0 controls are a fixed reference, and a difference between arms is a real difference.
- **Compare weight hashes, never `.pt` file hashes.** Those same two identical models have *different*
  file shas (torch stamps mtimes into the zip).
- **This cannot settle what the roster showed** (`05_analysis_plan.md` §3.1). The four SS-on roster
  models trained with `ss_feedback='mean'`, which C-259 now forbids. A null here answers the
  forward-looking question only.
- **The floor gate is re-run on the sweep's own ε=0 controls afterwards.** If they fail, the sweep is
  VOID whatever the treatment arms did.

## If something goes wrong

`results/run.log` is the narrative; `results/<arm>_run.log` is the raw pipeline output for one arm.
An arm that fails does not stop the sweep — it is logged and skipped, and `VERDICT.md` reports how many
arms actually completed. Fewer than 3 seeds per side means the exact test cannot reach p ≤ 0.05, so the
verdict will read **UNDERPOWERED** rather than pretending.
