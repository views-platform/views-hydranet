# Lesson curve — ready to launch, ~10 h to a verdict, ~16 h with the branch

**Pre-registration LOCKED 2026-08-18** (`05_analysis_plan.md`). This file exists so the run is a
command, not a reconstruction.

## One command

```bash
cd /home/simon/Documents/scripts/views_platform/views-hydranet
setsid nohup env BUDGET_HOURS=26 \
  bash reports/2026-08-18_lesson_curve_dossier/tools/run_curve.sh \
  > reports/2026-08-18_lesson_curve_dossier/results/curve_launcher.log 2>&1 < /dev/null &
```

Then the one check that catches a silent non-launch:

```bash
sleep 60 && tail -3 reports/2026-08-18_lesson_curve_dossier/results/run.log
```

Read `results/VERDICT.md` when it finishes. Its first line is one of **RISING / PLATEAU / UNDERPOWERED /
G1-STOP / VOID**. **Read the state before the numbers.**

`setsid` is not optional — plain background jobs get reaped when the assistant idles; setsid daemons
survive.

## What runs, in this order, and why this order

| stage | arms | ~h | why here |
|---|---|--:|---|
| **1a** | oracle (`use_real`) on `longzero_fortytwo` | 0.15 | completes the L=160 anchor — its control, CI and gate already exist |
| **1b** | `longzero_forty{three,four,five}` — L=160, ε=0 | 3.3 | **σ_seed**. Without it a one-seed lesson point cannot be read at all |
| **gate G1** | — | — | if `2.631 × σ_seed(R) ≥ 0.30`, stop. Seed noise would swamp the whole 0.4687 gap, and *that* is the finding |
| **2** | `fullzero_fortytwo` — L=300 + oracle | 2.1 | the production setting, never measured on this ruler |
| **3** | `sixhundredzero_fortytwo` — L=600 + oracle | 4.2 | never run in this repo, at any time |
| **4** | RISING → L=900 · otherwise → 2 more seeds at L=300 | 6.4 / 3.8 | pre-registered branch; skipped below 8 h remaining |

Stages 1–3 ≈ **9.8 h**. Every prefix is a complete result: the noise floor first, then each lesson point
complete with its own ceiling before the next begins.

## ⛔ Disk — the run will abort on its first assert as things stand

Measured 2026-08-18 04:30: **24 GB free**, and both `run_curve.sh` and `run_realism_arms.py` refuse
below **25 GB**. Reclaim before launching. Nothing below is a result:

| GB | what | why it is safe |
|--:|---|---|
| 33 | `/tmp/claude-1000/-home-simon-…-views-models/4bc3abf9-…/` | a scratchpad from a session dated **2026-08-10**, a different project |
| 12 | `views-models/models/violet_visitor/logs` | dead DEBUG logs; the pipeline writes fresh ones |
| 2.5 | `violet_visitor/data/generated/predictions_*` | leftover cube — it also **blocks** any realism arm on violet, which refuses when one is present |
| 2.5 | `shortzero_fortytwo/data/generated/predictions_*` | leftover 40-lesson cube. It was the fixture for the `ap_block_bootstrap` regression (see `07_experiment_log.md`); that check is done and recorded, so it is no longer needed |

```bash
du -sh /tmp/claude-1000/-home-simon-Documents-scripts-views-platform-views-models/4bc3abf9-*/        /home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/logs
# then remove whichever of the four you are happy to lose, and re-check:
df -BG /home/simon | tail -1
```

Left as a decision rather than done automatically: deleting is not reversible, and three of the four
live inside model directories.

## Preconditions the driver asserts (listed so a failure is legible)

- **Repo HEAD pinned and re-asserted before every arm.** A moved HEAD aborts rather than skips — arms
  across two HEADs are not comparable (F6).
- **≥ 25 GB free**, re-checked per arm. `run_realism_arms.py` refuses below 25 and each arm holds a
  ~2.5 GB cube transiently.
- **No leftover `predictions_*`** in an arm directory — two arms' cubes must never mix.
- **`diagnostic_visualizations: False`** in every arm, asserted at build time by `make_ss_arm.py`.
  Violet's is `True` and the per-origin biopsy costs ~28 min/origin ≈ **6 h per emit**.
- **The symmetric-difference assertion** — an arm whose resolved config differs from the floor in
  anything beyond `total_lessons` / seed is **not built** (F5).

## Facts worth having in hand when the numbers arrive

- **Training is bit-reproducible at fixed seed on this box** (M22: 190 weight tensors, identical
  sha256). So the ε=0 controls are a fixed reference and a difference between arms is a real difference.
  **Compare weight hashes, never `.pt` file shas** — torch stamps mtimes.
- **The anchor, on disk:** `longzero_fortytwo` C=0.4745, F=0.2569, **R=0.5415**, FG-A 28.30× PASS.
  Its ceiling: oracle h18 **0.4793**, i.e. retention **1.0101** — the rollout discards 46% of what the
  same weights achieve when fed the truth.
- **The prior is PLATEAU.** The v2 scoreboard's 3 seeds × 300 lessons give R = 0.4859 / 0.5505 / 0.5259
  (mean 0.5208, σ ≈ 0.0326) against 0.5415 at 160. Different snapshot and family, so not a substitute —
  but it is why the null had to be made *declarable* rather than assumed, and it is stated in the
  pre-registration so it cannot be retrofitted either way.
- **If σ_seed(R) > 0.0532, PLATEAU is unreachable by construction** and only a large rise is detectable.
  Pre-registered in §5, not discovered afterwards.
- **This consumes 3 arms of the parked SS sweep** — by design, not by accident. See `SCOPE.md`.

## Costs are extrapolated from two points

Measured: 40L = 18.9 min, 160L = 60.7 min total (`Done. Runtime` in the pipeline log). Per-lesson cost
**rose 22%** across that span, so the table above assumes it keeps rising. Two points cannot separate a
fixed overhead from a superlinear term. `TRAIN_TIMEOUT = max(21600, 36·L + 3600)` s sizes for the
pessimistic reading — 7 h at L=600, 10 h at L=900 — because there is **no mid-training checkpoint**
(`train_model.py:73` saves once, at the end) and a killed arm is lost entirely, not resumed.

## If something goes wrong

`results/run.log` is the narrative; `results/<arm>_run.log` is the raw pipeline output for one arm. An
arm that fails is logged; `verify_curve.py` reports how many points actually completed and will say
**UNDERPOWERED** rather than pretending. A `TIMEOUT` is logged by that name — the cost estimate was
wrong, not the arm. The run is resumable at arm granularity: relaunch the same command and any arm with
a score CSV is skipped.
