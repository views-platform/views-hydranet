# Harness audit — why runs kept dying, and what was actually wrong

**Date:** 2026-08-19 · triggered by the maintainer after 24 h in which every status update carried
"…and there was also a bug". **Status:** findings recorded; fixes and a 40-lesson smoke follow.

## 1. The pattern, stated plainly

Six arms were launched in 24 h. **Three completed.** The failures were not one problem:

| # | arm | died of | class |
|--:|---|---|---|
| 1 | L=300 attempt 1 | `CUDA error: unspecified launch failure` | **external** — machine physically moved |
| 2 | L=600 attempt 1 | SIGKILL, 7 h timeout, lesson 459/600 | **my judgement** — deadline known to be marginal, not acted on |
| 3 | L=300 attempt 2 | SIGKILL, 6 h timeout, 19 min into the emit | **my estimate** — timeout sized on a point estimate |
| 4 | *(the queue)* | `ABORT — 10G free`, idle 6.7 h | **my design** — a guard that aborts where it should wait |

Only #1 was outside my control. And the deeper issue is not any single defect: **the harness grew by
accretion under time pressure** — `run_curve.sh` → `run_followup.sh` → `run_followup2` → `3` → `4`, each
written while an arm was running, each fixing the previous one's blind spot. Bugs were found by *running
into them*, not by reading. That is the process failure the rest of this document exists to correct.

## 2. Defects found by reading (this audit)

Ordered by what they could cost.

### A · `--resume` fires on ANY artifact, and the artifact cannot prove its own training length
`run_lesson_arm.sh:70` — `PRIOR=$(ls -t "$A"/artifacts/*.pt | head -1)` and, if anything is there,
training is skipped. The artifact's sidecar (`*.pt.config.json`) carries `torch_seed`,
`output_distribution`, architecture — **but not `total_lessons`**. Meanwhile `arm_<label>.json` records
`total_lessons` **from the config**, not from the artifact.

So: edit an arm's `total_lessons`, re-run, and it silently emits the *old* model while labelling it with
the *new* length. A wrong number with a confident provenance record. This is the same class as the
2026-08-14 label/config mismatch the harness was built to prevent, reintroduced by a convenience path I
added at 08:37 today under pressure.
**Fix: resume must be explicit (`--resume`) and must refuse when the config is newer than the artifact.**

### B · Process detection by `pgrep -f` on generic strings
`run_followup4.sh:30-32` waits on `pgrep -f "main.py -r calibration"`. **My own diagnostic shell commands
match that string** — confirmed at 08:34, when a status check reported "STILL RUNNING" for a `/bin/bash -c`
wrapper. A scheduler can therefore stall indefinitely on a process that is not an arm. Also
`pgrep -f "run_arm.sh"` matches `run_lesson_arm.sh` by substring.
**Fix: one sequential scheduler holds a lock; no string matching.**

### C · Unchecked exit codes, then the evidence is deleted
`run_lesson_arm.sh:101-108` runs two bootstraps and `:112` writes the provenance JSON — **none of the
three exit codes is checked** — and `:138` then does `rm -rf "$P"`. If a bootstrap fails, the cube is
gone, `ap_ci_*.json` is missing, the inline gate crashes inside a heredoc whose traceback lands in
`run.log`, and the arm is still marked `ARM_DONE`. Silent, and unrecoverable without a re-run.
**Fix: check each, and do not delete the cube until every artefact that needs it exists.**

### D · An arm with a config record but no score is dropped without a word
`verify_curve.py:60` — `if not label or not score.exists(): continue`. A half-completed arm vanishes from
the verdict silently. Everything else in this harness reports what it skipped.
**Fix: append a note.**

### E · `verify_curve` crashes the whole verdict on one missing CSV column
`verify_curve.py:88-96` builds the ship panel with direct `rows[h]["crps_all"]` indexing. A score CSV
from an older scorer lacking one column raises `KeyError`, the outer handler turns it into
`# VOID — verify_curve itself crashed`, and a valid result is replaced by nothing.
**Fix: read those columns defensively; the ship panel is a bonus, not a gate.**

### F · An existing arm directory is reused without checking it matches
`run_followup4.sh:56` — `[ -d "$MODELS/$label" ] || build`. If a directory with that name exists from an
earlier, different configuration, it is used as-is. Same family as A.
**Fix: verify the resolved config matches the requested (lessons, seed) before reuse.**

### G · The oracle timeout is a flat 5400 s and does not scale
`run_lesson_arm.sh:162`. Emit cost does not depend on lesson count, but it *does* depend on machine
state — the same emit took 6 min cool and 24 min throttled. 90 min is probably enough and has never
fired; it is listed because it is the one remaining timeout sized on a cool-box measurement.

### H · Double logging
`log()` writes to stderr *and* tees to `run.log`; callers redirect `2>&1` into the same file, so every
line appears twice. Cosmetic, but it made the incident logs harder to read at exactly the wrong moment.

## 3. What was NOT wrong

Worth stating, because the failure list makes it look worse than it is:

* **The science harness held.** Every guard that exists did its job: the floor gate correctly failed a
  2-lesson arm and a 40-lesson arm; the symmetric-difference assertion never let a mis-specified arm be
  built; F1 held byte-exactly; `N = 170430` never drifted; the disk guard refused rather than filling the
  disk; the leftover-cube check refused rather than mixing two arms' cubes.
* **No result was corrupted.** Every failure was a *stop*, not a wrong number. The five completed
  160-lesson arms are clean and their verdict is unaffected.
* **The pipeline itself never failed.** `main.py` produced exactly what it was asked to, every time.

The defects above are all in the **scheduling layer I wrote**, not in the measurement.

## 4. Disposition

1. Fix A–F. G and H are noted; H is cosmetic and G has never fired.
2. **Consolidate the four scheduler scripts into one.** The accretion is itself the root cause; a single
   sequential queue removes the need for B entirely.
3. **A 40-lesson smoke through the full seam before any long run** — build, train, emit, score, both
   bootstraps, provenance, cube deletion, oracle, gate, sentinel — on a fresh arm, into a scratch
   results directory so it cannot contaminate the curve.

## 5. `/code-review medium` — 14 findings, 6 of which I had missed

Run against the same surface after my own pass. It found everything in §2 independently, plus six I
had not seen. Two are worth naming as *classes*, not items:

**It caught a defect I introduced an hour earlier.** After fixing the multiplier to track `n`
(AMENDMENT 3), `verify_curve` still computed the "a null is declarable" annotation from the hard-coded
`K_PRED` while the number printed beside it used `k_used`. At n=5–6 those differ (2.335 vs 2.631), so
`VERDICT.md` could have printed **"← too wide for a null" directly under a `# PLATEAU` heading**. A fix
that leaves an inconsistent copy of the old value behind is not a fix.

**It found the stale-state read.** `verify_curve`'s crash handler wrote `VERDICT.md` but left
`curve_state.json` holding the *previous* stage's verdict, and exited **0**. A driver reading that file
after a crashed verifier would act on a state that no longer existed — concretely, a stale `RISING`
sending it into a 900-lesson arm. Both files are now written together and the exit code is non-zero.

Full list, with disposition:

| # | finding | fixed |
|--:|---|---|
| 1 | bootstrap exit codes unchecked, then the cube is deleted | ✅ cube kept, arm exits non-zero |
| 2 | null-declarability annotation used `K_PRED`, verdict used `k_used` | ✅ |
| 3 | crash handler left `curve_state.json` stale and exited 0 | ✅ both files, exit 1 |
| 4 | resume records HEAD at **emit** time, blinding F6 to old weights | ✅ `--train-head` required, else provenance null |
| 5 | `ARM_DONE` unconditional — a failed oracle still read as complete | ✅ `ARM_INCOMPLETE_*` naming what is missing |
| 6 | `--gate` exit status discarded; empty "GATE written:" read as success | ✅ |
| 7 | measurement floor compared against `anchor[0]`, an arbitrary seed | ✅ selected by `REF_SEED` explicitly, unevaluable if absent |
| 8 | `mde_f` truthiness dropped 0.0; missing MDE gave a **false explanation** | ✅ `is not None`, and the detail names the real cause |
| 9 | F6 silently exempted arms with no fingerprint | ✅ flagged |
| 10 | unresolvable SHA gave an identical sentinel for every arm → F6 passes | ✅ returns None |
| 11 | `verdict_state` empty → stage 4 ran on an unknown verdict | ✅ scheduler retired |
| 12 | `wait_idle` blind to a sibling driver; per-script locks did not serialise | ✅ scheduler retired |
| 13 | `FOLLOWUP_COMPLETE` fired even when an arm was never attempted | ✅ scheduler retired |
| 14 | verdict encoded in the FLOORGATE **filename**; PASS and FAIL could coexist | ✅ stale tokens removed first |
| low | `df` unparseable → guard skipped rather than tripped | ✅ |
| low | double logging (`log` tees *and* writes to stderr) | pending — cannot edit a running script |

11–13 are gone because the four accreted schedulers (`run_curve.sh`, `run_followup{,2,3,4}.sh`) were
replaced by a single sequential **`run_queue.sh`**. It holds one lock, runs arms in order, and contains
**no `pgrep`** — the process-detection class cannot recur because nothing detects processes any more.

## 6. An honest footnote

While dry-testing the new queue's lock, I wrote a test whose second stage released the lock before the
command ran — so it **started a real 40-lesson arm**, which a 2-minute timeout then killed. Nothing was
lost (no artifact, no cube, directory removed) and it was caught within the minute, but it is precisely
the accident class this audit exists to close, committed by the person writing the audit. Recorded
because a process that only catches other people's mistakes is not a process.

## 7. Finding I — a long run's output must not live in the session scratchpad

The first 40-lesson smoke was launched with `RES_DIR` pointing at the assistant session's scratchpad
under `/tmp/claude-1000/<session-id>/scratchpad/`. That directory is **session-scoped**: when the
session ended it was wiped and recreated empty at 11:02, taking the live run's results directory —
`run.log`, the score CSV it was about to write, the sentinel — with it while the arm was still running.
The run died and left nothing to inspect.

`setsid` was doing its job; the process was detached and would have survived. What did not survive was
the place it was writing to. Detaching the process and then pointing it at ephemeral storage buys
nothing.

**Rule: `RES_DIR` must be inside the repository.** The smoke now writes to
`reports/2026-08-18_lesson_curve_dossier/smoke/`, which is durable, is not read by `verify_curve.py`
(that reads `results/` only), and therefore still cannot contaminate the curve. The scratchpad remains
fine for throwaway diagnostics; it is not fine for anything a run needs to still exist afterwards.

This is the same family as the rest of §2: not a wrong number, a **lost** one.

## 8. Status

| item | state |
|---|---|
| A–F, plus review findings 1–10 and 14 | fixed |
| 11–13 | dissolved — the four schedulers replaced by one `run_queue.sh` |
| G (oracle timeout) | fixed — 90 min → 4 h |
| H (double logging) | **outstanding** — cannot edit `run_lesson_arm.sh` while it is executing |
| I (scratchpad) | fixed — smoke writes into the repo |
| regression tests | 24 green, 3 new, sabotage-verified |
| 40-lesson smoke | **running** (2nd attempt), `smoke/run.log` |

The queue reused the existing arm directory on relaunch **after verifying its resolved config reads
`40:47:0.0`** — which exercises the finding-F fix on a real directory rather than a fabricated one.

## 9. The 40-lesson smoke — **PASS**, full seam, 2026-08-19 11:03–11:45

`run_queue.sh 40:47` into `smoke/`, a fresh seed on the new single scheduler.

| step | result |
|---|---|
| arm reuse | **reused** the existing dir after verifying its resolved config reads `40:47:0.0` (finding F) |
| train + emit | `rc=0`, 26 min, cube with **13 origins** |
| score | 22 columns, `N = 170430`, n_event 1343 / 1547 |
| AP bootstrap | `rc=0` — h1 mde 0.01570, h18 mde 0.00071 |
| retention bootstrap | `rc=0` — ratio **0.029431**, exactly `AP18/AP1` |
| exit codes | logged as `(AP rc=0, retention rc=0)` — finding C |
| cube deleted | **only after** all three artefacts existed — finding C |
| provenance | `resumed_from_artifact: false`, `head` and `emit_head` both recorded — finding 4 |
| oracle | `rc=0`, 12 min, under the new 4 h cap — finding G |
| **F1** | control h1 vs oracle h1, **|Δ| = 0.000e+00** |
| floor gate | **FAIL**, and correctly |
| sentinel | `ARM_DONE` (not `ARM_INCOMPLETE`) — control, oracle and gate all present |
| queue sentinel | `completed: shortzero_fortyseven · failed: none` |
| contamination | **0 mentions** of the smoke arm in the real `results/VERDICT.md` |

```
FG-A [FAIL] AP 0.00901 / prevalence 0.009077 = 0.99x  (need >= 5.0x)
FG-B [FAIL] AP(h1) 0.30613 / clim 0.29798 = 1.027x
FG-C [PASS] a 30% effect is 0.00631 AP; 3x MDE is 0.00212
```

**A 40-lesson model lands at 0.99× chance — indistinguishable from random ranking at h18.** The gate now
separates four rungs measured on identical machinery: **0.99× · 1.17× · 2.16× (FAIL) vs 28.30× (PASS)**.

Incidental, and relevant to the curve's scope: seed 47 at 40 lessons retains **0.0294** against seed 42's
**0.068** — a 2.3× spread at the same length. Seed variance is large at 40 lessons too, which is a
further reason that rung was excluded rather than merely deprioritised.

**Finding H is now fixed as well** (it could not be touched while the script was executing): `log()`
writes to `run.log` exactly once and to the terminal only when stderr is a TTY. Proven with a
redirect-emulating test — one line in, one line out.

## 10. Verdict

The harness is ready for long runs. Every defect in §2 and every actionable finding in §5 is fixed or
dissolved, the scheduler is one file with no process detection in it, and the full seam has been walked
end to end on a fresh arm with every guard firing in the right direction — including the one that was
supposed to fail.

