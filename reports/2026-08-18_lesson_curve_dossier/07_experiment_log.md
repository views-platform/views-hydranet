# Experiment log — the lesson curve

Append-only. Falsifier verdicts recorded **before** predictions are read.

---

### Harness verification · 2026-08-18 · **before any GPU**

Recorded here because these are measurements, not preparation, and one of them corrected a claim I had
written into the module docstring.

**1. The `ap_block_bootstrap.py` refactor is behaviour-preserving on real data.** `ap_origin_block_ci`
was changed to share its load/index step with the new ratio bootstrap. Re-run on the surviving 40-lesson
cube it reproduces the archived `2026-08-17_ss_retention_dossier/results/stage_a_ap_ci.json` **exactly**,
to every printed digit:

```
conda run -n views-hydranet-env python scripts/ap_block_bootstrap.py \
  --pred-dir .../shortzero_fortytwo/data/generated/predictions_calibration_20260817_153035 \
  --target sb --horizons 1,18 --n-boot 200 --seed 0
→ h1  ap 0.2888726240524795  lo 0.27667553726250277  hi 0.3055769409096746  mde 0.014450701823585904
  h18 ap 0.019622288747132736 lo 0.017583138589246157 hi 0.02253057794149111 mde 0.0024737196761224767
```

**2. The paired ratio bootstrap works, and it is not narrower than propagation — it is wider.** Same cube,
`--ratio --num-h 18 --den-h 1 --n-boot 200 --seed 0`:

```
ratio 0.06792713159128556   lo 0.059046465093222   hi 0.07984432192078814   mde 0.010398928413783065
ap_num 0.019622288747132736  ap_den 0.2888726240524795   n_origins 13
```

The point estimate is exactly `AP(h18)/AP(h1)`. **I had written in the docstring that independent
propagation "overstates the width". Measured, it does the opposite here** — naive propagation gives
mde ≈ 0.0092 against the paired 0.0104, ~13% *narrower*. The docstring was corrected to say the bias has
no predictable sign and that propagation is simply not a valid construction when the two horizons share
origins. Recorded because the claim was wrong before it was checked, and this is where that gets said.

**3. The pairing property is sabotage-verified.** Injecting the unpaired bug (a second, independent
origin draw for the denominator) turns `test_both_horizons_see_the_SAME_resampled_cells_every_replicate`
red; the `-k` filter was confirmed to select 1 test and deselect 7, so it genuinely ran. This is the
2026-08-17 scar — a filter that matched no tests and reported zero failures — being explicitly not
repeated.

**4. A repo gate caught the decision rule in the wrong place.** The first version put the rule in
`tools/verify_curve.py` and tested it from `tests/`. `test_P5b_no_tracked_test_runtime_loads_gitignored_reports_path`
went red: `reports/` is gitignored, so a tracked test loading a dossier tool **fails in any fresh clone
or CI**. Its named remedies are "move the tool under the tracked package, or guard on availability" —
and a `pytest.skip` guard would mean the rule that ~14 GPU-hours feed has no test in CI, which is how a
decision rule quietly stops meaning anything.

Fixed by taking the shape `scripts/floor_gate.py` already established: the **rule** moved to
`scripts/lesson_curve_gate.py` (pure function over parsed arm records, 21 never-skipping tests, a pinned
`rule_md5`), and `verify_curve.py` became reading and rendering only. This is a better boundary than the
one I started with, and I did not find it by taste — a test did.

**5. Arm labels for the new lesson counts are legal.** `arm_label` for 600/900 yields
`sixhundredzero_fortytwo` / `ninehundredzero_fortytwo`, both matching `^[a-z]+_[a-z]+$`, and every
existing label (`longzero_fortytwo`, `shortzero_fortytwo`, `longhalf_fortyfive`) is unchanged.

**6. The verdict harness reproduces the anchor from the real archived data.** `verify_curve.py` run
against the live results directories finds `longzero_fortytwo` in the SS dossier, computes
C=0.4745 / F=0.2569 / **R=0.5415** and FG-A **PASS**, and returns **UNDERPOWERED** with the correct
reason ("1 arm(s) at L=160; sigma_seed needs at least 3"). Those match the pre-registration's table
digit for digit, and the state is refused rather than guessed — which is the behaviour the four-state
design exists for.

**7. Decision-rule md5 `5d6a256bb2b41485220d033cd0bfbc87`**, pinned in `05_analysis_plan.md` §6 and
stamped on every `VERDICT.md`. Changing θ, k, G1 or the anchor length moves it, so a threshold relaxed
after seeing a control cannot pass unnoticed. Tested.

---

### Harness dry-run · 2-lesson arm, full seam · 2026-08-18 03:22–04:22 · **PASS**

`RES_DIR=<scratch> run_lesson_arm.sh tinyzero_fortytwo --gate`, so a deliberately floor-limited arm
could exercise every seam **without** writing into the real curve's results, where `verify_curve.py`
would have picked it up.

| step | outcome |
|---|---|
| build (`make_ss_arm --lessons 2`) | symmetric-difference assertion passes: only `total_lessons` and the C-259 `ss_feedback` insertion differ from the floor |
| `total_lessons` read from the arm's own config | 2 → `TRAIN_TIMEOUT` 21600 s, logged before any GPU work |
| train + emit | rc=0, 28 min, cube with **13 origins** |
| score | 22-column CSV, `N=170430` at every horizon |
| bootstrap (AP) | h1 mde 0.00910, h18 mde 0.00061 |
| bootstrap (**retention**, the new paired mode) | ratio 0.19441, mde 0.03327, `n_origins` 13 |
| `arm_<label>.json` | weight-tensor sha256, lessons, seed, HEAD — read from the config, not the shell |
| cube deleted | yes, before the oracle ran |
| **oracle (`use_real`) on a freshly-cloned arm dir** | rc=0, 24 min — **the one seam nothing had tested** |
| floor gate | writes `FLOORGATE_tinyzero_fortytwo_FAIL` |
| sentinel | `ARM_DONE_tinyzero_fortytwo` |

**F1 holds byte-exactly.** Control h1 `0.054523146320288096`, oracle h1 `0.054523146320288096` —
|Δ| = **0.000e+00**. Step 1 has no feedback, and on a fresh arm at a lesson count nothing has run before,
the two paths agree exactly. That is the falsifier the whole oracle design leans on, confirmed on real
output rather than argued from the code.

**The floor gate fired, which is the point of running a 2-lesson arm.**

```
FG-A [FAIL] AP 0.01060 / prevalence 0.009077 = 1.17x (need >= 5.0x)
FG-B [FAIL] AP(h1) 0.05452 / clim 0.29798 = 0.183x
FG-C [PASS] a 30% effect is 0.00742 AP; 3x MDE is 0.00182
```

Three rungs now separate cleanly on the same gate: **1.17× (2 lessons, FAIL) · 2.16× (40, FAIL) ·
28.30× (160, PASS)**.

**⚠️ What this is NOT.** `tinyzero_fortytwo` fails FG-A, so by this dossier's own §6 F4 its numbers are
**not evidence about the curve** and are quoted here only as harness output. Read the temptation and
resist it: the arm's ceiling O=0.0648 against 0.4793 at 160 lessons would make a lovely story about the
ceiling being trained, and it is exactly the story a floor-limited vehicle is licensed to tell falsely.
The gate exists so that number never enters an argument. The arm directory was deleted afterwards for
the same reason.

**Cost note for the estimates in `LAUNCH.md`.** The 28-min train+emit and 24-min oracle here both ran
**contended** — the full 1573-test suite and other work shared the box for much of it. Uncontended
measurements on comparable vehicles are ~10 min (free-running emit) and ~6 min (oracle). The curve runs
unattended, so the `LAUNCH.md` table is the right basis; this run is not a counter-measurement, and is
recorded as contended so nobody later reads it as one.

---

### Stage 1 · sigma_seed at L=160 · 2026-08-18 04:49–08:25 · **COMPLETE**

Four independent 160-lesson training runs, ε=0, one variable (`torch_seed`). All four **PASS FG-A**.

| seed | C = AP h1 | F = AP h18 | R = F/C |
|--:|--:|--:|--:|
| 42 | 0.4745 | 0.2569 | **0.5415** |
| 43 | 0.4510 | 0.2834 | 0.6284 |
| 44 | 0.4641 | 0.2683 | 0.5780 |
| 45 | 0.4591 | 0.3052 | 0.6648 |

**sigma_seed(R) = 0.0544** — 9.0% of the mean 0.6032. The first between-run variance ever measured on
this vehicle. G1 does **not** fire (k·sigma = 0.1431, far below 0.30), so the curve is worth running.

**Three findings, none of which needed a treatment arm.**

1. **Seed 42 is the worst of the four.** The anchor the whole rollout programme is benchmarked against —
   violet_visitor's 0.54 retention — is a **below-average draw**. Typical 160-lesson retention is
   **~0.60**. Nothing is wrong with the 0.54 number; it was simply never known to be low.
2. **Seed noise lands almost entirely on retention, not the ceiling.** sigma(C)/mean(C) = **2.1%**;
   sigma(R)/mean(R) = **9.0%**. One-step skill is highly reproducible across training runs; the ability
   to survive its own output is 4× less so. That is a statement about where training variance lives, and
   it was not predicted.
3. **We landed 2% over the pre-registered power boundary** (0.0544 vs 0.0532) ⇒ AMENDMENT 2.

**F6 fired and was wrong** ⇒ AMENDMENT 1. Verdict was VOID on a two-commit-id span whose
`views_hydranet/` tree hash is identical at both ends (`ca41c3f5`, 0-line diff).

---

### Stage 2 · L=300 seed 42 · **FAILED — `CUDA error: unspecified launch failure`**

Died at ~lesson 192/300 after 66 min, `rc=1`. Not a timeout (cap was 6 h) and not a modelling failure:

```
training_engine.py:965 in training_loop -> w_loss.backward()
RuntimeError: CUDA error: unspecified launch failure
```

**Cause: the machine was physically moved mid-run** (maintainer, ~09:31). The GPU recovered on its own —
the next arm started clean, no CUDA errors, 27% utilisation at 68 °C — so only the in-flight process was
lost. There is **no mid-training checkpoint** (`train_model.py:73` saves once, at the end), which is why
66 minutes went with it.

Recorded rather than quietly retried because it is the second time an environmental event has destroyed
a long arm on this box (the ensemble dossier lost 5.5 h to an unattended-upgrades reboot). The class is
real: **a long HydraNet arm is an indivisible unit, and this machine is not a quiet one.**

Requeued by `tools/run_followup.sh`, which also carries AMENDMENT 2's seeds. Priority order is the
missing curve **point** first, then the power — so an interruption leaves the more valuable half done.

---

### Cost model FALSIFIED · 2026-08-18 13:05 · **training cost per lesson climbs steeply with `total_lessons`**

`LAUNCH.md` extrapolated per-lesson training cost from two points (40L and 160L) as
`0.3058 · (L/160)^0.144`, and `SCOPE.md` §8 warned that two points cannot separate a fixed overhead
from a superlinear term. They could not, and the extrapolation was wrong by **2.3×**.

| L | measured min/lesson | source | predicted |
|--:|--:|---|--:|
| 160 | **0.306** | seeds 43–45, 49 min train | — |
| 300 | **0.344** | the CUDA-killed arm, 192 lessons in 66 min | 0.336 ✓ |
| 600 | **0.839** | 255 lessons in 214 min | 0.372 ✗ **2.3× low** |

**160 → 300 is nearly flat; 300 → 600 is 2.4×.** The likely mechanism is **C-301** showing up as a cost
signal: `curriculum.py:85` normalises the cooling slope by `total_lessons`, so a 600-lesson run holds the
event threshold high for far longer, and the volume sampler works much harder to find qualifying cells.
Consistent with the observation that the process is **CPU-bound, not GPU-bound** — 1336% CPU across 20
cores at **12% GPU utilisation**. If that is the mechanism, cost is not a property of length alone but of
length *through the curriculum*, which is the same confound the analysis already declares.

**Consequences, acted on the same hour:**

1. **A 900-lesson arm is ~13.5 h, not the ~6.2 h `LAUNCH.md` claims.** `run_curve.sh` was stopped before
   its stage-4 branch could spend that on a single point while the 300-lesson point was still missing.
   The running L=600 arm was left alive deliberately — killing the driver reparented it and it kept
   advancing.
2. **`TRAIN_TIMEOUT` was 36·L + 3600 s, i.e. 7 h at L=600 — inside the plausible range of the run it was
   guarding.** Raised to 72·L (12 h at L=600, 18 h at L=900), which carries ~1.4× margin over the worst
   measured rate. Unchanged at L ≤ 300, where the 6 h floor already dominates. Applied by
   `run_followup2.sh` *after* the in-flight arm ends, because editing a bash script that bash is
   currently reading corrupts execution.
3. The L=600 arm sped up markedly in its second half (25 s/lesson at lesson 258 vs a 50 s/lesson
   average), consistent with the curriculum explanation, and is now projected to finish inside its
   original timeout.

**The honest reading:** the pre-registration flagged this extrapolation as an extrapolation and it still
cost a stopped driver and a re-planned schedule. Naming a weakness is not the same as being protected
from it — the timeout should have been sized for the stated uncertainty, not for the point estimate.

---

### L=600 attempt 1 · **KILLED BY ITS OWN TIMEOUT** at lesson 459/600 · 2026-08-18 16:31:48

`rc=124` after 420 min. The 7 h cap (`36·L + 3600`) was **inside the plausible completion range of the
run it was guarding**, and the run lost the race.

**This was called and called wrong.** At 13:05, with the arm at lesson 255 and the timeout 3.5 h away,
the log records the risk explicitly — "finish at average rate 17:54, at recent rate 15:42, timeout
16:31, between the two". I sampled the instantaneous rate (25 s/lesson at lesson 258), concluded it
would land inside, and chose not to intervene. The average over the full run was **0.915 min/lesson**;
the fast stretch did not hold. **7 GPU-hours lost to a decision made on a 25-second sample when a
3.5-hour average was available.**

The right move at 13:05 was to restart the arm under the corrected timeout, paying 3.5 h to protect 7 h.
Recorded because the cost model being wrong (previous entry) is a forgivable extrapolation, while
choosing not to act on a known-marginal deadline is a judgement error, and they should not be filed
together.

**Measured rate table now has three points at L=600 and they disagree with each other by 2×** — 0.42
min/lesson instantaneous at lesson 258, 0.839 average to lesson 255, 0.915 average to lesson 459. The
per-lesson cost is not stationary within a run, which is itself consistent with the curriculum
explanation (C-301): the sampler's workload tracks the event threshold, and the threshold moves
throughout training. **Any future timeout must be sized on the worst full-run average, never on an
instantaneous rate.**

**Disposition:** timeout corrected to `72·L` (12 h at L=600 vs 9.2 h measured, 18 h at L=900). Retry
queued **last** rather than next — see `run_followup3.sh`. The reorder is scheduling only; no
pre-registered quantity moved.

---

### Queue after the kill · 2026-08-18 16:34

| order | arm | ~h | why this position |
|--:|---|--:|---|
| 1 | `fullzero_fortytwo` L=300 | 2.0 | **running**; first longer-length magnitude readout |
| 2 | `longzero_fortysix` L=160 | 1.1 | cheap; AMENDMENT 2 power |
| 3 | `longzero_fortyseven` L=160 | 1.1 | cheap; takes n to 6 |
| 4 | `sixhundredzero_fortytwo` L=600 | 9.2 | 3rd attempt, expensive, twice-failed — goes last |

Previously 300 → 600 → seeds. Moved because the L=600 arm has now failed twice for unrelated reasons
and costs 9 h, and the seeds cost 1.1 h each. Under the new order every prefix is worth having: a
magnitude readout at a longer length, then a **declarable** null on retention, then the expensive point.
`verify_curve.py` is re-run after each seed so the verdict is current even if the L=600 retry dies again.

---

### ROOT CAUSE of the day's failures: the GPU is thermally throttled to 24% · 2026-08-18 23:55

Three arms died on 2026-08-18 for three apparent reasons. Two of them share one real cause.

```
clocks.sm 735 MHz   clocks.max.sm 3105 MHz   ->  23.7% of maximum
temperature 86 C     power.draw 14.9 W
AC connected (battery charging, 93%), power profile "balanced"
```

**Identical configs run 2–3× slower than they did this morning.** `longzero_forty{three,four,five}`
trained 160 lessons in **49–73 min** between 05:07 and 08:25. `longzero_fortysix` — same config, same
seed family, same code — is on track for **~156 min**. The change coincides with the machine being
moved at ~09:31.

**This invalidates every cost figure in the dossier**, because they were all measured on a cool box.
The "cost per lesson climbs with `total_lessons`" entry above is now **confounded with thermal drift**:
the L=600 measurement (0.839 min/lesson) was taken in the afternoon, the L=160 measurements (0.306) in
the early morning. The C-301 curriculum explanation may still be right, but **this data cannot separate
it from throttling**, and the entry should not be cited as if it could. A clean test needs the same
lengths measured back to back at a stable clock.

**Two timeout kills follow directly from it:**

| arm | died | how far |
|---|---|---|
| `sixhundredzero_fortytwo` | 7 h cap | lesson 459/600 |
| `fullzero_fortytwo` | 6 h cap | **lesson 300/300, 19 min into the emit** |

The second is the instructive one. **Training finished. The artifact was written**
(`calibration_model_20260818_221401.pt`, 22:14:01). The timeout killed the *emit*, and the default
recovery — rerun the arm — would have thrown away **5.6 h of completed training** to redo work that was
already on disk.

**Fixes applied:**

1. `run_lesson_arm.sh` now **resumes from a saved artifact** rather than retraining. `train_model.py:73`
   writes the `.pt` the moment training ends, so a killed emit costs one emit, not a whole run. This
   turns the worst failure of the day into a ~30–60 min recovery.
2. Timeout raised **36·L → 72·L → 150·L** (2.5 min/lesson). We have now lost ~12 GPU-hours to timeouts
   and **zero** to hangs; the asymmetry is settled and the guard should be generous.
3. Queue reordered so the cheap recovery runs first: **L=300 from artifact → seeds 46/47 → L=600**,
   with `verify_curve.py` re-run after each so the verdict is never stale.

**Owed to the maintainer, not fixable from here:** the box needs airflow. At 24% clock everything
downstream — cost model, timeouts, schedule — is measuring the cooling system, not the model.

---

### CORRECTION to the previous entry · 2026-08-19 00:24 · **the bottleneck is the CPU, not the GPU**

The entry above named a *GPU* thermal throttle as the root cause. That reading was wrong in a way that
matters, and the numbers that refute it were available when I wrote it.

```
CPU  i9-13900H   avg core 1839 MHz of 5400 max (34%)   x86_pkg_temp 100 C   TCPU 99 C
GPU               615 MHz of 3105 (20%)   83 C   12.3 W   utilisation 25%
python 1351% CPU  (13.5 of 20 cores)      load average 15.7
```

**A GPU that is thermally throttling draws its full power budget and sits at its temperature limit.**
This one draws **12 W** at **25% utilisation**. It is idling because it is starved, not throttled — the
low clock is ordinary power management. The real limiter is the **CPU at 100 °C, clocked to 34%**, and
the reason the CPU is the limiter at all is that this workload is **CPU-bound in the volume sampler**
(`views_hydranet/utils/volume_sampler.py` searching for cells above the curriculum threshold), which is
the same mechanism C-301 predicts — just bounded by the wrong device from what I claimed.

**What survives from the previous entry:** identical configs really are running 2–3× slower than this
morning; the cost figures really are confounded by machine state; the timeout and resume-from-artifact
fixes are right regardless. **What does not survive:** "the GPU is thermally throttled to 24%" as the
cause. It is a symptom of CPU starvation.

**Also competing:** an unrelated `ffmpeg` audiobook job (~99% of one core, launched from a different
session) and a long-running `transmission-gtk` (3%). Neither is the main cause — the thermal ceiling is —
but they are on the same budget.

**Consequence for the programme, and it is not small:** if HydraNet training is CPU-sampler-bound rather
than GPU-bound, then the global-server scale-up ([[project_global_server_endgame]]) is bounded by
sampler throughput and core count, not by GPU memory or FLOPs. That is a different provisioning
question than the one currently assumed. Flagged, not investigated — it needs a clean profile on an
unloaded machine, not an inference from a contended one.

**Method note.** Twice today I diagnosed from an instantaneous reading (a 25-second lesson rate, then a
single `nvidia-smi` line) when a fuller picture was one command away. Same error class both times.

---

### Seed 46 lands · AMENDMENT 2 works · 2026-08-19 01:48

`longzero_fortysix`: C 0.4648, F 0.2880, **R 0.6196**. FG-A PASS.

**sigma_seed(R): 0.0544 (n=4) → 0.0477 (n=5).** With the multiplier at its correct n=5 value the
prediction bound is `k·sigma = 0.1113` against `theta = 0.14` — **a null is now declarable**, which is
exactly what the amendment was run to buy. Mean retention across five runs is **0.6065**; seed 42
remains the lowest of the five.

**Implementation bug found while reading the output.** `K_PRED` was hard-coded at **2.631**, the n=4
value, while §5 states the multiplier as `t(n-1,0.95)·sqrt(1+1/n)` and AMENDMENT 2 tabulates it for
n=5,6. At n=5 the correct value is **2.335**. The hard-coding was **conservative** — a wider bound makes
both RISING and PLATEAU *harder*, so it never manufactured a positive — but it silently discarded the
power the extra seed was run to buy, reporting `0.1254` (undeclarable) where the pre-registered formula
gives `0.1113` (declarable). Fixed with an eight-entry t-table; `rule_md5` is unchanged because
`K_PRED` remains the pinned n=4 reference and the formula was always n-dependent. 21 tests green.

### Disk exhaustion halted the queue for 6.7 h · 2026-08-19 01:49 → 08:36

`followup4` hit its own `>= 25 GB` guard at **10 GB free** and **exited**. Nothing ran between 01:49 and
08:36. Cause was external: an unrelated audiobook job in `~/Downloads/audio_books` (now 141 GB) filled
the disk overnight.

Two fixes:

1. **The guard now pauses instead of aborting** — it re-checks every 10 min and gives up only after
   12 h. A transient squeeze on a shared box should stall a queue, not end it. Aborting cost 6.7 h of
   idle GPU on a machine that was otherwise free.
2. **21 GB reclaimed**, all of it re-derivable and none of it an artifact: the 40-lesson fixture cube
   (its regression check is done and recorded), the partial cube + `_pf_staging` left by the killed
   L=300 emit, and the **seven `_quarantine_` Epic #263 cubes**.

**On deleting the quarantine cubes:** this reverses my own recommendation of 2026-08-18, which was
"keep them". That advice was given at 73 GB free; at 10 GB the trade is different. What is lost is a
cube re-derivable in ~10–20 min from a preserved `.pt`, whose scores are already in tracked CSVs, and
whose reproducibility EXP-00 *already proved* — the proof is what makes the object redundant. All 20
`.pt` artifacts are intact. Recorded as a reversal rather than presented as the plan all along.

**Also fixed:** the L=300 recovery would have aborted on the partial cube its own killed emit left
behind — `run_lesson_arm.sh` refuses to start when `predictions_*` is present. The follow-up now clears
it first. Found by reading the script, before it ran.

---

