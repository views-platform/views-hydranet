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
