# Structure-penalty arm — OUTCOME

**Pre-registration:** `reports/2026-10-21_structure_penalty_arm_prereg.md` (amended 4×, last `7a290ed`,
2026-10-25T23:59:52+01:00 — **2 h 31 m before the run started**).
**Verdict: the arm is NEGATIVE on every proper score, and the two predictions it PASSED both
passed for a reason that makes them uninformative.** The finding is not the sign of the result.
It is that **the penalty term and the B-1 diagnostic disagree about the same model by sign.**

---

## 1. What ran

| | |
|---|---|
| views-models branch / commit | `feat/607-struct-penalty-arm` / `fa2de298` |
| checkpoint | `calibration_model_20261026_040937.pt`, sha256 `aab48beb4fa8480da1552694351774b094abba0450fa270ccdffb66a85b23696` — **digest re-verified here on disk, not taken on report** |
| trained | 2026-10-26T01:43:09Z → 04:09:42Z, 300 lessons, mode `production`, region `land` |
| hardware / cost | RTX PRO 4500 Blackwell, 87 GB RAM, 128 vCPU · **~$3.00** |
| the ONE changed key | `'struct_penalty_weight': 0.416904` on `heavy_freighter` |
| control | `calibration_model_20261025_132124.pt`, git `c8ae1803`, same 300 lessons, `STATUS: OK` |
| data | ⭐ all 2,834 files / 1.4 GB at `~/views-diagnostics/hf_arm_20261026/` |

⭐ **The weight is the PRE-REGISTERED one.** The prereg's original `8.89624` was retracted in
`de4fd3e` — three independent errors (derived on amber_kestrel's `truncated_nb`; a superseded v1
script; gradient-matched against **one** of the three regression targets `training_engine.py:995`
sums over). `0.416904` was committed to §3c at 2026-10-25T23:59, and the pod began training at
2026-10-26T01:43. ⇒ **This is a pre-registered result, not an exploratory one.** The `models`
session flagged the opposite possibility, correctly and without knowing the amendment existed.

### 1b. ⭐ The experiment is clean — measured by the `models` session, not assumed

| | |
|---|---|
| `queryset_digest` | **IDENTICAL** to the control: `a7fce3a6cfdc4c8b5dbb46be37a3a1ddc75f3c80ae090d06cfc5c5928a1bc685`, pgm, months 121–504 |
| truth grids `y_reg`, `y_cls` | **BYTE-IDENTICAL** arm vs control in all three training windows at lesson 300 |
| trajectory at lesson 0 | **NINE columns byte-identical**: `loss_reg`, `loss_cls`, `gate_logit_mean`, `b_{sb,ns,os}`, `g_{sb,ns,os}`. Only `raw_grad_norm` differs — which is the manipulation. |
| environment | `views-hydranet==0.1.7`, `views-pipeline-core==3.3.4`, `torch==2.14.1` — 176-line pinned set, **exactly the control's** |

⇒ The floating `views-hydranet~=0.1.1` PyPI range (`pod_run_model.sh:172`) **did not bite**. That
was HARD-3 of the 2026-10-26 readiness falsification; it was a live risk, and it resolved
favourably by luck rather than by a pin. ⛔ **It is still unpinned for the next arm.**

---

## 2. Gate A — the penalty is live

`struct_pen` has no logged magnitude (**C-365**), so the manipulation check used lesson-0
`raw_grad_norm`:

| | |
|---|---|
| arm | 1794.6770552604437 |
| control | 1591.1967992930588 |
| Δ | **+203.480256 = +12.788%** |

⚠️ **One-sided by construction:** it can condemn an inert knob; it cannot certify a correct one.
It did not condemn. Gate B (throughput) 33.8 s/lesson vs the control's 31.6, threshold 63.0.

---

## 3. ⛔ THE SCORECARD — all five pre-registered predictions

| # | quantity | control | **arm** | prediction | verdict |
|---|---|---|---|---|---|
| **P-1** ⚪ manipulation | structure penalty, trained ckpt, 12-draw mean | 1.7223 | **0.3540** | gap closes ≥ 50% | ✅ **PASSED** — 79.4% closed |
| **P-2** guardrail | B-3 level, implied | 95.9% | **1026.4%** | must not degrade | ⛔ **BLOWN, 10.7×** |
| **P-3** ⭐ the hypothesis | B-2 slope `b`, 336-mo, 1,431 cells | 0.4996 | **0.8066** | rises to ≥ 0.55 | ✅ **PASSED** — ⚠️ see §5 |
| **P-4** | B-1 at d=1, in-sample E–W, **1/n intercept** | 0.287 | **0.085** | rises ≥ 0.10 ⇒ ≥ 0.387 | ⛔ **FAILED, OPPOSITE SIGN** |
| **P-5** | FAO-02 `crps_all` | — | 8–38% worse | not worse by > 5% | ⛔ **FAILED** |

**2 passed, 3 failed.** ⛔ **And the two that passed are the only two quantities on the board that
are sensitive to the SPREAD of the predictions rather than their accuracy.** That is §5.

### 3a. The held-out scores (`metrics/`, 6 metricframes — the authoritative scores)

| target | metric | arm | control | ratio | better |
|---|---|---|---|---|---|
| `lr_sb_best` | CRPS | 0.08876 | 0.08219 | 1.08× | CONTROL |
| `lr_sb_best` | QS_sample | 0.07271 | 0.06232 | 1.17× | CONTROL |
| `lr_sb_best` | MCR_sample | 2.16320 | 0.39742 | 5.44× | CONTROL |
| `lr_ns_best` | CRPS | 0.01956 | 0.01605 | 1.22× | CONTROL |
| `lr_ns_best` | QS_sample | 0.01867 | 0.01333 | 1.40× | CONTROL |
| `lr_ns_best` | MCR_sample | 4.06027 | 0.37719 | 10.76× | CONTROL |
| `lr_os_best` | CRPS | 0.01449 | 0.01052 | 1.38× | CONTROL |
| `lr_os_best` | QS_sample | 0.01765 | 0.00953 | 1.85× | CONTROL |
| `lr_os_best` | MCR_sample | 7.22472 | 0.35506 | 20.35× | CONTROL |
| `by_sb_best` | Brier_cls_sample | 0.00270 | 0.00269 | 1.00× | CONTROL |
| `by_ns_best` | Brier_cls_sample | 0.00126 | 0.00153 | 0.82× | **ARM** |
| `by_os_best` | Brier_cls_sample | 0.00134 | 0.00132 | 1.02× | CONTROL |

**Control better on 11 of 12.** ⭐ **The gate barely moved** (all three Brier within 0.82–1.02×),
which is the signature of a penalty acting on the body and leaving the gate alone — consistent
with `_structure_penalty`'s gate being **detached** by design.

⛔ **MCR crossed 1.0 and overshot, monotonically in target sparsity.** Control MCR 0.355–0.397 —
it UNDER-predicts, the timid-prophet failure the MCR guardrail exists to catch. Arm 2.16 → 4.06 →
7.22 for sb → ns → os. ⇒ The penalty did not fail to act on magnitude. **It acted far too hard,
in the direction the control was deficient, and hardest where the target is rarest** — which is
exactly §3c.1's measurement that non-state carries 0.374910 of the 0.827130 summed penalty
(**45.3%**) and the largest penalty gradient (3.7407e+00) despite being the rarest target.

---

## 4. ⭐⭐ THE FINDING — two instruments for one quantity, disagreeing by SIGN

Both of these were built to answer *"does this model carry its due spatial structure?"*
Both were run on **the same two checkpoints**, and for the penalty on **the same window**
(rows 191–223, cols 350–382, land 99.4%), the same seed, the same 12-draw dropout sweep.

| instrument | control | arm | direction |
|---|---|---|---|
| **`_structure_penalty`** — the loss term, Σ over blocks (2, 8) | 1.7223 | **0.3540** | ⭐ **4.87× BETTER** |
| **B-1** — γ_model(d)/γ_truth(d) at d = 1, in-sample E–W | 0.400 | **0.090** | ⛔ **4.44× WORSE** |

⛔ **They cannot both be describing the same change.** And B-1 is worse at **every** scale, not
just the finest:

| d | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| control, in-sample E–W | 0.400 | 0.488 | 0.572 | 0.697 | 0.872 | 0.929 |
| **arm**, in-sample E–W | **0.090** | **0.181** | **0.351** | **0.523** | **0.609** | **0.676** |
| control, out-of-sample E–W | 0.183 | 0.196 | 0.243 | 0.394 | 0.578 | 0.716 |
| **arm**, out-of-sample E–W | **0.088** | **0.168** | **0.326** | **0.496** | **0.556** | **0.636** |

### 4a. The mechanism, read off the term's own docstring

`training_engine._structure_penalty`:

> *"The penalty is a ONE-SIDED difference of LOGS: a 10× dispersion **SHORTFALL** costs the same at
> every intensity, and **excess dispersion costs nothing**."*

⇒ The term says *"your within-block dispersion is below the truth's — raise it"*, and **places no
cost whatever on raising it too far.** Nothing in the objective opposes it except the body NLL,
which the frozen balancer weights **0.5**. The cheapest available route to satisfying it is to
inflate the field.

**That is what the model did.** `models` measured the conditional mean μ̂ falling 0.85 → ~0.10 in
the control and **RISING 0.85 → ~9 in the arm**, while `loss_reg` barely moved — a mean/dispersion
trade along an NB likelihood ridge, with a higher θ-CoV panel in the arm, which is consistent.

⚠️ **Flagged as the leading hypothesis, NOT as a measurement.** Nobody has run the
profile-likelihood check that would establish the ridge. What IS measured: the penalty fell 4.87×,
B-1 fell 4.44×, μ̂ rose ~10×, MCR rose 5–20×.

### 4b. ⛔ Why this is a defect in the term and not a dose error

A smaller weight reaches the same place more slowly. A one-sided penalty with no opposing term
has **one** direction available to it and **no** stationary point in that direction short of the
body NLL's own resistance. Halving the weight halves the pressure; it does not introduce a cost
for overshoot, which is the thing that is missing.

⇒ **The next move is not a weaker weight. It is a two-sided penalty, or none.**

---

### 4c. ⭐⭐ A MATCHED IN-TRAINING WINDOW — the same finding, on byte-identical truth

§4 is measured at the two trained checkpoints. This subsection measures the same thing *during*
training, on a window both runs saw, and it is the strongest single piece of evidence in this
document because the two arms are compared against **the same truth array**.

- control: `~/views-diagnostics/hf_control_20261025/diagnostics/251026_1043/02_training_dynamics/biopsy_stage_5:_training_forensic_(lesson_297_win_1).npz`
- arm: `~/views-diagnostics/hf_arm_20261026/plots/02_training_dynamics/biopsy_stage_5:_training_forensic_(lesson_297_win_1).npz`

⭐ `y_reg` **and** `y_cls` are byte-identical between the two files (`np.array_equal`, asserted in
the reader, not assumed). Same lesson, same window, same months 293–298.

**POPULATION:** 6 steps × 32×32 × 3 targets = 18,432 cell-month-targets.
**TRANSFORM:** log1p counts, as stored. **ESTIMATOR:** plain mean / percentile / correlation over
that population.
⛔ This is an **in-sample curriculum window**. It is not the held-out score of §3a and must never
be quoted beside it.

| mean (log1p) | truth | body | gate | forecast | forecast ÷ truth |
|---|---|---|---|---|---|
| control | 0.0147 | 0.0720 | 0.0107 | 0.0163 | **1.11×** |
| arm | 0.0147 | **1.3449** | 0.0119 | **0.1247** | **8.51×** |

**Reading 1 — the level.** The control's emitted field matches the truth's level to 11% on this
window. The arm's is 8.51× it. Consistent in sign and rough size with P-2's held-out 95.9% → 1026.4%,
and arrived at by a different instrument on a different population, so the two are corroborating,
⛔ not combinable.

Every number in this subsection is produced by `reports/2026-10-26_matched_window/matched_forensic_window.py`, which asserts the truth
equality before it compares anything and exits VOID if it fails.

**Reading 2 — the arm raised the FLOOR, it did not raise the PEAKS.**

| body percentile | p50 | p75 | p90 | p99 | max |
|---|---|---|---|---|---|
| truth | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5.8377 |
| control | 0.0060 | 0.0357 | 0.1584 | 1.3240 | 2.8413 |
| arm | 0.1862 | **2.9228** | 3.5370 | 4.0684 | 4.3522 |

The body's mean rose **18.7×** while its maximum rose **1.53×**. p75 rose **82×**. A quarter of the
arm's body field sits above 2.92 in log1p (≈ 17 counts) on a window where **over 99% of the truth
is exactly zero**. The control's body spans p50→p99 by a factor of 221; the arm's by 22.
⇒ the intervention did not sharpen the map, it flooded it — which is what a one-sided dispersion
penalty with no cost for overshoot (§4a) predicts.

**Reading 3 — placement, and ⛔ two correlation conventions that disagree.**

| | Pearson | | Spearman | |
|---|---|---|---|---|
| | control | arm | control | arm |
| body vs truth | +0.4861 | **+0.0933** | +0.1470 | +0.0981 |
| body vs gate | +0.8516 | **+0.2418** | +0.7890 | **+0.8920** |
| gate vs truth | +0.6213 | +0.6065 | +0.1497 | +0.1498 |

- **The gate is untouched** on both conventions — corroborating §3a's Brier 0.82–1.02×.
- **body vs truth falls on both conventions.** Placement got worse, not better.
- ⛔ **body vs gate falls 3.5× on Pearson and RISES on Spearman.** Per the standing rule, this is
  reported as a split and not resolved to one number. The honest reading of the split is that the
  arm's body still **ranks** cells roughly as the gate does — better, even — and has lost the
  **amplitude** relationship, because its values are compressed against a high floor.
- ⭐ That is the same shape as P-3 (§5): the scale statistic `b` improved while the correlation `r`
  fell. Two independent populations, two independent instruments, one pattern — **ranks preserved,
  magnitudes destroyed.**

⚠️ Note against over-reading: the control's body-vs-truth Spearman is only +0.147. The control did
**not** place well in rank terms either. The claim here is strictly comparative — the arm placed
*worse* — not that the control was good.

### 4d. ⭐⭐ C-366 ANSWERED — the two instruments measure DISJOINT properties

`reports/2026-10-26_body_optimisation_gap/c366_two_instruments.py`, both instruments
re-implemented from their own source and run on **one shared field**: the matched
`lesson_297_win_1` biopsy, byte-identical truth, 18 fields (6 steps × 3 targets), 18,432
cell-values. Space is **swept**, not chosen — counts (the penalty's native inputs, reconstructed
as `y_hat_cls * expm1(y_hat_reg)`) and log1p.

**⛔ F1 did not fire.** The sign disagreement **reproduces on the shared field**, so it is a
property of the two definitions and not of the two populations:

| space | penalty ctrl → arm | B-1 d=1 | d=2 | d=4 | d=8 |
|---|---|---|---|---|---|
| counts | 0.5468 → **0.1164** (4.70× better) | 0.3618 → 0.3635 ⚠️ tied | 0.7411 → **0.6226** | 0.9786 → **0.8160** | 1.0245 → **0.9432** |
| log1p | 0.3075 → **0.0403** (7.63× better) | 0.3801 → **0.1988** | 0.7121 → **0.4107** | 0.9477 → **0.6305** | 1.0459 → **0.8862** |

⚠️ **At d=1 in counts the two arms are tied** (0.3618 vs 0.3635). The disagreement is firm at
**d ≥ 2 in both spaces** and at d=1 in log1p only. ⛔ Do not quote the d=1 figure alone.

**The four hypotheses, all pre-committed in the script's docstring before it ran.**

| | claim | verdict |
|---|---|---|
| **H1** | the penalty is homogeneous in amplitude — scaling the field moves it with nothing rearranged | ✅ ×4 takes it 0.5468 → **0.2029** (counts) |
| **H2** | B-1 is **exactly invariant** to that scaling | ✅ **EXACT.** max \|change\| over d ∈ (1,2,4,8) = **0.000e+00** at ×0.5/×2/×4; 4.4e-16 at ×100 |
| **H3** | the penalty is **exactly invariant** to permuting cells *within* each block | ✅ **EXACT.** shuffle within 2×2 → penalty change **+0.000e+00**, block variance unchanged at k=2 *and* k=8 (≤5.7e-14), while B-1 d=1 jumps **0.3618 → 0.6856** |
| **H4** | adding unstructured noise **improves** the penalty and **raises** B-1 | ✅ penalty 0.5468 → 0.3226 → 0.2446 → 0.1933 at τ = 0.5/1/2 sd, B-1 d=1 0.3618 → 0.5008 → 0.7054 → **0.9185** |

⇒ **The penalty sees how much dispersion there is and is blind to where it sits, at the very
scale it is evaluated on. B-1 sees where it sits and is blind to how much.** They were never two
estimates of one quantity, so a disagreement by sign is not a contradiction. **C-366 dissolves.**

**⭐ And the penalty half of the arm's change is fully accounted for by AMPLITUDE alone.**
Step 4 swept scale × smoothing passes against the arm's observed pair. In counts the closest cell
is **scale ×8 with zero smoothing** — penalty **0.1256** against the arm's **0.1164** (within 8%)
— and by H2 that costs B-1 *exactly nothing*. ⇒ **The term bought amplification, not structure.**

**⛔ AND HALF MY OWN MECHANISM IS REFUTED BY THE SAME BATTERY, which is why it is here.**
Two readings of the arm have been offered in this programme and the battery kills both:

1. ⛔ **"The penalty was satisfied by flooding the field with noise"** — stated in §4c's framing
   and repeated to Simon. **Refuted by H4:** noise *raises* B-1 (to 0.9185 at τ=2 sd). The arm's
   B-1 **fell**. Whatever the arm did, it was not adding unstructured noise.
2. ⛔ **"The field was smoothed"** — my own replacement guess, offered before measuring.
   **Refuted by the lag profile.** A 3×3 box smooth hits short lags hardest and leaves d=8
   untouched (B-1 d=1 → 0.1526, d=8 → **1.0303**). The arm lost structure at **every** lag with
   d=8 at **0.9432**. Local smoothing has the wrong shape.

⛔ **AMENDED 2026-10-27b — THE NULL IS MUCH LESS SURPRISING THAN THIS DOCUMENT RECORDS, because
the headroom it was measured against was overstated 3×.**

This arm was released by amendment **X-1** of `2026-10-25_sharpness_ceiling_prereg.md`, which held
that B-1's `0.391` sat against a ceiling of **~1.0** — a ~61% shortfall, i.e. a lot to win. **X-1 is
now retracted.** Its evidence was B-1's formula applied to two halves of the TRUTH, which is ~1.0
for any dataset including white noise, because two halves of one dataset share a roughness and a
variance by construction. The real ceiling is the cross term, **0.498**.

| ceiling | `0.391` is… | headroom the arm was chasing |
|---|---|---|
| ~1.0 (as released) | 39% of achievable | ~61% |
| **0.498 (real)** | **78.5%** | **~21%** |

⇒ **The body was already carrying ~78% of the structure the truth can even demonstrate.** A null
from an arm chasing the remaining ~21% is close to what the ceiling implies — so the question is no
longer *"why did a 61% gap not close?"* but *"why did a 21% gap not close?"*, which is a far smaller
and far less mysterious question. ⚠️ **The arm's measurements are unaffected** — every number in
this document stands; what changes is the size of the prize it was scored against.

⛔ **The verdict is NOT reversed and the money is not recovered.** The arm still ran, still returned
a null, and its own term and its own diagnostic still disagreed by sign — that finding is
independent of the ceiling and is the more valuable half of this document. What is withdrawn is the
framing of the null as unexplained.

⇒ **The arrangement half of the arm's change remains unexplained** — a narrower open question
than C-366 was, and one that now has a sharp constraint on it: the arm's loss is **scale-free
across lags 2–8**, which neither of the two cheap mechanisms produces.

---

## 5. ⚠️ P-3 PASSED, AND THE PASS IS AN ARTEFACT — read this before quoting 0.8066

`body_level_selection_check.py`, **same instrument, same session, both arms**, population fixed at
the 1,431 truth-positive cells, 336-month base-rate truth estimator:

| | control | arm |
|---|---|---|
| **`b`** — the pre-registered quantity | 0.4996 | **0.8066** ✅ |
| **`r`** — rank correlation, same fit | **+0.6833** | **+0.6090** ⛔ |
| level (implied) | 95.9% | **1026.4%** ⛔ |

⛔ **`b` rose while `r` FELL.** `b` is the OLS coefficient of log-truth on log-μ: widening the
predictor's dynamic range raises `b` without the predictions becoming more accurate, and here the
agreement actually got *worse*. ⇒ **P-3 passes as written and tells us nothing we wanted to know.**

⚠️ **This is the pre-registration's fault, not the arm's.** §4 chose a scale-sensitive statistic
as the headline for an intervention whose only mechanism is to change scale. The prereg should
have pre-committed `r` alongside `b`, or pre-committed `b` *conditional on the level guardrail
holding*. It did neither. **W-63.**

---

## 6. P-4 scored on its pre-committed estimator — and the estimator now exists as code

§4b fixed **the 1/n intercept, not the 4-draw value**, as P-4's scoring quantity, before any arm
number existed. New cubes: `d4_result_ARM_hf0126_{s0..s3,p1A..p3B}.npz`, the three **disjoint**
2-vs-2 partitions of four draws.

| | n=1 | n=2 | n=4 | 1/n fit | residuals |
|---|---|---|---|---|---|
| **arm** B-1 d=1 | 0.1072 (sd 0.0010, 4 cubes) | 0.0967 (sd 0.0005, 6 halves) | 0.0900 | **0.0847 + 0.0227/n** | ±0.0006 |
| control B-1 d=1 | 0.7547 (sd 0.0075) | 0.5288 (sd 0.0033) | 0.4000 | **0.2870 + 0.4700/n** | ±0.007 |

| | |
|---|---|
| control intercept | 0.287 |
| **arm intercept** | **0.085** |
| prediction | rises ≥ 0.10 ⇒ **≥ 0.387** |
| observed move | **−0.202** |
| split-half noise floor, arm (equal draws, 3 disjoint partitions) | **0.0010** |
| **gap / floor** | **202×** |

⭐ **A second-order finding worth keeping:** the arm's draw-count bias is **0.0227**, against the
control's **0.4700** — **21× smaller**. The control's B-1 was 28% MC-dropout artefact because its
field variance was small; the arm's field variance is dominated by its inflated mean, so dropout
noise is a negligible share of it. ⇒ **The same diagnostic has a 21× different noise content on
two checkpoints of the same architecture.** Any future B-1 comparison must sweep draws on *both*
sides; a single-sided correction would have been wrong by 0.45 here.

⛔ **`split_draws.py` is now a committed instrument** (`reports/2026-10-25_arm_readiness/`). The
control's splits were made ad hoc in a shell on 2026-10-25 and that code died with the shell —
so from then until now, **the estimator §4b pre-registered for scoring the headline prediction
existed only as a command someone had typed.** That is C-106's shape applied to an estimator:
second instance of "a load-bearing thing with no durable home" recorded today, the first being
the `models` session's abort gate lost to an overwritten plan file.

---

## 7. ⛔ THE SEVENTH INSTANCE

`reports/2026-10-21_structure_penalty_arm_prereg.md` §4, written before the run:

> *"35% is a deliberate, low prior, and it is the honest one: **seven interventions have failed,
> four of them on the same lever.**"*

And the method review of the sibling fork asked outright whether this was *"the seventh instance
waiting to happen."*

**It was.** M45 (`truncated_nb` ×1170 → −0.238 AP), M62 (`act_ratio` ×3.0), M64 (×56), and now
this: **an intervention that made the model predict more, and lost on every proper score.**
`project_firing_is_not_the_lever` records the rule as *"AP loss scales with how much the model
FIRES"*; §3a's MCR 0.36 → 2.2–7.2 is that rule, measured one more time.

⭐ **What is new here, and worth the $3:** the previous four were interventions that increased
firing *as a side effect*. This one increased it as its **designed mechanism**, with a
pre-registered diagnostic that was supposed to detect the intended effect — and the diagnostic
**reported success while the outcome failed**. ⇒ The lever is not merely unhelpful; **the
instrument we built to steer it points the wrong way.**

---

## 8. ⚠️ LIMITS — pre-registered, and binding

1. ⛔ **n = 1 against n = 1.** Nothing here bounds run-to-run variance. σ̂ is still unmeasured
   (views-models #572/#566). **The 1.08× CRPS gap on `lr_sb_best` is NOT distinguishable from
   seed noise.** The 5–20× MCR gap, the 0.202 B-1 gap (202× its floor) and the 10.7× level gap
   are far too large to be seed noise, and those carry the verdict.
2. ⛔ **§7.8's second-seed trigger did not fire** — it was defined for *"the screen comes back
   positive and near the ask"* (P-3 ∈ [0.50, 0.60] or P-4's intercept ∈ [0.34, 0.44]). P-4 came
   back at 0.085, P-3 at 0.8066 with `r` falling. Neither is near its ask in the sense the trigger
   meant. ⇒ **No second seed is owed, and the reason is the pre-committed rule, not a judgement
   made after seeing the numbers.**
3. ⚠️ **The split-half floor still does not bound seed variance.** It splits one checkpoint's
   draws. Unchanged from §7.8.
4. ⚠️ **The penalty-share figures are TWO POINTS, NOT A TRAJECTORY** (C-325, narrowed not
   retired). They cannot show when a term drifted, a spike that resolved, or a divergence that
   reversed. **C-365's fix — one trajectory column per additive loss term — would have made §4
   a curve instead of two points, and it is now the single highest-value change for the next arm.**
5. ⚠️ **Region is recorded as ENFORCED, not measured.** `PROVENANCE` records that
   `pod_run_model.sh:279-282` refuses any region but `land`; nothing in a `.pt` or its sidecar
   records the region, and the net is fully convolutional. Weaker than a measurement.
6. ⚠️ **P-1 is scored on the SUMMED penalty over blocks (2, 8), not per block.** The prereg asked
   for "≥ 50% at `k = 2` **and** `k = 8`". The instrument reports the sum. ⇒ **P-1's pass is
   reported at lower resolution than it was pre-registered at**, and a per-block split is owed
   before anyone leans on it.

### 8b. ⛔ The verification failure rate behind §1b and §3a was THREE

The `models` session self-reported, unprompted, that a `/falsify` pass on *"I can terminate the
pod"* was FALSIFIED with three hard findings, all since remediated and registered (C-179/C-180/
C-181, plus instances on C-161 and C-169):

1. It synced one of two diagnostics folders and verified the sync with a count taken **inside the
   folder it had synced**. The 13 per-origin `03_model_reasoning` biopsies were nearly destroyed.
2. It skipped `slice_gate_posterior.py`, which the run guide mandates before teardown, and
   substituted a raw superset for the validator. Run afterwards it came back clean — **but that
   was not known when the pod was declared safe to kill.**
3. A digest comparison of its own produced **empty hashes and reported agreement**.

⇒ **Recorded here because §1b's cleanliness evidence and §3a's scores rest on that session's
verification, and its measured failure rate on the day was three.** The checkpoint digest and the
B-1/penalty/P-3 numbers in this document were re-derived here from the artefacts on disk; the
`queryset_digest`, byte-identity and trajectory-column claims were **not** independently re-run
and carry that session's verification only.

---

## 9. Where the two gate artefacts are, and why they are not interchangeable

| | |
|---|---|
| `gate_draws/gate_h1.tar.zst` (43.8 MB) | the h=1 slice, **VALIDATED**: `slice_gate_posterior.py` exit 0, 13 origins × 3 targets, every row count exactly 64,818, 16 draws consistent across targets, all probabilities finite in (0,1). sha `c13e86a55aac430465405c4eb95fec53fa2d682aee708b7378f0662b1efa6e08` |
| `gate_draws/by_*.tar.zst` (683 MB, 78 members) | the full posterior, byte-identical to the pod per member — but **none of those structural checks ran on it**. sha `80bedf6816d3d6100912849b5bac095c728977a660e4bb4abf423fb28394aeec` |

⛔ **The control has NO h=1 slice** (`~/views-diagnostics/hf_control_gate_draws/`, 662 MB, raw
only, 117 entries = 13 origins × 3 targets × 3 files). ⇒ A slice-vs-slice gate comparison requires
making one from the control's raw archive first. **Given §3a — the gate moved 0.82–1.02× — there
is currently nothing for a gate-side readout to read, so this is recorded and not chased.**

---

## 10. What this closes and what it opens

**CLOSED.** The structure penalty as written is not a usable lever. Not at this weight, and not at
a smaller one — §4b. ⛔ **Do not re-propose a one-sided dispersion penalty** without first
answering §4's disagreement.

**OPEN, in priority order.**
1. ⭐⭐ **Why do `_structure_penalty` and B-1 disagree by sign?** Both claim to measure carried
   spatial structure. One is within-block dispersion on 32×32 training windows, one-sided, in
   logs; the other is a variance-normalised variogram ratio on the 360×720 grid. **Until that is
   resolved, neither can be trusted as a target, and B-1 has been the body programme's primary
   diagnostic since D12.**
2. ⭐ **C-365** — a trajectory column per additive loss term. Needs a 0.1.8 release. It converts
   §4 from two points into a curve and is the cheapest large improvement available.
3. ⭐ **Pin the pod's views-hydranet.** `~=0.1.1` floats; it resolved to 0.1.7 by luck this time.
   `pod_run_model.sh` is on the never-modify list, so this needs an env seam or an explicit
   decision to accept the risk.
4. **A two-sided dispersion penalty**, if §4's disagreement resolves in the term's favour. A
   symmetric log penalty has a stationary point; the current one does not.
5. **σ̂** (#572/#566) — still unmeasured, and limit 1 will bind on every future n=1 screen.

---

## Provenance

| | |
|---|---|
| arm data | `~/views-diagnostics/hf_arm_20261026/` — 2,834 files, 1.4 GB |
| control data | `~/views-diagnostics/hf_control_artifact/`, `hf_control_gate_draws/`, `hf_control_inputs/` |
| new cubes | `reports/2026-10-12_underfire_diagnosis/d4_result_ARM_hf0126{,_s0..s3,_p1A..p3B}.npz`, `d12_result_ARM_hf0126*.npz` |
| new instrument | `reports/2026-10-25_arm_readiness/split_draws.py` |
| readouts re-run here | `body_scoreboard.py`, `body_level_selection_check.py`, `d12_variogram.py`, `struct_pen_at_checkpoint.py` — **both arms, same session, same instrument** |
| ⛔ the pod | **TERMINATED.** HydraNet is not bit-reproducible (C-317). Nothing more can be taken from it. |
