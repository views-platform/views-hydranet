# Pre-analysis plan — is the deployment state outside the range the model was trained on?

**Status: LOCKED.** Committed alone, before any tool in `tools/` existed. `git log` proves the ordering.

---

## 0. Where this came from, and a correction

The user asked whether the training curriculum runs "random windows → conflict" or "conflict → random
windows". Investigating that surfaced `CurriculumLearner` and, following the thread, a structural
asymmetry between how the recurrent state is built in **training** and in **deployment**.

⚠️ **A claim made before this investigation was wrong and is retracted here.** I stated that "training
windows are 36 timesteps" while inference runs 335 + 36. **False.** `VolumeSampler._generate_window`
slices `vol_data[:, r0:r0+dim, c0:c0+dim, :]` on axes `("T","H","W","C")` — the **full time axis is
kept**. A training window is cropped **spatially**, not temporally, and `h` runs the full training
length. `time_steps=36` is the forecast horizon (`Field(description="Checksum for 'steps'")`), not a
training window length. **The temporal asymmetry I proposed does not exist.**

**What the investigation actually found is spatial**, and it is sharper:

| | training | deployment |
|---|---|---|
| state shape | `[1, C, 32, 32]` (`window_dim=32`) | `[1, C, H_grid, W_grid]` (full map) |
| where the state runs | patches **selected by an activity threshold** | **the entire map**, including everything that would never qualify as a training window |
| re-initialised | once per window, zeros | once per origin, zeros |

So the state has only ever been run over **activity-selected patches**, and at forecast time it must be
maintained **everywhere**. Two further mechanisms make this more than a framing point: the architecture
is a **BatchNorm recurrent U-Net**, so (a) BN statistics are gathered on activity-selected patches and
applied to the full map (the C-184 family), and (b) a U-Net's bottleneck resolution and effective
receptive field differ between a 32×32 and a full-grid input.

## 1. Hypothesis

**H:** the recurrent state arriving at the rollout origin lies outside the range of states the model
produces on training-distribution input — i.e. free-running begins from an out-of-distribution state.

**Why it is worth testing.** It would explain our single best result without extra assumptions: freezing
the **cell** state is worth **+0.039 AP@h18** (M38/M39). If the deployment state is out of range, the
clamp helps because it **pins the state in a regime the model recognises**. No other story we hold
explains M38 mechanistically. It also gives the zero collapse a cause that is not feedback — which
matters, because **every feedback lever we have tried has failed** (M30–M33 SS, M42 ITF), and M42 showed
the damage is a monotone function of exposure dose rather than of curriculum direction.

**H0 (the null we must be able to accept):** the deployment state sits inside the training-input range;
the collapse is a *dynamical* property of free-running, not distribution shift. **This is a real
possible outcome and closes the thread.**

## 2. The one variable

**The input distribution the trained model is run on.** Weights, artifact, months, and code are held
identical across regimes. Nothing is retrained.

## 3. Method — and why it needs no GPU-hours of training

We are **not** instrumenting training. The question is what state distribution the **trained** model
produces on **training-like** input versus **deployment** input; that is answerable with the existing
checkpoint by forward passes alone.

⚠️ **Stated limitation, not to be glossed:** this is *not* "the states as they were during training" —
the weights moved throughout training. It is the trained model's response to two input distributions.
That is the deployment-relevant question, and it is the only one available without retraining, but the
write-up must not claim the stronger thing.

**Vehicles: TWO seeds — `fullzero_fortytwo` and `fullzero_fortythree`** (both L=300 ε=0 controls,
already trained and already scored). Seed 43 is the seed M38/M41 and the σ_max work used, so its
published free-running numbers apply as a cross-check; seed 42 is the independent replicate. **`f` is
computed per seed and both are reported.** A verdict requires **both seeds to land in the same branch of
§4**; a split (one OUT-OF-RANGE, one IN-RANGE) is reported as **INCONCLUSIVE — SEED-SPLIT**, never
resolved by picking one.

**Regimes captured (state = both halves; report `hidden` and `cell` separately, since M39 established
the cell carries the entire freeze effect):**

| | regime | what is run | state captured at |
|---|---|---|---|
| **R1** | training-like | 32×32 windows drawn by the **real `VolumeSampler`** at the **real curriculum thresholds** — sampled at three points of the schedule, `ratio = 0.665` (roof), `0.35` (mid), `0.05` (floor) | final step of each window |
| **R2** | deployment | full grid, `origin` history steps | `t == origin` — literally the existing `state_anchor` |
| **R3** | free-running | full grid, `t > origin` | every free step |

**Statistics.** Not `max|h|` alone — C-308 was caused by a single summary number reading plausibly while
describing the wrong thing. Per channel: mean, sd, and the 1/25/50/75/99% quantiles of the state over
cells; plus the pooled distribution. R1 aggregates over ≥100 windows so its range is a distribution, not
one draw.

## 4. Decision rule — registered BEFORE any capture, with thresholds justified externally

Build, **per channel**, the R1 (training-like) `[1%, 99%]` interval. Let **`f`** = the fraction of R2's
(cell × channel) values falling outside their channel's interval.

**By construction, `f = 0.02` is what chance alone produces** (a 98% interval). Thresholds are set as
multiples of that chance rate, not from our numbers:

| `f` | verdict |
|---|---|
| **≥ 0.20** (10× chance) | **OUT-OF-RANGE** — free-running starts from a state the model was not trained to hold. Reframes the rollout problem; the next lever is the training input distribution, not the feedback. |
| **≤ 0.05** (2.5× chance) | **IN-RANGE** — H is dead. The collapse is dynamics, not distribution shift. Say so and close the thread. |
| between | **INCONCLUSIVE** — report `f` and both intervals, claim nothing. |

**No branch may be overridden by an argument not written above** (C-305: a registered rule fired, was
overridden on grounds it did not contain, and was written up as "no branch matched"). If the rule fires
a branch we dislike, the branch stands and the objection goes in the log as a **separate** paragraph.

## 5. Falsifiers — pre-committed

- **F1 — zero init.** The captured state at the first step must be **exactly zero** in every regime
  (`init_hTtime` returns `torch.zeros`). Non-zero ⇒ the capture is attached to the wrong call. *This is
  the direct C-308 guard: that defect was a probe reading the wrong phase while every downstream check
  passed.*
- **F2 — same vehicle.** For **seed 43** R3 must reproduce the published free-running collapse
  (**65.6 → 1.6**) **within 5%** — this is a deterministic re-run of the same artifact over the same
  months, so a larger deviation means a different vehicle or config, not noise. For **seed 42** no such
  number has been published, so its collapse is recorded as a **new measurement with no threshold**
  (it is subject only to F5). Failing F2 on seed 43 voids the entire run.
- **F3 — the geometry control (CRITICAL).** Run the model on a **32×32 crop of the full grid at the same
  months**, and compare those states to the **same cells** taken from the full-grid R2 run. If they
  differ materially, then **input size alone** moves the state and **R1-vs-R2 is confounded by U-Net
  geometry rather than by data distribution** — the headline comparison would then be measuring image
  size, not data distribution. **Without F3 this experiment cannot distinguish its hypothesis from an
  image-size artifact.**

  **F3 is a HARD STOP.** If it fires, the run **aborts**: `f` is not computed, no verdict is rendered,
  and §4 is not consulted. The result is written up as *"the measurement was confounded and no
  comparison was made"*. It is explicitly **not** downgraded to a caveated headline — reporting a
  confounded number alongside a caveat is how **C-308** happened (a probe read the wrong phase, every
  downstream guard passed, and the plausible-looking number was published).
- **F4 — the curriculum actually bites.** Windows drawn at `ratio=0.665` must have measurably higher
  event density than at `ratio=0.05`. If not, R1 is not sampling what we believe and the whole R1 arm is
  void.
- **F5 — no silent NaN/inf** in any captured state.

## 6. What each outcome buys

- **OUT-OF-RANGE** ⇒ M38 gets a mechanism, and the lever moves to the **training input distribution** —
  which is where the user's original question pointed: the curriculum decides what the state is ever run
  over, and it has **never been varied** (27 of 27 configs identical, `{'value':}` not `{'values':}` in
  every sweep file — verified 2026-08-23).
- **IN-RANGE** ⇒ a clean negative that kills a plausible story cheaply, and redirects attention to
  free-running dynamics.

## 7. False-negative mode and reopen trigger (C-307 discipline, written before the result)

**Registered false-negative mode:** R1 uses the **final trained weights** on training-like input. If the
state distribution the model was *actually* trained on differed (because the weights moved), an
IN-RANGE verdict would not rule out that the *training-time* state distribution was different. **An
IN-RANGE result therefore closes "the deployed model is being run out of range", NOT "training never
saw a different state regime."**

**Reopen if:** anyone instruments a real training run's states (≈5 GPU-h/seed); or a curriculum arm is
run and changes rollout behaviour, which would imply the input distribution matters after all.

## 8. Scope

Two seeds, one architecture, `sb`. No training. No config changes. No claim about AP — this measures **states**,
not skill; any link to AP is inference and must be labelled as such.

---

# AMENDMENT 1 — R1 moves onto the same months as R2 (recorded before any tool ran)

**Defect in the locked §3.** R1 was specified as windows drawn by `VolumeSampler` from the **training
volume**, while R2 runs on the **evaluation** volume. Those are different months. The comparison would
therefore vary **two** things at once — spatial selection **and** partition — violating §2's "one
variable" and making any `f` uninterpretable: an out-of-range result could be months, not geometry.

**Fix.** R1 is drawn from **the same handler and the same months as R2**. The only thing that differs
between R1 and R2 is then **which cells the state is run over**: activity-selected 32×32 patches (R1)
versus the entire map (R2). Patch anchors are chosen by the **same** `SAMPLING_STRATEGY_REGISTRY`
function and the same `min_events` the real sampler uses, at the same three schedule points
(`ratio = 0.665 / 0.35 / 0.05`), so the *selection rule* is production; only the months are shared with
R2 rather than taken from the training partition.

**What this costs, stated plainly.** R1 is no longer literally "input the model trained on" — it is
"input distributed the way training input was *selected*, on the months R2 uses". The hypothesis is
about **spatial selection**, and this isolates exactly that.

**R1b (added, cheap).** The same activity-selected patches on the **training** months. This measures the
months axis directly instead of leaving it as an assumption. If R1 ≈ R1b, months are irrelevant and the
§4 verdict rests on selection alone. If R1 ≠ R1b, that is itself a finding and the write-up must say the
two axes are not separable in this design.

**§4 is unchanged** and is computed on **R1 (shared months)**. R1b is reported alongside and **may not be
substituted into the decision rule** — swapping which arm feeds a registered rule after seeing the
numbers is exactly C-305.

**F3 is unchanged and remains a HARD STOP**, and under this amendment it becomes the *licence* for the
whole comparison: it holds location fixed and varies only patch-vs-full-grid, so passing it is what
permits R1 and R2 to be read as differing in *selection* rather than in *image size*.

---

# AMENDMENT 2 — F2 was premised on determinism the inference path does not have (before any tool ran)

**Defect in the locked §5.** F2 required reproducing seed 43's published collapse (**65.6 → 1.6**)
**within 5%**, justified as "a deterministic re-run of the same artifact over the same months". **That
premise is false.** `hydranet_inference.py:350-359` calls `model.eval()` **and then**
`set_locked_dropout(True)`, with `reset_locked_dropout()` refreshing the mask **once per posterior
sample** (ADR-057). The production rollout is therefore **MC-dropout stochastic**, and the published
65.6 came from one posterior sample's mask. Reproducing it to 5% would require restoring a global RNG
state that depends on everything executed before it — unreproducible in a standalone probe. **As
written, F2 would have aborted a correct run for the wrong reason.**

**Fix, two parts.**

1. **The measurement runs with dropout OFF** — `model.eval()` and `set_locked_dropout` left untouched
   (standard dropout is identity under `eval()`). This makes all regimes **deterministic and directly
   comparable**, so the one variable stays the *input*, not the dropout mask. Dropout noise would
   otherwise inflate both R1's and R2's spreads and blur `f` in an uncontrolled way.
2. **F2 is re-specified** as: on seed 43, R3's collapse ratio `‖state at origin‖ / ‖state at last free
   step‖` must be **≥ 10×**. The published value is ~40×; **10× is a deliberate factor-of-4 slack**. Its
   job is to catch *"wrong vehicle / wrong phase"* — which would show a ratio near **1** — not to certify
   numerical agreement with a dropout-active run it cannot match. The measured ratio is reported
   verbatim next to the published 40× so any drift is visible rather than absorbed.

**This makes F2 weaker, and that is the honest trade.** It no longer proves we are on the identical
trajectory; it proves we are on a trajectory that collapses. Recorded as a **registered limitation**, not
presented as an equivalent check.

---

# AMENDMENT 3 — F3 gets a numeric rule, and a disclosure

## ⚠️ Disclosure first

**F3's locked wording — "if they differ materially" — is qualitative**, and a smoke run of
`capture_regimes.py` (2 patches, 2 locations, seed 42) **has already shown me F3 values before any
numeric threshold existed**: `rel_abs_diff ≈ 0.09–0.10`, with patch |state| mean 0.1434 vs the same
cells of the full grid 0.1458.

**This is the C-305 shape** — a qualitative registered rule with room for a post-hoc criterion — and it
is disclosed here rather than resolved silently. Everything below is therefore **not blind**. The
threshold is set from an argument that does not reference those numbers, and **the expected outcome is
stated up front** so this cannot later be presented as a blind test.

## The design flaw the smoke run exposed

A 32×32 patch and the full grid **must** disagree near the patch edge: the patch's border cells have no
neighbours where the full grid does, and the model is convolutional. **A single pooled `rel_abs_diff`
therefore confounds an unavoidable boundary artifact with the thing F3 is meant to detect** (whether
input *extent* changes the state where the data is identical). ~10% pooled over a 32×32 patch is exactly
what a boundary ring of a few cells would produce, and the locked F3 cannot tell that apart from a real
geometry effect.

## The rule

Split each F3 patch into **border** (cells within `b` of the edge) and **interior** (the rest), and
report `rel_abs_diff` separately for each. `b` is set to the model's receptive-field radius, read off
the architecture — **not tuned**. F3 is judged on the **interior** only:

| interior `rel_abs_diff` | F3 |
|---|---|
| **≤ 0.02** | **PASS** — where data is identical and boundary effects are excluded, patch and full grid produce the same state. R1-vs-R2 differences are then attributable to *selection*, not image size. |
| **> 0.10** | **FAIL → HARD STOP.** Input extent alone moves the state; §4 is not consulted and no verdict is rendered. |
| 0.02–0.10 | **HARD STOP as well**, reported as *"geometry contributes an amount comparable to what we are trying to measure."* |

**Justification, independent of the observed values:** `f` in §4 asks whether R2 sits outside a 98%
interval built from R1. For that comparison to mean anything, any *systematic* patch-vs-full-grid offset
must be small next to the interval's own width. 0.02 is the same order as the 2% tail mass the interval
is defined by; anything at 0.10 is five times that and would dominate. The 0.02/0.10 pair is read off
§4's own construction, not off the smoke output.

**Expected outcome, stated before the run:** interior **PASSES** and the border carries most of the
pooled ~10%. **If that is what happens, it is a confirmed expectation, not a discovery** — and if the
interior fails, the run stops and this experiment produces no verdict at all.

**The pooled number is still reported** next to the split, so the border effect is visible rather than
defined away.

---

# AMENDMENT 4 — the expectation in AMENDMENT 3 is withdrawn; F3 will probably HARD STOP

Written **before** the F3 run, because the reasoning changed and the prediction must change with it.

## The receptive field compounds across timesteps

Reading `HydraBNrecurrentUnet_06_LSTM4.py`: 3×3 convs (`padding=1`), two `MaxPool2d(2,2)`, a bottleneck
conv, two `ConvTranspose2d(2, stride=2)` and decoder 3×3 convs. Accumulating jump and radius gives a
**single-pass receptive-field radius of ≈ 12–13 cells**.

**But the state is recurrent.** `h_t` feeds the forward pass that produces `h_{t+1}`, so the region of
input influencing a cell's state **grows by ~12 cells per timestep**. R1/R2/F3 digest **384** steps.
After ~2 steps the influence region already spans a 32×32 patch; after 384 it vastly exceeds it.

**Therefore a 32×32 patch run for 384 recurrent steps has no boundary-free interior.** AMENDMENT 3's
`b = receptive-field radius` would leave a 6×6 "interior" that is, after two timesteps, just as
boundary-contaminated as the border. **AMENDMENT 3's stated expectation — "interior PASSES" — is
withdrawn. The honest prediction is that F3 FAILS and the run HARD STOPS.**

## Why this is not a reason to weaken F3

The temptation is to loosen F3 so the experiment can produce a number. **That is precisely what F3 exists
to prevent**, and C-308 is what it looks like when a confounded measurement is published because it
looked plausible. **F3 stands as written in AMENDMENT 3.** If it fails, this design yields **no verdict**,
and that is the correct result rather than a disappointing one.

## What the failure would actually mean — a real finding, not just a dead end

If F3 fails, it says something true and non-obvious: **boundary effects are not a nuisance in this
design, they are a property of training.** Training runs the state over 32×32 patches for the full
sequence, so the training states genuinely *are* boundary-dominated, while deployment states are not.
**Geometry is not separable from "training-like" here — it is part of what training-like means.** R1 and
R2 would then differ in selection *and* in geometry inseparably, which is exactly what §2 forbids.

## The successor design, named now so it cannot be reverse-engineered later

If F3 hard-stops, the follow-up is a **new pre-registration**, not an amendment to this one:

> Within the **single full-grid R2 run** — no patches anywhere, so no geometry variable exists — compare
> the state distribution over cells that **would** have qualified inside a selected training window
> against cells that **never** would. Same run, same months, same image size; the only thing that varies
> is which cells are read out.

That tests the same hypothesis (does the model hold state over territory it never trained on?) with the
confound removed by construction. **It is written down here, before F3 runs, so that if F3 does fail the
successor cannot be presented as a fresh idea that happened to arrive after an inconvenient result.**

---

# AMENDMENT 5 — closing two gaps in §4 before `f` is computed

§4 says "build, per channel, the R1 interval" but **R1 is three ratio points, and the state has two
halves**. Left unspecified, both would be choices made after seeing `f` — the C-305 shape again. Fixed
here, before the analysis step runs.

**(a) The interval pools all three ratios.** `ratio = 0.665 / 0.35 / 0.05` are three points on **one**
schedule the model traverses within a single run, so the union is the training diet it actually
experienced. Per-ratio `f` is reported as **secondary** and **may not be substituted into the verdict**.

**(b) The verdict is rendered on the CELL half.** Justified externally by **M39**: freezing `cell` alone
reproduces the entire freeze effect (+0.036 of +0.036) while `hidden` alone does nothing (−0.005). The
cell is therefore the half whose regime demonstrably matters for rollout skill. **`hidden` is reported
beside it, always.** If the two halves land in different §4 branches, the result is reported as
**HALF-SPLIT** and the cell branch is *not* silently taken as the answer.

## F4 disclosure

F4's registered wording — windows at `ratio=0.665` must have "measurably higher" event density than at
`0.05` — **is qualitative, and the capture has already shown me the values**: mean event density
**4.548** (thr=143) / **5.985** (thr=75) / **3.966** (thr=10), identical across both seeds (anchor
selection depends on the data and the tool's own RNG, not on the model — a consistency check that
passed).

**F4 PASSES on its registered comparison** (4.548 > 3.966) but **only by 15%, and non-monotonically** —
the middle ratio is the densest. Recorded as a **weak pass with a caveat**, not a clean one: the
production `sigmoid` anchor strategy plus up-to-`dim` spatial jitter evidently blunts the threshold, so
**the curriculum separates the training diet far less than its 0.665 → 0.05 range suggests**. That is
itself a finding about the curriculum and is carried into the write-up regardless of what `f` does.
