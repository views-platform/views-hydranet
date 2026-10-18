# Pre-analysis plan — what compresses the gate's logits, and can a one-parameter fix repair it?

**Date:** 2026-10-18 · **Seat:** views-hydranet (this session) · **Status:** written before any Tier-1 measurement
**Predecessor (the fact this plan explains):** `reports/2026-10-16_pos_weight_arm_outcome/OUTCOME.md` @ rev 3. **Read its §2 and §8.4 first.**
**Artifacts:** `amber_kestrel` (`pos_weight = 2.0`, the delivered baseline) and `copper_kestrel` (`pos_weight = 1.0`, killed). Both on disk, both `truncated_nb`, `soft_gate`, 300 lessons.
**Cost:** Tier 1 and Tier 2 are **$0** and CPU-only. Tier 3 is **gated** and requests nothing until Tier 1 has named a cause.

**Authorship and inputs.** Written in this seat because pre-registration for views-hydranet experiments lives here — `reports/` holds 48 prior pre-analysis plans, every one committed from this repository. A draft written from the `views-models` seat is carried as **input, not as a base**: `inputs/views_models_draft_v3.md`, with this session's review of its v2 at `inputs/hydranet_review_of_draft_v2.md` and the superseded v2 with its errors intact at `inputs/superseded_draft_v2_with_errors.md`. The error trail is kept deliberately — a corrected document that hides what it corrected is worth less than one that shows it. **Candidates H1/H2/H3, experiment E6, and the F1 rank-identity invariant originate in this seat; the under-dispersion finding itself, the remedy ladder R1–R5, the origin-fold cross-validation and the GPU gate originate in the `views-models` seat.** Material rewritten rather than inherited, because a document carries its author's blind spots and the two seats have different ones.

⛔ **Division of labour, fixed before anything runs.** This seat writes and owns this plan. The `views-models` seat **attacks it and executes the measurements**, and does not decide which of its own objections survive. That division is what produced every correction of the last two days; the previous draft collapsed it by having one seat write, integrate its own criticism, and declare the result final — which is the "auditor may not be the author" rule this project enforces in `falsify` guard mode, broken in the document that cites it.

---

## 0. In plain words

The gate's probabilities are **squashed**. When it says 2% the real rate is about 1.3%; when it says 85% it is about 90%. It is over-confident about safe places and under-confident about dangerous ones, crossing over around 0.3. We know the shape precisely. We do not know what causes it.

This plan lists every cause we can think of, says in advance what each predicts, says what would prove each wrong, and commits to spending nothing on hardware until the free measurements have named one.

**The two leading suspects are both defects in this repository, not in the model's idea.** One is a post-training recalibration step that fits its statistics to the wrong kind of data. The other is a regularisation setting ten times its usual value. If either holds, the model is less broken than two days of work has been implying.

---

## 1. The fact to be explained — ONE fact

Fit `logit(observed rate) = a + b·logit(gate)` over gate bins. An honest gate gives `a = 0, b = 1`. Measured at h=1 on 13 out-of-sample months (457–469), 842,634 cell-months per arm:

| fit weighting | baseline `amber_kestrel` (w=2) | arm `copper_kestrel` (w=1) |
|---|---|---|
| bins with n ≥ 1000 | **1.146** | **1.419** |
| all non-empty bins | **1.177** | **1.251** |
| n-weighted, all bins | **1.139** | **1.456** |

**`b > 1` in all six fits — the gate is `under-dispersed`** (GLOSSARY §5b). Any proposed cause must reproduce three properties:

1. **Present in BOTH arms** ⇒ not caused by `pos_weight`. Reinforced analytically: at the weighted-BCE optimum `logit(s*) = log(w) + logit(p)` **exactly**, so `pos_weight` has zero slope degree of freedom (OUTCOME §2.4).
2. **Worse in the arm** (1.42 vs 1.15 central) — but see §3.4: at n=1 run per arm this is not attributable to `pos_weight`.
3. **Invisible to the pre-registered statistic**, because 96% of cell-months sit in the `[0.00, 0.01)` bin where the signed error is +0.0004.

⚠️ **One fact, not two.** An earlier draft treated a missing `−log(w)` intercept signature as a second fact. It is not: the per-bin deviation ranges 1.91 logits, so there is no single intercept, and a bin-unweighted fit reproduces to 0.0001 from bin weighting plus the slope defect alone (OUTCOME §8.6). Withdrawn.

⚠️ **`b` is a LOWER BOUND on the true compression.** Fitting on a noisy, binned predictor attenuates the slope toward 1 by two mechanisms: classical attenuation, and selection into extreme bins by noise. Both push `b` down. So `b > 1` survives two downward biases, and any σ-derived prediction inverting `T = b` is a **floor**, not a point.

---

## 2. Candidate causes

### The chain: C1 → C2/C3. MC-dropout averaging is a consequence, not a cause.

The reported gate is `mean(sigmoid(logit))` over `D = 4` locked-dropout draws, and averaging probabilities through a sigmoid compresses them: `E[sigmoid(z+ε)] ≈ sigmoid(z/T)`, `T = sqrt(1 + πσ²/8) > 1`. Right sign, real mechanism.

⛔ **But training cancels it.** Training runs `LockedDropout` with `locked=False` (`locked_dropout.py:61-62`) — a fresh mask on **every forward call**, ~383 per sequence inside one `backward()`. Setting the expected weighted-BCE gradient to zero over both `y` and `ε` gives

```
E_ε[ sigmoid(z + ε) ]  =  w·p / (1 + (w−1)·p)
```

**The stationary condition is on the mean of the sigmoid over masks — exactly the functional inference reports.** The compression is absorbed into the weights during training. **C1 therefore contributes nothing independently** and bites only through a displacement from that optimum:

- **C2 — training/inference asymmetry.** Training: ~383 **i.i.d.** masks per sequence, averaged inside the loss. Inference: **one** mask held fixed across the whole 36-step roll (deliberately, C-113, `hydranet_inference.py:536-540`), four such trajectories. Different count **and** different correlation structure, so the ε distribution differs and the absorption is imperfect. Remedy would be a training change.
- **C3 — non-convergence.** Partly answered: the class head's logit **level** converges by ~lesson 150 (§6, measured). ⚠️ Necessary, not sufficient — the instrumentation records no **spread** (§7 prerequisite).

### C4 — aggregation artefact

The slope is pooled over 13 origins and all land cells. If the gate is conditionally honest but we are mixing populations with different base rates, the pooled slope could be an artefact of the mixing. **This candidate attacks our own headline and belongs here for that reason.**

### C5 — measurement noise. RULED OUT. Attacked by both seats and survived.

Noise in the predictor makes the calibration curve **flatter** (`b < 1`) by both attenuation mechanisms above. We measure `b > 1` in all six fits — the opposite sign. **An entire class of explanations is dead at zero cost.** Both seats attacked this independently; neither broke it.

### C6 — a downstream compression site. ANSWERED AND CLOSED.

Traced at `ce56244`. Between the gate head's final linear layer and the reported number there is: `dec_conv4_head1_class` (`HydraBNrecurrentUnet_06_LSTM4.py:660`) → `out_class1 = H1_class` (`:661`, **raw — no activation, no clamp**, against `:643` where the regression head *does* apply one) → `torch.concat` (`:737`) → `ModelOutput.cls` (`:739`) → `torch.sigmoid` → `mean` over D=4. **No clamp, no normalisation, no epsilon, no clipping, no second averaging.** The only nonlinearity-then-average in the reporting path is C1's. Closed.

### ⛔ H1 — BatchNorm recalibration fitted on the wrong distribution. **CO-LEAD**

`bn_recalibrate: True` in both arms. `_recalibrate_bn` (`training_engine.py:1294-1325`) runs **after training**, resets every BatchNorm's running statistics and re-accumulates them forward-only with `momentum=None` (cumulative average). What it accumulates over:

- `n_windows = config.get("bn_recal_windows", windows_per_lesson * 10)` = **30 windows** (neither arm sets the key; `windows_per_lesson: 3`).
- `planner.get_lesson(w)` for **`w in range(30)`** — lesson indices **0–29 of 300**.
- `get_intensity_ratio` is `ratio = max_ratio + b·step` with `b = (−max_ratio + min_ratio)/(total_steps·slope_ratio) < 0` (`curriculum.py:79-93`), so the threshold **cools from hot to cold**. Indices 0–29 are the **hottest 10% of the schedule.**
- `window_dim: 32`, `batch_size=1`, `training_augmentation=False` (clean data), **dropout deliberately left on**.

⛔ **So the BatchNorm statistics that ship inside the artifact are estimated over 30 of the most activity-enriched 32×32 windows in the entire curriculum — and are then applied, frozen, to the whole quiet 180×180 global land grid at inference.**

The class head has BatchNorm immediately upstream of its final linear layer (`bn_dec_conv1_head1_class`, defined `:249`, used `:653`). A fixed σ estimated on an activity-enriched population is **too large** for the global grid, so eval-mode BatchNorm over-divides, shrinking the features entering the head and **multiplicatively compressing the logit**.

It reproduces every property in §1: a **pure slope error in logit space**; applied **after the last gradient step**, so no training optimum can absorb it; **present in both arms**; independent of `pos_weight`. And it is exactly C-184's blind spot — that fix was validated on flipping seed-bimodal eval collapse (*"6/6 bad seeds GOOD"*, its own docstring), **never on calibration slope**.

⚠️ **30 windows is a very small cumulative average,** so H1 additionally predicts the slope is **unstable run-to-run** while stable across origins within a run.

### ⛔ H2 — `weight_decay: 0.1` on the class heads. **CO-LEAD**

Both arms set `weight_decay: 0.1` — **ten times the AdamW default.** `choose_scheduler` builds `AdamW(unet.parameters(), weight_decay=config["weight_decay"])` (`utils/utils.py:277-282`) with **no exclusion list**, so `dec_conv4_head{1,2,3}_class` are decayed. The exclusion mechanism exists and is used twice for the balancer (`training_engine.py:113`, `:136`); the heads are simply not in it.

Decoupled decay shrinks the head's weights multiplicatively every step, and shrinking a final linear layer compresses its output range — the textbook cause of under-confidence in a regularised classifier.

⛔ **H2 is the candidate that predicts the sign *a priori*, with no statistic required.** `E[s] = w·p/(1+(w−1)·p)` is the stationary point of the **loss**; AdamW optimises **loss + penalty**, whose optimum is displaced toward smaller weights, hence **less extreme logits**. Weight decay moves the optimum away from calibration *by construction, in the compressing direction*. It is also why §6's finding that the level has converged does **not** rescue C1: the model converged to the penalised optimum, which is not the calibrated one.

### H3 — a resolution limit. The null, carried explicitly.

If the features cannot resolve per-cell `p` — separating "quiet" from "active" but not 2% from 8% — then the best achievable curve **is** flatter than truth, `b > 1` is an honest statement of limited resolution, and **no monotone recalibration can fix it**; it can only re-centre it. This is the world in which R1 "succeeds" on calibration and buys nothing real, and **F7 is the falsifier that detects it.**

### C7 — horizon. Unmeasurable today, and the one that matters for the product.

Everything here is **h=1**. The delivery is **36 months**, and compression should worsen with horizon as the rollout injects noise at every step. **The gate posterior at h≥2 is not retained** (#558 / C-161). This plan establishes a slope at h=1 and **must not generalise it to the delivered horizons.** That inability is the strongest argument yet for #558 and is recorded as a finding, not worked around.

---

## 3. ⛔ Skepticism ledger

### 3.1 The honest prior, with a number

Six interventions on this platform have failed and four failed on the same lever. **But this is not an intervention** — Tier 1 is a diagnosis on arrays already in hand, and Tier 2 is a post-hoc transform of a stored array. The thing that can fail is not "the fix doesn't work" but **"we never identify the cause"**, and that is the outcome I weight highest: **P(Tier 1 names a single cause) ≈ 0.45**; **P(it excludes at least one co-lead) ≈ 0.80**. Excluding one is worth having and is the realistic win.

### 3.2 Two of my own candidates are unfalsifiable-by-Tier-1 if E6 is ambiguous

H1 and H2 are separated by **one** experiment (E6). If E6 returns an intermediate result, Tier 1 cannot discriminate and **F6 fires**, sending the question to Tier 3 rather than letting either candidate be asserted. I am pre-committing that outcome now because the temptation afterwards will be to read an intermediate result as supporting whichever candidate I prefer — and both are mine.

### 3.3 ⛔ E6's test is biased IN H1's FAVOUR by sample size, and the bias is not small

E6 fits `b` on 32×32 windows and compares to the global grid. The windows have **far fewer cells**, so their per-bin observed rates are noisier, so **attenuation pushes `b_window` DOWN** — toward H1's prediction of `b ≈ 1`, for a reason that has nothing to do with BatchNorm. Neither seat caught this in three rounds.

**Control, pre-registered: E6c.** Subsample the **global grid** to the same cell count as the window set and re-fit. If `b_grid_subsampled` already falls to within 0.05 of `b_window`, **the entire E6 contrast is a sample-size artefact and F5 fires.** E6 is not interpretable without E6c, and E6c runs first.

### 3.4 The between-arm difference is not attributable, and I said otherwise once

`b` differs between arms (1.146 vs 1.419) and AP differs by +0.0075. **Neither is attributable to `pos_weight`.** The unit of randomisation is the **training run** and there is **one per arm**; the 13 months are correlated reads of the same two artifacts, not replicates of the intervention. The seed sd on AP is **~0.019** (`2026-10-16_pos_weight_prereg.md` VOID rule 4), so +0.0075 is **0.39 of one seed sd**. By §1's exact identity `pos_weight` has no mechanism to move a rank-based score at all. ⛔ **This session asserted "pos_weight is inert" too strongly in peer review and had to narrow it:** the correct claim is about the *mechanism*, and observed artifact differences are run-level. H1 independently predicts such run-level variance via 30-window BatchNorm statistics.

### 3.5 ⛔ Every scalar in this plan is subject to the sweep rule, including mine

**Three claims were published and withdrawn in this programme for the same defect** — a point summary of a quantity that varies across its own free parameter and straddles the value being tested: `implied_w` (6.8×/12.8×, crosses 1.0); a fitted intercept (1.91 logits, crosses zero); a "fit-free" ratio (0.628→1.209, crosses 1.0). Two were mine.

⛔ **Standing rule, in force for every number this plan produces: sweep whatever free parameter it depends on and show the family does not cross the value being tested.** E6 has **two** such parameters (bin count and window count) and §7 locks the sweep for both. **F4 voids any result that does not survive it.**

### 3.6 We tune under one score and judge under another

CRPS and LogScore prefer different departures (Machete 2012), and FAO-02 locks CRPS + AP as evaluation. A recalibration optimises calibration, which is neither. **Expect R1 to improve calibration and move nothing else — that is F7's whole point, not a surprise.**

---

## 4. Pre-registered predictions — and ⛔ what is NOT a prediction

**Three of the five Tier-1 items are measurements, not tests. Labelling them honestly is a change from the previous draft, which counted all five as experiments.**

| # | what it is | reason |
|---|---|---|
| **E1** | ⛔ **A MEASUREMENT, not a test.** | Its σ bands were derived by inverting `b` values already published in OUTCOME §2 and quoted in §1 here. Both seats have read them. Agreement would be **consistency, not confirmation**. It is reported as a measurement with no pass/fail. |
| **E2** | ⛔ **A DECISION TABLE, not a prediction.** | Its three outcomes (`b→1`, `b>1`, `b<1`) exhaust the real line and cannot fail. It earns its place by committing **in advance** to dropping remedy R2 if the third fires — that is a pre-commitment, not a risky prediction. |
| **E3** | measurement | Separates ensemble compression from network compression. No pre-committed threshold is defensible yet. |
| **E4** | ⛔ **CARRIES A BLIND PREDICTION** | Fit `T` from σ on 12 origins, predict `b` on the **held-out 13th before fitting it there**. Neither seat has seen per-origin values. |
| **E6** | ⛔ **CARRIES THE LOAD-BEARING BLIND PREDICTION** | The window-grid slope **exists in no artifact**. Neither seat can have seen it. It is the only experiment that discriminates the two co-leads. |

### The blind predictions, written down before measurement

**E6, point predictions, on the n≥1000 fit at the locked 12-bin resolution:**

| world | `b_window` | `b_grid − b_window` |
|---|---|---|
| **H1 alone** (BatchNorm) | **1.00 ± 0.10** | **≥ +0.10** |
| **H2 alone** (weight decay) | **1.15 ± 0.10** | **≤ +0.05** |
| **both contributing** | 1.05 – 1.10 | +0.05 – +0.10 |

`b_grid = 1.146` is the published reference. **The tolerance is ±0.10, not Tier 2's ±0.05**, and deliberately so: the three published weightings of `b_grid` already span 0.043 (1.139–1.177), so a discrimination threshold must exceed the spread the weighting choice alone produces. A gap smaller than 0.05 is not resolvable by this instrument and must not be read.

**E4:** per-origin `b` within **±0.10** of the pooled 1.146 if the defect is structural.

---

## 5. ⛔ Falsifiers — pre-committed. TIER 1 FIRST.

**The previous draft had six falsifiers and all six governed Tier 2. Tier 1 had predictions with no pre-committed consequence — the condition under which results get reinterpreted afterwards. These are the Tier-1 falsifiers it lacked.**

| | fires if | then |
|---|---|---|
| **F1** | `b_grid − b_window` **≥ +0.10**, sign preserved across all three bin resolutions, **and E6c shows subsampling alone moves `b_grid` by < 0.05** | ⛔ **H2 is DEAD as the primary cause.** Weight decay compresses the weights themselves and must show up on both populations. **H1 is the cause.** Fix is `bn_recal_windows` / the recal distribution — a bounded local change. |
| **F2** | `b_window > 1.10` **and** `|b_grid − b_window| ≤ 0.05` | ⛔ **H1 is DEAD.** BatchNorm statistics are correct for the window population by construction, so if the windows are equally compressed the cause is not the recal distribution. **H2 or H3.** |
| **F3** | `b_window ≤ 1.10` **and** `b_grid ≤ 1.10` | **VOID, not a result.** The grid slope is published at 1.146; reproducing it below 1.10 means the harness does not reproduce the finding and nothing downstream is interpretable. Stop and fix the harness. |
| **F4** | the **sign** of `b_grid − b_window` changes across the three locked bin resolutions, **or** `b_window` varies by **> 0.10** across the three locked window counts | ⛔ **VOID by §3.5's standing rule.** The measurement depends on its own free parameter. Report the family, assert nothing. |
| **F5** | **E6c:** subsampling the global grid to the window cell count moves `b_grid` **by ≥ 0.05 toward `b_window`** | ⛔ **E6 IS A SAMPLE-SIZE ARTEFACT** (§3.3) and cannot discriminate. H1 and H2 both survive; the question goes to Tier 3. **E6c runs BEFORE E6 is read.** |
| **F6** | E6 returns `+0.05 < b_grid − b_window < +0.10`, or any outcome F1–F5 do not cover | **INCONCLUSIVE, not negative** (C-307). Tier 1 has failed to discriminate. **Neither co-lead may be asserted**, and Tier 3's seed replicate becomes the next step. Pre-committed in §3.2. |
| **F7** | **Tier 2:** R1 reaches `b ≈ 1` held-out while **AP and sharpness are unchanged within the ~0.019 seed sd** | ⛔ **H3 — the gain is COSMETIC.** A calibration fix that adds no resolution has repaired a reporting defect, not the forecast. **Must be reported as such**, not as an improvement. |
| **F8** | **Tier 2:** R4 (isotonic) beats R1 held-out by **> 20%** on absolute calibration error | the defect is **not a pure slope**; OUTCOME §2 is incomplete *again* and must be revised a third time. |
| **F9** | **Tier 2:** R1's fitted `T` varies **> 15%** across the 13 origin folds | no well-defined global temperature; the simple fix is unavailable and a per-origin correction is a different, larger proposal. |
| **F10** | **Tier 2:** rank identity fails under R1 or R3 | ⛔ **implementation bug. Stop.** See §7's invariant — AP movement **alone** is not sufficient to fire this. |
| **F11** | **Tier 2:** `act_ratio` leaves its guardrail band under any remedy | the recalibration broke the composed forecast — the CRPS trap in reverse. The firing-rate guardrail is carried forward verbatim because it is the only reason the last experiment's CRPS artefact read as a failure. |
| **F12** | any Tier-1 statistic is reported without its **signed** value alongside its absolute one, or on non-common bin support across arms | **VOID.** These are the two defects that hid the finding for two readings. |

---

## 6. What is already settled, and must not be re-measured

- **C6 is closed** — the reporting path contains no compression site (§2).
- **C5 is ruled out** — noise gives `b < 1`; we measure `b > 1` (§2).
- **C1 is not an independent cause** — training calibrates the reported functional (§2).
- **Under-training is excluded on the LEVEL.** `copper_kestrel`'s trajectory CSV, 300 lessons: block means of the class-head logit mean `−6.937 → −7.588 → −7.987 → −8.121 → −8.030 → −7.984`; OLS slope over lessons 100–200 **−0.0019/lesson**, over 200–300 **+0.0003/lesson** — opposite sign, an order of magnitude below the within-block sd (~0.09). **Converged by ~lesson 150.**
- **`pos_weight` cannot be the cause** — exact `log(w)` intercept shift (§1).
- ⛔ **A fit-free test of the stationary condition is impossible for `w ≠ 1`** (OUTCOME §8.6). Do not re-attempt it; the estimator's bias flips sign with bin width and the family crosses 1.0.
- **The 16 posterior columns hold 4 distinct gate values.** `np.repeat(prob_thwc[..., None], posterior_K, -1)` at `hydranet_inference.py:1707-1709` — the head-sample axis `K` is pure duplication for the gate. **There is no 4-vs-16 draw-count contrast in the substrate**; any experiment premised on one is void.

---

## 7. Method — LOCKED before execution

**Shared.** `NP_SEED = TORCH_SEED = 42`; `BASE_ORIGIN = 456`; `N_ORIGINS = 13`; `LEADIN_START = 421`; CPU; `model.eval()` then `set_locked_dropout(True)`, `reset_locked_dropout()` once per pass, `D = 4`. Target **`sb`** (`by_sb_best` gate against `lr_sb_best` truth) throughout — the target P2 established. P2 bin edges `[0, .01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01]`. **Common bin support across every compared pair, always.** **Signed error reported alongside absolute, always.** Positive fraction pooled, never a median over cells. Slopes under all three weightings (n≥1000, all bins, n-weighted), never one. Logits recovered as `log(p/(1−p))`, clipping at 1e-9 recorded with a saturation count. Every artifact verified by sha256 against its `.pt.sha256` and its sidecar's `output_distribution` before use.

### ⛔ E6 — fully locked, because it carries the load-bearing blind prediction

The previous draft left window count, sampler configuration, origins, target, tolerance and bin resolution unfixed. All six are fixed here.

- **Sampling rule — matched to `_recalibrate_bn` exactly.** `VolumeSampler(handler, config)` + `CurriculumLearner(config, handler)`; `planner.get_lesson(w)` for **`w in range(n_windows)`** starting at **w = 0**; `sampler.get_batch(target, threshold, batch_size=1)`; `window_dim = 32`. This reproduces the recal pass's **selection rule** (`training_engine.py:1312-1314`), including that indices 0–29 sit at the hot end of the cooling schedule. Pattern already exercised at `reports/2026-10-13_composition_arm_outcome/r3_per_target.py:93-108`.
- ⛔ **Months are held IDENTICAL to the global-grid fit: 457–469, teacher-forced lead-in from 421.** The recal pass runs on training months, but changing both the spatial selection **and** the months would confound the contrast. **E6 isolates the spatial selection only**, and this plan does not claim otherwise. That the months differ from the real recal pass is a **stated limitation**, not an oversight.
- **Window-count sweep (free parameter 1):** `n_windows ∈ {30, 60, 120}`. **30 is the match** (`windows_per_lesson * 10`); the other two test stability. **F4 voids if `b_window` moves > 0.10 across them.**
- **Bin-resolution sweep (free parameter 2):** the locked 12 P2 edges, plus **40** and **200** equal-mass bins. **F4 voids if the sign of `b_grid − b_window` changes across them.** The 12-bin fit is the one the §4 point predictions are judged on.
- **Tolerance:** ±0.10, justified in §4 by the 0.043 spread across published weightings.
- **E6c runs first** (§3.3): global grid subsampled to the window cell count, same seed, same bins, 10 subsample replicates, report mean and sd of `b`.
- **Report:** per-bin `n` for both populations side by side, and the window set's pooled positive rate against the global grid's 0.003448 — so a reader can see how different the two populations are.

### Tier 2 — remedies

All are post-hoc monotone transforms of the stored gate. **Cross-validated by origin: fit on 12, evaluate on the held-out 1, 13 folds.** Fitting and evaluating a recalibration on the same data is how one gets a free pass, and this plan forbids it.

| # | remedy | params | why it is in the comparison |
|---|---|---|---|
| **R1** | **temperature scaling**, `logit/T` | 1 | the fix that matches the measured defect (GLOSSARY §5b) |
| **R2** | report `sigmoid(mean logit)` | 0 | ⚠️ **conditional on E2.** §2 predicts it **overshoots** to `b < 1`, because training calibrated `mean(sigmoid)`. **Dropped if E2's third outcome fires** — it is a different estimator with its own bias, not a free fix |
| **R3** | Platt / vector scaling, `a + b·logit` | 2 | the generality control. **If R3 does not beat R1 held-out, one parameter is enough and we stop** |
| **R4** | isotonic | non-parametric | **the falsifier for the slope diagnosis itself** (F8) — the ceiling on any monotone recalibration |
| **R5** | apply the same correction to **both** arms and report the Murphy **calibration–refinement decomposition** | — | see below |

⛔ **R5 is a localisation, NOT a resurrection of the killed arm.** A per-arm fitted temperature drives both arms to `b ≈ 1` **by construction**, so comparing calibration residuals afterwards is near-vacuous. Monotone transforms **cannot add resolution**. So the decomposition is the correct frame: the calibration component is removable, the refinement component is rank-based, and `pos_weight` provably cannot move ranks at its optimum (§1). **Pre-registered readings:** if resolution is equal within the ~0.019 seed sd, the `pos_weight` axis is confirmed inert and retired; if it differs by more, the two **artifacts** differ structurally — which **cannot be attributed to `pos_weight` at n=1 run per arm**, and which H1 independently predicts via 30-window BatchNorm statistics. Either way the killed arm stays killed.

### ⛔ The internal-validity invariant, and why it is not "AP must not move"

R1 and R3 are strictly monotone in the scalar gate, so they preserve ranks and AP is rank-based. **But "AP must be exactly unchanged" is the wrong invariant on two counts:**

1. **float32 tie creation.** A strictly monotone map can send distinct float32 inputs to the same output. With 842,634 values, 96% of them below 0.01 and a minimum of 2.0e-5, collisions are near-certain, and `average_precision_score` groups ties — so AP can move by a rounding artefact.
2. ⛔ **R2 is NOT a monotone transform of the gate.** It is a different functional of the **draw vector**: two cells can share `mean(sigmoid)` and differ in `mean(logit)`, and their order can genuinely flip. **R2 can and probably will move AP, legitimately, and that is a result, not a bug.**

**Locked invariant:** assert **rank identity** — `argsort` equality, or Kendall τ exactly 1.0 — between pre- and post-transform gate for **R1 and R3 only**, and report the count of newly created ties. **F10 fires only if rank identity fails.** R2 and R4 are excluded from it by construction.

### Prerequisite, not an experiment

⛔ **C3 cannot be settled from existing artifacts.** The forward hook records `_o.detach().mean().item()` (`training_engine.py:1430-1432`) — one scalar, **no sd**. And `trajectory_log_path` is set for **`copper_kestrel` only** (`config_hyperparameters.py:233`); `amber_kestrel` sets none, so **no control baseline exists.** Adding an sd to that hook is a one-line change with no effect on training dynamics, and it must land **before** any Tier-3 training arm, or that arm will be as unreadable as these two. Tracked as issue 7 of OUTCOME §9.

---

## 8. Decision rules

1. **E6c, then E6.** If **F5** fires, stop interpreting E6 and go to §9.
2. If **F1** fires → H1 is the cause. The remedy is a bounded local change to the recal distribution (`bn_recal_windows`, or sampling the recal windows from the inference population). **Propose it as an ADR, do not implement it inside this plan.**
3. If **F2** fires → H1 is dead. Proceed to Tier 2; if **F7** then fires, the answer is H3 and the honest report is that the gate is at its resolution limit.
4. If **F3**, **F4** or **F6** fires → **INCONCLUSIVE.** Nothing is asserted. Tier 3's seed replicate becomes the next proposal.
5. **Tier 2 runs regardless of Tier 1's verdict**, because R1 is worth having even if the cause is unknown — ⚠️ but **F7 governs how it is reported**, and a cosmetic gain must be called cosmetic.
6. ⛔ **No GPU time is requested or spent until Tier 1 has returned a verdict under rules 1–4.** Written down because this is the step that gets quietly skipped.

---

## 9. Tier 3 — gated, and not requested by this plan

Only these need hardware, and only if Tier 1 points at them. Each must carry the `act_ratio` guardrail.

- **T3a — a seed replicate.** The one measurement that would make any between-artifact difference attributable. ~12 h, ~$16/arm — ⚠️ **and that figure is inherited, not measured by this session.** The last plan's budget was wrong by ~4× for exactly this reason; it must be re-derived from a MANIFEST before being quoted to anyone.
- **T3b — a training arm** on whichever of `weight_decay` or the recal distribution Tier 1 implicates.
- **T3c — #558 / C-161**, retaining the gate posterior at h≥2. The only route to knowing whether the defect worsens across the 36 delivered horizons. A code change plus a re-run, not a cheap fix, and this plan's inability to reach h≥2 is the evidence for prioritising it.

---

## 10. Out of scope

- Opening the nine issues in OUTCOME §9 — **Simon's repos, Simon's call.**
- Any change to `models/`, any `run.sh`, or the committed `p2_gate_calibration.py` / `vehicle.py`.
- Implementing any remedy in production. Tier 2 produces a **measurement of what a remedy would buy**, not a shipped recalibration.
- The **denominator arm** (`reports/2026-10-15_loss_balance_prereg.md`) — untouched, different half of the loss, still unrun.
- **M-1 / C-162** — the balancer's learning rate never reaching its parameter group. Latent; both arms froze the balancer.
- `p2_gate_calibration.py`'s own `verdict` field. It stamps **both** arms "D2 — already calibrated", the arm passing by 0.0009 while under-stating by 0.1036. **Unusable as evidence here** (OUTCOME §9 issue 1).
- Generalising any h=1 result to h=36 (C7).

---

## 11. Definition of done, and the stop rules

**Done** when E6c, E6 and E1–E4 have run on both arms; every falsifier F1–F6 has been evaluated and the verdict recorded against §8's rules; Tier 2's R1–R5 have run under origin-fold cross-validation with F7–F12 evaluated; and an outcome document exists stating which candidate survived, or that none did.

**Stop rules.** No new questions mid-plan — a candidate not in §2 is a note for the next plan, not an addition to this one. No GPU without §8 rule 6. **A null at Tier 1 is INCONCLUSIVE, not negative** (C-307) — with n=1 run per arm and no seed replicate, failure to discriminate is a limit of the instrument, not evidence about the model. ⛔ **And no scalar leaves this plan without its parameter sweep** (§3.5).

---

## 12. Glossary

Every term checked against `reports/GLOSSARY.md`. **§5b was added for this work** (2026-10-18) and locks `the calibration level` (`a`), `the calibration slope` (`b`), `under-dispersed`, `over-dispersed` and `temperature scaling`, with the banned aliases — including that **`timid` (§5) is about the BODY and must not be used for the gate**, and that *"the gate is miscalibrated"* names neither axis and is drift. `the gate`, `the body`, `soft_gate`, `AP`, `crps-all`, `act_ratio` and `calibrated` are unchanged from their existing entries.
