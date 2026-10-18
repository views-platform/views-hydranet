# Pre-analysis plan — what compresses the gate's logits, and can a one-parameter fix repair it?

**Date:** 2026-10-18 · **Seat:** views-hydranet (this session) · **Status:** written before any Tier-1 measurement
**Predecessor (the fact this plan explains):** `reports/2026-10-16_pos_weight_arm_outcome/OUTCOME.md` @ rev 3. **Read its §2 and §8.4 first.**
**Artifacts:** `amber_kestrel` (`pos_weight = 2.0`, the delivered baseline) and `copper_kestrel` (`pos_weight = 1.0`, killed). Both on disk, both `truncated_nb`, `soft_gate`, 300 lessons.
**Cost:** Tier 1 and Tier 2 are **$0** and CPU-only. Tier 3 is **gated** and requests nothing until Tier 1 has named a cause.

**Authorship and inputs.** Written in this seat because pre-registration for views-hydranet experiments lives here — `reports/` holds 48 prior pre-analysis plans, every one committed from this repository. A draft written from the `views-models` seat is carried as **input, not as a base**: `inputs/views_models_draft_v3.md`, with this session's review of its v2 at `inputs/hydranet_review_of_draft_v2.md` and the superseded v2 with its errors intact at `inputs/superseded_draft_v2_with_errors.md`. The error trail is kept deliberately — a corrected document that hides what it corrected is worth less than one that shows it. **Candidates H1/H2/H3, experiment E6, the F1 rank-identity invariant and R5's framing originate in this seat; the under-dispersion finding itself, the remedy ladder R1–R5 (including R5's existence), the origin-fold cross-validation and the GPU gate originate in the `views-models` seat.** ⚠️ Corrected 2026-10-18 at that seat's request, against its own interest: R5 is theirs, but the argument that a per-arm fitted temperature makes its calibration comparison near-vacuous — and therefore its final decomposition framing — is this seat's correction of their overreach. See Amendment 1. Material rewritten rather than inherited, because a document carries its author's blind spots and the two seats have different ones.

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

⛔ **So the BatchNorm statistics that ship inside the artifact are estimated over 30 of the most activity-enriched 32×32 windows in the entire curriculum — and are then applied, frozen, to the whole quiet 360×720 global grid (64,818 land cells) at inference.**

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

---

# AMENDMENT 1 — E6 could not discriminate as specified. 2026-10-18, BEFORE any Tier-1 run.

**Source:** four attacks from the `views-models` seat, the division of labour working as designed. **All four are upheld. Two are upheld with their severity raised.** Every number below was reproduced in this seat before being accepted.

⛔ **The headline: F1 as written would have declared H1 the cause on a measurement artefact.** The plan is amended before anything ran, so nothing is retracted — but the design that was committed two commits ago was not sound, and it was my own lead candidate's test that was broken in my own candidate's favour for the **second** time (§3.3 was the first).

---

## A1 — ⛔ UPHELD. Gate-support mismatch alone moves `b` by 0.099 against F1's 0.10 threshold.

Reproduced independently from `p2_result_amber.npz`, same code, same bins, same forward pass; **the only thing varying is which part of the gate range `b` is fitted over:**

| population | cells | pos rate | bins | `b` | gap vs 1.146 |
|---|---|---|---|---|---|
| full global grid (published) | 842,634 | 0.00345 | 5 | **1.146** | — |
| `gate ≥ 0.001` | 88,806 | 0.03231 | 5 | **1.047** | **+0.099** |
| `gate ≥ 0.005` | 46,135 | 0.06034 | 5 | 1.117 | +0.029 |
| `gate ≥ 0.01` | 34,117 | 0.08005 | 4 | 1.170 | −0.024 |
| `gate ≥ 0.02` | 23,668 | 0.11201 | 4 | 1.178 | −0.032 |
| `gate ≥ 0.05` | 12,774 | 0.19156 | 3 | 1.086 | +0.060 |

**F1 fired on `b_grid − b_window ≥ +0.10`. A support shift alone produces +0.099** — one thousandth short, with no BatchNorm involved, from one array.

⛔ **And the family is NON-MONOTONE** (1.146 → 1.047 → 1.117 → 1.170 → 1.178 → 1.086; range **0.131**), so the confound cannot even be sign-corrected. The window set's pooled positive rate will land inside 0.03–0.11 and its cell count (≤ 30×1024 = 30,720) inside 23k–88k — **exactly the rows where `b` swings most.**

**Their three sub-points are all correct:**
- **E6c was structurally blind to it.** Random thinning preserves the gate distribution; it controls sample size and leaves support identical. §3.3's attenuation confound and this support confound are **different axes**, and E6c covered only the first.
- **F1's conjunction did not protect against it.** A support artefact is a stable feature of the data: it preserves its sign across bin resolutions and passes a random-subsample control cleanly. The triple condition filtered noise; this is not noise.
- ⛔ **This is the bin-support defect a THIRD time in this programme** (it voided the first reading at 5-vs-4 bins). **F12 aimed at the old axis** — "non-common bin support *across arms*" — and the new comparison is **across populations.** A guard written for the last instance did not cover the next one.

## A2 — ⛔ UPHELD, AND UNDERSTATED. 96.5% of cells in a 32×32 window have truncated context.

They flagged this as possibly overstated and asked this seat to settle it from the architecture. **It is understated.** Receptive field to `dec_conv4_head1_class`, computed layer by layer (`RF += (k−1)·jump`, `jump *= s`):

```
enc_conv0 k3s1 RF=3 j=1 · pool0 k2s2 RF=4 j=2 · enc_conv1 k3s1 RF=8 j=2
pool1 k2s2 RF=10 j=4 · bottleneck k3s1 RF=18 j=4 · upsample0 RF=18 j=2
dec_conv0_class k3s1 RF=22 j=2 · upsample1 RF=22 j=1 · dec_conv1_class k3s1 RF=24
dec_conv4_class k3s1 → RF = 26 px per side
```

| pass | cells with FULL context | **truncated** |
|---|---|---|
| **32×32 window** | 6×6 = 36 of 1,024 = 3.5% | ⛔ **96.5%** |
| **360×720 grid, ALL cells** | 334×694 = 231,796 of 259,200 = 89.4% | 10.6% |
| **360×720 grid, the 64,818 SCORED LAND cells** | 64,368 = 99.31% | ⛔ **0.69%** (450 cells) |

All convolutions are `padding=1`, so a truncated cell's receptive field is filled with **zeros, not neighbours**. And this is the U-Net path **alone**: the recurrent cell state is zero-initialised and built across the lead-in on that same 32×32 window, and M54 established the state carries a spatial map of where conflict will be — so its reach is truncated too. Effectively **every** cell in a window pass is computed from less context than its global-pass counterpart.

Their correction to this seat's mechanism statement is also accepted and is better than what the plan said: **BatchNorm is in eval mode with frozen per-channel statistics and does not re-estimate from the batch.** Running on hot windows does not "make the statistics correct"; it changes the input distribution so the activations entering BatchNorm resemble those the statistics were estimated from. H1's mechanism holds, stated that way.

### ⛔ But their proposed fix does not work, and the reason changes what H1 *is*

They proposed running both a crop and a window pass and pre-registering the difference as the confound. The crop cannot serve that role: **a crop of the global pass carries the global pass's activations, so BatchNorm mis-normalises it exactly as it mis-normalises the whole grid.** The crop does not test H1 — **it is the perfect control for A1's selection axis**, because it holds the computation fixed and varies only which cells are selected.

And the deeper point: **the activity mismatch and the context mismatch are inseparable, because the recalibration pass itself ran on 32×32 window passes** (`training_engine.py:1312-1314`, `window_dim: 32`, `batch_size=1`). The recal distribution is *defined by* window-passes. So H1 is not "the recal windows were too hot" — **H1 is "the recal distribution is not the inference distribution", and it differs on at least two axes**, activity and spatial context, the second of which neither seat had identified.

**Both axes have the same fix** (recalibrate on the inference distribution), so E6 does not need to separate them — it needs to separate **H1 from H2**. But this changes what a positive E6 licenses: it licenses *"recalibrate on the inference distribution"*, **not** *"the windows were too hot"*. Recorded so the remedy is not mis-specified later.

## A3 — ⛔ UPHELD. No power analysis, and E6c already computes the number the plan threw away.

The conclusive zones were `b_window ≤ 1.046` and `b_window > 1.10`; the inconclusive band is **0.054 wide**. The plan never stated the expected standard error on `b_window`, fitted on ~30k cells instead of 843k with far noisier per-bin rates.

**If `sd(b_window) ≳ 0.027` — half the band — F1/F2/F6 are decided by sampling noise.** And §4's tolerance justification was **category-wrong**: the 0.043 spread across published weightings is a **systematic range**, not a standard error, and cannot size a power threshold. That error is mine.

E6c's 10 replicates give `sd(b)` at window sample size directly, and the plan used them only for a bias check. **Adopted at zero extra cost.**

## A4 — ⛔ UPHELD (minor). F3's harness check was too loose.

`b_grid` at the 12 locked P2 edges is recomputed from the same `.npz` with the same code, so it must reproduce **1.146 exactly**, not merely exceed 1.10. A harness returning 1.12 is broken and F3 would not have fired.

---

## The amended design

**`b_crop` replaces `b_grid` as E6's reference.** Crop and window are **the same cell locations**, so A1's support confound is removed *by construction* rather than by matching.

| quantity | computation | cells | isolates |
|---|---|---|---|
| **`b_grid`** | 360×720 pass | all 842,634 | the published fact (1.146) |
| **`b_crop`** | 360×720 pass, **re-binned** to the window set's cell locations | window set | ⛔ **the SELECTION/support axis** — free, a re-binning of the `.npz` already on disk |
| **`b_window`** | 30 × 32×32 window passes | window set | selection **+** the recal-distribution match (activity **and** context) |

- ⛔ **H1's signal is `b_crop − b_window`** — same cells, same truth, differing only in how each cell was computed.
- **`b_grid − b_crop` is the measured selection confound**, reported as a number rather than assumed away.
- ⚠️ **Residual, stated not hidden:** binning is on the *predicted* gate, which differs between the two passes, so bin populations still differ on the same cells. Mitigated by common gate support and by reporting per-bin `n` for both. This is as clean as the instrument gets.

### Amended falsifiers — these REPLACE F1, F3, F5 and extend F12

| | fires if | then |
|---|---|---|
| **F1′** | `b_crop − b_window ≥ max(0.10, 3·sd(b_window))`, sign preserved across all three bin resolutions and all three window counts | ⛔ **H2 is DEAD as primary cause. H1 is the cause** — and the remedy is *"recalibrate on the inference distribution"*, not *"use cooler windows"* (A2). |
| **F2′** | `b_window > 1.10` **and** `\|b_crop − b_window\| ≤ max(0.05, 1.5·sd(b_window))` | ⛔ **H1 is DEAD.** H2 or H3. |
| **F3′** | `b_grid` at the 12 locked edges **≠ 1.146** to three decimals | ⛔ **VOID — the harness does not reproduce the published fact.** Stop and fix it. The old `≤ 1.10` clause is retained as a separate, looser catch. |
| **F5′** | **E6c:** `\|b_grid − b_crop\| ≥ 0.05` | ⛔ **the selection axis is live.** Not a void — `b_crop` is still the valid reference — but **`b_grid` must not appear in any E6 comparison**, and the magnitude is reported as the measured size of the confound that nearly broke F1. |
| **F5b** | ⛔ **POWER GATE. `sd(b)` across E6c's 10 replicates ≥ 0.027** | ⛔ **F6 fires BEFORE E6 is read.** The instrument cannot resolve the inconclusive band and no E6 result may be interpreted. Zero extra work — the replicates are already specified. |
| **F12′** | any comparison is reported on non-common bin support **across arms OR across populations** | **VOID.** Extended from the arms axis, which is the axis the last instance of this defect used. |

### Amended method

- **E6c runs first** and now reports **three** things: `b_crop`, `b_grid − b_crop`, and `sd(b)` across 10 replicates. Replicates are **stratified to the window set's gate histogram**, not random — random thinning cannot see the support axis (A1).
- **Both passes are run**: the crop (free) and the window pass. **Their difference is H1's signal**, not a confound to be removed.
- `n_windows ∈ {30, 60, 120}` and bin resolutions {12 locked, 40, 200} unchanged — but the sweep now applies to `b_crop − b_window`, and **F4 voids on a sign change in the difference**, not in either term alone.
- **Report the truncated-context fraction (96.5% vs 0.69% on the scored land cells) alongside every E6 number**, so no reader mistakes the window pass for the inference computation. ⚠️ Corrected in Amendment 2 — the grid is 360×720, not 180×180, and the scored population's truncation is 0.69%, not 26.8%.

### What this amendment does not fix

⛔ **E6 still cannot separate H1's two axes** — activity mismatch and context mismatch — because the recalibration pass differs from inference in both. **That is a finding about the harness, not a gap in this plan**, and it is the strongest argument for the one-line fix: make the recal pass sample the inference distribution, at which point both axes close together and the question does not need separating.

### Pattern, recorded

Three attacks in this programme have now landed on **this seat's own decisive test for this seat's own lead candidate** — §3.3 (sample size), A1 (support), A2 (context). Each was found by the other seat or by applying the sweep rule to my own side. **An author cannot test what they did not imagine, and the author's lead candidate is the hypothesis they imagine least critically.** The `falsify` guard-mode rule — the auditor may not be the author — is doing work here that no amount of care in a single seat would have done.

---

# AMENDMENT 2 — H4 added as a fourth candidate. It was already on record in this repo. 2026-10-18, BEFORE any Tier-1 run.

**Source:** three further attacks from the `views-models` seat. **All three upheld.** One of them — H4 — changes what a positive E6 **licenses**, which is the failure mode F1′ was written to prevent, one level further out than Amendment 1 reached.

---

## ⛔ H4 — TRAIN/INFERENCE GEOMETRY MISMATCH. **Co-lead, and the one that changes the remedy.**

**Verified premise.** Training runs exclusively on **32×32 crops** (`window_dim: 32`, `windows_per_lesson: 3`, both arms byte-identical). Inference runs **one untiled pass over the full grid** (`hydranet_inference.py:1172`, `_, seq_len, _, H, W = full_tensor.shape`; no tiling anywhere on that path — the only matches for `tile` in the file are the word *quan**tile***). **The network is evaluated on an input geometry it never saw once in training.**

### ⛔ It was identified in this repository six weeks ago, verified in code, and neither seat found it

`reports/2026-09-05_direct_multi_horizon_dossier/02b_design_review_rulings.md:20`, committed **`968b788`, 2026-09-06**, under the heading *"Architecture facts the design assumed wrongly — all verified in code"*:

> **"Training crops are 32×32 (`window_dim`), deployment 180×180. A K=36 full-res forecaster reaches 36 cells — wider than the training crop; long-horizon dynamics would be learned from boundary."**

**Four rounds of candidate enumeration across two seats missed a candidate already written down, verified, and committed in our own `reports/`.** Recorded as a process finding in its own right: neither seat searched the repository's own record before enumerating. The candidate list was built from what we could each see from our seat and from the code we happened to open.

### The second limb, which that ruling also quantifies

The same document establishes that the four ConvLSTM cells are **parallel, not stacked**, at full resolution, with `pool0`/`pool1` **outside** the recurrence — so **the recurrent state propagates one grid cell per month.** Over a 36-month lead-in its spatial reach is **36 cells, wider than the 32-cell training crop.**

⛔ **So in training the state's spatial reach was clipped by the crop boundary on every single window. At inference it is not clipped at all.** That is a second, independent geometry mismatch, and unlike the padding limb it is not confined to a margin — it affects the state everywhere.

### Why it is not separable from H1 by the Amendment-1 design

H4 predicts the window pass (the trained geometry) is honest and the global pass (an unseen geometry) is compressed: **`b_crop − b_window > 0`, bit-for-bit the same signature as H1, same sign, same magnitude class.** The crop carries the global pass's computation under H4 too, so `b_crop` does not separate them.

**And the remedies are not the same size:**

| | remedy | cost |
|---|---|---|
| **H1** | re-estimate BatchNorm statistics on the inference distribution | one-line, bounded, cheap |
| **H4** | the training geometry does not match inference — train on larger windows, or infer in tiles | architectural, expensive, and it puts **the inference path** in question rather than one post-training step |

⛔ **F1′ as written fires on `b_crop − b_window ≥ threshold` and licenses "recalibrate on the inference distribution." Under H4 that fix changes little and we would have bought the lesson with a Tier-3 arm.**

⚠️ One honest qualification in H1's favour: the H1 fix is a **partial** H4 fix, because re-estimating BatchNorm statistics on the inference distribution also re-normalises full-context activations. So "H4 holds" does not make the cheap fix worthless — it makes it **incomplete**, and the distinction matters for what gets promised.

---

## A2 — ⛔ UPHELD. F2 could falsely kill H1, and Amendment 1 made this worse.

Amendment 1 redefined H1 as *"the recal distribution is not the inference distribution"* — a claim about **distribution match**. But the BatchNorm statistics come from curriculum lessons 0–29 inside training months **121–456**, and E6's windows are at **457–469**. **E6's windows are therefore not the recal distribution** — they are a different draw from a similar selection rule, on data the statistics never saw.

An imperfect distribution match produces `b_window > 1.10` with a small gap **while H1 is true.** F2′ read that as *"H1 is DEAD"*, converting a known limitation of the proxy into a negative result about the hypothesis. **That is my error and it was introduced by my own re-framing in Amendment 1.**

**F2″ replaces F2′:** the verdict is **"H1 UNSUPPORTED ON THIS PROXY — Tier 3 required to kill it."** The alternative, reproducing the recal months, confounds time with selection and is still refused.

## A3 — ⛔ UPHELD, and the correction is larger than they argued. The grid is 360×720.

`vehicle.py:23` — `GRID_ROWS, GRID_COLS = 360, 720`; the volume is `[1, T, 3, 360, 720]` (`:212`); 64,818 land cells of 259,200. **180×180 = 32,400 is fewer than the land-cell count, so the figure could never have been right.** It is the **africa** grid (`africa_me_legacy`, 13,110 cells — see `2026-10-12_underfire_diagnosis_prereg.md:211`), carried into a global-model claim. Corrected throughout this plan, in `OUTCOME.md` rev 3, and in project memory.

**And the relevant denominator is the scored population, not the grid:**

| population | truncated context |
|---|---|
| 32×32 window | ⛔ **96.5%** (988 of 1,024) |
| 360×720 grid, all cells | 10.6% — *their figure, correct for all cells* |
| 360×720 grid, **the 64,818 scored land cells** | ⛔ **0.69%** (450 cells) |

**The contrast A2's case rests on is 96.5% vs 0.69%, not 96.5% vs 26.8%.** Correcting it widens the gap by a further factor of ~39.

---

## The amended design

### ⛔ My own counter-proposal is DEAD, and I checked before offering it

I had intended to propose **E6e**: split the global grid into rim and interior cells and compare slopes, testing H4 without any window pass. **It is unpowered.** Land cells by distance to the grid boundary:

```
  0–3 px:    96 cells (0.15%)      13–30 px:  2,032 (3.13%)
  3–7 px:   132 cells (0.20%)      30–60 px: 12,861 (19.84%)
  7–13 px:  222 cells (0.34%)      60+ px:   49,475 (76.33%)
```

**450 truncated land cells at a 0.345% positive rate is ~1.6 expected positives.** No calibration slope can be fitted on that. E6e is not in this plan.

### E6d — ADOPTED as specified, and it is the only viable signature discriminator

Per cell on the **same** cells, compute `logit(p_crop) − logit(p_window)` and regress it on **distance-to-window-edge**. The mechanisms have different spatial signatures:

- **H1** is a frozen per-channel scale error at BatchNorm; aggregated to the head it is approximately a single global logit scale — **no reason to correlate with edge distance.**
- **H4** is a context/padding effect and **must concentrate by distance to the pass boundary**, since that is what sets how much of a cell's receptive field is zeros.

It has power where E6e does not, because the **window** pass is where truncation actually varies: edge distance spans 0–15 across a 32-wide patch and 96.5% of cells sit below the 13-px full-context margin. **Pre-registered as a PRECONDITION on reading F1′**, the way E6c gates E6.

### ⛔ E6f — NEW, and it is the definitive test: measure the REMEDY, not the signature

A signature test infers the cause; this tests the fix. **Re-estimate the BatchNorm running statistics on the inference distribution** — forward-only over the 360×720 grid at the lead-in months, `momentum=None`, exactly as `_recalibrate_bn` does but sampling the inference geometry — then re-score the gate and refit `b`.

| outcome | reading |
|---|---|
| `b → 1` within the power band | ⛔ **H1 is the binding cause and its cheap remedy WORKS.** H4 is at most secondary. This is the result that licenses the one-line fix — and it licenses it because the fix was *measured*, not inferred from a signature. |
| `b` unchanged | ⛔ **H1's remedy does not work.** Either H1 is dead or (per the qualification above) the geometry limb dominates. **H4 or H2 or H3.** |

**Cost:** CPU forward passes only — no gradients, no optimiser, no training. ⛔ **On a scratch copy of the artifact, never written back.** Re-estimating BatchNorm statistics changes the numeric posterior of every logged metric (C-112/C-119 comparability), so this is a diagnostic and must never be confused with a shipped change.

**F1‴ replaces F1′:** H1 may be declared the cause **only if E6f also shows `b → 1`.** A positive `b_crop − b_window` with E6f unchanged is **H4-or-H2, not H1** — which is exactly the mis-licensing this amendment exists to prevent.

---

## Pattern, updated

**Four attacks have now landed on this seat's own decisive test for this seat's own lead candidate** — §3.3 sample size, A1 support, A2 context, and now H4 remedy mis-licensing. Three came from the other seat; one from applying the sweep rule to my own side; and **one of them was sitting in this repository's own records, committed, verified, and never searched for.**

The lesson is not that the test was badly built. It is that **a decisive test for a hypothesis you hold is the single place an author's imagination is weakest**, and that no amount of care in one seat substitutes for another seat trying to break it. The `falsify` guard-mode rule — the auditor may not be the author — has now earned its place four times in two days.

⚠️ **And if H4 holds, the consequence should be stated rather than left implied:** the compression would be a defect in **the inference path** of this repository, not primarily a calibration problem — and the whole `pos_weight` programme, the temperature-scaling remedy, and two days of calibration work would be treating a symptom of a geometry mismatch.

---

# AMENDMENT 3 — the record search, done properly. E6's central confound and H4's second limb were both already measured here. 2026-10-18, BEFORE any Tier-1 run.

**Source:** a systematic prior-art search of `views-hydranet/reports/` by the `views-models` seat, after Amendment 2 recorded that one candidate had been missed. **It is not one candidate. It is five, and one of them is a committed tool.** Every claim below was verified in this seat against the committed file before acceptance.

⛔ **Two of the hits are not hypotheses. They are (i) the confound that dominates E6 as specified, measured at 142×, and (ii) an empirical refutation of a limb I added to H4 in Amendment 2 — a prediction this repository registered, ran and scored WRONG in August.**

---

## 1 — ⛔ UPHELD. E6 must judge the INTERIOR. This repo wrote the amendment and then measured the artefact.

`reports/2026-08-23_state_range_dossier/tools/capture_regimes.py:253-257`, committed:

> *"**AMENDMENT 3: judge the INTERIOR**, because a patch and the full grid **MUST** disagree at the patch edge (border cells have no neighbours the full grid has). **Pooling the two confounds an unavoidable boundary artifact with the geometry effect F3 tests for.**"*

And `07_experiment_log.md:54` carries the measurement of exactly that contrast:

| | pooled `rel_abs_diff` | **interior** `rel_abs_diff` | gap |
|---|---|---|---|
| seed 42 | ≈ 0.10 | **0.0006** | ⛔ **142×** |
| seed 43 | ≈ 0.10 | 0.0059 | 13× |

**E6 as specified pools the whole 32×32 window.** On this repository's one measurement of the patch-vs-grid contrast, the pooled quantity is up to **142×** the interior quantity. So `b_window` as written would be **predominantly a boundary artefact**, and F1‴/F2″ would attribute it to BatchNorm or use it to weaken H1. **The same amendment, for the same reason, on the same geometry, was already written down — and I re-derived the design without it.**

### Locked: E6 is interior-only, with the pooled value reported beside it

- **Margin = 13 cells**, read off the architecture as `RF/2` with RF = 26 (Amendment 1). The prior tool parameterises this the same way — `--border`, *"Read off the architecture (3×3 convs, two 2×2 pools …)"* (`capture_regimes.py:139-142`).
- **Interior = the central 6×6 of each 32×32 window**, 36 cells.
- **Power, computed rather than assumed:** 30 windows × 36 cells × 13 scored months = **14,040 cell-months**, ≈ **421–1,544 positives** at the 0.03–0.11 hot-window positive rate. Workable; 60 and 120 windows scale it 2× and 4×. ⛔ **F5b's power gate is computed on the INTERIOR set, not the pooled set.**
- **The pooled value is reported alongside, always**, so the gap is visible rather than assumed away — this repo's own number for it is 142×.

## 2 — ⛔ UPHELD. H4's recurrent limb is REFUTED, and I reproduced a prediction my own ledger scores as wrong.

Amendment 2 argued: the state propagates one cell per month, reaches 36 cells over the lead-in, wider than the 32-cell crop, so **unlike padding it is not confined to a margin**. That is, almost verbatim, AMENDMENT 4 of the August dossier (`capture_regimes.py:256-257`: *"the receptive field compounds ~12 cells PER TIMESTEP over 384 steps, so no part of a 32×32 patch stays boundary-free"*) — **registered before the run, and scored** (`07_experiment_log.md:60-67`):

> **"AMENDMENT 4 predicted F3 would fail. It passed, comfortably."** … **"The theoretical receptive field is not the effective one."** The recurrence is **contracting** — F2 measures `max|h|` **65.622 → 1.832 over 30 free steps, a 35.8× collapse** — *"so influence decays faster than it propagates."*

⛔ **A contracting recurrence does not carry boundary contamination 36 cells inward.** The limb is **withdrawn**. The theoretical receptive field was the wrong instrument, which is the same error AMENDMENT 4 made, and I made it again with the answer in the repository.

### What survives of H4, at its real strength

- ⛔ **The recurrent-reach limb: WITHDRAWN**, refuted at 0.0006 interior.
- **The padding limb: SURVIVES** — but E6 now excludes it by construction (item 1), so it is not what E6 measures.
- **The untested step: state agreement → gate slope.** The prior work scopes itself explicitly (`07_experiment_log.md:96`): *"2 seeds, one architecture, `sb`, **states only. No AP was measured and none may be inferred.**"* So **F3 does not license a claim about the calibration slope.**

**H4 is therefore WEAKENED, NOT DEAD: plausible for outputs, disconfirmed for interior states, strongest in the padding margin, and demoted from co-lead.** H1 and H2 return to being the co-leads.

⚠️ **And it rescues E6d.** That discriminator was proposed in fear that no boundary-free interior existed. F3 demonstrates one does, comfortably. E6d has a clean reference, for a reason neither seat derived.

## 3 — ⛔ UPHELD, and there is a FIFTH hit neither seat cited. I also found why we missed them.

Verified prior art in this repository for our candidate list:

| candidate | prior art | what it is |
|---|---|---|
| **C1/C2** | `2026-08-03_lockeddropout_calibration_dossier/05_analysis_plan.md:11` | A **LOCKED** pre-analysis plan on LockedDropout and posterior calibration: *"**posterior spread**, improving calibration if the current posterior is **too tight**"*, with PIT / coverage / sharpness endpoints. We derived the probit/Jensen argument from first principles over three rounds. |
| **H4** | `968b788` + AMENDMENT 3 (the confound) + F3 (the measurement) | items 1 and 2 above |
| **H3** | `2026-10-12_underfire_next_step.md:76`; `PREDICTION_LEDGER.md:125` (W-17) | *"concentration of gradient is not the same as capacity to fit those cells individually"*; only **11.29%** of 905,289 params are shared, the gate has its own ~400k decoder |
| — | `archived/2026-06-08_magnitude_calibration_dossier/`, `2026-07-20_distributional_head_dossier/` | two further calibration pre-analysis plans |
| ⛔ **the instrument** | **`reports/2026-06-20_gate_calibration_dossier/` + `scripts/gate_reliability.py`** | **see below** |

### ⛔ The fifth hit: the instrument already exists, committed, with its own pre-registration

`reports/2026-06-20_gate_calibration_dossier/05_analysis_plan.md` is a pre-registered **gate reliability** plan (C-147) whose primary instrument is *"**Reliability diagram** — binned (deciles) mean-π vs empirical onset frequency, with per-bin counts"*, with a falsifier for exactly our direction — *"**F-underconfident:** reliability curve **above** the diagonal … gate too timid"* — a logged outcome in `07_experiment_log.md`, and a committed tool: **`scripts/gate_reliability.py`, 11 KB, `eb48924`, 2026-06-20.**

**P2 reinvented that instrument.**

⚠️ **Scoping it honestly, so this does not become the mirror error.** That era's vehicles ran **`pos_weight = 10.0`**, pre-foundation, and reported mean π 0.012 against 0.0077 prevalence — miscalibrated **HIGH**. Its H1 and F-underconfident are about the curve lying **below or above the diagonal**: a **LEVEL** claim. ⛔ **The level/slope decomposition — `a` versus `b`, and that `pos_weight` can only move `a` — is genuinely new, and so is the under-dispersion finding.** What already existed is **the instrument and the question**, not the answer.

### ⛔ The mechanism, which is sharper than "we did not search"

`MEMORY.md` — the index loaded at the start of every session in this repo — files `project_t0_calibration_failure.md`, `project_gate_resolution_finding.md` and `project_proper_score_gate_finding.md` under:

> *"Superseded/history (folded into the foundation) … (zero-smear + timid-body threads)"*

**The prior art was not merely unsearched. It was actively filtered out by a marker that said not to look.** And the marker was correct about the *conclusions* — those threads were superseded — while **silently retiring the instruments with them.** `gate_reliability.py` is not superseded; it is a working binned reliability diagram with bootstrap CIs and join guards.

### Locked controls — their proposal, plus the two the mechanism requires

1. ⛔ **Candidate enumeration is not closed until `reports/` has been searched for prior art on each candidate, and the search terms and the hits are recorded in the plan.** Their proposal, adopted verbatim as a numbered step in §3 rather than as a lesson, *because a lesson will not fire*.
2. ⛔ **`scripts/` is searched too.** A reports-only search finds the dossier but not the tool, and the tool is the part that was rebuilt.
3. ⛔ **A "superseded" marker retires a CONCLUSION. It never retires a tool, an instrument, or a falsifier.** Anything marked superseded must still be searched, and this one must be written into the memory index itself or it will filter the next search the same way.

## 4 — E6f's three holes. All accepted; (a) is material.

**(a) ⛔ F1‴ must be an EQUIVALENCE test against 1.0, not a movement test.** Amendment 2 records, against its own framing, that the H1 fix is a *partial* H4 fix. So if H4 is binding and re-estimating BatchNorm partially repairs it, `b` moves substantially toward 1 and F1‴ names H1 falsely. With `sd ≈ 0.027`, **`b = 1.06` is improved, is 2.2 sd from 1.0, and is NOT H1 confirmed.** F1⁗ requires `|b − 1| ≤ 2·sd(b)` with the power band stated, **and** `b` indistinguishable from 1.0 by that interval — not merely closer than before.

**(b) The re-estimation months are now LOCKED to the training months, 121–456.** Leaving them unspecified was the same defect class as the original E6 — and the choice is not cosmetic: training months is a **one-off deployable change**, while re-estimating on the evaluation window makes the statistics **origin-dependent**, which is a different and larger remedy. **Two distinct fixes were hiding under "the inference distribution."** The plan claims the one-line change, so it locks the one-line change.

**(c) E6f is PAIRED on the same scratch pipeline.** Per C-112/C-119, re-estimating BatchNorm changes every logged metric's numeric posterior, so post-fix `b` is compared against `b` computed **on the same scratch copy before the fix** — never against the published 1.146. **F3′'s exact-reproduction precondition extends to E6f's own baseline.**

---

## Candidate standing after Amendment 3

| | candidate | standing |
|---|---|---|
| **H1** | BatchNorm recalibration distribution | ⛔ **CO-LEAD** |
| **H2** | `weight_decay: 0.1` on the class heads | ⛔ **CO-LEAD** |
| **H4** | train/inference geometry | **WEAKENED, demoted.** Recurrent limb withdrawn (refuted at 0.0006 interior); padding limb excluded from E6 by the interior rule; the state→slope step untested and explicitly unlicensed by F3's scope |
| **H3** | resolution limit | the null, carried, with prior art |
| **C1/C2** | MC-dropout chain | not independent causes (Amendment 1), with prior art |

## Pattern, final form

Five attacks have now landed on this seat's own decisive test for this seat's own lead candidate, and **three of the five were already answered in this repository's committed record.** One of them — the 142× boundary confound — is the single largest effect anyone has measured in this programme, and it was in a file whose docstring explains why E6 as specified cannot work.

⛔ **The lesson is not "search harder."** It is that **a conclusion marked superseded took its instruments and its confound measurements down with it**, and that four rounds of two-seat enumeration built the candidate list from what each seat could see rather than from what the project already knew. The three locked controls in item 3 are the response; they are mechanical, and they are in §3 as steps, not as advice.

---

# AMENDMENT 4 — E6 was unpowered, the bin rule was the binding constraint, and the window budget was imaginary. 2026-10-18, BEFORE any Tier-1 run.

**Source:** a power analysis from the `views-models` seat, pre-empting F5b's own computation. Reproduced in this seat to the digit. **Items 1 and 2 upheld. Item 3(a) adopted. Item 3(b) is CLOSED by the value only this seat could read — and it did not need to be open.**

---

## 1 — ⛔ UPHELD. At the interior sample the locked bin rule cannot fit at all.

Reproduced here on the global `.npz`, 20 replicates, locked P2 edges:

| sample | rule | bins | `b` | `sd(b)` |
|---|---|---|---|---|
| full grid | `n ≥ 1000` | 5 | 1.146 | — |
| **30 windows (14,040 cm)** | **`n ≥ 1000`** | — | ⛔ **NO FIT — fewer than 2 bins clear the threshold** | — |
| 30 windows | `n ≥ 200` | 2 | 1.121 | **0.207** |
| 120 windows | `n ≥ 50` | 5.5 | 1.138 | 0.051 |

**Amendment 3's 421–1,544 positives was the right number and the wrong constraint. BINS were binding.** At `n ≥ 200` a two-bin fit has zero residual degrees of freedom — a line through two points — with `sd = 0.207` against a **0.054** discrimination window.

⛔ **And `n ≥ 1000` was a free parameter carried over from a sample 60× larger and never re-examined.** Sweeping it moves `b` across 1.12–1.18 with `sd` 0.05–0.21. **My own §3.5 standing rule obliged me to sweep it and I did not.** Sixth instance of that defect class, and the first where the rule was already written in the same document.

## 2 — ⛔ UPHELD. Amendment 3's interior rule and F5b's power gate were jointly unsatisfiable.

Margin 13 on a 32×32 window keeps 36 of 1,024 cells — it costs **96.5% of the sample**, and a binned-slope estimator cannot survive that at the specified window counts. Two corrections, each individually right, in direct conflict. Not a specification slip.

## 3(a) — ⛔ ADOPTED. Fit the slope by maximum likelihood. It removes a free parameter.

`logit P(y=1) = a + b·logit(gate)`, one logistic regression over every cell, no binning. Reproduced here:

```
full grid, BIN-FREE:   a = +0.1584   b = 1.1875        (binned n>=1000 gave 1.146)

sd(b):            bin-free      binned
  30 windows       0.0761        0.207      2.7x better
  60 windows       0.0624        0.150
  120 windows      0.0528        0.081
```

It uses every cell rather than a handful of bin means, and ⛔ **it eliminates the bin-threshold free parameter entirely**, which the sweep rule otherwise obliges us to sweep and report.

⚠️ **Their caveat against their own proposal is upheld and locked:** the reference number changes. **F3′'s exact-reproduction precondition retargets to `b = 1.1875` bin-free**, the binned 1.146 remains the published fact, and **every population must be compared under the same estimator** or the switch itself becomes the effect.

## 3(b) — ⛔ CLOSED. F3 passed at border = 13. Margin 8 is NOT empirically licensed.

They could not read the value and correctly flagged it as the thing that decides whether E6 is cheap. Read here:

- `capture_regimes.py:139-145` — `--border`, **`default=13`**, help text: *"receptive-field radius ~12–13 … **NOT tuned**"*.
- `results/regimes_fortytwo.json:200` and `regimes_fortythree.json:200` — **`"border": 13`** in both seeds.

**F3's 0.0006 interior PASS was measured at margin 13.** It licenses nothing about margin 8.

⛔ **And there is a stronger reason the escape hatch was never available.** F3 measures agreement of **recurrent states**. The **gate's** receptive field is an *architectural fact*: with RF = 26, a cell within 13 px of the patch edge has zero padding **inside its receptive field by construction**. No border sweep of a state-agreement test can license a smaller margin for the gate output. The contamination at margin 8 might be small, but its size is exactly what is unknown — and assuming it is small is the error AMENDMENT 4 made in August.

## ⛔ 4 — THE ACTUAL WAY OUT: the window budget was imaginary. 300 windows passes the gate and is nearly free.

| windows | cell-months | `sd(b)` bin-free | power gate (< 0.027) |
|---|---|---|---|
| 30 | 14,040 | 0.0761 | fails |
| 60 | 28,080 | 0.0624 | fails |
| 120 | 56,160 | 0.0528 | fails |
| ⛔ **300** | **140,400** | **0.0218** | ✅ **PASSES** |

**And 300 window passes cost almost nothing.** A 32×32 window is 1,024 cells against the global grid's 259,200 — **1/253 of the area** — and the global pass is already run **four times** per arm for `D = 4`. So **300 window passes ≈ 1.2 global-pass-equivalents.**

⛔ **I misread a distribution-matching constant as a budget.** `n_windows = 30` came from `bn_recal_windows = windows_per_lesson × 10`, which exists to **match the recal selection rule**, not to cap cost. Matching the recal *count* was never required; matching its *selection rule* is. Nothing in the plan ever justified 30 as affordable-vs-not, and the whole "cheap or expensive" framing of item 3(b) rests on a constraint that does not exist.

### Locked: how to add windows without breaking the distribution match

⛔ **`get_lesson(w mod 30)` for `w in range(N)`, `N = 300`** — the hot-end thresholds are preserved exactly while spatial coverage increases 10×. **Extending `w` to `0..299` would walk the entire curriculum and destroy the match**, which is the obvious wrong way to do this and is forbidden here explicitly.

Window-count sweep becomes **{100, 200, 300}**, all three above the fit floor, with F4 still voiding on a sign change in `b_crop − b_window`.

## 5 — ⛔ E6d is promoted again: it MEASURES the margin instead of assuming it

E6d regresses `logit(p_crop) − logit(p_window)` on distance-to-window-edge. **That is the contamination-versus-margin curve for the gate itself** — the quantity 3(b) wanted and F3 cannot supply.

**Locked order: E6d runs FIRST and the margin is read off it**, as the smallest distance at which the crop–window difference is indistinguishable from its interior asymptote. Margin 13 remains the **pre-registered default** and the fallback if E6d is itself unpowered. This converts the margin from an architectural assumption into a measurement, using the discriminator the other seat proposed for a different purpose.

## Their caveat, upheld

Their `sd` figures come from random subsamples of the **global, cold** gate distribution. Hot windows have higher positive rates and will populate the upper range better, so the true `sd` is probably somewhat lower. **An upper bound on available power, not a measurement of it. E6c on the real window set is the authority and if it disagrees, it wins.** Recorded so the 300 figure is read as a floor, not a target.

## Running order, final

```
E6d  (margin calibration)  ->  E6c (selection confound + sd -> F5b power gate)
     ->  E6 (b_crop vs b_window, interior, bin-free, N=300)  ->  E6f (the remedy test, F1'''')
```

## Pattern

Six attacks now, and this one found that **two of my own corrections were jointly unsatisfiable** — the interior rule I locked in Amendment 3 and the power gate I adopted in Amendment 2. Neither was wrong. Together they were impossible, and I committed both without checking they could hold at once.

⚠️ **And the binding constraint turned out to be a number I had never questioned**, inherited from a function whose purpose was distribution matching rather than budgeting. Both seats then spent a round optimising around it — them proposing a smaller margin, me verifying it was unlicensed — before either of us asked whether the constraint was real.

---

# AMENDMENT 5 — the 300-window power claim is void. Overlapping draws are correlated observations. 2026-10-18, BEFORE any Tier-1 run.

**Source:** the `views-models` seat, attacking the figure it had itself supplied. **All three items upheld.** Item 1 withdraws the conclusion Amendment 4 rested on.

---

## 1 — ⛔ UPHELD. The 300 windows OVERLAP, so `sd(b)` is larger than either seat estimated, and "300 passes the gate" is VOID.

Verified in `views_hydranet/utils/volume_sampler.py`:

- `:44` — `self.rng = np.random.default_rng(seed)`, seeded **once** at construction.
- `:79-81` — `_select_anchor(activity, threshold, self.min_events, self.rng, self.config)`.
- `:84-85` — `r0 = np.clip(r_anc - self.rng.integers(0, dim), 0, h_max - dim)`, same for `c0`: **a ±32 jitter and nothing more.**
- `:137` — `qualified_count = len(np.argwhere(activity >= threshold))`, returned at `:144` *"for transparency"*.

So `w mod 30` draws **ten windows per threshold index**, each anchored in the set of cells with `activity ≥ threshold`, displaced by at most one window width. ⛔ **At the hot end of the cooling schedule the qualified set is SMALL by construction — that is what makes those indices hot.** Ten windows from a small anchor set with a ±32 jitter **overlap heavily.**

**(a) `300 × 36 × 13 = 140,400` counts overlapping interior cells repeatedly.** The honest figure is the **union** of the 300 interior 6×6 regions, which at hot-end thresholds may be a small fraction of the nominal count.

**(b) ⛔ And the power inference is void.** `sd(b) = 0.0218` was obtained by subsampling 140,400 **distinct** cells **uniformly at random** — the maximum-information case. Overlapping draws from a small anchor set are **positively correlated observations** carrying less independent information than the same number of distinct cells, so **the true `sd` at 300 windows EXCEEDS 0.0218 and may not clear the 0.027 gate at all.**

⛔ **Amendment 4's conclusion — "300 windows PASSES the power gate" — is WITHDRAWN.** It was an inference from an optimistic bound presented as an estimate. The other seat supplied the bound and has withdrawn it; this seat drew the conclusion and withdraws that.

### Locked — the power gate is evaluated on the realised sample, not the nominal one

Before E6 may be read, **E6c reports all three**:

1. **`qualified_count` at each of the 30 threshold indices** — already returned by the API, never used.
2. **The count of DISTINCT interior cell-months in the union** of the realised window set.
3. ⛔ **`sd(b)` bootstrapped over WINDOWS, not over cells** — the window is the sampling unit, so resampling cells understates the variance of a clustered design.

**F5b's power gate is evaluated on (3).** ⛔ **And the window count is raised until the UNION clears the floor, never until the nominal count does.** No window-count figure appears in this plan as sufficient; sufficiency is a measured property of the realised draw.

## 2 — ⛔ UPHELD. The 1/253 area argument is the `overhead → 0` limit. The number comes out of the plan.

Cost is not proportional to area, because **300 window passes run the full recurrent lead-in 300 times.** The timestep count is identical for a window pass and a global pass; only the per-timestep spatial work scales:

```
global pass:    T_steps x (overhead + 259,200 x c)
300 windows:  300 x T_steps x (overhead +  1,024 x c)
```

So "≈1.2 global-pass-equivalents" is the best case of a model that **ignores the dominant term at this tensor size** — at 32×32 on CPU, per-op dispatch plausibly dominates arithmetic, and we pay **300× the timestep evaluations** (~15,000 recurrent steps against ~49 for one global pass).

⛔ **The figure is struck from the plan.** The conclusion survives with margin — this is minutes of CPU, and the budget is not the constraint — but it is stated as *"minutes of CPU, not a budget constraint, **to be measured on first run**"* rather than as a ratio that will be cited.

⚠️ **One lever this seat should name rather than leave implicit:** `get_batch` already takes `batch_size`, so the windows can be drawn in batches instead of 300 sequential passes, which would collapse the dispatch overhead and make the area argument actually apply. **But batching changes the tensor shape, and `LockedDropout` caches masks by shape (`locked_dropout.py:66-73`), so it changes the realised dropout draw — and the recal pass it is matched against used `batch_size=1`.** It is therefore a decision to be locked, not an optimisation to assume. **Locked: `batch_size=1`, matching `_recalibrate_bn`.** Cost is not the binding constraint, so the match is worth more than the speed.

## 3 — ⛔ UPHELD (guard). The 300-draw sequence is now load-bearing and must be pinned.

`self.rng` is seeded once at construction, so the draw sequence is reproducible **only if nothing else consumes that generator in between** — and with `w mod 30` the sequence determines the sample.

**Locked:** record `np_seed` and **hash the realised 300 `(r0, c0, threshold)` tuples**, asserting the hash on any re-run. A reordering or an extra consumer is then detected rather than silently changing the sample. Same class as F3′'s exact-reproduction precondition, which is why it belongs beside it.

---

## What this does to E6's standing

E6 is **not yet shown to be powered at any window count.** That is the honest state, and it is a better state than Amendment 4 left it in, because the sufficiency criterion is now a measurement on the realised draw rather than a number inherited from a uniform-subsampling idealisation.

⛔ **If E6c's union is small enough that no affordable window count clears the gate, F6 fires and Tier 1 cannot discriminate H1 from H2.** That outcome is pre-committed here rather than discovered later, and it would make the seed replicate (T3a) the next step rather than a fallback.

## Pattern

Seven rounds. This one is the first where **the attacking seat withdrew its own number and the authoring seat withdrew the conclusion built on it** — and the error was the same in both directions: a quantity computed under an idealisation (uniform random subsampling) applied to a design that violates it (clustered, overlapping draws).

⚠️ **And it is the second time in this programme that a constraint went unexamined because it arrived as a number in a config.** First `n_windows = 30` read as a budget when it was a distribution-matching constant; now `140,400` read as a sample size when it was a nominal product. The sweep rule in §3.5 catches a scalar that varies with its own free parameter. **Neither of these was that — both were numbers whose *meaning* was assumed, not whose value was unstable.** That is a distinct failure mode and the controls in §3 do not cover it; stated here, unfixed, because neither seat has a mechanical control for it yet.
