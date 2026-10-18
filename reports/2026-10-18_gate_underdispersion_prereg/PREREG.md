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

---

# AMENDMENT 6 — a control for the provenance-mismatch class. ADOPTED with two amendments, one of which is required for its only load-bearing part to work. 2026-10-18.

**Source:** the `views-models` seat, answering the open failure mode Amendment 5 recorded as having no mechanical control. **The class is five instances, not two, and it is not confined to numbers.** Adopted into §3 as numbered steps, with two amendments from this seat and the authors' own limits preserved verbatim.

---

## The class, stated properly

Every instance: **a value or label used in a role other than the one that produced it. The value is correct; the provenance → use mapping is wrong.**

| # | value | setter's question | use's question |
|---|---|---|---|
| 1 | `SHIPPED_POS_WEIGHT = 2.0` | what did `violet_visitor` ship? | ⛔ what did THIS run use? |
| 2 | `verdict = "D2"` | which branch did the tolerance take? | is the gate calibrated? |
| 3 | `180×180` | what is the **africa** grid? | what is the **global** grid? |
| 4 | `n_windows = 30` | how many windows match recal? | what can we afford? |
| 5 | `140,400` | how many cell-months nominally? | how much independent information? |
| 6 | `"Superseded/history"` | are these **conclusions** retired? | is anything in these threads live? |

⛔ **Why §3.5's sweep rule cannot catch any of them: all six are perfectly stable.** 30 is exactly 30; 2.0 is exactly `violet_visitor`'s 2.0. The sweep rule tests whether a value moves across its own free parameter. **Stability was never the problem — meaning was.** A genuinely distinct failure mode, and instance 6 shows the class covers **labels**, not only numbers, which makes Amendment 3's control 3 one instance of the general rule rather than a separate rule.

### ⛔ Instance 1 is a live defect in this programme's committed evidence. Verified here.

`p2_gate_calibration.py:91` sets `SHIPPED_POS_WEIGHT = 2.0`, commented *"violet_visitor config_hyperparameters.py:84"*. At `:185` the D2 branch assigns `pw = SHIPPED_POS_WEIGHT`, and `:199` writes it to the `.npz` under the field name **`pos_weight`**. Measured from the committed arrays:

```
amber  (control, RAN at pos_weight 2.0):   npz records pos_weight=2.0  verdict=D2
copper (ARM,     RAN at pos_weight 1.0):   npz records pos_weight=2.0  verdict=D2   <-- WRONG
```

⛔ **The arm's own stored evidence misrecords the single variable the arm changed.** A future reader trusting the field name would conclude `copper_kestrel` ran at 2.0.

And there is a third layer: because `pw` is assigned **inside the verdict branch**, the field's meaning depends on which branch fired — a D1 outcome would have written `1.0`. So `pos_weight` in that `.npz` is neither "what the run used" nor "what `violet_visitor` shipped" but **"the value this branch happens to carry."**

⚠️ **No conclusion in `OUTCOME.md` is affected** — the `gate` and `truth` arrays are measurements and are sound, and the report states `copper`'s 1.0 from the config, not from the `.npz`. But the evidence is self-contradicting and it is now **issue 10** in OUTCOME §9.

---

## The control, as adopted

**(1) The provenance line, at point of use.** For every value or label that gates a decision, one line: the `file:line` that **SETS** it — not the line that reads it — and **the question its setter was answering.** If that question differs from the question now being asked, the value is **re-derived, not reused.** Six for six above, each mismatch visible in one line with no measurement.

**(2) ⛔ THE TEETH — the typed restriction.** Tag every gating value `measured` / `configured` / `nominal-derived`, then:

> **A value that was not MEASURED may never establish sufficiency.**

A nominal product cannot gate a power claim; a config default cannot establish a budget; a hardcoded constant cannot record a run's parameter; a branch label cannot stand as a verdict. Mechanical — it needs no insight into meaning, only a tag checked against the kind of claim being made.

**(3) The trigger, so it fires at all.** Every value appearing in **a falsifier, a gate, a threshold, or a sufficiency claim**. In a pre-registration that set is small and can be listed outright — *"and a plan that cannot list them does not know what it is conditioning on."*

**(4) ⛔ THE BIAS CHECK.** In every instance the assumed meaning was **the one convenient to the argument being made**: `30`-as-ceiling made the design look constrained, which justified a smaller-margin proposal; `140,400`-as-sample made it look powered, which a conclusion needed.

> **When a value's role is ambiguous, ask whether the reading you adopted made your own argument easier. If it did, re-derive it.**

Unlike an exhortation, it has a yes/no answer. **In all six instances the answer is yes.**

---

## ⛔ AMENDMENT A — (2) does NOT catch instance 5 as written, and 5 is the instance its authors supplied

They claim (2) catches 1, 2, 4 and 5. **It catches 1, 2 and 4. It misses 5**, and the miss is structural.

The sufficiency claim at Amendment 4 did not rest on `140,400`. It rested on **`sd(b) = 0.0218`** — and that number *was* measured: a real bootstrap over real subsamples of a real array. Tagged honestly, it is `measured`, and (2) would have **permitted** it to establish sufficiency. What was wrong was not its provenance but its **input**: it measured uniform subsampling of a design that is clustered.

**Required amendment, without which (2)'s teeth do not reach its own authors' instance:**

> ⛔ **Tags PROPAGATE. A measured quantity computed from a `configured` or `nominal-derived` input inherits the weaker tag.** `sd(b)` measured over subsamples of a nominal cell count is `nominal-derived`, not `measured`, and may not establish sufficiency.

That restores (2) to 1, 2, 4 **and** 5, and it keeps it mechanical — tag propagation is checkable without understanding what any value means.

## ⛔ AMENDMENT B — (3)'s trigger set misses instance 1, which is the only one that corrupted an artifact

`SHIPPED_POS_WEIGHT` appears in **no falsifier, gate, threshold or sufficiency claim.** It appears in a **stored output field**, and that is exactly where it does its damage: the harm is done to a *future reader* who trusts the field name, not to a decision made at the time.

> ⛔ **The trigger set also includes: any value written into a stored artifact under a named field.** The field name is a claim about the value's meaning, and it is the claim a later reader will rely on.

Instance 1 had survived two readings of `OUTCOME.md`, a full `/review-diff`, and this programme's own four-round evidence verification, because nothing ever asked what the field name was asserting.

---

## Limits — preserved as the authors stated them, because they are the honest part

- **(1), (3) and (4) are DISCLOSURE controls.** They fire only if the provenance line is written honestly. **(2), with Amendment A, is the only part with teeth.**
- It does not help where a value is so familiar it never registers as an input — which is why (3) **bounds** the trigger set rather than saying "be careful".
- (2) cannot reach labels, because a label is not tagged `measured`/`configured`. Instance 6 needs Amendment 3's control 3 as a separate rule.
- ⛔ **It caught none of the six prospectively. All six were found by the other seat, after the fact.** A control derived from six post-hoc catches, which has never fired prospectively, is **a hypothesis about process, not a demonstrated one.** It goes into §3 **labelled UNTESTED**, and the first prospective catch or miss should be recorded against it.

⚠️ **And the authors' disclosure applies to this seat too:** the control is proposed by the seat that supplied instance 5, amended by the seat that supplied instances 3 and 4, and instance 1 is in a harness this seat wrote and twice verified. **No seat here has clean hands, which is the strongest argument for a mechanical control over a careful one.**

---

# AMENDMENT 7 — Amendment A had no stopping condition, and instance 1's history is not what I wrote. 2026-10-18. Final amendment.

**Source:** the `views-models` seat. **Both items upheld. One corrects a false claim this seat committed one commit ago.**

---

## 1 — ⛔ UPHELD. Amendment A, as worded, forbids the measurement the plan depends on.

> *"A measured quantity computed from a `configured` or `nominal-derived` input inherits the weaker tag."*

**Every measured quantity in this plan has configured inputs** — bin edges, `np_seed`, `D = 4`, `LEADIN_START`. Applied literally, every measurement becomes `nominal-derived`, **no sufficiency claim is ever permitted, and E6c's own `sd(b)` — the measurement Amendment 5 made the authority — is forbidden by my own rule.** Either it gets ignored or it paralyses the plan. Neither is a control. **The rule ate itself and I did not notice.**

**Qualifier adopted, which keeps it mechanical:**

> ⛔ **Taint propagates only through an input that STANDS IN FOR the object the claim is about.**
> **Test: does the measurement's input differ from the population the claim concerns?**

| quantity | claim about | input | verdict |
|---|---|---|---|
| `sd(b) = 0.0218` | variance of the realised window draw | uniform subsamples of the global array | **differs → TAINT** |
| E6c's `sd(b)` | variance of the realised window draw | **the realised window draw** | same → no taint |
| bin edges, seed, `D` | not a substitute for any population — part of the estimator's definition | — | no taint |

One yes/no question, no understanding of meaning required. It restores (2) to catching instances 1, 2, 4 and 5 **without** forbidding the measurement the plan rests on.

## 2 — ⛔ UPHELD, AND IT CORRECTS A FALSE CLAIM I COMMITTED IN `0966fad`.

Amendment 6 and OUTCOME §9 issue 10 said instance 1 survived *"because nothing ever asked what the field name was asserting."* **False. Something did.** `evidence/EVIDENCE.md:17-20`, under its own heading:

> **"### TRAP — do not cite the `pos_weight` field inside the npz**
> `p2_result_amber.npz` and `p2_result_copper.npz` BOTH store `pos_weight = 2.0`. That field is the harness's hardcoded `SHIPPED_POS_WEIGHT` constant (violet_visitor's value), not a value read from the model. `p2_gate_calibration.py:185` assigns `pw = SHIPPED_POS_WEIGHT` under verdict D2. **It is meaningless for these two arms.** Cite the run.log lines above instead."

Verified present, plus further mentions at `:41` and `:155`. Written at evidence-staging by the `views-models` seat, which is why **no conclusion ever rested on it.** OUTCOME §9 issue 10 is corrected in the same commit as this amendment.

⛔ **So the failure is not detection. It is that a documented defect in a stored artifact became a CAVEAT instead of a FIX.** A different failure, with a better control:

> ⛔ **A caveat does not travel with the artifact.** A known-wrong stored field must be **fixed or renamed**, never annotated in a separate document. The `.npz` still says `pos_weight = 2.0`, **three files away** from the note that corrects it, so anyone who loads it without reading `EVIDENCE.md` gets the wrong answer.

### The mechanical form for stored artifacts is a NAMING obligation, not a disclosure

This supersedes Amendment B's provenance-line treatment for stored fields:

> ⛔ **A stored field's name must be true of its value under the WEAKEST reading a later reader could take.**

`pos_weight` asserts *"the pos_weight of this run"*. `shipped_pos_weight_reference` would have asserted nothing false, **and the defect could not have existed.**

⛔ **And this is where the third layer settles the argument.** Because `pw` is assigned *inside* the verdict branch, a D1 outcome writes `1.0` — so the field means *"whatever this branch carries"*, neither the run's value nor `violet_visitor`'s. **No provenance line about a branch-dependent field could be written truthfully in the first place**, which is why disclosure was never the right instrument here and naming is.

**One strengthening from this seat, since the branch dependence is the root:**

> **A stored field whose value depends on which control-flow branch produced it must encode that branch, or be split into separate fields.** A single name cannot be true of two values chosen by a condition the name does not mention.

**Remediation is small, concrete, and Simon's:** rename the field in the harness and regenerate, or drop it. Filed as OUTCOME §9 issue 10.

---

## Close of the attack phase

Eight rounds. **The last three improved the CONTROL, not the experiment** — and the other seat's judgement that this is the end of useful attack is correct and I accept it.

The experiment has been as good as it can get without data since Amendment 3. E6's honest state — **not shown to be powered at any window count, with F6 pre-committed if no affordable count clears the gate** — is a fact about the world, not a gap in the plan. Further rounds would refine a process rule neither seat has yet run, which is worth less than running E6d and E6c and finding out whether any of this is powered.

### What the attack phase produced, for the record

| | |
|---|---|
| rounds | 8 |
| amendments | 7 |
| claims withdrawn, this seat | 6 — including a power conclusion, an architectural limb refuted by our own ledger, a grid dimension, a tolerance justification, two jointly-unsatisfiable corrections, and a false claim about a defect's history |
| claims withdrawn, other seat | 4 — including its own headline statistic, its own power bound, and the limb it supplied to H4 |
| defect classes named | 3 — unstable scalars (§3.5), provenance mismatch (Amendment 6), caveat-instead-of-fix (here) |
| prior art found in this repo, unsearched | 5, including a committed instrument and the 142× confound that would have dominated E6 |
| controls adopted | 7, **all labelled UNTESTED** — none has fired prospectively |
| cost | **$0.** Nothing on a GPU. Sigma unmeasured, so E1's prediction is unspent. |

⚠️ **The honest summary: the experiment is no better than it was at Amendment 3, and the plan around it is much better.** Eight rounds of adversarial review found no reason to change what E6 measures — only six reasons the measurement as specified would not have meant what it claimed.

**Next act is execution, and it is not either seat's to start.** The first three measurements are free and have not been asked for; the issues in OUTCOME §9 are Simon's repos and Simon's call.

---

# AMENDMENT 8 — F6 RULING, and E6f dominated E6 all along. 2026-10-18, after E1/E2/alignment.

**Source:** the `views-models` seat executed E1, E2 and the E6d alignment check, then stopped before the window forward-pass build and asked this seat to rule on F6. **The ruling is below. It is neither of the two options offered.**

---

## 1 — ⛔ THE ANCHOR-SET MEASUREMENT IS CONFIRMED, AND I FOUND THE MECHANISM

Their measurement: `planner.get_lesson(w)` for `w ∈ 0..29` yields **4–5 qualified anchor cells** per stratum and only **6 distinct `(target, threshold)` pairs**, cycling five times. Verified in code and by derivation:

- `curriculum.py:102` — `subject = self.subjects[global_step_idx % len(self.subjects)]`. Six subjects (3 regression + 3 classification) ⇒ indices 0–29 cycle **six** subjects five times.
- `curriculum.py:85-93` — `ratio = max_ratio + b·step`, then ⛔ **`ratio = min(ratio, max_ratio × roof_ratio)`**. With `max_ratio 0.95`, `min_ratio 0.05`, `slope_ratio 0.75`, `roof_ratio 0.7`, `total_steps = 300 × 3 = 900`:

```
b = -0.001333 /step          roof clip = 0.95 x 0.7 = 0.665
ratio at steps 0,6,12,18,24,29  =  0.665 at every one
⛔ the clip RELEASES at step 214 = lesson 71 of 300
   -> steps 0..213 all carry the IDENTICAL ratio
distinct (target,threshold) pairs over indices 0..29  =  6
```

Their reported thresholds reproduce exactly: `183 / 0.665 = 275.2`, `55 / 0.665 = 82.7`, `105 / 0.665 = 157.9` as the three subject maxima, and `by_sb_best` carries the same 183 as `lr_sb_best` because both count non-zero months.

⛔ **So it is not merely "the hot end". The curriculum's intensity ratio is CLIPPED FLAT for the first 214 of 900 steps**, and `bn_recal_windows = 30` draws entirely from inside that flat region. The 30 is not 30 strata, or even 30 thresholds. **It is 6 strata of 4–5 eligible anchor cells.**

### ⛔ This is a PRIMARY FINDING, not a precondition check — and it lands on C-184

**The BatchNorm running statistics shipped inside both delivered artifacts were re-estimated over windows anchored on 4–5 cells per stratum, across 6 strata — a handful of conflict hotspots — and are then applied, frozen, to 64,818 land cells globally.**

That is not an activity-enriched sample. **It is a degenerate one.** H1's premise — *the recal distribution is not the inference distribution* — is now established **by direct measurement of the recal distribution**, far more strongly than E6 could ever have shown it. ⛔ **H1 no longer needs E6 to be plausible.**

And it qualifies C-184 itself: that fix was validated on flipping seed-bimodal eval collapse (*"6/6 bad seeds GOOD"*) and it **worked**. So C-184 repaired the collapse **while being structurally incapable of producing representative statistics** — 15 BatchNorm layers' running means and variances estimated from 30 forward passes over 6 strata. Both things are true, and the second was never checked because the first was the only endpoint. **Filed as OUTCOME §9 issue 11.**

## 2 — ⛔ THE F6 RULING: F6 DOES NOT FIRE

F6 is worded *"fires if E6 returns `+0.05 < b_grid − b_window < +0.10`, or any outcome F1–F5 do not cover."* **It is evaluated on E6's RESULT. E6 has not run.** And Amendment 5 made E6c — not this seat's judgement — the authority on sufficiency, explicitly: *"sufficiency is a measured property of the realised draw."*

⛔ **Ruling (ii) would be me overriding my own decision rule on a precondition that rule does not cover** — which is the "decide afterwards what we predicted" failure the whole falsifier structure exists to prevent. **F6 does not fire. The anchor-set measurement is not an E6 result.**

The executing seat's lean toward (i) is correct **by this plan's own structure**, and for the reason they gave: 4–5 anchors is a property of the anchor set, not of the realised cell union, and ±32 jitter over ~30 overlapping anchors does not cap the union at 4–5 cells.

## 3 — ⛔ BUT NEITHER OPTION IS THE RIGHT NEXT STEP. E6f DOMINATES E6, AND ALWAYS DID.

Every one of the eight attack rounds was about **E6's** confounds:

| round | E6 confound | does it touch E6f? |
|---|---|---|
| §3.3 | sample-size attenuation | **no** |
| A1 | gate-support mismatch (0.099 vs a 0.10 threshold) | **no** |
| A2 | truncated spatial context, 96.5% vs 0.69% | **no** |
| Am. 3 item 1 | patch-boundary artefact, 142× | **no** |
| Am. 4 | bin rule unfittable at the interior sample | **no** |
| Am. 5 | overlapping draws, correlated observations | **no** |
| now | 6 strata, 4–5 anchors | **no** |

**E6f re-estimates the BatchNorm statistics on the inference distribution and re-scores the SAME global-grid population.** It is a **paired test on identical cells**: no window geometry, no interior margin, no support matching, no anchor diversity, no binning asymmetry, no clustered sampling. **Not one of the seven confounds applies to it.**

⛔ **And Amendment 2 already made E6f REQUIRED to name H1** (F1‴, strengthened to the equivalence test F1⁗ in Amendment 6). **So E6 was never sufficient and E6f was always necessary.** Eight rounds of adversarial review hardened the experiment that was not on the critical path, while the one that was sat in the plan unattacked since Amendment 2.

**This is the single largest structural error in the programme and it is mine** — I added E6f as a supplement to my own lead candidate's signature test, and never noticed it superseded it.

### Locked ordering, replacing Amendment 4's

```
1. E6c       free, runs anyway — F5'/F5b are live and it bounds what E6 can contribute
2. ⛔ E6f    THE DISCRIMINATOR. Paired, same population, immune to all seven confounds
3. E6/E6d    demoted to CORROBORATION. Run only if E6c shows it is powered; a null here
             no longer blocks a verdict, because E6f does not depend on it
```

⚠️ **E6f's cost is UNMEASURED and must not be claimed.** It needs forward-only passes over the 360×720 grid across the training months to accumulate statistics — more than E6c, no gradients, no GPU. **Measure it on the first pass and stop if it exceeds minutes**, per Amendment 5's struck ratio: this plan does not get to assert a cost again.

## 4 — E1 AND E2, RECORDED AGAINST THEIR PRE-REGISTRATIONS

**E1 — `sigma` BELOW the pre-registered floor in both arms, every summary, family does not cross it:**

| arm | floor | median σ | implied `T` | published `b` |
|---|---|---|---|---|
| amber | ≥ 0.85 | **0.459** | 1.040 | 1.146 |
| copper | ≥ 1.20 | **0.594** | 1.067 | 1.419 |

**E2 — removing the Jensen average barely moves the slope:**

| arm | `mean(sigmoid)` | `sigmoid(mean logit)` | share of excess explained |
|---|---|---|---|
| amber | 1.1875 | 1.1713 | **8.6%** |
| copper | 1.3596 | 1.3335 | **7.2%** |

⛔ **E2's pre-registered SECOND outcome fires in both arms: the compression is in the NETWORK, not the reporting path.** R2 is not dropped (`b < 1` did not fire) but recovers under a tenth of the defect and is **quantitatively pointless as a remedy**.

✅ **And Amendment 1's ATTACK 1 is confirmed with a number.** The derivation said C1's Jensen compression is absorbed at the loss's stationary point and contributes nothing independent. **Measured contribution: ~8%.** The residual the remaining candidates must explain is **1.171 (amber) / 1.334 (copper)**.

⚠️ **E1 was labelled a MEASUREMENT, not a test** (§4), so σ landing below its floor is **not** a falsification of anything — it is the first honest number on the axis, and the bands were always derived by inverting published `b`. Recorded as such.

**Alignment proven by content, not by offset arithmetic:** full-block equality over 300 timesteps × 11 channels gives a unique match at `(r0, c0) = (92, 496)`, and single-slice matching would have been insufficient because channels 3 and 4 are constant-valued (57 spurious candidates on one slice). **The geometry is sound; plumbing is not what blocks E6.**

---

# AMENDMENT 9 — E6c has run. E6 is not worth running. ✅ And a control fired prospectively for the first time — against its author. 2026-10-18.

**Source:** the `views-models` seat executed E6c on the realised draw. **This is the first Tier-1 sufficiency measurement, and it settles E6's status by measurement rather than argument.**

---

## 1 — E6c, locked rule `get_lesson(w mod 30)`, interior margin 13

| N win | nominal | UNION | dup | cell-months | pos | `b_crop` | `sd(b)` window-boot | gate |
|---|---|---|---|---|---|---|---|---|
| 30 | 312 | 294 | 1.1× | 3,822 | 22 | 1.0380 | 0.0666 | **FAIL** |
| 60 | 556 | 528 | 1.1× | 6,864 | 125 | 1.1160 | 0.1269 | **FAIL** |
| 120 | 1,007 | 862 | 1.2× | 11,206 | 131 | 1.1677 | 0.1002 | **FAIL** |
| 300 | 2,421 | 1,838 | 1.3× | **23,894** | **314** | 1.1730 | **0.0470** | **FAIL** |

`b_grid` (bin-free, full population) = **1.1875**. **F5′ selection confound = `b_grid − b_crop(300)` = +0.0145.**

⛔ **E6 fails the power gate at every count** — 0.0470 against 0.027 at 300 windows, still 1.7× over. And the `sd` sequence is **non-monotone** (0.0666 / 0.1269 / 0.1002 / 0.0470), so extrapolating to a sufficient count is unsound **by §3.5's own rule**, and no count is offered. **No affordable window count is demonstrated to clear the gate.**

## 2 — ⛔ MY AMENDMENT 5 CELL-MONTH FIGURE WAS 6× TOO HIGH. The binding term is LAND FRACTION.

Amendment 5 argued the honest figure is the union, then **printed `300 × 36 × 13 = 140,400` in its own table as "cell-months".** Measured: `300 × 36 = 10,800` nominal interior cells, of which only **2,421 are land (22.4%)** and **1,838 are distinct** ⇒ **23,894 cell-months and 314 positives**, not 140,400. **The ±32 jitter displaces interiors over ocean.**

The 22.4% land fraction is itself a consistency check: the grid is 64,818 land of 259,200 = **25.0%** land, so a semi-randomly jittered window set landing at 22.4% is exactly what it should be. **The number was always recoverable and I never recovered it.**

### ⛔ The sharper lesson: a number can be corrected once and remain wrong for a second reason

`140,400` was corrected in Amendment 5 **for independence** — overlapping draws carry less information. It remained wrong **for land fraction**, an entirely separate defect, and the correction gave no hint of it. **Fixing the reason a number is wrong does not make the number right.** This is the seventh instance of the Amendment 6 class and the second on this single figure.

## 3 — ✅ FIRST PROSPECTIVE FIRE OF ANY CONTROL, AND IT FIRED AGAINST ITS AUTHOR

Amendment 6 labelled all seven adopted controls **UNTESTED**, noting none had ever fired prospectively and that *"a control derived from post-hoc catches is a hypothesis about process, not a demonstrated one."*

**Amendment 5's control fired.** It locked: *"E6c reports … the count of DISTINCT interior cell-months in the union"* and *"the window count is raised until the UNION clears the floor, never until the nominal count does."* **E6c measured instead of multiplying, and caught a 6× error in the amendment that created the control.**

⛔ **Recorded against the UNTESTED label: one of seven controls has now fired prospectively, and it caught its own author's arithmetic.** That is the strongest available evidence that making the measurement the authority — rather than the careful reasoning of whoever wrote the plan — is the part that works. It is also exactly the clean-hands argument Amendment 6 made, now demonstrated rather than asserted.

## 4 — ⛔ BOTH OF THE OTHER SEAT'S OWN ATTACKS OVER-ESTIMATED THEIR REALISED MAGNITUDE

Recorded next to the attacks, at that seat's request:

- **A1 (support mismatch).** Worst case was that support shift alone moves `b` by **0.099**, against F1's 0.10 threshold — the number that forced the whole `b_crop` redesign. **Realised: +0.0145**, about **7× smaller**. `b_crop(300) = 1.1730` against `b_grid = 1.1875` — the cells these windows select are **essentially ordinary** in their calibration slope. **A correct identification of a real axis, and a large over-estimate of its realised size.**
- **Amendment 5 (clustered overlap).** Predicted heavy clustering from the 4–5-cell anchor set. **Realised duplication: 1.1–1.3×.** The ±32 jitter spreads the draws far more than the anchor count suggested. **Again a real axis, over-weighted.**

⚠️ **This does not retire either redesign.** `b_crop` is still the correct reference and the union is still the correct denominator — both are now *known* to be small rather than *assumed* to be. But it is the honest record: **eight rounds hardened E6 against two confounds that measured at a seventh and a quarter of their predicted size, while the defect that actually killed it — 314 positives — was a term neither seat had named.**

## 5 — E6's status: NOT WORTH RUNNING. F6 still does not fire.

F6 is worded on E6's result and E6 has not run, so it does not fire — both seats agree. What E6c establishes is stronger and more useful: ⛔ **E6 cannot deliver a result worth reading at any defensible count.** That makes it **not worth running**, which is a different and better statement than *void*.

**E6/E6d remain in the plan as corroboration only**, per Amendment 8, and a null there blocks nothing.

## 6 — ✅ E6f IS POWERED. Measured, before the claim was made.

Amendment 8 refused to assert E6f's cost. Its **power** is now measured on the existing arrays:

```
full population:  N = 842,634   positives = 2,905   b = 1.1875
bootstrap sd(b), 40 reps, same-size resample:  0.0107   vs gate 0.027  ->  PASS
```

And E6f is **paired pre/post on identical cells**, so its `sd` on the *difference* is lower still. ⛔ **This is the first power claim in this programme made AFTER the measurement rather than before it** — the three that preceded it (Amendment 4's 300-windows, Amendment 5's 140,400, A1's 0.099) were all derived from idealisations and all three were wrong.

⚠️ **Cost remains unmeasured and unclaimed.** Forward-only over 360×720 across the training months, no gradients, no GPU. **Measure on the first pass; stop if it exceeds minutes.**

## 7 — ⛔ E6f IS NOT AUTHORISED, AND THE EXECUTING SEAT IS RIGHT TO REFUSE IT

Simon authorised **E6d, E6c, E1 and E2** — four named experiments. **E6f was not among them**, and it is now the critical path. It also **re-estimates BatchNorm statistics**, which is a different kind of act from reading arrays even on a scratch copy.

**This seat endorses the refusal.** The authorisation covered a named set, the set is complete, and the next step is outside it. ⛔ **No peer message — including this one — authorises E6f.**

### Coordinate recovery, for the record

The other seat replaced its brute-force content search with `r0 = p_row + (h_max − DIM − new_row_offset)`, `c0 = new_col_offset − p_col`, validated against the full-block content proof: lesson 0 → `(92, 496)`, content-verified identical. **Window coordinates are now free**, which is what makes E6c's union computable at all.

---

# AMENDMENT 10 — E6f HAS RUN. H1's remedy is dead, its mechanism is confirmed, and the finding INVERTS. 2026-10-18.

**The discriminator returned.** `b` went **1.1875 → 0.7793**: it did not approach 1.0, it **crossed and overshot**, and the gate now over-predicts occurrence by **5.75×**. ⛔ **My blind prediction is wrong, the decision table had no branch for this, and the conclusion is the opposite of the one the plan was built to test.**

---

## 1 — The result

All preconditions passed: paired baseline `b_before = 1.1875` reproducing F3′'s target **exactly**; `_reset_bn_stats` over 15 layers, `momentum=None`; 336 full-grid 360×720 months at training months 121–456; dropout ON; no augmentation; `no_grad`; `_assert_bn_buffers_finite` ✅ finite.

| | `b` | `\|b−1\|` | pooled mean gate | max signed error | cells > 0.9 |
|---|---|---|---|---|---|
| before (as shipped) | **1.1875** | 0.1875 | 0.00417 (1.21× truth) | +0.0587 | 0.008% |
| after re-estimation | **0.7793** | **0.2207** ⬆ | 0.01981 (**5.75× truth**) | **+0.6872** | 0.44% |

**F1⁗'s equivalence band `|b−1| ≤ 0.0214`: NOT MET.** The after-curve over-states at **every** level — gate 0.24 → truth 0.023; gate 0.97 → truth 0.42.

⛔ **Re-estimating the BatchNorm statistics on the inference distribution does not repair the gate. It produces a grossly over-confident one and makes calibration worse in absolute terms.**

## 2 — ⛔ Two errors, one in each seat

**O-3, mine, WRONG — and it was the headline blind prediction of this programme.** I predicted `b` would land at 1.10–1.16, reasoning that `weight_decay: 0.1` pins the trained optimum and BatchNorm re-estimation *"cannot undo what the weights are."* **It moved 2.2× the distance to 1.0 and past it.** Changing nothing but 15 layers' running means and variances swung the slope by **0.41 — more than twice the entire defect.** BatchNorm statistics are the **dominant** control on this gate's calibration and I allowed them almost none. Logged as **W-25**.

**F1⁗'s decision table was NON-EXHAUSTIVE, and I signed it off.** Three outcomes were pre-registered — `b→1`, `b≈1.19`, `b≈1.10` — and **none covers crossing 1.0**. The script fell through to `else` and printed *"PARTWAY — improved but NOT equivalent"*, which is false: `|b−1|` got **worse**. ⚠️ The other seat wrote it, but **I attacked it in Amendment 6, strengthened it to an equivalence test, and approved the three branches.** And the symmetry is exact: I criticised E2's table for being **exhaustive** and therefore unfalsifiable, then missed that this one was the **opposite** failure. Logged as **W-26**.

## 3 — H1: remedy DEAD, mechanism CONFIRMED

- ⛔ **The remedy is dead — actively harmful, not inconclusive.** The one-line change this plan has treated as the cheap fix destroys the gate. **And it would blow F11's firing guardrail**: a mean gate at 5.75× the true rate would wreck `act_ratio` and the composed forecast. F11's trap, in the direction we never pre-registered.
- ✅ **The mechanism is confirmed with enormous leverage** — far stronger evidence than E6 could ever have produced, which retrospectively justifies Amendment 8's ruling that E6f dominated E6.

## 4 — ⛔ THE INVERSION: the criterion was wrong, and C-184 is arguably right BY DESIGN

The other seat's mechanism is correct and self-consistent: hot windows have **high** activation variance, so eval-mode BatchNorm divides by a large σ and **shrinks** logits (under-confident, `b > 1`); the quiet global grid has **small** variance, so it divides by a small σ and **amplifies** them (over-confident, `b < 1`). It explains the sign of **both** endpoints.

They read this as *"the degenerate recal sample is accidentally closer to honest than its own stated intent."* ⛔ **I think it is not accidental, and the reason inverts H1.**

**`_recalibrate_bn` calls `train(ctx, batch[0], …)` (`training_engine.py:1321`), which puts the model in TRAIN mode — so BatchNorm normalises each window by ITS OWN batch statistics while accumulating the running averages.** During actual training, the same thing happens on the same 32×32 activity-seeking crops. **So the head's weights were trained against activations standardised by hot-window moments.**

⛔ **The criterion for BatchNorm eval statistics is therefore not "match the inference distribution". It is "give the head the input scale it was trained against."** By that criterion:

- the shipped hot-window recal **reproduces the training condition** → `b = 1.19`;
- the global-grid recal **does not** → `b = 0.78`.

**C-184's sample is not accidentally better. It is closer to the condition BatchNorm eval statistics exist to reproduce.** H1 as written — *"the recal distribution is not the inference distribution, therefore mismatch"* — had the criterion backwards, and E6f is the measurement that shows it.

### ⛔ Which means the root cause is the TRAINING GEOMETRY, and H1 fuses into H4

BatchNorm computes per-channel moments over `(N, H, W)`. On a 32×32 activity-seeking crop the spatial population is **hot**: a conflict cell sits near the batch mean. On the 360×720 grid the population is **mostly empty**: the same cell sits far above the mean. **"Standardised" means two different things, and the head only ever saw the first.**

⛔ **So no choice of eval statistics can fix this, because the two populations have genuinely different conditional distributions — and the cause is that training crops are activity-seeking.** H1's mechanism is real, its remedy is dead, and its **root** is H4's selection limb, which Amendment 3 demoted on the strength of a states-only measurement that explicitly said *"No AP was measured and none may be inferred."* **H4 returns, as the cause of H1.**

## 5 — ⛔ THE GOODHART TRAP, forbidden before anyone proposes it

`b` crossed 1.0, so **by continuity there exists a recal distribution giving `b = 1.000` exactly** — some blend between hot windows and the global grid. **Tuning the recal distribution to hit `b = 1` is the "honest `pos_weight`" error one level up:** a nuisance parameter with no principled value, fitted to the endpoint it is judged on, which would also move the firing rate as a side effect.

⛔ **Pre-emptively forbidden by this plan.** It is the obvious next idea, it is wrong for exactly the reason OUTCOME §8.1 is wrong, and it would be the eighth instance of a knob tuned to a target it cannot principledly aim.

## 6 — R1 is STRENGTHENED, and that was not expected

Neither endpoint of the BatchNorm axis is calibrated, the axis overshoots violently, and its gain cannot be aimed without tuning to the endpoint. **Temperature scaling sets the slope to 1.0 by construction, with one parameter, without touching the model, the weights, or the firing rate's dependence on them.**

> ⛔ **The expensive axis is unaimable. The free axis is exact.**

## 7 — H2 is neither confirmed nor refuted, and my reasoning for it is damaged

Nothing here tests `weight_decay`. But **the premise I argued H2 from — that the weights pin the slope — is now in doubt**, because BatchNorm statistics moved it 0.41 with the weights untouched. H2 stays live and its argument is weaker than Amendment 8 presented it.

## 8 — Candidate standing

| | candidate | standing after E6f |
|---|---|---|
| **H1** | BatchNorm recal distribution | **mechanism CONFIRMED (0.41 leverage); remedy DEAD; criterion INVERTED** — the shipped sample is closer to the training condition, not further from honest |
| **H4** | train/inference geometry | ⛔ **RETURNS as H1's root cause.** Activity-seeking 32×32 crops make BatchNorm's "normal" activity-conditional. Previously demoted on a states-only measurement that forbade exactly this inference |
| **H2** | `weight_decay: 0.1` | live; its supporting argument damaged |
| **H3** | resolution limit | live, untested |
| **R1** | temperature scaling | **strengthened** — the only axis that sets the slope exactly without an unaimable gain |

⛔ **Nothing here authorises Tier 3.** A training arm — the only thing that could test H4's root or H2 — is Simon's, and this result does not change that.

---

# AMENDMENT 11 — I withdraw half of Amendment 10. No fixed BatchNorm statistic is correct, including the "principled" third one. 2026-10-18.

**Source:** the `views-models` seat, measuring the training schedule. **Their arithmetic is verified exactly. It refutes the second half of Amendment 10's inversion — and the argument that refutes it also kills the third option they proposed.**

---

## 1 — Their schedule measurement, verified

```
roof 0.665, floor 0.05, b = -0.001333/step, total 900 steps
  steps at roof 0.665 :  214  (23.8%)   <- the recal draw (0..29) lives entirely here
  steps declining     :  461  (51.2%)
  steps at floor 0.05 :  225  (25.0%)
  MEAN ratio over training = 0.3540    recal sample = 0.665    ⛔ 1.88x hotter
  median 0.3507;  50.8% of training runs below ratio 0.36
```

Every figure reproduces. **The head's final weights were fitted predominantly on much cooler crops than the recalibration samples, and a quarter of training ran at the floor.**

## 2 — ⛔ WITHDRAWN: "C-184 is arguably right by design"

Amendment 10 §4 made two claims. **The first stands, the second is wrong.**

- ✅ **Stands:** `_recalibrate_bn`'s docstring states the wrong criterion (*"so eval-mode matches the true activation distribution"*), and H1 was formulated against it. The other seat adds that this is the **textbook** criterion — BatchNorm running statistics exist to estimate the **training** distribution's moments so eval reproduces the input scale the downstream weights were fitted against — which makes the point independent of E6f. **Adopted into OUTCOME issue 12 as the first-principles form.**
- ⛔ **WITHDRAWN:** *"the shipped hot-window sample is closer to the condition BatchNorm eval statistics exist to reproduce"* and *"C-184 is arguably right by design."* **At 1.88× the mean training ratio, the shipped sample is not the training distribution either.** It is closer than the global grid and still badly mis-specified. **Right as a direction, wrong as a verdict** — their words, and they are correct.

## 3 — ⛔ AND THE ARGUMENT THAT KILLS THE THIRD OPTION TOO

They propose re-estimating over the full curriculum schedule (`get_lesson(step)` for `step ∈ 0..899`), arguing it is principled because the distribution is fully specified before measurement and is not fitted to the endpoint. **The procedural distinction is real and their self-protection is right. But the option is not principled, and the reason is the same one that explains both endpoints.**

⛔ **In train mode BatchNorm standardises PER BATCH.** `training_engine.py:1027` — `ctx.model.train()` — so every forward during training normalises that window by **its own** moments. **The head was therefore never trained against any fixed set of moments at all.** It was trained against activations standardised *within each batch*, on batches that were **activity-selected**.

Three consequences:

1. **There is no "the training distribution's moments" the head was fitted against.** The criterion Amendment 10 stated — *"give the head the input scale it was trained against"* — was **underspecified**, and correctly specified it is **unsatisfiable by any fixed statistic.**
2. **Averaging moments across a heterogeneous schedule reproduces per-batch standardisation at NO step.** A mean over ratios 0.665→0.05 is not what step 50 saw, nor step 800.
3. ⛔ **So the full-schedule recal is a third arbitrary point on the same unaimable axis, not a principled option.** It would return some intermediate `b` — plausibly near 1.0, as they honestly flagged — and that number would mean nothing more than the other two.

### Why this explains the signs, in principle rather than empirically

The head learned a mapping from **"activity relative to a hot local neighbourhood"** to probability, because that is the only reference frame it ever saw.

| eval statistics | what `x̂ = 0` means to the head | effect | measured |
|---|---|---|---|
| hot moments on the global grid | the global spread is **smaller** than assumed, so `x̂` compresses toward 0 | head **under**-reads | `b = 1.19` |
| global moments on the global grid | `x̂ = 0` now means *quiet globally*, so `+2σ` is a far stronger claim than in training | head **over**-reads | `b = 0.78` |

⛔ **No fixed moments can reproduce a per-batch, activity-selected reference frame. The axis is unaimable in principle, not merely in practice** — which is a stronger statement than Amendment 10's, and it is why §5's Goodhart ban extends to every point on the axis including the pre-specified ones.

**Recommendation: the full-schedule recal does NOT go to Simon as a candidate.** It is the same axis, and its result would not be interpretable.

## 4 — Their two H4 strengthenings, both accepted

**1. The 0.41 leverage is a LOWER BOUND on the train/deploy mismatch.** BatchNorm statistics are only the *adjustable* channel; **the weights themselves were fitted on activity-seeking crops and no choice of eval statistics touches that.** So H4's scope exceeds the BatchNorm channel, and E6f measured the part that happens to be tunable. Accepted — it means E6f understated H4 rather than testing it.

**2. ⛔ A scale fix cannot address a shape problem — and this is now the central conclusion.** Matching moments does not make the head's learned *function* appropriate to the globe, because the two populations have different conditional shapes.

> **`pos_weight` moved the intercept when the defect was the slope. BatchNorm moments move the scale when the defect is the reference frame. The same error, one level up.**

That is the third independent route to R1, and the strongest: ⛔ **every axis this programme has examined — `pos_weight`, the NLL denominator's weighting, the recal distribution — is a scale or level knob, and the defect has been a shape defect every time.** Temperature scaling is the only intervention considered that acts on the shape parameter directly.

## 5 — Standing

| | candidate | standing after Amendment 11 |
|---|---|---|
| **H1** | BatchNorm recal distribution | mechanism CONFIRMED (0.41, a lower bound) · remedy DEAD · criterion corrected · ⛔ **and NO fixed statistic satisfies the corrected criterion** |
| **H4** | train/inference geometry | **root cause**, scope **larger** than the BatchNorm channel, conditional-shape argument intact and now central |
| **H2** | `weight_decay: 0.1` | live, premise damaged |
| **H3** | resolution limit | live, untested |
| **R1** | temperature scaling | ⛔ **strengthened a third time** — the only axis that acts on shape rather than scale |

**W-26 stays as one ledger entry naming both seats**, per their request and because it is accurate: they wrote a non-exhaustive table, this seat attacked it for the opposite failure and approved it.

⛔ **Nothing authorises Tier 3, and this amendment removes a candidate rather than adding one.**

---

# AMENDMENT 12 — the DIAGNOSIS has converged. The REMEDY has never been measured. 2026-10-18.

**Source:** the `views-models` seat proposed the close and invited disagreement on exactly that point. ⛔ **I disagree, on one axis, and it is the cheapest thing in this plan.**

---

## 1 — ADOPTED: the two criteria are in irreconcilable conflict

Their framing is better than Amendment 11's "unaimable axis" and belongs in issue 12:

```
(a) running stats = E[training batch moments]  ->  x_hat matches what the head SAW
(b) running stats = eval-data moments          ->  x_hat is correctly STANDARDISED for the data
```

**They coincide only when training batches are i.i.d. samples of the eval distribution. Activity-seeking crops are not.** So `b = 1.19` and `b = 0.78` are **both correct implementations of incompatible criteria.**

⛔ **The axis has no correct point, not merely no reachable one.** Adopted — it is strictly stronger than what Amendment 11 claimed.

## 2 — ADOPTED with a sharpening: RECLASSIFY the full-schedule recal as a DEFECT

They are right that killing it as a remedy does not dispose of it. `_recalibrate_bn` iterates `get_lesson(w)` for `w ∈ range(30)` — lesson indices **0–29 of 900, the first 3.3% of steps, all inside the roof-clipped flat region** (the hottest 23.8%). As an estimator of `E[training batch moments]` that is **biased**; the full-schedule draw is the **unbiased** implementation of the same criterion.

⛔ **My sharpening: it is a defect under EITHER reading of the docstring, which makes the issue airtight regardless of what its author intended.**

| docstring read as | the defect |
|---|---|
| **(b)** *"matches the true activation distribution"* = inference-time | **wrong population entirely** — curriculum windows are not the global grid |
| **(a)** = training-time | **biased estimator** — 3.3% of steps, hottest 23.8%, 1.88× the mean ratio |

**There is no reading under which the implementation satisfies its own stated purpose.** Filed as **OUTCOME issue 13**, with the one-line character of the fix noted (`get_lesson` over the full step range) and ⛔ **an explicit warning that fixing it is NOT expected to repair the calibration** — so no later reader treats the issue as a remedy and re-runs E6f's mistake in a third place.

## 3 — ACCEPTED: the close must not read as "nothing"

My *"I would rather hand him nothing than hand him a measurement neither of us could read"* was about that one measurement, but they are right that it must not colour the close. What this programme produced:

- a **named, quantified defect in a shipped model** — under-dispersion, `b = 1.19` against 1.00 honest, six fits, surviving ten rounds of two-seat attack;
- the mechanism **localised** — BatchNorm statistics carry **0.41** of leverage, a lower bound, with the root in activity-seeking training crops;
- a **remedy killed by measurement** rather than by argument, **before it was shipped as advice**;
- **13 issues**, several in production code, two of which (the silent `D2` stamp, the recal bias) would have gone unnoticed indefinitely.

## 4 — ⛔ MY DISAGREEMENT: TIER 2 HAS NEVER RUN, SO R1 IS UNMEASURED

`results/` holds `e1_sigma.py`, `e2_meanlogit.py`, `e6c_power.py`, `e6d_align.py`, `e6f_bn_reestimate.py`. **Nothing from Tier 2. R1 through R5 have not been executed once.**

⛔ **We are about to recommend a remedy on three theoretical routes with zero measurement — after spending two days establishing that asserting remedies on theory is how we got here.** The pattern this plan named as its own main finding is the one the close would repeat.

**And the plan already contains the falsifiers that would stop it.** They are pre-registered, they are unrun, and each has a different failure mode:

| | fires if | consequence |
|---|---|---|
| **F7** | R1 reaches `b ≈ 1` held-out while **AP and sharpness are unchanged within the ~0.019 seed sd** | ⛔ **H3 — the gain is COSMETIC.** A calibration fix that adds no resolution has repaired a reporting defect, not the forecast |
| **F8** | isotonic beats R1 held-out by **> 20%** on absolute calibration error | the defect is **not a pure slope** — OUTCOME §2 is incomplete a third time |
| **F9** | R1's fitted `T` varies **> 15%** across the 13 origin folds | **no well-defined global temperature** — the simple fix is unavailable |

⛔ **F7 is the one that matters, and H3 is still live and untested.** Three independent routes argue R1 acts on the **right parameter**. ⛔ **None argues it buys anything.** If the gate is at its resolution limit, R1 sets `b = 1.000` and the forecast is unchanged — and §3.6 of this plan says so in advance: *"Expect R1 to improve calibration and move nothing else — that is F7's whole point, not a surprise."*

### And it is the cheapest thing in the plan

R1–R4 are post-hoc monotone transforms of the **stored gate and truth arrays**. No model, no forward pass, no volume build. **Seconds of CPU on arrays already on disk — cheaper than E1, E2, E6c, E6d and E6f, every one of which has run.** F9's 13 origin folds are a loop over a 13-element axis in an array we have loaded repeatedly today.

⛔ **The most expensive experiments ran and the cheapest one did not, because it was filed as a remedy rather than as a test.**

## 5 — So: the diagnosis has converged. The remedy has not been tested.

**Agreed and settled:** the defect, its magnitude, its localisation, H1's death as a remedy, the criterion conflict, the Goodhart ban, the scale-versus-shape generalisation, and that the root cause is a training-side proposal this programme never scoped.

⛔ **Not settled, and one tier of free work away from being settled:** whether temperature scaling buys anything, or whether F7 fires and the honest answer is *"the gate is at its resolution limit, the calibration number can be fixed, and the forecast cannot."*

**That is a materially different thing to tell Simon**, and it costs seconds. **Recommending R1 without F7 would be the fourth time in this programme a remedy was advanced on its mechanism rather than its effect** — after `pos_weight`, the NLL denominator, and the BatchNorm recal.

**Revised recommendation to Simon: run Tier 2 — it is seconds — and report R1 with F7's verdict attached.** If F7 fires, R1 is still worth doing and must be described as cosmetic. If it does not, R1 is the first intervention in this programme with a measured benefit.

---

# AMENDMENT 13 — Tier 2 has run. ✅ R1 confirmed for the SHIPPED model, ⛔ fails for the arm, and "cosmetic" was my error. 2026-10-18.

**Source:** the `views-models` seat executed Tier 2 after this seat refused the close. **It changed what we tell Simon twice over, and it took minutes.** The refusal was correct.

---

## 1 — The results

**AMBER — the shipped baseline:**

| | `b` held-out | `\|gate−obs\|` | AP | sharpness |
|---|---|---|---|---|
| raw | 1.1917 | 0.00981 | 0.54101 | 0.03439 |
| **R1** (1 param) | **1.0378** | **0.00287** | 0.54101 | 0.03380 |
| R3 (2 param) | 1.0037 | 0.00335 | 0.54101 | 0.03572 |
| R4 (isotonic) | 0.9907 | 0.00482 | 0.53837 | 0.03555 |

`T` mean **0.8708**, spread **0.8%** across 13 origin folds.

| | verdict |
|---|---|
| **F2** `\|b−1\| = 0.0378` vs 0.05 | ✅ PASS |
| **F9** `T` spread 0.8% vs 15% | ✅ PASS — **a single global temperature IS well-defined** |
| **F8** R4 0.00482 vs R1 0.00287 | ✅ PASS — **isotonic is 68% WORSE; the defect IS a pure slope** |
| **F7** `ΔAP = 0.00000` exactly | ⛔ **FIRES** |
| **F10** rank identity True, 14 new ties | ✅ PASS |

✅ **R1 cuts held-out calibration error by 71% with ONE parameter, stable to 0.8% across folds, and the non-parametric ceiling cannot beat it.** ⛔ **The first intervention in this programme with a measured benefit.**

**COPPER — the killed arm:**

| | `b` held-out | `\|gate−obs\|` |
|---|---|---|
| raw | 1.3633 | 0.00878 |
| **R1** | **1.2482** | **0.01136** ⬆ **worse than raw** |
| R3 | 1.0028 | 0.00929 |
| **R4** | 0.9928 | **0.00430** ← best |

**F2 FIRES** (`|b−1| = 0.2482`) and **F8 FIRES** (isotonic beats R1). ⛔ **R1 is not the matching fix for the arm.**

## 2 — ✅ Their reconciliation is right, and it survives on a DEFENSIBLE estimator

They explain F8's asymmetry by copper's large intercept, citing the **binned** `a = +1.645` from §1. ⚠️ **That number is confounded** — OUTCOME §8.6 established the bin-unweighted intercept is reproducible to 0.0001 from bin weighting plus the slope defect alone. So I refitted **bin-free**:

```
amber   bin-free  a = +0.1584   b = 1.1875
copper  bin-free  a = +1.2980   b = 1.3596      <- 8.2x amber's intercept
```

✅ **The asymmetry is real and their mechanism holds on the clean estimator.** Amber's defect is **slope-only** (intercept ≈ 0.16); copper's is **slope AND level** (1.30), and one parameter cannot address two. **R3's two parameters fix it exactly** (`b = 1.0028`).

⛔ **So "the defect is a pure slope" is TRUE for the shipped model and FALSE for the arm** — and it was predictable from a number published in rev 2 that neither seat connected to F8. Their mechanism for `T`: NLL-fitted temperature optimises **likelihood, not the slope**, giving `T = 0.9155` where the slope needs `1/1.36 = 0.73`.

## 3 — ⛔ "COSMETIC" WAS MY ERROR. I retract it.

I told Simon that if F7 fired I would describe R1 as **cosmetic**. ⛔ **That is wrong and their correction is right.**

AP is rank-based and `R1` is monotone, so `ΔAP = 0` **by construction** — ✅ which is §3.2's structural argument confirmed numerically to five decimals, and a theorem rather than a measurement. But **the composed forecast is `gate × magnitude`**, so changing the stated probabilities **changes predicted counts**, and **AP cannot see that.**

> ⛔ **R1 buys HONEST PROBABILITIES and nothing on the rank axis. Whether that is worthless or the entire point depends on whether the stated probabilities are consumed AS probabilities** — which, for a system delivering probabilistic forecasts to external partners, is a **product question, not a statistical one.**

**That is Simon's judgement and this plan puts it to him as one, not as a verdict.**

## 4 — ✅ AND THE FIRING-RATE RISK I WAS WORRIED ABOUT DOES NOT MATERIALISE

`T = 0.8708 < 1` means `logit/T` **amplifies** the logits, and most cells sit at very negative logits — so I expected the mean gate to **fall**, worsening the under-firing that started this whole programme. Measured on amber:

```
true positive rate         0.003448
raw   mean gate            0.004167   (1.209x truth)
R1    mean gate            0.003278   (0.951x truth)     <- 21.3% lower
```

✅ **It falls, and it falls TOWARD the truth**: `|1.209 − 1| = 0.209` → `|0.951 − 1| = 0.049`. **R1 improves the aggregate occurrence rate as well as the slope.** The adverse direction I feared does not occur.

**Free structural check:** cells above gate 0.5 are **1296 → 1296, unchanged** — `logit/T` preserves sign, so the count above 0.5 is invariant. Another confirmation of rank preservation, at no cost.

⚠️ **F11 REMAINS UNRUN and R1 is NOT claimed to pass it.** `act_ratio` and CRPS are properties of the **composed** forecast, which Tier 2 does not build, and a 21% drop in mean gate is a ~21% drop in expected counts. **The guardrail that caught the CRPS artefact has not been applied to R1.** Their flagging this rather than glossing it is the right call, and it is the last gate before R1 goes anywhere near production.

## 5 — H3 is NOT confirmed. Their point 4 is correct.

F7 firing is **consistent with** a resolution limit but cannot establish one, because **AP is unchanged by construction under a monotone map — a test that cannot fail cannot confirm.** H3 stays live and untested, exactly as before. ⚠️ Worth noting this is the same reasoning I used against E2's exhaustive table, applied to my own hoped-for conclusion.

## 6 — Standing, final

| | status |
|---|---|
| **R1 on amber (shipped)** | ✅ **CONFIRMED by measurement** — 71% calibration gain, 1 parameter, `T` stable to 0.8%, isotonic cannot beat it, `ΔAP` exactly 0, aggregate rate improved. ⚠️ **F11 outstanding** |
| **R1 on copper (killed arm)** | ⛔ **FAILS.** Its defect is slope **and** level; R3 is the matching fix there |
| **"pure slope"** | ⛔ **arm-specific** — true for the shipped model, false for the arm |
| **H1 / H4 / H2 / H3** | unchanged from Amendment 12 |

⛔ **Nothing authorises F11, production use, or Tier 3.** What goes to Simon: **R1 is confirmed for the model that shipped, with F11 as the one remaining gate, and the "cosmetic" framing withdrawn** — the question is whether honest probabilities are worth having, and that is his.

---

# AMENDMENT 14 — a third place prior art hides, and the control from Amendment 3 was still too narrow. 2026-10-18.

**Source:** the `views-models` seat, reversing one of my corrections and finding the third piece of prior art in the process.

---

## 1 — ⛔ My correction was wrong, and the mechanism is worse than the error

I told them `scripts/sharpness_scorecard.py` does not exist. **It does** — 10,615 bytes, with the inline MCR at `:161`. **Their citation was correct and so was mine: there are TWO MCR implementations**, an inline `pp.mean()/tp.mean()` and the shared `calculate_mcr_native`.

⛔ **The mechanism:** I wrote `grep … <path> | head -4 || find . -name …`. **The pipe makes the exit status `head`'s — which is 0 — so `||` never fired and `find` never ran.** I read an empty section as evidence of absence. **The command was structurally incapable of testing the claim it was used to support**, and a grep that finds nothing is indistinguishable in the output from a grep that never ran. Logged as **W-27**: same class as the eight prior existence errors, **new mechanism** — not skipping a check, but running one that could not fail and believing its silence.

## 2 — ⛔ The third piece of prior art, and the most on-point conceptually

`scripts/audit_run.py` has carried, in its `PRIMARY` metric set, all along:

- **`range_compression`** (`:55-60`, `:130`) — *"1 = full dynamic range kept; **>>1 = compression** (the floor's blob ~20×)"*. ⛔ **The magnitude twin of the slope defect this programme spent three days and thirteen amendments naming on the gate.**
- **`gate_brier`** (`:135`) — a committed proper calibration score for the gate.
- **The MCR demotion** (`:107-108`, `:137`) — *"MCR is a ratio of pooled means and HIDES per-cell failure… **DO NOT read as calibration**… the user caught this 3×."*

⚠️ **It would not have handed us the slope**, and the record should not imply it: `range_compression` is on the **magnitude** field, conditions on the **outcome**, and is a ratio of dynamic ranges rather than a regression coefficient. `gate_brier` is proper but does not decompose into slope and level. **What existed was the vocabulary and the suspicion, not the measurement.**

## 3 — ⛔ The control extension: metric definitions inside scoring functions are a third hiding place

Amendment 3's control says to search `reports/` **and** `scripts/` for prior art per candidate. **We did search `scripts/` — and still missed this, because we searched for instruments by FILE and this is a metric DEFINITION inside a scoring function's output dict.**

> ⛔ **Extension, adopted: searching for files is not enough. Search for the CONCEPT — grep the metric names and the mechanism words (`compression`, `calibration`, `slope`, `reliability`) across scoring functions and metric dicts, not only for plausibly-named modules.**

Three hiding places now: a dossier (`gate_calibration_dossier`), a tool (`gate_reliability.py`), **and a metric dict inside a scoring function.** Each was found by the other seat, after the fact, and none by the control that was supposed to catch it.

## 4 — The MSLE claim, narrowed as offered

The magnitude limb **is** written down — in the `views-models` seat's project memory (`project_mcr_timid_prophet`). ⛔ **It appears nowhere in views-hydranet's `reports/`, `GLOSSARY.md` or project memory.** So my stronger claim ("neither limb was written") is **withdrawn**; the useful half stands: **the rule lived in one seat's memory and in neither repository's text**, which is why the occurrence limb was never derived from it. Corrected in OUTCOME §3.5.

## 5 — My answer to their question: the demotion DOES apply to `act_ratio`

They asked whether MCR's per-cell demotion extends to `act_ratio`, since it is also a ratio of pooled means. ⛔ **Yes, and the proof was already in §3.5 before they asked.** The `TP = 0` case — a model firing the right *number* of times on entirely wrong cells scores exactly 1.0 — **is the demotion's complaint verbatim, transposed to the occurrence axis.** Twins in construction, twins in defect, and **only one carries the warning.** Written into §3.5 as a table, with `precision_at_k` named as `act_ratio`'s committed per-cell companion.

⛔ **Neither MCR nor `act_ratio` is a calibration metric. Both are bias metrics, and both have per-cell companions already in the tree.**
