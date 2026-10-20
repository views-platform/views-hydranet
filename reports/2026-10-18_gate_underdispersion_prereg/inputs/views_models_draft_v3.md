# PRE-REGISTRATION v3 — why is the gate under-dispersed, and what fixes it

Joint draft v3, views-models + views-hydranet seats, 2026-10-18. Nothing executed. Tier 1 and Tier 2
request **zero GPU time and zero dollars**.

Supersedes v1 and v2. All 11 of hydranet's v1 changes and all 7 of its v2 changes applied. **v2 §2.0's rebuttal of ATTACK 1 and ATTACK 3 is WITHDRAWN** — it failed on Jensen's inequality (§2.0) and on pseudo-replication (§2.0 end). Both verified independently by the views-models seat.

**Predecessor:** `reports/2026-10-16_pos_weight_arm_outcome/OUTCOME.md` @ `ce56244` (rev 2), which
establishes the fact this document tries to explain. Read its §2 first.

> **UNCOMMITTED, deliberately.** `reports/` is gitignored in views-hydranet (`.gitignore:203`),
> which is why `OUTCOME.md` needed a forced add. If this is committed: `git add -f` is required, and
> **all three files should go in ONE commit** — `PREREG.md`, `HYDRANET_REVIEW_v2.md` and
> `superseded_v2_with_errors.md` — so the error trail is attached to the correction rather than
> arriving separately. Which repo owns a document explaining a views-hydranet model from
> views-models arrays is a two-repository judgement and is Simon's; **both sessions declined to make
> it.** Written into the views-hydranet working tree without touching its git state: `HEAD` was and
> remains `ce56244` on `outcome/pos-weight-arm`, clean, verified independently by both seats.

---

## 1. The fact to be explained — ONE fact. v2's "second fact" is withdrawn.

Provenance (hydranet d): slopes are fitted on `p2_result_{amber,copper}.npz`, `gate` =
`(4, 13, 64818)`, a fresh local CPU forward pass (`p2_gate_calibration.py:109-119`, `sigmoid(raw
class-head logit)`, D=4 locked-dropout passes, torch 2.10), averaged at `:123`. **Not** the
16-column pod slices, which are a second unvalidated measurement (E5').

**THE FACT.** Fit `logit(obs) = a + b*logit(gate)`. `b > 1` in all six fits — 1.146 / 1.177 / 1.139
(baseline), 1.419 / 1.251 / 1.456 (arm). The gate's logits are compressed; truth moves faster than
the gate. Present in both arms, worse in the arm, invisible to the pre-registered statistic (96% of
cell-months sit in one bin where the signed error is +0.0004).

⛔ **v2's FACT 2 ("the level — the `log(w)` signature is absent") IS WITHDRAWN. It was wrong.**
v2 read the *fitted intercept* as the `log(w)` signature. It is not. The slope-free form — the
condition is exactly `logit(obs) - logit(gate) = -log(w)` in every bin, with no fitted slope to
absorb anything — gives, for the baseline (theory `-0.6931`):

```
per bin:  -1.01 -0.57 -0.27 -0.13 -0.15 +0.13 +0.03 +0.08 +0.24 +0.34 +0.49 +0.90
unweighted over bins : +0.0069
n-weighted by mass   : -0.9825     <-- RIGHT SIGN, 0.29 logits from theory
```

**The signature is PRESENT.** v2's `+0.1749` was an artefact of unweighted bin fitting, and
reproduces arithmetically: `a = ybar - b*xbar`, bin centroid `xbar = -0.9488` (gate ~0.279) while
96% of the data sits at gate ~0.005 (logit -5.29), so the intercept at gate=0.5 is a long
extrapolation from where almost no data is. `-0.9419 - 1.1771*(-0.9488) = +0.1749`, reproduced to
0.0001 by the bin-unweighted mean deviation plus the slope defect. **No `log(w)` information
survives in it.** The `+0.0069` is ~0.04% of the data outvoting the mass.

And there is **no intercept to test**: the deviation spans `-1.01 … +0.90`, **1.91 logits**. Fitting
one number to it is the same structural error that voided "the honest `pos_weight` is ~1.5"
(OUTCOME §8.1) — a point summary of a quantity varying by more than its own magnitude and crossing
the tested value. That class voids a claim; it voids this one.

**There is one fact to explain: the slope.**

---

## 2.0 Distance from the loss's stationary point — ⛔ NULL. Unmeasurable at `w != 1`.

hydranet's derivation, adopted: for weighted BCE with `s = sigmoid(z+eps)`,
`E_eps[s] = phi(p) = w*p/(1+(w-1)*p)`, hence `logit(s*) = log(w) + logit(p)` exactly.

⛔ **v2 TESTED THIS WRONG, AND THE ERROR WAS JENSEN'S INEQUALITY — the mechanism this whole
investigation is about.** The condition is **per cell**: `mean_i E[s_i] = mean_i[phi(p_i)]`. v2
computed **`phi(pbar)`** — the function at the mean, not the mean of the function. Verified
independently by the views-models seat:

```
phi''(p) = -2w(w-1)/(1+(w-1)p)^3        CONCAVE for w>1,  EXACTLY ZERO for w=1
```

⛔ **So the bias exists in ONE arm only. At w=1, phi is the identity and the Jensen gap is
identically zero, invariant to bin count.** Measured gap in v2's statistic: **+0.001569**
(baseline), **+0.000000** (arm). **v2's statistic was structurally incapable of comparing the two
arms, and it penalised precisely the one it was used to indict.**

Corrected, `mean_i[phi(p_i)]` with `p_i` from equal-mass gate bins, extended to the per-cell limit
(reproduced independently by the views-models seat):

| bins | `mean_i[phi(p_i)]` | ratio obs/pred, baseline (w=2) | arm (w=1) |
|---|---|---|---|
| `phi(pbar)` — **v2's error** | 0.006871 | 0.606x | 0.945x |
| 12 | 0.006631 | 0.628x | 0.945x |
| 40 | 0.006200 | 0.672x | 0.945x |
| 200 | 0.005490 | 0.759x | 0.945x |
| 1000 | 0.005303 | 0.786x | 0.945x |
| 5000 | 0.005284 | 0.789x | 0.945x |
| 25000 | 0.005218 | 0.799x | 0.945x |
| 100000 | 0.004981 | 0.837x | 0.945x |
| 300000 | 0.004450 | 0.936x | 0.945x |
| **400000** (2.1 cells/bin) | **0.004223** | **0.987x** | 0.945x |
| **421317** (exactly 2 cells/bin) | **0.004174** | **0.998x** | 0.945x |
| 500000 (1.7 cells/bin) | 0.004040 | **1.032x** | 0.945x |
| **per-cell** | **0.003448** | **1.209x** | 0.945x |

⛔ **NULL. THE RESIDUAL'S SIGN IS NOT ESTABLISHED, AND v3's FIRST DRAFT OVERSTATED IT TOO.** An
earlier v3 reported ">= 0.786x, direction unchanged". That was wrong in the same class as the two
claims it replaced. The estimator is biased in **both** directions and the family **crosses 1.0**:

- **coarse** binning destroys within-bin variance in `p`; `phi` concave ⇒ `mean[phi]` too HIGH ⇒
  ratio too low (0.628)
- **fine** binning replaces `p` with its own Bernoulli sampling noise, maximally spread ⇒
  `mean[phi]` too LOW ⇒ ratio too high (**1.209**)
- the limit is **analytic, not empirical**: as bins refine, `p_b -> y_i in {0,1}`, and `phi(0)=0`,
  `phi(1)=1` for **every** `w`, so `mean_i[phi(y_i)] = pbar` exactly and the ratio tends to
  `obs/pbar = 1.209`. Verified for w = 1, 2, 5.

⛔ **And the crossing is FINITE and BRACKETED, not merely asymptotic.** The family reaches
**0.998x at 421,317 bins** and **1.032x at 500,000**, so it passes through exactly 1.0 between 2.0
and 1.7 cells per bin — a computable resolution, not a degenerate limit. This closes the only
objection available to the null: one cannot dismiss the crossing as an unreachable asymptote.

Reproduced by **three independent implementations** — sorted-index cumulative-sum binning, quantile
edges via `np.quantile` + `np.digitize` (hydranet), and a vectorised `cumsum`-and-slice over the
sorted order (views-models seat). All three agree to the printed digit at every shared resolution
(0.628 / 0.759 / 0.786 / 0.799 / 0.837). **The sweep is not an artefact of any one binning scheme.**

**No bin resolution is unbiased, the family is monotone between 0.628 and 1.209, and it passes
through the value being tested at finite resolution. Nothing in the data says where on that curve
the truth sits.**

One caveat recorded so the strengthened argument is not itself overstated: at ~2 cells per bin the
per-bin rate `p_b` can only take the values {0, 0.5, 1}, so that end of the sweep is a degenerate
estimator of `p` just as the coarse end is. **The point is not that 0.987x is the answer — it is
that no point on the curve is privileged, and the curve contains 1.0.** The verdict is NULL from
either direction.

### ⛔ The general result — a fit-free test is IMPOSSIBLE in principle when `w != 1`

This is the keeper, because it forecloses re-attempts rather than just retiring this one.

The condition is `E[s_i] = phi(p_i)` with `phi` **nonlinear** in a `p_i` that is **never observed** —
we only ever see one Bernoulli draw from it. Any estimator of `mean_i[phi(p_i)]` built from binned
outcome frequencies therefore inherits a Jensen bias at the bin level, whose sign and magnitude are
set by the bin width, a free parameter. Removing that bias requires a **model for `p`** — i.e. a
fitted calibration curve — which reintroduces precisely the fitting the test existed to avoid.

**At `w = 1`, `phi(p) = p` is linear, the Jensen gap vanishes at every resolution, and the fit-free
test exists and is valid.**

⛔ **So the asymmetry in v2's §2.0 was never a fact about the two models. A valid fit-free test
exists for one arm and cannot exist for the other. That is a property of `pos_weight`, not of the
networks** — and it is the cleanest statement of why this test could never have done the job asked
of it.

**Recorded verdict:** the baseline's distance from its own stationary condition is **UNMEASURED** —
not small, not large, not signed. The arm sits at **0.945x**, a valid measurement, invariant to bin
count. This supports nothing, which is correct.

**This is the third instance of one defect class** and the pattern is now named in the design:
`implied_w` (varies 6.8x/12.8x, crosses 1.0 — OUTCOME §8.1), the fitted intercept (1.91-logit range,
crosses zero — §1), and this estimator (0.628-1.209, crosses 1.0). **Pre-registered rule for the
rest of this programme: before any scalar summary is reported, sweep whatever free parameter it
depends on and show the family does not cross the value being tested.**

### What the residual does and does not license

⛔ **It is NOT evidence of non-convergence, and the test cannot see the end of training.** The
shipped artifact is **post-BN-recal**: `_recalibrate_bn` (`training_engine.py:1294-1301`) runs
*after* the last gradient step, resets every BN's running stats and re-accumulates them
forward-only, rescaling the features feeding the head. **So the stationary condition can hold
exactly at the last gradient step and be violated in the artifact we measured.** This test
therefore **cannot discriminate C3 ("never reached it") from H1 ("reached it, then BN recal moved
us off")** — and those have opposite implications for C1.

**ATTACK 1 stands, in its narrowed form** (hydranet change 5), which both seats adopt:

> C1's Jensen compression is absorbed by the trained weights at the LOSS's stationary point. The
> shipped artifacts are displaced from that point by two identified mechanisms — `weight_decay: 0.1`
> (which displaces the total objective's optimum, by construction, in the compressing direction) and
> post-hoc BN recal (which acts after the last gradient step). C1 contributes compression ONLY
> through that displacement, which is H2 and H1. It remains not an independent cause.

Corroborated by (f): the gate-logit level converged by ~lesson 150 (last-100 drift +0.0003/lesson,
opposite in sign to the early drift, an order of magnitude under the within-block sd). A
non-converged network would not show that. Practical effect on the design is nil — C1 stays demoted
— but it changes what E1/E2 **mean**: a non-zero sigma with `b` unexplained by it points at H1/H2,
not at C1.

### ⛔ ATTACK 3: phrasing conceded by hydranet, substance accepted by us. The AP evidence is withdrawn.

v2 cited AP moving (+0.0075, CI [+0.0014, +0.0136], 11/13 months) as proof that resolution changed
and the axis is not inert. **That inference is withdrawn — it is pseudo-replication.**

The unit of randomisation for the intervention is the **training run**, and there is **one run per
arm**. Pairing by month replicates the **evaluation**, not the intervention; the 13 months are 13
correlated reads of the same two artifacts. A month-wise CI can answer "is this difference
consistent across months" and **cannot** answer "is it attributable to `pos_weight`". The CI is
tight because the reads are correlated, not because the effect is resolved.

The relevant variance component is already in our own record: **`2026-10-16_pos_weight_prereg.md:175`
— "n=1 SCREEN at 300 lessons. Against the measured AP seed sd <=0.019"**. **+0.0075 is 0.39 of one
seed sd, and v2's entire CI lies inside it.** Verified independently.

**AP +0.0075 is recorded as NOT ATTRIBUTABLE. The number is kept; the inference is dropped.**

The slope difference (1.146 vs 1.419) is likewise **not** attributable to `pos_weight`: it is
evidence **for H1**, because `bn_recal_windows` defaults to `windows_per_lesson * 10`
(`training_engine.py:1303`), making the BN running stats a high-variance cumulative estimate over
few windows — **H1 predicts the calibration slope varies run to run.** With no seed replicate we
cannot separate that from a `pos_weight` effect, which is exactly the point.

Correct statement about the mechanism, adopted:

> `pos_weight` has no mechanism by which it can change ranks. At the loss's stationary point its
> effect is exactly `logit(s*) = log(w) + logit(p)` — an intercept shift with zero slope effect.
> Differences between the two artifacts are RUN-LEVEL and, with n=1 training run per arm, cannot be
> attributed to `pos_weight`.

F1's "exactly unchanged" remains the right invariant for a monotone transform of **one** artifact's
gate — all F1 governs — and is untouched by any between-artifact comparison.
## 2. Candidate causes, restructured as one chain plus three architectural candidates

### The chain (hydranet change 2): C1 is a consequence, not a cause
`C1` (MC-dropout Jensen temperature, `E[sigmoid(z+eps)] ~= sigmoid(z/T)`, `T = sqrt(1+pi*sigma^2/8)`)
is real pointwise and **cancels at the optimum**. It can only bite through:
- **C2 — training/inference eps mismatch.** CONFIRMED FROM CODE, and sharper than v1 stated.
  Training: `locked_dropout.py:61-62`, `locked=False`, a **fresh mask on every forward**, ~383
  i.i.d. masks averaged inside one backward. Inference: **one mask locked across the whole 36-step
  roll** (`hydranet_inference.py:536-542`, C-113), 4 of them. Different count *and* different
  correlation structure. Rate `0.15` in both arms, byte-identical; 5 of 15 dropout sites on the sb
  gate path, one of them (`h1_class`) directly on the final linear's input.
- **C3 — non-convergence.** See below: **unresolvable from existing artifacts.**
- **H3 — resolution limit.** The null.
- **H2 — weight decay.** The optimised objective is loss **+ penalty**, so its optimum is not the
  loss optimum. This is the mechanism §2.0 shows we need.

### C4 — promoted from robustness check to C1/C2's falsifier (hydranet change 11)
Fit `b` per origin and per intensity stratum. Also the carrier of the only genuinely blind
prediction available — see §3.

### C5 — measurement noise. RULED OUT. Attacked by both seats, survived.
Two independent mechanisms push `b` **down**: classic attenuation, and binning on the noisy
predictor (live here — `verify_slope.py` bins on `gm` itself). `b > 1` survives both. The class is
dead. **Consequence (hydranet g): the measured `b` is the compression attenuated by residual
4-draw noise, so `b` is a LOWER BOUND on `T`.** v1's two-sided sigma bands were therefore biased
low and would have fired on a supporting result. Corrected in §3.

### C6 — a downstream compression site. ANSWERED AND CLOSED.
Complete path from `dec_conv4_head1_class` to the npz: `torch.concat` -> `torch.sigmoid` -> `mean
over D=4`. **No clamp, no normalisation, no epsilon, no clipping, no second averaging.** The class
head applies no activation (`HydraBNrecurrentUnet_06_LSTM4.py:660-661`), unlike the regression head
at `:643`. Verified independently by the views-models seat. **Every compressor is upstream, inside
the network.**

### C7 — horizon. Unmeasurable; the strongest argument for #558 / C-161.
All of this is h=1. The delivery is 36 months. Must not be generalised.

### H1 — BN recalibration on the wrong distribution ⛔ **CO-LEAD CANDIDATE**
`bn_recalibrate: True` in both arms. `_recalibrate_bn` (`training_engine.py:1294-1301`) resets every
BN's running stats and re-accumulates them forward-only over **32x32 activity-seeking curriculum
windows** (positive rate 0.004508). Inference then runs eval-mode BN with those fixed stats on the
**180x180 global land grid** (positive rate 0.003448). `bn_dec_conv1_head1_class` sits immediately
upstream of the final linear. If the recal distribution's activation variance exceeds the global
grid's, eval-mode BN divides by too large a sigma, shrinking the features entering the head and
**multiplicatively compressing the logit**.

Applied **post-training**, so no optimum can absorb it. Present in both arms. Right sign. Explains
THE FACT — a pure slope defect in logit space. **And its own docstring
confirms the blind spot:** C-184 was "Validated 2026-06-27: flips 6/6 bad seeds GOOD and preserves
2/2 good" — validated on seed-bimodal eval collapse, **never on calibration slope**.

### H2 — `weight_decay: 0.1` on the head ⛔ **CO-LEAD CANDIDATE**
`0.1` in both arms — **10x the AdamW default**. `utils.py:278` passes it to
`AdamW(unet.parameters(), ...)` with no exclusion list. The exclusion mechanism **exists** —
`sigma_params` and `log_var_params` get explicit `weight_decay: 0.0` groups
(`training_engine.py:113,136`) — and the three class heads are simply not in it. Verified
independently by the views-models seat.

Decoupled decay shrinks the head's weights every step; shrinking the final linear compresses the
logit range. **It is the one candidate that predicts `b > 1` at full convergence** — which (f)'s
converged logit level says is the regime we are in.

⛔ **H2's support is RE-BASED (hydranet change 4). It does NOT rest on v2's 39% figure**, which was
Jensen-biased and is withdrawn. It rests on the displacement argument, which is stronger because it
needs no statistic at all: `E[s] = phi(p)` is the stationary point of the **loss**, while AdamW
optimises **loss + decay penalty**. The penalty's optimum is systematically less extreme in logit
space, so **weight decay moves the optimum away from calibration by construction, in the compressing
direction** — it predicts the observed sign a priori. The residual in §2.0 is then a *predicted
displacement*, not evidence of non-convergence.

### H3 — resolution limit. The null hypothesis, carried explicitly.
If the features cannot resolve per-cell `p`, then `b > 1` is an honest statement of limited
resolution, not a defect, and **no monotone recalibration can fix it** — R1 would "succeed" on
calibration and buy nothing. Answered on the rank axis (AP, sharpness), which no remedy here moves.

---

## 3. TIER 1 — on-disk, read-only, zero dollars

Substrate verified 2026-10-18: 78 pod slices (2 arms x 13 origins x 3 targets, `(64818,16)`) plus
the two local npz (D=4). Probabilities unsaturated (2.0e-5 … 0.996), so logits invert exactly.

| # | experiment | pre-registered prediction |
|---|---|---|
| **E1** | across-draw logit sd `sigma`; derive `T` | **ONE-SIDED (change 3):** `sigma >= 0.85` baseline, `>= 1.20` arm. `b` is a lower bound on `T`, so sigma **above** the floor is CONSISTENT with the chain, not falsifying |
| **E2** | refit slope on `sigmoid(mean logit)` | **THREE outcomes (change 4):** `b->1` = C1 uncancelled; `b` stays `>1` = compression is in the network; **`b<1` = training absorbed it and R2 OVERSHOOTS** — hydranet's derivation predicts this third, and if it fires **R2 is dropped**, not celebrated |
| **E3** | slope on each single draw | single-draw `b` below mean-gate `b`; if single draws are already `b>1.1` the compression is in the weights |
| **E4** | `b` per origin (13) and per intensity stratum | `b` within +/-0.10 across origins if structural. **Carries the blind prediction** |
| **E5'** | **REPLACED (change 5).** K is pure duplication for the gate — `np.repeat(prob_thwc[...,None], posterior_K, -1)` at `hydranet_inference.py:1707-1709`, verified independently. 16 columns hold **4 distinct values repeated 4x**, so there is no 4-vs-16 variance contrast in the substrate. Now: **local (torch 2.10) vs pod (torch 2.14) determinism check** on the 4 distinct values | agreement to ~0.002, as the earlier cross-check showed |
| **E6** | ⛔ **NEW (change 6) — THE DISCRIMINATING EXPERIMENT.** Fit the slope on 32x32 activity-seeking curriculum windows (the BN-recal distribution) and compare to the global grid. Machinery exists: `VolumeSampler` + `CurriculumLearner`, driven this way at `reports/2026-10-13_composition_arm_outcome/r3_per_target.py:93-108` | **H1: `b ~ 1` on windows, `b ~ 1.15` globally.** **H2: `b > 1` on BOTH**, since decay compresses the weights themselves. This single measurement separates the two co-leads |

**Blindness repair (change 11).** v1 called E1 a risky prediction. It was not: the sigma bands were
derived by inverting `b` values already published in OUTCOME.md §2 and quoted in §1 here, so
agreement would be consistency, not confirmation. Both seats agree this must be said plainly rather
than claimed as rigour. The genuinely blind axes, chosen because **neither seat has seen these
numbers**: (i) the per-origin and per-stratum `b` of E4 — fit `T` from sigma on 12 origins, predict
`b` on the held-out 13th before fitting it there; (ii) **E6's window-grid slope**, which does not
exist in any artifact yet and is the best blind test available.

**Ordering, pre-registered:** E1 recorded before E2. E6 runs alongside E1/E2, not behind them.

**C3 is UNRESOLVABLE from existing artifacts (change 10).** The hook records
`_o.detach().mean().item()` — a single scalar mean, **no sd, no quantiles** (`training_engine.py:1430-1432`,
verified independently). The statistic C3 needs is not in the file. The CSV exists for **copper
only** (`config_hyperparameters.py:233`; amber sets no `trajectory_log_path`), so C3 could never be
answered for the baseline. What *is* answerable is the level, and hydranet measured it: the head's
logit mean converges by ~lesson 150 (last-100 drift +0.0003/lesson, opposite in sign to the early
drift and an order of magnitude below the within-block sd). **Necessary but not sufficient** — a
converged mean is compatible with an expanding spread. Adding an sd to that hook is a one-line
change and is listed as a **prerequisite**, not an experiment. (`fed_grad_max` is 0.0 on all 300
rows and is inert.)

---

## 4. TIER 2 — remedies. Zero dollars. Cross-validated by origin (fit 12, evaluate 1, 13 folds).

| # | remedy | params | role |
|---|---|---|---|
| **R1** | temperature scaling `logit/T` | 1 | the fix matching FACT 1 |
| **R2** | report `sigmoid(mean logit)` | 0 | **conditional on E2.** Dropped if E2 gives `b<1` |
| **R3** | Platt / vector scaling | 2 | the "more general" control: it can absorb a level offset as well as a slope, so **if it does not beat R1 held-out, one parameter is enough and we stop.** (v2 justified R3 by "FACT 2", now withdrawn — no level defect is established, so R3's role is purely to bound R1's sufficiency) |
| **R4** | isotonic | non-param | the falsifier for the pure-slope reading |
| **R5'** | **re-aimed (change 8)** | — | see below |

### F1 corrected — hydranet ATTACK 2, accepted in full
v1's "AP moves at all => bug" was wrong twice, and one way would have sent us chasing a phantom:
- **R2 is NOT monotone in the gate.** It is a different functional of the draw vector: two cells can
  share `mean(sigmoid)` and differ in `mean(logit)`, and their order can legitimately flip. R2 is
  **excluded from F1**, and its AP movement is a **result**.
- **float32 can create ties.** With 96% of 842,634 values inside `[0, 0.01)` and a minimum of
  2.0e-5, dividing logits by `T~1.15` and re-sigmoiding will collide some; `average_precision_score`
  groups ties, so AP can move by rounding alone.

**Replacement:** assert **rank identity directly** — `argsort` equality (equivalently Kendall tau
exactly 1.0) pre- vs post-transform — and report the count of newly created ties. AP movement under
R1/R3 is a bug **only if rank identity also fails**.

### R5' — re-aimed, and the verdict is neither "killed" nor "retired"
hydranet is right that v1's R5 was near-vacuous: temperature is fitted **per arm**, so both land at
`b ~ 1` by construction and the comparison removes the thing being measured. Accepted. By the
Murphy calibration-refinement decomposition, a monotone fit removes the calibration component and
cannot touch resolution.

R5' reports, for both arms: the **full calibration-refinement decomposition**, each arm's corrected
distance from its own stationary point (**arm 0.945x; baseline UNMEASURABLE in principle** — §2.0,
not v2's 0.606x and not v3-draft-1's 0.786x), AP, and the sharpness term. Pre-registered readings:

| if | then |
|---|---|
| the kill rested **wholly** on the removable calibration component, AND resolution is equal within the **seed** sd | the axis is inert on this evidence and the question retires |
| resolution differs by **more than the ~0.019 seed sd** | the two artifacts differ structurally. ⛔ **With n=1 training run per arm this CANNOT be attributed to `pos_weight` rather than run-level variance** — and H1 predicts exactly such run-level variance via `bn_recal_windows`. Attribution requires a seed replicate, which is Tier 3 |

⛔ **Pre-committed, so it cannot be forgotten later: a month-paired CI is NOT replication of the
intervention.** The unit of randomisation is the training run; n=1 per arm; the 13 months are
correlated reads of the same two artifacts. Every between-arm difference in this document is
reported against the **seed** sd, never against a month-wise CI. This is the error v2 made with AP
(+0.0075 = 0.39 of one seed sd) and it is pre-registered here so the corrected design cannot repeat
it.

### Falsifiers, Tier 2
| F | fires if | consequence |
|---|---|---|
| **F1'** | rank identity fails under R1 or R3 | implementation bug. Stop. (AP movement alone is not sufficient) |
| **F2** | R1 held-out slope outside `1.00 +/- 0.05` | temperature is not the matching fix; diagnosis wrong |
| **F3** | R4 beats R1 held-out by >20% on absolute calibration error | not a pure slope; OUTCOME §2 incomplete again |
| **F4** | R1's `T` varies >15% across the 13 folds | no well-defined global temperature; the simple fix is unavailable |
| **F5** | R1 "succeeds" on calibration while AP and sharpness are unchanged | **H3.** The gain is cosmetic and buys nothing real. Must be reported as such |
| **F6** | `act_ratio` leaves its guardrail band under any remedy | the recalibration broke the composed forecast — the CRPS trap in reverse |

CRPS is reported **only** alongside `act_ratio`, and **always with its eval_type labelled** (all
three of month-wise / step-wise / time-series-wise, never one unlabelled number).

---

## 5. TIER 3 — gated
**No GPU time is requested or spent until Tier 1 has named the cause.** Written down because it is
the step that gets quietly skipped. If Tier 1 points at them (~12 h, ~$16 per arm):
- **T3a (H2):** `weight_decay: 0.01`, or a `weight_decay: 0.0` group for the three class heads —
  the mechanism already exists at `training_engine.py:113,136`.
- **T3b (H1):** `bn_recalibrate` over global-grid windows, or disabled, with C-184's seed-collapse
  check carried as a guardrail — it is the thing that flag exists to prevent.
- **T3c (C2):** inference-time mask structure matched to training, against C-113's reasons.
- **T3d (C3):** the one-line sd addition to the trajectory hook, on **both** arms.
- **T3e (C7):** #558 / C-161 — the gate posterior at h>=2. The only route to the delivered horizons.

Every arm carries the `act_ratio` guardrail verbatim.

## 6. Not doing
No honest `pos_weight` hunt (`implied_w` varies 6.8x/12.8x and crosses 1.0 — OUTCOME §8.1). No
spending in Tiers 1-2. No touching `models/`, any `run.sh`, or `p2_gate_calibration.py`. No use of
that harness's `verdict` field — it stamps **both** arms "D2 / already calibrated", the arm passing
by +0.0009 while under-stating by 0.1036 at p=0.12; it fails silently. No generalising h=1 to h=36.
No opening the five argued-for issues — Simon's call.

## 7. Method, locked
Seeds `42`; `BASE_ORIGIN 456`; `N_ORIGINS 13`; `LEADIN_START 421`; CPU. P2 bin edges unchanged.
**Common bin support across arms, always** — the 5-vs-4 asymmetry is the defect that voided the
first reading. **Signed error reported alongside absolute, always** — taking the absolute value is
what hid this for two readings. Positive fraction pooled, never a median over cells. Slopes under
all three weightings. Logit clipping at 1e-9 recorded with a saturation count.
