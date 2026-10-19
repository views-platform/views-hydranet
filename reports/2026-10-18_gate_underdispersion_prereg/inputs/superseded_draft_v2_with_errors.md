# PRE-REGISTRATION v2 — why is the gate under-dispersed, and what fixes it

Joint draft, views-models + views-hydranet seats, 2026-10-18. Nothing executed. Tier 1 and Tier 2
request **zero GPU time and zero dollars**.

Supersedes PREREG_DRAFT.md. All 11 of hydranet's requested changes applied. **Two of its three
attacks are accepted; ATTACK 1 and ATTACK 3 are rebutted on a fit-free measurement** (§2.0).

---

## 1. The facts to be explained — now TWO, not one

Provenance, corrected per hydranet (d): the slopes are fitted on `p2_result_{amber,copper}.npz`,
`gate` = `(4, 13, 64818)`, a **fresh local CPU forward pass** (`p2_gate_calibration.py:109-119`,
`sigmoid(raw class-head logit)`, D=4 locked-dropout passes, torch 2.10), averaged at `:123`.
**Not** the 16-column pod slices. The 16-column `y_pred_h1.npy` object is a *second, unvalidated*
measurement — see E5'.

Fit `logit(obs) = a + b*logit(gate)`. At the weighted-BCE optimum theory demands **exactly**
`a = -log(w)`, `b = 1` (derivation §2.0).

| | a predicted | a measured (3 fits) | b predicted | b measured (3 fits) |
|---|---|---|---|---|
| baseline `amber` (w=2) | **-0.6931** | **+0.063 / +0.175 / +0.026** | 1 | 1.146 / 1.177 / 1.139 |
| arm `copper` (w=1) | **0.0000** | **+1.645 / +0.902 / +1.777** | 1 | 1.419 / 1.251 / 1.456 |

**FACT 1 — THE SLOPE.** `b > 1` in all six fits. The gate's logits are compressed; truth moves
faster than the gate. Present in both arms, worse in the arm, invisible to the pre-registered
statistic (96% of cell-months sit in one bin where the signed error is +0.0004).

**FACT 2 — THE LEVEL. New in v2.** The `log(w)` intercept signature is **absent**. The baseline's
intercept should be `-0.693` and is `+0.026 … +0.175` — wrong sign, ~0.72-0.87 logits off, in all
three fits. Any mechanism proposed must now explain **both**.

---

## 2.0 ⛔ REBUTTAL — both "at the optimum" arguments fail, because we are not at the optimum

hydranet's derivation is correct and we adopt it. For weighted BCE with `s = sigmoid(z+eps)`:

```
dL/dz = -w*y*(1-s) + (1-y)*s
E over y~Bern(p) and eps, set to 0   =>   E_eps[s] = w*p / (1 + (w-1)*p)
and hence                                 logit(s*) = log(w) + logit(p)     EXACTLY
```

Two consequences hydranet draws from it, both conditioned on the stationary condition **holding**:
- **ATTACK 1:** the condition is on `E_eps[sigmoid(z+eps)]` — the very functional inference reports
  — so Jensen compression is absorbed during training and C1 predicts no bias at convergence.
- **ATTACK 3:** `pos_weight` is exactly a `log(w)` intercept shift, so it cannot change ranks, AP or
  resolution; the axis is **inert** and the question should be retired, not reopened.

**We tested the condition. It fails, and fails hardest on the delivered model.** Fit-free — no
binning, no regression, no extrapolation, pooled over 842,634 cell-months:

| arm | positive rate | predicted mean gate `wp/(1+(w-1)p)` | observed mean gate | ratio |
|---|---|---|---|---|
| baseline `amber` (w=2) | 0.003448 | 0.006871 | 0.004167 | **0.606x** |
| arm `copper` (w=1) | 0.003448 | 0.003448 | 0.003260 | **0.945x** |

**The baseline misses its own stationary condition by 39%.** The arm is within 5.5% of its.

Therefore:
1. **ATTACK 1 is conditionally correct but its condition is false.** C1's compression is not
   absorbed, because training did not reach the point at which absorption happens. C1 is still not
   *independent* — it is entangled with whatever holds the network off the optimum — so we accept
   the demotion from "lead candidate" while rejecting "predicts zero compression".
2. **ATTACK 3's inertness cannot be asserted for these networks.** It is a statement about the
   optimum. It is also independently contradicted by two of our own measurements, both of which
   would be null under an exact intercept shift:
   - the **slopes differ**: 1.146 vs 1.419. A pure intercept shift cannot change a slope.
   - **AP moved**: 0.5324199835833106 -> 0.5384912009101893, paired by month +0.0075, CI
     **[+0.0014, +0.0136]**, 11/13 months positive. The CI excludes zero. "Held within the
     pre-registered +/-0.010" is not "exactly unchanged". A pure intercept shift cannot change AP.
3. **This promotes H2 and H1.** A mechanism that holds the network off its own loss optimum, in the
   compressing direction, post- or during training, is exactly what H1 and H2 describe. FACT 2 is
   the signature of that mechanism, and it was invisible until the `log(w)` prediction was derived.
4. **Recorded against our own kill:** the arm we killed sits at 0.945x its own stationary condition;
   the baseline we shipped sits at 0.606x. The arm gets the aggregate *level* nearly right and the
   *shape* badly wrong; the baseline gets the level badly wrong and the shape less wrong. This is
   not a resurrection — see R5' — but it belongs in the record.

---

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
FACT 1, and FACT 2 as well if the shrinkage is toward a non-zero centre. **And its own docstring
confirms the blind spot:** C-184 was "Validated 2026-06-27: flips 6/6 bad seeds GOOD and preserves
2/2 good" — validated on seed-bimodal eval collapse, **never on calibration slope**.

### H2 — `weight_decay: 0.1` on the head ⛔ **CO-LEAD CANDIDATE**
`0.1` in both arms — **10x the AdamW default**. `utils.py:278` passes it to
`AdamW(unet.parameters(), ...)` with no exclusion list. The exclusion mechanism **exists** —
`sigma_params` and `log_var_params` get explicit `weight_decay: 0.0` groups
(`training_engine.py:113,136`) — and the three class heads are simply not in it. Verified
independently by the views-models seat.

Decoupled decay shrinks the head's weights every step; shrinking the final linear compresses the
logit range. **It is the one candidate that predicts `b > 1` at full convergence** — which §2.0 and
(f) together say is the regime we are in — and it predicts FACT 2 directly, because it moves the
optimum away from the loss optimum in the compressing direction.

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
| **R3** | Platt / vector scaling | 2 | if it does not beat R1 held-out, one parameter is enough. **Addresses FACT 2, which R1 cannot** |
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

But §2.0 rejects the conclusion drawn from it. **Inertness is a property of the optimum, and neither
arm is at it** — and the slope difference and the AP movement both contradict it directly.

R5' therefore reports, for both arms: the **full calibration-refinement decomposition**, the
distance of each arm from its own stationary condition (0.606x / 0.945x), AP with its paired CI, and
the sharpness term. Pre-registered readings:

| if | then |
|---|---|
| the kill rested **wholly** on the removable calibration component, AND resolution is equal within CI | the axis is inert and the question retires — hydranet's reading |
| resolution differs (as the +0.0075 CI currently indicates) | `pos_weight` perturbed the network beyond an intercept shift, which **cannot happen at the optimum** and is therefore evidence for H1/H2/C3. Not a resurrection of the arm; a localisation of the defect |

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
