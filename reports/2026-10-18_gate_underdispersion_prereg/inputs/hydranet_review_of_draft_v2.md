# views-hydranet review of PREREG_v2.md §2.0

**Verdict: §2.0's fit-free test has a hole, it is quantified below, and the hole is Jensen's
inequality — the same mechanism this whole investigation is about.** Test (1), the intercepts, is
confounded with the slope defect and carries no independent information. ATTACK 1 and ATTACK 3 both
survive in corrected form. H2 does **not** lose its support — it gains better support than §2.0 gave
it, from a different argument.

Everything below is read-only, computed from `reports/2026-10-16_pos_weight_arm_outcome/evidence/
p2_result_{amber,copper}.npz` at views-hydranet `ce56244`. I have still not touched sigma.

First, housekeeping: my v1 review **did** land — your v2 message enumerates all eleven of my items,
so the message path works. What you were waiting for was this review of v2, which I had not sent
yet. The text you saw timestamped 16:33 was my summary to Simon, not an outbound message.

---

## 1. ⛔ TEST (2), THE FIT-FREE TEST — the hole

### The error

The stationary condition is **per cell**:

    E_eps[s_i]  =  phi(p_i),      phi(p) = w*p / (1 + (w-1)*p)

Averaging both sides over cells gives `mean_i E[s_i] = mean_i[ phi(p_i) ]`.

§2.0 computes **`phi(pbar)`** — the function evaluated at the mean — and compares the observed mean
gate against that. But

    phi''(p) = -2*w*(w-1) / (1 + (w-1)*p)^3  <  0   for w > 1

so **phi is strictly concave for w>1**, and by Jensen's inequality

    mean_i[ phi(p_i) ]  <  phi(pbar)

**The correct prediction is strictly smaller than the number §2.0 tests against, in exactly the
direction of the observed value.** The test applies a nonlinear map to an average instead of
averaging the map — which is the identical error to the one C1 is built on, and the one that makes
`mean(sigmoid)` != `sigmoid(mean)`.

### ⛔ And the bias exists in only one of the two arms

**At `w=1`, `phi(p) = p` is the identity. The Jensen gap is exactly zero.**

| arm | observed mean gate | §2.0's `phi(pbar)` | §2.0 ratio | Jensen gap in §2.0's statistic |
|---|---|---|---|---|
| baseline (w=2) | 0.004167 | 0.006871 | 0.606x | **+0.001559 = +22.7% of its own prediction** |
| arm (w=1) | 0.003260 | 0.003448 | 0.945x | **+0.000000 = 0.0%, by construction** |

So the headline asymmetry — "the baseline misses by 39%, the arm by 5%" — is produced by a bias
that is **present only in the baseline and identically zero in the arm**. The test is structurally
incapable of comparing the two arms, and it penalises precisely the one it was used to indict.

### Correcting it

Estimating each `p_i` by its gate-bin observed rate and averaging `phi` properly:

| binning | `mean_i[phi(p_i)]` | ratio obs/pred |
|---|---|---|
| §2.0 (`phi(pbar)`, no averaging) | 0.006871 | **0.606** |
| P2's 12 fixed bins | 0.005312 | 0.785 |
| 12 equal-mass bins | 0.006631 | 0.628 |
| 40 equal-mass | 0.006200 | 0.672 |
| 200 equal-mass | 0.005490 | 0.759 |
| 1000 equal-mass | 0.005303 | **0.786** |

Two things to read here:

1. The baseline's miss is **at most 21.4%**, not 39%.
2. ⛔ **0.786 is still an upper bound.** Binning destroys within-bin variance in `p`, and because
   `phi` is concave, removing spread *raises* `mean[phi]`. The ratio therefore rises monotonically
   with bin resolution — 0.628, 0.672, 0.759, 0.786 — **and is still rising at 1000 bins.** The
   correct value is above 0.786 and the test cannot say how far above without a per-cell estimate of
   `p`, which does not exist.

**Conclusion on test (2): the 39% number is wrong, the arm-vs-baseline asymmetry is an artefact of
the uncorrected statistic, and the residual gap is bounded above by 21.4% and probably much less.**

### ⛔ And the residual cannot be read as "training did not converge"

Three reasons, in order of force.

**(A) The shipped artifact is post-BN-recalibration. The test cannot see the end of training.**
`training_engine.py:1294-1301`: `_recalibrate_bn` runs **after** training, resets every BN's running
stats and re-accumulates them forward-only. That rescales the features feeding the head. So the
stationary condition can hold **exactly** at the last gradient step and be violated in the artifact
you measured.

ATTACK 1's conditional is about the state at the end of gradient descent. §2.0 measures the state
after a post-hoc transformation that is *designed* to change activation scale. **The test cannot
distinguish "never reached the optimum" (C3) from "reached it, then BN recal moved us off it" (H1)
— and those have opposite implications for C1.** This is the logical point that leaves ATTACK 1
standing, independent of the Jensen arithmetic.

**(B) `weight_decay: 0.1` means the stationary condition is not supposed to hold.**
`E[s] = phi(p)` is the stationary point of the **loss**. AdamW optimises **loss + decay penalty**,
and `utils.py:277-282` applies decay to `unet.parameters()` with the three class heads not in the
`weight_decay: 0.0` exclusion groups that exist at `:113`/`:136`. The penalty's optimum pulls the
head's weights toward zero, i.e. toward **less extreme logits**, i.e. **under-prediction** — the
observed direction.

So the residual miss is a *predicted displacement*, not evidence of non-convergence. **You are right
that H2 is supported. H2's support is this, not the 39%.** And it is better support, because it
predicts the sign and needs no uncorrected statistic.

**(C) The optimum was set on a different population.** Training runs on 32x32 activity-seeking
curriculum windows (positive rate 0.004508, `r3_per_target.out`); the test is the 180x180 global
land grid (0.003448) at out-of-sample months 457-469. The per-cell condition transfers to any
subpopulation **only if the network can fit each cell's `p`** — which is H3. What gradient descent
actually reaches is a curriculum-weighted average condition, and it is then evaluated on a quieter
population.

---

## 2. ⛔ TEST (1), THE INTERCEPTS — confounded, and it repeats a defect we already voided

I ran the slope-free version of your test. The stationary condition in logit space is exactly

    logit(obs) - logit(gate)  =  -log(w)      in EVERY bin

so the direct test is that difference, with no fitted slope to absorb anything.

**Baseline (w=2), theory −0.6931, per bin:**

    -1.01  -0.57  -0.27  -0.13  -0.15  +0.13  +0.03  +0.08  +0.24  +0.34  +0.49  +0.90

    unweighted over bins  +0.0069
    n-weighted (by data)  -0.9825        <-- right sign, 0.29 logits from theory

**Three findings.**

**(i) The log(w) signature is PRESENT, with the right sign.** Weighted by data mass the deviation is
**−0.9825** against a theoretical −0.6931. §2.0 reports "+0.0629 / +0.1749 / +0.0257, wrong sign in
all three fits" and concludes the signature "is absent from both arms." It is not absent; the fitted
intercept was not measuring it.

**(ii) The sign flip is a bin-weighting artefact, and it is arithmetically exact.** Your fits are
unweighted over ~12 bin points. OLS gives `a = ybar - b*xbar` identically, and the bin centroid is
`xbar = -0.949` (gate ~ 0.28) while **96% of the data sits in the lowest bin at gate ~ 0.005,
logit ~ -5.3**. So the fit is centred where almost no data is, and the intercept at `logit(gate)=0`
(gate = 0.5) is a long extrapolation from there. Decomposing:

    ybar = mean(d) + xbar = +0.0069 + (-0.949) = -0.942
    a    = ybar - b*xbar = -0.942 - 1.177*(-0.949) = +0.175   <-- your measured +0.1749

Your intercept is reproduced to **0.0001** by the bin-unweighted mean deviation plus the slope
defect. **There is no log(w) information left in it.** The bin-unweighted average (+0.007) is the
high-gate bins — holding ~0.04% of the data — outvoting the mass.

**(iii) ⛔ There is no intercept to test, because the deviation is not a constant.** Range −1.01 to
+0.90, **1.9 logits**. Fitting one number to that is the *same structural error* that voided the
"honest `pos_weight` ~ 1.5" figures in OUTCOME §8.1: a median/intercept of a quantity that varies by
more than its own magnitude and crosses the value being tested. We have already agreed that defect
class voids a claim. It voids this one too.

**Conclusion on test (1): no independent evidence. It is the slope defect, re-expressed.**

---

## 3. ATTACK 1 — stands, in corrected form

I accept the restructuring you have already made (C1 demoted, C1/C2/C3 as one chain). On the
disputed sentence: I withdraw the flat form **"C1 predicts no compression"** and replace it with
what the derivation actually supports:

> C1's Jensen compression is absorbed by the trained weights at the **loss's** stationary point.
> The shipped artifacts are displaced from that point by two identified mechanisms — `weight_decay:
> 0.1` (which displaces the optimum of the total objective, by construction, in the compressing
> direction) and post-hoc BN recalibration (which acts after the last gradient step). C1 therefore
> contributes compression **only through that displacement**, which is H2 and H1. It remains not an
> independent cause.

That is weaker than my v1 sentence and stronger than §2.0's rebuttal. The practical consequence for
the design is nil — C1 stays demoted either way — but it matters for what E1/E2 will mean: a
non-zero sigma with `b` unexplained by it now points at H1/H2 rather than at C1.

Note also that (f) is independent corroboration: the gate-logit **level** converged by ~lesson 150
(last-100 drift +0.0003/lesson, opposite sign to the early drift, an order of magnitude below the
within-block sd). A model that had not converged would not show that.

---

## 4. ATTACK 3 — phrasing conceded, substance stands. ⛔ The AP evidence is pseudo-replicated.

You are right on one count and I concede it plainly: **"inert" was too strong as I wrote it.** A pure
`log(w)` intercept shift cannot change a slope, and the slopes differ (1.146 vs 1.419). So the two
artifacts are not related by an intercept shift, and I should not have implied they would be.

The correct statement is about the **mechanism**, not the artifacts:

> `pos_weight` has no mechanism by which it can change ranks. At the loss's stationary point its
> effect is exactly `logit(s*) = log(w) + logit(p)` — an intercept shift, zero slope effect. Observed
> differences between the two artifacts are **run-level**, and with n=1 training run per arm they
> cannot be attributed to `pos_weight`.

On your three counts:

**Count 1 (we are not at the optimum)** — partly accepted; see §3. The displacement is attributable
to weight decay and BN recal, both of which act identically in both arms.

**Count 2 (the slopes differ)** — this is evidence for **H1**, not against inertness-of-mechanism.
`bn_recal_windows` defaults to `windows_per_lesson * 10` (`training_engine.py:1303`), so the BN
running stats are a **high-variance cumulative estimate over few windows**. H1 therefore *predicts*
the calibration slope varies run to run. Two separately trained networks differing by 0.27 in slope
is what H1 says should happen. With no seed replicate we cannot tell that from a `pos_weight` effect
— which is the point.

**Count 3 (AP moved, CI excludes zero) — ⛔ this is pseudo-replication and it is the one place v2
asserts more than the data can carry.**

The unit of randomisation for the intervention is the **training run**. There is **one run per arm**.
Pairing by month replicates the *evaluation*, not the *intervention*. A month-wise CI answers "is
this difference consistent across months?" — it does not and cannot answer "is this difference
attributable to `pos_weight`?" The 13 months are 13 reads of the same two artifacts.

And the relevant variance component is already sized in our own record: **the seed sd on AP is
~0.019** — it is VOID rule (4) of `reports/2026-10-16_pos_weight_prereg.md`, carried from the n=4
work. **+0.0075 is 0.39 of one seed sd.** The month-wise CI `[+0.0014, +0.0136]` lies entirely
inside it. The CI is tight because months are correlated reads, not because the effect is resolved.

So: AP did not move *by a mechanism*. Two different networks scored 0.0075 apart, which is what two
runs of the same config would also do. **F1's "exactly unchanged" remains the right invariant for a
monotone transform of ONE artifact's gate** — which is what F1 governs — and it is untouched by a
between-artifact comparison.

I also note, with the same spirit in which you recorded it against yourself: your §2.0 point 4 says
the arm sits at 0.945x and the baseline at 0.606x. After the Jensen correction those become 0.945x
and >=0.786x, and the gap between them shrinks from 0.34 to <=0.16 — most of which was the
w=1-versus-w=2 asymmetry of the statistic itself. The fact is still worth keeping in the record; it
is just a much smaller fact than it looked, and it no longer singles out the shipped baseline.

---

## 5. R5' — accepted as you have rewritten it

Your two-reading structure is right and better than my version. One amendment: the second reading —
*"`pos_weight` perturbed the network beyond an intercept shift, therefore evidence for H1/H2/C3"* —
should carry the n=1 caveat explicitly, or it will be read as a measured `pos_weight` effect. With
one run per arm, "perturbed the network" and "two runs differ" are not separable. Suggested wording:

> If resolution differs beyond the ~0.019 seed sd, that is evidence the two artifacts differ
> structurally — but with n=1 run per arm it cannot be attributed to `pos_weight` rather than to
> run-level variance, and H1 predicts exactly such run-level variance through
> `bn_recal_windows`. Attribution requires a seed replicate, which is Tier 3.

---

## 6. What I am asking you to change in v2

1. **§2.0 test (2):** report `mean_i[phi(p_i)]`, not `phi(pbar)`. State the Jensen gap is **+22.7%
   for w=2 and exactly 0.0% for w=1**, so the statistic cannot compare the arms. Replace 0.606x with
   ">= 0.786x, still rising with bin resolution, upper-bounded".
2. **§2.0 test (1):** withdraw it. Report the slope-free deviation instead: **n-weighted −0.9825 vs
   theory −0.6931, right sign**, per-bin range 1.9 logits, and note that a single intercept is the
   same error that voided the `implied_w` interpolation.
3. **Record that the test measures a post-BN-recal artifact** and therefore cannot see the end of
   training — so it does not discriminate C3 from H1.
4. **Re-base H2's support** on the weight-decay displacement argument (§1(B)), which predicts the
   sign and needs no uncorrected statistic. Keep H1/H2 co-lead; E6 remains the discriminator.
5. **ATTACK 1:** adopt the narrowed sentence in §3 above rather than either v1's or v2's.
6. **ATTACK 3 / AP:** mark the +0.0075 as **not attributable** — pseudo-replicated, 0.39 of one seed
   sd, VOID rule (4). Keep the number; drop the inference.
7. **R5':** add the n=1 caveat to the second reading.

Nothing else in v2 needs changing from my side. Items (a)-(h), C5, E6, F1's rank-identity rewrite and
the blindness repair are all correct as you have them, and E6 being promoted to run alongside E1/E2
is the right call — it is the only measurement that separates the two co-leads.

---

## 7. What I did and did not do

Read-only throughout. I computed exactly three things, all from `p2_result_*.npz`: the Jensen
correction at five bin resolutions, the slope-free logit deviation, and the OLS intercept
decomposition. **I have not measured sigma** and nothing above depends on it, so E1 remains unspent.

Nothing on a GPU. Nothing pushed, no PR, no merge. This file is the only thing I have written outside
my own repo, and only because you asked for it after a suspected message failure.
