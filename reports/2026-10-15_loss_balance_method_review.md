# Method review — the A-vs-B fork on the task-loss balance

**Date:** 2026-10-15 · **Target:** `reports/2026-10-15_loss_balance_prereg.md`
**Panel:** 7 seats, fanned out as independent agents (no cross-contamination).
**Chair's framing (standing):** no seat's criticism is accepted at face value. Every checkable claim
was verified against the code, the arrays or the source, and a seat that asserted something
checkable and wrong is recorded as wrong.

---

## RULING

**A — demoted from an experiment to a correctness fix. B struck. N on "the balance is the lever".**

**All seven seats ruled A or N. Not one backed B** — including Kendall (who authored the balancer)
and Sutton (seated specifically to argue for learning it rather than hand-setting it).

---

## 1. Three things the prereg got wrong

### ⛔ 1.1 B is not a fallback. This repository has already measured it as pathological.

`reports/results_balancer_bisect.md` (2026-06-05) is a **pre-registered, device-matched GPU/GPU
bisect** whose verdict line reads: *"H CORROBORATED — the C-111 fix (un-freezing the MultiTaskLoss
balancer) is the cause of the C-113 acute runaway."* Free-running attractor: **trainable log ~15–17
(`expm1` ~1e6–1e7, PATHOLOGICAL)** against **frozen log ~4–5 (~70–215, healthy)** — and the frozen
arm held across **all three** synthetic seeds and both `freeze_h` settings.

`reports/technical_risk_register.md:429` records the disposition: *"Fix direction: **regularise** the
balancer (bound/decay `log_vars`, lower its LR), **not permanent freeze**"*.

**So B-as-proposed — unfreeze, unregularised, `log_vars` outside the gradient clip (C-314 item 3) —
is precisely the one form the repository has already ruled against.** The prereg's §7a instead cited
Kendall's Table 1 and dismissed the repo's own experience as *"the guard bug, not the method"*. That
was wrong, and it omitted a dedicated pre-registered experiment on exactly this intervention.

⚠️ Held honestly: the bisect is single-seed per arm, is a synthetic-seed diagnostic, and concerns
the **free-running rollout** (C-113), not T=0 allocation. It does not prove B would fail at the fork's
question. It does establish that B is not a safe fallback, and the register's "regularise, don't
unfreeze" direction makes an unregularised B the wrong form of the idea.

### ⛔ 1.2 The 0.5 on the body has no derivation — and the joint likelihood IS reachable under the freeze

The Kendall seat read the paper. The factor of 2 is real and is in §3.2 eq (10):
`≈ (1/2σ₁²)L₁(W) + (1/σ₂²)L₂(W) + log σ₁ + log σ₂`. But it descends from eq (5),
`log p(y|f^W(x)) ∝ −(1/2σ²)‖y − f^W(x)‖² − log σ` — **the Gaussian exponent**, where `L₁` is a
*squared residual*. The classification branch gets `1/σ²` with no 2 because the temperature enters
as `Softmax((1/σ²)f^W(x))` (eq 8).

Our body loss is an **NB NLL — already a normalised log-density**. The 1/2 has nothing behind it.
`is_regression=True` on a likelihood-valued task is a category error.

**Verified consequence:** with `is_regression=False`, `coeffs = 1/((0+1)·1 + eps) = 1.0` and
`log(stds + eps) = log(1 + 1e-8) ≈ 0`. So **A's denominator + `is_regression=False` on the body = the
exact joint hurdle NLL, with the balancer still frozen.** The prereg's §1/§2 claim that the joint
likelihood is unreachable under the freeze is **false**.

⚠️ But do not bundle it into this arm. The control measured 0.593 *with* the 0.5, so keeping
`is_regression=True` leaves the denominator as the single variable. Flip the flag in a later arm.

### ⛔ 1.3 F8 cites a dead number, and F2 is sized ~4× too coarse

- **C-119 is RESOLVED** (`technical_risk_register.md:38` and `:5518`; bit-identical same-seed runs,
  78/78 `y_pred` `array_equal`, `daab1c1`). The *"~20% training variance"* figure is the **pre-fix**
  measurement. F8's C-119 clause must go; the live noise floor is the **0.019** seed sd.
- `OUTCOME.md:67-68` already demonstrates the design's real resolution: paired by month, the gate
  loss is **mean −0.0299, 95% CI [−0.0401, −0.0197], 0 of 13 months favouring the arm** — a
  half-width of **~0.010**, not the 0.0413 month-to-month sd the prereg used. F2/P-3 should use the
  paired-by-month contrast.

---

## 2. Four findings no seat raised (chair's own verification)

### 🔑 2.1 Only 11.3% of the model is shared — and under AdamW the constant is invisible on the other 88.7%

Measured on the delivered control artifact (`4acbf4d3…`):

| group | tensors | params | % of model |
|---|---|---|---|
| REG-only decoder (`*_reg`) | 54 | 402,540 | **44.39%** |
| CLS-only decoder (`*_class`) | 54 | 401,673 | **44.29%** |
| shared `bottleneck_conv` | 1 | 73,728 | 8.13% |
| shared encoder (`enc_conv0/1`) | 2 | 23,904 | 2.64% |
| shared BN + 4 ConvLSTM cells (4 channels each) | — | ~2,000 | ~0.2% |
| **shared total** | | **~100,000** | **~11.3%** |

The body and gate have **separate decoders**. They share only the encoder, the bottleneck, the
recurrent core and their BatchNorm.

And the optimiser matters: `torch.optim.AdamW` (`utils/utils.py:260`). **A per-task constant on a
loss is exactly invisible on parameters that only that loss touches**, because Adam normalises by the
gradient's own second moment. Verified numerically — 60 AdamW steps, scale 1.0 vs 65.5, on a
parameter touched by one loss: **max|difference| = 0.0**. On a parameter shared by two losses the
same test gives a real difference (1.6e-3), because there the constant tilts the summed gradient's
**direction**.

⛔ **So the 65× acted only as a direction tilt on ~11% of the weights.** That is far narrower than
the prereg's framing ("the term scale is the allocation rule for shared capacity"), and it cuts both
ways: a tiny shared bottleneck is where capacity conflict bites hardest, *or* each head's private
400k-parameter decoder absorbs the imbalance and the effect is second-order. **Unresolved, and it is
exactly what §4.1's free test settles.**

### 🔑 2.2 Under B the balancer would train at a constant LR while the network decays

`choose_scheduler` builds `WarmupDecayLearningRateScheduler` over an `AdamW` constructed with
**`unet.parameters()` only**, so `base_lrs` is captured with **one** param group. `log_vars` are
added as a *second* group later (`training_engine.py:134-136`), and `_LRScheduler.step()` zips
`optimizer.param_groups` against `get_lr()`, which returns `[lr] * len(self.base_lrs)` — **length 1**.
`zip` truncates, so **group 1's LR is never updated**, and `add_param_group` gave it the optimiser
default `lr = config["learning_rate"] = 0.001`.

Verified empirically after 900 steps: network group `6.011e-04` (on schedule), `log_vars` group
`1.000e-03` (**unchanged**).

`optimizer.step()` sits **inside** the window loop (`training_engine.py:1577`, indent 16, against the
loop at `:1378`, indent 12), so 300 lessons × 3 windows = **900 optimizer steps**. Under Adam a
persistent-sign gradient moves a parameter by ≈lr per step, so the balancer's **total travel budget
is ≈0.90 nats** against the **2.585 / 3.037 / 4.135 nats** its own fixed point requires for the three
gates.

⛔ **B cannot reach its fixed point on this vehicle.** It would test a ≤2.3× nudge, confounded with a
decaying network LR. (The Kendall seat reached ~0.81 nats by assuming the schedule *does* apply to
`log_vars` — right ballpark, wrong mechanism. The real number is 0.90 and the real finding is the
silent LR inconsistency, which is a defect in its own right.)

⚠️ Also noted, not chased: `optimizer.zero_grad()` is at `:1361`, **outside** the window loop, so
gradients accumulate across the 3 windows of a lesson while still stepping each window.

### 🔑 2.3 The "both halves under-reach" fact dissolves — and was never evidence for N

The prereg called this *"the most interesting unexplained fact in the record"* and treated it as the
strongest case for N. It has **two separate explanations, one per half**, and therefore points to no
common cause:

- **The gate.** Its objective is **improper**. Gneiting & Ranjan (2011) Thm 1: for proper `S₀` and
  non-constant weight `w`, `S = w(Y)·S₀` is improper and its expected-score minimiser is the *tilted*
  law. For `pos_weight = 2`, the weighted-BCE first-order condition gives `g* = wp/(1+(w−1)p)` ≈
  **2p**. This is already in our own output — P2's `s*(w=2)` column. So the gate's own objective wants
  **2p**, and the gate delivers **p**. Under-reaching its objective means **landing on the truth**.
  That is the gate being better than what it was trained on, not a defect.
- **The body.** `θ = 0.1544` against a conditionally-honest `0.4323`, in the `total_count`
  parameterisation where `Var = μ + μ²/θ` — so **lower θ is MORE over-dispersion**. Machete (2012)
  Prop 2.1: the log score prefers to err with *more* uncertainty. An NLL-trained body on a
  heavy-tailed rare target buys insurance against `−log p → ∞` by inflating dispersion. The
  over-dispersion is what its objective asks for.

**Two halves, two different mechanisms, both literature-grounded. The common-cause reading is
retired.**

### 🔑 2.4 The real argument for A is an estimator argument, not a balance argument

`weighted_nll_mean` divides by `weight.sum().clamp_min(1e-8)` — **the per-step positive count, a
random variable**. Measured positive cells per 32×32 window, from the delivered arm's own
per-window fractions:

| target | mean P | median P | min P | windows averaging <1 positive | <2 |
|---|---|---|---|---|---|
| `lr_sb_best` | 11.57 | 9.69 | 0.00 | 25% | 28% |
| `lr_ns_best` | **1.16** | **0.88** | 0.00 | **61%** | **78%** |
| `lr_os_best` | 4.24 | 4.90 | 0.02 | 33% | 42% |

A zero-positive step yields a graph-connected **exact 0** (the function's own docstring). A
one-positive step weights that **single cell's NLL at full strength**.

⛔ **So for `lr_ns_best` the body term is a lottery**: silent in most steps, then a single cell at
~1000× leverage. `E[Σ/N_pos] ≠ E[Σ]/E[N_pos]`, so the gradient is both **biased and high-variance**.
Dividing by N (A) is the unbiased Monte-Carlo estimator of the hurdle log-score's body term.

**This reframes A entirely.** It is not a rebalancing lever whose effect on AP we predict — it is the
removal of a biased estimator. And it **refutes the prereg's own §3.1 objection**: `ns` ending up at
0.058× its gate term is not "effectively unsupervised", it is the likelihood stating honestly that a
target with 0.88 positives per window carries almost no magnitude information. The control's
near-equal per-target ratios (0.65 / 0.53 / 0.56) are the **artefact** — produced by diluting the
body with zeros carrying no magnitude signal (measured: zeros supply **51.7% / 88.0% / 67.8%** of the
control's body term).

---

## 3. Seats recorded as wrong

| seat | claim | chair's verdict |
|---|---|---|
| Murphy | *"a truncated body fitted on positives only is relieved of explaining the zeros, so it should report lower dispersion — the hurdle working"* | **REFUTED, twice.** θ is `total_count`, so θ 0.154 < 0.432 is **more** over-dispersion, the opposite direction. And 0.1544 was measured on the **control (plain `nb`)** — a non-truncated body — so the premise does not apply to the artifact the number came from. |
| Gelman | *"one new point placed between two known points on the same side of a 1-D curve has no discriminating power at any n"* | **REFUTED as stated.** A (0.225) is **not** between control (0.593) and arm (35.2); it is beyond the control, on the far side. ⚠️ Residual that survives: A lands 0.368 from the control on a scale whose damaging contrast spans 34.6, so **A ≈ control under both hypotheses** — but A-vs-**arm** is still discriminating, and the arm is the pre-registered contrast. |
| Gneiting | *"median \|gate − p\| = 0.0054 exceeds the base rate 0.0034, so it is not evidence of calibration"* | **REFUTED as stated.** It compares a cross-bin median absolute error against a *global* base rate. The 5 bins with n≥1000 span observed p from **0.0002 to 0.239**; per-bin relative errors are 50% / 34% / 3.4% / 4.1% / 2.3%, and the branch test's tolerance was the 0.0298 half-separation. P2's actual job — separating D1 (`2p/(1+p)`, 0.0573) from D2 (`p`, 0.0054) — is a 10× margin and unaffected. ⚠️ Residual: the gate's *relative* calibration in the two lowest bins, which hold ~99% of the mass, is genuinely poor. |
| Murphy | *"Cragg (1971) §3 has no decomposition lemma; eqs (7)/(9) are a truncated-**normal** two-part model, not a count hurdle; the separability result you want is Mullahy (1986) eq (16) p. 346"* | ⛔ **NOT VERIFIED by the chair — I did not read the PDF.** If true it is a **library data-integrity defect**: our `Cragg1971` claim is marked *proven/passage* with provenance *"Section 2, equations (7) and (9)"*, and we have built the separability argument on it. **Must be checked against the source before either paper is cited again.** |
| Kendall | *"Σ lr = 0.8122, so v travels ≤0.81 nats"* | **Conclusion right, mechanism wrong.** The schedule never reaches `log_vars` (§2.2); the budget is 0.90 nats at a constant lr. The ruling against B stands and is better grounded than the seat's version. |
| Sutton | *"the bitter lesson is a category error here; six scalars do not scale with computation"* | **CONFIRMED**, and credited: the seat seated to argue for B dismantled its own argument on the record. |

---

## 4. Recommendation — ordered, costed

### 4.1 ⛔ FIRST, and free: the trunk gradient-conflict test

**Three seats proposed this independently** (Murphy, Gelman, operational) and it is the one test that
can invalidate the entire diagnosis. Allocation **requires conflict**. On the **shared ~11.3%** only
(§2.1), at both delivered artifacts, compute per target pair:

- `‖∇_shared L_body‖` against `‖∇_shared L_gate‖` — GradNorm's actual quantity (Chen et al. 2018:
  imbalance *"manifest[s] as imbalances between backpropagated gradients"*, not as loss values), and
- `cos(∇_shared L_body, ∇_shared L_gate)`, pairwise across all six tasks.

**Pre-committed reading.** If the shared-trunk gradient norms are already comparable despite the 65×
loss-value gap, **or** the cosine is ≥ 0, then there is nothing to allocate, the loss-value ratio
measured nothing, and the ruling becomes **N with no arm bought**. The pairwise form also separates
"gate vs body" from "`lr_ns_best` eats the trunk".

**Cost: $0.** Both artifacts are on local disk. This runs before anything else.

### 4.2 Free: the prereg corrections

Strike B from §7 and cite `results_balancer_bisect.md:9-13`; drop the C-119 clause from F8; re-size
F2/P-3 to the paired-by-month ±0.010; demote A's hypothesis from "recovers AP" to "restores the
unbiased estimator"; replace §3.1's objection with §2.4's reframe; record §2.1–§2.3.

### 4.3 Free: Gneiting's one-parameter composed-CRPS recalibration

On the delivered weights, rescale `θ` (and separately a Platt slope on `g`) to minimise the
**composed** 16-member CRPS on validation. Minutes on CPU, no retraining. It either quantifies the
NLL-vs-CRPS divergence the prereg names as a limitation, or **retires it as measured-and-negligible**
for the price of an afternoon. GR07 §9.1's optimum-score estimation, and FAO-02-legal because it
optimises the locked metric.

### 4.4 $2.25: run A, as hygiene

`is_regression` left alone, so the denominator stays the single variable. F1 (the §6a pre-flight) is
a free hard stop before the spend.

### 4.5 Later arms, better motivated than A was

- **`is_regression=False` on the body** ⇒ the exact joint hurdle NLL (§1.2).
- **`pos_weight → 1.0`.** Gneiting derived it independently from G&Ranjan Thm 1 and from CRPS's k=0
  threshold term being the occurrence Brier score at weight 1; it matches P2's measured implied
  **1.0365**. ⚠️ This is a **better-motivated single-variable arm than A** on skill grounds, and it
  is the one the panel's reasoning most strongly supports. It is *not* in the current prereg.

### 4.6 Instrumentation, free, do regardless of the ruling (Sutton seat's main contribution)

Log per step the 6 raw task losses, the 6 coefficients and the derived `0.5·Σbody/Σgate`; assert a
declared band and key guards on **magnitude, never on the sign of a log** (C-312's exact failure).
Set `trajectory_log_path` default-on and widen it from 2 pooled columns to **6 per-task** — r3 showed
the finding was per-target. Then 0.593 → 35.2 is visible at step one of the next arm instead of
forensically.

---

## 5. Strongest dissent, kept live

**Harrell, and it is cheaper than the fork.** The information arithmetic (RMS 2nd ed. §4.4 limiting
sample size): `m_gate/m_body = 3(1−q) ≈ 2.99` at any rarity, so the occurrence question structurally
carries **3×** the information of the magnitude question — which is why the gate is calibrated and
the body is not. His decisive experiment is **not on the fork**: two single-task arms, gate-only and
body-only on the same trunk, **~6 h, ~$4.50**. If each alone reaches its own optimum, allocation is
real. If each alone still under-reaches by ~2–3×, the cause is common (identification / sparsity) and
the balance question is closed permanently. Nothing in the record does this.

⚠️ §2.3 weakens his premise — the two halves' shortfalls now have separate explanations — but the
ablation remains the only direct test of whether the shared trunk constrains either head at all, and
it subsumes §4.1's question with a measurement rather than a proxy.

**Sutton's dissent, also live:** the hand-engineering that should go is the hurdle itself. A per-cell
**categorical over ~32 log-spaced count bins** — one head, one loss, no gate, no balancer, no
denominator ambiguity, zero as a learned bin rather than an imposed spike. 3 seeds × ~3 h ≈ **$7**,
needs a head rewrite. He grounds it in our own plateau: retention 0.690 → 0.692 from 160 to 300
lessons (**+0.0014**) is, by his criterion, the signature of a method that has stopped scaling.

---

## 6. Methodological risks (register-compatible — NOT appended; route via `register-risk`)

| ID | Tier | Trigger | Location | Narrative |
|---|---|---|---|---|
| **M-1** | **2** | Any arm that unfreezes `freeze_multitask_balancer` | `utils/utils.py:258-273`, `train/training_engine.py:134-136` | The LR scheduler's `base_lrs` is captured over `unet.parameters()` only; `log_vars` are added as a later param group and `_LRScheduler.step()` zips against a length-1 `get_lr()`, so the balancer trains at a **constant `lr=0.001`** while the network follows WarmupDecay. Verified: after 900 steps, network `6.011e-04`, balancer `1.000e-03`. Silent; no error signal. Total balancer travel ≈0.90 nats against the 2.6–4.1 its own fixed point needs. |
| **M-2** | **2** | Any arm whose body loss uses the positives denominator on a rare target | `distributions/nb_core.py:62-79` | The denominator is the per-step positive count — a random variable that is **0 in most steps for `lr_ns_best`** (median 0.88 positives per window; 61% of windows average <1). Zero-positive steps contribute an exact 0; one-positive steps weight a single cell at full strength. The gradient is biased (`E[Σ/N] ≠ E[Σ]/E[N]`) and high-variance, with no error signal. |
| **M-3** | **3** | Citing `Cragg1971_LimitedDependentVariables` for hurdle separability again | `~/brain/9_library/papers/_meta/Cragg1971_LimitedDependentVariables.json` | A seat reports, from the PDF, that eqs (7)/(9) are a truncated-**normal** two-part model with no decomposition lemma, and that the separability result is **Mullahy (1986) eq (16) p. 346**. Our claim is marked *proven/passage* with that provenance and the separability argument rests on it. **Chair has not verified.** |
| **M-4** | **3** | Reading the `0.5` body coefficient as derived | `utils/mtloss.py:65`, `utils/utils.py` (`is_regression` construction) | Kendall's factor of 2 descends from the **Gaussian exponent** (eq 5 → eq 10), where the task loss is a squared residual. An NB NLL is already a normalised log-density, so `is_regression=True` applies an underived 1/2 to a likelihood. No published extension of homoscedastic-uncertainty weighting to a count or hurdle likelihood exists. |
| **M-5** | **3** | Sizing any falsifier from C-119's variance figure | `reports/2026-10-15_loss_balance_prereg.md` §5 F8; `reports/2026-10-12_composition_arm_prereg.md` §5 | C-119 is **RESOLVED**; its "~20% training variance" is a **pre-fix** measurement and is not a live noise floor. Two pre-registrations size falsifiers from it. The live floor is the 0.019 seed sd, and the demonstrated paired-by-month resolution is ±0.010. |

---

## 7. Library grounding

**Held and used:** `Kendall2018_MultiTaskUncertainty` (read in full by the Kendall seat — §3, §3.1,
§3.2, eq 5/8/10, Table 1, App. A.1/B), `Mullahy1986_ModifiedCountDataModels`,
`Cragg1971_LimitedDependentVariables` (⚠️ provenance disputed, M-3), `Machete2012` (Prop 2.1, 3.6),
`Gneiting2007_StrictlyProperScoringRules` (§4.2, §9.1, eq 6, eq 44),
`Gneiting2011_ComparingDensityForecasts` (Thm 1 — the `pos_weight` impropriety),
`Hersbach2000_DecompositionCRPSEnsemble`, `Jordan2019_EvaluatingProbabilisticForecasts`,
`Lerch2017_ForecastersDilemmaExtremeEvents`, `Kong2020_DeepHurdleNetworks`.

**Fetched from the web this review (none held locally):** Kurin et al. 2022 *In Defense of the
Unitary Scalarization* (arXiv 2201.04122) — ⚠️ **does not evaluate uncertainty weighting**, so it
neither validates nor refutes B; Xin et al. 2022 (arXiv 2209.11379) — *"MTO methods do not yield any
performance improvements beyond what is achievable via traditional optimization approaches"*;
GradNorm (Chen et al. 2018) — imbalance acts **through the gradient**; Groenendijk et al. 2021
(arXiv 2009.01717) — uncertainty weighting fails `Σαᵢ = 1` and is therefore coupled to the global LR;
Standley et al. 2020 (task *grouping*, not weighting); Rasp & Lerch 2018 (CRPS-trained networks);
Dimitriadis–Gneiting–Jordan–Vogel triptych (arXiv 2301.10803).

**Gaps worth fetching:** Kurin 2022 and Xin 2022 (the two large-scale MTO null results, both now
load-bearing), Groenendijk 2021 (the LR-coupling defect, which is M-1's literature anchor), Chen 2018
(GradNorm, the gradient-vs-loss distinction that §4.1 tests), Cameron & Trivedi 2013 ch. 4.
