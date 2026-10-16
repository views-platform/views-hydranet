# Pre-analysis plan — does normalising the body NLL over all cells restore the balance?

**Written 2026-10-15, BEFORE the arm has been trained.** One arm, one variable, one readout.
Follows the measured outcome of the `gated_truncNB` composition arm
(`reports/2026-10-13_composition_arm_outcome/`), which is closed.

⛔ **Nothing launches on my say.** Simon approves the code change and the spend.

⛔ **This is a NEW pre-registration, not an amendment.** `reports/2026-10-12_underfire_diagnosis_prereg.md`
has met its definition of done and its stop rules forbid new questions.
`reports/2026-10-12_composition_arm_prereg.md` pre-registered a *composition* change; this is a
*training-objective* change. Neither is extended.

---

## 0. The established facts this starts from

All measured on the two delivered 2026-10-12/13 global-land vehicles, on the real curriculum
windows, in `eval()` mode, with `target_weights` asserted all-1.0 (C-87).
Drivers: `r2_loss_scale.py` (`a904673`), `r3_per_target.py` (`abf4293`); results `64607cd`;
the correction to my own earlier reading, `8f0610d`.

| | | source |
|---|---|---|
| every views-models config pins `freeze_multitask_balancer: True` | so `log_vars` sit at their **zero initialisation** | 10/10 configs |
| ⇒ the objective is the **constant form** `total = 0.5·Σbody + 1.0·Σgate` | the 0.5 is `1/((is_regression+1)·1)`; a regression task can never get 1.0 | `utils/mtloss.py:61-66` |
| nothing reads the task-loss terms; no feedback path exists | — | `mtloss.py`, `training_engine.py:855-868` |
| pooled positive-cell fraction in the sampled windows | **0.004508** (1/222) | r2 |
| **control** (`nb`): `0.5·body` against `gate` | **0.6×** | r2 |
| **arm** (`truncated_nb`): `0.5·body` against `gate` | **35.2×** | r2 |
| the family swap alone, on **identical predictions** | **65.5×** | r2 |

**The mechanism, stated plainly.** `weighted_nll_mean` (`distributions/nb_core.py:62-79`) divides by
`weight.sum()`, and the truncated family passes the `y>0` mask while `nb` passes `weight=None`. So
the **NLL denominator is coupled to the family**: swapping the family silently swapped the
denominator from all cells to positive cells, multiplying the body task-loss term by ~222 (and ~65×
after the truncated law's own renormalisation by `1 − NB(0)`). On a **shared trunk** with a frozen
balancer, the term scale *is* the allocation rule for shared capacity — so the trunk was optimised
for magnitude and the gate inherited whatever was left.

### ⛔ What is NOT the finding

The positive-cell denominator is **not a defect**. It is deliberate, documented as *"the defining
guarantee #3"*, pinned by `tests/distributions/test_truncated_negative_binomial.py:104-126`
(`test_nll_reduces_over_positives_not_all_cells`), and implemented independently a second time in
the legacy `utils/truncated_nb_loss.py:92`. Under **Mullahy (1986)** separability a positive
constant on one part of a hurdle cannot move that part's own MLE — so the denominator is
statistically **immaterial for independently parameterised parts**. I claimed it was a defect and
withdrew that.

⛔ **Citation corrected 2026-10-15 (Amendment 1, §4).** This originally cited **Cragg (1971)** §2
eqs (7) and (9). Cragg **never states separate estimation** — his only relevant line is *"Maximum
likelihood estimates may be calculated by iterative methods"*, on the **full** likelihood — and his
model II is a continuous **truncated normal**, not a count hurdle.
**`Mullahy1986_ModifiedCountDataModels`** states separability verbatim (*"can be obtained by separate
maximization of Λᵖ¹ and Λᵖ²"*) and is the right model class. Cragg remains the correct citation for
the two-part *model* and for the likelihood *factorisation*, not for separate estimation.

**What makes it matter here is that our shared trunk violates Mullahy's premise.** The parts are not
independently parameterised; they share every encoder weight. That is the whole finding.

### ⛔ A false claim in our own prose (a new C-303 instance)

Three places assert that the equal-weight sum **is** the joint hurdle-NB NLL:

- `views_hydranet/utils/weighted_bce_loss.py:6-7` — *"Being a proper NLL, it is additive (in nats)
  with the truncated-NB body, so their equal-weight sum is the joint hurdle-NB NLL (no reg-vs-cls
  balancer — C-141)."*
- `views_hydranet/utils/truncated_nb_loss.py:4-6` — the same claim.
- views-models `models/amber_kestrel/configs/config_hyperparameters.py:43` (on `development`).

**All three are false as written**, on three independent counts: the denominators differ (positives
vs all cells), the weights are 0.5 and 1.0 rather than equal, and `loss_class_pos_weight = 2.0`
makes the gate term a *weighted* Bernoulli NLL rather than a proper one. The prose is not fixed by
this arm; it is fixed *by* this arm, because under the all-cells denominator the first count stops
being true.

---

## 1. Hypothesis

**H1.** The `gated_truncNB` arm's loss of gate skill is an **allocation** effect, not a limit of the
composition. Normalising the zero-truncated body's NLL over **all cells** — the denominator the joint
hurdle NLL actually has — restores the body term to the same order as the gate term, and the gate
recovers without any change to the composition, the family, or `pos_weight`.

---

## 2. The intervention — ONE variable

The **NLL denominator** of the zero-truncated body, per target. Nothing else.

| | arm-as-run (`amber_kestrel`) | this arm ("A") |
|---|---|---|
| **the NLL denominator**, body | **positives** (`weight = y>0`) | **all cells** (`weight=None`) |
| `output_distribution` | `truncated_nb` | `truncated_nb` — **unchanged** |
| `forecast_composition` | `soft_gate` | `soft_gate` — **unchanged** |
| `freeze_multitask_balancer` | `True` | `True` — **unchanged** |
| `loss_class_pos_weight` | 2.0 | 2.0 — **unchanged** |
| `body_supervision` | `all` | `all` — **unchanged** |
| seed 42, 300 lessons, `window_dim` 32, `windows_per_lesson` 3, `sigmoid`, `min_events` 5 | — | identical |

⛔ **The denominator must become a config axis, not a second meaning for an existing dial.** It is
today an undeclared consequence of `output_distribution`, which is itself already dual-purpose: it
is read at `utils/utils.py:183` to build the **training loss** and at
`utils/hydranet_inference.py:266-270` to pick the **emit-time draw**. One value, two jobs. Per C-85
every scale is a config field from day one, and per the standing convention it is **opt-in with no
shadow default** (AST-enforced, `tests/test_falsification_magic_numbers.py:31`), with a
reject-if-ignored `model_validator`.

⛔ **Ask before writing it:** which non-training forwards traverse the new path? The C-328 precedent
(augmentation leaking into BatchNorm recalibration) says this question is answered up front, not
after.

---

## 3. ⛔ Skepticism ledger

1. **A does not equalise the three targets; the `nb` family's all-cells denominator does.** r3 measured the
   body-budget shares as 0.580 / 0.101 / 0.319 under `nb` and 0.449 / 0.217 / 0.334 under the
   truncated family. Projected forward, A gives per-target `0.5·body/gate` of **0.295 / 0.058 /
   0.166** — a **5× spread**, against the control's **0.593** for the same ratio (r3; the same
   quantity r2 reports pooled as 0.6×). **`lr_ns_best`'s
   magnitude ends up effectively unsupervised**, 0.058× its own gate term and ~9× below the control.
   This is the clearest reason A might be the wrong point even if it is the *defined* point.
2. **A overshoots past the control.** `0.5·Σbody/gate`: as-run **35.2**, control **0.593**,
   A **0.225**. A does not land between them; it lands on the far side. The frozen balancer's 0.5
   coefficient is the reason — a regression task structurally cannot be weighted above 0.5.
3. **Seventh intervention on the same lever.** Six have failed and four failed on firing (M45
   `truncated_nb` ×1170 → −0.238 AP; M62 ×3.0; M64 ×56). This one does not touch the fed-back field
   or the gradient through it — it changes the NLL denominator — which is a real difference in
   kind. But M45 also had a mechanism story.
4. **We are tuning under NLL and scoring under CRPS + AP.** See §4a. This is the reservation I am
   least able to retire.
5. **The whole argument rests on a mechanism nobody has instrumented *during* training.** r2/r3 are
   reconstructions from delivered weights in `eval()` mode, on a sampler whose RNG stream is not
   production's. They establish the magnitudes, not that the magnitudes caused the AP loss. See §6
   for what the arm logs so that stops being true.
6. **My own position oscillated A → B → A**, each move driven by a named new fact. Recorded in §8,
   because the oscillation itself is a finding about how determined this choice is.

---

## 4. Pre-registered predictions

Primary first. The **contrast set is the arm**, not the control — A changes one variable relative to
`amber_kestrel` and two relative to `violet_visitor`.

| | endpoint | prediction | threshold |
|---|---|---|---|
| **P-1** | pooled `0.5·Σbody / gate` at lesson 1, measured by the §6 pre-flight | **0.225** | within **±30%** (0.16–0.29) ⇒ the intervention did what it is defined to do |
| **P-2** | per-target `0.5·body/gate`, same readout | 0.295 / 0.058 / 0.166 | each within ±30% |
| **P-3** | gate AP, all land cells, **paired by month** over the 13 h=1 months | **recovers toward the control's 0.5631** | the paired mean difference against the arm-as-run is **> +0.010** (the design's demonstrated half-width — Amendment 1, §3) |
| **P-4** | emitted positive fraction, 100-cell set, h=1 | stays in the composition arm's pre-registered **[0.62, 0.76]** | the composition is unchanged, so this should not move |
| **P-5** | `act_ratio` at h18 | **< 3×** the control's | the M45 lever, unchanged from the composition prereg |
| **P-6** | `crps_all` | does not worsen | **> 5%** worse fails FAO-02's own bar |

### 4a. ⛔ The limitation, stated up front — we optimise one score and judge by another

**Machete (2012)** (`Machete2012`, C-1419–C-1422) establishes that **CRPS and the logarithmic score
prefer different directions of departure**, and that CRPS cannot distinguish distributions with
equal L2 CDF error. We are re-allocating the trunk under an **NLL** objective and scoring the result
under **CRPS + AP**, which FAO-02 has LOCKED.

**So the weight that minimises the training objective need not maximise the score we judge by.** A
null on P-3 therefore does *not* establish that the allocation was already right — it is also
consistent with the allocation being right under NLL and CRPS not rewarding it.

⚠️ **This applies to A and to B identically and is not a reason to prefer either.** It is pre-registered
here so that it cannot be discovered after a null and used to rescue the hypothesis.

---

## 5. Falsifiers — pre-committed

| | fires if | then |
|---|---|---|
| **F1** | the §6 pre-flight shows pooled `0.5·Σbody/gate` outside **0.11–0.45** (±2× the prediction) | **VOID** — the intervention is not the intervention. Do not train. |
| **F2** | the **paired-by-month** gate-AP difference against the arm-as-run is **below −0.010** | **KILL.** The allocation story is wrong-signed; the balance was not what cost the gate. |
| **F3** | gate AP recovers on P-3's paired contrast but `crps_all` worsens by **> 5%** | the body paid for it. A **finding**, not something to tune away. |
| **F4** | `act_ratio` at h18 **> 5×** the control's | M45's mechanism reproduced; the denominator is not the fix. |
| **F5** | emitted positive fraction leaves **[0.55, 0.85]** | the composition moved when only the loss was supposed to — a confound. **VOID.** |
| **F6** | the arm's config differs from `amber_kestrel` anywhere but the new denominator axis | **VOID** |
| **F7** | artifact arrives without all three files + `PROVENANCE` with `lessons ≥ 300` and `mode: production` | **VOID** — reuse `2026-10-12_underfire_diagnosis/vehicle.py`'s gate verbatim |
| **F8** | training diverges, NaNs, or the stability gate trips (C-184 BatchNorm seed-bimodality) | **VOID, not negative** (C-307) |

⛔ **Pre-committed consequence.** This is an **n=1 SCREEN at 300 lessons**. A gate-AP change that is
neither a recovery past P-3's +0.010 nor a collapse past F2's −0.010 is recorded as **INCONCLUSIVE,
not negative** (C-307). It cannot ship on that basis without a seed replication.

⚠️ **And n=1 resolves a SIGN, not a size.** A's predicted recovery (~+0.031 back toward 0.5631) is
**1.6× the 0.019 seed sd**. The paired-by-month design is sharp enough to say which direction the
arm moved; it is not sharp enough to measure how far.

---

## 6. Method

### 6a. ⛔ The pre-flight gate that does not exist yet — wire it first

`reports/2026-10-13_composition_arm_outcome/OUTCOME.md` closes with the requirement and nothing
implements it: *no config-identity check can see a loss magnitude.* `scripts/arm_identity_check.py`
compares **config values**; the composition arm shipped a **65× change in a task-loss term** and
every check passed.

**Assert, before the arm trains, on one sampled lesson:**

1. the **count of cells entering each task's backward graph** (6 tasks), and
2. the resulting **per-task task-loss terms** and the pooled `0.5·Σbody/gate`,

against the §4 table. F1 reads this gate. This is the arm's cheapest and most informative readout
and it costs no GPU time.

### 6b. ⛔ Log the per-task split — it was unset on BOTH delivered runs

`trajectory_log_path` (`training_engine.py:1319-1340`) is the **only** sink for `lesson_reg` /
`lesson_cls`, and it was `None` on the control and on the arm. That is why r2/r3 had to be
reconstructed from weights, and why I wrongly told Simon the offline wandb logs would have the
split — `run.log` carries only the total (`Final Lesson Loss: 3789.684814`). **Set it.**

### 6c. ⚠️ The `log_var` gradient readout — and the honest correction to it

I proposed logging the balancer's gradients during A's run as a free read of the direction B would
move. They *are* populated: `training_engine.py:114-118` states that without the optimizer group
*"log_vars accumulate gradients but are never stepped, so they stay frozen at their zero
initialization and the balancer is inert."* Two things I then checked:

- ⛔ **The gradient is a deterministic function of the task loss and nothing else.** With
  `stds = exp(v/2)`, the term is `L/((r+1)e^v) + v/2`, so `∂/∂v = −L/((r+1)e^v) + 1/2`, which at
  `v = 0` is **`−L/(r+1) + 1/2`**. It carries **no information beyond `L`**, which §6b already logs.
  The readout is therefore **redundant, not free information**. §6b is the instrumentation that
  matters.
- ⛔ **The raw `.grad` is a cumulative sum, not a per-lesson gradient.** The only `zero_grad` in the
  engine is `optimizer.zero_grad()` (`training_engine.py:1361`), and under the freeze `log_vars` are
  not in the optimizer — so their `.grad` is **never zeroed** and accumulates monotonically across
  all 300 lessons. Anything logged must be the **delta** between successive reads (read-only),
  never the raw value, and never zeroed in place (that would be a mutation for a redundant readout).

**What survives, and it is better than the readout:** the stationary point is analytic. Setting
`∂/∂v = 0` gives `v* = log(2L/(r+1))`, hence `coeff* = 1/(2L)` and a **weighted contribution of
exactly 0.5 nats for every task, whatever its family**. So **B's fixed point equalises all six
task-loss terms by construction**, giving `0.5·Σbody/gate = 1.000`. Computed on r3's arm medians:

| task | L (arm) | `v*` | `coeff*` | weighted |
|---|---|---|---|---|
| `lr_sb_best` body | 2.4784 | 0.908 | 0.2017 | 0.500 |
| `lr_ns_best` body | 1.1962 | 0.179 | 0.4180 | 0.500 |
| `lr_os_best` body | 1.8392 | 0.609 | 0.2719 | 0.500 |
| `lr_sb_best` gate | 0.0377 | −2.585 | 13.26 | 0.500 |
| `lr_ns_best` gate | 0.0080 | −4.135 | 62.50 | 0.500 |
| `lr_os_best` gate | 0.0240 | −3.037 | 20.83 | 0.500 |

⚠️ **Three caveats, pre-registered so this is not over-read.** (i) It is the stationary point of the
*instantaneous* objective treating `L` as fixed; `L` moves as the trunk reallocates, so it is a
coupled fixed point, not a prediction of where a 300-lesson run ends. (ii) Three of the six `v*`
land **below −2**, outside the interval over which Kendall's Appendix B demonstrates
initialisation-robustness — which concerns initialisation, not where an optimum may sit. (iii) It is
derived from **our code**, so it describes where B goes; it says nothing about whether that is a
good place to go, which is exactly what §7a says is underived.

⛔ **And it does not reopen the A-vs-B order.** I have moved position three times; a fourth on a
derivation that only describes B's destination would be noise. Recorded as a prediction **for the
fallback**, and §8 states why the ordering stands.

### 6d. Standing gates (the #311 precedent — no gate in this repo is repo-wide)

- `scripts/floor_gate.py` **FG-A and FG-C on the control arm before any treatment arm runs** (C-299,
  zero extra GPU).
- the **potency pre-flight** on the arm's own config **and** at a trained checkpoint (C-324/C-325 —
  a mechanism measured only at initialisation has twice been recorded as ruling a mechanism out,
  with the effect arriving 33 lessons later).
- `scripts/arm_postflight.py`, `scripts/arm_identity_check.py`, the **weight-hash post-condition**,
  the recursive `kill_tree` fix (C-326).

### 6e. Readout

- **Primary slice h=1**, the only matchable one — teacher-forced in both input and state
  (`hydranet_inference.py:1219-1221`, verified in P1). h18/h36 read **only** for F4.
- **Statistic:** pooled mean. ⛔ **Not** the median over cells — it is quantised at 1/16 = 0.0625,
  coarser than the effects compared.
- ⛔ **Never mix two statistics.** Concretely, in this document: the per-target medians in §4 are
  **not summable** — summing r3's per-target medians gives 39.6 where r2's pooled median-of-sums is
  **35.2**. The pooled figure is r2's; the per-target figures are r3's; rows must not be compared
  across tables.

### 6f. Contrast set, pinned by hash

| | artifact | sha256 | dist | git |
|---|---|---|---|---|
| **contrast (the arm)** | `calibration_model_20261013_010739.pt` | `2a5efc2a67c08eac…` | `truncated_nb` | `73c3f42` |
| reference only | `calibration_model_20261012_033112.pt` | `4acbf4d3c2b457b0…` | `nb` | `f99413e` |

⚠️ **Amendment 3 of the composition prereg applies:** the data is not pinned by anything in this
repository. Pull FRESH; never validate on cached disk data.

---

## 7. Decision rules

| outcome | decision |
|---|---|
| **F1 fires** | do not train. The denominator axis is mis-wired; fix and re-run the pre-flight. No spend. |
| **P-3 recovers, no falsifier** | the allocation story holds. The *next* question is the per-target spread (§3.1), **not** another arm — a decision, not a sweep. |
| **F2 fires** | **KILL A.** The balance was not what cost the gate. B becomes uninteresting for the same reason, and the open question returns to why both halves sit below their own objectives. |
| **F3 fires** | recorded as the pre-registered magnitude-vs-occurrence trade, with §4a's caveat live. |
| **INCONCLUSIVE** (neither recovery nor collapse) | **replicate A at a second seed.** ⛔ **NOT B** — struck, see §7a. ⛔ Not a sweep of denominators. |
| **F5/F6/F7/F8** | VOID. Re-run or abandon; never reported as negative. |

### 7a. ⛔ Option B is STRUCK — it is not a fallback, and this repository already measured it

**Struck 2026-10-15 (Amendment 1, §1).** `reports/results_balancer_bisect.md` is a **pre-registered,
device-matched GPU/GPU bisect** whose verdict reads: *"H CORROBORATED — the C-111 fix (un-freezing
the MultiTaskLoss balancer) is the cause of the C-113 acute runaway."* Free-running attractor
**trainable log ~15–17 (`expm1` ~1e6–1e7, PATHOLOGICAL)** against **frozen log ~4–5 (~70–215,
healthy)**, and the frozen arm held across **all three** seeds and both `freeze_h` settings.
`technical_risk_register.md:429` records the disposition: *"Fix direction: **regularise** the
balancer (bound/decay `log_vars`, lower its LR), **not permanent freeze**"*.

⛔ **So B-as-written — unfreeze, unregularised, `log_vars` outside the gradient clip — is precisely
the form this repository has already ruled against.** The text below was my original case for B and
cited Kendall's Table 1 over our own experiment. It is kept, struck, so the error is on the record
rather than edited out.

⚠️ **What is NOT struck:** a *regularised* balancer (bounded or decayed `log_vars`, lower LR) is the
register's own stated fix direction and remains open. It is **not** this plan's fallback and would
need its own pre-registration, including **M-1** below.

⛔ **M-1, found 2026-10-15 and fatal to B as specified** (`reports/2026-10-15_loss_balance_method_review.md`
§2.2): the LR scheduler captures `base_lrs` over `unet.parameters()` **only**
(`utils/utils.py:258-273`), `log_vars` are added as a **later** param group
(`training_engine.py:134-136`), and `_LRScheduler.step()` zips against a length-1 `get_lr()`. So the
balancer's LR is **never updated** and keeps the optimiser default `0.001` while the network decays
on WarmupDecay. Verified after 900 steps: network `6.011e-04`, balancer `1.000e-03`. Total balancer
travel ≈ **0.90 nats** against the **2.585 / 3.037 / 4.135 nats** its own fixed point needs — **B
cannot reach its fixed point on this vehicle.**

#### ~~The original case for B, struck — kept for the record~~

**B** = unfreeze the balancer on `amber_kestrel` only (`freeze_multitask_balancer: False`, which is
already the config **default**), letting the learned log-variances allocate the trunk.

- ⛔ **The method gap — recorded so it is not re-proposed** (the same treatment as Horizon Forcing's
  disqualification in the professor-forcing dossier). **Kendall (2018)**'s derivation
  (`Kendall2018_MultiTaskUncertainty`, *"Multi-Task Learning Using Uncertainty to Weigh Losses"*,
  Kendall, Gal & Cipolla) covers **Gaussian regression** and **softmax classification**. Our body is
  a **two-parameter NB NLL** and our gate a **pos-weighted** Bernoulli NLL — neither. The `log σ`
  regulariser that stops a task being absorbed is **unproven for a count likelihood**. B is
  therefore **plausible-but-underived for our objective**, not the principled choice. ⚠️ There is
  **no non-Gaussian extension to fetch**: "Cipolla, Gal & Kendall 2018" is a reordered author list
  for the paper we already hold. The gap is in the literature, not in our retrieval.
- ✅ **What Kendall does establish, and it favours learning where the derivation applies:** Table 1 —
  uniform weighting 50.1% IoU, **worse than single-task 59.4%**; grid-searched 62.8%; **learned
  63.4%**. Appendix B — initialisation-robust for `log σ² ∈ [−2, 5]`, noise terms converge in **~100
  iterations** against 30,000+ for the network. Our C-312 trouble was the **guard bug**, not the
  method.
- ⚠️ **`log_vars` are outside the gradient clip** (C-314 item 3 — the clip covers
  `model.parameters()` only). Under the freeze this is harmless because they are never stepped.
  **Under B it is live**, and it is a VOID condition, not a guard to add mid-arm (C-112).
- ✅ **The comparability objection is narrower than I first said.** Unfreezing on `amber_kestrel`
  **alone** changes no roster model and no logged roster result. C-112/C-119 only bite if the config
  **default** moves — and the default is already `False`.
- **Predicted destination:** §6c's table. `0.5·Σbody/gate → 1.000` by construction.

### 7b. ⛔ No surrogate rig; no 2×2

- A **mini environment / synthetic DGP** to iterate on this was considered and **rejected**: the
  repo's own precedent is that cheap testbeds mislead here (M8's state-freeze at 40 lessons was 3.6×
  off the L=300 truth; M1's persistence ordering flipped entirely between 40L and 300L; C-325
  records two mechanism tests measured at initialisation that were recorded as ruling a mechanism
  out, with the explosion arriving 33 lessons later). And a self-chosen synthetic DGP can be made to
  show what its author expects, which is the one objection no result from it can answer.
- A **2×2** (Kendall vs GradNorm × hurdle vs NB) was proposed by the library session and is
  **declined**: four arms at n=1 against a ~0.019 seed sd, one requiring GradNorm implemented from
  scratch. That is C-307's pattern — a cheap screen recorded as a closure. ⚠️ And *"publishable"* is
  not this programme's objective function.

---

## 8. ⛔ Record — things that exist only in session memory and would otherwise die here

1. **My position on A-vs-B moved A → B → A.** Named causes, in order: (i) A first, because it is the
   defined point; (ii) B, when I read Kendall's Table 1 and Appendix B and judged the learned
   balancer both better-evidenced and init-robust; (iii) back to A, when the library session showed
   Kendall's derivation does not cover an NB likelihood, removing B's claim to being *principled*.
   **The oscillation is itself the finding:** there is no authoritative scale for a shared-trunk
   hurdle, so the choice is **which measurement is more interpretable**, not which is correct. A is
   one defined, deterministic point with a per-target failure mode we have already projected (§3.1).
   B is a coupled fixed point reached through an underived regulariser with an unclipped parameter.
2. **Simon's standing intuition was A first, B as the fallback. It stands**, and I was wrong to talk
   him off it mid-conversation.
3. **Composition self-consistency, measured:** `E[composed] = g·μ/(1−p₀)` recovers `μ` only if
   `1−p₀ = g`. On the 100-cell in-sample set (P0, coarse): `1−p₀ = 0.471` against `g = 0.695` ⇒
   **~1.47× over-prediction**. Not this arm's question; recorded because it is the magnitude-side
   consequence of the same coupling.
4. **Mullahy (1986)** (`Mullahy1986_ModifiedCountDataModels`, *proven/passage* §2 pp. 345–346) is
   the formulation our arm already implements — a binomial plus a truncated-at-zero count model. Its
   *argued* passage (pp. 344–345) reads overdispersion as *"misspecification of the parent
   distribution in which the relative probabilities of zero and positive outcomes implied by the
   parent are not supported by the data"* — which is a literature-grounded reading of M6's θ̂ 0.154
   against a conditionally-honest 0.432.
5. ⚠️ **An assumption from the library session not to build on:** it applied Xu (2023) under
   *"your case: UCDP is high-reporting."* That is its inference about our data, not a finding, and
   conflict-event under-reporting is contested. The hurdle-vs-zero-inflated conclusion must not rest
   on it.
6. ⚠️ **No held paper addresses learned task balance in a deep hurdle model** — Kong (2020)
   hand-tunes a λ; Musat (2026) is silent. I read that as **caution at n=1, not opportunity**: no
   prior art means no documented failure modes.
7. ⚠️ **Tweedie is not a shortcut around this.** It relocates the coupling into ρ and φ rather than
   removing it, is untested above ~97% zeros, and the standard implementations are a severe
   approximation (Damato 2025, C-1863).

---

## 9. ⛔ Definition of done, and the stop rules

**Done when:** the §6a pre-flight numbers are on record; a number for P-1 through P-6; every
falsifier marked fired or not; one sentence saying whether the denominator was the allocation
problem; a `found, not chased` list.

**Then stop.** Specifically:

- ⛔ **One arm. No sweep.** If it fails, the next step is a decision (§7), not another arm.
- ⛔ **Nothing is added mid-run.** Discoveries go on `found, not chased`.
- ⛔ **F1 is a hard stop before any spend.** F2 and F8 are hard stops on the readout.
- ⛔ **A null is INCONCLUSIVE, not negative** (C-307).
- Budget: **one pod, ~3 h, ~$2.25.** The §6a pre-flight is free and runs first.

---

## 10. Out of scope

- **B.** It is the fallback (§7a), not a second arm in this plan.
- **The per-target spread** A leaves behind (§3.1). Named, projected, and deliberately not fixed
  here — fixing it would mean a second variable (`target_weights`, which C-87 asserts are all 1.0).
- **Splitting `output_distribution` into a train axis and an emit axis.** The dial is dual-purpose
  (`utils/utils.py:183` vs `utils/hydranet_inference.py:266-270`) and that is a real design gap, but
  this arm does not need it: A changes the denominator, not the family, so train and emit stay the
  same law. ⚠️ Noted, not fixed.
- **The magnitude ceiling** (#241) and the free-running roll's contribution (C-113 territory).
- **Why both halves sit below their own objectives.** Still the most interesting open question in
  the record, and still not this experiment's question.
- **Fixing the false prose in §0** beyond this arm's own files. ⛔ The views-models occurrence
  (`config_hyperparameters.py:43` on `development`) is raised as a **GitHub issue, never a commit** —
  the standing rule for that repository. Issue number to be filled in when filed.

---

## 11. Glossary

Checked against `reports/GLOSSARY.md`. **Two corrections to my own brief:**

1. ✅ **`gated_truncNB` IS already locked** (§2c) for *"soft: per draw `Bernoulli(gate) ×
   TruncNB(μ,θ)`"* — I claimed there was no locked name for a soft-gated zero-truncated body. There
   is, and it is this arm's composition, unchanged.
2. ⛔ **The glossary had no words for the task-loss balance at all**, and §3's **NB loss** row listed
   `TruncatedNB` as a *banned alias* — written when the only truncated NB was the discredited legacy
   loss, and now in direct conflict with the live `truncated_nb` family and with `gated_truncNB`.

Per the standing rule a missing word is **added to the glossary**, never worked around. Added as
**§3b — the task-loss balance**: *the balancer*, *frozen balancer*, *task-loss term*, *the NLL
denominator* (values **positives** / **all cells**), *truncated NB loss*; and the stale banned alias
scoped to the legacy loss.

---

**Seams.** Design critique → `expert-method-review` (not run on this arm; the design is one
constant). Code → `expert-code-review` once the denominator axis is written. Risks → `register-risk`
(the C-303 instance in §0, the §6a harness gap, and §4a as a methodological risk). Literature →
the `library` session, which supplied §4a, §7a and §8.4–8.7. Claims → `falsify`, **after**.

---

# AMENDMENT 1 — four corrections, and the gradient test that was supposed to kill this plan. 2026-10-15, BEFORE the arm trains.

⛔ **Why amending a pre-registration is legitimate here, stated so it can be challenged.** No arm has
run. Every input below predates the arm: three corrections come from a 7-seat method review of *this
document* (`reports/2026-10-15_loss_balance_method_review.md`), and the fourth from a measurement on
the **two already-delivered artifacts**, not on the arm. Two of the four **tighten** a falsifier and
none loosens one. This follows the composition prereg's own Amendment 1–3 pattern. ⚠️ The original
text is struck through or quoted, never deleted.

## 1. ⛔ B is STRUCK (§7, §7a)

I offered B as the pre-registered fallback and dismissed this repository's prior trouble with it as
*"the guard bug, not the method."* **That was wrong, and a dedicated pre-registered experiment
already said so** — `reports/results_balancer_bisect.md`, 3 seeds, device-matched: un-freezing the
balancer **is** the cause of the C-113 runaway. I cited Kendall's Table 1 over our own data.

Compounding it, **M-1** (§7a above): the balancer's param group never receives the LR schedule, so it
trains at a constant `0.001` and can travel ≈0.90 nats against the 2.6–4.1 its own fixed point needs.
**B could not have tested Kendall's method on this vehicle even if it were safe.**

The INCONCLUSIVE branch of §7 now reads *replicate A at a second seed*.

## 2. ⛔ F8 no longer cites C-119's variance figure (§5)

C-119 is **RESOLVED** (`technical_risk_register.md:38`, `:5518`; same-seed runs bit-identical, 78/78
`y_pred` `array_equal`, `daab1c1`). Its *"~20% training variance"* is the **pre-fix** measurement and
is not a live noise floor. **A dead number was sizing a VOID rule in a committed pre-registration.**

## 3. P-3 and F2 re-sized to the paired-by-month contrast (§4, §5)

I sized both on the **0.0413** month-to-month sd. `OUTCOME.md:67-68` already demonstrates this
design's real resolution: paired by month, the composition arm's gate loss was **mean −0.0299, 95% CI
[−0.0401, −0.0197], 0 of 13 months favouring the arm** — a half-width of **~0.010**, four times
finer. Both thresholds are now ±0.010 on the paired difference, and §5 records that n=1 resolves a
**sign, not a size**.

## 4. Cragg → Mullahy for separability (§0)

Adjudicated against both PDFs (method review §3a). Cragg **never states separate estimation** and his
model II is a continuous **truncated normal**; Mullahy states it verbatim three times and is the
right model class. ⚠️ Separately, our library claim is graded **`proven`** for a statement the paper
does not make — flagged for `/library`, **not** edited by hand.

## 5. ⛔ Two further errors in this document, corrected

- **§1/§2 said the joint hurdle NLL is unreachable under the freeze. False.** With
  `is_regression=False` the coefficient is exactly `1/((0+1)·1+eps) = 1.0` and the barrier
  `log(1+eps) ≈ 0`, so **A's denominator + that flag = the exact joint hurdle NLL**, balancer still
  frozen. ⚠️ Deliberately **not** bundled into this arm: the control measured 0.593 *with* the 0.5, so
  keeping `is_regression=True` leaves the denominator as the single variable. It is a later arm, and
  Kendall's factor of 2 descends from the **Gaussian exponent** (eq 5 → eq 10, where the task loss is
  a squared residual), so applying it to an already-normalised NB log-density has no derivation
  behind it.
- **§3.1's objection was backwards.** I wrote that A leaves `lr_ns_best` *"effectively
  unsupervised."* Measured: zeros supply **51.7% / 88.0% / 67.8%** of the *control's* body term, so
  **the control's near-equal per-target ratios are the artefact**, produced by diluting the body with
  cells that carry no magnitude information. A's spread is the likelihood being honest: with a
  **median 0.88 positive cells per window** for `ns` (61% of windows averaging under one), there is
  almost no magnitude signal to supervise. **And the real argument for A is an estimator argument:**
  the positives denominator is a random variable, usually 0 for `ns`, so `E[Σ/N_pos] ≠ E[Σ]/E[N_pos]`
  — the gradient is biased and high-variance. A is the unbiased estimator, not a tuning lever.

## 6. 🔑 The gradient test — the plan's own precondition, and it HELD

The method review found that a loss-value ratio has no direct causal path here: only **11.29%** of
the model is shared (102,240 of 905,289 — body and gate have separate ~400k-param decoders), and
under **AdamW** a per-task constant is **exactly** invisible on that task's private parameters
(verified: max\|difference\| = **0.0** at scale 1 vs 65.5 over 60 steps). So the 65× can act *only* by
tilting the summed gradient's direction on the shared subset — which nobody had measured.

`r4_grad_conflict.py` (committed `1ebcc92` **before** the run; output `ca3d7f8`) measured it on both
delivered artifacts. **The pre-committed N-branch did not fire:**

| | control (`nb`) | arm (`truncated_nb`) |
|---|---|---|
| loss-value ratio (r2) | 0.593 | 35.2 |
| **shared-trunk ‖grad‖ ratio** (K=24 / K=8) | **0.312 / 0.325** | **10.497 / 9.436** |
| **cos(∇body, ∇gate)** | **+0.7244 / +0.7216** | **−0.2033 / −0.1807** |
| windows in conflict | **0%** | **75%** |

59× in loss became **~32×** in shared-trunk gradient norm and the cosine **flipped sign**. The gate's
own gradient barely moved (sb 0.086 → 0.098); the body's grew **13–63×** and out-competed it.
**Stability pre-registered and confirmed:** tripling the BPTT depth moved the cosine by **0.003**
(control) and 0.023 (arm), so the verdict is not a truncation artifact. K=1 was dropped — ~3 h of 13
cores for the block with no BPTT at all, i.e. the least faithful to production.

### ⚠️ But it relocated the conflict, and §3 should have said this

```
CONTROL  diagonal +0.8606 / +0.8552 / +0.8482     off-diagonals and body-vs-body all NEGATIVE
ARM      diagonal -0.4491 / +0.0580 / +0.3072     body-vs-body now POSITIVE (+0.28, +0.60)
```

In the control **each target's own body and gate cooperate** (+0.85) and the three *targets* compete.
In the arm the three bodies gang up and **sb's own body turns against its own gate (−0.45)** — sb
being the target with the most data and the one whose AP is scored.

⛔ **So "the body drowned the gate" is right in the arm and wrong in the control.** The control's
shared trunk has no body-vs-gate conflict at all; its conflict is **cross-target**. A new hypothesis
this plan does not test, raised by the Gelman seat: the rarest target may be eating the trunk and
starving both halves of the other two — which is the only *allocation* story that explains a
common-mode shortfall across the gate/body split. **Named, not chased.**

## 7. ⚠️ A better-motivated arm than A now exists, and it is not in this plan

**`pos_weight` 2.0 → 1.0.** Derived independently by the Gneiting seat from **Gneiting & Ranjan
(2011) Thm 1** (a weighted proper score is improper; the weighted-BCE optimum is `g* = wp/(1+(w−1)p)`
≈ **2p**), matching **P2's measured implied 1.0365** and CRPS's `k=0` threshold term being the
occurrence Brier score at weight 1. ⛔ **It also dissolves the "both halves under-reach" puzzle this
plan calls the most interesting unexplained fact:** the gate's objective wants 2p and the gate
delivers p, so under-reaching its objective means **landing on the truth**. Tracked as **O-2** in
`reports/PREDICTION_LEDGER.md`. Not pre-registered; the chair's call whether it precedes A.

## 8. What is unchanged

The intervention (§2), the predicted magnitudes (§4 P-1/P-2), the free pre-flight gate and its hard
stop (§5 F1, §6a), the instrumentation (§6b), the standing gates (§6d), the contrast set (§6f), the
stop rules (§9) and the scope (§10). **A still runs as a correctness fix with its hypothesis demoted,
not as a skill claim.**
