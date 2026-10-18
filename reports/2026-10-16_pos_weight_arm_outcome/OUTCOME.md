# Outcome — the `pos_weight` arm. KILLED, and it found something bigger on the way out.

**Date:** 2026-10-17 · **Pre-registration:** `reports/2026-10-16_pos_weight_prereg.md` (commit `02d1294`, written before the run)
**Arm:** `copper_kestrel` — the experimental run, gate trained at `pos_weight = 1.0`
**Control:** `amber_kestrel` — the already-delivered run, gate trained at `pos_weight = 2.0`
**Readout measured by:** the `views-models` session. **Written up and independently re-verified by:** this session.
**Evidence:** `evidence/EVIDENCE.md` (273 lines of primary data) + the `.npz` arrays and scripts beside it.
**Revision 3** (2026-10-18): §2.4's central identity upgraded from approximate to **exact**; §8.4's leading candidate **closed by derivation** and replaced with two named ones; §8.6 added; three issues added to §9. Rev 2 changed the *name* of the finding; rev 3 changes what to **do** about it. Nothing in §1–§6 is retracted.

---

## 0. In plain words, before any numbers

The model has two halves. One decides **where** conflict happens (the **gate**). The other decides **how much** (the **body**).

The gate is trained with a setting called `pos_weight`, which was set to **2**. Setting it to 2 tells the gate: *when conflict does happen, count that twice as much*. There is a theorem saying this is the wrong thing to do — it mathematically guarantees the gate will report wrong probabilities. The honest setting is **1**.

So we trained one model at **1** instead of **2**, and asked: does the gate now report honest probabilities?

**It got worse. Twice as wrong.** The arm is killed.

But here is the part worth staying awake for. At `pos_weight = 1` the gate **should** have been honest, because the theorem says 1 is the correct value. It wasn't.

Looking at *why* found something better than the experiment was designed to produce:

> ⛔ **The gate's probabilities are squashed toward zero. It over-states small risks and under-states large ones — and `pos_weight` cannot fix that, at any value.**

`pos_weight` can only shift the whole set of probabilities up or down. The gate's problem is not that they are too high or too low; it is that the **spread is too narrow**. At `pos_weight = 2` the shift happened to sit where most of the data is, so the gate *looked* about right wherever we measured it. At 1 the shift moved and the squashing became visible.

It was never a hidden second problem. **It was always there, in the control, in plain sight — our measurement was looking at the one place it does not show.** 96% of all cell-months sit in a single bin where the gate's error is 0.0004.

⚠️ **This is a correction to an earlier reading.** The first version of this document — following the readout it was written from — said there was "a second, independent downward bias" that removing the tilt "revealed". That was the right instinct with the wrong mechanism, and §2 now carries the corrected version. The error was in the evidence's interpretation, not in its measurements.

---

## 1. The verdict

### Pre-registered predictions

| | endpoint | predicted | **measured** | |
|---|---|---|---|---|
| **P-1** | median \|gate − p\| (calibration) — **primary** | ≤ 0.010, no worse than control | control **0.0154** · arm **0.0303** | ⛔ **MISSED — 1.96× worse** |
| **P-2** | median implied `pos_weight` | in [0.90, 1.15] | control **1.3090** · arm **0.6252** | ⛔ **MISSED — both out, opposite sides** |
| **P-3** | gate AP, paired by month | **unchanged**, ±0.010 | 0.5324 → **0.5385**, paired **+0.0075** | ✅ **HELD** (with a note, §4.3) |
| **P-4** | `crps_all` | no worse, ≤2% better | **37–83% "better"** | ⚠️ **MISSED UPWARD — an artefact, §3** |
| **P-5** | emitted positive fraction, 100 cells, h=1 | in [0.55, 0.85] | control **0.4999** · arm **0.3947** | ⛔ both outside — **cannot discriminate, §5.2** |
| **P-6** | `act_ratio` at h18 | ≤ control | control **2.3520** · arm **0.4363** | ✅ **HELD** — and it is what caught §3 |

### Pre-registered falsifiers

| | | |
|---|---|---|
| **F1** calibration worse than control | ⛔ **FIRES** | 0.0303 vs 0.0154 |
| **F2** paired gate AP below −0.010 | no | AP went *up* |
| **F3** `crps_all` worse by >5% | no | it "improved" — see §3 |
| **F4** `act_ratio` above control's | no | it halved |
| **F5** positive fraction outside [0.55,0.85] | fires **literally** | but the control fails it too — §5.2 |
| **F6** any second config difference | tripped **literally** | ⚠️ **ruled NOT VOID by Simon, 2026-10-17** — §6 |
| **F7** artifact completeness | pass | |
| **F8** divergence / instability | no | 300/300 lessons, no failure signature |

### ⛔ The kill

The pre-registration's kill criterion (§5) was **F1 or F2**. **F1 fired.** The arm is **KILLED**.

**One sentence on whether removing the improper tilt bought anything** (prereg §9's definition of done): **No — it cost calibration, roughly doubling the gate's error, and in doing so exposed a second and larger problem that the improper tilt had been masking.**

⛔ `pos_weight` returns to **2.0** for all subsequent arms, per the pre-registered decision rule — **not because 2.0 is right** (it is provably not), but because changing it is not the fix and the arm said so.

---

## 2. ⛔ The finding: the gate is UNDER-DISPERSED. No `pos_weight` can fix it.

### 2.1 What the theorem predicted, and what actually happened

`weighted_bce` at `pos_weight = w` has its optimum at `s* = w·p / (1 + (w−1)·p)`.

**Gneiting & Ranjan (2011), Theorem 1:** weighting a proper scoring rule by a function of the *outcome* makes it **improper** — the minimiser becomes a tilted law rather than the truth. So `w ≠ 1` is **mathematically guaranteed** to miscalibrate, and `w = 1` is the only principled value.

| | trained at | median implied `w` | reading |
|---|---|---|---|
| control | 2.0 | **1.3090** | over-states, as the theorem predicts for w>1 |
| arm | 1.0 | **0.6252** | ⛔ **under-states** — the theorem predicts 1.0 here |

So the theorem's prediction fails at `w = 1`. The question is why.

### 2.2 ⛔ The answer: look at the SIGN. P-1 takes the absolute value and cannot see it.

Signed error, `gate − observed p`, every bin, both arms *(independently reproduced here; see §4.2)*:

| gate bin | n control | **control signed** | n arm | **arm signed** |
|---|---|---|---|---|
| [0.00, 0.01) | 808,517 | **+0.0004** | 818,958 | **+0.0008** |
| [0.01, 0.05) | 21,343 | **+0.0100** | 16,679 | −0.0091 |
| [0.05, 0.10) | 5,713 | **+0.0157** | 3,104 | −0.0516 |
| [0.10, 0.20) | 3,275 | **+0.0154** | 1,677 | −0.1036 |
| [0.20, 0.30) | 1,265 | **+0.0261** | 727 | −0.1438 |
| [0.30, 0.40) | 700 | −0.0308 | 421 | −0.1469 |
| [0.40, 0.50) | 525 | −0.0062 | 291 | −0.1892 |
| [0.50, 0.60) | 387 | −0.0209 | 246 | −0.2293 |
| [0.60, 0.70) | 334 | −0.0517 | 237 | −0.1943 |
| [0.70, 0.80) | 291 | −0.0587 | 178 | −0.1646 |
| [0.80, 0.90) | 219 | −0.0531 | 105 | −0.0825 |
| [0.90, 1.01) | 65 | −0.0419 | 11 | −0.0823 |

```
control signs:  + + + + +  - - - - - - -      ⛔ crosses zero at 0.30
arm signs:      +  - - - - - - - - - - -         same rotation, whole line dropped
```

⛔ **The control over-states where the data is and under-states where it is confident. That is a ROTATION — a wrong slope — not an offset.** The arm is the same rotation with the line pushed down.

### 2.3 Confirmed parametrically

Fitting `logit(observed p) = a + b·logit(gate)`. An honest gate gives `a = 0, b = 1`:

| | n≥1000 bins | all bins | n-weighted |
|---|---|---|---|
| control | a=+0.063 **b=1.146** | a=+0.175 **b=1.177** | a=+0.026 **b=1.139** |
| arm | a=+1.645 **b=1.419** | a=+0.902 **b=1.251** | a=+1.777 **b=1.456** |

**`b > 1` in all six fits: the gate's logits are COMPRESSED — truth moves faster than the gate does.** The control's *level* is roughly right (`a ≈ 0`) with the wrong *slope*. The arm is wrong in both.

⚠️ Method note: the n-weighted fits differ slightly between the two sessions that computed them (control b 1.139 here vs 1.134; arm b 1.456 vs 1.508), because the weighting scheme differs. **`b > 1` is unanimous across every fit either session ran**, which is the load-bearing part; no single coefficient should be quoted as *the* slope.

### 2.4 ⛔ Why this is the mechanical reason no `pos_weight` works

In logit space:

- `s* = wp/(1+(w−1)p)` is **EXACTLY a constant shift of `log w`** — it moves **`a`** and nothing else.
- Temperature scaling is `logit / T` — it moves **`b`**.

⛔ **The first line is an identity, not an approximation** (rev 3 — rev 2 said "approximately" and understated the document's own strongest claim):

```
s*/(1 − s*) = [wp/(1+(w−1)p)] / [(1−p)/(1+(w−1)p)] = wp/(1−p)
⟹  logit(s*) = log(w) + logit(p)          exactly, for all w > 0 and all p ∈ (0,1)
```

No small-`p` limit, no linearisation. **At the weighted-BCE optimum `pos_weight` has provably ZERO slope effect.**

**The defect is in `b`. `pos_weight` does not have the degree of freedom the defect requires** — not approximately, not mostly, but as a matter of algebra. Tuning `w` only slides the crossing point along a curve that keeps the wrong shape.

⚠️ Two consequences worth stating, because both were reached only after the peer review of rev 2:
- The identity gives a **testable signature**: the fit's intercept should be `−log(w)`. On the data-weighted statistic the control gives **−0.9825** against a theoretical **−0.6931** — right sign, 0.29 logits out. (A *bin-unweighted* fit gives `+0.175` and appears to have the wrong sign; that is an artefact of bin weighting, not a finding. See §8.6.)
- Because the effect is an exact intercept shift, **`pos_weight` cannot change ranks, AP, or resolution at its own optimum.** §4.3's measured AP residual is therefore not a `pos_weight` effect; see the note added there.

That is why `w = 2` "worked": it put the crossing point above the bulk of the data, so the gate looked calibrated *where the median samples it*. `w = 1` dropped the curve and the compression became measurable.

### 2.5 ⚠️ And the instrument already knew

`p2_gate_calibration.py:194` carries, as its VOID-branch message:

> *"The gate cannot be repaired by a scalar `pos_weight` — the arm needs a new design."*

**That is the correct diagnosis, written into the harness, and the branch never fired** — because it is gated on monotonicity and resolution, both of which this gate passes. The instrument had the right sentence behind the wrong condition. See §9's issue 4.

### 2.6 What survives from the first reading

The *shape* claim stands and is strengthened: the arm's error grows monotonically with the gate's confidence across four populated bins spanning 0.0008 to 0.1036. **A monotone ordering over that range is not seed noise**, so the shape is measured even at n=1. What changed is its name and its cause: **under-dispersion (slope), not bias (level)** — and it is **present in the control too**, not revealed by the arm.

⛔ **The claim that must stay fenced is the CAUSE.** n=1, no seed replication. We have measured the shape; we have not identified what produces the compression. **Rev 3 narrows the field to two named candidates and one free experiment that separates them — §8.4.** It does not resolve it.

## 3. ⛔ The CRPS result is an artefact. It must never be quoted as a gain.

**`eval_type == "month-wise"`** (see §4.5 — the selection was positional, not by name, and the label was initially missing):

| target | control | arm | change |
|---|---|---|---|
| sb | 0.1594 | 0.1010 | **−36.6%** |
| ns | 0.1027 | 0.0173 | **−83.2%** |
| os | 0.0265 | 0.0109 | **−58.8%** |

⚠️ The artefact conclusion is **robust across all three `eval_type`s** — step-wise gives −46.0% / −83.3% / −58.0%, and time-series-wise matches step-wise to 4 dp — so nothing here depends on which was selected.

**This is not skill, and it is not a second fact.** It is §1's P-6 seen through a blind spot.

The field is **99.7% zero**. On such a field, a forecast that fires less scores better *by construction*. `activation_metrics.py` says so in its own words:

`reports/2026-07-29_v2_scoreboard_dossier/tools/activation_metrics.py:3-4,19`:

> *"crps_all / MCR go BLIND when the field is ~99% zero: a forecast that almost never fires still [scores well]… This is the falsifier crps_all cannot see."*

The arm **halved its firing rate** — `act_ratio` at h18 went **2.3520 → 0.4363**, and emitted mass at h=1 dropped ~33%. The CRPS "improvement" and the firing collapse are **the same event described twice**, once by a metric that can see it and once by a metric that cannot.

⚠️ And 0.4363 is below 1.0, which is the **#258 under-firing collapse signature** — the arm is not merely less eager, it is now firing at *less than half* the true rate.

### The methodological win

**`act_ratio` was pre-registered as a guardrail specifically because CRPS cannot see firing rate.** It was required in advance, it fired, and it is the only reason this looks like a kill rather than a 37–83% success story.

**If `act_ratio` had not been pre-registered, this arm would have been reported as a large win.** That is the entire argument for pre-registration, demonstrated on one arm.

---

## 4. What this session verified independently — and one thing the readout missed

The readout was produced by another session. Per standing practice its claims were re-derived here from the raw arrays before being written down.

### 4.1 Everything load-bearing reproduces exactly

Recomputed from `p2_result_amber.npz` / `p2_result_copper.npz` with an independently written statistics block:

```
P-1  control 0.0154   arm 0.0303   ratio 1.96x      (claimed 0.0154 / 0.0303)
P-2  control 1.3090   arm 0.6252   sign flip YES    (claimed 1.3090 / 0.6252)
AP   control 0.5324199835833106     arm 0.5384912009101893
pos  control 0.4999038461538461     arm 0.3946634615384615
act  control 2.3519639144639144     arm 0.4362748737748737
sha  control 2a5efc2a…              arm f286e4a3…
```

**No discrepancies.** The weight shas inside each `.npz` match the delivered artifacts, so the harness measured these two runs and no others.

### 4.2 ⛔ A comparability issue the readout did not flag — and it strengthens the kill

`P-1` and `P-2` are medians over calibration bins with **n ≥ 1000**. The two arms do not have the same such bins:

- **control:** 5 bins — `[0.00, 0.01, 0.05, 0.10, 0.20)`
- **arm:** **4 bins** — `[0.00, 0.01, 0.05, 0.10)`

The arm's `[0.20, 0.30)` bin holds only 727 cells and drops out, **because the arm fires less**. So the published medians are taken over **different supports**, which is a comparability defect in the statistic.

Recomputed on the **common four bins only**:

| | as published (own bins) | **common bins** |
|---|---|---|
| control median \|gate − p\| | 0.0154 | **0.0127** |
| arm median \|gate − p\| | 0.0303 | **0.0303** |
| **ratio** | 1.96× | ⛔ **2.39×** |
| control median implied `w` | 1.3090 | **1.5381** |

**The verdict survives and gets stronger: on a like-for-like support the arm is 2.4× worse, not 2.0×.**

⚠️ **The defect is in the pre-registered METHOD, not in one implementation.** The pod's own gate slice shows the same 5-vs-4 split (control 5 bins / 0.0175 / 1.3571; arm 4 bins / 0.0298 / 0.6047). **Both measurement paths inherit it because both implement the prereg's `n ≥ 1000` rule faithfully.** It is not a coding slip in either.

⛔ **And following the correction one level down VOIDS the interpolated "honest `pos_weight`" entirely — both the original ≈1.55 and my corrected ≈1.41.** There is no scalar to interpolate *to*. `implied_w` across the usable bins:

| | min | max | range | crosses 1.0? |
|---|---|---|---|---|
| control | 0.405 | 2.737 | **6.8×** | **YES** |
| arm | 0.289 | 3.706 | **12.8×** | **YES** |

A median over 4 or 5 of 12 equally-weighted bins was being read as *"the effective `pos_weight`"* while the underlying quantity varies by nearly an order of magnitude **and changes the sign of the error within each arm**. Both numbers were medians of a non-constant. See §2.4 for why no scalar can be the answer, and §8.1 for what replaces it.

### 4.3 P-3 held, as the pre-registration structurally predicted — with a note

Prereg §3.2 argued *in advance* that AP **cannot** move by this arm's mechanism: the tilt `s* = wp/(1+(w−1)p)` is **strictly monotone in p**, and average precision depends on the gate **only through ranks**, so a pure recalibration is invisible to it.

**Measured: 0.5324 → 0.5385.** Essentially unchanged, as predicted. **A falsifiable structural prediction, made before the run, and correct.**

⚠️ The note: paired by month the difference is **+0.0075**, bootstrap CI **[+0.0014, +0.0136]**, with the arm higher in **11 of 13** months. The CI **excludes zero** and its upper bound sits just outside the ±0.010 band. So this is **held-with-a-note, not a clean null** — there is a small, consistent positive residual that the monotone argument does not explain. Candidate explanations: seed noise (n=1, nothing separates it), or the altered loss changing what the trunk learned. **Unresolved, and §7 of the prereg forbids reading it as confirmation of the hypothesis.**

⛔ **Rev 3 — the residual is NOT ATTRIBUTABLE, and the CI is measuring the wrong variance component.** The unit of randomisation for this intervention is the **training run**, and there is **one run per arm**. Pairing by month replicates the *evaluation*, not the *intervention* — the 13 months are 13 correlated reads of the same two artifacts, which is why the CI is tight. The prereg's own VOID rule (4) sizes the **seed** sd on AP at **~0.019**, and **+0.0075 is 0.39 of one seed sd**, so the whole CI lies inside it. By §2.4's exact identity `pos_weight` has no mechanism to move a rank-based score at all, so the right reading is: *two separately trained networks scored 0.0075 apart.* **Keep the number; drop the inference.** Attribution would need a seed replicate.

### 4.4 Traps in the evidence, confirmed

| | |
|---|---|
| ⛔ **Do not cite `pos_weight` from the `.npz` files.** | **Both** store `2.0`. Verified: it is the harness's hardcoded `SHIPPED_POS_WEIGHT` constant (`p2_gate_calibration.py:185`), not a value read from the model. **Meaningless here.** |
| ✅ **The authoritative arm identity** | the library's own log line, `weighted_bce_loss.py:41`, on the *constructed* criterion: control `run.log:430` → **2.0**; arm `run.log:95` → **1.0**. This is exactly what the prereg's §6a pre-flight demanded. |
| ⚠️ **Archival asymmetry, not a run difference** | the arm's delivery ships `configs_as_run/` (6 files); the control's does not, so the control's as-run config must be read from git at `73c3f42`. Also the control's gate slice is nested one level deeper (`gate_h1/gate_h1/`) than the arm's (`gate_h1/`). The two arms are **not symmetric as records**, which is a provenance weakness worth fixing for future arms. |
| ⛔ **Do not cite `verdict` from the `.npz` files either** | **Both** store `verdict='D2'` — *"already calibrated, w_eff=1"*. The branch is `err1 <= sep and err1 < err2` where `sep = median(\|2p/(1+p) − p\|)/2` (`p2_gate_calibration.py:170,184`). Measured: control err1 0.0154 vs sep 0.0245 (passes by +0.0090); **arm err1 0.0303 vs sep 0.0313 — passes by nine ten-thousandths** while under-stating by 0.1036 at p≈0.12, a factor of 1.7 in probability. `sep` scales with `p`, so on a sparse field the tolerance inflates until it admits almost anything. **Not usable evidence for either arm.** See §9's issue 4. |

---

## 5. ⛔ Two defects in the pre-registration — mine, not the run's

### 5.1 The absolute thresholds were mis-sized, so two falsifiers cannot discriminate

- **P-1's ≤0.010** was sized on `violet_visitor`'s published 0.0054 — a different, smaller-extent vehicle. **The control fails it** (0.0154).
- **P-5's [0.55, 0.85]** was sized on an expectation that the composition arm *itself missed* (it predicted [0.62, 0.76] and measured 0.4999). **The control fails it too** (0.4999).

**So F1 and F5 as written cannot distinguish arm from control.** What actually discriminates is the **paired contrast** — same script, same data, same stack, same calibration method — **0.0303 against 0.0154** (2.39× on a common support, §4.2), and that is unambiguous.

⛔ **The verdict stands. One of its stated justifications does not.** Anyone re-checking the arithmetic will find this, so it is stated here rather than discovered later. **Lesson for the next pre-registration: size absolute thresholds on the actual control, or state them as paired differences from the start.**

### 5.2 The pre-registration contradicts itself on F6

**§6 of the prereg MANDATES** setting `trajectory_log_path`. **F6 VOIDS the experiment** on any second config difference. The arm therefore differs from the control in two values:

1. `loss_class_pos_weight` 2.0 → 1.0 — the variable under test
2. `trajectory_log_path` — absent in the control, set in the arm

**The same document both requires the second difference and voids the run for having it.** That is a drafting defect and it is mine. It is **fixed in the prereg's text** as part of this write-up so the next arm does not inherit it.

---

## 6. ⚠️ The F6 ruling — recorded as a dated human decision, with the conflict left visible

**Simon ruled the experiment NOT VOID on 2026-10-17**, after being shown the §5.2 contradiction and the cost of the alternative (re-run the control with logging enabled: ~12 h, ~$16).

**The substantive basis, read from the code rather than asserted:** `training_engine.py:1423-1470` registers a **read-only forward hook** on `dec_conv4_head{1,2,3}_class` that accumulates `_o.detach().mean().item()` and writes a CSV. No RNG draw, no gradient path, no parameter touched. The comment at `:1422` states the default case is *"no hooks, no file, byte-unchanged."*

⛔ **This is recorded as a ruling, not laundered into "only one thing changed."** The point of a pre-registration is defeated if an inconvenient clause is quietly dropped once the results are known. The clause tripped; a human overrode it, on a stated basis, on a stated date, having been shown what it would cost not to. **That is the honest form of this and it is what the record should contain.**

✅ **And F6's scope claim is now measured, not assumed.** The as-run control config (git `73c3f42`) was diffed against the arm's captured `configs_as_run/config_hyperparameters.py`: **exactly two keys differ** — `loss_class_pos_weight` 2.0→1.0 and the added `trajectory_log_path`. Nothing else. The `copper_kestrel` PR commit (`628d5d80`) did touch the control's config file (24 insertions) but changed **zero** `'key': value` lines — comments and docstring only.

⚠️ The residual risk is still named, not dismissed: the hook is read-only *by inspection*, not by measurement. Nobody ran the control twice — once with logging, once without — to prove byte-identity. **That measurement does not exist, and the ruling rests on a code read.**

---

## 7. Method, and every caveat that bears on the reading

- ⛔ **n = 1 per arm. No seed replication.** The pre-registration pre-commits that a screen **can kill an idea** and **cannot license a fleet change**. §2's mechanism is a **reading of the pattern** — strongly supported by the Gneiting & Ranjan theorem and by the monotone structure in §2, but **not separated from seed noise by measurement**. It must not be asserted as established.
- **Calibration method, unchanged from P2:** posterior-mean gate over D=4 locked-dropout draws; bin edges `[0, .01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01]`; medians over bins with n ≥ 1000; `NP_SEED = TORCH_SEED = 42`; `BASE_ORIGIN = 456`; `N_ORIGINS = 13`; `LEADIN_START = 421`; CPU. The positive fraction is a **pooled mean**, never a median over cells (draws quantise at 1/16).
- **Two independent paths agree.** Local inference and the pod's own gate slice give P-1 within ~0.002 (control 0.0154 / 0.0175; arm 0.0303 / 0.0298), so the headline is not an artefact of either path. ⚠️ **Rev 3 — what this cross-check does and does not establish.** Both paths average over **D = 4** draws, not 4 against 16: `hydranet_inference.py:1707-1709` fills the gate cube with `np.repeat(prob_thwc[..., None], posterior_K, axis=-1)`, so the pod slice's 16 columns hold **4 distinct gate values repeated 4×** — the head-sample axis `K` is **pure duplication for the gate** and genuinely varies only the magnitude draws. So `g.mean(1)` over 16 columns is arithmetically the 4-draw mean, and `slice_calib.py:32`'s comment *"posterior-mean gate over the 16 draws"* is literally wrong. The agreement is therefore a **cross-stack** check (torch 2.10 vs 2.14) on an identical `D=4` structure — which is what we needed — and **not** evidence about draw count.
- **Validation before interpretation.** P2's statistics block applied verbatim to the stored `violet_visitor` array reproduces its published 0.0054 / 0.0573 / 1.0365 **exactly**; the control's gate AP (0.5324) and positive fraction (0.4999) reproduce the composition arm's published values **exactly**. The harness is wired correctly.
- **Stack skew.** Pod torch 2.14.1, local torch 2.10.0+cu128; numpy matches at 1.26.4. Mitigated twice: both arms measured on the **same** local stack, and cross-validated against the pod's own slice.
- **Library version matched.** The pod ran `views-hydranet==0.1.3`; the local checkout is 0.1.3 with no library drift since the release tag.
- ⚠️ **One harness override, and it names a defect.** `vehicle.py` hardcodes `EXPECTED_SIDECAR["output_distribution"] = "nb"` and **VOIDs on anything else** — so it refuses **both** kestrels, which are `truncated_nb`. A scratchpad copy overrode that one value after asserting the committed file's sha256 (`a77044813c50b8f8052094bb0c913b7d7ab23839d5d4d67041bcb49f2466a4d8`) was unchanged; nothing else was altered. **The instrument refuses every arm it was built to compare.** That is a latent generality defect and it deserves its own issue (§9).
- **CRPS comparability checked:** both arms' metricframes carry `data_version = 504`, `scoring_code_version = 2.0.0`, the same partition `{'train': (121,456), 'test': (457,504)}` and `level = pgm`, selected on `metric == "CRPS"`, `group_id == "mean"`.
- ⚠️ **The CRPS row was selected POSITIONALLY, not by name.** Three rows match that selection (month-wise, step-wise, time-series-wise) and the readout took `v[sel][0]` — the first in array order, which happens to be month-wise in all six cells. The numbers are right and consistently the same `eval_type` on both sides, so the comparison is valid, but **a reordering by the scorer would silently switch `eval_type` with no error raised.** That is the same failure class as `vehicle.py`'s hardcoded family above — selecting by position or by hardcoded constant rather than by name — and it is what made the missing label invisible.

---

## 8. Found, not chased

⚠️ **Everything in this section is derived and NOT pre-registered.** None of it is a result.

### 8.1 ⛔ There is NO honest `pos_weight`. Do not go looking for one.

Two numbers were put forward and **both are void**: the readout's ≈1.55 and this session's corrected ≈1.41. Not merely support-dependent — **ill-posed**. `implied_w` varies 6.8× (control) and 12.8× (arm) across bins and changes the sign of the error inside each arm (§4.2), so both figures were medians of a non-constant.

⛔ **Rev 3 — this is the first of THREE instances of one defect class, and the class now has a control.** All three were point summaries of a quantity that varies across its own free parameter and **straddles the value being tested**:

| instance | the free parameter | the family | crosses |
|---|---|---|---|
| `implied_w` ≈ 1.5 (this section) | bin membership | 6.8× control, 12.8× arm | **1.0** |
| the fit's intercept (§8.6) | bin weighting | 1.91 logits across bins | **zero** |
| a "fit-free" stationary-condition ratio (§8.6) | bin count | 0.628 → 1.209 | **1.0** |

**Standing rule, adopted jointly with the `views-models` seat on 2026-10-18 and now in force for the rest of this programme: before any scalar summary is reported, sweep whatever free parameter it depends on and show the family does not cross the value being tested.** Had it been in force, none of the three would have been published — two of them mine. The sweep is cheap; it is the only check that separates a measurement from an artefact of its own tuning.

⛔ **`pos_weight` is the wrong shape of correction** (§2.4): it moves the level `a`, and the defect is in the slope `b`. Searching over `w` can only slide the crossing point along a curve that stays bent. **Any `w` that produces good calibration numbers will have produced them by putting the crossing where the median happens to look** — which is precisely the situation we are in now with `w = 2`, and precisely the failure this arm exposed.

### 8.2 ⛔ Temperature scaling is not merely "principled" — it is the MATCHING fix

This is now a mechanism rather than a preference:

| | moves | |
|---|---|---|
| `pos_weight` | the level `a` | wrong knob |
| **temperature scaling** (`logit / T`) | **the slope `b`** | ⛔ **the broken parameter** |

So temperature scaling has **exactly the one degree of freedom that is broken**, and by §4.3's rank argument it **provably cannot damage AP** — monotone transformations are invisible to a ranking score.

Cost: a one-parameter fit on arrays already in hand. **Seconds on CPU, no GPU run, no arm.**

⚠️ Limit: applying it across all 36 horizons needs the gate posterior retained at h≥2, which is exactly **#558 / C-161**. At **h=1 it is possible today** from `gate_h1/`. ⚠️ And it is a *post-hoc* repair — it fixes the emitted probabilities without touching whatever in training compresses them, which §8.4 says is the real target.

### 8.3 `pos_weight` cannot be learned

It lives **inside** the loss, so minimising that loss with respect to it is degenerate. Any search over it is an outer loop, not gradient descent.

### 8.4 The real target is whatever COMPRESSES the gate's logits

§2 identifies the defect as under-dispersion with `b ≈ 1.14` in the control and `b ≈ 1.42–1.46` in the arm. **Rev 3 narrows the field.** Rev 2 named four speculative candidates and pointed the reader at the MC-dropout posterior mean as the leading, cheap-to-check one. **That candidate is now closed on theory, and two better ones are named.**

#### ⛔ CLOSED by derivation: the MC-dropout posterior mean is NOT an independent cause

Rev 2 reasoned that averaging `D = 4` draws of a sigmoid shrinks extremes toward the centre. The algebra is right — `E[sigmoid(z+ε)] ≈ sigmoid(z/T)` with `T = sqrt(1 + πσ²/8) > 1` — but **training cancels it.** Training runs `LockedDropout` with `locked=False` (`locked_dropout.py:61-62`), i.e. a **fresh mask on every forward call**, so the loss averages over masks. Setting the expected weighted-BCE gradient to zero over both `y` and `ε`:

```
E_ε[ sigmoid(z+ε) ]  =  w·p / (1 + (w−1)·p)
```

**The stationary condition is on the MEAN OF THE SIGMOID over masks — which is exactly the functional inference reports.** The Jensen compression is absorbed into the weights during training. It therefore contributes nothing on its own, and bites only to the extent that something displaces the model from that optimum. Which is the next two items.

⚠️ This also means a plausible "free fix" is **not** free: reporting `sigmoid(mean logit)` replaces a calibrated functional with an uncalibrated one and may **overshoot** to `b < 1`.

#### ⛔ CO-LEAD CANDIDATE — BatchNorm recalibration fitted on the wrong distribution

`_recalibrate_bn` (`training_engine.py:1294-1301`, `bn_recalibrate: True` in both arms) resets every BatchNorm's running statistics **after training** and re-accumulates them forward-only with `momentum=None` over **real 32×32 activity-seeking curriculum windows** (positive rate 0.004508). Inference then runs the **360×720 global grid (64,818 land cells)** (0.003448) with BatchNorm in eval mode, applying those fixed statistics.

The class head has BatchNorm immediately upstream of its final linear layer (`bn_dec_conv1_head1_class`, defined at `HydraBNrecurrentUnet_06_LSTM4.py:249`, used at `:653`). If the recalibration distribution's activation variance exceeds the global grid's — which an activity-enriched window selection would produce — eval-mode BatchNorm divides by too large a σ, **shrinking the features entering the head and multiplicatively compressing the logit.**

It fits every property of the defect: a **pure slope error in logit space**; applied **after the last gradient step**, so no training optimum can absorb it; **present in both arms**; independent of `pos_weight`. And it is precisely C-184's blind spot — that fix was validated on flipping seed-bimodal eval collapse (*"6/6 bad seeds GOOD"*, per its own docstring), **never on calibration slope**.

#### ⛔ CO-LEAD CANDIDATE — `weight_decay: 0.1` on the class heads

Both arms set `weight_decay: 0.1` — **ten times the AdamW default** — and `choose_scheduler` builds `AdamW(unet.parameters(), weight_decay=config["weight_decay"])` (`utils/utils.py:277-282`) with **no exclusion list**. So `dec_conv4_head{1,2,3}_class` are decayed. The mechanism for excluding a group already exists and is used twice for the balancer (`training_engine.py:113`, `:136`); the heads are simply not in it.

Decoupled decay shrinks the head's weights multiplicatively every step, and shrinking a final linear layer **compresses its output range** — the textbook cause of under-confidence in a regularised classifier. Critically, it interacts with the item above: `E[s] = w·p/(1+(w−1)·p)` is the stationary point of the **loss**, while AdamW optimises **loss + penalty**, whose optimum is displaced toward smaller weights, i.e. **less extreme logits**. **Weight decay moves the optimum away from calibration by construction, in the compressing direction** — so it predicts the sign *a priori*, with no statistic required.

#### The null, carried explicitly — a resolution limit

Under-dispersion is also the generic signature of a predictor whose features cannot resolve per-cell `p`. If the model separates "quiet" from "active" but not 2% from 8%, the best achievable curve **is** flatter than truth, `b > 1` is an honest statement of limited resolution, and **no monotone recalibration can fix it** — it can only re-centre it. This is the hypothesis under which §8.2's temperature scaling "succeeds" on calibration and buys nothing real.

#### ✅ Ruled out cheaply: it is not under-training

`copper_kestrel`'s trajectory CSV (the per-lesson instrumentation shipped in PR #398, enabled for the arm only) gives the class-head logit mean per lesson. Block means: `−6.937 → −7.588 → −7.987 → −8.121 → −8.030 → −7.984`. OLS slope over lessons 100–200 is **−0.0019/lesson**; over 200–300 it is **+0.0003/lesson** — opposite in sign and an order of magnitude below the within-block sd (~0.09). **The head's logit LEVEL is converged by roughly lesson 150.** ⚠️ Necessary, not sufficient: the hook records only the mean (§9 issue 7), so a still-expanding *spread* cannot be excluded from existing artifacts.

#### The free experiment that separates the two co-leads

Fit the calibration slope on **32×32 activity-seeking curriculum windows** — the BatchNorm-recalibration distribution — and compare it to the global-grid slope. `VolumeSampler` + `CurriculumLearner` are already driven this way at `reports/2026-10-13_composition_arm_outcome/r3_per_target.py:93-108`.

- **BatchNorm candidate predicts:** `b ≈ 1` on the windows, `b ≈ 1.15` globally.
- **Weight-decay candidate predicts:** `b > 1` on **both**, since decay compresses the weights themselves.

⚠️ Its method must be locked before it runs — the window slope has **two** free parameters (bin count and window count) and §8.1's standing rule applies to it. That is the job of `reports/2026-10-18_gate_underdispersion_prereg.md`.

**Treatments are secondary. This is a diagnosis task on arrays already in hand, not an arm.**

### 8.5 An endpoint the control's own write-up left open is now closed

`2026-10-13_composition_arm_outcome/OUTCOME.md:22` recorded `act_ratio` as *"not measured — h≥2 gate posterior deliberately not retrieved (#558)"*.

⛔ **That was wrong about what `act_ratio` needs.** It reads the **emitted composed draws** (`lr_sb_best`, shape (2333448, 16) = 64,818 cells × 36 months × 16 draws, `time` 457–492), which **did** come home in `draws/`. Only the raw `by_*` gate posterior at h≥2 died with the pod, and `act_ratio` does not use it.

**The control's `act_ratio` at h18 is 2.3520**, back-filled into that document as part of this write-up. The earlier reasoning there was **wrong, not merely unlucky** — it confused two different archived objects.

### 8.6 ⛔ A methodological result: §2.4's identity CANNOT be tested fit-free when `w ≠ 1`

§2.4's exact identity invites a check that needs no fitting: the stationary condition is `E[s_i] = φ(p_i)` with `φ(p) = wp/(1+(w−1)p)`, so compare the observed mean gate against `mean_i[φ(p_i)]`. **That test does not exist for `w ≠ 1`, and the reason is worth recording because two sessions walked into it.**

`p_i` is **never observed** — only a Bernoulli draw from it. Any estimator built from binned outcome frequencies therefore carries a **bin-level Jensen bias**, and `φ` is strictly concave for `w > 1` (`φ'' = −2w(w−1)/(1+(w−1)p)³ < 0`). The bias **flips sign with bin width**:

| bins | cells/bin | `mean_i[φ(p_i)]` | ratio obs/pred |
|---|---|---|---|
| `φ(p̄)` — no averaging at all | — | 0.006871 | **0.606** |
| 12 | 70,220 | 0.006631 | 0.628 |
| 200 | 4,213 | 0.005490 | 0.759 |
| 1,000 | 843 | 0.005303 | 0.786 |
| 100,000 | 8.4 | 0.004981 | 0.837 |
| 400,000 | 2.1 | 0.004223 | **0.987** |
| 500,000 | 1.7 | 0.004040 | **1.032** |
| per cell (analytic) | 1.0 | `p̄` = 0.003448 | **1.209** |

Coarse bins destroy within-bin variance in `p`, raising `mean[φ]` and depressing the ratio. Fine bins replace `p` with its own sampling noise — at ~2 cells per bin `p_b ∈ {0, ½, 1}` — lowering `mean[φ]` and inflating the ratio. The per-cell limit is analytic: `φ(0) = 0` and `φ(1) = 1` for **every** `w`, so `mean_i[φ(y_i)] = p̄` exactly.

⛔ **No resolution is unbiased, the family is monotone, and it crosses 1.0 between 2.0 and 1.7 cells per bin.** The point is **not** that any one figure is right — it is that **no point on the curve is privileged and the curve contains 1.0**. Debiasing would require a *model* for `p`, i.e. the fitting the test existed to avoid.

**At `w = 1`, `φ(p) = p` is linear, the Jensen gap is exactly zero at every resolution, and the test is valid.** The arm sits at **0.945×**.

⚠️ **So the apparent asymmetry between the arms — "the control misses its own optimum badly, the arm nearly hits it" — is not a fact about the two models. A valid fit-free test exists for one arm and cannot exist for the other.** That is a property of `pos_weight`, and the control's distance from its stationary condition is **UNMEASURED**: not small, not large, not signed.

The same applies to the intercept signature in §2.4: the per-bin deviation `logit(obs) − logit(gate)` ranges **1.91 logits** (−1.01 to +0.90), so there is no single intercept to test. A bin-unweighted fit is reproducible to **0.0001** from bin weighting plus the slope defect alone, with no `log(w)` information left in it. Weighted by data mass the signature is present at **−0.9825**. Three implementations (quantile-digitize, sorted-index split, vectorised cumsum) agree at every shared resolution, so none of this is an artefact of a binning scheme.

**Provenance:** derived jointly with the `views-models` seat across three adversarial rounds, 2026-10-18. Three of this session's claims and two of theirs were withdrawn in the process; the error trail is kept at `reports/2026-10-18_gate_underdispersion_prereg/superseded_v2_with_errors.md`.

---

## 9. What this closes, what it opens, and what it leaves alone

**Closed:**
- `pos_weight = 1.0` as a skill or correctness improvement. **Killed.** It is theoretically the right value and empirically worse — and §2.4 explains why that is not a paradox: `w` cannot address the actual defect at any value, so being "principled" about `w` was always going to be beside the point.
- ⛔ **Tuning `pos_weight` at all.** Not just 1.0 — the whole axis (§8.1).
- The composition arm's open `act_ratio` endpoint (§8.5).

**Opened:**
- ⛔ **The gate is under-dispersed — its logits are compressed, `b ≈ 1.14` in the control** (§2). The main finding. It is present in the control, so it is a property of the platform and not of this arm.
- ⛔ **No `pos_weight` can fix it**, because `w` moves the level and the defect is in the slope (§2.4). That closes a whole direction of work.
- **Temperature scaling is the matching fix** (§8.2) — the one knob that moves `b`, provably harmless to AP, seconds on CPU.
- **What causes the compression** (§8.4) — narrowed by rev 3 to **two co-lead candidates**, both defects in views-hydranet and both acting outside the training optimum: **BatchNorm recalibration fitted on 32×32 curriculum windows and applied to the 360×720 grid**, and **`weight_decay: 0.1` on the class heads**. The MC-dropout posterior mean is **closed by derivation** — training calibrates the very mean-of-sigmoid that inference reports. Under-training is **ruled out** (the head's logit level converges by ~lesson 150). One free CPU experiment separates the two survivors; its method must be locked first.

**Unaffected and still pending:**
- **The denominator arm** (`body_nll_denominator`, `reports/2026-10-15_loss_balance_prereg.md`). Untouched by this result — different half of the loss, different mechanism, measured gradient evidence of its own. ⚠️ Note its config PR is expected to go **red** in views-models CI by design: the pin-floor guard will refuse the key because the runner's pin floor is 0.1.1 and the key needs 0.1.3.
- **M-1 / C-162** — the balancer's learning rate never reaching its parameter group. Latent; this arm kept the balancer frozen.
- **The joint-vs-marginal question** — the loss is a sum of per-target NLLs, i.e. a product of marginals, and conditional independence across violence types is assumed and untested.

**Issues this readout argues for**, worst first:

1. ⛔ **`p2_gate_calibration.py`'s `sep` tolerance stamps a badly miscalibrated gate as "D2 — already calibrated".** The arm passes by **0.0009** while under-stating by 0.1036 at p≈0.12. `sep` scales with the base rate, so on a sparse field it inflates until it admits almost anything, and the verdict is written into every stored `.npz` (§4.4). **Higher severity than the rest, because it PASSES SILENTLY** where `vehicle.py` at least refuses loudly. ⚠️ And `:194` already contains the correct diagnosis — *"the gate cannot be repaired by a scalar pos_weight"* — behind the wrong condition (§2.5).
2. ⛔ **P-1 as pre-registered cannot see either defect found by this arm.** It is a median of *absolute* errors over 4–5 equally-weighted bins, one of which holds **96%** of the cell-months. It discards the sign (so it cannot see the rotation, §2.2) and it silently changes support between arms (so it cannot see the comparability problem, §4.2). **The next pre-registration should read the per-bin curve, or fit level and slope, rather than a median of absolute errors.** This is the methodological lesson of the arm and it sits beside §5.1's threshold-sizing lesson.
3. `vehicle.py`'s hardcoded `output_distribution: "nb"` refuses every arm it was built to compare (§7).
4. **Selecting metricframe rows positionally** (`v[sel][0]`) rather than by name — three `eval_type`s match and a scorer reordering would switch silently (§7). Same failure class as 3.
5. Deliveries are asymmetric in what they archive — `configs_as_run/` present for the arm, absent for the control; the gate slice nested one level deeper on the control (§4.4).
6. #558 / C-161 — the gate posterior at h≥2 not coming home — now blocks §8.2's all-horizon application, which is a **second concrete cost** for an issue that previously only cost retrieval convenience.
7. ⛔ **The per-lesson gate instrumentation I shipped in PR #398 records the MEAN only, so it cannot answer the question that is now live.** `training_engine.py:1430-1432`'s forward hook accumulates `_o.detach().mean().item()` — one scalar per lesson, **no sd, no quantiles**. The live question (§8.4) is whether the logit *spread* is still expanding, and the statistic is not in the file. Adding an sd is a one-line change with no effect on training dynamics. ⚠️ Compounding it: `trajectory_log_path` was set for **`copper_kestrel` only** (`config_hyperparameters.py:233`); `amber_kestrel` sets none, so **no control baseline exists and never did.** A diagnostic enabled on one arm of a two-arm contrast cannot answer a question about either.
8. **`fed_grad_max` is `0.0` on all 300 rows** of the arm's trajectory CSV — an inert column in the same instrumentation. C-324 shape: tested, logged, and carrying no information.
10. ⛔ **The committed evidence arrays MISRECORD the arm's defining variable.** `p2_gate_calibration.py:91` hardcodes `SHIPPED_POS_WEIGHT = 2.0` (commented *"violet_visitor config_hyperparameters.py:84"*); `:185` assigns it to `pw` inside the **D2 verdict branch**; `:199` writes it to the `.npz` as the field **`pos_weight`**. Measured from the committed arrays: **`copper_kestrel`'s `.npz` records `pos_weight=2.0`, and `copper_kestrel` ran at 1.0.** A reader trusting the field name would conclude the arm never changed its one variable. Worse, because `pw` is assigned inside the branch, the field means *"the value this branch happens to carry"* — a D1 outcome would have written 1.0. ⚠️ **No conclusion here is affected** (the `gate`/`truth` arrays are measurements and the report takes 1.0 from the config).

    ⛔ **CORRECTION (rev 3a).** An earlier wording of this issue said the defect *"survived … because nothing asked what the field name was asserting."* **That is false and I withdraw it.** `evidence/EVIDENCE.md:17-20` carries a dedicated heading — *"**TRAP — do not cite the `pos_weight` field inside the npz**"* — with the correct diagnosis, naming the hardcoded constant and `:185`'s branch assignment, written at evidence-staging. Detection worked. **The failure is that a documented defect in a stored artifact became a CAVEAT instead of a FIX** — and a caveat does not travel with the artifact. The `.npz` still says `pos_weight = 2.0`, three files away from the note that corrects it, so anyone who loads it without reading `EVIDENCE.md` gets the wrong answer. ⛔ **Remediation is small and concrete and is Simon's call:** rename the harness field to something true under the weakest reading (e.g. `shipped_pos_weight_reference`) and regenerate, or drop the field. See `2026-10-18_gate_underdispersion_prereg/PREREG.md` Amendments 6–7.
11. ⛔ **C-184's BatchNorm recalibration estimates 15 layers' running statistics from a DEGENERATE sample, and nobody checked because the collapse was the only endpoint.** Measured 2026-10-18: `planner.get_lesson(w)` for `w ∈ 0..29` — the indices `_recalibrate_bn` uses — yields only **6 distinct `(target, threshold)` pairs** with **4–5 qualified anchor cells each**. The mechanism is the `roof_ratio` clip (`curriculum.py:91`): `ratio = min(ratio, 0.95 × 0.7) = 0.665` holds for steps **0–213 of 900**, so the recal draw sits entirely inside a flat region of the curriculum. **The statistics shipped in both artifacts were estimated over a handful of conflict hotspots and are applied frozen to 64,818 land cells.** C-184 was validated on flipping seed-bimodal eval collapse (*"6/6 bad seeds GOOD"*) and it worked — so it repaired the collapse **while being structurally incapable of producing representative statistics.** Both are true; the second was never an endpoint. This is now the measured premise of H1 (§8.4). See `2026-10-18_gate_underdispersion_prereg/PREREG.md` Amendment 8.
12. ⛔ **C-184's recalibration is arguably RIGHT BY DESIGN, and the docstring states the wrong criterion.** `_recalibrate_bn`'s docstring says it re-accumulates *"over real curriculum windows, so eval-mode matches the true activation distribution"* — implying the target is the **inference** distribution. Measured 2026-10-18 (E6f): re-estimating on the actual inference distribution moves the gate's calibration slope **1.1875 → 0.7793**, crossing 1.0 and leaving the gate over-predicting occurrence by **5.75×**. The reason is that `:1321` calls `train(...)`, so during both training and recalibration BatchNorm normalises each 32×32 activity-seeking crop by **its own batch statistics** — so the head's weights were trained against hot-window moments. ⛔ **The criterion for BatchNorm eval statistics is "give the head the input scale it was trained against", not "match the inference distribution"** — and by that criterion the hot-window sample is correct. The docstring asserts the other criterion, and issue 11's degeneracy finding is therefore a finding about **the training crop selection**, not about the recal pass. See `2026-10-18_gate_underdispersion_prereg/PREREG.md` Amendment 10 §4.
9. ⛔ **The repo's lint gate is structurally blind to every script in `reports/`.** `.gitignore:203` contains `reports/`, and `ruff` honours gitignore patterns **even for files that were force-added and are tracked** (confirmed by `git ls-files`). So `ruff check .` — what CI runs, and what it passes — silently skips the whole directory; `ruff check --no-respect-gitignore reports/` reports **461 errors**. Every dossier script this programme has linted was linted only because it was named explicitly on the command line. The gate would never have caught any of them.
