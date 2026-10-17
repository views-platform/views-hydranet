# Outcome — the `pos_weight` arm. KILLED, and it found something bigger on the way out.

**Date:** 2026-10-17 · **Pre-registration:** `reports/2026-10-16_pos_weight_prereg.md` (commit `02d1294`, written before the run)
**Arm:** `copper_kestrel` — the experimental run, gate trained at `pos_weight = 1.0`
**Control:** `amber_kestrel` — the already-delivered run, gate trained at `pos_weight = 2.0`
**Readout measured by:** the `views-models` session. **Written up and independently re-verified by:** this session.
**Evidence:** `evidence/EVIDENCE.md` (273 lines of primary data) + the `.npz` arrays and scripts beside it.

---

## 0. In plain words, before any numbers

The model has two halves. One decides **where** conflict happens (the **gate**). The other decides **how much** (the **body**).

The gate is trained with a setting called `pos_weight`, which was set to **2**. Setting it to 2 tells the gate: *when conflict does happen, count that twice as much*. There is a theorem saying this is the wrong thing to do — it mathematically guarantees the gate will report wrong probabilities. The honest setting is **1**.

So we trained one model at **1** instead of **2**, and asked: does the gate now report honest probabilities?

**It got worse. Twice as wrong.** The arm is killed.

But here is the part worth staying awake for. At `pos_weight = 1` the gate **should** have been honest, because the theorem says 1 is the correct value. It wasn't. And it was wrong in the **opposite direction** from before — at 2 the gate *overstated* conflict, at 1 it *understated* it.

**That means something else in the gate is pushing its probabilities down, and setting `pos_weight` to 2 had been accidentally cancelling it out.** We have been running a known-wrong setting that was silently correcting an unknown-wrong one.

Finding the second problem is worth more than the experiment was designed to produce.

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

## 2. The finding that matters more than the verdict

### The theorem

`weighted_bce` at `pos_weight = w` has its optimum at

```
s* = w·p / (1 + (w−1)·p)
```

**Gneiting & Ranjan (2011), Theorem 1:** weighting a proper scoring rule by a function of the *outcome* makes it **improper** — its minimiser becomes a tilted law rather than the truth. So `w ≠ 1` is **mathematically guaranteed** to produce a miscalibrated gate, and `w = 1` is the only principled value.

### What the theorem predicts, and what happened

| | trained at | implied `w` the gate actually shows | reading |
|---|---|---|---|
| control | 2.0 | **1.3090** | over-states conflict, as the theorem predicts for w>1 |
| arm | 1.0 | **0.6252** | ⛔ **under-states** — the theorem predicts 1.0 here |

**At `w = 1` the gate should have been honest. It is not, and it errs the other way.**

A tilt cannot explain an error that appears when the tilt is *removed*. So:

> ⛔ **There is a second, independent downward bias in the gate. The platform has been unknowingly compensating for it with an improper loss tilt.**

`w = 2` was pushing predictions up; something else was pushing them down; the two partially cancelled. Taking away the up-push revealed the down-push.

### How the bias behaves — structured, not noise

From the per-bin table (verified independently, §4.2), the arm's calibration error **grows monotonically with the gate's own confidence**:

| gate bin | n | mean gate | observed p | error |
|---|---|---|---|---|
| [0.00, 0.01) | 818,958 | 0.0012 | 0.0003 | 0.0008 |
| [0.01, 0.05) | 16,679 | 0.0224 | 0.0314 | 0.0091 |
| [0.05, 0.10) | 3,104 | 0.0705 | 0.1221 | **0.0516** |
| [0.10, 0.20) | 1,677 | 0.1409 | 0.2445 | **0.1036** |
| *(below the n≥1000 threshold, for shape only)* | | | | |
| [0.30, 0.40) | 421 | 0.3472 | 0.4941 | 0.1469 |
| [0.50, 0.60) | 246 | 0.5471 | 0.7764 | **0.2293** |

**The more confident the gate is, the more it under-states.** That is a systematic, ordered failure — it is not seed noise, and it tells us the bias is multiplicative-ish in the probability rather than a constant offset. **This is the single most useful diagnostic in the readout and it is new.**

---

## 3. ⛔ The CRPS result is an artefact. It must never be quoted as a gain.

| target | control | arm | change |
|---|---|---|---|
| sb | 0.1594 | 0.1010 | **−37%** |
| ns | 0.1027 | 0.0173 | **−83%** |
| os | 0.0265 | 0.0109 | **−59%** |

**This is not skill, and it is not a second fact.** It is §1's P-6 seen through a blind spot.

The field is **99.7% zero**. On such a field, a forecast that fires less scores better *by construction*. `activation_metrics.py` says so in its own words:

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

⚠️ **But one derived number in the evidence must be corrected.** `EVIDENCE.md` §2 interpolates the honest `pos_weight` using the control's **1.3090**, giving **≈1.55**. On the common support the control's implied `w` is **1.5381**, and the same interpolation gives

```
1 + (1.000 − 0.6252) / (1.5381 − 0.6252) = 1.41
```

So the estimate is **support-dependent: ≈1.4 to ≈1.55**. The qualitative conclusion is unchanged — the honest value lies in (1, 2), it is **not** 2.0, and the direction of 2.5 is ruled out — but any single number quoted for it is an artefact of which bins were included, and that must travel with it. See §8.1.

### 4.3 P-3 held, as the pre-registration structurally predicted — with a note

Prereg §3.2 argued *in advance* that AP **cannot** move by this arm's mechanism: the tilt `s* = wp/(1+(w−1)p)` is **strictly monotone in p**, and average precision depends on the gate **only through ranks**, so a pure recalibration is invisible to it.

**Measured: 0.5324 → 0.5385.** Essentially unchanged, as predicted. **A falsifiable structural prediction, made before the run, and correct.**

⚠️ The note: paired by month the difference is **+0.0075**, bootstrap CI **[+0.0014, +0.0136]**, with the arm higher in **11 of 13** months. The CI **excludes zero** and its upper bound sits just outside the ±0.010 band. So this is **held-with-a-note, not a clean null** — there is a small, consistent positive residual that the monotone argument does not explain. Candidate explanations: seed noise (n=1, nothing separates it), or the altered loss changing what the trunk learned. **Unresolved, and §7 of the prereg forbids reading it as confirmation of the hypothesis.**

### 4.4 Traps in the evidence, confirmed

| | |
|---|---|
| ⛔ **Do not cite `pos_weight` from the `.npz` files.** | **Both** store `2.0`. Verified: it is the harness's hardcoded `SHIPPED_POS_WEIGHT` constant (`p2_gate_calibration.py:185`), not a value read from the model. **Meaningless here.** |
| ✅ **The authoritative arm identity** | the library's own log line, `weighted_bce_loss.py:41`, on the *constructed* criterion: control `run.log:430` → **2.0**; arm `run.log:95` → **1.0**. This is exactly what the prereg's §6a pre-flight demanded. |
| ⚠️ **Archival asymmetry, not a run difference** | the arm's delivery ships `configs_as_run/` (6 files); the control's does not, so the control's as-run config must be read from git at `73c3f42`. The two arms are **not symmetric as records**, which is a provenance weakness worth fixing for future arms. |

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

⚠️ The residual risk is named, not dismissed: the hook is read-only *by inspection*, not by measurement. Nobody ran the control twice — once with logging, once without — to prove byte-identity. That measurement does not exist, and the ruling rests on a code read.

---

## 7. Method, and every caveat that bears on the reading

- ⛔ **n = 1 per arm. No seed replication.** The pre-registration pre-commits that a screen **can kill an idea** and **cannot license a fleet change**. §2's mechanism is a **reading of the pattern** — strongly supported by the Gneiting & Ranjan theorem and by the monotone structure in §2, but **not separated from seed noise by measurement**. It must not be asserted as established.
- **Calibration method, unchanged from P2:** posterior-mean gate over D=4 locked-dropout draws; bin edges `[0, .01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01]`; medians over bins with n ≥ 1000; `NP_SEED = TORCH_SEED = 42`; `BASE_ORIGIN = 456`; `N_ORIGINS = 13`; `LEADIN_START = 421`; CPU. The positive fraction is a **pooled mean**, never a median over cells (draws quantise at 1/16).
- **Two independent paths agree.** Local inference and the pod's own gate slice give P-1 within ~0.002 (control 0.0154 / 0.0175; arm 0.0303 / 0.0298), so the headline is not an artefact of either path.
- **Validation before interpretation.** P2's statistics block applied verbatim to the stored `violet_visitor` array reproduces its published 0.0054 / 0.0573 / 1.0365 **exactly**; the control's gate AP (0.5324) and positive fraction (0.4999) reproduce the composition arm's published values **exactly**. The harness is wired correctly.
- **Stack skew.** Pod torch 2.14.1, local torch 2.10.0+cu128; numpy matches at 1.26.4. Mitigated twice: both arms measured on the **same** local stack, and cross-validated against the pod's own slice.
- **Library version matched.** The pod ran `views-hydranet==0.1.3`; the local checkout is 0.1.3 with no library drift since the release tag.
- ⚠️ **One harness override, and it names a defect.** `vehicle.py` hardcodes `EXPECTED_SIDECAR["output_distribution"] = "nb"` and **VOIDs on anything else** — so it refuses **both** kestrels, which are `truncated_nb`. A scratchpad copy overrode that one value after asserting the committed file's sha256 (`a77044813c50b8f8052094bb0c913b7d7ab23839d5d4d67041bcb49f2466a4d8`) was unchanged; nothing else was altered. **The instrument refuses every arm it was built to compare.** That is a latent generality defect and it deserves its own issue (§9).
- **CRPS comparability checked:** both arms' metricframes carry `data_version = 504` and `scoring_code_version = 2.0.0`, selected on `metric == "CRPS"`, `group_id == "mean"`.

---

## 8. Found, not chased

⚠️ **Everything in this section is derived and NOT pre-registered.** None of it is a result.

### 8.1 The honest `pos_weight` is somewhere near 1.4–1.55, and chasing it would be a hack

Interpolating the effective tilt to 1.0 gives **≈1.41** on a common bin support and **≈1.52–1.55** on each arm's own bins (§4.2). Two points, n=1 each.

⛔ **And tuning to it would still be the wrong move**: good calibration numbers, underlying defect intact, cancellation valid only at the data distribution it was tuned against, **silent failure under distribution shift**. It would reproduce exactly the situation we are in now — a number that looks right for a reason nobody has written down.

### 8.2 The principled treatment is post-hoc recalibration

Platt scaling or temperature scaling on the gate. By §4.3's rank argument it **provably cannot damage AP**, because monotone transformations are invisible to a ranking score. Cost: a one-parameter fit on arrays already in hand — seconds on CPU, **no GPU run**.

⚠️ Limit: applying it across all 36 horizons needs the gate posterior retained at h≥2, which is exactly **#558 / C-161**. At **h=1 it is possible today** from `gate_h1/`.

### 8.3 `pos_weight` cannot be learned

It lives **inside** the loss, so minimising that loss with respect to it is degenerate. Any search over it is an outer loop, not gradient descent.

### 8.4 The real target is the downward bias itself

§2 says there is a second, independent mechanism pushing the gate's probabilities down, and §2's monotone structure says it scales with the gate's own confidence. **Treatments are secondary. Identifying that mechanism is the next question**, and it is a diagnosis task on arrays already in hand, not an arm.

### 8.5 An endpoint the control's own write-up left open is now closed

`2026-10-13_composition_arm_outcome/OUTCOME.md:22` recorded `act_ratio` as *"not measured — h≥2 gate posterior deliberately not retrieved (#558)"*.

⛔ **That was wrong about what `act_ratio` needs.** It reads the **emitted composed draws** (`lr_sb_best`, shape (2333448, 16) = 64,818 cells × 36 months × 16 draws, `time` 457–492), which **did** come home in `draws/`. Only the raw `by_*` gate posterior at h≥2 died with the pod, and `act_ratio` does not use it.

**The control's `act_ratio` at h18 is 2.3520**, back-filled into that document as part of this write-up. The earlier reasoning there was **wrong, not merely unlucky** — it confused two different archived objects.

---

## 9. What this closes, what it opens, and what it leaves alone

**Closed:**
- `pos_weight = 1.0` as a skill or correctness improvement. **Killed.** It is theoretically the right value and empirically worse, which is the interesting part, not a paradox to be re-tested.
- The composition arm's open `act_ratio` endpoint (§8.5).

**Opened:**
- ⛔ **The second downward bias in the gate** (§2) — the main finding, and the next thing to chase.
- The gate's **confidence-dependent** miscalibration structure (§2) — a diagnostic lead that did not exist before this arm.
- Post-hoc recalibration as the principled treatment (§8.2), cheap and available at h=1 today.

**Unaffected and still pending:**
- **The denominator arm** (`body_nll_denominator`, `reports/2026-10-15_loss_balance_prereg.md`). Untouched by this result — different half of the loss, different mechanism, measured gradient evidence of its own. ⚠️ Note its config PR is expected to go **red** in views-models CI by design: the pin-floor guard will refuse the key because the runner's pin floor is 0.1.1 and the key needs 0.1.3.
- **M-1 / C-162** — the balancer's learning rate never reaching its parameter group. Latent; this arm kept the balancer frozen.
- **The joint-vs-marginal question** — the loss is a sum of per-target NLLs, i.e. a product of marginals, and conditional independence across violence types is assumed and untested.

**Issues this readout argues for:**
1. `vehicle.py`'s hardcoded `output_distribution: "nb"` refuses every arm it was built to compare (§7).
2. Deliveries are asymmetric in what they archive — `configs_as_run/` present for the arm, absent for the control (§4.4).
3. #558 / C-161 — the gate posterior at h≥2 not coming home — now blocks §8.2's all-horizon application, which is a second concrete cost for an issue that previously only cost retrieval convenience.
