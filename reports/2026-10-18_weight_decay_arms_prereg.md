# Pre-analysis plan — is `weight_decay` the cause of the gate's under-dispersion?

**Date:** 2026-10-18 · **Phase 3**, superseding the Phase-2 slope survey · **Written before any pod starts.**
**Owner:** views-hydranet seat. **Executing seat:** views-models, which **attacks this plan first and does not decide which of its own objections survive.**
**Predecessors:** `2026-10-18_gate_underdispersion_prereg/RESULTS.md` (Phase 1 record) · `RND_PLAN.md` · `2026-10-18_gate_phase2_prereg.md` (superseded, with its four amendments kept as the trail)
**Authorised:** five parallel pods and the arm set below — Simon, 2026-10-18, relayed by the `views-models` seat. ⛔ **This plan does not authorise anything; it is what must exist before the pods start.**

---

## 0. In plain words

The gate's probabilities are squashed. We have ruled out the obvious causes and the cheap fixes. **One suspect is left that we can test at source: `weight_decay: 0.1`** — ten times the usual value, applied to every parameter including the layer that produces the gate.

**This is a cause test, not a product test.** We train small models with the knob changed and look at one number. ⛔ **Nothing here can tell us whether a resulting model is better to serve** — that needs the forecast stage, which we are deliberately not running.

⚠️ **And the honest headline about the design: with two runs per setting, the yardstick we measure the effect against is itself uncertain by a factor of thirty.** §4 derives that and §5 changes the reading rules because of it.

---

## 1. Why Phase 2 died, and why this is worth money

**Simon, 2026-10-18:** *"the models being served are wrong and bad and we are trying to fix them."*

⛔ **He is right, and the Phase-2 slope survey is withdrawn.** Measuring condemned models answers nothing that changes what gets built. ⚠️ **And the provenance detour that consumed the previous round was chasing a question nobody needed answered** — recorded, because it cost both seats a round.

**What makes a cause test worth money instead:** `weight_decay: 0.1`, `bn_recalibrate: True`, `window_dim: 32` and `dropout_rate: 0.15` are **byte-identical across both Phase-1 arms**. ⛔ **The defect is in machinery shared by every hydranet this pipeline trains — so the replacements inherit it too.** That is the difference between a survey of models being thrown away and a test of the thing doing the throwing.

⚠️ **The unidentifiable-serving-weights finding stands and is unaffected** (`2026-10-18_gate_phase2_prereg.md` §2, Amendments 2–4). It is a separate issue and does not depend on this experiment.

---

## 2. The cost structure, and ⛔ whose catch it was

**Simon asked whether the 13 moving origins are needed. They are not** — the slope was never measured from the forecast stage. It comes from the local teacher-forced sweep that produced `b_before = 1.1875` in **554 s** on his machine.

| stage | time | status |
|---|---|---|
| train, 300 lessons | ~2 h 15 m | **REQUIRED** — it is what we are changing |
| forecast / 13 origins | ~8 h 53 m | ⛔ **NOT REQUIRED — skipped entirely** |
| local slope measurement | ~9 min | free, CPU, after the fact |

⛔ **~12 h / ~$16 per arm becomes ~2 h 15 m / ~$3–4.** ⚠️ **Both figures are the other seat's and UNMEASURED by this one. Eleven of eleven cost estimates across both seats have been optimistic. F7 stops the set after arm 1 if its wall time exceeds 2× the estimate.**

⚠️ **#402's `to_cube_samples` work would give a 3.4× ceiling on the stage we are no longer running.** Irrelevant here and deliberately not pulled in — recorded so it is not re-proposed.

---

## 3. The arms

Base config **`amber_kestrel`**, so controls are directly comparable to the published `b = 1.1875`. Everything identical except the varied key.

| arm | name | `weight_decay` | seed | role |
|---|---|---|---|---|
| **1** | **`silver_kestrel`** | `0.1` (current) | **42** | control · ⛔ **and the determinism precondition** |
| **2** | **`bronze_kestrel`** | `0.1` (current) | **43** | ⛔ **RUN-TO-RUN VARIANCE — what makes arms 3/4 readable** |
| **3** | **`cobalt_kestrel`** | `0.01` | **42** | intervention |
| **4** | **`indigo_kestrel`** | `0.01` | **43** | intervention variance |
| **5** | **`russet_kestrel`** | `0.1`, exempting the 3 class heads | **42** | exploratory — ⛔ **NOT a controlled cell, and NOT authorised by this plan** (§8) |

**Names are the `views-models` seat's call** (their CLAUDE.md). **The kestrel family is deliberate**, marking the lineage: `amber` (shipped baseline) → `copper` (`pos_weight` arm) → these five (weight-decay arms of the same line).

✅ **And the names deliberately encode NOTHING about the configs.** Their reasoning, which this seat endorses and would have raised: a name like `wd001_kestrel` **asserts a value that can drift from the config and then lie** — ⛔ **the provenance-mismatch class applied to a directory name.** **The config is the only place the config is stated.**

**Seeds named by value** (their point 5). ⚠️ **This seat proposed `4711` and was wrong** — that is `capture_regimes.py`'s own **anchor-RNG** default, not a model seed, which is the provenance-mismatch class a fourteenth time and caught before it reached a config. **`43` is the repository's established pair for `42`**, verified: 4 roster configs at `np_seed`/`torch_seed` 42 and 2 at 43, and the paired-run dossier names its own runs `fortytwo`/`fortythree`. **Their value stands.**

⛔ **Arm 2 is the point of the design.** amber and copper differ in slope by **0.17** from one config key at n=1 each, and the effect expected from `weight_decay` is **~0.19**. **Those are the same size.** Without arm 2 no result is interpretable at any price.

**The full config diff of each arm against `amber_kestrel` must be printed in this plan before the pods start**, so identity is checkable rather than asserted (their point 5, adopted).

---

## 4. ⛔ THE DESIGN'S CENTRAL WEAKNESS, DERIVED

The other seat's point 1 — *"the decision threshold is measured by the experiment, not fixed in advance"* — is right and is a genuine improvement on Phase 1's absolute bands. ⛔ **But with n=2 per cell the measured threshold is a lottery, and the plan must say so before it runs.**

For `X₁, X₂ ~ N(0, σ²)` iid, the observed spread is `|X₁ − X₂|`:

```
E|X₁ − X₂|  = 2σ/√π      ≈ 1.128 σ
sd|X₁ − X₂| = σ√2·√(1−2/π) ≈ 0.853 σ
⛔ so the observed spread is known to ±76% of itself
5th–95th percentile of the observed spread:  0.088 σ … 2.782 σ  =  a 31.5× RANGE
```

**Derived, and confirmed by simulation at 200,000 draws.**

⛔ **So a hard falsifier of the form "the effect exceeds the control spread" is a coin flip dressed as a decision.** A lucky small spread (0.09σ) makes any effect look decisive; an unlucky large one (2.78σ) hides a real effect. **This is the same class of error as Phase 1's absolute bands, one level up: a threshold whose own uncertainty was never stated.**

### Two structural mitigations, both free

**(i) POOL the two within-pair spreads.** Arms 1–2 give one estimate of σ at `wd = 0.1`; arms 3–4 give another at `wd = 0.01`. **Pooling gives 2 df instead of 1.** Still weak, still free, and it halves the threshold's variance.

**(ii) ⛔ PREFER DOSE-RESPONSE OVER THRESHOLD-CROSSING.** Monotonicity across three `weight_decay` values is **robust to a badly-estimated σ**; a single threshold comparison is not. **The alternative explanations — nondeterminism, seed, and run-to-run drift — do not produce monotone dose-response.** ⛔ **This is the strongest evidence a five-cell design can buy here, and it is why §8 recommends re-spending arm 5.**

---

## 5. ⛔ Readings — THREE-WAY, not binary, and disjoint

Let `S` = the pooled within-pair spread in `b` (§4(i)), and `Δ` = the mean `b` at `wd = 0.01` minus the mean `b` at `wd = 0.1`. **Evaluate in order; first match is the reading.**

| # | condition | consequence |
|---|---|---|
| **W-0** | any arm VOID under F1–F5 | ⛔ **the cell is VOID.** With 5 cells and no spares, **one void control collapses the variance estimate** and the set is INCONCLUSIVE |
| **W-1** | `Δ < −2S` (lower decay → **lower** `b`, i.e. toward honest) | ⛔ **`weight_decay` is implicated as a cause.** Evidence about the CAUSE only — **never a ship decision** (§6) |
| **W-2** | `\|Δ\| ≤ 2S` | ⛔ **INCONCLUSIVE at this n** — not a null. ⚠️ **With `S` known to ±76%, this is the EXPECTED cell unless the effect is large.** Report `Δ`, `S`, and the implied minimum detectable effect |
| **W-3** | `Δ > +2S` (lower decay → **higher** `b`) | ⛔ **wrong direction — a finding.** `weight_decay` is not the cause and the mechanism argued in Phase 1 is refuted |
| **W-4** | W-2 fires **and** no arm's `b` falls below ~1.10 | ⛔ **evidence for H3 / H4 — the resolution limit or the geometry mismatch — and the post-hoc patch (R1) is the CEILING.** A real outcome, not a washout (their point 4, adopted) |

⛔ **`2S` is a stated convention, not a significance level**, and §4 is why: with 2 df there is no honest `p`. **The reading reports `Δ/S` and lets the reader see how thin it is.**

---

## 6. ⛔ NO ARM CAN BE DECLARED SHIPPABLE FROM THIS EXPERIMENT

Their point 3, adopted and strengthened, because it is the single most likely misreading.

**Without the forecast stage there is no `act_ratio`, no CRPS, no AP.** ⛔ **The firing-rate guardrail — the thing that caught the 37–83% false CRPS win in `OUTCOME.md §3` — is UNAVAILABLE in this design.** And `RESULTS_LEDGER.md`'s eligibility conjunction is satisfiable by exactly that artefact (`RESULTS.md` §9 / M84).

> ⛔ **A slope improvement here is evidence about the CAUSE. It is not, and cannot be, evidence that a model is better to serve.** Any arm that improves the slope must be re-run through the forecast stage with the firing-rate guardrail before the word "better" is used about it.

---

## 7. Arm 1 — ⛔ MY RULING: FRESH, and for a different reason than the one offered

The open question put to this seat: fresh run (costs a pod, tests determinism) or reuse `amber_kestrel` as arm 1 (frees a pod, assumes determinism)?

**Ruling: FRESH** — and ✅ **Simon has since ruled the same, directly: _"I agree with the fresh run, don't reuse amber."_** (relayed by the `views-models` seat; recorded as a relay, and it matches the independent ruling below rather than replacing it).

⚠️ **But their argument for it is not the reason, and I think it is wrong.**

**Their argument:** the determinism assumption carries the whole variance estimate. ⛔ **It does not.** What arms 1 and 2 measure is **total run-to-run variance at fixed config** — and that is the right quantity *whichever mechanism produces it*. If training is deterministic, the spread is seed variance; if not, it is seed plus nondeterminism. **Both are "run-to-run variance at fixed config", which is exactly what §5 compares `Δ` against.** The decomposition is interesting and not load-bearing.

**The reason fresh is nonetheless required:**

⛔ **`amber_kestrel` was trained on a different pod session at a different time under a different code sha.** Using it as one cell of a five-cell controlled experiment imports **every difference between then and now** — library version, environment, pod image — into the variance estimate. **And the variance estimate is the thing the entire design rests on.**

> **A variance estimate must come from two runs in the same environment, or it is not a variance estimate for this experiment.**

⚠️ **This is the provenance-mismatch class again, and it would have been the thirteenth instance:** `amber_kestrel`'s setter question was *"deliver the control arm for the pos_weight experiment"*; using it here asks *"what is run-to-run variance in this arm set"*. **Different question ⇒ re-derive, not reuse.**

✅ **And the determinism test becomes a free by-product rather than the justification.** Arm 1 should reproduce amber's weights `sha256 2a5efc2a67c08eac…`; **both readings are pre-registered** (their point 2):

- **reproduces** ⇒ training is deterministic across two weeks and a code change — **a stronger result than the experiment needed**, and arm 2's difference is then purely the seed.
- ⛔ **does not reproduce** ⇒ training is non-deterministic at fixed seed, **which alone could explain the amber/copper 0.17 gap**; Phase 1's *"they differ in one config key"* framing is undermined, and **`RESULTS.md` §1 and `OUTCOME.md` need qualifying.** ⚠️ **The variance estimate remains valid either way** — this is why the determinism question is a by-product rather than the justification.

⛔ **And the other seat's framing of the non-reproducing branch is stronger than mine and is adopted: it would be the most consequential outcome in the set.** If training is non-deterministic at fixed seed, then **the amber/copper 0.17 slope gap may never have been the config key at all** — and `RESULTS.md` §1, `OUTCOME.md` §2 and every "they differ in one config key" sentence in this programme need re-scoping. **Not a washout. The biggest result available from these five pods.**

---

## 8. ⛔ ARM 5 — NOT AUTHORISED BY THIS PLAN, AND I WILL NOT MAKE THE CODE CHANGE ON A RELAY

Arm 5 needs a **views-hydranet library change**: `dec_conv4_head{1,2,3}_class` added to a `weight_decay: 0.0` parameter group. The mechanism exists — `training_engine.py:113,136` already create such groups for `sigma_params` and `log_var_params` — and the class heads were simply never added.

⛔ **Three reasons it is held:**

1. **It is a production code change, not a config edit**, and it arrived via a **peer relay of Simon's words.** ⚠️ **A peer cannot carry authorisation for a change to this repository's training path, and I am not treating it as one.** **Simon's direct word is required.**
2. **It moves the code sha**, so the config-identity check must be told to accommodate it **deliberately** rather than fire — which means the change has to be made, reviewed and pinned *before* the pods, not alongside them.
3. ⛔ **It is not a controlled cell.** It varies *which parameters are decayed* while arms 1–4 vary *how much*. **A different axis, in a single exploratory cell, with a code change attached — three confounds in one slot.**

### ⛔ RECOMMENDATION TO SIMON: re-spend arm 5 as a third `weight_decay` value

**Drop arm 5; run `weight_decay: 0.001`, seed 42, as arm 5′.** Then the set is `0.1 / 0.1 / 0.01 / 0.01 / 0.001` — **three points on one axis.**

**Why this is better, and it is §4's fix rather than a preference:**

- ⛔ **It buys DOSE-RESPONSE, which is robust to the badly-estimated σ that is this design's central weakness.** Monotone `b` across three decay values is evidence the two-point comparison cannot give at any n=2.
- **No code change, no sha move, no new axis.** Config-only, so F6-style identity stays checkable.
- **The head-exemption question survives as a separate, better-scoped experiment** with its own review — which is what a production training-path change deserves.

⚠️ **This is a recommendation, not a change: Simon authorised a specific arm set and altering it is his call, not this seat's.** The plan is written for **both** variants; §5's readings are unchanged except that W-1/W-3 gain a monotonicity clause under the 5′ variant.

---

## 9. Falsifiers

| | fires if | then |
|---|---|---|
| **F1** | an arm's config differs from `amber_kestrel` anywhere but its named key and seed | ⛔ **VOID that arm.** The full diff is printed in §3 before the pods start |
| **F2** | training diverges, NaNs, or the C-184 / C-119 stability gate trips | ⛔ **VOID, not negative** (C-307) |
| **F3** | the local slope harness fails to reproduce amber's `b = 1.1875` bin-free to three decimals on amber's own weights | ⛔ **VOID everything.** The paired-baseline precondition E6f met exactly |
| **F4** | the sign of `Δ` changes across the three bin resolutions, or bin-free and binned disagree in sign | ⛔ **VOID** by the sweep rule |
| **F5** | an arm's artifact arrives without all three files + `PROVENANCE` with `lessons ≥ 300` and `mode: production` | ⛔ **VOID** |
| **F6** | ⛔ **any arm is described as better to serve, or any slope result is read as a ship decision** | ⛔ **a REASONING error, not a result** (§6). Retract and rewrite |
| **F7** | arm 1's wall time exceeds **2×** the ~2 h 15 m estimate | ⛔ **STOP and report** before the rest of the set |
| **F8** | any cost, power or sufficiency figure appears in the readout without a measurement behind it | ⛔ **VOID that figure.** Eleven of eleven optimistic |
| **F9** | the readout quotes `Δ` without `S` and `Δ/S` beside it | ⛔ **VOID the sentence.** §4 is why |

---

## 10. Method — LOCKED

- **Train only.** `total_lessons: 300`, no forecast stage, no origins. ⛔ **`to_cube_samples`/#402 explicitly not pulled in.**
- **Slope measured locally, after the fact**, by the Phase-1 harness: `BASE_ORIGIN = 456`, `N_ORIGINS = 13`, `LEADIN_START = 421`, `D = 4`, `set_locked_dropout(True)`, CPU, target `sb`. ⛔ **Bin-free maximum likelihood is the estimator**; binned fits at the locked P2 edges reported alongside, never instead.
- **Every artifact verified** by `sha256` against its `.pt.sha256` and its sidecar before use. ⚠️ `vehicle.py` hash-denies four known africa artifacts and **records that region is not a read-back** (`:187-191`) — the new arms are global, and that must be **confirmed from the run's own provenance, not inferred from a date.**
- **Signed error reported alongside absolute.** **Common bin support across every compared pair.**
- ⛔ **Record per arm:** `b` (bin-free + three binned weightings), `a`, bootstrap `sd(b)`, `n`, positives, weights `sha256`, code sha, wall time, and the full config diff.
- ⛔ **No value written into a stored artifact under a name that is not true of it under the weakest reading a later reader could take** (`OUTCOME.md §9` issue 10).

---

## 11. Out of scope

- **Shipping anything** (§6). **The forecast stage** and therefore `act_ratio`/CRPS/AP.
- **The body (#530)** — Simon's sequencing. ⚠️ But a gate change moves the delivered totals; `RND_PLAN.md` §6.
- **`h ≥ 2`** — #558. Everything is h=1.
- **H1's remedy** — dead by measurement (`RESULTS.md` §6). ⛔ **Tuning the recal distribution remains banned.**
- **The Phase-2 slope survey** — withdrawn (§1). Its unidentifiable-serving-weights finding stands separately.
- ⛔ **Arm 5's library change** — held for Simon's direct word (§8).

---

## 12. Definition of done, and the stop rules

**Done** when all authorised arms have a recorded `(b, a, sd, n, positives, sha, code sha, wall time, config diff)` row or a recorded VOID; F1–F9 are evaluated; §5's reading is stated with `Δ`, `S` and `Δ/S`; and the outcome document **does not describe any arm as better to serve.**

**Stop rules.** No new questions mid-set. ⛔ **F7 stops after arm 1.** A null is INCONCLUSIVE, not negative — **W-2 is the expected cell and the plan says so in advance.** No scalar without its parameter sweep. No cost or power figure without a measurement. ⛔ **The executing seat attacks this plan before running it, and does not decide which of its own objections survive. Neither does this seat.**

---

# AMENDMENT 1 — my dose-response mitigation is wrong, their effect size is void, and the honest critical value is 4.30·S. 2026-10-18, BEFORE any pod starts.

**Source:** three attacks from the `views-models` seat. **All three upheld.** ⛔ **Attack 1 kills the mitigation this seat built in §4, and their replacement is harsher than either of us wrote.**

---

## A1 — ⛔ UPHELD. Monotonicity is an ORDERING test, so replication buys it NOTHING.

Simulated here, 200,000 draws per cell, under the null:

| n per dose | P(strictly monotone in the hoped direction) |
|---|---|
| 1 | 16.58% |
| 2 | 16.55% |
| 4 | 16.83% |
| 16 | 16.68% |

**Theory `1/6 = 16.67%`. ⛔ FLAT in n, exactly as they claim** — under the null the three cell means are exchangeable, so all `3! = 6` orderings are equiprobable **whatever the replication.**

⛔ **So §4(ii) is wrong as a test.** I proposed dose-response because it is *"robust to a badly-estimated σ"* — **true, and the cost is that it is also BLIND to σ, which is the one thing the five pods exist to produce.** It swaps a noisy threshold for a fixed 1-in-6 error rate and **forfeits the benefit of the replicates.** A one-in-six chance of a spurious monotone result is roughly the chance of publishing *"weight decay causes the under-dispersion"* **from pure run noise.**

### ✅ REPLACEMENT, adopted — and it preserves the intuition while fixing the test

> **Fit `b = c + k·log₁₀(weight_decay)` across ALL runs and test `k` against the pooled within-cell spread.** Report `k`, its interval, and the pooled `S`.

**It uses every run rather than three means, it uses the magnitudes, it has a real null distribution, and it DOES improve with n.** ⛔ **The reframe that makes it work: dose-response as a SLOPE, not as an ORDERING.** And it subsumes both of §4's mitigations — pooling is how `S` is formed, and a monotone trend is the `k ≠ 0` case with magnitude attached.

⛔ **And it is why a third dose point matters: with two points `k` is just the difference and there is no trend to test.**

### ⛔ MY ADDITION: their replacement inherits a 2-df denominator, so the critical value is 4.30·S

Five runs, with within-cell spreads available from **two** cells only (the third dose has n=1 and contributes no variance information) ⇒ **the pooled denominator has 2 df.**

```
t(0.975, df=2)  = 4.303      <- the honest two-sided critical value
t(0.975, df=3)  = 3.182
t(0.975, df=4)  = 2.776
```

⛔ **So §5's `2S` thresholds were far too lenient, and they were mine.** The honest minimum detectable effect is **~4.30·S**, not `2·S`. **§5's W-1 and W-3 are re-cut accordingly**, and this is a second, independent reason W-2 is the expected cell.

## A2 — ⛔ UPHELD. The ~0.19 effect size is VOID, and the design's power is UNKNOWN rather than low.

Their own figure, withdrawn by them: **0.19 is the size of the WHOLE DEFECT** (1.19 → 1.00). ⛔ **It assumes weight decay explains all of it.** We have **no** estimate of the weight-decay component — **H2 predicts the SIGN at convergence and nothing about the magnitude.**

**If weight decay explains a third of the defect, the effect is 0.06 — against observed run-to-run gaps of 0.17.**

⛔ **So the design's power is not low. It is UNKNOWN — and no arrangement of five pods fixes that, because the unknown is the effect size, not the variance.** Written in, because "low power" invites "add more pods" and that is the wrong response.

⚠️ **And both of their numbers in this plan are now unmeasured, one of them unjustifiable:** the ~2 h 15 m wall time (unmeasured, which this seat flagged) and the ~0.19 effect (no derivation, which they have withdrawn). **Twelve for twelve across both seats, and the twelfth is theirs.** **F7's stop remains, anchored to an unmeasured figure, and the plan now says so where the stop is defined.**

## A3 — ⛔ UPHELD. Pooling assumes equal variance, and less decay plausibly means MORE.

Pooling arms 1–2 with arms 3–4 requires σ to be equal across cells. ⛔ **Weight decay is a regulariser; removing it plausibly increases run-to-run variability as well as moving the mean.** If σ differs, the pooled estimate is wrong for both and the trend test's denominator is mis-sized.

⛔ **And with n=2 per cell the assumption cannot be tested** — two spreads differing 3× is unremarkable under equality, by §4's own 31.4× range.

**Adopted as written:** stated as a **load-bearing assumption**; **both within-pair spreads reported separately beside the pooled one**; and ⛔ **a large discrepancy VOIDS the pooled denominator rather than being averaged into it** (new, **F10**).

### ⛔ MY ADDITION: this interacts with A1's replacement in the worst place

Under A3, σ should be **largest at the lowest decay**. And the lowest-decay cell is the **third dose point** — which is **n=1** and is the **highest-leverage point on the trend axis.**

⛔ **So the point that contributes most to `k` is simultaneously the least replicated and plausibly the most variable.** That is not fixable within five pods, and it bears directly on the choice below.

---

## ⛔ THE THIRD DOSE: `0.001` vs `0.0`. My ruling, and it reverses my own recommendation.

They put the tension and declined to choose. **Both cannot fit one slot:**

```
0.1 / 0.01 / 0.001   log-spaced, a testable trend, WEAKER lever
0.1 / 0.01 / 0.0     the strongest existence test, -inf on a log axis, NO trend fit
```

**Ruling: `0.0`.** ⛔ **This reverses the `0.001` I recommended in §8, and A2 is why.**

**The reasoning:** a trend test needs an effect size to be powered, and ⛔ **A2 establishes that we do not have one.** An **existence** test does not. With an unknown effect size, **the maximal lever maximises the chance of observing anything at all** — and the two nulls are not equally informative:

- **a null at `0.0`** ⇒ removing decay **entirely** does not move the slope ⇒ ⛔ **H2 is dead**, cleanly.
- **a null at `0.001`** ⇒ we have learned **almost nothing**; the effect may simply need full removal.

> ⛔ **With an unknown effect size, prefer the maximal lever over the parameterised trend.** The trend cannot be powered against an effect we cannot size; the existence test only needs the effect to exist.

✅ **And `weight_decay: 0.0` is config-only** — no library change, no code sha move, unlike arm 5's head exemption. **It is the cheap version of the strong test.**

### ⚠️ And a free parameter neither of us had named: the trend AXIS

`log₁₀` is not derived. **The effect of weight decay on a calibration slope has no known functional form**, so linear is equally undefended — and ⛔ **choosing the axis that admits your third point is choosing a parameterisation to fit the data.**

**Pre-registered:** if a trend is fitted at all, ⛔ **the axis is swept (log and linear) and `k`'s SIGN must agree across both**, per §3.5's rule. **If it does not, the trend is a parameterisation artefact and is void.** Under the `0.0` ruling the trend fit degrades to the two replicated points plus a separate existence arm, **which is stated rather than papered over.**

---

## §5 re-cut, and §4 corrected

| # | condition | consequence |
|---|---|---|
| **W-0** | any arm VOID under F1–F5, or **F10** fires | ⛔ **INCONCLUSIVE.** One void control collapses the variance estimate |
| **W-1** | `Δ < −4.30·S` | ⛔ **`weight_decay` implicated as a cause.** Evidence about the CAUSE only — never a ship decision (§6) |
| **W-2** | `\|Δ\| ≤ 4.30·S` | ⛔ **INCONCLUSIVE at this n — and the EXPECTED cell, for two stated reasons: a 2-df denominator and an unknown effect size.** Report `Δ`, both spreads, `S`, `Δ/S` and the implied minimum detectable effect |
| **W-3** | `Δ > +4.30·S` | ⛔ **wrong direction — a finding.** H2's mechanism refuted |
| **W-4** | W-2 fires **and** no arm's `b` falls below ~1.10 | ⛔ **evidence for H3 / H4, and R1 is the CEILING.** A real outcome |

⛔ **§4(ii)'s dose-response mitigation is WITHDRAWN as a test** and retained only as the *motivation* for a third dose point, now expressed as A1's slope test where a third point is fitted at all.

**New falsifier:**

| | fires if | then |
|---|---|---|
| **F10** | the two within-pair spreads differ by more than **3×** | ⛔ **the pooled denominator is VOID.** Report both separately; do not average. A3's equal-variance assumption has failed and the trend test has no valid denominator |

---

⛔ **Two of the three things this seat contributed to §4 are now corrected by the other seat: the `2S` threshold (too lenient by 2.15×) and the dose-response mitigation (a fixed 1-in-6 error rate).** The derivation they verified — the 31.4× range on a 2-run spread — stands, and it is what made both errors visible. **Nothing is running and no model directory exists.**

---

# AMENDMENT 2 — `S` was ambiguous in my own text, arm 5 needs `5.27·S`, and arms 1–2 retroactively test Phase 1. 2026-10-18.

**Source:** the `views-models` seat, following Amendment 1's `t`-correction one step further. **Their two numbers verified. ⛔ And doing so exposed a unit ambiguity in my own amendment and a consequence larger than the weight-decay question.**

---

## 1 — ⛔ `S` WAS AMBIGUOUS IN AMENDMENT 1, AND THE TWO READINGS DIFFER BY √2

Amendment 1 wrote the thresholds as multiples of *"the pooled within-cell spread `S`"*. ⛔ **"Spread" and "sd estimate" are not the same quantity**, and for a pair they differ by exactly `√2`, since the unbiased sd estimate from two points is `σ̂ = |x₁−x₂|/√2`:

```
if S ≡ σ̂ (pooled SD)        W-1/W-3 crit = 4.303·S     arm 5 = 5.270·S
if S ≡ |x₁−x₂| (the spread)  W-1/W-3 crit = 3.042·S     arm 5 = 3.726·S
```

⛔ **Their 4.30 and 5.27 are correct ONLY under the first reading — and §4 of this plan was written about the SPREAD** (the 31.4× range is a property of `|X₁−X₂|`). **So my own amendment mixed the two.**

> ⛔ **LOCKED: `S ≡ σ̂`, the pooled within-cell SD estimate, with `σ̂ = |x₁−x₂|/√2` for a pair.** Every threshold in this plan is a multiple of **that**, and the readout must print `σ̂`, the raw spreads, and the conversion.

⚠️ **This is the provenance-mismatch class in my own text, one amendment after I wrote the rule into the plan** — a value whose meaning was assumed rather than stated. **Caught by following their arithmetic, not by re-reading mine.**

## 2 — ⛔ UPHELD. Arm 5 is n=1 against a mean of n=2, so its threshold is `5.27·S`

`4.30·S` is the critical value for a **difference of two means**. Arm 5 is a **single run** against the control pair's mean:

```
Var(x − ȳ) = σ²(1 + 1/2) = 1.5σ²   ⇒   crit = 4.303·√1.5 = 5.270·S
```

**23% harsher, and it applies to the arm this seat just made the centrepiece.** Verified. **W-1/W-3 keep `4.30·S`** (both cells n=2); ⛔ **arm 5's existence test takes `5.27·S`.**

## 3 — ⛔ UPHELD AND IT IS THE HEADLINE: `S` must be under 0.0356 or the design cannot detect a COMPLETE fix

The whole defect is `1.1875 − 1.0 = 0.1875`. For arm 5 to detect **all** of it:

```
5.27·S < 0.1875   ⇒   ⛔ S < 0.0356
```

**Where `S` plausibly sits:**

| | value | status |
|---|---|---|
| sampling-only `sd(b)` at full population | **0.0107** | ⛔ **MEASURED — a hard FLOOR** |
| implied by the amber/copper gap of 0.17 if it were all run-to-run | **≤ ~0.120** | an UPPER bound (`0.17/√2`) |
| ⛔ **the feasibility boundary** | **0.0356** | **sits INSIDE that range** |

```
S = 0.02  ->  detectable 0.105   WORKS        (defect 0.1875)
S = 0.04  ->  detectable 0.211   MARGINAL — cannot detect a complete fix
S = 0.08  ->  detectable 0.422   HOPELESS
```

⛔ **The design is genuinely marginal and nobody knows which side of the line it falls on.** Not a reason to cancel it — **a reason to say so before it runs**, which gives W-2's expected-cell status a **third** independent reason and now a **quantitative boundary.**

## 4 — ✅ ADOPTED: arms 1–2 are the FEASIBILITY TEST, and `S` is a primary deliverable

Their reframe, and it is the right one. **The five pods are not interchangeable: arms 1–2 produce `S`, and `S` decides whether arms 3–5 can answer anything.**

> ⛔ **`S` is pre-registered as a primary deliverable in its own right, not as a denominator.**
> - **`S < 0.036`** ⇒ the design can detect a complete fix; read the arms as planned.
> - **`S > 0.036`** ⇒ ⛔ **the arms cannot resolve even a total effect. The correct response is MORE REPLICATES PER CELL, NOT MORE DOSES.** Report `S`, report the arms as uninformative, and let `S` size the next experiment.

✅ **And it is a real finding either way — the first measurement of run-to-run variance in this training pipeline, which nothing in the record has.** Deliverable even if every slope comes back null.

✅ **It also answers Amendment 1's own objection** that *"low power invites add-more-pods"*: **with `S` measured, "more pods" stops being a guess and becomes arithmetic.** Which this seat has now done:

### ⛔ MY ADDITION: the replicate table, so the next experiment is pre-sized

For the 2-cell comparison at equal `n`, detectable effect `= t(0.975, 2(n−1))·√(2/n)·σ̂`:

| n per cell | df | detectable | needs `σ̂ <` |
|---|---|---|---|
| **2** | 2 | **4.303·σ̂** | **0.0436** |
| **3** | 4 | **2.267·σ̂** | **0.0827** |
| 4 | 6 | 1.730·σ̂ | 0.1084 |
| 6 | 10 | 1.286·σ̂ | 0.1458 |
| 8 | 14 | 1.072·σ̂ | 0.1748 |

⛔ **Going from n=2 to n=3 nearly DOUBLES the detectable range** — `0.0436 → 0.0827` — because `df` goes 2→4 and `t` collapses 4.303→2.267. **That is the single highest-leverage change available to this design, and it costs two pods.** Recorded so that if `S > 0.036`, the next proposal is **n=3 on two cells (6 pods)** rather than more doses — and the number is already derived.

## 5 — ⛔ AND THE CONSEQUENCE LARGER THAN THE WEIGHT-DECAY QUESTION: arms 1–2 retroactively test whether PHASE 1 was ever readable

`amber` and `copper` differ by **0.17** in `b`, at **n=1 each**. That difference has **no** error bar anywhere in the record. With `S` measured, it acquires one:

```
se(amber − copper) = σ̂·√2      informal 2-sigma needs σ̂ < 0.17/(2√2)
⇒ ⛔ Phase 1's comparison was readable only if σ̂ < 0.0601
```

⛔ **So if `S > 0.06`, the `pos_weight` comparison was never resolvable at n=1 per arm** — and `RESULTS.md` §1, `OUTCOME.md` §2 and every *"they differ in one config key"* sentence in this programme need re-scoping. **Pre-registered as a reading:**

| | condition | consequence |
|---|---|---|
| **W-5** | `S > 0.0601` | ⛔ **PHASE 1's two-arm comparison was never resolvable.** The under-dispersion *finding* survives — it is present in **both** arms and in six fits — but the **amber-vs-copper DIFFERENCE** (0.17, and the R1-works/R1-fails asymmetry built on it) is not separable from run-to-run noise. `OUTCOME.md` and `RESULTS.md` require an amendment, and it is **not** optional |

⚠️ **This is independent of the determinism branch in §7.** That one asks whether a fixed seed reproduces; **this one asks whether the observed gap exceeds the noise, whatever its mechanism.** ⛔ **Arms 1–2 therefore audit the previous phase as a by-product of enabling this one** — and that is the strongest reason to run them even if arms 3–5 return nothing.

---

## 6 — Tidy-up: the trend axis is NOT IN FORCE

✅ Their catch: with the third dose at `0.0`, **`log(0)` is undefined, so there is no log axis and the trend fit is off the table.** Amendment 1's log-vs-linear sweep requirement is therefore **moot for this design.**

> ⚠️ **It is retained in this document as the rule that WOULD apply if `0.001` were ever chosen, and explicitly marked NOT IN FORCE** — so the plan does not carry a dead requirement a later reader tries to satisfy.

⛔ **The design is now three pairwise existence tests, not a trend**, and §5's readings are the pairwise ones.

---

## State

⛔ **Both of the other seat's substantive numbers in this plan are withdrawn (`0.19`) or superseded (the arm-1 rationale); this seat has now had five corrected** — the `2S` band, the dose-response mitigation, the `4711` seed, the `0.001` recommendation, and now **the ambiguous `S`.**

⛔ **And this round's net effect is to make the design look WORSE than when Simon authorised it.** He should know that before any pod starts, and both seats are telling him. **Nothing is running and no model directory exists.**

---

# AMENDMENT 3 — the dominating allocation, and ⛔ the planned set's headline threshold is contingent on an assumption it cannot test. 2026-10-18.

**Source:** the `views-models` seat applying this seat's own n-table to the pods in hand rather than to a future experiment. ⛔ **Their arithmetic verified exactly. And verifying it exposed something that makes their case stronger than they argued.**

---

## 1 — ✅ THEIR ALLOCATION, VERIFIED

Same five pods, detecting the whole defect `0.1875`:

| allocation | df | crit | max `σ̂` |
|---|---|---|---|
| planned — `2×0.1` vs `2×0.01` | 2 | 4.303·S | 0.0436 |
| planned — arm 5, `1×0.0` vs the `0.1` mean | 2 | 5.270·S | 0.0356 |
| ⛔ **theirs — `3×0.1` vs `2×0.0`** | **3** | **2.905·S** | **0.0645** |
| alt — `2×0.1` vs `3×0.0` | 3 | 2.905·S | 0.0645 |
| alt — `4×0.1` vs `1×0.0` | 3 | 3.558·S | 0.0527 |
| alt — `3×0.1` vs `1×0.0` (+`1×0.01`) | 2 | 4.968·S | 0.0377 |

**+48% on the two-cell comparison, +81% on the arm-5 existence test, for the same five pods** — `df` 2→3 collapses `t` 4.303→3.182, and the unbalanced `√(1/3+1/2)` is cheaper than `√(1+1/2)`.

⛔ **And it is this seat's own n=3 argument applied to the allocation rather than to a hypothetical follow-up. The df gain was available inside the five pods the whole time and I did not look.** Their catch.

⚠️ `2×0.1 + 3×0.0` is **statistically identical** (0.0645) — so their orientation, putting the **third run on the safe cell**, is the right one and is not arbitrary.

## 2 — ⛔ WHAT I FOUND VERIFYING IT: the planned arm-5 threshold is CONTINGENT, and collapses below the sampling floor

`5.270·S` for arm 5 assumes `S` is **pooled across the `0.1` and `0.01` cells** — because arm 5's own cell (`0.0`, n=1) contributes **no** variance information. Computed from the comparison alone:

```
arm 5 vs the 0.1 pair, sigma_hat from that pair only:  df = 1
  t(0.975,1) = 12.706   crit = 12.706 x sqrt(1.5) = 15.56 * S
  ⇒ max sigma_hat = 0.0120
```

⛔ **So if F10 fires — the two within-pair spreads differing by more than 3×, which A3 says is unremarkable at n=2 — arm 5's threshold collapses from `5.27·S` to `15.56·S`, and its tolerable `σ̂` falls from 0.0356 to 0.0120.**

⛔ **0.0120 is BELOW the measured sampling-only floor of 0.0107.** **Arm 5 then becomes undetectable in principle, not merely underpowered.**

### ⛔ And the assumption cannot be checked for the planned arm 5, even in principle

A3 established that pooling is a **load-bearing, untestable** assumption at n=2. **It is worse than that for the planned set:**

- **Planned:** arm 5's `σ̂` is borrowed from two cells that **do not include its own.** The `0.0` cell has **n=1**, so its variance **cannot be estimated at all** — ⛔ **there is nothing for F10 to check it against.** The headline 0.0356 rests on equal variance between `0.0` and cells it was never compared to, and **weight decay is a regulariser, so `0.0` is precisely where σ is most likely to differ.**
- **Theirs:** `σ̂` is pooled across **exactly the two cells being compared, both replicated.** The assumption is still made — but ⛔ **F10 can actually test it**, because both cells produce a spread.

> ⛔ **The planned allocation's headline number depends on an assumption that its own design makes unfalsifiable, in the cell where it is least likely to hold. Theirs does not.**

**That is the strongest argument for their allocation and neither of us had it.** It is not about power; it is about whether the stated threshold means anything.

## 3 — Their counter-argument, quantified — and it is weaker than they stated

Their honest objection: the allocation **concentrates all intervention risk on `weight_decay: 0.0` being trainable at all**, with no intermediate dose to fall back on. **The risk is real** — `0.0` is no regularisation whatever and nothing in the record says this architecture trains stably there.

**But the failure modes are not as asymmetric as that implies:**

| | if `0.0` will not train |
|---|---|
| **planned** (1 run at `0.0`) | existence test **GONE**; the `0.01` comparison survives at max `σ̂` 0.0436; `σ̂` from 2 pairs = **2 df** |
| **theirs** (2 runs at `0.0`) | **one** voids → 3 vs 1 survives **degraded but alive** (max `σ̂` 0.0377); **both** void → no intervention, but `σ̂` from n=3 = **2 df — the SAME as planned** |

⛔ **So total failure of their allocation lands on exactly the `σ̂` quality the planned set would have delivered anyway** — and `σ̂` is the primary deliverable and the Phase-1 audit. **The downside is bounded below by the planned set's baseline.**

⛔ **And the planned set does NOT hedge the trainability risk.** It has **one** zero-decay run; if that voids, the maximal lever is gone too — with **less** replication on the risky cell than theirs. **What the planned set genuinely retains is a DIFFERENT, graded lever (`0.01`) — not a fallback for the same one.** That is a real thing to want and it is the honest form of their objection.

## 4 — ⛔ MY RECOMMENDATION, with the risk left where it belongs

They offered no recommendation. **I will give one, because the evidence is now one-sided on everything except the trainability question:**

> ⛔ **Recommend `3×0.1 + 2×0.0`**, on four grounds: **+48%/+81%** on the only quantity that decides readability; **3 df on `S`**, which is a primary deliverable; a **sharper Phase-1 audit**, which is deliverable whatever the arms do; and ⛔ **it does not stake its headline on an untestable cross-cell variance assumption**, which the planned arm 5 does and which collapses it below the sampling floor if F10 fires.

⚠️ **The trainability risk is Simon's to price and I cannot price it** — nothing in the record says `wd=0.0` trains on this architecture, and I will not guess. **If he prefers the graded `0.01` cell as insurance against learning nothing at all, that is a coherent choice and the plan supports it.**

**Pre-registered either way:**

| | fires if | then |
|---|---|---|
| **F11** | a zero-decay run diverges, NaNs, or trips the C-184 / C-119 stability gate | ⛔ **VOID for that arm** (C-307, **not negative**). **The readout must state how many zero-decay runs survived BEFORE any slope is read**, and a single-survivor existence test is reported at **4.968·S**, not 2.905·S |

## 5 — ✅ On the Phase-1 audit

Their endorsement is adopted, and ⛔ **they are right that it gets BETTER under their allocation — 3 df on `σ̂` instead of 2** — which sharpens the `0.0601` boundary that decides whether the amber/copper comparison was ever resolvable. **A second reason to prefer it, and the one that survives whatever the arms do.**

⛔ **W-5 stands: if `S > 0.0601`, the `pos_weight` difference was never separable from run noise, the R1-works/R1-fails asymmetry goes with it, and the amendment to `OUTCOME.md`/`RESULTS.md` is not optional.**

---

⛔ **Six of this seat's items corrected in this document; two of the other seat's withdrawn.** Neither seat has defended a position once, and the allocation that now looks best is one neither of us proposed when Simon authorised the set. **Nothing is running and no model directory exists.**
