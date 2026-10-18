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

⛔ **CORRECTED 2026-10-18 (Amendment 9). THIS IS THE APPROVED SET AND THE ONLY ONE. F1's authority is this table.**

| name | `weight_decay` | seed | role |
|---|---|---|---|
| **`silver_kestrel`** | `0.1` (current) | **42** | control replicate |
| **`bronze_kestrel`** | `0.1` (current) | **43** | control replicate |
| **`cobalt_kestrel`** | `0.1` (current) | **44** | control replicate |
| **`indigo_kestrel`** | **`0.0`** | **42** | intervention — paired to `silver` (same seed) |
| **`russet_kestrel`** | **`0.0`** | **43** | intervention — paired to `bronze` (same seed) |

**The three replicates deliver `σ̂` (M1).** The two interventions are each a **one-key contrast** against their same-seed control, which is what makes them declarable under `test_roster_conformance`'s arm guard — declaring them against `amber` would make them three-key contrasts, which `:66-70` names as confounded.

<details><summary>⛔ SUPERSEDED — the originally authorised set, struck in place, never run</summary>

| arm | name | `weight_decay` | seed | role |
|---|---|---|---|---|
| ~~1~~ | ~~`silver_kestrel`~~ | ~~`0.1`~~ | ~~42~~ | ~~control · the determinism precondition~~ |
| ~~2~~ | ~~`bronze_kestrel`~~ | ~~`0.1`~~ | ~~43~~ | ~~run-to-run variance~~ |
| ~~3~~ | ~~`cobalt_kestrel`~~ | ~~`0.01`~~ | ~~42~~ | ~~intervention~~ |
| ~~4~~ | ~~`indigo_kestrel`~~ | ~~`0.01`~~ | ~~43~~ | ~~intervention variance~~ |
| ~~5~~ | ~~`russet_kestrel`~~ | ~~`0.1`, heads exempt~~ | ~~42~~ | ~~exploratory, not authorised~~ |

**Superseded by Amendments 3 (allocation), 4 (third dose = `0.0`) and 5 §2 (seeds 42/43/44).** ⛔ **Three names map to different configs here than in the approved table above** — `cobalt`, `indigo` and `russet`. **This is why the trail is kept rather than deleted: a reader who finds an old message must be able to see which mapping is live.** The determinism precondition was withdrawn in Amendment 6; the head-exemption arm was never authorised and is replaced by `0.0`.

</details>

**Names are the `views-models` seat's call** (their CLAUDE.md). **The kestrel family is deliberate**, marking the lineage: `amber` (shipped baseline) → `copper` (`pos_weight` arm) → these five (weight-decay arms of the same line).

✅ **And the names deliberately encode NOTHING about the configs.** Their reasoning, which this seat endorses and would have raised: a name like `wd001_kestrel` **asserts a value that can drift from the config and then lie** — ⛔ **the provenance-mismatch class applied to a directory name.** **The config is the only place the config is stated.**

**Seeds named by value** (their point 5). ⚠️ **This seat proposed `4711` and was wrong** — that is `capture_regimes.py`'s own **anchor-RNG** default, not a model seed, which is the provenance-mismatch class a fourteenth time and caught before it reached a config. **`43` is the repository's established pair for `42`**, verified: 4 roster configs at `np_seed`/`torch_seed` 42 and 2 at 43, and the paired-run dossier names its own runs `fortytwo`/`fortythree`. **Their value stands.**

⛔ **The replicates are the point of the design.** amber and copper differ in slope by **0.17** from one config key at n=1 each. ⚠️ **The "~0.19 expected effect" that stood here is WITHDRAWN** (Amendment 2 A2: it was the size of the *whole* defect and assumed `weight_decay` explains all of it). **We have no estimate of the effect's magnitude, which is why `σ̂` is the primary endpoint and not a denominator.** Without the replicates no result is interpretable at any price.

**The full config diff of each model against `amber_kestrel` must be verified before the pods start**, so identity is checkable rather than asserted. ⚠️ **It cannot be printed here until the five directories exist** (views-models #567/#568), so this is **not an unmet precondition of the plan** — it is **views-models #569**, owned by this seat, and its method is fixed in `2026-10-18_weight_decay_arms_launch_plan.md` §1.4 plus Amendment 8's F13: **six-file byte identity (`config_queryset.py` first), then `CORE_GENOME` per pair.**

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

**(i) POOL the within-cell spreads.** ⛔ **Under the approved set this is superseded by Amendment 3's allocation:** the three `wd = 0.1` replicates give `σ̂` on **2 df** directly, and the two `wd = 0.0` runs add a third, for **3 df** pooled across the compared cells. **Better than the 2 df this paragraph was written for.**

**(ii) ⛔ PREFER DOSE-RESPONSE OVER THRESHOLD-CROSSING.** Monotonicity across three `weight_decay` values is **robust to a badly-estimated σ**; a single threshold comparison is not. **The alternative explanations — nondeterminism, seed, and run-to-run drift — do not produce monotone dose-response.** ⛔ **This is the strongest evidence a five-cell design can buy here, and it is why §8 recommends re-spending arm 5.**

---

## 5. ⛔ Readings — THREE-WAY, not binary, and disjoint

Let `S ≡ σ̂` = the pooled within-cell **sd estimate** in `b` (Amendment 2 §1 locks the units), and `Δ` = the mean `b` at **`wd = 0.0`** minus the mean `b` at `wd = 0.1`. ⛔ **`0.0`, not `0.01`** — the third dose was ruled to `0.0` in Amendment 4. **Evaluate in order; first match is the reading.**

| # | condition | consequence |
|---|---|---|
| **W-0** | any model VOID under F1–F5 | ⛔ **the cell is VOID.** ✅ **CORRECTED (Amendment 10): the approved allocation bought a SPARE on the control side.** With **three** control replicates, one void leaves two and `σ̂` survives at df=1. **Two void controls, or both interventions void, collapses it.** The original wording assumed 5 cells with no spares and was more pessimistic than the design |
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
| **F1** | a model's config differs from `amber_kestrel` anywhere but the key and seed named in **§3's APPROVED table** (as corrected by Amendment 9) | ⛔ **VOID that model.** ⚠️ **F1's authority is §3's approved table only — never the superseded one kept beneath it.** The diff is verified by views-models #569 before the merge, not printed here |
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
| **W-1** | `Δ < −C` | ⛔ **`weight_decay` implicated as a cause.** Evidence about the CAUSE only — never a ship decision (§6) |
| **W-2** | `\|Δ\| ≤ C` | ⛔ **INCONCLUSIVE at this n — and the EXPECTED cell**, for an unknown effect size. Report `Δ`, every within-cell spread, `S`, `Δ/S` and the implied minimum detectable effect |
| **W-3** | `Δ > +C` | ⛔ **wrong direction — a finding.** H2's mechanism refuted |

⛔ **`C` CORRECTED (Amendment 10). The `4.30·S` written here was derived for two n=2 cells and the approved set is n=3 vs n=2:** `C = 2.905·S` **pooled (df=3)**, or `C = 3.928·S` **conservative (σ̂ from the three controls only, df=2) when F10 fires.** ⚠️ **And W-2's "2-df denominator" is stale — it is 3 df pooled.**
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

**23% harsher, and it applies to the arm this seat just made the centrepiece.** Verified **for the allocation this section was written against.**

⛔ **VOID BY CONSTRUCTION (Amendment 10). The approved set has NO n=1 cell**, so there is no separate "arm 5 existence test" to carry a harsher threshold. **W-1/W-3 and the existence test are the SAME comparison and take the same value** — `2.905·S` pooled, `3.928·S` conservative. **This section is retained as the derivation that produced the 1.5-factor; its application to the approved design is withdrawn.**

## 3 — ⛔ UPHELD AS ARITHMETIC — ⚠️ **but the BOUNDARY IS CORRECTED to 0.0645 / 0.0477 (Amendment 10).** `0.0356` was derived for n=1 vs n=2 and the approved set is n=3 vs n=2

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

⛔ **CORRECTED (Amendment 4): `0.0120` is ABOVE the measured floor of `0.0107`, by +12.6% — the original sentence had the inequality reversed, in the line carrying this plan's recommendation.** The correct statement loses nothing and is stronger:

> ⛔ **`σ̂` would have to land in `[0.0107, 0.0120]` — a 12.6% window sitting directly ON a floor that cannot be gone below.** Training stochasticity would have to contribute **essentially nothing** beyond pure sampling noise. Given that amber and copper differ by **0.17**, that is not a narrow requirement — **it is close to requiring that run-to-run variance does not exist.**

⚠️ *"Below the floor"* invited the reply *"then the floor is wrong"*; **a 12.6% window on top of the floor is a statement about the DESIGN, not about the floor.** The other seat's reframing, adopted.

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

---

# AMENDMENT 4 — a reversed inequality in the sentence carrying the recommendation, and both seats now agree on the allocation. 2026-10-18.

## 1 — ⛔ MY ERROR, in the load-bearing line

Amendment 3 wrote: *"0.0120 is BELOW the measured sampling-only floor of 0.0107."* ⛔ **`0.0120 > 0.0107`, by +12.6%. The inequality was reversed** — and it was the sentence the recommendation rested on. **Corrected in place in §2 of Amendment 3, with the error stated rather than silently edited.**

**The conclusion survives and is stronger in their form:** `σ̂` would have to land in a **12.6% window sitting ON a floor that cannot be gone below**, i.e. training stochasticity contributing **essentially nothing** beyond sampling noise — against an observed amber/copper gap of **0.17**. ⚠️ **And "below the floor" invited "then the floor is wrong", where "a 12.6% window on the floor" is a claim about the design.** Theirs is the better sentence.

### ⛔ The pattern, which is worth more than the instance

**Both of this seat's arithmetic errors in this document are in the threshold sentences that carry recommendations:**

| | error | where |
|---|---|---|
| Amendment 2 | `S` stated in ambiguous units — **spread vs sd, off by √2** | the line defining every threshold |
| Amendment 3 | **a reversed inequality** | the line carrying the recommendation |

⛔ **Not random slips: both are in the numbers doing the persuading.** Logged as **W-28**, with the control that follows: **any inequality or unit in a sentence that carries a recommendation gets evaluated numerically before the sentence is committed, not read back.** Both of these were caught by the other seat following the arithmetic, and neither by re-reading mine.

## 2 — ✅ BOTH SEATS NOW AGREE ON THE ALLOCATION

They have joined the recommendation: ⛔ **`3×0.1 + 2×0.0`** — on power (+48%/+81%), on **3 df for `S`**, on the **sharper Phase-1 audit**, and above all on ⛔ **not staking the headline on an untestable assumption.**

**And they verified, independently, the failure-mode table that weakened their own objection:**

```
both 0.0 runs survive   df=3   2.905·S   max σ̂ 0.0645
one voids               df=2   4.969·S   max σ̂ 0.0377   degraded but ALIVE
both void               σ̂ on n=3 = 2 df  ⛔ IDENTICAL to the planned set's delivery
```

✅ **F11 confirmed to rounding (4.969 vs my 4.968).** ✅ **And they adopted the orientation point** — `2×0.1 + 3×0.0` is statistically identical, so putting the third run on the **safe** cell is *"the difference between a choice and an accident."*

## 3 — ⚠️ WHAT REMAINS UNPRICED, BY BOTH SEATS

⛔ **Neither seat will price the probability that `weight_decay: 0.0` trains on this architecture.** Nothing in the record speaks to it. The other seat's framing of the choice is the honest one and is adopted verbatim:

> **Simon's choice is between the better-designed experiment and the one more likely to return something rather than nothing.**

**That is a risk judgement, not an arithmetic one, and it is his alone.**

## 4 — The state of the plan

⛔ **Seven of this seat's items corrected in this document; three of the other seat's withdrawn.** The recommended allocation is one **neither seat proposed** when the pods were authorised, and the argument that now decides it — untestability rather than power — **neither seat had two rounds ago.**

**The design has improved four times since authorisation, and every improvement came from the other seat checking the one who wrote it.**

**Nothing is running. No model directory exists. Three things await Simon's direct word: the allocation, the trainability risk, and — if he keeps the head-exemption arm instead — the library change in this repository.**

---

# AMENDMENT 5 — ⛔ the prereg tests `b` alone, and weight decay moves `a` too. A real effect could have read as a null. 2026-10-18.

**Source:** the `views-models` seat, asked for a fifth improvement and returning two. **Both upheld.** ⛔ **The first is mechanistic and would have been a design failure; our own record contains the case that proves it.**

---

## 1 — ⛔ UPHELD. The gate heads carry DECAYED BIASES, so H2 predicts movement in BOTH `a` and `b`.

**Verified in the architecture:**

```
:251  dec_conv4_head1_class = nn.Conv2d(base, output_channels, 3, padding=1)        <- no bias= arg -> bias=True
:289  dec_conv4_head2_class = nn.Conv2d(base, output_channels, kernel_size, ...)    <- bias=True
:339  dec_conv4_head3_class = nn.Conv2d(base, output_channels, kernel_size, ...)    <- bias=True

the encoder, deliberately NOT:
:196-201 enc_conv0 ... bias=False     :207 enc_conv1 bias=False     :212 bottleneck_conv bias=False
```

⛔ **So the three gate heads carry biases, the encoder explicitly does not, and AdamW decays every parameter in the group** — the only exclusions are `sigma_params` and `log_var_params` (`training_engine.py:113,136`). **The head's bias is decayed.**

```
weight shrinkage  ->  MULTIPLICATIVE on the logit  ->  the SLOPE  (b)
bias shrinkage    ->  ADDITIVE       on the logit  ->  the LEVEL  (a)
```

⛔ **H2 therefore predicts movement in both, and this plan's readings were written on `b` alone. If weight decay acts mainly through the bias, every arm returns a `b` null and we conclude H2 is dead when it is not.**

⚠️ **And their sharpest point: we measured the `a`/`b` distinction being decisive three days ago and then wrote a prereg testing one of them.** Tier 2: amber's defect is pure slope (`a = +0.16`), copper's is slope **and** level (`a = +1.30`) — **which is exactly why R1 worked on one and failed on the other.**

### ⛔ AND OUR OWN RECORD CONTAINS THE CASE THAT PROVES A `b`-ONLY DESIGN MISLEADS

**E6f.** `b` moved **1.1875 → 0.7793** — a **0.41** slope movement, twice the whole defect, straight past 1.0. ⛔ **Read on `b` alone that is a spectacular success.** What actually happened:

| | `b` | `\|b−1\|` | mean gate | max signed error |
|---|---|---|---|---|
| before | 1.1875 | 0.1875 | 1.21× truth | +0.0587 |
| after | **0.7793** | **0.2207** ⬆ | **5.75× truth** | **+0.6872** |

⛔ **Calibration got WORSE in absolute terms and the gate became grossly over-confident — and a `b`-only reading would have called it the best result in the programme.** **That is the argument for item 1, it is from our own measurements three amendments ago, and neither seat connected it until now.**

### Required changes, adopted

1. ⛔ **Report `a`, `b` AND the held-out calibration error `|gate−obs|` per arm.** `b` remains the **primary** endpoint — under-dispersion *is* the slope, and that is H2's hypothesis — but ⛔ **`a` gets its own threshold and a level-only effect is a POSITIVE result, not a null**; and **the calibration error is the composite, reported because E6f proves a large slope move can coincide with calibration worsening.**
2. ⛔ **`σ̂` is estimated SEPARATELY for `a`, for `b`, and for the calibration error.** They are different statistics with different variances, and **using `b`'s spread as `a`'s denominator would be W-28's units error in a new place** — their catch, and they are right that it is the same failure.
3. **F10, F11, `4.303`/`2.905`/`4.969` and the `S ≡ σ̂` locking apply to each endpoint independently**, with its own `σ̂`.

### Readings, extended

| # | condition | consequence |
|---|---|---|
| **W-1** | `Δb < −4.303·σ̂_b` (or `−2.905·σ̂_b` under the recommended allocation) | ⛔ **weight decay implicated in the SLOPE defect** — H2's hypothesis. Cause only, never a ship decision |
| **W-1a** | `\|Δa\| > ` its own threshold, with `Δb` null | ⛔ **weight decay moves the LEVEL and not the slope. A POSITIVE finding, not a null** — it means the decayed **bias** is doing the work and the under-dispersion has another cause |
| **W-1b** | both move | ⛔ **report both and the composite.** A slope gain with a level loss is **E6f's signature** and must not be reported as an improvement |
| **W-2** | all endpoints within threshold | INCONCLUSIVE at this n — the expected cell |
| **W-3** | `Δb > +` threshold | wrong direction — H2's mechanism refuted |
| **W-4** | W-2 **and** no arm's `b` below ~1.10 | evidence for H3/H4; R1 is the ceiling |
| **W-5** | `σ̂_b > 0.0601` | ⛔ **Phase 1's amber-vs-copper comparison was never resolvable** — non-optional amendment to `OUTCOME.md`/`RESULTS.md` |

## 2 — ⛔ UPHELD. Five runs, two seeds named. The third was unnamed.

The allocation is `3 × 0.1 + 2 × 0.0` — **five runs — and the plan named only 42 and 43.**

⚠️ **An unnamed seed is the provenance gap this programme has now found fourteen times.** *"Another seed"* in a plan becomes *whatever the operator typed* in a config, and then **the run that produced a number cannot be reproduced from the document.**

**Adopted, and it buys a free pairing:**

```
wd 0.1   seeds 42, 43, 44
wd 0.0   seeds 42, 43
```

⛔ **Two of the three cross-cell comparisons are then seed-paired**, removing seed as a between-cell nuisance for those two, while the unpaired third contributes to `σ̂` without confounding the contrast.

✅ **`44` verified inside the repository's established range:** roster `torch_seed` values are **42, 43, 44, 45, 46, 47**. **It is not invented here.**

## 3 — Scope: these are changes to the PLAN, not to the pods

⛔ **Neither item changes the arm set, the cost, the code, or anything that runs** — item 1 extends what is *recorded and read*, item 2 names a value the plan had left blank. **Both tighten rather than loosen, and neither needs Simon's word.** This seat reached that conclusion independently of their saying so.

⚠️ **Still awaiting his direct word, unchanged:** the allocation, the `wd = 0.0` trainability risk, and — if he keeps the head-exemption arm instead — the library change in this repository.

## 4 — What they checked and found clean

The `vehicle.py` accommodation transferring (all five arms inherit amber's `truncated_nb`, so the one documented override covers them), the truth window being a within-design constant, and W-28's control. **Recorded, because a negative result from a check is a result.**

---

⛔ **Eight of this seat's items corrected in this document. The fifth improvement was the one that mattered most: a design that tested one of two parameters a known mechanism moves, written by the seat that had measured the distinction being decisive.** **Nothing is running and no model directory exists.**

---

# AMENDMENT 6 — ⛔ two of this seat's own positions cannot both hold, and determinism is untestable in this design. 2026-10-18.

**Source:** the `views-models` seat, comparing two of this plan's own paragraphs. ⛔ **Upheld. It is a logical inconsistency between two positions this seat argued, and it was not visible in either paragraph alone.**

---

## 1 — ⛔ THE CONTRADICTION

**Position A** — §7, this seat's reason for refusing to reuse amber:

> *"`amber_kestrel` was trained in a different pod session under a different code sha. Reusing it imports every difference between then and now into the variance estimate the design rests on."*

**Position B** — §7's pre-registered reading for arm 1:

> *"does not reproduce ⇒ training non-deterministic at fixed seed."*

⛔ **Position B compares the fresh run to amber across exactly the environment gap Position A says is large enough to corrupt a variance estimate. If the environment differs enough to invalidate a variance estimate, it differs enough to produce a sha mismatch on its own.**

**So a mismatch has at least three candidate causes and the design contains no discriminator:**

1. training non-determinism at fixed seed
2. a library / environment difference
3. a code-sha difference

⛔ **The mismatch branch is therefore not a reading. It is an attribution with three explanations and no way to choose between them.**

## 2 — ⛔ AND DETERMINISM IS UNTESTABLE HERE REGARDLESS. Verified for BOTH allocations.

Determinism requires **two runs identical in every input** — same config, same seed. Checked:

```
recommended  (0.1,42) (0.1,43) (0.1,44) (0.0,42) (0.0,43)   identical (config,seed) pairs: 0
planned      (0.1,42) (0.1,43) (0.01,42) (0.01,43) (0.0,42)  identical (config,seed) pairs: 0
```

⛔ **Zero in both. Determinism was never testable in this design, under either allocation** — not merely unsupported by the sha comparison, but absent from the design entirely. **Neither seat noticed, through four amendments of allocation argument.**

## 3 — ⛔ MY ADDITION: the asymmetry is a CONJUNCTION against a DISJUNCTION, and that is why their fix is right

They recommend keeping the sha comparison and dropping only the mismatch inference, on the grounds that *a match would be genuinely informative in the reassuring direction.* **That is correct, and the reason is worth stating because it says exactly how much a match buys:**

```
a MATCH    requires non-determinism AND environment AND code-sha differences to be ALL NULL
           -> a CONJUNCTION of three nulls -> strong evidence for EACH of them
a MISMATCH requires AT LEAST ONE of the three to be non-null
           -> a DISJUNCTION -> evidence for NONE of them individually
```

⛔ **So the comparison is worth running and only half of it is readable** — which is an unusual and legitimate shape for a check, and it must be labelled as such rather than presented as a two-branch reading.

### Adopted

> **The sha comparison is RECORDED FOR PROVENANCE.** A **match** across a changed environment and code sha is informative in the reassuring direction and should be reported as such. ⛔ **A MISMATCH IS NOT EVIDENCE ABOUT TRAINING DETERMINISM** — the environment and code sha also differ, and nothing in this design separates the three causes. **The determinism reading is withdrawn.**

## 4 — ✅ And their correction of their own attribution is right

They put the non-reproducing branch in, with the line this seat adopted verbatim — *"the amber/copper 0.17 gap may never have been the config key at all."*

⛔ **That sentence survives and is important. It belongs to W-5 — the Phase-1 audit via `σ̂` — which HAS a discriminator, and not to the sha comparison, which has none.** They attached a sound claim to the wrong test, and **the claim is unharmed by moving it.**

**W-5 is unchanged:** if `σ̂_b > 0.0601`, the amber-vs-copper difference was never separable from run noise, whatever its mechanism — ⛔ **and that holds without needing to know anything about determinism**, which is precisely why it is the version that works.

## 5 — ⛔ AND THIS SEAT DECLINES TO ADD A DETERMINISM TEST

The obvious response to §2 is to recover determinism by adding a matched pair. ⛔ **Declining, and recording why so a later reader knows it was considered rather than missed:**

**This seat's own argument, which they quoted back, established the decomposition is not load-bearing:**

> *"Arms 1–2 measure total run-to-run variance at fixed config — the right quantity whichever mechanism produces it. Deterministic ⇒ seed variance; non-deterministic ⇒ seed plus nondeterminism. Both are what §5 compares `Δ` against."*

⛔ **So recovering determinism would be scope creep for a question the design does not need answered.** A cheaper variant exists — two short runs (e.g. 10 lessons) identical in every input, which would test the training loop's determinism without 2 h 15 m — ⚠️ **but its cost is unmeasured, it may not hold at 300 lessons even if it holds at 10, and it answers a question that is not on this plan's critical path.** **Noted as available, deliberately not proposed.**

## 6 — ✅ A control this seat stated and they named as one

This plan recorded their clean checks (the `vehicle.py` accommodation transferring, the truth window as a within-design constant, W-28's control) on the stated ground that *"a negative result from a check is a result, and a reviewer who only reports hits leaves the next reader unable to tell what was looked at."*

⛔ **They identify that as a control in its own right, and it is: it is the search-provenance rule turned around.** Stating what was checked and came back clean is **what makes the absence of a finding mean anything.** **Adopted as a standing requirement for both seats' reviews.**

---

## State

⛔ **Nine of this seat's items corrected in this document.** **Item 1 came from the architecture; item 6 came from comparing two of this plan's own paragraphs** — ⛔ **neither was visible in the prereg's own text, which is the generalised form of W-28: re-reading your own document cannot find an error whose evidence is in a file, or a contradiction whose halves are in different sections.**

**This item tightens a reading and removes an unsupported inference. It changes nothing that runs and needs no authorisation.** ⚠️ **Simon's word is still required on: the allocation, the `wd = 0.0` trainability risk, and the library change if he keeps the head-exemption arm.**

**Nothing is running and no model directory exists.**

---

# AMENDMENT 7 — ⛔ THE ENDPOINTS ARE RESTRUCTURED. Measurements first; the causal question is demoted to a note. 2026-10-18.

**Source:** Simon, 2026-10-18, after the design churn: *"when you (or any other session) get things right it is about the 'what'. It is the 'how' and 'why' you struggle with."*

⛔ **He is right, and this repository's own record says so before he did.** `PREDICTION_LEDGER.md`'s by-kind table: **derivations 10 right / 1 wrong; causal narratives 0 right / 8 wrong.** He arrived at it independently, from watching the work.

**He has decided the experiment runs.** This amendment changes what it is pointed at.

---

## 1 — ⛔ The error this corrects, which was mine

The plan's primary question was *"is `weight_decay` the cause of the under-dispersion?"* ⛔ **That is a WHY.** It is the only kind of output that can come back inconclusive, because a causal verdict requires the effect to beat the noise — and §4's feasibility boundary showed we cannot know in advance whether it will.

⛔ **This seat then judged the whole experiment on that weakest output and recommended dropping it.** That is the category error Simon named, applied to the recommendation itself: **the experiment's strongest outputs are measurements, and measurements return a value unconditionally.**

> ⛔ **Pointed at the WHATs, this experiment cannot come back empty.**

---

## 2 — PRIMARY ENDPOINTS. All four return a value whatever happens.

| # | the question | what delivers it | can it fail? |
|---|---|---|---|
| **M1** | ⛔ **How much does this training pipeline vary run to run, at fixed config?** `σ̂` from the three identical-config runs (`wd 0.1`, seeds 42/43/44) | 3 runs, 2 df | **No.** It is a spread of three numbers |
| **M2** | ⛔ **Does `weight_decay: 0.0` train at all?** | the two zero-decay runs | **No.** Yes or no, and nobody currently knows |
| **M3** | **What are `a`, `b` and the held-out calibration error at each setting?** Reported as a table of measurements with no causal claim attached | all five runs + the local sweep | **No.** Five rows of numbers |
| **M4** | ⛔ **Does the calibration slope reproduce across identical runs?** M1 expressed as a readable fact rather than a denominator | the three identical runs | **No** |

⛔ **M1 is the primary endpoint of this experiment.** Nothing in this repository's record contains a run-to-run variance figure for the training pipeline. It is the most reusable number available: **it sizes every future experiment, and it is the denominator that every single-run comparison in this programme has silently been missing.**

### ✅ And one DERIVED reading stays primary, because it is arithmetic and not a story

**W-5 — what `σ̂` implies about the comparison this programme has been reasoning from.** `amber` and `copper` differ by **0.17** in `b` at **n=1 each, with no error bar anywhere in the record.** `se = σ̂·√2`, so:

> ⛔ **If `σ̂ > 0.0601`, the amber-vs-copper difference was never separable from run noise** — and the R1-works/R1-fails asymmetry built on it needs re-scoping, which makes the amendment to `OUTCOME.md`/`RESULTS.md` **non-optional.**

⚠️ **This stays primary deliberately.** It is a **derivation from a measurement**, not a causal narrative — the category the ledger records as 10-for-11, not the one it records as 0-for-8.

---

## 3 — ⛔ DEMOTED: the causal question becomes a secondary note

**W-1, W-1a, W-1b, W-2, W-3, W-4 are retained unchanged in their arithmetic and are no longer the point.**

> **Reported as: "the effect of `weight_decay` on `a` and `b`, with its uncertainty, at this n."** With `Δ`, `σ̂`, `Δ/σ̂`, and the threshold stated. ⛔ **Not as a verdict on whether `weight_decay` causes the under-dispersion.**

⛔ **And the readout may not assert a cause.** The strongest permissible sentence is *"the measured effect is X ± Y, which is / is not larger than the run-to-run variation at this n."* **F12 fires on any sentence in the readout that states or implies `weight_decay` is or is not the cause of the under-dispersion** — because the design cannot support it and because this is the error class the whole restructure exists to avoid.

**W-2 is no longer "the expected cell."** It is **the expected state of one secondary note**, which is a different and much smaller claim. ⛔ **The experiment's expected outcome is now four numbers and one derivation, not an inconclusive verdict.**

---

## 4 — What does NOT change

**Nothing about what runs.** Same five pods, same configs, same seeds, same allocation (`3 × weight_decay 0.1` at seeds 42/43/44; `2 × weight_decay 0.0` at seeds 42/43), same cost, same launch path, same gating questions settled in `2026-10-18_weight_decay_arms_launch_plan.md` §1.

**Every falsifier is retained:** F1–F11 unchanged, plus **F12** above. `S ≡ σ̂` as locked in Amendment 2. The per-endpoint `σ̂` requirement from Amendment 5. The sha comparison recorded for provenance only, per Amendment 6.

⛔ **This is a restructure of what the experiment is FOR, not a redesign of the experiment.** The five pods were already going to produce M1–M4; the plan simply did not treat them as the product.

---

## 5 — The honest case, restated

⛔ **Previously:** *"expected outcome INCONCLUSIVE for three independent reasons; cannot say any model is better to serve."*

✅ **Now:** **four measurements that return values unconditionally, plus a derivation that puts the first error bar on a comparison three days of work has rested on.** The causal question is a note attached to that, with its uncertainty stated.

⚠️ **Unchanged and still true: no arm can be declared shippable** (F6) — no forecast stage means no `act_ratio`, no CRPS, no AP, so the firing-rate guardrail is unavailable.

---

# AMENDMENT 8 — ⛔ M1 has a validity condition, and the cells it depends on are the only ones with no guard. 2026-10-18.

**Source:** the `views-models` seat, following this seat's byte-identity catch one step further and finding a worse gap in its own issue. ⛔ **Verified here and it is wider than either of us stated.**

---

## 1 — Every pair guard is keyed to `EXPERIMENT_ARMS`, so a replicate has NO pair guard of any kind

> ✅ **FIXED by views-models #566 / PR #582, merged `4c29fb9e`. Amendment 11 records what changed and what it did NOT fix.** The description below is kept as the state this amendment was written against; the struck block is **no longer the code.**

~~`tests/test_roster_conformance.py` — all five pair guards are `@pytest.mark.parametrize("arm", sorted(EXPERIMENT_ARMS))`:~~

```
~~:765  the declared-contrast check~~
~~:798  differs-from-its-control-only-on~~
~~:880  byte-identity over ARM_IDENTICAL_FILES   <- the one they found~~
~~:926  (pair guard)~~
~~:949  (pair guard)~~
```

✅ **Now (verified in the merged tree):** five of the seven run over `PAIRED_MODELS`, which is `EXPERIMENT_ARMS ∪ SEED_REPLICATES`, and two keep `EXPERIMENT_ARMS` because they are contrast assertions a replicate must NOT satisfy:

```
:873   test_the_arm_is_its_declared_values                                    arm    <- CONTRAST, stays
:906   test_the_arm_differs_from_its_control_only_on_its_declared_variable    arm    <- CONTRAST, stays
:1013  test_the_paired_model_shares_its_controls_data_configuration_byte_for_byte   paired   <- the one they found
:1059  test_the_paired_model_is_evaluated_the_same_way_as_its_control         paired
:1078  test_a_paired_model_is_neither_a_roster_member_nor_an_ensemble_constituent   paired
:1097  test_the_paired_models_rationale_survives_readme_regeneration          paired
:1183  test_a_paired_model_is_a_candidate                                     paired
```

⛔ **So it is not "the byte-identity test misses replicates". It is that the entire pair-guard apparatus is keyed to `EXPERIMENT_ARMS`, and a model outside it is unguarded as a pair in every respect** — no genome check, no byte-identity, nothing. **That was the gap; it is now closed by widening five of the seven.**

And `:883-890` states the hazard in its own words: `REGION = "land"` → `"africa_me_legacy"` *"trains it on 13,110 Africa/ME cells against a control trained on 64,818 global land cells, **makes the comparison meaningless, and leaves every test in the repository green**."*

## 2 — ⛔ AND THE CONSEQUENCE FOR M1, WHICH IS WHY THIS MATTERS MORE THAN A HYGIENE GAP

**The asymmetry is the point:**

| | supposed to | guarded? |
|---|---|---|
| the two **arms** (`indigo`, `russet`) | **differ**, on `weight_decay` | ✅ yes — they are in `EXPERIMENT_ARMS` |
| the three **replicates** (`silver`, `bronze`, `cobalt`) | be **identical** | ⛔ **no — nothing** |

⛔ **The cells with no guard are precisely the cells M1 depends on.** `σ̂` is the spread of the three identical-config runs. **Any config divergence among them feeds straight into `σ̂` as variance.**

⛔⛔ **And here is why no reading in this plan could catch it:** a divergence would **inflate** `σ̂`, which raises every threshold, which makes every effect fail to reach significance — **producing exactly the outcome already pre-registered as expected (W-2, inconclusive).** **A corrupted `σ̂` is indistinguishable from the honest expected result.** It does not look like a failure; it looks like a measurement.

> ⛔ **So #569 is not pre-merge hygiene. It is the VALIDITY CONDITION for the epic's primary output**, and it must be applied to the three replicates with **more** care than to the two arms — the opposite of where the repository's guards point.

## 3 — Locked: `σ̂` is never reported without its validity condition attached

> ⛔ **M1's readout must carry a positive assertion that the three replicates are byte-identical to each other and to `amber_kestrel` across all six `ARM_IDENTICAL_FILES`, and identical in `CORE_GENOME` except `np_seed`/`torch_seed`.**
> **`σ̂` quoted without that assertion is VOID** — not because it is wrong, but because nothing distinguishes training variance from config divergence once they are in the same number.

**This is the same shape as `OUTCOME.md §3.5`'s standing rule** — *never rank on CRPS alone, read it with `act_ratio`* — applied to the primary endpoint of this experiment: **a number that cannot be read without its companion.**

**New falsifier:**

| | fires if | then |
|---|---|---|
| **F13** | `σ̂` is reported without the byte-identity and genome assertion for all three replicates | ⛔ **VOID the number.** Nothing separates training variance from config divergence |

## 4 — What this does not change

**Nothing about what runs.** Same five pods, same configs, same seeds, same allocation, same cost. ⛔ **This adds one assertion to a readout and one falsifier.** It is a check on M1, discovered in code, and it is the category the record says this seat is reliable in.

---

# AMENDMENT 9 — ⛔ §3's arm table was the SUPERSEDED set, and F1's authority was that table. 2026-10-18.

**Source:** the `views-models` seat, reading this pre-registration end to end. ⛔ **Upheld in full. As written, F1 would have voided three of the five models AFTER they trained — three pods and the entire intervention cell, lost to a stale table in my own document.**

---

## 1 — The contradiction, verified

`§3:54-58` still carried the originally authorised set. **Three names mapped to different configs than the approved allocation:**

| name | §3 said | approved |
|---|---|---|
| `cobalt_kestrel` | `0.01` / seed 42 | **`0.1` / seed 44** |
| `indigo_kestrel` | `0.01` / seed 43 | **`0.0` / seed 42** |
| `russet_kestrel` | `0.1`, heads exempt / 42 | **`0.0` / seed 43** |

⛔ **And `F1:178` named §3 as its authority** — *"the full diff is printed in §3 before the pods start."* **Evaluated as written against §3's table, `cobalt`, `indigo` and `russet` each differ from their "named key and seed", so all three VOID after ~2 h 15 m of training each.**

⚠️ **No amendment superseded §3.** Amendments 3, 4 and 5 §2 changed the allocation in their own sections and **never went back to fix the table they invalidated.** That is on me: **I wrote six amendments on top of a section I had made wrong and did not re-read it once.**

## 2 — ⛔ It was stale in FOUR ways, not one

Found while fixing it:

1. **The three config mappings** above.
2. **`§3:66` carried the "~0.19 expected effect"** — ⛔ **withdrawn in Amendment 2 A2** as the size of the *whole* defect. A withdrawn number was still doing load-bearing work in a live section.
3. **`§3:54` called `silver` "the determinism precondition"** — ⛔ **withdrawn in Amendment 6.**
4. **`§4(i)` and `§5` still described the comparison as `0.01` vs `0.1`** — ⛔ the third dose was ruled to **`0.0`** in Amendment 4, so the readings' own `Δ` was defined on a dose that is not in the design.

## 3 — What is corrected

- **§3 now leads with the APPROVED table**, with the superseded one **struck in place beneath it** in a collapsed block, labelled, with the reason the trail is kept: ⛔ *"a reader who finds an old message must be able to see which mapping is live."*
- **F1 re-pointed:** its authority is **§3's approved table only, never the superseded one**, and it states that the diff is verified by views-models **#569** before the merge rather than printed here.
- **§3's config-diff precondition** now says it **cannot** be printed until the directories exist, so it reads as **#569's scope** rather than an unmet precondition of this plan.
- **§3:66** replaced: the replicates are the point, and the `~0.19` is marked withdrawn with the reason.
- **§4(i)** corrected: the approved allocation gives **3 df** pooled, better than the 2 df that paragraph was written for.
- **§5's `Δ`** corrected to `wd = 0.0` minus `wd = 0.1`, with `S ≡ σ̂` per Amendment 2 §1.

## 4 — ⛔ The lesson, and it is the one W-28 already named

**Six amendments were written on top of a section they invalidated, and none of them re-read it.** Each amendment was internally correct and checked against the *measurement* it concerned; **none was checked against the document it was amending.**

> ⛔ **An amendment that changes a design must name every section it invalidates, or the plan's own body becomes the stalest thing in it.**

⚠️ **This is the exact generalisation of W-28 the other seat and I agreed on two days of work ago** — *"re-reading your own document cannot find a contradiction whose halves are in different sections"* — **and it then happened across nine amendments of my own.** Adopted as a standing requirement: **every future amendment carries a line naming the sections it supersedes, or it is incomplete.**

## 5 — Still outstanding, and it is theirs to tell me

⚠️ They have read §§0–12 and Amendments 3, 4, 5 in full and **skimmed 1, 2, 6, 7, 8.** I have corrected every stale reference a grep for the superseded values finds in the **live** sections. ⛔ **The amendments' own internal text is historical and correctly so** — it records superseded reasoning deliberately — **but if anything in 1, 2, 6, 7 or 8 is asserted as live and contradicts the approved set, it is a defect and I want it reported.**

---

# AMENDMENT 10 — ⛔ EVERY THRESHOLD IN THIS PLAN WAS DERIVED AT n=2 vs n=1. The approved set is n=3 vs n=2. 2026-10-19.

**Source:** the `views-models` seat, answering Amendment 9 §5's request to report anything in Amendments 1–8 asserted as **live** and contradicting the approved allocation. ⛔ **They found it, it is the thresholds themselves, and it is the worst class of defect in this document: a pre-registered DECISION RULE that would misfire.**

---

## 1 — Reproduced, then recomputed

I verified their arithmetic numerically before changing anything:

| | `se` | `df` | `t(.975,df)` | crit | boundary for `0.1875` |
|---|---|---|---|---|---|
| **plan as written** — n=1 arm vs n=2 control mean | 1.2247 | 2 | 4.3027 | **5.270·S** | **0.0356** |
| ✅ **approved, pooled** — n=2 vs n=3 | 0.9129 | **3** | 3.1824 | **2.905·S** | **0.0645** |
| ✅ **approved, conservative** — `σ̂` from the 3 controls only | 0.9129 | 2 | 4.3027 | **3.928·S** | **0.0477** |

⛔ **The plan's critical value is 1.81× too harsh.** `0.0356` is wrong; the boundary is **0.0645** pooled or **0.0477** conservative.

## 2 — ⛔ WHY THIS IS WORSE THAN A STALE NUMBER: it is a decision rule that misfires

§4 pre-registered: *"`S > 0.036` ⇒ the arms cannot resolve even a total effect… the correct response is MORE REPLICATES."*

⛔ **So in the band `0.0356 < S < 0.0645` the plan commits IN ADVANCE to declaring the design uninformative when it is in fact informative** — and that band sits squarely between the measured sampling floor (**0.0107**) and the amber/copper upper bound (**~0.120**), which is exactly where this plan says nobody knows which side the answer falls.

**A stale figure misleads a reader. A stale decision rule produces the wrong action automatically.**

## 3 — ⛔ WHICH DENOMINATOR IS LOCKED, AND IT DECIDES WHETHER TWO READINGS CAN BOTH BE POSITIVE

Their observation, and it must be settled in advance rather than discovered in the readout:

```
pooled boundary        0.0645   is ABOVE W-5's separability boundary 0.0601
conservative boundary  0.0477   is BELOW it
```

**So:**
- **Pooled, with `σ̂ ∈ (0.0601, 0.0645)`:** ✅ the design is readable **and** ⛔ the amber/copper gap was **not** separable. **Both readings positive.**
- **Conservative, with `σ̂ ∈ (0.0477, 0.0601)`:** ⛔ the design is **not** readable **but** ✅ the amber/copper gap **was** separable. **The opposite pair.**

> ⛔ **LOCKED: pooled (`C = 2.905·S`, df=3) is PRIMARY. The conservative value (`C = 3.928·S`, df=2) applies if and only if F10 fires** — the within-cell spreads differing by more than 3×, which A3 says is plausible because weight decay is a regulariser and removing it may raise variance. **Both boundaries are stated here so neither can be selected after the fact.**

## 4 — The three further consequences of the same root

1. **§5's `4.30·S` (Amendment 1's re-cut) is stale** — corrected in place to `C`, with both values given. ⚠️ **This seat called §5's original `2S` "too lenient by 2.15×"; `4.30·S` is now too harsh by 1.48×.** **Two corrections in opposite directions to the same threshold, neither caught by re-reading.** W-2's *"2-df denominator"* is also stale — it is **3 df**.
2. ⛔ **Amendment 2 §2 is VOID BY CONSTRUCTION, not merely restated.** There is no n=1 cell, so **there is no separate existence test to carry a harsher threshold. W-1/W-3 and the existence test are the same comparison.** A live section asserting two different thresholds for one comparison is the shape that costs a readout. Marked void in place; the 1.5-factor derivation is retained as history.
3. ✅ **And one thing got BETTER and the plan did not say so.** W-0 read *"with 5 cells and no spares, one void control collapses the variance estimate."* ⛔ **With three control replicates, one void leaves two and `σ̂` survives at df=1. The approved allocation bought a spare on the control side.** Corrected — **the plan was more pessimistic than the design.**

## 5 — ⚠️ And `σ̂`'s definition is under-specified for n=3

Amendment 2 §1 locked `S ≡ σ̂` with `σ̂ = |x₁−x₂|/√2` — **pair-specific.** ⛔ **For three replicates it is the ordinary sample sd of the three values**, and pooling across cells is the usual `√(Σ(nᵢ−1)sᵢ² / Σ(nᵢ−1))`. **Their catch. Locked: `σ̂` is the pooled sample sd; the pair formula was a special case of it and is not the definition.**

## 6 — ✅ What they checked and found LIVE AND CORRECT

Recorded because a negative result from a check is a result: **Amendment 2 §1's `S ≡ σ̂` lock** (modulo §5 above), **§4's forward-sizing table** (explicitly for sizing the *next* experiment, so `n=2 → 0.0436` is not this design's own threshold), **Amendment 6** (which states "verified for BOTH allocations"), **Amendment 7's M1–M4**, and **Amendment 8's pair-guard gap.**

## 7 — ⛔ THE PATTERN, THIRD INSTANCE AND THE SAME ROOT AS AMENDMENT 9

**Amendment 9:** nine amendments written on top of §3, none re-reading it.
**Amendment 10:** ⛔ **four amendments written on top of Amendment 2's thresholds, none recomputing them when the allocation changed.**

> ⛔ **Amendment 3 changed the allocation and did not recompute the thresholds that the allocation determines.** The standing requirement adopted in Amendment 9 — *every amendment names the sections it supersedes* — **would have caught this one too, and it was adopted one amendment too late.**

**Strengthened: an amendment that changes `n`, the number of cells, or their sizes must recompute every threshold derived from them, or it is incomplete.** ⛔ **A threshold is not a number in a document. It is a function of the design, and changing the design silently invalidates it.**

---

## 8 — The insertion-point ruling, corrected and strengthened

✅ **Their correction: it is BOTH sigmoids, not `:1428` alone.** `:1325` feeds `_sample_feedback` at `:1331` exactly as `:1428` feeds it at `:1438`, and both then feed `acc_probabilities` at `:1337`/`:1460`.

⛔ **So the sigmoid is the SHARED UPSTREAM of both consumers — which is a stronger argument for the boundary than the single-site version.** Correcting at either sigmoid necessarily touches the autoregressive feedback. **The emitted gate cube is the only point downstream of the split.**

✅ **And the inert-knob trap is confirmed with its mechanism.** `extra = "allow"` at `config_initializer.py:1561`, and the near-miss guard `reject_near_miss_keys` at `:1503` **cannot** catch it: it returns early when `abs(len(a) − len(b)) > 1`, and `gate_temperature` (16) vs `gate_threshold` (14) differs by **2**. **Verified.** ⛔ **So a `gate_temperature` key validates silently, does nothing, and no guard in the library says so — C-324's signature, confirmed by construction rather than by analogy.**

---

# AMENDMENT 11 — ⛔ F13 HAS THE SAME GAP THE GUARDS HAD, AND IT FAILS IN THE OPPOSITE DIRECTION. 2026-10-19.

**Source:** the `views-models` seat, reporting S1 (#566) merged as PR **#582** / `4c29fb9e`, six checks green. They flagged two renamed names. ⛔ **Verified in the merged tree — and the rename is the smallest of the three things this changes.**

---

## 1 — ✅ The names, verified rather than taken on report

| what | status |
|---|---|
| five pair guards renamed `test_the_arm_*` → `test_the_paired_model_*`, argname `arm` → `paired`, set `EXPERIMENT_ARMS` → `PAIRED_MODELS` | ✅ **confirmed** at `:1013/:1059/:1078/:1097/:1183` |
| two CONTRAST guards keep `test_the_arm_*` / `arm` / `EXPERIMENT_ARMS` | ✅ **confirmed** at `:873/:906` — and this is right, not an oversight: they assert a model **differs** from its control, which a replicate must **not** satisfy |
| `PAIRED_MODELS = EXPERIMENT_ARMS ∪ SEED_REPLICATES` | ✅ confirmed, `:229-232` |
| **`ARM_IDENTICAL_FILES` keeps its name** | ✅ confirmed, `:171` region — **F13's citation stands unchanged** |
| `CORE_GENOME` keeps its name | ✅ confirmed — F13's second citation stands |

**Amendment 8 §1 is corrected in place** with the stale line-number block struck rather than deleted. ⚠️ **Note the shape: Amendment 8's text was not wrong when written and is wrong now, because the code it described was fixed in response to it.** That is the good case, and it still has to be marked, or a reader six months out finds a false description of a file and no way to tell which direction the staleness runs.

## 2 — ⛔ THE REAL FINDING: F13 PERMITS EXACTLY THE DEFECT THEIR NEW GUARD FORBIDS

Their most consequential addition is the seed's **inverse**: nothing required a replicate's seed to **differ** from its control's, so a replicate carrying its control's own seed *"passed every guard and contributed zero to σ̂."*

⛔ **F13, as this document locked it in Amendment 8 §3, has the identical hole:**

> *"…identical in `CORE_GENOME` **except** `np_seed`/`torch_seed`."*

⛔ **"Except" EXEMPTS those keys from the identity requirement. It does not require them to differ.** Three replicates all carrying seed 42 satisfy F13 word for word, and the readout would print the assertion as a **pass**.

**The design is safe — the plan names 42/43/44 (Amendment 5 §2, §831) and views-models declares them per replicate. But F13 is the thing the number is checked against, and it would not catch it.**

## 3 — ⛔⛔ AND IT FAILS IN THE OPPOSITE DIRECTION TO THE ONE AMENDMENT 8 WAS WRITTEN FOR, WHICH IS WORSE

Amendment 8 §2's entire argument was that config divergence **inflates** `σ̂`, raising every threshold, making every effect fail — *"indistinguishable from an honest null"*, i.e. **mimicking the outcome already pre-registered as expected.** Conservative, and therefore survivable.

⛔ **Seed collision does the reverse.** Every threshold in this plan is `C = k·σ̂` — `σ̂` is in the **denominator** of the comparison. So:

| corruption | effect on `σ̂` | effect on `C` | what it mimics |
|---|---|---|---|
| config divergence (Amendment 8) | **inflates** | rises | ⚠️ a null — the expected result. Conservative |
| **seed collision (this amendment)** | ⛔ **deflates toward 0** | ⛔ **falls toward 0** | ⛔ **a RESOLVED EFFECT. Anti-conservative** |

⛔ **With `σ̂` under-stated, any nonzero difference clears the threshold.** And the plan cannot lean on determinism to bound how far it deflates — **Amendment 6 withdrew the determinism precondition** — but the direction does not depend on determinism at all: identical seeds can only ever make three runs **more** alike than three distinct seeds would, so `σ̂` can only be **under-stated**, and a threshold divided by an under-stated spread can only be **too permissive**.

> ⛔ **This is the first corruption in this plan that would manufacture a positive result rather than suppress one.** Everything else on record here fails safe. **This one does not, and F13 as written would sign it off.**

## 4 — F13 amended

| | fires if | then |
|---|---|---|
| ~~**F13**~~ | ~~`σ̂` is reported without the byte-identity and genome assertion for all three replicates~~ | ~~VOID the number~~ |
| ✅ **F13 (amended)** | `σ̂` is reported without **(a)** byte-identity across all six `ARM_IDENTICAL_FILES`, **(b)** `CORE_GENOME` identity except `np_seed`/`torch_seed`, **AND (c)** a positive assertion that the three `np_seed`/`torch_seed` values are **PAIRWISE DISTINCT** | ⛔ **VOID the number.** (a) and (b) alone admit a zero-spread cell |

⛔ **(c) is not a tightening of (b). It is the direction (b) cannot express**, because an exemption and a requirement are different statements about the same key, and this document had only written one of them.

## 5 — ⚠️ AND F13 CURRENTLY HAS NO MECHANICAL BACKING. THE PEER SAID SO AND IT MUST BE IN THIS PLAN.

Their words, and this is the part a reader of the merge would otherwise get wrong:

- **`SEED_REPLICATES` is `{}`** (verified, `:161`). So the replicate guards **collect zero cases**, and `PAIRED_MODELS` currently equals `EXPERIMENT_ARMS`. **The guards that back F13 exist and cannot fire.**
- They proved all five mutations against a **synthetic** pair (`bright_starship` → `heavy_freighter`). ⛔ **Their own framing, kept verbatim because it is the honest distinction: that is evidence the guards are NOT DECORATIVE — it is NOT evidence they HOLD on the real `silver`/`bronze`/`cobalt` configs.**

> ⛔ **So F13 is satisfied by a human assertion at readout time, not by the suite, until views-models #567 declares the three replicates.** **#569 must therefore assert (a), (b) and (c) directly against the three real configs and record the values — not record that the suite is green**, because while `SEED_REPLICATES` is empty a green suite says nothing about them. ✅ Added to #569's scope.

## 6 — C-163, and it is the same lesson as Amendments 9 and 10

They registered **C-163** (Tier 2): C-120 closed with a rule asserted as *generalisable* — *"an escape hatch must be guarded at the same scope it operates"* — and this change broke it three ways at once **in the sibling of the file that cites it.** Their transferable form:

> ⛔ **When a declaration gains a sibling, audit the DECORATORS, the HATCHES and the INVERSE assertions.**
> **A rule in a register is a rule someone must remember; a guard is one the suite enforces.**

⛔ **That is this document's Amendment 9 and 10 lesson in different vocabulary, and the third place it has now been learned in two days:**

| | the sibling | what was not re-audited |
|---|---|---|
| **Amendment 9** | a new arm table | the **old** table, still cited as authority by F1 |
| **Amendment 10** | a new allocation | every **threshold** the allocation determines |
| **Amendment 11** | a replicate beside an arm | the **inverse** assertion, in their guards **and in F13** |

⛔ **All three are the same defect: a new case was added, and the statements that were only true of the old case were not revisited.** Amendment 9's standing requirement catches the first two by making an amendment name what it supersedes. **It does not catch the third**, because F13 did not supersede anything — it was incomplete from the moment a second kind of pair existed.

> ⛔ **ADOPTED, and it is the stronger rule: when this plan gains a new KIND of unit, every falsifier must be re-read for the direction it does not constrain.** An exemption is not a requirement, and a falsifier that only forbids one sign of an error is half a falsifier.
