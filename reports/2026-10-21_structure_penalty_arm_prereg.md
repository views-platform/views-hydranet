# Pre-registration — the structure-penalty arm (`struct`)

**2026-10-21.** Written **before** anything is built or run. ⛔ **This verb writes the plan only.**
Design notes behind it: `2026-10-21_loss_arm_design_notes.md` (with its §5b seam read). Ruler:
`2026-10-21_body_scoreboard.md`, baselined on the control **before** this plan existed.

---

## 1. Hypothesis

> **The body's field is over-smooth, too flat and too timid because the objective is minimised by a
> smooth field — not because the network cannot represent a sharp one.** Adding a term that penalises
> a mismatch in *within-block dispersion* at small spatial scales will move the body's magnitude
> calibration, not only its sharpness.

⚠️ **The hypothesis is the second clause.** That the term moves *sharpness* is near-certain and is a
manipulation check, not a finding (§5). The claim with content is that **level and slope move too**.

## 2. The intervention — ONE variable

Add a fifth additive term at the reduction site, in the pattern
`train/training_engine.py:989-998` already uses four times:

```
loss = loss + struct_penalty_weight * struct_pen        # the new line
```

**The term**, per regression target `j`, over `k ∈ {2, 8}`:

> `struct_pen = Σ_k | log( Var_within-k-block(pred) + ε ) − log( Var_within-k-block(truth) + ε ) |`

| slot (C-351) | value |
|---|---|
| population | every `k×k` block of the training window's grid, **no admission filter**, zeros kept. ⚠️ Blocks are the 32×32 curriculum windows' own sub-blocks, so the population is whatever the curriculum feeds — recorded because it is **not** the 64,818-cell land grid the scoreboard uses |
| transform | `log(Var + 1e-6)`, so the penalty is scale-free and a 10× dispersion error costs the same at every intensity |
| estimator | within-block variance of the **composed** field `gate × mu` against the truth's counts, both unconditional (`W-52`'s lesson: `mu` alone is a *conditional* mean and may not be compared to counts) |

⛔ **The gate is `.detach()`ed inside the term.** The composed field is the only object comparable to
the truth, but detaching the gate means **gradients flow into the body alone**. ⇒ One variable: the
body's spatial structure. The gate is a fixed multiplier.

**Everything else is held:** family `truncated_nb`, `body_nll_denominator` unchanged,
`loss_class_pos_weight` 2.0, `forecast_composition` `soft_gate`, `freeze_multitask_balancer` **True**,
the balancer untouched (the term is added *after* it, so the 6-vector and all per-task instrumentation
are byte-identical).

## 3. The weight, derived rather than swept

⛔ `struct_penalty_weight` is a free parameter and the frozen balancer will not tune it. Instead of a
sweep (3–4× the cost), it is **set once by measurement, before the arm runs**:

> Measure `‖∂struct_pen/∂θ‖` and `‖∂body_NLL/∂θ‖` on the real vehicle at initialisation, on CPU, and
> set `struct_penalty_weight` so the two are **equal**. Report the measured value.

⚠️ **This is GradNorm's idea used as a one-time constant, not as a dynamic algorithm** — the
distinction matters because the dynamic version was declined on cost grounds. **Pre-committed
fallback:** if the derived weight makes the term exceed **50%** of the total loss at initialisation,
halve it until it does not, and report that this happened.

## 3b. ⛔ AMENDMENT 2026-10-21, BEFORE ANY ARM RUNS — the term is ONE-SIDED, and the weight is derived

**§2's term was symmetric (`|log Var_pred − log Var_true|`). It is now one-sided:**

> `struct_pen = Σ_k max(0, log Var_true − log Var_pred)`

⛔ **Why, and it was measured rather than reasoned.** On the derivation window — **2 positive cells
in 1,024**, representative of the curriculum's 1-in-222 rate — the symmetric form's value was
**12.381** against the one-sided **0.224**. ⇒ **98.2% of the symmetric penalty was punishing the
model for having MORE dispersion than a truth that has none**, i.e. enforcing **flatness** on a
97.8%-empty field. It would have fought the defect it exists to fix, and the arm would have been
worse than useless — it would have been *informatively* wrong in the opposite direction.

✅ **The asymmetry is justified by a pre-committed falsifier, not by convenience.** D12's **F1** —
*"the model's γ exceeds the truth's at any d"* — **never fired**: 4/4 window × direction cells, 480
distance bins, the model below the truth at **every** scale. **The defect has one direction, so the
penalty has one direction.**

### The derived weight (§3), measured

`struct_weight_derivation.py` on a **fresh** model built from the control's own config sidecar, one
real window chosen by **land coverage only**, CPU, **8.39 s / 2.90 GiB**:

| | value |
|---|---|
| body NLL / its gradient norm | 3.220172 / **5.932e+01** |
| structure penalty / its gradient norm | 0.223680 / **3.334e+00** |
| grad-norm-matched weight | **17.7925** → the term would be **55.3%** of the loss |
| ⛔ §3's 50% fallback | **FIRED once** |
| ⭐ **`struct_penalty_weight` = ** | **8.89624** (term at **38.2%** of the loss) |

⚠️ **Measured at initialisation, which is C-325's exact trap** (two mechanism tests measured at init
were recorded as ruling a mechanism out; the arm's gradient exploded 33 lessons later). ⇒ **This is a
starting scale, not a claim the two terms stay balanced.** The per-task trajectory log is what would
show them drifting apart, and §7.5 already requires it to be on.

### Implementation status

✅ **Built and committed** (`6de8afb` and before): `_structure_penalty` + `_block_variance` in
`train/training_engine.py`, the fifth additive term at the reduction site, `struct_penalty_weight`
declared in the schema with `reject_inert_struct_penalty`, CIC field count 101 → 102. **13 tests,
2288 passing, ruff clean.** Mutation-tested: the un-detached gate, a constant penalty, the removed
divisibility guard, a changed block set, the removed `clamp_min`, the symmetric form and a flipped
direction are **all caught**. ⚠️ **Not independent** — I wrote the code and the tests.

⛔ **And one process failure, recorded because it is the third instance:** the one-sided change was
mutation-tested while **uncommitted**, so `git checkout --` reverted the mutation and ate the change
with it (`feedback_commit_before_mutation_testing`, written after the first two). Recoverable here,
but it also produced a **false mutation result** — the file had silently reverted to the committed
symmetric version, so one mutation looked caught by the wrong test and I briefly recorded the
smoothing test as weaker than it is. **Re-run against committed code with an assertion that each
mutation actually applied.**

## 4. Pre-registered predictions, with a prior

| # | quantity | baseline (control) | prediction | my confidence |
|---|---|---|---|---|
| **P-1** ⚪ *manipulation check* | B-1 at the **trained** scales | — | the within-block dispersion gap closes by **≥ 50%** at `k = 2` and `k = 8` | **85%** — if this fails the term is inert, not wrong |
| **P-2** ⭐ **the hypothesis** | **B-3 level**, in-sample, truth-positive cells, 336-month truth | **24.7%** | rises to **≥ 40%** | **35%** |
| **P-3** ⭐ **the hypothesis** | **B-2 slope `b`**, same population | **0.4562** | rises to **≥ 0.55** | **35%** |
| **P-4** | B-1 at **untrained** scales (d = 1, 4, 16, 32) | 0.391 / 0.491 / 0.810 / 0.870 | d = 1 rises by **≥ 0.10** absolute | 60% |
| **P-5** | FAO-02 `crps_all` | the control's | **not worse** by more than FAO-02's 5% margin | 55% |

⛔ **35% is a deliberate, low prior, and it is the honest one:** seven interventions have failed, four
of them on the same lever. ⚠️ A 35% prior means a null is the **expected** outcome and must not be
written up as a surprise.

## 5. ⛔ The readout, and why B-1 is NOT the verdict

**A within-block dispersion penalty and B-1 (the γ-ratio profile) are nearly the same object.** ⇒
Training on one and reading out on the other is **Goodhart by construction**.

| role | number |
|---|---|
| ⚪ **manipulation check** | **B-1 at k = 2, 8** — did the term do what it was told. Necessary, and **not evidence of improvement** |
| ⭐ **the verdict** | **B-2 (slope) and B-3 (level)** — magnitude calibration, which the term does **not** optimise |
| ⭐ **the verdict** | **FAO-02: `crps_all` primary, with QS99 / Brier / MCR guardrails** — unchanged and untouched |
| ⚠️ **partial independence only** | B-1 at **d = 1, 4, 16, 32** while training at `k = 2, 8`. **This reduces the leak; it does not remove it**, and no claim will be made as if it did |

⛔ **`AP` is not a verdict quantity here** — it is blind to spatial arrangement by construction
(`scripts/ap_block_bootstrap.py:19`), and C-352 records that one of the platform's three AP
constructions is coupled to dispersion. It is reported as an FAO-02 guardrail and read no further.

## 6. ⛔ Falsifiers, committed now

| ID | fires if | consequence |
|---|---|---|
| **F1** | B-1 at `k = 2, 8` does **not** move | ⛔ the term is **INERT** ⇒ **VOID, not negative** — an implementation failure (C-324's class), and the arm is re-run after a fix, not written up |
| **F2** | ⭐ B-1 moves but **B-2 and B-3 do not** | ⛔ **NEGATIVE, and the strongest possible one:** the objective can be made to produce structure, and structure does not buy magnitude calibration. ⇒ The loss is **not** the lever for the body's timidity, and the three remaining mechanisms (BN recal, `weight_decay`, recurrent state) become the live list |
| **F3** | `crps_all` degrades beyond FAO-02's 5% margin | ⛔ **NEGATIVE** — sharpness bought with calibration, the wall `project_body_knob_quest` and `project_body_mask_sweep_negative` both hit. ⚠️ **Not a trade to be argued afterwards** |
| **F4** | the arm's weight hash equals the control's | ⛔ **VOID** — the config never reached the model (C-324/C-331) |
| **F5** | non-finite loss or gradient at any lesson | ⛔ **VOID** — report, do not interpret |

## 7. ⛔ VOID conditions and the gates

1. **n = 1 at 300 lessons is a SCREEN.** A null is **INCONCLUSIVE, not negative** (C-307), measured
   against the ~**0.019** seed sd. ⛔ Only F2 and F3 produce a *negative*; absence of effect does not.
2. **C-184 BatchNorm seed-bimodality / C-119 ~20% training variance** — an unstable run is **VOID**.
3. ⛔⛔ **THE W-22 GATE, and it is the one most likely to be skipped.** The pod installs
   `views-hydranet~=0.1.1` **from PyPI** (`tools/podrun/pod_run_model.sh:171-173`) and
   `HydraNetConfig` sets `extra="allow"`, so a config key that exists only on a local branch is
   **silently accepted and ignored** — the arm returns **byte-identical to its control** and looks
   like a clean null. ⇒ **Assert, on the pod, that the installed package contains
   `struct_penalty_weight` before the run starts.** This cost a $2.25 confounded null once already.
4. **Standing gates** (the #311 precedent): `scripts/floor_gate.py` FG-A and FG-C on the **control**
   first; the potency pre-flight on the arm's config **and at a trained checkpoint** (C-324/C-325);
   `scripts/arm_identity_check.py`; `scripts/arm_postflight.py`; the weight-hash post-condition;
   `kill_tree` (C-326).
5. **Instrumentation that must be on, because it was off on both delivered runs:**
   `trajectory_log_path` set, and the per-task loss split logged (#398's six columns).

## 8. Contrast set

Against the **control** `calibration_model_20261013_010739.pt` (sha `4acbf4d3c2b457b0…`,
`truncated_nb`, git `f99413e` lineage) — the artifact the whole scoreboard is baselined on. ⛔ One
variable relative to it: `struct_penalty_weight` from 0 to the derived value.

## 9. Cost

⚠️ **ESTIMATE, marked as such** — the standing rule is that a cost is measured before it is quoted,
and nothing has run. By analogy with prior arms: **~$2.25, one arm, 300 lessons** on datafactory, plus
minutes of CPU for the weight derivation and the scoreboard readout. ⛔ The actual figure gets measured
and reported.

## 10. ⛔ What this plan does NOT do

- ⛔ **No `views-models` commits.** The arm needs a roster config, which lives there ⇒ **a GitHub issue
  only**, per the standing rule.
- ⛔ **It does not touch FAO-02.** The evaluation framework is unchanged; this changes a *training*
  objective, and that distinction is already on the record (`2026-10-15_loss_balance_prereg.md`).
- ⛔ **It is not the gate version.** The clean proper-score form of this idea exists only for the gate
  (`1 − Π(1−gᵢ)` is closed form and differentiable, the body's block sum is not). ⇒ **The elegant
  version is the one we are not doing**, deliberately — see
  `feedback_the_body_is_the_target_not_the_gate`.
- ⛔ **It cannot separate the four smoothness mechanisms.** A positive result says the objective is
  *sufficient* to move the body, not that it was the binding constraint.
- ⛔ **Nothing is launched** by this file. The next steps are: derive the weight (CPU, minutes), build
  the term behind a default-off flag, open the views-models config issue, run the gates.

## 11. Vocabulary

⚠️ `reports/GLOSSARY.md` has no term for this object. ⇒ **the structure penalty** is added to the
glossary in the same commit as this plan, rather than a synonym being invented here.
