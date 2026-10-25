# Pre-registration — the structure-penalty arm (`struct`)

> ## ⛔ AMENDMENT 2026-10-25 — THE VEHICLE CHANGED, AND ONE OF THE TWO HYPOTHESES IS WITHDRAWN
>
> Nothing below is deleted. Every original value is kept and annotated.
>
> **What happened.** §8 contrasted against `truncated_nb` `2a5efc2a…` because `vehicle.py` recorded it
> as *"the only one still on disk"* — the `nb` vehicle `4acbf4d3…` that every baseline in §4 was
> measured on had been **deleted**. On 2026-10-25 `body_scoreboard.py` was run against the
> `truncated_nb` vehicle for the first time and returned **B-1(d=1) = 1.462**, i.e. *above* its due
> share of spatial structure. ⇒ **The arm's premise does not exist on that vehicle.**
>
> A live `nb` vehicle now exists — **`heavy_freighter` `0ca3f31f…`**, trained 2026-10-25 on
> views-hydranet 0.1.7, global land, 300 lessons (`reports/2026-10-24_control_vehicle_prereg.md`).
> `vehicle.py` **Amendment 5** admits it; exactly one F6 key differs from the deleted vehicle,
> `dropout_rate` 0.125 against 0.15, and the non-comparability rule is stated at that site.
>
> **Measured on it, like-for-like** (population: the 1,431 cells with truth > 0 in 421–456; truth
> estimator: the 336-month base rate — the same combination that produced 24.7% and 0.4562, via
> `body_level_selection_check.py`):
>
> | | the deleted `nb` (what §4 is written against) | **`heavy_freighter`, live** | verdict |
> |---|---|---|---|
> | **B-1** at d = 1, in-sample E–W | 0.391 | **0.400** | ✅ **reproduces** |
> | **B-2** slope `b` | 0.4562 | **0.4996** | ✅ **reproduces** |
> | **B-3** level | 24.7% | **95.9%** | ⛔ **does NOT reproduce** |
>
> ⇒ **Two independent models, trained nine days apart at different dropout rates, agree on the
> spatial-structure deficit and on the spread deficit.** The *level* deficit is gone: this vehicle is
> at 95.9% of the right size on event cells, against a P-2 target of 40%.
>
> **Consequences, pre-committed before the arm runs:**
>
> | | |
> |---|---|
> | **P-2 (level) is WITHDRAWN** | the deficit it targets does not exist on this vehicle. It becomes a **guardrail**: B-3 must not *degrade*. ⛔ It may not be reported as a hypothesis test |
> | **P-3 (slope) STANDS**, restated against **0.4996** | ≥ 0.55 remains a genuine ask — about +0.05 |
> | **P-4 (B-1 d=1) STANDS**, restated against **0.400** | +0.10 absolute ⇒ ≥ 0.50. ⛔ **SUPERSEDED the same day by §4b — 0.400 is 28% MC-dropout artifact; the baseline is 0.287 and the target is 0.39** |
> | **§5's verdict shrinks to B-2 alone** | plus FAO-02. B-3 moves from verdict to guardrail |
> | **F2 is rewritten** | "B-1 moves but B-2 does not" — B-3 can no longer carry a negative |
> | **§8's contrast set is corrected** | it paired the `truncated_nb` *filename* with the `nb` *sha*. Verified: `calibration_model_20261013_010739.pt` is `2a5efc2a…`, not `4acbf4d3…` |
>
> ⚠️ **This makes the arm WEAKER AND HONEST.** It now tests **one** proposition, not two: *can a
> spatial term be made to produce structure, and if so does magnitude spread follow?* ⛔ And §1b's
> counterweight `C-3661` bears directly on it — the one published model that bought sharpness came
> out **under-dispersed**, with heavy-intensity decay at long lead. The literature supports the
> **mechanism** and is **neutral-to-negative on this verdict**. That is the state of the evidence and
> it is not to be softened.
>
> **The ceiling is no longer an open assumption.** `reports/2026-10-25_sharpness_ceiling_prereg.md`
> measured the achievable value of B-1 by splitting the truth into independent halves and applying
> B-1's own estimator: **0.983–1.009 across d = 1…32**. ⇒ The arm's implicit ceiling of 1.0 is
> **measured, not assumed**, and P-4 needs no capping. ⛔ That ceiling is an UPPER bound — systematic
> provider geocoding survives a split-half comparison.
>
> ### ⛔ SECOND AMENDMENT, SAME DAY — the derived weight was also wrong, three separate ways
>
> **`struct_penalty_weight` = 0.416904**, not §3b's **8.89624**. See **§3c**. The three errors: the
> old number was derived on the **`truncated_nb`** vehicle (not `nb`), it was **v1's** value that
> **v2 had already superseded** on disk, and it was matched against **one** of the **three** targets
> the loss sums over. ⭐ The third was found by Simon asking whether there is a penalty weight per
> violence type. There is not — **one global scalar, `tw = 1.0` per target**, and the **rarest**
> target (non-state) carries **45.3%** of the summed penalty.

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

## 1b. ⛔ Literature grounding — ADDED 2026-10-25, and it was missing

⚠️ **This plan went to the pre-flight gate with no citations at all.** The design notes behind it
carried four sources; this prereg carried none, and the four it carried were all *supporting*. Every
counterweight below was absent from both documents. Claim IDs are from `~/brain/9_library`, read
2026-10-25.

### ⭐⭐ The mechanism is a theorem, not an analogy

| source | claim | what it establishes |
|---|---|---|
| **Bishop 1994** `C-1522` | minimising a sum-of-squares error gives outputs approximating the **conditional average** of the target, `f(x) = ⟨t\|x⟩`, in the limit of enough data and a flexible enough network | ⭐⭐ §1's mechanism is a **limit result**, not a tendency. At its optimum a per-cell loss **is** the smeared field |
| **Bishop 1994** `C-1528` | on a multi-valued inverse problem a least-squares net "gives a very poor fit **because** it approximates the conditional average"; a mixture-density net recovers the modes | the worked failure case, from 1994 |

### ⭐⭐ It is the loss, not the architecture — someone ran that control

| source | claim | what it establishes |
|---|---|---|
| **Mathieu 2016** `C-3634` | a multi-scale Laplacian-pyramid architecture enlarges the effective receptive field and leaves sharpness **identical at 24.7** vs the single-scale ℓ2 model | ⭐⭐ the only experiment in the library that **separates** loss from capacity, and it lands on §1's side |
| **Mathieu 2016** `C-3633` | Adv+GDL 31.5/0.91/**25.4** vs ℓ2 27.6/0.86/**24.7**; *"the loss function, not the architecture"* | the effect size, on images |

### ⭐ Not an MSE quirk — any per-cell loss does it

| source | claim | what it establishes |
|---|---|---|
| **Ayzel 2020 (RainNet)** `C-3654` | trained with **`logcosh`** (≈ MAE, a median) and still loses power at **16 km and below**, at a **single** 5-min step | ⛔ switching to a robust/median loss does **not** escape it |
| **Isola 2017** `C-3642` | L1 **and** L2 blur; they capture low frequencies and fail on high-frequency crispness; "L1 incentivizes greyish, averaged colors" | two loss families, same defect |
| ⭐ **Shi 2015 (ConvLSTM)** `C-3292` | **our own architecture's paper** records that ConvLSTM blurs at longer lead times and attributes it to the model "blur[ring] predictions to minimize error under uncertainty" | the architecture we use says so about itself |

### ⭐ Why a *spatial* term is the right SHAPE of fix

| source | claim | what it establishes |
|---|---|---|
| ⭐ **Isola 2017** `C-3644` | a **70×70** patch discriminator is optimal; a **1×1 "PixelGAN" adds colour diversity but NO spatial sharpness**; PatchGAN "effectively imposes a Markov random field assumption, penalising structure at the patch scale" | ⭐⭐ **a per-cell critic cannot fix a per-cell defect.** The structural argument for `k > 1`, measured |
| **Mathieu 2016** `C-3632` | the Gradient Difference Loss penalises the **gradient difference** between prediction and truth (unlike total variation, which penalises the gradient itself) | ⛔ `struct_pen` is in GDL's family, generalised across scales — and §2's one-sided form is the same distinction |
| **Ravuri 2021 (DGMR)** `C-3657` | a U-Net on per-grid-cell regression had effective resolution **8 km at T+30, 32 km at T+90** on a 1 km grid; *"CSI does not penalise loss of small-scale structure"* | ⭐ the defect is invisible to the standard metric — which is §5's reason `AP` is not a verdict quantity |
| **Ravuri 2021** `C-3659` | **56 Met Office forecasters preferred the sharp model 88% / 89%** of the time (p < 10⁻⁴), calling the smooth one *"too bland"* and *"blocky"*; "standard metrics and expert judgments are not mutually indicative of forecast value" | the operational relevance, from forecasters rather than from a score |

### ⛔ THE COUNTERWEIGHTS — every one of these was absent, and they are the reason F2/F3 exist

| source | claim | what it costs this arm |
|---|---|---|
| ⛔ **Ravuri 2021** `C-3661` | **DGMR ensembles are UNDER-DISPERSED** vs PySTEPS on rank histograms, and heavy-rainfall intensity **decays** at T+90 | ⛔⛔ the sharp fix carries its own calibration cost — and **under-dispersion is a defect we already have** (the gate's logits are compressed, slope b > 1 in all six fits). This arm could trade one of our problems for the other, and nothing in the readout would catch that unless `crps_all` moves. ⚠️ Named here as a **limitation, not a falsifier** — adding a dispersion readout is out of scope for this plan |
| ⛔ **Ravuri 2021** `C-3658` | without grid-cell regularisation **CSI degrades**; without the discriminators, **best per-cell CSI but poor spectrum** (λ = 20 selected **for CRPS**) | ⛔ the sharpness/per-cell trade is **measured**, not hypothetical. One picks a point on that curve. ⇒ This is **F3** with a number behind it, and it is why the weight is derived rather than chosen for effect |
| ⛔ **Gneiting 2007** `C-3526` | the governing paradigm is **maximising sharpness SUBJECT TO calibration** — calibration is the constraint, sharpness the objective within it | ⛔ **F3 is not a preference, it is the field's standard.** Sharpness bought with calibration is not a trade to be argued afterwards |
| ⛔ **Mathieu 2016** `C-3633` | the winner was **Adv+GDL** (31.5), not GDL alone; adversarial-only reached 30.6 | ⛔ **`struct_pen` is the cheap, stable, non-GAN HALF of the literature's fix.** A partial effect is the *expected* outcome, which is consistent with §4's 35% prior and must not be read as a failure of the mechanism |
| ⛔ **Ayzel 2020** `C-3656` | proposes three remedies for the smoothing — #1 **"a spectral power penalty in the loss"** — and calls all three **UNTESTED** | ⛔ **our exact remedy is on the record as never having been run.** No one has published the result we are about to measure, in either direction. That is the value of the arm and also why no prior is borrowable |

### ⛔ THE GAP — nothing in the library is a count field

Every source above is **images or radar**: continuous, dense, and scored with L1 / L2 / logcosh / CSI /
PSD. ⛔ **Nothing in the library applies a spatial-structure penalty to a ~99.7%-zero COUNT field under
a hurdle likelihood.** The closest analogue is DGMR noting **89% of UK cells contain no rain**
(`C-3661`) — and they required importance sampling to train on it at all.

⇒ **Consequence, pre-committed:** the mechanism transfers by argument, **not** by precedent. No effect
size from any paper above may be quoted as an expectation for this arm. §4's priors stand on our own
measurements (B-1 0.391 at d = 1; B-3 24.7%) ⛔ **[AMENDED 2026-10-25: `B-3 24.7%` is
WITHDRAWN as a prior — it is **95.9%** on the vehicle this arm now runs on, so it grounds nothing. The
live priors are **B-1 0.400** and **B-2 0.4996**, both on `0ca3f31f…`. And with the level hypothesis
gone, counterweight `C-3661` is no longer a limitation at the edge of this plan — it is evidence
against its single remaining verdict, and must be read as such.]** and on the seven-failure base rate, and on nothing else.

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

⛔⛔ **THE TABLE BELOW IS RETRACTED — ALL THREE OF ITS NUMBERS. It is kept because each of its
three errors is a different kind of error, and the third was found by a question from Simon rather
than by any check in this document.** The replacement is the §3c block that follows it.

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

## 3c. ⛔⛔ AMENDMENT 2026-10-25 — THE PINNED WEIGHT WAS WRONG THREE SEPARATE WAYS

⛔ **`struct_penalty_weight = 8.89624` is VOID.** The replacement, derived on the vehicle that will
actually train and over the terms the loss actually contains, is **0.416904** — **21× smaller**.
`struct_weight_derivation_v3.py`, output in `struct_weight_derivation_v3.out`, **2.79 s / 2.90 GiB /
$0**.

⭐ **The three errors are independent, and they are worth separating because each one is a different
failure of this document rather than three symptoms of one.**

| # | the error | how it was caught |
|---|---|---|
| **1** | ⛔ **WRONG FAMILY.** Both earlier derivations ran on `calibration_model_20261013_010739.pt` — that is **amber_kestrel, `truncated_nb`** (sha `2a5efc2a…`). The arm runs on **`nb`**. The family swap alone moved the body NLL from 3.2202 to 0.4251 on the same window and the same seed, which is the same ~65× family effect §8 already measured for the body/gate ratio. | reading the two `.out` files while re-running; nothing in this prereg asserted which vehicle the weight came from |
| **2** | ⛔ **SUPERSEDED BY ITS OWN RE-RUN.** §3b's table reports **v1** (17.7925, halved to 8.89624). A **v2** run after the family-mean fix exists on disk and gives **27.0768 with no halving**. `2026-10-12_underfire_diagnosis_prereg.md:833-834` records 8.89624 as VOID and 27.0768 as its replacement — **so this prereg has been contradicted by a sibling prereg since 2026-10-21 and neither noticed.** | `struct_weight_derivation_v2.out` |
| **3** | ⛔ **ONE TARGET, APPLIED TO THREE.** The derivation called `_structure_penalty` **once**, on `SB`. `training_engine.py:995` accumulates it over **every** regression target and `:1127-1128` multiplies that **sum** by the single scalar. A weight matched against one of three terms lands the term at the wrong strength. | ⭐ **Simon's question** — *"do we have one for each type of violence?"* — not by any gate in this document |

### The replacement derivation, all three targets, vehicle `0ca3f31f` (`nb`, dropout 0.125)

| target | positive cells | body NLL | ‖∂NLL/∂θ‖ | struct pen | ‖∂pen/∂θ‖ |
|---|---|---|---|---|---|
| `ged_sb_best` | 2 | 0.425147 | 9.5437e-01 | 0.207398 | 3.0310e+00 |
| `ged_ns_best` | 2 | 0.386821 | 9.9605e-01 | **0.374910** | **3.7407e+00** |
| `ged_os_best` | 2 | 0.435904 | 1.7008e+00 | 0.244822 | 2.1537e+00 |
| **summed, tw = 1.0 each** | | **1.247872** | **2.1886e+00** | **0.827130** | **5.2496e+00** |

| | value |
|---|---|
| grad-norm-matched weight | **0.416904** |
| the term's share of the total loss | **21.7%** |
| §3's 50% fallback | **did NOT fire** |
| ⭐ **`struct_penalty_weight` = ** | **0.416904** |

### ⛔ 3c.1 There is NO per-violence-type penalty weight, and the equal weighting is not neutral

`struct_penalty_weight` is **one global scalar** (`config_initializer.py`, `Field(default=0.0,
ge=0.0)`). The only per-target dial is `target_weights`
(`config_initializer.py:204`, `Field(default=None)`), read at `training_engine.py:927-929`:

```python
tw = 1.0
if target_weights is not None:
    tw = target_weights[idx.reg_names[j]]
```

⛔ **`heavy_freighter` sets no `target_weights`** — the key is absent from the delivered artifact's
config sidecar and from the roster config. ⇒ **`tw = 1.0` for all three violence types**, so
state-based, non-state and one-sided receive **equal** structural pressure despite positive rates of
**0.009465 / 0.000861 / 0.004783** — an ~11× spread.

⚠️ **And the equal weighting is actively lopsided, measured rather than argued.** In the table above,
**non-state — the rarest target — contributes the LARGEST share of the summed penalty (0.374910 of
0.827130 = 45.3%) and the largest penalty gradient (3.7407e+00).** ⇒ The term points its hardest
structural pressure at the target with the least data to support it. That is a property of the arm
**as designed**, recorded here so the outcome is not read as a surprise, and it is **not** changed by
this amendment: introducing `target_weights` would be a **second variable** and the arm stays at one.

⚠️ **Scope of this fix.** It corrects the *scale* of a term whose *direction* is unchanged, and it
leaves C-325 exactly where §3b left it: still measured at **initialisation**, still one window, one
seed, one step. The 21× move is itself the argument for why an init-measured scalar deserves the
trajectory log §7.5 requires.

### ⚠️ 3c.2 Two things this amendment had to verify first, both of which could have failed silently

- ⛔ **The import path.** v1 failed with `_structure_penalty() takes 3 positional arguments but 4
  were given`: run from the `reports/` subdirectory, the **installed** `views_hydranet` **0.1.5**
  (3-arg) shadowed the working tree (4-arg). The env is a **regular install, not a dev install** —
  site-packages holds a real `views_hydranet` plus `views_hydranet-0.1.5.dist-info` and there is no
  editable finder. v3 prepends the repo root to `sys.path` and **prints the resolved
  `views_hydranet.__file__`** so a reader can see which copy answered. ⭐ And the throwaway venv this
  nearly cost: `git diff v0.1.7 -- views_hydranet/train/training_engine.py` is **empty**, so the
  working tree is byte-identical to the `v0.1.7` tag for this file and the 4-arg signature is
  **0.1.7's**, not a local edit.
- ⛔ **The head really is three targets wide.** Slicing `out.reg[:, j*npar:(j+1)*npar]` past the end
  of the tensor returns an **empty** slice in torch rather than raising, so a 2-target head would
  have produced a plausible-looking table. Checked explicitly: `out.reg` is **(1, 6, 32, 32)** = 3 ×
  2 NB params, `out.cls` is **(1, 3, 32, 32)**.

### ⚠️ 3c.3 A finding about this prereg, not about the weight

⛔ Error #2 means **§7's pre-flight gates cannot catch a stale pinned constant.** They assert the
config matches the prereg; they do not assert the prereg matches the last derivation that ran. Error
#1 means **no gate recorded which checkpoint a derived constant came from.** ⇒ A derived constant
needs its **vehicle sha and its script version named in the same sentence as its value**, which is
why the §3c table above names `0ca3f31f` and `struct_weight_derivation_v3.py` inline. Routed as a
gate gap, not fixed here.

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
| ⛔ **AMENDED 2026-10-25** | the three rows below are kept verbatim; the live baselines and their status are in the header amendment | | | |
| **P-2** ⛔ **WITHDRAWN** | B-3 level | **95.9%** on this vehicle, not 24.7% | ⇒ no deficit to close. Now a **guardrail**: must not degrade | — |
| **P-3** ⭐ **the hypothesis, restated** | B-2 slope `b`, 1,431 truth-positive cells, 336-month truth | **0.4996** | rises to **≥ 0.55** | **35%**, unchanged |
| **P-4** **restated** | B-1 at d = 1, in-sample E–W | **0.400** | rises by **≥ 0.10** ⇒ **≥ 0.50** | 60%, unchanged |
| ⛔ **P-4 SUPERSEDED, see §4b** | the **same** quantity, extrapolated to infinite MC-dropout draws | **0.287** | rises by **≥ 0.10** ⇒ **≥ 0.39** | 60%, unchanged |
| **P-2** ⭐ **the hypothesis** | **B-3 level**, in-sample, truth-positive cells, 336-month truth | **24.7%** | rises to **≥ 40%** | **35%** |
| **P-3** ⭐ **the hypothesis** | **B-2 slope `b`**, same population | **0.4562** | rises to **≥ 0.55** | **35%** |
| **P-4** | B-1 at **untrained** scales (d = 1, 4, 16, 32) | 0.391 / 0.491 / 0.810 / 0.870 | d = 1 rises by **≥ 0.10** absolute | 60% |
| **P-5** | FAO-02 `crps_all` | the control's | **not worse** by more than FAO-02's 5% margin | 55% |

⛔ **35% is a deliberate, low prior, and it is the honest one:** seven interventions have failed, four
of them on the same lever. ⚠️ A 35% prior means a null is the **expected** outcome and must not be
written up as a surprise.

## 4b. ⛔ AMENDMENT 2026-10-25, BEFORE THE ARM RUNS — P-4's BASELINE WAS 28% MEASUREMENT ARTIFACT

⛔ **P-4's old baseline 0.400 and old target 0.50 are both VOID. The baseline is 0.287 and the target
is 0.39.** The prediction, the quantity and the step size are unchanged; only the two numbers move,
and they move because the old ones were measured through an instrument that reads its own sampling
noise as signal.

### What was wrong

B-1 is read from `d4_result*.npz`, which stores μ for each **MC-dropout draw**. Each draw is a
different stochastic forward pass, and randomness between draws is **spatial variability** as far as
a variogram is concerned. ⇒ Fewer draws ⇒ a noisier μ field ⇒ the model's field looks **rougher** ⇒
**B-1 reads higher**. The control has 4 draws, so its B-1 carries four draws' worth of that noise.

Measured on the control's own stored draws — 11 cubes, all through `body_scoreboard.py` itself, CPU,
$0 (`reports/2026-10-25_arm_readiness/draw_count_dependence.csv`):

| MC-dropout draws | cubes | B-1 at d = 1, in-sample E–W | sd |
|---|---|---|---|
| 1 | 4 singles | 0.7547 | 0.0075 |
| 2 | 3 disjoint 2-vs-2 partitions (6 halves) | 0.5288 | 0.0033 |
| **4** | the full cube | **0.4000** | — |
| **∞** | 1/n fit `0.2870 + 0.4700/n` | **0.2870** | — |

⇒ **About 0.11 of the control's 0.400 — roughly 28% — is our own MC-dropout noise.** P-4 asked for a
**+0.10** rise in a quantity whose artifact term at n = 4 is **0.1175**: the yardstick carried an
error the size of the effect.

⭐ **The direction of the error FAVOURS this arm's premise, and that is why it must be written down
before the run rather than after.** The body's spatial-structure deficit is **deeper** than this
document has claimed throughout — 0.287 of its due structure, not 0.400 — so there is *more*
headroom, not less. An amendment made after seeing the arm's result could not be distinguished from
choosing the baseline that flattered it.

### The new numbers, and where each comes from

| | value | derivation |
|---|---|---|
| **baseline** | **0.287** | the 1/n fit's intercept on the control's 11 cubes |
| **step** | **+0.10** | ⭐ **UNCHANGED** — the same absolute step §4 already committed to |
| **target** | **≥ 0.39** | 0.287 + 0.10, i.e. Simon's standing rule *"rewrite those two targets to sit above where the model already is"* applied to the corrected baseline |

⛔ **The step size is not re-chosen here.** Had it been, this amendment would be selecting an effect
size with the instrument's behaviour already known. The only inputs are the corrected baseline and
the rule that was already in force.

### ⛔ How P-4 is to be READ — fixed now, before any arm number exists

> **P-4 is judged on the ∞-draw extrapolated value, not the 4-draw value.** The arm's
> `d4_result` cube is split into the same 4 singles and 3 disjoint 2-vs-2 partitions, each is run
> through `body_scoreboard.py`, and `B-1(d=1)` is regressed on `1/n_draws`. **The intercept is the
> number P-4 is scored against.** Cost: 4 + 6 variogram runs, ~17 min CPU, $0.

⚠️ **Why this clause exists in this form.** Both readings will be available once the arm lands — the
4-draw value and the intercept — and they can disagree about pass/fail. Naming the estimator *after*
the numbers exist is the discretion this pre-registration is for removing. ⛔ Both numbers are to be
**reported**; only the intercept **decides**.

### ⚠️ What this amendment does NOT claim

- **It does not fix P-3.** It does not need to: measured through its own instrument
  (`body_level_selection_check.py`, 336-month estimator), the slope moves **+0.008** across a 4× change
  in draws and the 4-draw value 0.4996 sits **0.003** from its fitted asymptote 0.5024. ⇒ **P-3 is
  draw-count robust and its baseline and target stand unchanged.**
- **It does not establish that the artifact cancels arm-vs-control.** It does not. The artifact's size
  depends on how much dropout variance the μ field carries, and the structure penalty changes field
  dispersion **by design** — so assuming cancellation would assume the arm has no effect. That is the
  whole reason P-4 moves to the intercept instead of keeping the 4-draw value on both sides.
- **It does not settle the 1/n form by theory.** The form is **measured**: residuals ±0.007 for B-1 and
  ±0.0001 for P-3's slope across a 4× range. ⛔ This is explicitly *not* `W-60`'s error, where a 1/n
  premise was **assumed** across a log transform and withdrawn — but it is **three draw counts fitting
  two parameters**, one residual degree of freedom, so the intercept is **indicative**. The robust
  claims are the sign and the order of magnitude.
- **It does not touch the noise floor, which is now measured and is not the problem.** Same draw count
  on both sides, the three partitions give a max |A−B| of **0.0080** on B-1 and **0.0021** on P-3's
  slope — **12× and 24× below** their asks. The `/falsify` finding that no floor existed is answered;
  what replaced it is this bias, which is a different defect.

### ⚠️ Retroactive scope, recorded rather than quietly left

The same artifact inflates **every** 4-draw B-1 number this programme has published, including the
D12 headline *"the body carries 0.391 of its due spatial structure at 1 cell"* on the deleted `nb`
vehicle. ⛔ Those figures are not corrected here — the deleted vehicle cannot be re-measured — but
they are **upper bounds on the model's structure**, i.e. the deficit they describe is understated.
Registered as part of this amendment's trail; cross-ref **C-128** (the locked-dropout posterior,
which already recorded that 11.2% of the body field's spatial variance at 4 draws is MC noise and
that it is **spatially correlated**, lag-1 +0.27 to +0.46 — this amendment is that fact arriving in
a headline).

## 5. ⛔ The readout, and why B-1 is NOT the verdict

**A within-block dispersion penalty and B-1 (the γ-ratio profile) are nearly the same object.** ⇒
Training on one and reading out on the other is **Goodhart by construction**.

| role | number |
|---|---|
| ⚪ **manipulation check** | **B-1 at k = 2, 8** — did the term do what it was told. Necessary, and **not evidence of improvement** |
| ⭐ **the verdict** | **B-2 (slope) and B-3 (level)** — magnitude calibration, which the term does **not** optimise |
| ⛔ **AMENDED 2026-10-25** | **the verdict is B-2 (slope) ALONE.** B-3's deficit does not exist on `0ca3f31f…` (95.9%), so it becomes a guardrail that must not degrade and cannot carry a verdict either way |
| ⭐ **the verdict** | **FAO-02: `crps_all` primary, with QS99 / Brier / MCR guardrails** — unchanged and untouched |
| ⚠️ **partial independence only** | B-1 at **d = 1, 4, 16, 32** while training at `k = 2, 8`. **This reduces the leak; it does not remove it**, and no claim will be made as if it did |

⛔ **`AP` is not a verdict quantity here** — it is blind to spatial arrangement by construction
(`scripts/ap_block_bootstrap.py:19`), and C-352 records that one of the platform's three AP
constructions is coupled to dispersion. It is reported as an FAO-02 guardrail and read no further.

## 6. ⛔ Falsifiers, committed now

| ID | fires if | consequence |
|---|---|---|
| **F1** | B-1 at `k = 2, 8` does **not** move | ⛔ the term is **INERT** ⇒ **VOID, not negative** — an implementation failure (C-324's class), and the arm is re-run after a fix, not written up |
| **F2** | ⭐ B-1 moves but **B-2 and B-3 do not** ⛔ **[AMENDED 2026-10-25: "B-2 alone". B-3 is a guardrail on this vehicle and cannot fire F2.]** | ⛔ **NEGATIVE, and the strongest possible one:** the objective can be made to produce structure, and structure does not buy magnitude calibration. ⇒ The loss is **not** the lever for the body's timidity, and the three remaining mechanisms (BN recal, `weight_decay`, recurrent state) become the live list |
| **F3** | `crps_all` degrades beyond FAO-02's 5% margin | ⛔ **NEGATIVE** — sharpness bought with calibration, the wall `project_body_knob_quest` and `project_body_mask_sweep_negative` both hit. ⚠️ **Not a trade to be argued afterwards** |
| **F4** | the arm's weight hash equals the control's | ⛔ **VOID** — the config never reached the model (C-324/C-331) |
| **F5** | non-finite loss or gradient at any lesson | ⛔ **VOID** — report, do not interpret |

## 7. ⛔ VOID conditions and the gates

1. **n = 1 at 300 lessons is a SCREEN.** A null is **INCONCLUSIVE, not negative** (C-307), measured
   against the ~**0.019** seed sd. ⛔ Only F2 and F3 produce a *negative*; absence of effect does not.
2. **C-184 BatchNorm seed-bimodality** — an unstable run is **VOID**. ⛔ **CORRECTED 2026-10-25:**
   this clause read "/ C-119 ~20% training variance". **C-119 is RESOLVED and that figure is a
   PRE-FIX measurement** (C-357), so it is not a live noise floor and must not appear beside a VOID
   rule as if it were. The live floor is §7.1's **≤0.019 seed sd**; the demonstrated paired-by-month
   resolution is **±0.010**. This is registered risk **M-5** appearing in a third pre-registration.
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
`truncated_nb`, git `f99413e` lineage) — the artifact the whole scoreboard is baselined on.

⛔ **CORRECTED 2026-10-25 — this names a file/sha pair that does not exist, and the vehicle is
replaced.** Verified on disk: `calibration_model_20261013_010739.pt` is **`2a5efc2a67c08eac…`**
(`truncated_nb`, amber_kestrel); **`4acbf4d3…`** was the `nb` vehicle
`calibration_model_20261012_033112.pt`, which is **DELETED**. The row above pairs one vehicle's
filename with the other's digest.

⇒ **The contrast set is now `heavy_freighter` `0ca3f31ff86ab348…`** — `nb`, global land, 300 lessons,
views-hydranet 0.1.7, trained 2026-10-25, with `PROVENANCE` and a 172-package `ENVIRONMENT` beside it.
⛔ One variable relative to it: `struct_penalty_weight` from 0 to the derived value, trained from the
**same config** on the **same package version**. ⛔ One
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
