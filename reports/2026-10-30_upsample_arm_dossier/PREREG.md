# PREREG — the upsample remedy, as candidate 7 of the 2026-08-24 architecture bake-off

**2026-10-30. Committed ALONE and before any architecture code exists** (D-5, and the bake-off
dossier's own Phase 2 gate: *"`05` committed alone"*).

⛔ **This document is the verdict rule. It is written before the numbers exist and I may not change
it after seeing them.** If I later wish to, §A3.3 states what that costs.

---

## 0 · Why this arm exists, and why it is NOT new work

`M92`/`M93` localised the body's spatial smoothness to a named defect:
`ConvTranspose2d(kernel=2, stride=2)`, where **kernel = stride ⇒ the windows never overlap**, so
output at `(2i+a, 2j+b) = W[:,:,a,b] · x[i,j]` — **each of the four in-block positions gets its own
free weight matrix.** The stage **replicates and rescales; it never interpolates.**

Measured at `up1`, dropout off, teacher-forced: ρ(1) **+0.123146** against ρ(2) **+0.920900**, with
channel-wise ρ(1) from **−0.724900** to **+0.718700** (negative ⇒ adjacent cells
**anti**-correlated). End to end: fed **+0.344210** → encoder **+0.313470 / +0.443300**
(roughness-**neutral**, bracketing its own input) → output **+0.958400**.

⇒ ⭐ This is **candidate 7 of `reports/2026-08-24_architecture_bakeoff_dossier/`**, not a new
programme. It is registered, run, scored and logged through that dossier's existing harness, and
its read-out appends to that dossier's append-only `07_experiment_log.md` as **EXP-02**.

---

## 1 · ⛔⛔⛔ THE PRIOR, AND IT PREDICTS THIS ARM FAILS

`07_experiment_log.md` EXP-01: **12 arms, 6 architectures × 2 seeds, L=300, 12/12 scored, zero
failed arms. NOTHING PROMOTED** — five INCONCLUSIVE, and **`ShallowPool` REJECT** (−0.0178 /
−0.0150 Δ AP@h18).

⭐ **`ShallowPool`'s docstring named `M92`'s mechanism two months before I measured it:** *"the
bottleneck sits at ¼ resolution and every placement decision is made from features reconstructed
from a quarter-scale map."*

| horizon | h1 | h6 | h12 | h18 | h24 | h30 | h36 |
|---|--:|--:|--:|--:|--:|--:|--:|
| **arms better** | **9** | 9 | 5 | 5 | 2 | **0** | **0** |
| arms worse | 3 | 3 | 7 | 7 | 10 | **12** | **12** |

- ⭐⭐ **12/12 negative at h30 AND h36** from six unrelated mechanisms — `2⁻¹²` under independent
  signs — and the gradient is **monotone**: 9 → 9 → 5 → 5 → 2 → 0 → 0.
- ⭐⭐ **"Handed a real field, every candidate matches or slightly beats the control's ceiling."**
  Oracle @h18: control **0.4974 / 0.5014**, candidates to **0.5077**. ⇒ **Every one sharpened the
  per-frame map. None converted to rollout skill.**
- ⭐⭐⭐ **"The damage is ordered by how aggressively each architecture intervenes."** The two that go
  furthest to preserve spatial resolution are worst at h36 — **DualStream −0.0567**, **ShallowPool
  −0.0475**, 3–4σ, the largest effects in the table.
- **"The body is untouched; this is entirely the gate's placement degrading through the loop."**
  Δ`crps_all`@h18 within ±0.006 for every arm.
- The mechanism, with literature that predicted it first — `Aceituno2025_TemporalHorizons` Thm 4.6:
  minima found on short horizons do **not** generalise to long ones, gap `O(e^{λ(T_h−T_l)})`.
  **We train at one step (ε=0, teacher-forced) and evaluate at thirty-six.**

⇒ ⛔ **`M92`/`M93` are TEACHER-FORCED T=0 measurements — precisely the quantity EXP-01 showed does
not transfer.** This arm sharpens the per-frame map at the **finest** scale, so this repository's own
prior art predicts it **improves the oracle and loses at h36.**

### ⭐⭐⭐ Which is why it is worth running: the hypothesis has never been tested PROSPECTIVELY

EXP-01's horizon gradient is explicitly labelled **post-hoc**: *"It cannot promote anything under §4
and is not offered as a verdict."* ⇒ A **7th unrelated mechanism, predicted in advance**, is its
**first prospective test** — informative either way. That, not the fix, is what this arm buys.

---

## 2 · ⛔⛔ THE REMEDY — revised before any code, because the obvious one is disqualified

`base = total_hidden_channels = 32` (`violet_visitor/configs/config_hyperparameters.py:38`). The 12
upsample modules hold **245,760** of the incumbent's **905,289** parameters.

| remedy | upsample params | total | Δ vs incumbent |
|---|--:|--:|--:|
| ⛔ `nearest + Conv2d(3×3)` — **what #491 proposed** | 552,960 | 1,212,489 | **+33.9%** |
| ⭐⭐ `nearest + Conv2d(2×2)` | 245,760 | 905,289 | **+0.0% — EXACT** |
| ⭐ `nearest + Conv2d(1×1)` | 61,440 | 720,969 | **−20.4%** |

⛔ **+33.9% is 5.6× the most capacity-confounded arm in the dossier** (WideMemory, +6.0%), whose
table concludes *"no arm is capacity-confounded in the way first feared."* ⇒ **A 3×3 arm would be
uninterpretable whichever way it went, and it is NOT RUN.** ⚠️ #491 §5 named the `9/4` kernel ratio
and never computed what it meant for the total — the same error shape as the `0.499` comparator.

**What each surviving remedy does to the mechanism, derived:**

- **`nearest + 1×1`** — every phase applies the **same** matrix to the same replicated value ⇒ ⭐⭐
  **no phase pattern can exist, by construction.** No mixing in the stage; mixing stays with
  `dec_conv0/1`, which is already 3×3 and already does it. **−20.4% params.**
- **`nearest + 2×2`** — at `(even, even)` the window sees four copies of `x[i,j]` and applies the
  **summed** kernel; at the other three phases it **straddles block boundaries and mixes
  neighbours.** ⇒ Residual phase structure is a **constraint on shared weights**, not a free
  parameter. **+0.0% params — the `AntiAliasedPool` standard**, which the dossier calls *"the
  cleanest possible experiment in the whole space"* precisely because it adds zero parameters.

⇒ ⛔ **The remedy is NOT chosen by argument. `G-F` (§4) measures all three modes on CPU and the one
that removes the phase signature at the lowest parameter cost takes the GPU slot** (D-4: sweep the
free parameter before committing a scalar decision).

---

## 3 · What is measured (C-351, every slot)

| slot | value |
|---|---|
| population | the bake-off's own: `fullzero_*` controls, seeds **42** and **43**, L=300, paired |
| truth | the dossier's battery — gate `AP`/`Brier`/`precision_at_k`/`act_ratio`/`n_false_pos`; body `crps_all`/`crps_events`/`crps_none`/`size_ratio`/`mcr_*`/`pos_mcr` |
| estimator | `score_v2_horizons.py`, unchanged, at **h1/6/12/18/24/30/36**, plus the **oracle ceiling** and **retention** per arm |
| **inference mode** | ⭐ **AUTOREGRESSIVE ROLLOUT** — ⛔ **the opposite of `M92`'s teacher-forced cube, and that is the point** (`C-399`) |
| free parameters | the **remedy** (swept on CPU in `G-F`, §4); the **seed** (42, 43) |
| the one varying config key | ⛔ **`model`. Nothing else.** Per the dossier: *"the only config key that differs between an arm and its control is `model`."* |

⚠️ **Parameter count is reported beside every result**, as the dossier requires of all six prior
candidates.

---

## 4 · `G-F` — the mechanism check. CPU, minutes, zero cost.

`gf_phase_signature.py`, committed before it runs. Reuses `stage_roughness.py`'s `pair_rho` and its
ramp / checkerboard / iid-noise controls **verbatim**.

⭐ **Three modes, UNTRAINED, one seed** — control, `nearest+2×2`, `nearest+1×1`. ⛔ **Untrained vs
untrained is the only clean comparison:** the replacement convs have no trained weights, so
comparing them against a *trained* transposed conv would measure training, not geometry.

**The statistic:** the gap `ρ(2) − ρ(1)` at `up1`, and the channel floor at ρ(1).

⚠️ **`G-F` is a real test, not a tautology.** `Upsample(nearest)` makes the conv's input blocky in
2×2 blocks, and a window centred at an even position covers a different arrangement of those blocks
than one at an odd position ⇒ **a phase pattern CAN survive the remedy.**

⛔⛔ **CORRECTED 2026-10-30 against the source.** This paragraph attributed that caution to Odena et
al. 2016 — *"the fix reduces checkerboarding; it does not abolish it."* **Odena's wording is
STRONGER than that**: *"they do fix the checkerboard artifacts"*, qualified only by *"we don't
necessarily think that either approach is the final solution to upsampling."* ⇒ **The
"may survive" argument is MY OWN derivation** (the parity of a small window over a 2×2 block grid),
not the source's, and it should never have been hung on a citation. ⭐ `G-F` then settled it **in
Odena's favour**: the gap went to **−0.1984**, i.e. the pattern was removed, not merely reduced.

⛔ **`G-F` is NOT a gate on whether the arm runs.** The arm's premise is `M92`, already measured.
`G-F` only establishes which remedy addresses it. Per **S-2**, a `G-F` failure lowers confidence and
closes nothing.

---

## 5 · PRE-REGISTERED PREDICTIONS

### 5a · ⭐⭐ Simon's predictions — HIS, attributed. The first non-mine rows in the ledger.

| id | prediction | how it settles |
|---|---|---|
| **S-1** | *"We'll do the tests. You will find out that it doesn't work. I will ask you if you're sure that it really doesn't work. You will review your own test and you will tell me that actually your test doesn't prove that it doesn't work. And we are left with zero knowledge."* | ⭐ By **my own conduct**, against §A3's rule — committed here, before any number exists |
| **S-2** | A FAIL does not kill an arm; it **lowers a probability**. *"We have so much evidence"* — things were rejected here and later turned out to be real | ⭐ **ALREADY CONCEDED**, verified against the record below. Adopted as a standing rule in §6, not re-tested |

⭐ **S-2's evidence, checked in the repository rather than agreed to reflexively:** **M1**
(persistence beats us — measured at 40 lessons, nobody re-ran the race; at L=300 the *worst* of 4
seeds beats a fairly-scored persistence at *every* horizon); **M8** (freezing state recovers AP — 40
lessons; its 0.0912 is **3.6× below** an L=300 model's untouched 0.3298); **ZINB leads `crps_all`**
(v1/40-lesson, overturned); **the amount ceiling** (declared *"irreducible"*, reopened by Epic #230
as tail shape); and ⭐⭐ the rule *"superseded retires a conclusion, never a tool"* — which exists
**because** that marker hid a working reliability diagram and the 142× boundary measurement from a
four-round search. Plus `RESEARCH_PROGRAMME.md:504` in the register's own voice: *"none of them was
scored on the frozen ruler with a noise floor. That is the reason to reopen — not a new idea."*

### 5b · My predictions — stated against the MEASURED prior, not intuition

⚠️ `M92`'s four predictions were all refuted because **every one assumed the upsampling would BLUR
when it phase-stripes.** These are therefore anchored to EXP-01's measured gradient.

| id | prediction | threshold, declared now |
|---|---|---|
| **P-20** | `G-F`: at least one remedy removes the phase signature | gap `ρ(2)−ρ(1)` **< 0.15** and channel floor **≥ −0.05** |
| **P-21** | ⭐ `nearest+1×1` shows a **smaller** residual gap than `nearest+2×2`, because it cannot form a phase pattern at all while 2×2 can | strict inequality |
| **P-22** | ⭐ The oracle ceiling matches or beats control, as all six prior candidates did | Δ oracle @h18 **≥ −0.005** |
| **P-23** | ⭐⭐ **THE ONE THAT MATTERS.** Δ AP is **positive at h1 and NEGATIVE at h36** — the 13th instance of a `2⁻¹²` pattern | sign test, **both** seeds |
| **P-24** | ⭐ The h36 loss is **among the worst in the set**, because this arm sharpens the finest scale and EXP-01 found damage ordered by intervention aggressiveness | Δ AP @h36 **≤ −0.040** (cf. DualStream −0.0567, ShallowPool −0.0475) |
| **P-25** | The body is untouched | \|Δ `crps_all` @h18\| **≤ 0.006** |
| **P-26** | Parameter count matches §2 **exactly** for the chosen remedy | printed, then compared to 905,289 / 720,969 |

---

## 6 · ⛔⛔ THE VERDICT RULE — committed before the numbers exist

**Primary endpoints: `h1` AND `h36`.** ⛔ **`h18` is NOT an endpoint.** EXP-01's own stated design
error was choosing it: *"h18 was the worst available primary endpoint… it sits almost exactly at the
sign change and reports 'nothing is happening' for a pattern that is unmistakable at h1 and h36."*
⇒ **Not repeated.**

| branch | condition | ⇒ consequence, pre-committed |
|---|---|---|
| **CONFIRMS** | P-23 holds: Δ AP > 0 at h1, < 0 at h36, **both** seeds | ⭐⭐ The horizon-gradient hypothesis gets its **first prospective confirmation from a 7th mechanism.** ⇒ The architecture axis closes **on evidence rather than fatigue**, and the named follow-up — a **loss with horizon > 1**, which the dossier says *"has never actually been tried here, despite three experiments that resembled it"* — is promoted. ⛔ `M92`'s lever is **NOT** promoted. ⚠️ What would reopen it: a multi-step loss arm, under which a sharper per-frame map is no longer penalised. |
| **FALSIFIES** | Δ AP **≥ 0 at h36 on both seeds** | ⭐⭐⭐ The strongest result available here. Spatial architecture is **live again**, `M92` becomes actionable, and EXP-01's 12/12 needs re-reading as *six mechanisms that happened to share a flaw* rather than a law. |
| **MIXED** | seeds disagree at h36 | ⚠️ **Under-powered at n=2 — which is EXP-01's own n, and it was not enough to separate five arms from zero at h18.** ⇒ Lowers confidence, **closes nothing**, and names the seed count needed, **derived** from the observed spread rather than assumed. |

### ⛔⛔ The anti-S-1 devices. These are the point of this document.

1. ⛔ **An oracle-only / T=0-only improvement is declared a FAIL, here, in advance.** Six arms
   already did exactly that and none promoted. **I may not report it as encouraging.**
2. ⛔ **I may not report an `h18` verdict**, in either direction.
3. ⛔⛔ **If the result is negative and I then wish to argue the test was inadequate, that argument
   must first be logged as a `W-` row against myself in `PREDICTION_LEDGER.md` — BEFORE it is
   made.** This makes S-1's step 4 **expensive and visible** instead of free.
4. ⛔ **The ONE void condition for `G-F`, named now so every other post-hoc objection is disallowed
   by my own committed text:** if the **control** (`ConvTranspose2d(k=2,s=2)`, untrained, same
   seed) does **not** show the signature (gap **< 0.40**), the instrument is not detecting what it
   detected on trained weights and `G-F` is void. ⛔ **Nothing else voids it.**
5. ⚠️ **Per S-2, no branch above says "closed".** Every branch moves a probability **and names in
   the same sentence what would reopen it.**
6. ⛔ **The remedy is chosen by `G-F`'s numbers, not by argument** (§2). If I want a remedy `G-F`
   did not favour, that is a change to this document and falls under device 3.

---

## 7 · GATES

| gate | assertion |
|---|---|
| **G-F** | §4 — three modes untrained, plus the ramp/checkerboard/iid-noise controls on the ρ estimator **before any model is loaded** |
| **G-20** | No `ConvTranspose2d` remains on **any of the 12** upsample paths. ⛔ Changing only the `reg` path would make the gate's placement incomparable, and EXP-01 says the effect **is** the gate's placement |
| **G-21** | Every stage's output **shape** equals the incumbent's, at /2 and /1, all three heads, both branches |
| **G-22** | ⭐ The candidate is **NOT INERT** vs the incumbent. `tests/architectures/test_candidate_mechanisms.py` exists because *"**every** candidate's distinguishing mechanism could be removed, inverted or neutered and the whole suite stayed green"* — a neutered arm would run 2 GPU-h and report a null reading as *"this architecture does not help"* rather than *"this architecture was never applied"* |
| **G-23** | ⭐ Parameter count **printed** and equal to §2's figure for the chosen remedy |
| **G-24** | ⛔ **C-384** — MC-dropout via `set_locked_dropout(True)`, **never** a bare `model.train()`; assert **no `_BatchNorm` in train mode**. A bare `train()` put 15 of them there and **inverted the sign of a headline** |
| **F1** | `verify_bakeoff.py` re-asserts the arm's `model` **independently of the builder's declaration** |

---

## 8 · ⚠️ LIMITS, stated before the run

- ⚠️ **n=2 seeds.** EXP-01's own power, and it was **not** enough at h18. ⇒ Mitigated by taking
  **h1 and h36**, where the prior effect is 3–4σ and the sign test is `2⁻¹²` — **not** by claiming
  n=2 is sufficient.
- ⛔ **No σ̂ for Δ AP at h1 or h36 exists.** `C-357` has fired twice on arms sized against a σ̂ that
  does not exist for the quantity measured. ⇒ **No significance is claimed.** The verdict rule is a
  **sign test across two seeds**, which is what n=2 can support.
- ⛔ **This tests the REMEDY, not `M92`.** `M92` is already measured and is not re-opened by this
  arm's outcome. What is tested is whether removing the phase mechanism changes rollout skill.
- ⚠️ **`sb` is the headline target**, as in EXP-01.
- ⚠️ **The honest prior is that this arm loses** (§1). It is framed as a prospective test of the
  horizon hypothesis for exactly that reason, so a negative result is a **result**, not a wasted
  night.
- ⛔ **Out of scope, recorded as Simon's call:** the multi-step loss (promoted only under CONFIRMS),
  and `RP-5` (the θ-captured-by-zeros argument) — which sits on the **body** half, and EXP-01 found
  the body untouched.

---

## 10 · ⛔⛔ AMENDMENT 1, 2026-10-30 — the control must be RE-RUN, and the harness needs siblings

**Written before the arm is built, after `G-F` and after reading the harness.** Two facts invalidate
§3's claim that *"paired controls are free"*.

### 10a · The `fullzero_*` controls no longer exist, and their scores are not a valid pairing

⛔ `ls views-models/models/ | grep fullzero` is **empty** — the control model directories are gone.
Their **score CSVs** do survive in-repo (`2026-09-03_drivers_and_dynamics_dossier/results/`,
`2026-09-04_input_noise_dossier/results/s1/`), so a comparison is *possible*.

⛔⛔ **But it would not be one-variable.** Those controls were trained on August's code; the arm
would train on the **0.1.7 working tree** — two releases later. ⇒ The comparison would vary
**`model` AND the code version**, which breaks the dossier's own protocol (*"the only config key
that differs between an arm and its control is `model`"*) and is the `W-51`/`C-351` failure in its
plainest form.

⇒ ⭐ **DECIDED: the control is re-run alongside the arm.** `HydraBNUNet06_LSTM4` becomes an arm of
this queue, built by the same builder, on the same commit, in the same run.

| | arms | seeds | total runs | est. GPU-h |
|---|---|---|---|---|
| §7 as first written | 1 | 2 | 2 | ~5 |
| ⭐ **amended** | **2** (arm + control) | **2** (42, 43) | **4** | **~10** |

⚠️ Still **local, free, overnight** on the RTX 4070. Simon authorised the increase before it was
scheduled.

### 10b · Three sibling tools, because EXP-01's must not be edited

⛔ `preflight.py:33`, `smoke.sh:26` and `launch_bakeoff.sh` each carry a **hard-coded
six-candidate list** and write EXP-01's sentinels. Editing them would alter the instruments that
produced a committed experiment. ⇒ **New files under `reports/2026-10-30_upsample_arm/tools/`**,
with their **own** `RES_DIR` and sentinels, reusing `run_queue.sh`, `make_arch_arm.py` and
`verify_bakeoff.py` **unchanged**:

| file | role |
|---|---|
| `preflight_exp02.py` | profiles the two architectures; writes this dossier's `PREFLIGHT_OK` |
| `smoke_exp02.sh` | ⭐ **the real gate** — 2 real lessons per architecture, measuring actual training footprint, the emit path and BatchNorm recalibration; writes `SMOKE_OK` |
| `launch_exp02.sh` | the four specs, in the pre-registered order below, via `run_queue.sh` |

⚠️ **The freshness gate is working as designed:** EXP-01's `PREFLIGHT_OK`/`SMOKE_OK` are now
**stale**, because today's commits touched model code and guard tests. That is a refusal to respect,
not to widen.

### 10c · One additive edit to the shared builder, instructed by the tool itself

⛔ `make_arch_arm.py`'s `_ARCH_TAG` table has **no entry for the incumbent**, so the builder
**cannot build a control** — which is exactly why EXP-01 paired against arms from an earlier
programme. Its own error says *"Add a tag to the table rather than improvising a name."*

⇒ Three **additive** rows, changing no existing behaviour. Labels must match `^[a-z]+_[a-z]+$`
(⛔ letters only — **no digits**, which is why the tags are spelled out):

| variant | tag | label at 300 lessons, ε=0, seed 42 |
|---|---|---|
| `HydraBNUNet06_LSTM4` | `ctl` | `ctlfullzero_fortytwo` |
| `UpsampleConv2` | `upctwo` | `upctwofullzero_fortytwo` |
| `UpsampleConv1` | `upcone` | registered for a follow-up; **not queued** |

### 10d · Queue order, pre-registered

⭐ **Control first, both seeds**, then the arm — EXP-01's rule (*"if the harness is going to fail,
it fails on the cheapest arm"*) applied here, with the added reason that **a control that fails to
reproduce its own historical AP tells us the code moved**, and that is worth knowing before two more
runs:

```
300:42:0.0::HydraBNUNet06_LSTM4  300:43:0.0::HydraBNUNet06_LSTM4
300:42:0.0::UpsampleConv2        300:43:0.0::UpsampleConv2
```

⚠️ **A new prediction this amendment creates, declared now:** **P-27** — the re-run control
reproduces EXP-01's AP@h18 of **0.3298** (seed 42) / **0.3318** (seed 43) to within **±0.010**.
⛔ **If it does not, the code moved between August and now, and `M94`'s arm comparison is the only
valid one in this document** — every cross-reference to EXP-01's absolute numbers would become
indicative rather than comparable. ⭐ This is the single most useful by-product of re-running the
control and it is the reason to run it first.

---

## 11 · ⛔⛔ AMENDMENT 2, 2026-10-30 — two confounds an independent review found, one fixed and one declared

An independent `/code-review max` of PR #492 found that the arm **was not the single-variable
experiment this document pre-registers.** Both findings verified here by measurement before acting.

### 11a · ⛔⛔ FIXED — the shared conv stack was a different random draw

`utils.init_weights` dispatches on `isinstance(m, nn.Conv2d)`, and **`nn.ConvTranspose2d` is not a
subclass of `nn.Conv2d`.** ⇒ The incumbent's twelve upsample stages are the **only** weights in the
network the production init pass (`training_engine.py:94-95`) never touches.

Two consequences, both measured at base=32 with the production `weight_init: 'xavier_norm'`:

| | before | after the fix |
|---|--:|--:|
| shared parameters differing after `model.apply(init_weights)` | ⛔ **53 of 121** | ✅ **0 of 121** |
| `upsample0_head1_reg` weight std — incumbent **0.03612** | 0.05131 (xavier) | 0.02549 (torch default) |

⭐ **The 53/121 was the serious half**: it included `enc_conv0`, `enc_conv1` and `bottleneck_conv` —
modules the candidate does not touch — because constructing twelve extra modules advances the
global RNG, so every later xavier draw diverged. ⇒ **The paired arm-vs-control contrast at n=2 was
not a minimal perturbation; it was a full re-draw of the shared conv stack.**

⚠️ **And nothing in the repository could see it**: at construction the two models share 121
parameters with **0** differing, so every construction-time check reads clean, and the test suite
only ever builds via `get_architecture(...)` and never calls the production init pass. **Two
independent review angles stopped at construction and concluded "load-bearing, not waste."**

**Fixed two ways in `upsample_interp.py`:** the replacement block captures and restores
`torch.random.get_rng_state()` so it is RNG-neutral, and `NearestConvUp.apply` does not recurse into
its conv, so the global init pass cannot reach it and it keeps torch's default — exactly like the
`ConvTranspose2d` it replaces. ⚠️ `Module.apply` is distinct from `_apply`, which is what
`.to(device)` uses, so device moves are unaffected (verified).

### 11b · ⚠️ DECLARED, NOT FIXED — the remedy changes the stage's output scale, irreducibly

| | weight std | stage output variance on N(0,1) |
|---|--:|--:|
| incumbent `ConvTranspose2d(k=2,s=2)` | 0.03612 | **0.1707** |
| `UpsampleConv2` | 0.02549 | **0.3277** (1.92×) |
| `UpsampleConv1` | 0.05100 | **0.3413** (2.00×) |

**Both numbers are fully explained and neither is a defect:**

- The **√2** weight-std difference is the **weight layout**: `ConvTranspose2d` stores
  `(in, out, kH, kW)` ⇒ fan_in **256**, `Conv2d` stores `(out, in, kH, kW)` ⇒ fan_in **512**, and
  torch's default `kaiming_uniform_(a=√5)` scales by `1/√fan_in`. Predicted 1.414, measured 1.417.
- The **~1.9× output variance** is the remedy's own arithmetic: a k=2 conv over a
  **nearest-replicated** field puts all four taps on the **same** input cell and sums them, so it
  accumulates ~4× the variance of the transposed conv's single tap, offset by the √2 smaller
  weights ⇒ 4/2 = 2×.

⇒ ⛔ **This cannot be removed without changing the remedy itself** (e.g. an undeclared 1/√2 scale),
and it should not be: it is what replacing the operator *means*. ⚠️ It was **7.6×** before 11a's fix
and is **1.9×** now, so the fix removed the avoidable part.

⚠️ **What it costs the comparison, stated plainly:** `bn_dec_conv0`/`bn_dec_conv1` immediately
follow the concat, so the **absolute** scale is absorbed by BatchNorm. What is **not** absorbed is
the **relative weighting of the upsampled half against the skip half** inside
`torch.cat([upsample(b), e1s], 1)` — and the arm's own hypotheses `P-13`/`P-15` are about exactly
that pair. ⇒ **A difference in the upsampled branch's contribution is part of the intervention, not
a nuisance**, and no reading of this arm may attribute an effect to *geometry alone*.

---

## 9 · Attribution

Predictions **S-1** and **S-2** are **Simon's**, quoted verbatim, and are logged as his in
`PREDICTION_LEDGER.md`. Every `P-` row is mine.
