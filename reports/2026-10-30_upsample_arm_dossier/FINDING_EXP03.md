# EXP-03 — the upsample arm read on the BODY. The gate win does not replicate; the body sharpens at one cell; and FAO-02's primary score is a ~100-cell measurement.

**Date** 2026-11-03 · **Arms** 11 of 12 delivered (`upctwofullzero_fortysix` lost — see §8) ·
**Scripts** `tools/exp03_pairs.py`, `tools/exp03_magnitude.py`, `tools/exp03_sharpness.py`,
`tools/exp03_crps_concentration.py` · **Outputs** the matching `results/exp03_*.out` / `.json`

> ⛔ **LABEL, fixed before any number was seen.** `PREREG_EXP03_SEEDS.md` AMENDMENT 1 specifies
> §6's "both seeds" as **"all six, strictly"**. Five complete pairs cannot satisfy a six-seed
> unanimity test, so **everything here is an UNDERPOWERED READING, not a §6 verdict.** What *is*
> robust to the missing pair is the **disagreement**, because a sixth seed cannot un-resolve a
> resolved one (§2.2).

---

## 0 · Summary

| question | answer | strength |
|---|---|---|
| Does the arm improve the **gate** (AP) at h1? | ⛔ **No reliable effect.** 3 seeds resolve positive, **1 resolves negative** | 5 pairs, paired origin-block CI |
| Does it hurt the gate at h36? | ⚠️ Possibly slightly. mean −0.0042, 4/5 negative, 1 resolved | 5 pairs |
| Does it make the **body sharper**? | ✅ **Yes — at d=1, h1, unanimous on 5/5 seeds**, +0.0504 B-1 | sign count only, no interval |
| Does it raise **body magnitude** on event cells? | ✅ **Yes — upper tail, unanimous on 5/5**, growing with quantile | sign count only, no interval |
| Does any of that improve **CRPS**? | ⛔ **No. Null everywhere, in every band.** | 5 pairs |
| Does it do anything at **h36**? | ⛔ **No.** | 5 pairs |
| ⭐⭐ Can CRPS *see* an effect like this at all? | ⛔ **Largely no — 118 cells carry 81% of it at h36** | descriptive, exact |

⇒ **Reading:** the intervention does what it was designed to do — it places mass better at one cell
and lifts the event-cell tail — and the programme's primary score is structurally incapable of
registering that. The arm is **not** a null; the **score** is near-blind to it. ⚠️ But the body
effects are sign counts across five seeds and **not** intervals, and today's gate result is a
standing warning about exactly that (§1.1).

## C-351 slots, once, for everything below

| slot | value |
|---|---|
| **population** | 13 origins × **64,818 global land cells = 842,634 cell-months**, per horizon per arm. ⛔ NOT `V2_TRUTH`, the retired 13,110-cell Africa+ME crop — AP and CRPS both depend on prevalence |
| **truth** | the arms' **shared** global pull, `data/raw/calibration_datafactory_df.parquet`, `lr_sb_best`. All twelve arms verified at one digest `d3636988b1085af7` |
| **transform** | **none.** The gate is already a probability; the draws are already count space |
| **estimator** | the arm's own **S = 16** posterior draws. `E[Y]` is their mean; the gate is the posterior-mean over draws; CRPS is the frozen plug-in on 16 members |
| **endpoints** | **h1 and h36 only.** ⛔ No h18 number appears anywhere, in either direction (`PREREG.md` §6 device 2) |
| **primitives** | `average_precision` and `crps_ensemble` **imported** from the frozen ruler `reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py`, never reimplemented |

⚠️ **Equal S is load-bearing** (epic #263). All arms are S=16, so the plug-in CRPS bias and the
gate-leg quantisation bias are **common** and *differences* are interpretable — while *levels* are
not comparable to anything scored at another S.

---

## 1 · The gate: ΔAP across five complete pairs

`results/exp03_pairs.out`. ΔAP = `upctwofullzero_<seed>` − `ctlfullzero_<seed>`, i.e. each arm
against its **own-seed** control, so the symmetric difference is exactly `{"model"}`. Paired
origin-block bootstrap, `n_boot=400` (the committed default in `scripts/ap_block_bootstrap.py`),
90% CI: one origin draw per replicate scores **both** arms on the same resampled origin set, so the
correlation between them is carried rather than assumed away.

**G-0, and it gates everything in this section.** Seeds 42 and 43 were already published at h1 as
**+0.0049** and **+0.0047**. The script refuses to print a single figure unless it reproduces both
within 0.0005. Measured: **|Δ| = 0.00001 and 0.00000. PASS.** A pipeline that cannot re-derive a
number we already hold is not measuring what it claims.

### h1

| seed | AP control | AP upctwo | ΔAP | 90% CI | MDE | resolved? |
|---|---|---|---|---|---|---|
| 42 | 0.5667 | 0.5716 | **+0.0049** | [+0.0028, +0.0067] | 0.0020 | **YES** |
| 43 | 0.5736 | 0.5783 | **+0.0047** | [+0.0014, +0.0079] | 0.0033 | **YES** |
| 44 | 0.5686 | 0.5696 | +0.0010 | [−0.0026, +0.0046] | 0.0036 | no |
| 45 | 0.5624 | 0.5688 | **+0.0064** | [+0.0031, +0.0094] | 0.0031 | **YES** |
| **47** | 0.5675 | 0.5612 | **−0.0063** | **[−0.0094, −0.0028]** | 0.0033 | **YES** |

mean **+0.00213**, between-seed sd **0.00511**, signs `++++−`, **unanimous: NO**, median within-seed
MDE **0.00327**.

### h36

| seed | AP control | AP upctwo | ΔAP | 90% CI | MDE | resolved? |
|---|---|---|---|---|---|---|
| 42 | 0.3238 | 0.3203 | −0.0035 | [−0.0085, +0.0012] | 0.0049 | no |
| 43 | 0.3152 | 0.3186 | +0.0034 | [−0.0014, +0.0080] | 0.0047 | no |
| **44** | 0.3305 | 0.3183 | **−0.0122** | **[−0.0200, −0.0028]** | 0.0086 | **YES** |
| 45 | 0.3227 | 0.3176 | −0.0051 | [−0.0174, +0.0108] | 0.0141 | no |
| 47 | 0.3212 | 0.3177 | −0.0035 | [−0.0079, +0.0003] | 0.0041 | no |

mean **−0.00419**, sd **0.00556**, signs `−+−−−`, **unanimous: NO**, median MDE **0.00485**.

### 1.1 ⛔⛔ The two-seed story is overturned, and how it failed matters

The published two-seed result was *"+0.0049 and +0.0047, each resolved independently, agreeing to
4%"* — and that description was accurate. It was also **an accident of which two seeds ran first.**
Seed 47 is **resolved negative**: its interval excludes zero. Seeds 45 (+0.0064) and 47 (−0.0063)
both resolve and point **opposite ways**, so this is a genuine seed-dependent effect, not two noisy
measurements straddling zero.

⭐ **The pre-commitment held.** Before these numbers existed I wrote: *"if the five-pair mean at h1
falls below ~0.002, or any seed flips sign, 'replicated win' is no longer the right description and
I'll say so."* The mean is +0.00213 and a seed flipped **resolved**. ⇒ The claim is withdrawn.

⚠️ **The generalisable lesson, and it is not "use more seeds".** Two independent seeds agreeing to
4% felt like strong evidence and was worth almost nothing, because **n=2 cannot distinguish a real
effect from a pair drawn from the same side of a wide distribution.** The between-seed sd (0.00511)
is **1.6× the median within-seed MDE** (0.00327) — so each seed's own interval was *tighter* than
the spread between seeds, which is precisely the configuration in which per-seed resolution is
misleading. ⇒ **A resolved per-seed interval is not evidence of a reproducible effect.**

## 2 · The pre-committed consequences

### 2.1 ⛔⛔ The seed route is CLOSED at h1

`PREREG_EXP03_SEEDS.md` pre-commits that **`n_required > 20` ⇒ the seed route is CLOSED**, and §6's
MIXED branch orders the count to be **derived from the observed spread** rather than assumed. Done,
two-sided α = 0.05, 80% power:

| endpoint | mean | sd (4 dof, 90% CI) | **n_required** | CI on n | verdict |
|---|---|---|---|---|---|
| **h1** | +0.00213 | 0.00511 (0.00332–0.01213) | **45 seeds** | 19 – 255 | ⛔ **CLOSED** |
| **h36** | −0.00419 | 0.00556 (0.00361–0.01318) | **14 seeds** | 6 – 78 | ✅ OPEN |

⇒ Resolving the h1 gate effect by buying seeds would take **45 arms**. At the measured ~4 h and
~$2.60 per arm that is ~$120 and ~180 GPU-hours **for one endpoint of one arm**, and the
pre-registration already declared that out of bounds. ⭐ **h36 remains reachable at 14.**

### 2.2 ⛔ The sixth pair cannot change the verdict — do not buy it

§6's branches are:

| branch | condition |
|---|---|
| CONFIRMS | ΔAP > 0 at h1 **and** < 0 at h36, **every** seed |
| FALSIFIES | ΔAP **≥ 0 at h36 on every seed** |
| MIXED | seeds disagree at h36 |

Both CONFIRMS and FALSIFIES require **unanimity**, and resolved seeds have already broken it in both
directions — seed 47 resolved **negative at h1** (kills CONFIRMS), seed 44 resolved **negative at
h36** (kills FALSIFIES). **A sixth seed cannot un-resolve a resolved disagreement.**

⇒ The branch is **determined by the five pairs in hand**. ⛔ This **retracts** the recommendation I
gave earlier the same day, that the honest reason to buy the sixth pair was to keep the
pre-registration whole: that reason has evaporated. For σ alone the sixth pair moves the CI width
from **3.65× to 3.11×** (against **31.26×** at two pairs), which does not justify a pod.

⚠️ **Knock-on:** ledger row `O-5` — the guard relaxation behind views-models #697 — was written to
settle **on seed 46's re-run**. If the re-run does not happen, **`O-5` stays OPEN**, and #697 should
still be merged on its own correctness merits rather than for this experiment.

---

## 3 · The body: magnitude on cells that fire

`results/exp03_magnitude.out`. ⛔ Reported as a **quantile sweep, never as `size_ratio`**: that
statistic is a median and was measured to be **exactly invariant** to a factor-1000 error on up to
**49%** of event cells (M3 Finding C), which is the very quantity under discussion here.

### h1 — ΔE[Y] on TRUE event cells (upctwo − control), by quantile

| quantile | 42 | 43 | 44 | 45 | 47 | |
|---|---|---|---|---|---|---|
| q0.10 | 0 | 0 | 0 | 0 | 0 | both arms silent |
| q0.25 | 0 | 0 | 0 | 0 | 0 | both arms silent |
| q0.50 | +0.8125 | −0.3750 | +0.1250 | −0.2500 | +0.3125 | mixed |
| q0.75 | +1.6250 | +0.8125 | +0.6250 | −0.0000 | +0.3750 | **no seed negative** |
| **q0.90** | **+1.6250** | **+0.8750** | **+0.9375** | **+0.3125** | **+1.1250** | ⭐ **5/5 up** |
| **q0.99** | **+0.8925** | **+4.0675** | **+2.3075** | **+1.2000** | **+6.0775** | ⭐ **5/5 up** |

Control median E[Y] on those cells: **1.44 – 3.31**. **Truth median: 6.00.**

⇒ Two things at once. **(a)** The body under-forecasts true event cells by **2–4× at the median** —
the timidity the foundation measured as size-ratio ~0.02 is still there at global scale. **(b)** The
arm lifts the **upper tail** toward truth on **every seed**, with the effect **growing monotonically
with the quantile** and the lower half not moving at all (q0.10/q0.25 are exactly 0 in both arms,
i.e. both are silent there).

**Share of true event cells where all 16 draws are ZERO** — the body going silent on a real event:
control **0.3019–0.3883**, upctwo **0.2912–0.3432**; mean Δ **−0.0219**, sd 0.0344, 2/5 higher. ⚠️
Leaning toward fewer silent cells, **not** unanimous.

### h36 — the body has already collapsed

q0.50 is **+0.0000 on all five seeds**, and the reason is stark: **the control's median E[Y] on true
event cells is 0.0000** against a truth median of **5.0**. More than half of true event cells get a
forecast of exactly nothing. The tail Δ is **not** unanimous (q0.90: +1.9125, −0.7250, +0.5000,
−1.3750, +0.1250). **Share silent on an event rises to 0.5627–0.6650.**

⇒ **There is nothing at h36 for this intervention to improve.** The failure there is not placement
or shape, it is that the body has faded to zero.

---

## 4 · The body: is it less SMEARED? — the question the arm was actually for

`results/exp03_sharpness.out`. This is what `UpsampleConv2` was adopted for: §3's magnitude is a
**proxy** for sharpness and cannot distinguish "same blur, scaled up" from "tighter placement".

**Instrument — B-1, the structure share**, mirroring
`reports/2026-10-26_body_optimisation_gap/c366_two_instruments.py::b1`:

```
B1(d) = [ gamma_pred(d) / gamma_truth(d) ] / [ var_pred / var_truth ]
```

`gamma(d)` is the method-of-moments semivariogram at E–W lag `d`. **B1 = 1.0 means the field carries
its full due share of short-range variance; B1 < 1 means SMEARED.** Swept over `d`, ⛔ never a
scalar (D-4) — the withdrawn "12.8× correlation length" extrapolated from a single lag.

⛔ **Why the committed `gamma_ew` is NOT imported.** It averages over every raster pair. On the
global 360×720 grid **75% of cells are ocean** (65,818 of 259,200 are land) and structurally zero in
**both** fields, so those pairs contribute 0 to numerator and denominator alike and drag every ratio
toward 1.0 — attenuating exactly the signal being measured. The land-masked form restricts to
land–land pairs and takes variance over land cells only. The committed script prints the land
fraction so the size of that effect is visible rather than asserted.

⛔ **No hardcoded grid offsets.** `sharpness_scorecard.py:34` pins `87/310/180`, which are
**Africa-only** and on a global arm would silently mis-place every cell. Row/col come from the
pull's **own** `row`/`col` columns (offset 1, as the config declares) and both are asserted on-grid.

### 4.1 G-0 — the instrument on fields whose answer is known by hand

| lag | B1(truth, truth) | B1(3×3-smoothed truth, truth) |
|---|---|---|
| 1 | 1.000000000000 | **0.2532** |
| 2 | 1.000000000000 | 0.4523 |
| 4 | 1.000000000000 | 0.9080 |
| 8 | 1.000000000000 | 1.0045 |
| 16 | 1.000000000000 | 1.0885 |
| 32 | 1.000000000000 | 1.1043 |

All four gates PASS: identity exactly 1.0 at every lag; the blur reads deeply smeared at d=1
(< 0.5); the blur signature **rises monotonically** toward 1.0 as the lag outruns the kernel; and
the blur has essentially no effect by d=32.

⭐⭐ **This handed us a built-in NEGATIVE CONTROL.** A 3-cell blur provably cannot change structure
at lag 32 — two cells 32 apart are uncorrelated either way. ⇒ **d=16 and d=32 cannot be moved by a
sharpness change.** Movement there would mean a scale/amplitude change masquerading as sharpening.
The discrimination was specified before the arms were scored.

### 4.2 Result — h1

| lag | control (mean of 5) | upctwo | Δ | per-seed signs | unanimous |
|---|---|---|---|---|---|
| **d=1** | **0.5648** | **0.6152** | **+0.0504** | **+++++** | ⭐ **YES** |
| d=2 | 0.6686 | 0.6662 | −0.0025 | −++++ | no |
| d=4 | 1.1742 | 1.1575 | −0.0167 | −+++− | no |
| d=8 | 1.2783 | 1.2490 | −0.0293 | −+++− | no |
| d=16 ⟵ *neg. control* | 1.3893 | 1.3563 | −0.0330 | −+++− | no |
| d=32 ⟵ *neg. control* | 1.3787 | 1.3365 | −0.0422 | −+−+− | no |

⭐⭐ **Unanimous and positive at d=1, and nowhere else** — including at the two lags a sharpening
physically cannot reach. That is the signature of **genuine sharpening**, not rescaling. Size:
+0.0504 against a deficit of 0.4352, i.e. it closes **11.6% of the gap to full short-range
structure** (a **+8.9% relative** gain) from one layer swap.

### 4.3 Result — h36: nothing

d=1 Δ **−0.0006**, signs `−+−−+`; no unanimity at any lag. Consistent with §3: there is no body left
to sharpen.

### 4.4 ⭐ A structural finding that is bigger than the arm

The body is **smeared at d=1** (0.5648) and **over-dispersed from d=4 outward** (1.17 → 1.39 at h1;
1.61 at d=32 for h36). ⇒ **It is too smooth locally AND too incoherent regionally** — it fails to
reproduce the long-range clustering real conflict has. The earlier single figure (**0.391** at 1
cell, on 32×32 windows) showed only the near end of that curve.

⚠️ **0.565 and 0.391 are NOT a like-for-like comparison** — different population (global land vs
32×32 curriculum windows), different masking, different vehicle. The reproducible claim is the
**two-sided shape**, not a change in the number.

---

## 5 · ⭐⭐⭐ The instrument finding: FAO-02's primary score is a ~100-cell measurement

`results/exp03_crps_concentration.out`. Prompted by Simon's objection, in his words: *"manipulated
nature of the data makes enormous CRPS very hard to use."* That is a claim about the instrument and
it is **measurable rather than arguable**.

⛔ **This does not propose a replacement for a locked score.** FAO-02 locks CRPS as primary with
QS99/Brier/MCR as guardrails, and twCRPS/LogScore/PIT are explicitly **rejected**. What follows is a
**decomposition** of the locked primary score by truth magnitude, which is permitted where a new
score is not. Bands were fixed **before** any arm was compared.

### 5.1 Where the score lives — control, a property of the DATA not the arm

**h36:**

| truth band | n cells | % of cells | CRPS/cell | **% of TOTAL CRPS** |
|---|---|---|---|---|
| 1–2 | 1,082 | 0.128% | 1.381 | 0.83% |
| 3–5 | 719 | 0.085% | 3.537 | 1.42% |
| 6–20 | 871 | 0.103% | 9.644 | 4.68% |
| 21–100 | 517 | 0.061% | 38.730 | 11.16% |
| **101+** | **118** | **0.014%** | **1,234.282** | **81.16%** |

**h1:**

| truth band | n cells | % of cells | CRPS/cell | **% of TOTAL CRPS** |
|---|---|---|---|---|
| 1–2 | 843 | 0.100% | 1.430 | 2.76% |
| 3–5 | 549 | 0.065% | 3.260 | 4.10% |
| 6–20 | 899 | 0.107% | 8.289 | 17.06% |
| 21–100 | 563 | 0.067% | 33.079 | 42.64% |
| **101+** | **51** | **0.006%** | **253.833** | **29.64%** |

⇒ ⭐⭐⭐ **At h36, 118 cells — 0.014% of 842,634 — carry 81% of the score.** At h1, 614 cells
(0.073%) carry 72%. **CRPS on this population is, to first order, a measurement of ~100 cells.**

**The mechanism is not mysterious:** CRPS is in **count units**, and truth is heavy-tailed (max
2,482). CRPS/cell therefore runs 1.4 → 1,234 across the bands, so the pooled mean is a
magnitude-weighted average dominated by the largest cells.

⚠️ **A correction to the folk belief in this programme.** The usual worry is that the 99.66% zero
cells swamp CRPS. **They do not.** `crps_none` is **0.0020–0.0034** against `crps_events` of
**13.2–14.5**, so event cells are 0.34% of the population and **96.2%** of `crps_all` at h1. The
pooled score is dominated by events — and *within* events, by the largest hundred.

### 5.2 ⇒ What this does to every null in this document

A dead-flat `crps_all` **cannot** refute a structural change spread over hundreds of thousands of
cells, because the instrument's resolution is set by ~100 extreme cells whose own CRPS/cell is
1,234 and whose between-seed sd is correspondingly huge. ⇒ **§3's and §4's nulls on CRPS are not
evidence of no effect; they are evidence the score cannot see one.**

⚠️ **And this is NOT specific to this arm.** It applies to **every CRPS number on the scoreboard**.
It joins two sibling instrument findings: FAO-02's 1% MCR margin sitting **84×** under seed noise
(RP-2b), and the battery's gate leg running on **255 distinct values** at S=16 (M3b).

### 5.3 The banded trade the pooled score hides

Δ CRPS/cell, upctwo − control (lower = better), 5 seeds:

**h1**

| band | control | upctwo | mean Δ | sd | signs |
|---|---|---|---|---|---|
| 1–2 | 1.506 | 1.522 | **+0.016** worse | 0.029 | `+-+++` **4/5 worse** |
| 3–5 | 3.306 | 3.337 | **+0.031** worse | 0.068 | `-++++` **4/5 worse** |
| 6–20 | 7.982 | 7.984 | +0.002 worse | 0.297 | `-++++` **4/5 worse** |
| 21–100 | 31.190 | 30.955 | **−0.236** better | 0.808 | `-+-++` 3/5 better |
| 101+ | 247.573 | 247.028 | **−0.544** better | 1.435 | `-+-+-` mixed |

**h36** shows the same shape: `6–20` **+0.023** worse (`-++++`, 4/5), `21–100` **−0.188** better,
`101+` **−0.517** better.

⇒ The arm makes **small-event cells slightly worse and large-event cells slightly better** —
coherent with §3's unanimous upper-tail lift. Pooled CRPS weights by magnitude, so it nets the trade
to approximately zero.

⚠️ ⛔ **Every band effect is ~1% or less of its band's level, with a between-seed sd comparable to or
larger than the effect.** The sign consistency in the three small bands (4/5, three bands running)
is **suggestive**; it is **not an interval**. This is a **LEAD, not a result**, and §1.1 is the
standing reason not to treat 4/5 as resolved.

### 5.4 QS99 — the licensed tail guardrail

| | control | upctwo | Δ | signs | |
|---|---|---|---|---|---|
| h1, all cells | 0.0269 | 0.0267 | −0.0002 | `-+++-` | not unanimous |
| h1, event cells | 7.4347 | 7.3171 | **−0.1176** (−1.6%) | `-+++-` | not unanimous |
| h36, all cells | 0.1963 | 0.1956 | −0.0007 | `-+-+-` | not unanimous |
| h36, event cells | 49.7816 | 49.5789 | −0.2027 | `-+-+-` | not unanimous |

Consistently slightly better in the mean, never unanimous. ⇒ Null.

⚠️ **Two limits on QS99, stated rather than buried.** **(a)** With **S=16** draws the 0.99 quantile
*is* essentially the top member, so this is a statement about the ensemble **maximum**, not a
well-estimated 99th percentile. **(b)** M5/#482 records that QS99 as `t0_score.py` computes it has
**never been reconciled** with FAO-02's guardrail wording — so it is usable as a within-experiment
contrast, not as a compliance number.

---

## 6 · Can diffuseness be penalised in training? — asked by Simon, answered from the repo

The idea: measure sharpness at step 1 and penalise later autoregressive steps for being less sharp,
one-sided and relative to step 1 so that sharpness is not demanded for its own sake.

### 6.1 ⛔ We already built this. It ran as an arm. It made the real diagnostic 4.44× WORSE.

`training_engine.py:402-476` already carries `_structure_penalty`: for `k ∈ (2, 8)`, the **one-sided
log shortfall of within-`k×k`-block variance** of the composed forecast against the observed counts,

```
sum_k mean( clamp( log(var_true + 1e-6) - log(var_pred + 1e-6), min=0 ) )
```

On the structure-penalty arm the **penalty improved 4.87×** (1.7223 → 0.3540) while **B-1 at d=1
worsened 4.44×** (0.400 → 0.090). See `reports/2026-10-21_structure_penalty_arm_prereg.md` and
`reports/2026-10-26_body_optimisation_gap/c366_two_instruments.py` (C-366).

### 6.2 ⭐ C-366 derived *why*, and the three reasons are the design spec for anything better

| derived property of the penalty | consequence |
|---|---|
| **H1 — homogeneous in amplitude.** Scaling the forecast by `c` scales every block variance by `c²` | ⛔ **satisfiable by scaling up**, with nothing rearranged |
| **H3 — exactly invariant to permuting cells WITHIN each block** (a variance depends only on a multiset; a within-2×2 permutation also leaves every 8×8 multiset unchanged) | ⛔ a "sharpness" loss that **cannot see where mass sits** |
| **H4 — unstructured noise IMPROVES it** (`var_pred` rises, the shortfall shrinks) | ⛔ **rewards noise** |

And **H2**: B-1 is **exactly invariant** to that same amplitude scaling, because γ and var both
scale by `c²` and the ratio cancels.

⇒ ⭐⭐ **The spec for a diffuseness penalty is: amplitude-INVARIANT and arrangement-SENSITIVE.**
B-1 is both (H2, and it collapses under the within-block permutation of H3). The block-variance
shortfall is **neither**. **A B-1-shaped penalty is the right shape; a variance-shortfall penalty is
provably the wrong one**, and the arm's 4.44× regression is that derivation showing up in training.

### 6.3 ⭐⭐ The relative-to-step-1 framing is well-founded — and stronger than argued

Simon's stated reason for anchoring to step 1 was that we should not demand sharpness for its own
sake. But **the target has no horizon**: truth at the h36 month is just another calendar month with
the same spatial statistics as the h1 month. The forecast's horizon is a property of the *forecast*,
not of the field being predicted.

⇒ **Structure share SHOULD be horizon-invariant**, and any decay with step is pure model pathology
rather than a property of the world. So the absolute target is defensible and the step-1 reference
is not needed to avoid over-sharpening. ⚠️ The step-1 anchor remains **useful** as a
variance-reduction device (it differences out whatever the model's own baseline sharpness is), but
it is not required for correctness.

### 6.4 ⚠️ But the measurement says this would target the wrong defect

| | h1 | h36 |
|---|---|---|
| B-1 structure share at d=1 (control) | 0.5648 | **0.5396** |
| median E[Y] on true event cells (control) | 1.81 | **0.0000** |
| share of true event cells fully silent | ~0.34 | **~0.60** |

⇒ **The body's spatial structure barely decays with horizon; its amplitude collapses to nothing.**
The autoregressive pathology measured here is **FADE, not SMEAR**. A diffuseness penalty aimed at
progressive smearing would be aimed at a property that is roughly horizon-stable.

⚠️ **The honest caveat on that conclusion: it rests on two endpoints.** h1 and h36 are the only
horizons measured, because §6 device 2 pins the endpoints — but they cannot distinguish "flat" from
"decays and recovers" or from "collapses after h12". ⇒ **The owed measurement is a per-step profile
of BOTH structure share and amplitude across all 36 steps**, which the delivered draws already
support (they carry every horizon, not just the endpoints). If structure is flat and amplitude
decays, the lever is fade (`reports/2026-09-01_fade_mechanism_dossier`,
`reports/2026-09-02_silence_vs_fade_dossier`). If structure degrades past h12 and the endpoints hid
it, the penalty is right and §6.2 gives its shape.

ℹ️ **Literature: NOT checked.** I do not know what the library holds on sharpness-penalised
autoregressive forecasting and have not looked. `/library search` is the instrument and it has not
been run. ⛔ Nothing in §6 should be read as a literature claim.

---

## 7 · What this document does NOT establish

- ⛔ **Not a §6 verdict.** Five pairs cannot satisfy "all six, strictly" (AMENDMENT 1).
- ⛔ **No h18 number**, in either direction (device 2).
- ⛔ **No comparison to EXP-01's absolute numbers.** `P-30`'s refetch moved the population 4.944×
  and voided that cross-reference (`W-122`). Every comparison here is **within** this experiment.
- ⛔ **Nothing about gate calibration.** AP is rank-based and **exactly invariant** to monotone
  miscalibration — `p → (p+1)/2` moves Brier by 0.249 and AP by **0.000000** (M3 Finding B). "ΔAP
  +0.005" means the *ranking* changed and says nothing about calibration.
- ⛔ **No intervals on the body results.** §3 and §4's headline claims are **sign counts across five
  seeds**. §1.1 is the standing demonstration that unanimity without an interval can mislead. ⇒ A
  paired bootstrap on the d=1 structure share and on the banded CRPS is **owed before either number
  is quoted outside this dossier**.
- ⛔ **`P-22`, `P-25`, `P-26` remain unread.** They are in the same pre-registration; leaving them
  unread while reporting `P-23` would be selective reporting (the plan's own DoD line 9).
- ⚠️ **One target only** (`sb`), one composition, one vehicle family.

## 8 · The missing twelfth arm

`upctwofullzero_fortysix` trained to completion (300/300, 203 min) and emitted all 13 origins, then
was refused at the collapse stage: its largest `pred_lr_os_best` was **11.31** against
`_check_scale`'s `MIN_PLAUSIBLE_MAX = 12.0`, whose premise is that a value that small suggests the
frame is still in log1p space.

**The floor was in the wrong place, derivably.** A frame genuinely in log1p space cannot exceed
`log1p(2482) = 7.817`; the refused value implies `expm1(11.31) = 81,633` counts, which no cell
reaches. ⇒ The band (≈8, 12) is a refusal region in which the hazard being tested for is
**impossible**. views-models **PR #697** moves the floor to **9.0**, derived from that and nothing
else; the verdict-neutrality test (the worst already-verified arm was **13.69**) shows it cannot
rescue or re-score any existing result. Logged as **`O-5`** in `reports/PREDICTION_LEDGER.md`
**before** any re-run, per §6 device 3.

⛔ **The arm's artifact was destroyed with its pod before the raw `origin_5` values were read**, so
the claim that 11.31 was a legitimately timid count-space forecast rests on a derivation plus eight
sibling arms and **not** on a measurement of the arm in question. ⇒ Per §2.2 the re-run is no longer
worth buying, which means **`O-5` stays open** and this question stays unsettled.

## 9 · ⚠️ Process failures in the session that produced this document

Recorded because a finding's reliability depends on how it was reached, and because two of these
could have silently corrupted the numbers above.

1. ⛔⛔ **`grep` in this shell silently under-reports.** It is a shell function wrapping ugrep with
   `--ignore-files`; `reports/` is gitignored (dossier files are force-added, so they *are* tracked,
   but ugrep honours the ignore file rather than the index). Measured: `grep -rn '28 min/origin' .`
   → **1 hit**; `command grep -rn` → **15 hits across 10 files**. ⇒ Every recursive search over
   `reports/` in this session under-reported, and **an absence proved nothing**. Use
   `command grep -r`.
2. ⛔ **Three gate-specification errors of my own, all caught by the gates.** A first truth lookup
   built a 24.9M-key Python dict and ran for 40 min before being killed (the vectorised form takes
   **1.9 s**). The B-1 G-0 first demanded the blur signature be `< 1` at **every** lag, which is
   wrong — a 3-cell kernel cannot reach lag 32 — and then asserted monotone **decrease** while the
   prose beside it correctly said "rises toward 1.0". ⚠️ In each case the **instrument** was right
   and the **gate** was wrong; the fix narrowed the gate to where the claim holds, which made it
   stronger, rather than widening a tolerance.
3. ⛔ **A hardcoded per-arm filename.** The `_lr.tar.zst` draws archive carries each arm's own run
   timestamp; spelling one arm's name made the script die on the second arm. Now globbed, with an
   assert that exactly one archive exists.
4. ⚠️ **Four of five readouts in this session were the GATE** on an experiment that was never about
   the gate, until Simon asked directly for magnitude. The gate readout is cheap (the delivered
   archive carries `time`/`unit` in band, needing no alignment) and the body needs real work — which
   is availability driving reporting, not reasoning.
5. ⛔⛔ **Unrelated to this document but part of the same session: 698 directories of diagnostic
   plot output under `reports/plots/diagnostics/` were destroyed** by a `find -maxdepth 1 -type d
   -newermt ... -exec rm -rf {} +` that matched the **parent** directory, whose mtime was current
   because that day's runs had written into it. Unrecoverable (ext4, no snapshot, never tracked).
   Nothing in this dossier depended on them.

## 10 · ⇒ Consequences

1. ⛔ **The h1 gate win is withdrawn.** The two-seed result was not reproducible.
2. ⛔ **The seed route is CLOSED at h1** (45 seeds needed, >20 pre-committed as closed). **h36 stays
   open at 14.**
3. ⛔ **Do not buy the sixth pair.** It cannot change the branch. `O-5` stays open as a result.
4. ✅ **The arm does sharpen the body at d=1 and does lift the event-cell tail, on all five seeds** —
   ⚠️ as sign counts, pending intervals.
5. ⭐⭐ **Report CRPS banded by truth magnitude alongside the pooled number, from here on.** It is a
   decomposition of the locked primary score, not a new score, and it would have shown §5.3's trade
   immediately instead of a null. **Owed across the whole scoreboard, not just this arm.**
6. ⭐ **A diffuseness penalty must be amplitude-invariant and arrangement-sensitive.** The existing
   `_structure_penalty` is neither and its arm regressed B-1 4.44×. **Do not re-use that shape.**
7. ⭐ **Next measurement, and it is cheap:** the per-step profile of structure share and amplitude
   across all 36 horizons, from draws already on disk. It decides whether the autoregressive lever
   is **fade** or **smear**, and §6.4's two-endpoint reading cannot.
8. **`P-22`, `P-25`, `P-26` are owed** regardless of any of the above.
