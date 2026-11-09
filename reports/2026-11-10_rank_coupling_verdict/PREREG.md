# PREREG — locate λ*: how good must the body's ranking be before the rank-coupled empirical draw is measurable at all?

**Epic #579 · Story S1 #580.** ⛔ **Committed before any tool in this dossier exists** (`D-5`).
`tools/` is absent at this commit; `git show --stat HEAD` shows exactly this one file.

Template: `reports/2026-11-04_delivered_gate_lookup/PREREG.md`, the strongest pre-registration in
the repo.

---

## 0 ⛔ WHY THIS DOCUMENT EXISTS, AND WHY IT IS FIRST

**A** (today) emits `gate × magnitude body`. **B** (proposed) emits
`gate × ranking body × empirical draw` — the body says *which cell is bigger*, and the magnitude
comes from the distribution of what actually happened at that rank.

B **has already been built and run once**, in `reports/2026-11-04_rank_coupled_body/tools/sim_rank_coupling.py`
(`R = 400`, `SEED = 0`, `LAMS = (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)` at `:79-84`; 19.9 s CPU).
Its committed result:

| arm | ordering | Spearman vs truth | gap vs its own `lam=0` control |
|---|---|---|---|
| `lam=0.00` | noise — **the no-op control** | — | 0 by construction |
| ⛔ **`REAL`** | the recovered regression output | **0.5125** | **−0.0137**, CI **[−0.1138, 0.0826]** ⛔ spans zero |
| `lam=0.75` | partially informative | 0.68 | ⛔ spans zero |
| `lam=1.00` | oracle | ~1.0 | ✅ **+0.4724**, excludes zero |

⛔⛔ **That output reads as "B does not work" and is indistinguishable from "the test cannot see
anything at our ranking quality."** The oracle separates, so the instrument has power. Nothing at
Spearman ≤ 0.68 separates, so the readability threshold sits **between 0.68 and 1.0**. Our body
ranks at **0.5125**. ⇒ The idea is currently **unfalsifiable**, and it was being read as falsified.

⚠️ **Four previous definitions of done were withdrawn**, each because it was written or revised
*after* a number was seen. That is the only reason this story is first and is committed alone.

---

## 1 THE POPULATION SLOTS, ALL NAMED (`C-351`)

| slot | value |
|---|---|
| **population** | the **gate-open** set over score months **463–469** (7 folds, 6-month rolling fit windows), **64,818** global `land` cells. ⛔ The open-cell count is **printed per fold, not assumed** |
| **truth** | `lr_sb_best`, **raw counts, no transform.** ⚠️ The `lr_` prefix does **not** mean transformed — `e1_data.py:16` asserts raw counts and `e1_data.py:160,174` read the column directly. **Verified in-run by printing the column maximum** |
| **estimator** | sd with **`ddof=1`, declared on every line that prints one.** CRPS printed **both** ways, fair and plug-in, with the estimator named on every line (`GLOSSARY.md` §5e) |
| **inference mode** | ⭐ **DELIVERED AUTOREGRESSIVE, h = 1.** No teacher-forced number is produced, in either direction |
| **arms** | **A** = the delivered emission. **B** = the rank-coupled empirical draw on the **same** cube. σ̂ is measured across the **6 control-form arms** (`calm/keen/mellow/nimble/quiet/rapid_orbiter`) — ⚠️ **6, not 5**: `quiet_lander` is a ghost directory with no artifacts, which removes a *lander*, not an orbiter. Verified on disk: all 6 orbiters carry `gate_endpoints.tar.zst` and exactly one `*_lr.tar.zst` |
| **the one varying thing** | ⭐ **the magnitude assignment.** Same gate, same open set, same bin edges, same truth, same cells, same `n`, same `S` |
| **swept, and reported whole** | λ (densified over 0.68–1.00) × ranker ∈ {recovered regression output, gate} × scale ∈ {raw, log1p} × `R` |

### ⚠️ Pre-declared limits of this population

- `S = 16` draws. The emission issues a **literal point mass at zero on 99.4569%** of h1 cells, so
  its CRPS there is exactly the truth value. ⇒ The background and event halves are reported
  **separately, with `n` beside every figure.**
- A's own comparator, re-measured, is `e1_baseline.out`: pooled **fair-crps 0.047390** over
  **453,726** cell-months, **1,572** events. ⛔ This is **not** `M89`'s number — different
  population, different checkpoint.
- ⛔ **Disk is 98% full (21 GB free).** Any instrument must stream one origin at a time, as
  `e1_data.load_draws_h1` already does (149 MB per `y_pred.npy`).

---

## 2 THE SINGLE SUPERIORITY CRITERION

> **The paired per-bin conditional magnitude-calibration difference between B and A, on the shared
> gate-open set. Headline bin = the lowest. 16 bins.**

Declared **ex ante** (`C-3531` `Gneiting2011`: post-hoc scoring-rule selection is inadmissible).

**Why conditional and never marginal:** B draws from the empirical marginal, so it is **marginally
calibrated by construction.** A pooled magnitude check is a no-op wearing a guardrail's clothes.

**Why not CRPS as the decider:** it ranks an **empty map top** — `size_ratio` 0.0000 cleared the
5% bar with **74.6%** of the win coming from confident zeros. CRPS is a report-only guardrail here.

⛔ **The blocker this is designed around.** `C-52` `Brocker2009`: *"It is impossible to say how
proper scores will rank unreliable forecast schemes even when one is sufficient for the other."*
A is miscalibrated (pooled `obs/pred` **1.2349–1.6848** across ten arms). ⇒ A proper-score
comparison against A **is not interpretable marginally.** Conditioning is the design response, not
a preference.

### ⚠️ A known limit, stated now rather than discovered later

The conditioning index is the mean of **16 draws**, so it is **quantised at 1/16 = 0.0625.** ⇒ A
sweep beyond 16 bins **cannot** probe the bottom bin — the bins would be finer than the index.
16 bins is the ceiling, and it is a property of `S`, not a choice.

---

## 3 THE MARGIN

```
PROMOTE B  ⟺  ( δ > 2 · σ̂_paired  on the superiority criterion )
               ∧ ( no guardrail moved against B )
```

⛔ **`δ` is in units of MEASURED seed dispersion, never percent** (`C-392`: seed noise on
`MCR_field` is **84×** FAO-02's 1% margin).

⛔⛔ **σ̂ is measured IN THIS RUN (S7 #586), at `ddof=1`, and printed BEFORE any A-vs-B number.**
**No borrowed σ̂.** The only σ̂ table available to us measures `truncated_nb` at dropout 0.15 with
expectation composition at **S=256**, against delivered `nb` at **S=16**, and **never states its
inference mode.** It does not apply and is not used.

⭐ **A B that changes nothing scores zero on the superiority criterion and passes every guardrail,
so it fails automatically.** That property is the point.

---

## 4 GUARDRAILS — ⚠️ ALL REPORT-ONLY

| guardrail | why report-only |
|---|---|
| **gate AP** | ⛔ B does not touch the gate. **Any movement at all is a VOID signal, not a result** |
| **fair CRPS, plug-in CRPS** | §2: ranks an empty map top; and `C-52` blocks it against a miscalibrated A |
| **variogram score** | spatial, not magnitude-conditional |
| **pooled `obs/pred`** | marginal ⇒ B passes by construction |
| **Spearman, `DSC_ORD`, `R*`, any rank metric** | ⛔ **B's own training objective.** §8 row 16 |
| **`MCR_field`** | pooled |
| **`pos_mcr`, `size_ratio`** | ⛔ both **select cells by the outcome** |

⛔ **No guardrail may be promoted to the decision number after the numbers exist.**

---

## 5 THE SIX VOID TRIGGERS

VOID is a **third outcome** — *the instrument could not answer* — and is ⛔ **never** reported as
"B is worse".

| # | trigger |
|---|---|
| **V1** | the `ap_ctl` anchor fails to reproduce `0.5667056384496834` to **1e-12** ⇒ the artifact chain is broken; **nothing is reported** |
| **V2** | A and B scored on different cells, different `n`, or different `S` |
| **V3** | ⭐ the **no-op control (λ=0) does not read zero** ⇒ the criterion is broken, not the arm |
| **V4** | ⭐ the **oracle control (λ=1) does not separate** ⇒ no power; we report that |
| **V5** | no scoreable artifact. ⛔ **A crash is never evidence** |
| **V6** | ⭐⭐ **λ\* exceeds the real ordering (0.5125)** — declared **in advance** as **answer 3** below. This is a **result**, not a failure |

---

## 6 ⛔ THE VERDICT RULE — ONE OF THREE, EACH OF WHICH ENDS THE QUESTION

| # | the answer | what we would say | what happens next |
|---|---|---|---|
| **1** | ⭐ **Yes — build it** | Replacing the body's magnitudes with an empirical draw improves magnitude calibration in the lowest prediction bin by **more than 2 σ̂**, and nothing else got worse | B becomes real work |
| **2** | ⛔ **No — and this is why** | It does not. **And we know the test could have seen an improvement, because the same test sees one when handed a perfect ranking** | **B is dead, properly** |
| **3** | ⚠️ **Not yet — and here is what's missing** | The test cannot see anything until the body ranks event cells at Spearman ≈ **λ\***. Ours ranks at **0.5125**. So the question is not the draw, it is **the ranking**, and the gap is λ\* − 0.5125 | ⭐ the question **moves**, with a target |

⛔ **Answer 3 is the one every previous attempt could not give.** Without it, a test too weak to see
anything reads identically to "B does not work", and the idea dies for the wrong reason — which is
exactly what happened.

**Why answer 3 is trustworthy:** the λ dial has a **known right answer at both ends.** At λ=0 the
ordering is noise, so the test *must* read zero. At λ=1 it is perfect, so the test *must* see it.
**Both ends already behave correctly** in the committed run. That is what licenses the middle.

---

## 7 PREDICTIONS, REGISTERED BEFORE THE RUN

⚠️ My base rate on this codebase: `reports/PREDICTION_LEDGER.md`, **WRONG 137 / RIGHT 89.**

| # | prediction |
|---|---|
| **P-1** | B does **not** clear 2 σ̂ at the real head's ordering |
| **P-2** | λ\* lands at Spearman **0.80–0.95**, **above** the real ordering of 0.5125 ⇒ **answer 3** |
| **P-3** | the recovered-regression ranker and the gate ranker are **not separated** |

---

## 8 ⛔ FOUR ANTI-FISHING DEVICES

1. ⛔ **A teacher-forced-only improvement is a FAIL, declared in advance.** Only the delivered
   autoregressive cube counts. ⭐ The oracle (λ=1) is a **calibration axis, not an arm**: no number
   built on truth may be reported as a statement about the model.
2. ⛔ **If the result is negative and the test is then argued to have been inadequate, that argument
   is logged as a `W-` row in `reports/PREDICTION_LEDGER.md` FIRST, before it is made.**
3. ⛔ **The sweep is reported whole.** Quoting the best cell of a λ × ranker × scale grid without
   the grid beside it is **not a result**.
4. ⛔ **No threshold in this document may change after the numbers exist.** A change is an
   **amendment**, appended and dated, never an edit.

---

## 9 ⛔ WHAT IS NOT EVIDENCE HERE

> **Nothing counts unless the code AND its inputs are both on disk and re-runnable.**
> A `.out` file is a note. A dossier claim is a note. A register entry is a note.

| struck | why |
|---|---|
| ⛔⛔ `mu` orders magnitude (`R*` 0.1386); `mu` vs `gate` p 0.3950; `G-ALIGN` passes | **`d4_result_CTLFULLZERO_FORTYTWO.npz` does not exist** — searched `/`. The 29 `d4_result*.npz` on disk are the 2026-10-12 vehicle, **not a delivered arm.** ⇒ `mu` is **unavailable for every delivered arm**, and `results/mu_ordering.out` has no input artifact behind it |
| ⛔ σ̂ 0.840477 / 0.759371 / 0.271747; CV 0.237–0.356; the 84× and 4.97 σ̂ bars | §3: wrong family, wrong dropout, wrong composition, wrong `S`, inference mode never stated |
| every `M-`, `C-`, and `§8` row quoted above, and the 74.6% figure | notes. Kept out of every load-bearing position; used only to explain *why* a device exists |

⚠️ **`τ = 0.20` is an unverified transcription.** It is a module constant at
`sim_rank_coupling.py:79` and a docstring sentence at `e1_data.py:26`. Nothing reads it from an arm
config. ⇒ **S4 #583 reads it from the arm config and asserts it.** Until then no τ-dependent number
is load-bearing.

⚠️ **The sim's gate is not A's gate.** `sim_rank_coupling.py:138,141` draw **Bernoulli(π)**, while
A ships **`threshold_gate`**. The sim *does* use τ for the open set (`:185 open_ = pi >= TAU`), so
the discrepancy is in the **firing mechanism**, not the open-set definition. ⇒ "same gate, one
varying thing" **is not currently true**, and S4 #583 is what makes it true.

---

## 10 SCOPE

**IN:** offline re-run on committed cubes; fix the `ranks FIXED` ablation; extend 1 → 7 folds;
densify λ; the conditional magnitude readout; σ̂ in-run; the three-outcome verdict.

**OUT:** ⛔ production code (`views_hydranet/distributions/sampling.py:98-105` is where an
in-production rank-coupled emitter would go; changing it breaks 7 byte-exact goldens plus the h=1
feedback-invariance guard — **and it is not needed to answer the question**); training; GPU;
re-emitting `mu`; the **gate-conditioned** empirical lookup (that is **epic #559 / #565 / #566** —
a different index and a CRPS verdict); archiving the notes pile.

⛔ **Total: zero GPU, zero training, no production code, ~30 min CPU.**

---

## AMENDMENT A-1 — 2026-11-10 — the ratio criterion is UNDEFINED on the headline bin

⛔ **Appended, not edited.** §8 device 4 requires that a threshold which changes after the numbers
exist be recorded as a dated amendment rather than a revision. Nothing above this line is altered.

### What was measured

§2 fixed the superiority criterion as the per-bin **ratio** `mean_obs / mean_pred`, headline bin =
the lowest of 16. Measured on the shared gate-open set, 2,256 cell-months over 7 folds
(`results/s6_conditional_readout.out`):

| | |
|---|---|
| bin 0, `n` | **128** cell-months (5.6738% of the open set) |
| `pred_A` in bin 0 | ⛔ **exactly 0**, on all 128 cells — `max == 0`, verified bit-exact |
| cells in the whole open set with `pred_A == 0` | **128** — i.e. bin 0 **is** that set |
| events in bin 0 | 32 (25.0%), mean observed **3.8359** |

⇒ **Bin 0 is precisely the set of cells on which A issues a literal point mass at zero.** The ratio
divides by zero there, so `err_A` is `nan` and the pre-registered criterion **cannot be evaluated
on the bin the pre-registration nominated as the headline.**

⚠️ This is the same degeneracy that closed `M89` — a quantity that does not vary cannot be
conditioned on — arriving from the other side: here it is the *incumbent's* index that is
degenerate, not the candidate's.

### The amendment

> The superiority criterion becomes the within-bin **absolute conditional bias**,
> **`|mean_obs − mean_pred|`**, lower being better, with
> **`δ = bias_A − bias_B`** (positive ⇒ B better calibrated). Headline bin, bin count, population,
> open set, margin rule (`δ > 2·σ̂_paired`) and every VOID trigger are **unchanged**.

**Why this one and not another.** It is the ratio criterion's own numerator: the same quantity,
undivided. It is defined at `mean_pred = 0`, it keeps the criterion conditional and in count space,
and it measures within-bin conditional bias — the `MCB` component of `C-3572`
`Gneiting2023_ConditionalCalibration`. ⛔ It was **not** selected by looking at which arm it
favours; §9 below records what it does to the verdict, including against B.

**The ratio is retained as a report-only column** wherever it is defined (bins 1–15), so the
amendment adds a statistic and removes none.

### ⚠️ What this amendment does NOT license

- ⛔ It does not move the headline bin. Bin 0 remains the headline **because** it is where A is
  most degenerate, which is exactly where an empirical draw has the most room to help.
- ⛔ It does not change the margin. `δ` is still compared against `2·σ̂_paired`, measured in-run.
- ⛔ It does not retire `P-1`–`P-3`. They were registered against the mechanism, not the statistic.

---

## AMENDMENT A-2 — 2026-11-10 — the λ dial does not move the quantity λ* is stated in

⛔ **Appended, not edited.** §8 device 4. Nothing above is altered.

### What was measured

§1 lists `λ (densified 0.68–1.00)` as the sweep axis and §6 states λ\* in **Spearman vs truth on
event cells** (`rho_event`). The dial is inherited from `sim_rank_coupling.py:181`:

```
zt = (np.argsort(np.argsort(truth + jitter)) / truth.size - 0.5) * 2
key = lam * zt + (1 - lam) * noise / noise.std()
```

⛔ `zt` is a **global** rank over all 64,818 cells, of which **99.63% are zeros**. Measured on month
463 (`results/s8_lambda_star.out`, and reproduced directly):

| | |
|---|---|
| `zt` on the 64,579 zero cells | `[-1.000000, +0.992595]` |
| `zt` on the 239 event cells | ⛔ `[+0.992626, +0.999969]` — a band **0.37%** of the range wide |
| sd of `zt` **among events** | **0.002129** |
| sd of the noise term | 1.000000 |
| ⇒ event-ordering signal:noise at λ=0.95 | **0.04** |

⇒ **All events are crammed into the top 0.37% of the oracle key**, so at every λ < 1 the noise term
dominates *among* events. The sweep's own output shows it: `rho_event` is flat at **≈ −0.019 from
λ=0.00 through λ=0.95**, then jumps to **+0.9962** at λ=1.00, while the CRPS gap moves smoothly from
−0.14 to −4.26 across the same range.

⇒ ⛔⛔ **The λ dial parameterises event-versus-zero SEPARATION, not ordering AMONG events.** It is a
gate-like quality, not a body-ranking quality — and it is the one quantity λ\* was to be stated in.
⚠️ This also explains 2026-11-04's *"Spearman 0.68 at lam=0.75"*: that was measured on all firing
cells, i.e. the `rho_open` population, which the dial **does** move.

### The amendment

> 1. The λ grid is retained and reported whole, with an **AUC** column added so that it is read in
>    the quantity it actually moves (separation), never in `rho_event`.
> 2. A second dial **ν** is added, which moves ordering **among event cells** while holding
>    separation **exactly** fixed: among event cells, blend
>    `s = ν·z(rank(truth)) + (1−ν)·z(real_key)` and reassign the REAL key's **own value multiset**
>    on those cells in the order of `s`. ⭐ The multiset is unchanged, so AUC and every
>    separation-driven effect are held constant by construction.
> 3. **λ\* is located on ν**, and stated as *how much better the body's event ordering would have to
>    be for the improvement over the body's CURRENT ordering to exceed `2·σ̂`.*

⭐ **ν=0 reproduces the REAL arm exactly**, which makes it a second exact no-op control, and ν=1 is
perfect event ordering at the body's own separation.

⛔ **The criterion, the margin, the population, the open set and every VOID trigger are unchanged.**
This amendment changes a **sweep axis** that was measured to be degenerate; it does not change what
is being tested or what would count as a win. ⛔ It was written before the ν sweep was run.
