# M3 (#480) — the invariance table. Fourteen of twenty columns are functions of the mean field alone, and `matched_pred_thresh` keeps its promise in 52% of forecast-months.

**Date** 2026-10-28 · **Issue** #480 (epic #477, RP-1) · **Scripts**
`reports/2026-10-27_metric_redteam/invariance_probes.py` (committed `12e7e9a`, **before it ran**)
and `count_matching_audit.py` (committed before each of its two runs) · **Outputs**
`invariance_probes.out/.npz`, `count_matching_audit.out/.npz` · **165.7 s + 13.9 s, 9.93 GiB, CPU**

⭐ Every primitive **imported** from the frozen sources, never reimplemented. `score()` is
duplicated from M2 rather than imported — importing M2 would re-run its 60 s matrix — and **G-4
pins the duplicate**: it reproduces M2's committed `separation_matrix.npz` to **0.000e+00** on all
five deterministic forecasts.

**C-351 slots.** 13 months 457–469 × 64,818 land cells = 842,634 cell-months; FSS/AFSS/hit_rate on
the full 360×720 grid per month then averaged; count space, no normalisation; S = 256; CRPS via the
frozen plug-in. **Base** = the delivered cube's forecast **redrawn in this process**, so its
absolute numbers differ from M2's row by the MC draw — ⛔ every number below is a Δ between the base
and a perturbation scored **in the same process from the same base array**, so the draw cancels
exactly.

---

## ⭐⭐⭐ Finding A — the partition. Fourteen of twenty columns cannot see dispersion, by construction

Derived from the source **before any probe ran**, by asking which argument each column reads:

| reads | n | columns |
|---|---|---|
| `p` only | **2** | `AP`, `Brier` |
| `cs` beyond its row mean | **4** | `crps_all`, `crps_events`, `crps_none`, `QS99` |
| `ey = cs.mean(1)` only | **14** | `size_ratio`, `pos_mcr`, `MCR_field`, `FSS`×7, `AFSS`, `hit_rate`, `act_freq`, `Gini` |

**Confirmed to exactly 0.000e+00** by two perturbations that hold `ey` bit-exactly and change the
ensemble's shape completely:

| perturbation | the 14 mean-field columns | `crps_events` | `QS99` | `crps_all` |
|---|---|---|---|---|
| ensemble → point mass at its own mean (var 0) | **0.000e+00** | 0.384038 | 0.034385 | 0.003150 |
| ensemble → `{0, 2·ey}` alternating (var `ey²`) | **0.000e+00** | 1.412975 | 0.025138 | 0.002730 |

⇒ ⭐⭐⭐ **The ensemble's distributional shape — which is what under-dispersion *is* — is visible to
exactly four columns, all of them CRPS or the pinball guardrail.** Seventy percent of the battery
would return identical numbers for a point forecast and for a correctly dispersed ensemble with the
same mean.

⚠️ **This is a direct, derived constraint on RP-4** (draw count vs under-dispersion): there are only
four places dispersion can show up, and three of them are the same CRPS split. ⛔ It also means any
future "the arm is better calibrated" claim that cites a mean-field column is citing a column that
cannot see calibration.

⭐ **Bonus invariance, found by a gate rather than by a derivation.** `I-1c` shuffled the sample
columns of the ensemble and moved **nothing** — all twenty columns, Δ = 0. G-5 (non-vacuity)
flagged it as a probe that was never applied, and the reason is the finding: **every column treats
the ensemble as an unordered bag** (`crps_ensemble` sorts, the rest take a mean or a quantile).
⇒ Sample *order* carries no information anywhere in the battery. That was not on the derived list.

## ⭐⭐ Finding B — `AP` is blind to gate calibration, not only to magnitude

| perturbation | \|Δ AP\| | \|Δ Brier\| |
|---|---|---|
| `p → p²` (strictly monotone) | **0.000e+00** | 0.000446 |
| `p → (p+1)/2` (affine monotone) | **0.000e+00** | **0.248964** |

Derivation: `average_precision` is rank-based; `brier` is `mean((p−y)²)`. ⇒ **AP cannot see any
monotone miscalibration of the gate.** The second row is a gate squashed into [0.5, 1] — a forecast
claiming every cell on Earth is at least 50% likely to see conflict — and AP is **unchanged to
machine precision** while Brier moves by a quarter.

And three unrelated body perturbations (`×7`, `→ 0`, shuffled) moved `AP` and `Brier` by exactly
**0.000e+00** each. ⭐ **The ruler says this about itself and the consequence was never drawn:**
`apply_threshold_gate`'s comment reads *"body-only; p (AP/Brier) unchanged"* ⇒ **AP and Brier are
invariant to the entire `th_gated` composition axis by design**, which is why M67 measured
`AP = 0.736228` across all four cells of a 2×2.

## ⭐ Finding C — `size_ratio`'s tail invariance is a curve with a break at exactly 0.50

⛔ Swept (D-4), never a scalar. The top-k event cells **by the ratio `ey/truth`** — the statistic
`size_ratio` takes a median *of* — inflated ×1000:

| k / n_event | \|Δ size_ratio\| | \|Δ pos_mcr\| |
|---|---|---|
| 0.001 | **0.000000** | 14.290 |
| 0.010 | **0.000000** | 80.767 |
| 0.100 | **0.000000** | 312.536 |
| 0.250 | **0.000000** | 456.272 |
| 0.490 | **0.000000** | 563.412 |
| **0.510** | **267.152344** | 568.816 |
| 0.750 | 267.152344 | 606.560 |

⇒ **A forecast may be wrong by a factor of 1000 on up to 49% of its event cells and `size_ratio`
will not move at all** — while `pos_mcr`, the mean of the same ratios, moves by 563.
⚠️ The two columns are a median and a mean of **one** quantity; they are not two pieces of evidence
about magnitude, they are two summaries of the same vector with wildly different breakdown points.

⛔ **A correction to my own derivation, made before the run, not after.** The first version ranked
by `truth`. That moves cells from *below* the median to above it and shifts the median for **any**
k, which is a different and weaker claim. The invariance is about the tail of the **ratio**
distribution.

## ⭐ Finding D — `pos_mcr`'s weight is exactly `1/truth`, confirmed to 7.6e-17 against a closed form

Derivation: `∂pos_mcr/∂ey_i = 1/(n_event · truth_i)`. ⇒ adding the same total mass to the 100
smallest-truth event cells versus the 100 largest-truth ones must move `pos_mcr` by
`Σ(1/truth)/n_event` in each case — **predicted in closed form before measuring**:

| | predicted | measured | \|Δ\| |
|---|---|---|---|
| +1.0 on the 100 **smallest**-truth event cells | 0.03442341 | 0.03442341 | **7.6e-17** |
| +1.0 on the 100 **largest**-truth event cells | 0.00029859 | 0.00029859 | **1.3e-17** |

⇒ ⛔ **The same 100 units of mass move `pos_mcr` 115.3× more when added to the smallest truths.**
`pos_mcr` is a statement about one-event cells. ⚠️ `size_ratio` moved 1.97e-02 / 3.21e-04 on the
same two perturbations — a 61× asymmetry in the same direction, so the median inherits the
weighting in attenuated form.

## ⭐ Finding E — `Gini`'s floor is set by `act_freq`, which the battery already carries

Derivation: for a field with a fraction `f` of equal-valued cells and the rest zero, the
sorted-identity Gini is `1 − f + O(1/n)`; an unequal field scores higher. ⇒ `Gini ≥ 1 − act_freq −
1/n` for every forecast. **Confirmed on all 11 of 11** (`all-zero` returns `nan`, correctly):

| forecast | `act_freq` | `Gini` | derived floor | holds |
|---|---|---|---|---|
| oracle | 0.003448 | 0.998958 | 0.996551 | yes |
| `freq-right/mag-wrong` | 0.003448 | **0.996552** | **0.996551** | yes — **at the floor to 1e-6** |
| reference (delivered) | 0.012173 | 0.997411 | 0.987826 | yes |
| blurred r=4 | 0.131980 | 0.986870 | 0.868019 | yes |

⇒ **M2 measured `Gini`'s relative range at 1.22% and this is why**: on a 99.655%-zero field the
achievable span is about `f` wide. ⛔ **`Gini` is near-redundant with `act_freq`**, which separates
at 3828%. ⇒ The 2026-08-13 draft's "spatial concentration" leg is dead as a Gini, and **#483 (M6)
must not pick it** — `W-89`.

## ⛔⛔⛔ Finding F — `matched_pred_thresh` keeps its documented promise in 75 of 143 forecast-months, and it fails in THREE distinct ways

**This corrects C-388 and my own #479 comment.** Both said *"no published FSS number is affected —
every delivered forecast is denser than the truth."* ⛔ **That is false as stated.** I-8 showed the
`hit_rate == FSS(1)` identity breaking for **8 of 11** forecasts, not 2 — including the **delivered
model** (\|Δ\| 2.2e-04) and **climatology** (1.6e-04), where `AFSS` sits at 0.999999 rather than at
either extreme. So a second mechanism existed, it fires on **dense** forecasts, and C-388 did not
have it.

⭐ The second and third mechanisms were each **derived first and then measured**, with a gate that
would reject the account if it were incomplete. Both gates did their job — **G-7 failed**, which is
how mechanism (c) was found at all.

| mechanism | cause | forecast-months | over-selection |
|---|---|---|---|
| **(a)** | the `n_events`-th largest prediction is **0** ⇒ threshold `−1e-12` ⇒ **all 259,200** cells | **34** of 143 | up to 259,002 |
| **(b)** | the `n_events`-th largest value is **tied**; every duplicate clears `kth − 1e-12` | **31** of 143 | 1–12 cells |
| **(c)** | the `−1e-12` is **ABSOLUTE**; on a continuous field, cells strictly *below* the kth still clear it | **3** of 143 | 1–12 cells |
| — | exact, the function as documented | **75** of 143 | 0 |

**(b)'s cause is structural, not incidental:** `ey = cs.mean(1)` is the mean of S integer draws, so
it is **quantised to multiples of 1/256**, and on a ~99%-zero field the positive values pile onto
the smallest multiples — so the k-th largest is very often duplicated. ⇒ **Any ensemble mean field
is exposed to (b).**

**(c) was predicted in closed form and confirmed exactly.** G-8: the over-selection must equal the
count of values in `(kth − 1e-12, kth)`.

| forecast | month | kth | selected | n_events | over | in `(kth−1e-12, kth)` |
|---|---|---|---|---|---|---|
| blurred r=1 | 2 | 3.12472239 | 217 | 212 | **5** | **5** |
| blurred r=1 | 8 | 4.00792300 | 240 | 228 | **12** | **12** |
| blurred r=2 | 9 | 3.80346281 | 231 | 230 | **1** | **1** |

⇒ **An absolute epsilon on a field that has no absolute scale.** At a kth of order 100 it is a
tolerance of a few ulps; at order 1e-3 it swallows the whole neighbourhood.

### What this does and does not cost the record

- ⛔ **(a) never fires for the delivered model** (0 of 13 months) — that part of C-388 stands.
- ⛔ **(b) fires for it in 2 of 13 months**, over-selecting **exactly 1 cell** ⇒ the delivered
  model's own FSS *is* perturbed, at the 1e-4 level on the identity check, **not** at the 1.0 level.
  ⇒ The sentence "no published FSS number is affected" must become "**no published FSS number is
  affected by (a); (b) perturbs them at the 1e-4 level and the two months were never identified.**"
- ⛔⛔ **`persistence` hits (a) in 8 of 13 months.** It is a *published baseline* in this programme.
  No dossier currently reports an FSS or hit-rate number *for* persistence — checked — but ⛔ **any
  future one is worthless without a count-matching report beside it.**
- ⚠️ **UNMEASURED, and flagged rather than assumed:** whether (a) fires on the populations the
  published **B-1a/B-1b** FSS numbers were computed over (32×32 windows, land cells, a different
  `f_o`). (b) is generic to any ensemble mean field, so those numbers are exposed to it by
  construction. ⛔ This is the next check anyone touching B-1b owes.

## The gates

| gate | verdict | evidence |
|---|---|---|
| **G-0** frozen primitives reproduce their hand cases | **PASS** | crps 0.5, brier 0.0/0.25, AP 1.0 |
| **G-4** the duplicated `score()` == M2's committed npz | **PASS** | **0.000e+00** on all five deterministic forecasts |
| **G-5** every probe moves at least one column | ⛔ **FAIL** on `I-1c` | ⭐ and the failure *is* a finding — sample order is invariant everywhere |
| **G-6** every derived invariance holds to 1e-12 | **PASS** | every one at exactly **0.000e+00** |
| **G-7** a unique positive kth delivers exactly `n_events` | ⛔ **FAIL**, 3 months | ⭐ which is how mechanism (c) was found |
| **G-8** the over-selection == the near-tie count | **PASS** | 5/5, 12/12, 1/1 |

⚠️ **Two of six gates failed and both failures are results, not defects in the harness.** That is
what #480's acceptance criterion asked for: *"any derivation that fails to confirm is the more
interesting result."* ⛔ Note which way it went — **every derivation I wrote down held at exactly
0**; what failed were the two gates checking that the *harness* and the *shared function* behaved
as documented.

## ⚠️ What M3 does NOT establish

- ⛔ **Nothing about the plug-in CRPS bias.** All CRPS numbers here use the frozen plug-in, which is
  the point: **#481 (M4)** owns what it costs. The four shape-sensitive columns are shape-sensitive
  *and* biased.
- ⛔ **Nothing about `QS99`'s definition.** It is declared here as `t0_score.py` computes it and is
  **not** reconciled with the FAO-02 guardrail — **#482 (M5)**.
- ⚠️ **The invariances are exact for the perturbations applied.** A derivation confirmed on two
  perturbations is not a proof; it is a derivation that survived the two attacks designed to break
  it. The derivations themselves are from the source and stand on their own.
- ⚠️ **One base.** All probes run against the delivered forecast. A degenerate base would make I-3
  vacuous, so a single non-degenerate base was the right choice — but the invariances are claimed
  from the source, not from the base.

## ⇒ Consequences

1. ⛔ **#483 (M6) is unblocked and now scoped by a derivation, not a hunch.** The gap is: *nothing
   sees a magnitude error that preserves placement.* M3 says exactly why — 14 columns read only the
   mean field, 2 read only the gate, and the 4 that read the distribution read it pooled. The
   instrument must read **ranked magnitude against placement**, which is the one M67 already
   demonstrated (`spearmanr`, `+0.4925 → −0.1605`). ⛔ **Not Gini** (Finding E, `W-89`).
2. ⛔ **C-388 needs an amendment** for mechanisms (b) and (c) and for the corrected claim about
   published numbers.
3. ⚠️ **`size_ratio` and `pos_mcr` must never be quoted together as two findings** (Finding C).
4. ⚠️ **No mean-field column may be cited for a dispersion or calibration claim** (Findings A, B).
