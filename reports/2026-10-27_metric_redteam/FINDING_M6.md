# M6 (#483) — the placement instrument. A 100% magnitude scramble moves six placement columns by exactly zero and moves the instrument from 1.000 to −0.030.

**Date** 2026-10-28 · **Issue** #483 (epic #477, RP-1) · **Instrument**
`scripts/placement_rho.py` (new production module) · **Tests** `tests/test_placement_rho.py`
(21, independent oracle) · **Driver**
`reports/2026-10-27_metric_redteam/placement_instrument.py`, committed `0c4ea94` **before it ran**
· **Output** `placement_instrument.out/.npz` · **16.8 s, 8.27 GiB, CPU** · **all six validation
gates PASS**

⛔ **#483 was conditional and the condition was met.** Its own text: *"if M2's separation matrix is
clean, #444 is wrong and this story dies."* M2 is not clean, and M3 derived why — **14 of 20
columns are functions of the mean field alone**, 2 read only the gate, and the 4 that read the
ensemble read it pooled. ⇒ **#444 confirmed**, with the gap named: *nothing sees a magnitude error
that preserves placement.*

---

## The instrument, and why the other three candidates are dead

`placement_rho` = **Spearman rank correlation between predicted and observed magnitude over a
declared population**. One statistic. #483 listed four candidates and pre-selected none; three are
ruled out by **measurement**, not preference:

| candidate | why not |
|---|---|
| **Gini / spatial concentration** | ⛔ **DEAD.** M3's I-7 derived the floor `1 − act_freq − 1/n`, confirmed **11 of 11**; M2 measured the relative range at **1.22%**, the only column failing the 2026-08-13 draft's own <3% test. Near-redundant with `act_freq`. (`W-89`) |
| **activation frequency** | Already **in** the battery, and it is a *placement* statistic — it cannot close a *magnitude* hole. |
| **SAL / MODE** | A programme, not a story. #483 forbids it in writing. |
| **rank correlation** | ⭐ **Chosen.** M67 already used it to catch a composition defect AP scored 0.736228 on (`+0.4925 → −0.1605`); it is scale-free, so the known mass deficit cannot leak in; no new dependency. |

**C-351 slots.** 842,634 cell-months (13 months × 64,818 land cells); the **mean field**
`ey = cs.mean(1)`, count space, no normalisation; Spearman with average ranks; the instrument's own
**cell population and pooling are swept free parameters** (D-4). The battery's comparison numbers
are **read from M2's committed `separation_matrix.npz`**, never recomputed — so there is no third
copy of `score()` and the separation claim is made against what `FINDING_M2.md` already published.

---

## ⭐⭐⭐ Finding A — V-5. Six placement columns move by exactly 0 across a full magnitude scramble

Scramble a fraction `f` of the **oracle's** event-cell magnitudes among themselves. The nonzero
cell set is untouched — asserted every step, never assumed — so placement is preserved exactly.

| f | cells scrambled | **placement_rho** | \|Δ AP\| | \|Δ Brier\| | \|Δ FSS1\| | \|Δ FSS9\| | \|Δ AFSS\| | \|Δ hit_rate\| |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 0 | **1.000000** | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.10 | 290 | 0.890970 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.25 | 726 | 0.750752 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.50 | 1,452 | 0.477811 | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.75 | 2,179 | 0.249725 | 0 | 0 | 0 | 0 | 0 | 0 |
| **1.00** | **2,905** | **−0.030036** | **0** | **0** | **0** | **0** | **0** | **0** |

Every Δ is **exactly `0.000e+00`**, not "within tolerance". ⇒ ⭐⭐⭐ **A forecast can have 100% of
its event-cell magnitudes randomly permuted and six battery columns return bit-identical numbers,
while the instrument falls from 1.000 to −0.030, monotonically.** That is #444, closed, as a curve
rather than an anecdote.

## ⭐⭐ Finding B — V-2. The instrument separates both pairs the battery cannot

| pair | M2 columns giving them the SAME value to 1e-6 | `placement_rho` (events, pooled) |
|---|---|---|
| **oracle** vs **freq-right/mag-wrong** | **14 of 20** — AP, Brier, crps_none, MCR_field, FSS×7, AFSS, hit_rate, act_freq | **1.000000** vs **0.000000** — separation **1.000000** |
| **spiky-1/30** vs **all-zero** | **10 of 20** — size_ratio, FSS×7, AFSS, hit_rate | **0.521009** vs **0.000000** — separation **0.521009** |

⚠️ **14, not the 12 I reported from M2.** M2's "12 of 18" counted *scoring* columns; including
`MCR_field` and `act_freq` it is **14 of 20**. The larger number is the honest one for this pair.

## ⛔⛔ Finding C — the instrument's own free parameter can make it blind, and that was not predicted

| forecast | events/pooled | union/pooled | **all/pooled** |
|---|---|---|---|
| **freq-right/mag-wrong** | **0.000000** | 0.000000 | ⛔ **0.999998** |

On the all-cell population, the zeros tie with zeros and the positives tie with positives, and that
near-perfect agreement **swamps the ordering question entirely**. ⇒ ⛔ **`population='all'` is blind
to the exact defect this instrument was built for.** A caller who takes it gets a statistic that
cannot fail.

⭐ **This is the strongest single argument for D-4 I have produced.** The sweep was declared as
discipline; it turned out to decide *whether the instrument works at all*. `'events'` is therefore
not a default by preference — it is the population the number means something on, and that is now
written into the module's docstring where a caller will read it (`cf1db21`).

## ⛔ Finding D — it is not a standalone selector, and the reason is measured

| forecast | events/pooled |
|---|---|
| blurred-truth r=1 | **0.682500** |
| reference (delivered) | 0.629054 |
| **spiky-1/30** | **0.521009** |
| persistence | 0.427759 |
| climatology | 0.394095 |
| mag-right/freq-wrong | 0.008470 |

⛔ **`spiky-1/30` — all of a month's mass on 97 cells, the #258 defect — ranks ABOVE persistence
and climatology.** Its cells correlate with the truth even though its field is absurd. ⇒ A column
measuring conditional-magnitude *ranking* says nothing about placement, calibration or mass, and
selecting on it alone would select for the pathology the rest of the battery exists to catch. Also
in the docstring, as a measurement rather than a disclaimer.

⭐ **And another RP-10 datum:** `blurred-truth r=1` ranks magnitude **better than the delivered
model does** (0.6825 vs 0.6291). That is the *second* column to prefer the appropriately-smoothed
forecast, after `crps_events` in M2 (13.946 vs 14.487).

⚠️ **The delivered model beats persistence here (0.629 vs 0.428)** — the *opposite* ordering from
RP-6's recorded Spearman `0.303` vs `0.367`. ⛔ **Different populations; do not quote one against
the other (C-351).** ⇒ But it raises RP-6's real question: **is that recorded ordering
population-dependent?** RP-6 should answer that before it answers anything else.

## ⭐ Finding E — V-6. Not a relabelling of anything we already have

\|Spearman(placement_rho, column)\| across the eleven forecasts, highest first:

`crps_events` **0.7790** · `AP` 0.4772 · `Brier` 0.4635 · `QS99` 0.4374 · `FSS3/5/9/65` 0.4027 ·
`act_freq` 0.3954 · `FSS1/17/33` 0.3478 · `size_ratio` 0.2681 · `crps_all` 0.1549 ·
`hit_rate` 0.0793 · `Gini` 0.0669 · `crps_none` 0.0644 · `pos_mcr` 0.0456 · `MCR_field` 0.0257 ·
`AFSS` 0.0207

⇒ No column exceeds **0.78**, and the nearest is `crps_events` — the one column M2 found *does*
work on magnitude. ⚠️ Read that as corroboration, not redundancy: the two agree on direction and
disagree enough (0.78, not 1.0) that `crps_events` carries level information the rank statistic
discards, and vice versa.

## The gates

| gate | verdict | evidence |
|---|---|---|
| **V-1** the oracle scores exactly 1.0 in all six sweep cells | **PASS** | max \|ρ−1\| **1.110e-16** |
| **V-2** known-good above known-bad on both blind pairs | **PASS** | separations 1.000000 and 0.521009 |
| **V-3** scale-free (`×1000`, `^1.3`) | **PASS** | max \|Δ\| **0.000e+00** |
| **V-4** monotone non-increasing in the scramble fraction | **PASS** | 1.000000 → −0.030036 |
| **V-5** every placement column moves by exactly 0 | **PASS** | worst \|Δ\| **0.000e+00** |
| **V-6** not a monotone relabelling of an existing column | **PASS** | highest 0.7790 |

## ⛔⛔ The guard audit — two independent rounds, 54 mutations, and the second round was worse than the first

#483's acceptance requires `/falsify guard` on the new tests. It ran **twice**, each time in a
subagent with a **clean context** (the auditor may not be the author, and may not read the
reasoning that produced the guard). ⛔ **It did not pass first time, and the second round found
worse defects than the first.**

| round | mutations | caught | **SURVIVED** | guard verdicts |
|---|---|---|---|---|
| 1, against 21 tests | 27 | 14 | **13** | 0 decorative, **4 WEAK** |
| 2, against 31 tests | 27 | 15 | **12** | 0 decorative, **5 WEAK** |

⭐ Round 2 closed **12 of the 13** named survivors from round 1 — the thirteenth is a backstop the
source itself now admits is unreachable and does not claim to test. But it found **six new live
survivors**, and two were serious:

- ⛔⛔ **X1.** Replacing the monthly branch's `sel = inm & _select(pred, truth, population)` with
  `sel = inm` left **31/31 green** and made the monthly path compute the **all-cell** rho inside
  each month, ignoring the declared population. Measured on the module's own defect: it returns
  **0.9974** where the correct value is **0.0** ⇒ **the mutant restores the exact blindness the
  instrument was built to close**, through the documented public API.
- ⛔⛔ **X2.** Changing the default from `population="events"` to `"all"` was pinned by **nothing**,
  because every test passed `population=` explicitly. Same consequence: **0.9975 instead of 0.0**.

⭐ **The auditor also named the common root, and it is the sharper finding:** the monthly branch and
the argument defaults were **only ever exercised at `population='all'`** — the one population this
module's own docstring says *cannot fail*. ⇒ A suite can be thorough on a function's arithmetic and
blind on the single parameter that decides whether the arithmetic means anything.

Four more survived and are now closed: the monthly per-month refusal was unguarded, so deleting it
was invisible and replacing it with `continue` **silently dropped an unrankable month from the
mean**; `sweep`'s `month=None` path was never exercised, so a mutant writing `0.0` for the three
monthly cells — **three invented rows in a D-4 table** — stayed green; a silent
`np.clip(pred, 0, None)` survived because no fixture had a negative prediction, the companion hole
being that the invariance test covered positive **scalings** but never an affine **shift**; and a
silent `np.round(truth)` survived because every fixture's truth was integer counts.

**21 → 31 → 42 tests.** Each round's fixes are then **verified rather than trusted**:
`verify_guard_mutations.py` re-applies the auditor's own surviving mutations and reports whether
the strengthened suite fails. **9 of 10 now CAUGHT.**

⚠️ **The one that still survives is reported as surviving, not explained away.** Dropping the
`dtype=np.float64` coercion changes nothing observable for any input the contract accepts (int or
float arrays, and lists), so the auditor classed it an **equivalent mutant** and I agree — it is
unpinned because it has no consequence, which is a different thing from being unguarded.

⚠️ **Independence, stated precisely.** The *mutations* are the auditors' design; `verify_guard_mutations.py`
only executes them. So the independence that matters — who modelled the failure — holds; what the
verification lacks is an independent *choice* of mutations, which is why it is a check on a prior
audit and not a substitute for one.

## ⚠️ What M6 does NOT establish

- ⛔ **It is not adopted into FAO-02 and nothing is re-scored against it.** FAO-02 is locked;
  `lodestar_score.py` is frozen and untouched. This is an instrument that measures what they are
  blind to, quoted beside them.
- ⛔ **No verdict about any arm.** The numbers above are a *validation* of the instrument against
  known answers, not a scoreboard. The delivered model's 0.629 is reported to show the instrument
  returns a sane value on a real forecast, not as a result about the model.
- ⚠️ **One population family.** T=0, land cell-months, `ged_sb_best`, 13 months. The sweep covers
  the instrument's own free parameters, not the choice of target or horizon.
- ⚠️ **`spiky`'s 0.521 is a property of *this* spiky**, which picks cells by the model's own
  `gate × mu` index. A spiky forecast picking cells at random would score ~0.

## ⇒ Consequences

1. ⭐ **#444 can be closed** — it said *"a change can invert placement and every FAO-02 column stays
   quiet"*, and the quantified version is now measured: six columns bit-identical across a full
   scramble, one instrument that moves 1.03 over the same range.
2. ⛔ **Quote `placement_rho` only with `population='events'`, and only beside a placement column
   and a mass column.** Findings C and D are the two ways to misuse it, both measured.
3. ⚠️ **RP-6 has a new first question:** whether the recorded 0.303-vs-0.367 ordering survives a
   change of population.
4. ⚠️ **Remaining in epic #477:** #481 (M4, what the plug-in costs past verdicts), #482 (M5, the
   QS99 gap), #484 (M7, tracking). ⛔ The epic does not close until M4 and M5 report.
