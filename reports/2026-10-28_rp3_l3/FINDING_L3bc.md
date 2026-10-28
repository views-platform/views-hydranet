# L3b / L3c (#471) — it works on all three targets, and the published target is the best of the three

**Date** 2026-10-28 · **Scripts** `l3b_three_target_cube.py` (committed `4b1c3b7`),
`l3c_three_target_scores.py` + `binned_cdf.py` (committed `0b00a6e`), all before their runs ·
**Outputs** `l3b_three_target_cube.out`, `l3c_three_target_scores.out` · **Cost** 818.1 s + 49.3 s,
CPU, nothing retrained · **All gates PASS**

## 1 · ⭐⭐⭐ The cube is bit-identical to the published one, and that settles several things at once

C-394 found the vehicle on disk. `l3b` swept it for **all three targets in one pass** and gate
**G-2** compared the `sb` slice against `d4_result.npz`:

> all three arrays **(4, 49, 64818)** bit-identical, `np.array_equal`, **no tolerance**

⇒ In one assertion: the lead-in, the month range, the draw count, the dropout masks and the slice
arithmetic are proven identical, **and** `d4_result.npz` is proven to be what it claims, re-derived
from the vehicle it names. ⭐ It also extends **M74**'s determinism result — that cube was written
**2026-10-12/20** and re-derived **2026-10-28**: bit-identical across **16 days**, a different
process and a different working directory.

⭐ The sweep cost **nothing extra**. `out.reg` is `[1, n_targets*npar, H, W]` and `out.cls` is
`[1, n_targets, H, W]` — the model already computes all three on every forward pass, and
`d4_mu_on_the_grid.py:151,154` discards two at write time. Keeping all three added **zero** forward
passes and ~413 MB. ⇒ The argument in **#487**, now measured rather than reasoned.

| target | mu mean | theta mean | gate mean |
|---|---|---|---|
| `ged_sb_best` | 0.062514 | 0.000711 | 0.003538 |
| `ged_ns_best` | 0.076763 | 0.003038 | 0.008160 |
| `ged_os_best` | 0.174021 | 0.000412 | 0.002793 |

## 2 · ⛔ The answer to #471's second half: it generalises in SIGN, and loses more than half its size

Same split as the headline (fit 421–456, score 457–469). ⛔ No parametric comparator exists for
`ns` or `os` — `0.046023` and `0.050758` are `sb` numbers on those months and nowhere else — so the
margin is the **internal** one L3a used: CRPS at the best bin count against **B = 1**, the
unconditional empirical CDF of the same fit window, model never consulted.

| target | positive rate | B\* | CRPS(B\*) | CRPS(B=1) | **margin** |
|---|---|---|---|---|---|
| ⭐ `ged_sb_best` | 0.003448 | 8,000 | 0.039186 | 0.060323 | **−35.0%** |
| `ged_ns_best` | 0.001201 | 16,000 | 0.016166 | 0.021848 | **−26.0%** |
| ⛔ `ged_os_best` | 0.001291 | 2,000 | 0.006208 | 0.007294 | **−14.9%** |

⭐ **Three of three hold.** Conditioning on the model's own `gate × mu` index buys a real reduction
on every target. ⇒ The lookup is reading something genuine in the **trunk's ranking**, not something
peculiar to the `sb` head — which matters, because all three heads sit on one shared trunk.

⛔ **But the margin more than halves**, 35.0 → 26.0 → **14.9**. The headline was measured on the
**best of the three targets.**

⚠️ **And it is not simply a base-rate effect.** `ns` (0.001201) has a *lower* positive rate than
`os` (0.001291) and a *larger* margin. ⇒ Rarity alone does not order these; something about how well
the trunk ranks each target does.

⛔ **`os` is the weak case, and its curve says so independently of the margin.** Its ladder is
nearly flat from 1,000 bins upward — 0.006233 / 0.006208 / 0.006216 / 0.006288 / 0.006250 /
0.006239 across 1k–32k, a **0.9% span** — and its optimum sits at **2,000**, four times coarser
than `sb`'s. ⇒ The index carries **less information about `os`**, and extra resolution stops paying
early.

## 3 · ⛔⛔ Read with L3a, this is the second unrecorded selection in the same headline

| L3a | the published **score window** is the **easiest of eight** available in the cube |
| --- | --- |
| **L3c** | the published **target** is the **best of the three** the model emits |

⇒ Two independent favourable selections, in the same number, neither recorded anywhere. ⚠️ Neither
invalidates the **−15.3% / −17.6%** — its comparator sat on the same window and the same target, so
both selections are common to both sides and largely cancel. ⛔ What they mean is that the headline
is **the most favourable cell of a 8 × 3 grid**, and that must travel with it. **C-393 extended.**

## 4 · ⛔ What this still does not establish

- **No parametric comparison on `ns` or `os`.** The B=1 margin measures *conditioning value*, not
  *superiority over a parametric head*. ⛔ Nothing here says the lookup beats the NB emission on
  `ns` or `os`, and nothing here could.
- **One split.** L3a swept the boundary for `sb` only. The target sweep is at the published
  boundary alone.
- **One checkpoint, one seed.** ⛔ C-394 put the vehicle back on disk, but there is still **no
  second seed** in the `nb` @ 0.15 cell ⇒ **no σ̂ for CRPS on this route**. The 35.0 / 26.0 / 14.9
  ordering cannot be separated from noise. **Reproducible ≠ resolvable.**
- **`d4_mu_on_the_grid.py` was not modified** and the new cube lives in the RP-3 dossier under its
  own name. **#487** holds the deferred decision.

## 5 · The gates, and one of them is new in kind

- **G-S1 — source identity.** `binned_cdf.py` and `l3a_split_sensitivity.py` hold the same scorer,
  asserted by comparing their **parsed ASTs with docstrings excluded**. ⇒ Two scripts quoting one
  statistic are mechanically proven to be running one implementation, which is the defect C-363
  names, caught at the source level rather than after the numbers disagree.
- **G-S2 — the anchor**, 0.039186 / 0.039026 to **1e-6**, PASS, exactly as the bit-identical cube
  requires. ⚠️ Anchored on the **committed implementation**, never on M67's ledger value — W-98's
  lesson applied rather than repeated. M67's **0.039327** remains reproduced by **no**
  implementation (**−0.36%**), an open item of #460.
- G-1 sha match · G-3 non-vacuity, 3 of 3 target pairs differ · G-4 no BatchNorm in `train()`
  (C-384's defect inverted a headline's sign) · G-5 round-trip · G-S3 cube sha · G-S4 three distinct
  values · G-S5 void coverage 0.0000%.
