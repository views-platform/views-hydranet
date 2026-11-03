# M3b — a draw-derived gate costs AP a systematic **−0.0066** at S = 256, about 7× the seed sd. Brier is untouched.

**Date** 2026-11-03 · **Asked by** views-baseline#109 (the α-sweep) · **Script**
`gate_from_draws.py`, committed **before it ran** (`021b2dc`) · **Output** `gate_from_draws.out`
· ~90 s, ~8 GiB, CPU, $0

**C-351 slots.** Population = 13 months **457–469** × **64,818** land cells = **842,634**
cell-months, `ged_sb_best` counts, **2,905** positives (base rate **0.003448**), ordering
identical to M2's. Transform = none (count space, no normalisation). Estimator =
`p_hat = Binomial(S, p)/S` — the exact distribution of `(cs > 0).mean(1)` for a generator whose
per-cell occurrence probability is `p` — mean and sample sd (`ddof=1`) over **5** seeds.
`AP`/`Brier` **imported** from the frozen ruler, never reimplemented.

---

## 1 · The question, and why it is not rhetorical

M2 and M3 scored eleven forecasts that each supplied a gate array `p`. A pure **count generator**
has none — and `AP` and `Brier` are the only two battery columns that read `p` at all (M3 Finding
A). So scoring one requires **defining** a gate, and the harness's own precedent is the
generator's occurrence probability: `f_climatology` uses the fit window's positive rate
(`redteam_forecasts.py:188`). For a count generator that is `p = (cs > 0).mean(1)`.

⛔ **At S = 256 that estimate is quantised to 1/256 = 0.003906, which is coarser than this
population's base rate of 0.003448.** The step size of the score exceeds the prevalence it is
ranking, and `average_precision` is rank-based. ⇒ It was cheaper to measure than to argue, so
this is the measurement.

⚠️ **No prediction is claimed here, deliberately.** The concern above is what motivated the probe,
and my expectation was that the gate leg would prove unusable — but that expectation was **never
recorded before the first run**, only afterwards. ⇒ It earns **no `W-` row**, because a prediction
written down after the result is not a prediction, and claiming one would be the retroactive-probe
failure the falsification discipline names explicitly. The honest statement is: the quantisation
was derived in advance, the direction of its effect was not pre-committed.

## 2 · ⭐ The result

**G-0 PASS, and it is the load-bearing gate.** The continuous delivered gate reproduces M2's
committed `separation_matrix.npz` reference row at **AP = 0.736228** — the committed figure to
6 dp. That one check confirms the truth assembly, the population, the row ordering and the frozen
primitive together, and the script **refuses to report** if it fails.

| S | AP (mean ± sd, 5 seeds) | Brier | distinct `p` | ΔAP vs continuous |
|---|---|---|---|---|
| **256** | **0.729594 ± 0.000941** | 0.001682 | 255 | **−0.006634** |
| 1024 | 0.734417 ± 0.000233 | 0.001674 | 975 | −0.001810 |
| 4096 | 0.735898 ± 0.000158 | 0.001671 | 2,833 | −0.000330 |
| 16384 | 0.736278 ± 0.000168 | 0.001671 | 6,070 | +0.000050 |
| continuous | 0.736228 | 0.001671 | 840,883 | — |

⇒ ⭐⭐ **The bias is systematic, not noise: −0.0066 against a seed sd of 0.00094, a ratio of 7.0.**
It decays roughly as expected with S and is gone by S ≈ 16k. **Brier moves by 1.1e-05** — a
rank-based column pays for the quantisation and a squared-error column does not, which is the
same asymmetry M3 Finding B found from the other side.

⚠️ **255 distinct values out of 842,634 cells.** The gate leg of the battery is being computed on
a score with 255 levels. It still ranks — that is the finding — but every claim resting on it is
resting on 255 levels.

## 3 · ⇒ Consequences, and the one that binds

1. ⭐ **AP is usable for an α-sweep.** The bias is a property of S, not of α, so at fixed S it is
   common to every row and **cancels in a paired difference**.
2. ⛔⛔ **It does NOT cancel in a level.** An absolute AP from a draw-derived gate is **not
   comparable to any M2 row that carried a real gate**, including the delivered model's 0.736228 —
   the gap between them is 0.0066 of pure instrument. ⇒ Report α-**differences**, never α-levels,
   and never place a draw-derived AP in the same column as a gated model's.
3. ⚠️ **Equal S was already required** (epic #263: the reference's S must equal the arms'). This
   adds a second, independent reason — the gate leg's bias is S-dependent too, not just the
   plug-in CRPS's.

## 4 · ⚠️ What this does NOT establish

- ⛔ **The scope limit that matters is not the obvious one.** The probe perturbs the delivered
  cube's **gate**, so it isolates the cost of *estimating* an occurrence probability from S draws.
  It does **not** model the fact that the delivered cube's own `(cs > 0).mean(1)` sits **below**
  its gate, because a `gate × NB` draw can be zero while the gate fires. ⇒ For a generator with a
  separate gate the two quantities differ and **this probe does not bound that gap**. For a count
  generator there is no separate gate — `P(Y > 0)` *is* its occurrence probability — so the probe
  is apt for exactly the case it was built for and silent about the other.
- ⚠️ **One occurrence structure.** `p` is the delivered model's. The quantisation mechanism is
  generic and the base rate is this population's, so this is the **order of magnitude for any
  count generator on this population**, not a number for a specific one.
- ⚠️ **Independent Bernoulli draws.** `Binomial(S, p)` assumes the S draws are independent per
  cell. A generator with draw-to-draw dependence (common random numbers across α, for instance)
  would have a different sampling distribution for `p_hat`, though the same 1/S quantisation.
- ⛔ **Nothing about the four dispersion columns.** Their noise floors remain **UNMEASURED** on
  our side (RP-2b §5, blocked on #441).

## 5 · ⚠️ Provenance defect, recorded because the output cannot show it

The first run of this probe happened in a scratchpad **before** the script was committed, which
inverts the dossier's D-5 discipline (*script committed before it runs*). The committed copy was
then run from scratch and reproduced **every figure identically**, and the `.out` beside this
finding is from the committed version — so the numbers are sound. The ordering is logged anyway:
reproduction after the fact is not the same guarantee as commitment before, and the difference is
invisible in an output file.
