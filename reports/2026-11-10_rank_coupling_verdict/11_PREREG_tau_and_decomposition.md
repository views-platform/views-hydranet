# PRE-REGISTRATION — S11: sweep τ and the composition, and score with `MCB − DSC + UNC`

**Epic #579, story S11 (new).** ⛔ Committed **alone, before the tool exists** — the device S1 of
this epic could not honour, because there the tools were written first. Verifiable:
`git show --stat HEAD` must show **one** file.

---

## 0 ⛔ WHY THIS EXISTS — three corrections to S5–S9, all of them Simon's

E-3…E-7 reported *"B loses to A by **+8.52** CRPS, 7/7 folds"* and attributed the loss to B's
structure. ⛔ **Three things are wrong with that sentence and two of them change the result.**

1. ⛔⛔ **The cost was attributed to B and it belongs to τ.** Both arms were scored on A's own
   open set at `gate >= τ`, **τ = 0.20** — an unswept prior, set deliberately *below* this model's
   **measured calibrated 0.26** (`calm_orbiter/configs/config_hyperparameters.py:46-49`, quoted in
   `tools/arm_config.py`). ⭐ **MEASURED FIRST, before any outcome** (this is population metadata,
   not a result, and it is declared as having been measured before the predictions below):

   | τ | open cell-months | events | event rate | unique truths |
   |---|---|---|---|---|
   | 0.05 | 6,689 | 1,392 | **0.2081** | 115 |
   | 0.10 | 3,972 | 1,245 | 0.3134 | 113 |
   | ⬅ **0.20 (shipped)** | **2,256** | 1,034 | **0.4583** | 111 |
   | ⭐ **0.26 (calibrated)** | 1,749 | 939 | **0.5369** | 111 |
   | 0.35 | 1,291 | 796 | 0.6166 | 109 |
   | 0.50 | 877 | 638 | 0.7275 | 107 |
   | 0.65 | 572 | 468 | 0.8182 | 101 |
   | 0.80 | 303 | 271 | **0.8944** | 90 |

   ⇒ The "54% of open cells are not events" that E-7 §3 called **B's structural cost** is the
   **gate's false-positive rate at one unswept τ**, and it moves from 79% to 11% across the sweep.
   ⛔ I reported a scalar without sweeping its free parameter —
   `feedback_sweep_before_reporting_a_scalar`, the rule already on file.

2. ⛔ **"B cannot say *nothing here*" was false.** Occurrence is the **gate's** job, not the body's.
   A survives the 54% because A's *body* draws zero ~57% of the time — i.e. A's body is doing a
   **second, accidental gating**, and B removes it. The composition axis (`ADR-069`,
   `views_hydranet/distributions/composition.py`) is the designed place to put that back:
   `soft_gate` applies a **per-draw `Bernoulli(gate)`**, `threshold_gate` applies `gate >= τ`.
   ⇒ The body hurting is not the forecast hurting. The emitted forecast is the composition.

3. ⛔ **CRPS was the wrong headline and the repo already knew it.** `C-52` (`Brocker2009`): a proper
   score cannot rank miscalibrated schemes. On a population that is 54% zeros, CRPS is dominated by
   `UNC` and `MCB`; the ordering lives in `DSC`, the one component a raw CRPS does not separate.
   The governing reference is `MCB − DSC + UNC` (`Arnold2024`, `C-3572`,
   `reference_arnold2024_crps_decomposition.md`) and S5–S9 did not use it.

⚠️ **This is not a post-hoc criterion swap.** `DSC_ORD` / `R*` were derived on **2026-11-08**, in
`reports/2026-11-08_ordering_discrimination/01_MATH.md`, *before* this epic ran, and the
implementation (`tools/simulate_ordering_discrimination.py`: `crps_step`, `idr_fit`, `dsc_ord`) is
already committed and gated. ⛔ And **CRPS is retained and reported whole** — the decomposition is
an **identity**, not a replacement. Nothing is dropped; the same number is attributed.

---

## 1 C-351 SLOTS

**Population.** ⭐ **FIXED across every row**: the cells with `gate >= 0.05` on arm
`ctlfullzero_fortytwo`, score months 463–469, 7 folds, 6-month rolling fit windows — **6,689
cell-months, 1,392 events, 115 unique truths**, counts PRINTED not assumed.

⛔⛔ **Why the population is fixed and not each row's own open set.** The composition is part of the
**forecast**, not of the population. A cell that τ closes receives a **point mass at zero** — under
**both** arms. ⇒ every row is scored on **identical cells**, `UNC` is **constant** across the whole
sweep, and `MCB − DSC + UNC` is readable down the column. Scoring each row on its own open set
would make the rows incomparable, which is the defect that produced `+8.52` in the first place.

**Truth.** `lr_sb_best`, raw counts, no transform; column maximum printed.
**Pool.** Each fold's own fit window's strictly positive counts (unchanged from S5).
**Inference mode.** ⭐ DELIVERED AUTOREGRESSIVE, h = 1. **Arm.** `ctlfullzero_fortytwo`.
**Ensemble size.** ⭐ **S = 16 for both arms**, matched by construction (§4 `G-S6`).
**Estimator.** Plug-in CRPS via `crps_step` on the threshold grid, mean over the population.
⚠️ **Plug-in, not fair** — stated because the identity of §3 only closes when both sides use one
functional. Fair CRPS is printed **beside** it for continuity with S5, and is not decomposed.

---

## 2 THE ROWS — the sweep, reported WHOLE

| composition | setting |
|---|---|
| `threshold_gate` | τ ∈ {0.05, 0.10, **0.20**, **0.26**, 0.35, 0.50, 0.65, 0.80} |
| `soft_gate` | per-draw `Bernoulli(gate)`, no τ |
| `self_zeroed` | ⛔ not run — the delivered body is `nb`, which carries no zero spike |

Three arms **per row**, all composed identically:

| arm | ordering index |
|---|---|
| **A** | the delivered 16 body draws — the incumbent |
| **B/REAL** | rank-coupled empirical draw, ordered by **A's own composed point forecast** |
| **B/NOISE** | ⭐ the **potency control**: the same draw ordered by noise |

⛔ **The whole grid is printed.** Quoting the best cell of a τ × composition grid without the grid
beside it is not a result (`PREREG.md` §8).

---

## 3 ⭐⭐ THE DECOMPOSITION, AND THE IDENTITY THAT MAKES IT CHECKABLE

Per row, with `x` = the composed point forecast, **shared by A and B/REAL**:

```
UNC       = (1/n) Σ_i crps_step( Ĝ , y_i )           Ĝ = the pooled climatological CDF
CRPS_ORD  = (1/n) Σ_i crps_step( PAV-recalibrated F̂(x_i) , y_i )     ⬅ one isotonic fit per threshold
DSC       = UNC − CRPS_ORD                            the ordering information AVAILABLE in x
R*        = DSC / UNC ∈ [0, 1]
MCB(arm)  = CRPS(arm) − CRPS_ORD                      how much of it the EMISSION throws away
            CRPS(arm) = UNC − DSC + MCB(arm)          exactly
```

⭐⭐ **A and B/REAL are ordered by the SAME index**, so they share `UNC` **and** `DSC` and differ
**only in `MCB`**. ⇒ The comparison becomes exactly the right question — *given the ordering
information that is there, which emission wastes less of it?* — and

```
            ΔCRPS = CRPS(B) − CRPS(A) = MCB_B − MCB_A          EXACTLY
```

which is `G-S1`, a free arithmetic gate on the whole readout.

⚠️ `DSC_ORD = 0` means **no isotone-INCREASING** information, not none (`01_MATH.md` B3). The sign
of `x` is **pre-declared increasing** and `DSC_ORD(−x)` is printed beside it.

---

## 4 GATES — every one of them refuses, none of them repairs

| id | assertion |
|---|---|
| **G-S1** | ⭐⭐ `ΔCRPS == MCB_B − MCB_A` to **1e-12**, every row |
| **G-S2** | `MCB_A ≥ 0` **and** `MCB_B ≥ 0` — a theorem in-sample; a negative one means the PAV refit is wrong |
| **G-S3** | `DSC ≥ 0` — a theorem (`01_MATH.md` §3.4); ⛔ **a gate, not a result** |
| **G-S4** | `UNC` identical across **all** rows to 1e-12 — the fixed-population claim, asserted not stated |
| **G-S5** | A and B scored on the **identical** cell index, per row: same `n`, same index array |
| **G-S6** | B's cube has **exactly 16** draws, matching A — so no row's CRPS carries an `S` advantage |
| **G-S7** | the PAV fit is genuinely isotonic and weight-preserving (lifted from `01_MATH.md` `G-R3`) |

**VOID triggers.** ⛔ Any of these ⇒ **VOID**, never *"B is worse"* and never a recommended τ:
**V7** any gate above fails. **V8** `B/NOISE` is not worse than `B/REAL` in a row ⇒ that row has no
power and no statement is made about it. **V9** no scoreable artifact (a crash is never evidence).

---

## 5 ⛔ PREDICTIONS — logged BEFORE the run, as `W-137`…`W-140`

⚠️ Base rate `reports/PREDICTION_LEDGER.md`: **WRONG 140 / RIGHT 89**. All three of this epic's
previous predictions were **wrong**.

| # | prediction | confidence |
|---|---|---|
| **P-4** | τ = 0.26 (the calibrated value) shrinks `ΔCRPS` by **more than 20%** against τ = 0.20 | ~70% |
| **P-5** | `ΔCRPS` stays **positive** — B loses — at **every** τ in the sweep | ~60% |
| **P-6** | `soft_gate` gives a **smaller** `ΔCRPS` than `threshold_gate` at τ = 0.20 | ~55% |
| **P-7** | `R*` on the composed index **falls** as τ rises above 0.26 | ~50%, a coin flip, labelled |

---

## 6 ⛔ SCOPE

**IN:** τ and the composition as free parameters; the `MCB − DSC + UNC` attribution of the CRPS
already reported; `B/NOISE` as the per-row potency control.

**OUT:** ⛔ recommending a τ for production — this measures whether τ changes the **verdict**, not
what τ should ship, and a τ chosen on 7 folds of one vehicle is not a shipping decision.
⛔ training, GPU, production code, `mu` (unavailable), the h36 rollout, re-opening λ\*.

⚠️ **n = 1 vehicle.** `σ̂ = 0.4441` (E-5) is a within-B margin on the τ = 0.20 population and does
**not** transfer to a `ΔCRPS` on this fixed population. ⇒ **No margin is claimed here**; intervals
are origin-blocked over the 7 folds and that is all.
