# Pre-registration — the instrument battery (Stage 1)

**Written:** 2026-10-26, **before** the battery script exists.
**Branch:** `prereg/body-optimisation-gap`.
**Cost:** CPU, minutes, fields already on disk. No pod, no training, no draws from Simon.
**Governs:** C-370, C-371, C-372 · follows the dissolution of C-366 (`c3ec352`).

---

## 0. ⛔ What this is, and what it is NOT

This pre-registers a test **of three measuring instruments**, not of a model. Nothing is trained,
no checkpoint changes, and no claim about conflict is made. The object under test is:

| | instrument | implemented at |
|---|---|---|
| **I-1** | `_structure_penalty` — one-sided log shortfall of within-block variance | `training_engine.py:402-476` |
| **I-2** | **B-1** — `[γ_pred(d)/γ_true(d)] / [var_pred/var_true]` | `body_scoreboard.py:93-111` |
| **I-3** | **VS-p** — the variogram score, Scheuerer & Hamill 2015 Eq. 3 | not yet written |

⛔ **I-1's answers are ALREADY KNOWN** for three of the five transforms — they were measured this
morning (`c3ec352`). I-1 is therefore the **positive control**: it is here to prove the battery
reproduces a known result, and **its entries in §3 are not blind predictions and are not scored as
such.** Saying otherwise would be backfilling. I-2's and I-3's entries *are* blind.

---

## 1. Hypothesis

> **H.** A term is fit to be a training target for spatial structure only if it responds correctly
> to all five controlled transforms of a forecast field. Each of the three instruments fails at
> least one, and the pattern of failures — not any single verdict — determines whether a loss-term
> route to the body's structure defect exists at all.

**Why this and not an arm.** Three structure instruments have now been found to carry a
cheap-satisfaction route: I-1's amplification (measured, after $3 of pod time), I-3's variance
inflation (C-370, derived before building), I-2's impropriety as a target (`Bolin2023` — a
normalised ratio). **Only the one caught in advance was caught by running a battery first.**

---

## 2. The intervention — five transforms, one variable each

Applied to the **control's** composed forecast field, truth held fixed. Population: the matched
`lesson_297_win_1` biopsy, **byte-identical truth across both arms**, 18 fields (6 steps × 3
targets), 32×32, 18,432 cell-values. Space **swept**: counts and log1p.

| | transform | parameter swept |
|---|---|---|
| **T-1** | amplify, `f → c·f` | `c ∈ {0.5, 1, 2, 4, 8, 16}` |
| **T-2** | permute cells within each `k×k` block | `k ∈ {2, 8}` |
| **T-3** | 3×3 box smooth | `1, 2, 3` passes |
| **T-4** | **displace** — translate the field by `s` cells, scored on the valid overlap only | `s ∈ {1, 2, 3}` |
| **T-5** | **inflate per-cell predictive variance**, mean field held EXACTLY fixed | `τ²/v̄ ∈ {0, 0.25, 1, 4, 16}` |

⛔ **T-4 must crop to the valid overlap, not wrap.** A wrapped translation manufactures an edge
discontinuity that every one of these instruments would read as structure. If the implementation
wraps, the run is VOID.

⚠️ **T-5 is not defined for I-1 and I-2 as they are implemented.** Both consume a *point* field
(`log1p(E[y])`); neither has a variance channel. That asymmetry is itself a pre-registered
prediction (§3) and not a defect in the battery.

---

## 3. ⭐ PRE-COMMITTED PREDICTIONS

Written before the script exists. **`exact` means to floating-point, not approximately** — the
standard set this morning, where two invariances came back at `0.000e+00`.

| | **I-1 penalty** | **I-2 B-1** | **I-3 VS-p (p=2)** |
|---|---|---|---|
| **T-1 amplify** | ⚪ *known:* monotone **improvement**, unbounded | **EXACTLY flat** (re-check) | ⭐ **interior minimum at finite `c`** |
| **T-2 shuffle k=2** | ⚪ *known:* **exactly 0.000e+00** | **moves**, B-1(d=1) rises | ⭐ **moves — and MAY IMPROVE** |
| **T-3 smooth** | ⚪ *known:* **worsens** | **worsens** | **worsens** |
| **T-4 displace** | **worsens** | ⭐⭐ **EXACTLY invariant** | ⭐ **worsens**, via the cross term |
| **T-5 inflate variance** | **exactly invariant** (no variance channel) | **exactly invariant** (no variance channel) | ⭐⭐ **interior minimum at `τ* > 0`** |

**The four that carry the result, each with the derivation that makes it risky:**

**P-1 — I-2 is EXACTLY blind to displacement.** B-1 is built from `γ_pred`, `var_pred`, `γ_true`,
`var_true`. Every one is a functional of **one field alone**; no term ever pairs a model cell with
a truth cell. A translation leaves all four unchanged on the valid overlap. ⇒ **B-1 would score a
forecast with perfect structure in entirely the wrong place as perfect.** ⛔ If this holds, B-1
must be retired as the body programme's *placement* diagnostic — and placement is the arm's
measured defect (body-vs-truth Spearman +0.147 → +0.098, §4c). **That is a larger consequence than
the choice of term.**

**P-2 — I-3 has an interior minimum in `τ` (C-370).** For `p=2` with conditionally independent
heads, `E|Xᵢ−Xⱼ|² = (mᵢ−mⱼ)² + vᵢ + vⱼ`. Our field is too smooth, so the model's pairwise moment
sits **below** truth's; raising `v` lifts it toward truth, then past it. A two-sided score must
therefore have a minimum at some finite `τ* > 0` **with the arrangement untouched.**
⇒ **If `τ*` is materially greater than 0, C-370 is CONFIRMED and VS-p needs modification before any
arm. If VS-p worsens monotonically from `τ = 0`, C-370 is REFUTED** and I withdraw it.

**P-3 — I-3 has an interior minimum in `c` too, so it is not amplitude-blind.** By the same
algebra, scaling by `c` scales the model's pairwise moment by `c²`. ⚠️ **This is a limitation of my
own recommendation**: VS-p has a *bounded* amplification route where I-1's is unbounded. Better,
not clean.

**P-4 — I-3 may IMPROVE under T-2, and I would rather it did not.** VS-p's weights depend only on
the **distance** between a pair, never on where the pair is. It therefore constrains the
**variogram**, and two fields with the same variogram are indistinguishable to it. A within-block
shuffle raises short-lag differences, which moves the model's variogram *toward* truth's.
⇒ ⛔ **VS-p is a second-order-structure term and cannot place events.** If T-2 improves it, the
term is answering a different question from the one our defect poses, and §5's D-3 fires.

---

## 4. Method

1. Load the matched biopsy for both arms. **Assert `y_reg` byte-identical or exit VOID** — §4c's
   gate, reused verbatim.
2. Implement **I-3** fresh from Eq. (3) with **truncated inverse-distance weights, radius `r = 4`**
   — ⛔ **declared as a DEVIATION from the paper's own recommendation (C-371)**, which puts positive
   weight on all 4,201,373,124 pairs at grid scale. Propriety survives: the paper grants properness
   *"for any choice of nonnegative weights"*, and zero is nonnegative.
3. Re-use **I-1** and **I-2** verbatim from `c366_two_instruments.py`.
4. Apply T-1…T-5, sweeping each parameter. Report every value; **quote no scalar without its
   sweep** (C-351).
5. Report both spaces. **Never mix them in one statement.**

---

## 5. Decision rules, committed in advance

| | rule |
|---|---|
| **D-1** | **If P-1 holds (B-1 exactly displacement-blind):** B-1 is retired as the placement diagnostic, a replacement is required before any structure arm, and `2026-10-21_body_scoreboard.md` gains a correction. This fires **independently of the term choice.** |
| **D-2** | **If P-2 holds with `τ*` material:** C-370 CONFIRMED. VS-p does **not** go to an arm as written; the next step is a design change (nugget-cancelling weights, or a term on normalised increments), re-run through this same battery. |
| **D-3** | **If P-4 holds (T-2 improves VS-p):** VS-p is a second-order term that cannot place events. It is then **not the answer to our defect**, whatever it does elsewhere, and the loss-term route is reported as not ready. |
| **D-4** | **If no instrument passes all five:** say so plainly. ⛔ The conclusion is "no usable target exists today", **not** "pick the least bad" — picking the least bad is how `_structure_penalty` was chosen. |
| **D-5** | ⛔ **No arm is proposed from this document under any outcome.** C-372 blocks that until σ̂ exists. Stage 1 chooses an instrument; it does not authorise an experiment. |

---

## 6. Falsifiers and VOID conditions

| | |
|---|---|
| **F-1** | ⛔ **The battery is VOID if I-1's three known answers do not reproduce exactly** — monotone improvement under T-1, `+0.000e+00` under T-2 at `k=2`, worsening under T-3. If they do not, this is not the same instrument and nothing else in the run is interpretable. |
| **F-2** | VOID if the two arms' `y_reg` are not byte-identical. |
| **F-3** | VOID if T-4 wraps at the boundary rather than cropping to the valid overlap. |
| **F-4** | VOID if analytic VS-2 and a sampled estimate at `K = 4096` disagree by more than 1% on the untransformed field — that would mean the closed form is wrong, and every VS-p number with it. |
| **F-5** | ⚠️ **Not a VOID, a stated limit:** `p = 2` only. Scheuerer measures `p = 0.5` as the best discriminator for over- and under-dispersion, which is our failure mode. **A verdict here is a verdict about `p = 2`** and does not transfer. `p = 0.5` has no closed form and is Stage 2's business. |

---

## 7. Limits, binding

1. ⛔ **One window, one pair of checkpoints, in-sample.** 32×32, lesson 297. Nothing here
   generalises to the grid, and the instruments' behaviour on a 97.8%-empty global field is not
   measured by this.
2. ⛔ **This tests instruments against *synthetic* transforms, not against reality.** A term that
   passes all five can still be the wrong objective. The battery can only refute, never endorse.
3. ⚠️ **The transforms are the ones I could think of.** The whole reason this document exists is
   that three instruments carried loopholes their designers did not imagine. **A clean pass means
   "survived these five", and the next loophole is by definition not in this list.**
4. ⛔ **I am both the author and the auditor of the battery.** The independence `/falsify` guard
   mode requires is absent here. C-370 and C-371 were found by auditing my own plan, which is
   better than nothing and weaker than a second seat.

---

## 8. Provenance

| | |
|---|---|
| source method | `~/brain/9_library/papers/Scheuerer2015_VariogramScore.pdf` — Eq. (3) p. 1323, weights p. 1324, propriety p. 1324. ⚠️ **Claims not yet extracted** at the time of writing; read directly from the PDF. |
| fields | `hf_control_20261025/.../biopsy_stage_5:_training_forensic_(lesson_297_win_1).npz` and the arm's matched file |
| predecessor | `reports/2026-10-26_body_optimisation_gap/c366_two_instruments.py` (`c3ec352`) |
| risks | **C-370**, **C-371**, **C-372** (`5423738`); **C-366** updated, left open |
