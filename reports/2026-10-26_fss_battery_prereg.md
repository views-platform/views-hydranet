# Pre-registration — FSS through the five-transform battery (Stage 1b)

**Date:** 2026-10-26
**Branch:** `prereg/body-optimisation-gap`
**Predecessor:** `reports/2026-10-26_instrument_battery_prereg.md` (Stage 1, `c0cdd03`) and its run
(`b4da301`), which **retired B-1** as the placement diagnostic.
**Written before the script exists.** Same discipline as Stage 1.

---

## 0. ⛔ What this is, and what it is NOT

This puts **one** instrument — the Fractions Skill Score — through the **same five transforms**
that retired B-1 this morning. It is the discipline Stage 1 established, applied to Stage 1's own
proposed replacement. ⛔ **FSS is not adopted by this document on any outcome**; it is tested.

⚠️ **One result is already in hand and is NOT blind.** `fss_skill_threshold.py` (`8c2d537`)
confirmed Roberts & Lean's `scale_min = 2D` on **ideal** fields, and that `matched_pred_thresh`
forces `f_M = f_o` so `AFSS = 1`. Those are stated as **controls (F-1)**, not findings. Everything
about FSS **on our field** below is blind.

---

## 1. Hypothesis

> **H.** FSS is fit to replace B-1 as the body programme's **placement** diagnostic if and only if
> it (a) worsens monotonically under displacement, (b) localises each transform's damage **in
> scale**, and (c) carries no cheap-satisfaction route of the kind that killed `_structure_penalty`
> (amplification) and B-1 (displacement-blindness).

**Why FSS and not something new.** It is already built (`scripts/sharpness_scorecard.py:78`),
already tested (`tests/test_field_sharpness.py`), and its definition matches Roberts & Lean's
Eqs. (5)–(7) line for line. ⛔ The programme's recorded failure mode is adopting an instrument on
its definition; being already-built is a cost argument, not an evidence one.

---

## 2. The intervention — the same five transforms

Applied to the **control's** composed forecast field, truth held fixed. Population: the matched
`lesson_297_win_1` biopsy, byte-identical truth across arms, 18 fields (6 steps × 3 targets).

⭐ **METHOD CHANGE FROM STAGE 1, and it is an improvement.** Every transform runs on the **same
fixed 24×24 window** (`DISP_PAD=4` offset), not 32×32 for T-1/2/3/5 and 24×24 only for T-4. The
population is then **identical across all five**, so a difference between transforms is the
transform. Stage 1 had that confound in four of its five rows.

| | transform | parameter swept |
|---|---|---|
| **T-1** | amplify, `f → c·f` | `c ∈ {0.5, 1, 2, 4, 8, 16}` |
| **T-2** | permute cells within each `k×k` block | `k ∈ {2, 8}` |
| **T-3** | 3×3 box smooth | `1, 2, 3` passes |
| **T-4** | displace by `s`, fixed truth window, **no wrap** | `s ∈ {0, 1, 2, 3}` |
| **T-5** | inflate per-cell predictive variance, mean field EXACTLY fixed | `τ²/var(truth) ∈ {0, 0.25, 1, 4, 16}` |

**Free parameters, swept — not fixed and quoted (C-351):**

| | parameter | values |
|---|---|---|
| `n` | neighbourhood length | `{1, 3, 5, …, 23}` — the x-axis; FSS is defined on odd `n` |
| `space` | counts / log1p | both |
| `f_M/f_o` | the threshold rule | `{0.5, 1.0, 2.0}` × count-matched |

⚠️ **`n = 2N−1 = 47`, the asymptote, is unavailable on a 24-wide window.** `AFSS` is therefore
**not measured here**; it is derived analytically in §3 (P-F3) instead. Stated, not hidden.

**Aggregation, declared in advance:** FBS and its worst-case are **pooled across all 18 fields**
before the ratio is formed, not averaged per field. A 24×24 window holds ~2–3 events, so a
per-field FSS is dominated by its own event count. ⛔ This is the C-351 rule: the population is
"18 pooled fields", and it is stated with every number.

---

## 3. ⭐ PRE-COMMITTED PREDICTIONS

| | **FSS, count-matched** |
|---|---|
| **T-1 amplify** | ⭐⭐ **EXACTLY invariant**, `0.000e+00`, every `c`, every `n` |
| **T-2 shuffle** | **worsens at `n ≤ k`, recovers to ~baseline by `n ≫ k`** |
| **T-3 smooth** | ⭐ **monotone worsening — but WEAKLY**; the loophole to look for |
| **T-4 displace** | ⭐⭐ **worsens; `scale_min` strictly increases with `s`** |
| **T-5 inflate variance** | **exactly invariant** (no variance channel) |

**P-F1 — ⭐⭐ count-matched FSS is a function of the forecast's RANK ORDER ALONE.** `matched_pred_thresh`
picks the `n_events`-th largest forecast value; the binarised field is therefore determined by which
cells occupy the top `n_events` ranks, and by nothing else. Three exact consequences, all testable
to `0.000e+00`:

1. **T-1 invariance.** `f → c·f` for `c > 0` is strictly monotone ⇒ identical ranks ⇒ identical
   binary field ⇒ identical FSS at every `n`. ⭐ **FSS has NO amplification loophole** — the exact
   route that made `_structure_penalty`'s 4.4× gain pure amplification.
2. **counts ≡ log1p.** `log1p` is strictly monotone ⇒ the two spaces give **byte-identical** binary
   fields and **identical FSS**. ⭐ No other statistic in this programme has been space-invariant;
   every one has had to be reported twice and never mixed.
3. ⛔ **FSS is therefore completely blind to magnitude and to calibration.** It can only ever be
   the **placement** half of the diagnostic. ⛔ **It cannot be a training target on its own**, and
   anyone who adopts it as one has adopted a score a constant forecast of `c·f` satisfies equally.

⇒ **P-F1 is the prediction I most want to be wrong about consequence 3 and least expect to be.**

**P-F2 — ⭐ T-3 is where FSS should be weak, and that is the result that matters.** Count-matching
means FSS sees only *where the top `n_events` cells are*. A 3×3 box smooth spreads mass but does
**not** move the peaks, so the top-ranked cells stay near where they were. The repo's own
calibration table already hints at this: `fss_ratio` moves `1.000 → 0.897` under blur `σ=2` but
`→ 0.267` under a 3-cell displacement. ⇒ **FSS should worsen under T-3 monotonically but by far
less, per unit of damage, than under T-4.** ⛔ If so, the body's measured defect — over-smoothness
— is **not** what FSS is sensitive to, and a *structure* diagnostic is still missing even after
this document.

**P-F3 — AFSS = 1 analytically, so FSS carries ZERO bias information.** Roberts & Lean Eq. (8):
`AFSS = 2 f_o f_M / (f_o² + f_M²)`. Count-matching sets `f_M = f_o` ⇒ `AFSS = 1` identically.
Already confirmed numerically (`8c2d537`). ⇒ FSS is a **pure placement instrument by construction**.
⚠️ That is the right property for the job and it was **luck, not design** — nothing in the repo
records it as intentional. The `f_M/f_o ∈ {0.5, 2.0}` sweep exists to show what the instrument does
once that luck is removed.

**P-F4 — ⭐⭐ `scale_min` is a RULER, and this is the contrast with B-1.** On ideal fields
`scale_min = 2D` exactly (`8c2d537`, D = 1…11). On our field, events are blobs rather than a
1-cell band, so `2s` is an approximation. The committed prediction is the **direction and
monotonicity**: `scale_min(s=0) < scale_min(s=1) ≤ scale_min(s=2) ≤ scale_min(s=3)`, strictly
increasing overall. ⛔ The comparator is B-1 on the identical transform, which went
**0.2727 → 0.3338 — the wrong way** — while Spearman-vs-truth fell 41%.

---

## 4. Method

1. Load the matched biopsy for both arms. **Assert `y_reg` byte-identical or exit VOID** (F-2).
2. **Reproduce the ideal-band control** `scale_min = 2D` inside this run (F-1). If it does not
   reproduce, this is not the same instrument and nothing else is interpretable.
3. Re-use `fss` and `matched_pred_thresh` **verbatim** from `scripts/sharpness_scorecard.py` —
   the production functions, not a copy. Re-use the five transforms **verbatim** from
   `instrument_battery.py`.
4. Apply T-1…T-5 on the fixed 24×24 window, sweeping every parameter in §2.
5. Carry **R-1 (Spearman vs truth)** and **I-2 (B-1 at d=1)** alongside, unchanged, so the
   comparison with the retired instrument is like-for-like on identical fields.
6. Report every value. ⛔ **Quote no scalar without its sweep.**

---

## 5. Decision rules, committed in advance

| | rule |
|---|---|
| **D-1** | **If P-F4 holds (`scale_min` strictly increases with `s`):** FSS **replaces B-1** as the placement diagnostic. `reports/2026-10-21_body_scoreboard.md` gains the correction, and the bar is `FSS > 0.5 + f_o/2` on the stated population. |
| **D-2** | **If P-F2 holds (FSS weak on T-3):** FSS is adopted as a **placement** instrument ONLY. ⛔ The report must then say plainly that the body's **over-smoothness** defect has **no** diagnostic — not stretch FSS to cover it. |
| **D-3** | **If P-F1 holds:** FSS is recorded as a **DIAGNOSTIC and never a training target**, with the reason (rank-order ⇒ magnitude-blind) written into `reports/GLOSSARY.md`. |
| **D-4** | **If FSS does NOT worsen monotonically under T-3, or improves:** that is a blur loophole of the same class as I-1's amplification. Register it, and FSS does not replace anything until a transform that catches it exists. |
| **D-5** | **If FSS fails to localise T-2 in scale** (no recovery at `n ≫ k`): it is not scale-selective on our field, the `2D` ruler does not transfer from ideal fields, and D-1 is withdrawn. |
| **D-6** | ⛔ **No arm is proposed from this document under any outcome.** C-372 (σ̂ unmeasured, views-models #572/#566) blocks that. Stage 1b chooses a diagnostic; it does not authorise an experiment. |

---

## 6. Falsifiers and VOID conditions

| | |
|---|---|
| **F-1** | ⛔ **VOID if the ideal-band control does not reproduce** `scale_min = 2D` for `D ∈ {1,2,3,5,8,11}` on 128×128. |
| **F-2** | VOID if the two arms' `y_reg` are not byte-identical. |
| **F-3** | VOID if T-4 wraps at the boundary rather than cropping to a fixed truth window. |
| **F-4** | **P-F1's bar, written to be satisfiable.** The claim is an exact algebraic identity on a deterministic computation, so the bar is **byte-identical binary fields**. ⚠️ `matched_pred_thresh` subtracts `1e-12`, an absolute fudge sitting at a different scale in counts than in log1p, so a tie within `1e-12` could flip one cell. ⇒ **Report the COUNT of differing cells.** P-F1 is refuted only if **FSS values differ at the 4th decimal**; a nonzero cell count with identical FSS is an *observation*, not a refutation. ⛔ This wording exists because **four previous falsifiers in this programme were unsatisfiable by any correct answer** (W-69). |
| **F-5** | ⚠️ **Not a VOID, a stated limit:** `truth_thresh = 0` — "any event at all". A 24×24 window holds ~2–3 events, so a higher threshold leaves nothing to score. **A verdict here is a verdict about the any-event threshold** and does not transfer to intensity thresholds, which is what Roberts & Lean actually sweep. |

---

## 7. Limits, binding

1. ⛔ **One window, one pair of checkpoints, in-sample.** 24×24, lesson 297, ~2–3 events per field.
   Nothing here generalises to the 64,818-cell grid, and FSS's behaviour on a 99.65%-empty global
   field is **not measured by this**.
2. ⛔ **`n = 47` is unavailable**, so the asymptote is derived, not observed (§2, P-F3).
3. ⛔ **The battery can only refute, never endorse.** A term that passes all five can still be the
   wrong objective. **The transforms are the ones I could think of** — and this whole document
   exists because three instruments carried loopholes their designers did not imagine.
4. ⛔ **I am both the author and the auditor.** The independence `/falsify` guard mode requires is
   absent. P-F2 and the F-4 wording came from auditing my own plan, which is better than nothing
   and weaker than a second seat.
5. ⚠️ **The arm's FSS is reported as DESCRIPTIVE, not as a verdict.** `n=1` and C-307: a null is
   inconclusive, not negative.

---

## 8. Provenance

| | |
|---|---|
| source method | `~/brain/9_library/papers/Roberts2008_ScaleSelectiveVerification.pdf` — Eqs. (5)–(7) p. 82, Eq. (8) `AFSS` p. 82, `FSS_uniform = 0.5 + f_o/2` and `scale_min = 2D` p. 83. **Read directly from the PDF**; claims not yet extracted into the library. Fetched by `library` 2026-10-26. |
| implementation | `scripts/sharpness_scorecard.py:78` (`fss`), `:92` (`matched_pred_thresh`) — production, re-used not copied |
| fields | `hf_control_20261025/.../biopsy_stage_5:_training_forensic_(lesson_297_win_1).npz` + the arm's matched file |
| predecessors | `instrument_battery.py` (`b4da301`), `fss_skill_threshold.py` (`8c2d537`) |
| risks | **C-372** blocks any arm. **C-370**/**C-371** concern VS-p, not FSS. |

---

## Amendment 1 — 2026-10-26, after the run: TWO of my own pre-registrations were wrong

⛔ Recorded here rather than adjusted, because changing a pre-registration after seeing the number
it produces is worse than the error.

**A1-1 — P-F2 predicted "monotone worsening" under T-3. FSS is NOT monotone.** Measured at `n=1`,
counts: `0.5204 → 0.3571 → 0.3776 → 0.3673` for 0/1/2/3 smoothing passes. The **second** pass
scores **better** than the first. Same shape in log1p (`0.3469 → 0.3878 → 0.3673`) and at `n=3`,
`n=15`. The *direction* of P-F2 ("weak on blur") was right; the *monotonicity* was wrong, and
**D-4 fires on the version I committed.** Logged as **W-70**.

**A1-2 — D-2's stated consequence is refuted by this run's own control column.** D-2 commits me to
saying "the body's over-smoothness defect has **no** diagnostic". That is false: **B-1**, carried
alongside as the retired instrument, moves `0.4465 → 0.1710 → 0.1116 → 0.0855` under the same
three smoothing passes — strongly monotone, a 5.2× swing, while being **exactly flat** under
displacement (`0.4465 → 0.4466 → 0.4464 → 0.4422`). ⇒ **B-1 is a dispersion instrument and always
was.** This morning's verdict stands as narrowed: B-1 is retired **as the placement diagnostic**,
not as an instrument. I wrote D-2 assuming FSS was the only candidate on the table, with B-1's
replacement column already in my own script. Logged as **W-71**.

**Nothing else changed.** F-1, F-2, F-3, C-1, C-2 all PASS; no VOID condition fired; P-F1, P-F3 and
P-F4 hold as written; D-1, D-3, D-4, D-6 fire as written.
