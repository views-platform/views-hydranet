# Outcome — FSS through the five-transform battery (Stage 1b)

**Date:** 2026-10-26 · **Prereg:** `2026-10-26_fss_battery_prereg.md` (`571ea76`, committed before
the script existed) · **Run:** `2026-10-26_body_optimisation_gap/fss_battery.{py,out}`

**Population, stated with every number (C-351):** 18 pooled fields (6 steps × 3 targets), fixed
24×24 window, 10,368 cell-values, control checkpoint, `lesson_297_win_1`, **in-sample**. FBS pooled
across fields before the ratio. `f_o = 0.009452` on this window ⇒ **bar = 0.504726**.
Events per field: min 0, median 1, max 19.

**Gates:** F-1 (ideal-band `scale_min = 2D`) PASS · F-2 (truth byte-identical) PASS · F-3 (no wrap)
PASS · C-1 (`thresh_ratio(r=1)` ≡ production `matched_pred_thresh`) **0.000e+00** · C-2
(`fss_pooled` on one field ≡ production `fss`) **0.000e+00**. **No VOID condition fired.**

---

## 1. ⭐⭐ The headline: FSS replaces B-1 on placement, and the two instruments are complementary

Both carried on the **identical** fields, counts space, `n=1` for FSS:

| transform | FSS (`n=1`) | `scale_min` | B-1 (`d=1`) | Spearman vs truth |
|---|---|---|---|---|
| baseline | 0.5204 | 1 | 0.4465 | +0.1592 |
| **T-4 displace** `s=1` | 0.2143 | 3 | 0.4466 | +0.1517 |
| **T-4 displace** `s=2` | 0.0408 | 7 | 0.4464 | +0.1249 |
| **T-4 displace** `s=3` | **0.0000** | 9 | **0.4422** | +0.0940 |
| **T-3 smooth** 1 pass | 0.3571 | 3 | **0.1710** | +0.1571 |
| **T-3 smooth** 2 pass | **0.3776** | 3 | **0.1116** | +0.1563 |
| **T-3 smooth** 3 pass | 0.3673 | 3 | **0.0855** | +0.1544 |

⭐ **D-1 FIRES.** `scale_min` increases **strictly** with displacement (1 → 3 → 7 → 9) and FSS at
grid scale collapses to **exactly 0.0000** at `s=3`. **FSS replaces B-1 as the placement
diagnostic.** B-1 on the same fields moves by `0.0043` across the whole displacement sweep while
Spearman-vs-truth falls **41%**.

⚠️ **But the exact ruler does NOT transfer.** On ideal fields `scale_min = 2D` exactly (`8c2d537`).
Here: 3, 7, 9 against `2s` = 2, 4, 6 — **consistently high by 1–3 cells**, because our events are
blobs, not the paper's 1-cell band. ⇒ **`scale_min` is ORDINAL on our field, not a calibrated
ruler.** The prereg committed only to monotonicity, which is why this is a limit and not a W.

⭐ **And B-1 is rehabilitated in its proper role.** It is the one instrument here that moves
monotonically under blur — **5.2×** across three passes — while being flat under displacement. ⇒
**B-1 is a dispersion instrument and always was.** This morning's verdict narrows: retired **as the
placement diagnostic**, not as an instrument. See **W-71** — D-2 pre-registered the opposite.

---

## 2. ⭐⭐ P-F1 CONFIRMED EXACTLY: count-matched FSS reads RANK ORDER and nothing else

| comparison | differing binary cells | max \|ΔFSS\| |
|---|---|---|
| counts vs log1p | **0 / 10,368** | **0.000e+00** |
| T-1 `c = 0.5 / 2 / 4 / 16` | **0 / 10,368** each | **0.000e+00** each |

Three consequences, all exact:

1. ⭐ **No amplification loophole.** The exact route that made `_structure_penalty`'s 4.4× gain
   pure amplification is **unavailable** to FSS.
2. ⭐ **Counts ≡ log1p, byte for byte.** The first space-invariant statistic in this programme —
   every other one has had to be reported twice and never mixed.
3. ⛔ **FSS is totally blind to magnitude and calibration.** **D-3 FIRES: FSS is a DIAGNOSTIC and
   never a training target.** A forecast scaled by any `c > 0` satisfies it identically.

**P-F3 confirmed:** count-matching forces `f_M = f_o` ⇒ Roberts & Lean Eq. (8) gives `AFSS = 1.000000`.
FSS here is bias-free **by construction** — the right property, arrived at by luck (**C-374**).

---

## 3. ⭐ T-2: FSS localises damage IN SCALE — the property B-1 and VS-p both lack

| | `n=1` | `n=3` | `n=5` | `n=9` | `n=15` | `n=23` |
|---|---|---|---|---|---|---|
| baseline | 0.5204 | 0.7378 | 0.7813 | 0.8175 | 0.8839 | 0.9514 |
| shuffle `k=2` | **0.2959** | 0.7025 | 0.7697 | 0.8167 | 0.8863 | 0.9519 |
| shuffle `k=8` | **0.0714** | **0.3115** | **0.4900** | 0.7399 | 0.9039 | 0.9639 |

**D-5 does NOT fire.** Damage is confined to `n ≲ k` and recovers to baseline above it — `k=2` by
`n≈5`, `k=8` by `n≈15`. ⇒ **the scale-selectivity transfers from ideal fields to ours.** That is
what a diagnostic for a defect with a *scale profile* (D12: 0.391 at 1 cell, 0.870 at 32) needs.

⚠️ **Observation, not predicted:** at large `n` the shuffled field scores slightly **above**
baseline (`k=8`, `n=15`: 0.9039 vs 0.8839). Within-block permutation spreads events more evenly
across blocks, which marginally improves block-level fractions. Small, and not a route to anything.

---

## 4. ⛔ T-3: the loophole. FSS saturates on blur — and is NOT monotone

`n=1`, counts: `0.5204 → 0.3571 → 0.3776 → 0.3673`. **The second pass scores better than the
first.** Reproduced in log1p (`0.3469 → 0.3878 → 0.3673`) and at `n=3` and `n=15`.

⇒ **D-4 FIRES.** Blur is detected **once** — the first pass costs 0.163 — and further
over-smoothing is **free**. The mechanism is P-F1's: count-matching sees only which cells hold the
top `n_events` ranks, and a box smooth does not move peaks.

**The loophole is BOUNDED** (unlike I-1's unbounded amplification gain), so it is a blind spot
rather than a gradient to exploit. **Registered as C-373, Tier 2.** The mitigation is §1's pairing
with B-1, and the risk is the pairing being forgotten.

**W-70:** P-F2 predicted "monotone worsening — but weakly". The *weakly* was right and derived; the
*monotone* was assumed and is wrong.

---

## 5. T-5 and the threshold sweep

**T-5** — exactly invariant, as pre-registered. FSS consumes a **point** field and has no variance
channel, the same limitation as I-1 and I-2. ⇒ **no instrument tested in either battery can see
per-cell predictive variance.**

**Threshold rule** (`f_M/f_o`, both giving `AFSS = 0.80`):

| | `n=1` | `n=9` | `n=23` |
|---|---|---|---|
| `r = 0.5` | **0.5342** | 0.7113 | 0.8016 |
| `r = 1.0` | 0.5204 | 0.8175 | 0.9514 |
| `r = 2.0` | 0.4422 | 0.7262 | 0.7785 |

⛔ **Under-firing the event count IMPROVES grid-scale FSS** (0.5204 → 0.5342). Count-matching closes
that route; nothing in the repo records it as doing so. **C-374, Tier 3.**

---

## 6. The arm — DESCRIPTIVE ONLY (prereg §7.5; `n=1`, C-307)

| | `n=1` | `n=5` | `n=9` | `n=23` | Spearman |
|---|---|---|---|---|---|
| control | 0.5204 | 0.7813 | 0.8175 | 0.9514 | +0.1592 |
| arm | **0.5918** | **0.8815** | **0.9335** | **0.9869** | +0.1589 |

⚠️ **The arm reads BETTER on FSS at every scale** while composed-forecast Spearman is flat
(+0.1592 vs +0.1589). ⛔ **This is not a verdict and must not be quoted as one:** `n=1`, in-sample,
one window, and C-372 means no σ̂ exists to size it against. ⛔ **Do not set this beside §4c's
body-vs-truth Spearman fall (+0.147 → +0.098)** — that is a different statistic on a different
field, and mixing them is exactly the C-351/"never mix two statistics" failure.

---

## 7. Verdict

| rule | fires? | consequence |
|---|---|---|
| **D-1** | ⭐ **YES** | FSS replaces B-1 as the placement diagnostic; bar `0.5 + f_o/2` on the stated population |
| **D-2** | partially — ⛔ **its stated consequence is FALSE** | FSS is placement-only, **but** B-1 covers dispersion. **W-71** |
| **D-3** | ⭐ **YES** | FSS is a diagnostic, **never** a training target. Goes in the glossary |
| **D-4** | ⛔ **YES** | blur loophole; **C-373** |
| **D-5** | no | FSS *is* scale-selective on our field |
| **D-6** | ⭐ binding | **no arm proposed.** C-372 (σ̂ unmeasured, views-models #572/#566) blocks it |

**What is still missing after this document:** a diagnostic for **per-cell predictive variance**
(no instrument in either battery has a variance channel), and **σ̂**, without which no arm can carry
a decision rule.

⛔ **Limits unchanged from the prereg:** one window, one checkpoint, in-sample, ~1 event per field
median, `n=47` unavailable so `AFSS` is derived not observed, and **I am both author and auditor** —
the independence `/falsify` guard mode requires is absent.
