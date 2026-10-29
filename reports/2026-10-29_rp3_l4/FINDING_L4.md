# RP-3 L4 (#472) — the extrapolation limit, measured

**Date** 2026-10-29 · **Scripts** `l4_extrapolation_limit.py` (committed `f98cdb2`, before it ran),
`l4b_the_turnover.py` (committed `a6b772c`, before it ran), `l4c_turnover_attribution.py`
(committed `2e98635`, before it ran) · **Outputs** the three `.out` files beside them ·
**15 gates, 15 PASS**

**Population, once, for everything below (C-351).** Frozen `nb` cube `4acbf4d3…` (300 lessons,
dropout 0.15, ⛔ NOT the delivered weights — C-317). Fit 421–456, 2,333,448 cell-months; score
457–469, **842,634** cell-months, of which **2,905** (0.3448%) carry an event. `ged_sb_best`
counts, untransformed. Index `E[gate × mu]` per draw then averaged (C-386). Equal-**count** bins.
Plug-in `1/n²` CRPS, fair printed beside it. Not normalised. Fit-window maximum **1528**;
score-window maximum **2482**.

---

## 1 · The limit, as a curve

M67's stated limitation — *"it is a lookup and cannot emit beyond what its bin observed, so it
would fail on an unprecedented event"* — has been carried as an argument since 2026-10-22. It is
now measured. A score cell-month **exceeds** when its truth is larger than the largest outcome its
bin observed in the fit window.

| B | exceed / all | exceed / events | their CRPS share | `zero` | `tail` |
|---|---|---|---|---|---|
| 1 | 0.0001% | 0.03% | 4.88% | 0 | 1 |
| 250 | 0.0005% | 0.14% | 6.06% | 1 | 3 |
| 1,000 | 0.0015% | 0.45% | 7.19% | 5 | 8 |
| 4,000 | 0.0042% | 1.20% | 12.85% | 6 | 29 |
| **8,000** | **0.0074%** | **2.13%** | **14.62%** | **16** | **46** |
| 16,000 | 0.0146% | 4.23% | 17.52% | 43 | 80 |
| 32,000 | 0.0241% | 6.99% | 21.96% | 76 | 127 |

⭐ **Monotone in the bin count, as #472 predicted**: the event-cell rate runs 0.03% → 6.99% and the
CRPS share 4.88% → 21.96% across the ladder. ⇒ **the limit gets worse exactly as the method gets
better**, and a scalar here would have been meaningless.

⛔ **Void coverage is 0.000% at every rung** — no score cell-month lands in an empty fit bin, even
at B=32,000. The limit is never "the bin has nothing"; it is always "the bin has something and it
is too small".

---

## 2 · ⛔ Two different failures wear one name, and the issue's instinct was right

The script pre-registered the split, because it decides what L6 may consider:

| at B=8,000 | count | of events | mean CRPS | CRPS share |
|---|---|---|---|---|
| events the bin's law **covers** | 2,843 | 97.87% | 9.3366 | 80.39% |
| ⛔ `zero` — the bin saw **only zeros** | 16 | 0.55% | 1.0625 | **0.05%** |
| ⭐ `tail` — beyond the events it **did** see | 46 | 1.58% | 104.5553 | **14.57%** |

⭐⭐ **`tail` dominates `zero` by 291× in cost.** My pre-registered **H-L4d** said the opposite —
that `exceed_zero` would dominate, because L2 had just shown that 96.0959% of the window sits in
bins with no positive fit member. **It is refuted, and the issue as written was right:** the
relevant lever is a treatment of the **top** bin, not a floor on the low-index bins.

The reason is visible in the severity, and it is the index doing its job: the `zero` exceedances
are **tiny** (truth median 1, p90 1, max 2 — a bin that saw no events was mostly right, and a
single casualty costs ≈1), while the `tail` exceedances are **large** (truth − cap median 2, p90
46, **max 1514**). Big events land in high-index bins, which is what the index is for; what it
cannot do is say *how* big.

⚠️ But the tail exceedances are not wild. `truth / bin max` is **median 1.50×, p90 2.00×, max
3.00×** — the lookup's cap is the right order of magnitude and short by a factor of two.

---

## 3 · The verdict #472 asked for: **neither academic nor the blocker**

- **not academic** — 62 cell-months, **0.0074%** of the window, carry **14.62%** of the total CRPS
  at a mean of **77.85** against the event half's 10.80. A failure on 2.13% of events is not rare
  in the only sense that counts, and #472 said so in advance.
- **not the blocker** — 97.87% of events fall inside the law their bin observed, and even at
  B=32,000 the exceedances carry under a quarter of the score. ⛔ The method does **not** need a
  tail extension before it can be scored; it needs one before its tail behaviour can be
  *defended*.

⇒ ⭐ **H-L4a and H-L4b are both refuted, in the safe direction.** I predicted the event-cell rate
above 10% (measured **2.13%**) and the CRPS share above 50% (measured **14.62%**) — i.e. I
expected this story to kill the method and it does not.

---

## 4 · ⭐⭐ The ladder's turnover IS the limit, at its first step

The committed ladder turns over at B=16,000 (0.039186 → 0.039287 → 0.039493) and **M67 cited that
turnover as evidence of an out-of-sample optimum rather than an overfit**. Put it beside §1's rising
CRPS share and a different reading appears, which L4b derived from the two committed artifacts
**before it had a script**: split each rung's total into the exceeding cells and the rest.

| B | committed total | ⇒ predicted `mass_rest` | measured |
|---|---|---|---|
| 8,000 | 0.039186 | 0.033457 | **0.033458** |
| 16,000 | 0.039287 | 0.032404 | **0.032404** |
| 32,000 | 0.039493 | 0.030820 | **0.030819** |

⭐ **D-L4a held to 1.4e-6** (G-L4l), and **D-L4d** with it: `Δmass_exc` **+0.001155** exceeds
`Δtotal` **+0.000101** at the turnover.

### ⛔ And then the double measurement caught a selection effect

W-51 says numbers under different populations are not like-for-like, so L4b measured the split
**twice** — once by each rung's own exceeding set, once on the 62 cells held **fixed** at B=8,000.
The two tell different stories:

| B | the fixed 62 | the fixed 842,572 |
|---|---|---|
| 4,000 | 79.5542 | 0.034284 |
| **8,000** | **77.8475** ← argmin | 0.033461 |
| 16,000 | 79.7342 | **0.033423** ← argmin |
| 32,000 | 79.8830 | 0.033618 |

⛔ **H-L4e is REFUTED**: the complement does *not* keep improving. Its argmin is B=16,000, not
32,000 — the rest of the window turns over too, one rung later.

⭐⭐ **And the refutation is the finding.** The B-dependent `mass_rest` falls monotonically all the
way to 32,000 (**D-L4b**, measured and true), which reads as *"the method keeps improving on
everything it can represent"* — **and that reading is false.** As B grows the B-dependent `rest`
**sheds its worst members** (62 removed at 8k, 123 at 16k, 203 at 32k), so its mean can fall
without a single cell-month's forecast improving. ⇒ The monotone curve is **composition, not
improvement**, and it was visible only because the split was measured both ways. Registered as
**C-398**.

---

## 5 · The attribution, on a population that stays still

L4c measured — rather than read off a table — who pays for the turnover, on the fixed split where
each rung-to-rung change is exactly additive (**G-L4n**, re-adds to 1e-12):

| step | Δtotal | the 62's part | the rest's part | the 62's share |
|---|---|---|---|---|
| 8,000 → 16,000 | +0.000101 | **+0.000139** | −0.000038 | **+137.2%** |
| 8,000 → 32,000 | +0.000307 | +0.000150 | +0.000157 | +48.8% |

⭐ **D-L4e predicted all four contributions to 3.3e-7** and **D-L4f** held: the first turnover step
is **more than fully** the 62 cell-months' doing, with the rest of the window partially
**offsetting** it; by B=32,000 the two halves split the damage.

### ⚠️ The objection, and the check that answers it

The 62 are selected **at** the anchor rung, so their argmin there is partly a selection artifact.
L4c therefore repeats the attribution on a set that **does not know about B** — the 62 **largest
truths** (range 94–2482, overlapping the exceeding set in only **8** of 62, so a genuinely
different population):

- its curve's argmin is **B=4,000**, one rung *coarser* still;
- its part of the first turnover step is **+0.000220 of +0.000101 = 217.8%**.

⇒ ⭐⭐ **The anchor-free version agrees in direction and is twice as strong.** The finding survives
the objection to its own construction.

And the level is worth stating plainly: those 62 cell-months score **163.1542** at B=8,000, so they
carry `163.1542 × 62 / 842,634 = 0.012005` of the total **0.039186** — ⭐ **30.6% of the entire
score sits on 62 cell-months**, 0.0074% of the window. *(Arithmetic on the two measured numbers
above, not a separate measurement.)*

---

## 6 · What this hands L6, and what it does not

- ⭐ **The lever is the top bin**, not the low-index floor (§2). `exceed_zero` costs 0.05% of the
  score; a floor on the quiet bins would buy nothing here. ⛔ That is the opposite of what L2's
  open question suggested, and it is the cheapest thing L4 settles.
- ⭐⭐ **A single bin count is a compromise between two populations with different optima** — 4,000
  or 8,000 for the extremes, 16,000 for everything else (§4, §5). ⇒ A **non-uniform** rule — fine
  in the body, coarse in the tail — is a third option L6's framing (*tail extension vs scope the
  claim*) does not contain. ⛔ **Not built here**, as #472 instructs.
- ⭐ **The cap is short by a factor of ~2, not by an order of magnitude** (§2), which is what makes
  a bounded extension plausible rather than speculative.

## 7 · ⛔ What is not measured

- **No σ̂.** One cube, one target (`sb`), one split, n=1, 4 MC-dropout draws. Every number here is
  a point estimate with no replicate.
- **The comparator is internal.** Nothing here is scored against the delivered model or the best
  parametric arm; both remain ungated (M67's 0.000266 / 0.046023). The ladder is the method against
  itself.
- **One target.** `ged_ns_best` and `ged_os_best` are in the L3b cube and are not touched here.
- **The `tail` severity is a distribution over 46 cell-months.** Its p90 and max are single
  observations; the median is the only part of it with any stability.
- ⚠️ **"The turnover is the limit" is an attribution, not a mechanism.** It says where the cost
  moved, measured on fixed populations two independent ways. It does not say the lookup would score
  better at B=16,000 if its tail were extended — that is L6's experiment, not this one's claim.

## 8 · The gates

| gate | what it asserts | result |
|---|---|---|
| G-L4a | copy gate: `family_free_cdf.out`'s five numbers at B=8,000, + 1e-12 vs the frozen `binned_cdf` | PASS |
| G-L4b | the per-bin maximum, computed independently of the scorer, agrees with it exactly | PASS |
| G-L4c | the B=1 oracle: the rate equals `mean(y_sco > max(y_fit))` with no binning | PASS |
| G-L4d | the subset derivation: no exceeding cell-month has `truth == 0`, at every B | PASS |
| G-L4e | non-vacuity: 62 exceeding cell-months at the anchor | PASS |
| G-L4f | void coverage < 1% at every B (measured 0.000%) | PASS |
| G-L4g | monotonicity in the bin count (reported, not pass/fail) | PASS |
| G-L4h | the derived bound: the CRPS share ≤ the event half's 0.9502 | PASS |
| G-L4i | copy gate on both committed artifacts: the eight ladder totals and L4's anchor facts | PASS |
| G-L4j | the mass decomposition re-adds to each rung's own total, both splits, 1e-12 | PASS |
| G-L4k | the fixed set is the same 62 indices at every rung | PASS |
| G-L4l | D-L4a's three predicted `mass_rest` values | PASS |
| G-L4m | copy gate on `l4b_the_turnover.out` | PASS |
| G-L4n | the attribution re-adds to Δtotal, 1e-12, scale printed | PASS |
| G-L4o | D-L4e's four predicted contributions | PASS |

**Cost.** 46.3 s / 2.04 GiB, 25.8 s / 2.20 GiB, 25.5 s / 2.22 GiB. CPU, nothing retrained.

## 9 · Ledger

**M83**, **M84** in `RESULTS_LEDGER.md`. **W-103**–**W-106** and **R-46**–**R-49** in
`PREDICTION_LEDGER.md`. **C-398** in `technical_risk_register.md`.
