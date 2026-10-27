# S2 (#460) — the family-free conditional CDF reproduces, and its "out-of-sample optimum" does not

**Date** 2026-10-27 · **Issue** #460 (epic #458) · **Script**
`reports/2026-10-27_family_free_head/family_free_cdf.py`, committed at `ad3b007` **before it ran** ·
**Output** `family_free_cdf.out`

## C-351 slots

| slot | value |
|---|---|
| population | fit **36 months 421–456 × 64,818 land cells = 2,333,448** cell-months; score **13 months 457–469 × 64,818 = 842,634** cell-months |
| transform | count space, `ged_sb_best`, no normalisation |
| estimator | CRPS from the bin's **empirical CDF**, both the plug-in `1/n²` and the fair `1/(n(n−1))` columns, `n` = that bin's own fit population |
| index | `E[gate × mu]` — **per draw, then averaged** (⛔ not `E[gate]×E[mu]`, C-386) |
| binning | **equal-COUNT** bins on the fit window's index. ⚠️ M67 does not state which rule it used. |
| normalised | no |
| cube | `d4_result.npz` sha `4acbf4d3c2b457b0…`, family `nb`, 4 MC-dropout draws |

## What reproduces

| | M67 | here | |
|---|---|---|---|
| headline, B=8,000 | 0.039327 | **0.039186** | 0.36% apart |
| ladder: loses below 250 | yes | **0.048389 vs 0.046023** | ✅ |
| ladder: crosses ~1,000 | yes | **0.042003** | ✅ |
| ladder: argmin | 8,000 | **8,000** | ✅ |
| curve turns over at 16,000 | yes | **yes** (plug-in) | ✅ |
| event-cell law | 10.831 | **10.799** | 0.3% apart |
| score population | 842,634 | **842,634** | exact |
| zero fraction | — | 0.996552 | — |

**Void gates as declared before the run:** G-b **PASS**, G-c **PASS**, G-a **FAIL** (tolerance 5e-5,
actual difference 1.41e-4), G-d **FAIL** (background 0.001964 vs 0.000260).

⇒ **The method reproduces.** An independently written script, from a committed docstring, recovered
the ladder's four qualitative facts and the headline to 0.36% on a number that had **no code behind
it**. #443's core claim stands: a family-free lookup on the model's own `gate × mu` index beats the
best parametric arm, here by **−14.9%** (plug-in) / **−15.3%** (fair at 8,000) against M67's −17.6%.

## ⭐⭐ The finding: M67's "out-of-sample optimum, not an overfit" is the estimator's 1/n bias

| bins | n per bin | plug-in | fair |
|---|---|---|---|
| 4,000 | 583.4 | 0.040135 | 0.040050 |
| 8,000 | 291.7 | 0.039186 | 0.039026 |
| 16,000 | 145.8 | 0.039287 ⬆️ | 0.038971 ⬇️ |
| 32,000 | 72.9 | 0.039493 ⬆️ | **0.038856** ⬇️ |

⛔ **The plug-in turns over. The fair estimator never does** — it improves monotonically to 32,000
bins, the finest point swept, scored on 13 months the fit never saw.

**The mechanism is measured, not asserted.** Equal-count bins mean `n` halves every time the bin
count doubles, and the plug-in's bias is `E|X−X'|/(2n)`:

| bins | n per bin | plug − fair | `(plug−fair)·n` | ratio to previous |
|---|---|---|---|---|
| 8,000 | 291.7 | 0.000160 | 0.0467 | — |
| 16,000 | 145.8 | 0.000316 | 0.0461 | **1.975** |
| 32,000 | 72.9 | 0.000637 | 0.0464 | **2.016** |

`(plug−fair)·n` is flat to **1.3%** and the bias **doubles as `n` halves** — 1.975 and 2.016 against
a predicted 2.000. That is #459's signature to 2%, in exactly the three points where the turnover
was claimed.

⇒ ⛔ **M67's inference — "bin count swept, turns over at 16,000 ⇒ an out-of-sample optimum, not an
overfit" — is withdrawn.** The reasoning was sound given the number; the number was the ruler. The
turnover is our scoring estimator charging a per-bin small-sample penalty that grows as the bins
shrink, and it crosses the method's true gain at ~8,000.

⇒ ⭐ **The method is better than published and its ceiling is unlocated.** We stopped turning the
dial where a measurement artifact told us to stop.

⚠️ The product is **not** flat below 4,000 bins (0.0933 / 0.0747 / 0.0560 / 0.0525), for two
reasons that are both expected and neither of which weakens the above: wide bins mix heterogeneous
cells, so within-bin `E|X−X'|` is genuinely larger; and at B=50 the difference is 0.000002, at the
printed precision, so those rows carry ~50% relative uncertainty and are not interpretable.

## ⛔ What does NOT reproduce, left open

**G-d, the background term.** M67 reports the lookup **ties** the delivered model on the quiet
background, **0.000260 vs 0.000266**. This script measures **0.001964** — **7.6× higher**. The
event-cell half of the same gate reproduces to 0.3% (10.799 vs 10.831).

⇒ The split, not the method, is what differs. Candidates, none tested:

1. **The definition of "background."** Here it is cell-months with `truth == 0` (`~ev`). M67 may
   partition on a **low index** instead, which would route the high-index-but-empty cells — the
   expensive ones — into the event half.
2. **The binning rule.** Equal-count was this script's declared choice; M67 does not state one.
   Equal-width bins would place the background mass very differently.
3. **M67's own estimator.** It says *"same m=16 plug-in estimator"*, but a bin here holds ~292
   observations. If M67 subsampled each bin to m=16, its plug-in bias would be larger — and its
   headline **is** larger, 0.039327 vs 0.039186, in the right direction and roughly the right size.

⚠️ M67's two comparators (0.046023 best parametric, 0.050758 delivered) were **deliberately not
gated**: this script cannot tell which composition produced each, so they are quoted as targets
only. Whether they reproduce is open, not claimed.

## Cost

Wall time **5.3 s**, peak RSS **3.22 GiB**, CPU only.

## Consequences

- ⛔ **#443 keeps its finding and loses its overfit argument.** The monotone out-of-sample fair
  curve is *stronger* evidence against overfitting than a turnover was — it improves on held-out
  months all the way down to 73 observations per bin — but the specific sentence about the 16,000
  turnover is retracted.
- ⛔ **"8,000 bins" is not a result.** Any later work that treats it as the operating point is
  building on the artifact. The fair optimum is beyond 32,000 and unmeasured.
- ⭐ **#459 has claimed its first published conclusion**, within an hour of being measured, and the
  casualty was an inference, not a number. Every arm scored at low draw count carries the same
  penalty, by an amount that scales with that arm's own spread.
- 🔵 **Open:** the background decomposition (G-d), the two ungated comparators, and the location of
  the fair optimum beyond 32,000 bins.
