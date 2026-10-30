# L3a (#471) — the family-free lookup survives every split boundary available; the PUBLISHED split is the most favourable of the eight

> ⛔⛔⛔ **RETRACTED AS A FORECAST RESULT — 2026-10-29.** Every number below is computed from the
> frozen `nb` cube `d4_result.npz`, which is **TEACHER-FORCED**: `vehicle.py:301-307` writes the
> three observed fatality series into the model's three input channels and `d4_mu_on_the_grid.py:143-146`
> steps the model over that volume, so the model is handed last month's truth at every inference step
> and never consumes its own output. The index ranks events at **AUC 0.99777**; the delivered
> autoregressive forecast ranks them at **0.72340**, and only the delivered one decays with horizon.
> Re-running the recalibration with **only the index changed** scores **0.067752** against
> **0.060323** for no model at all.
>
> ⭐ **What still stands:** the arithmetic, the gates, and every comparison made **against the method
> itself or against B=1 on the same cube** — those are internally consistent and were scoped that way
> when written. Read them as **T=0 / teacher-forced** statements about emission families.
> ⛔ **What does not:** any reading of these numbers as forecast skill, and any comparison against the
> delivered model. See **M85**, **C-399**, **W-107**.

**Date** 2026-10-28 · **Script** `l3a_split_sensitivity.py` (committed `363c69f`, Amendment 1
`ba4d47d`, both before their run) · **Output** `l3a_split_sensitivity.out` · **Cube**
`d4_result.npz` sha `4acbf4d3c2b457b0…`, family `nb`, 4 draws · **Cost** 48.6 s, 3.16 GiB, CPU,
nothing retrained, no new forward passes · **All gates PASS**

## 1 · ⭐ The headline: the method holds on every window, including three that share no month with the published one

The internal margin is CRPS at the best bin count against CRPS at **B = 1**, the unconditional
empirical CDF of the same fit window — the model's `gate x mu` index not consulted at all. It is
the one comparison that transports across windows, because both halves of it move with the window.

| b | fit months | score window | B\* | CRPS(B\*) | CRPS(B=1) | **margin** | disjoint from 457–469? |
|---|---|---|---|---|---|---|---|
| 433 | 13 | 434–446 | 8,000 | 0.057239 | 0.086285 | **−33.7%** | ✅ yes |
| 437 | 17 | 438–450 | 16,000 | 0.060200 | 0.087141 | **−30.9%** | ✅ yes |
| 440 | 20 | 441–453 | 16,000 | 0.058662 | 0.083406 | **−29.7%** | ✅ yes |
| 443 | 23 | 444–456 | 16,000 | 0.057136 | 0.080029 | **−28.6%** | ✅ yes |
| 447 | 27 | 448–460 | 8,000 | 0.052147 | 0.073622 | −29.2% | no (4 mo) |
| 450 | 30 | 451–463 | 8,000 | 0.047518 | 0.069809 | −31.9% | no (7 mo) |
| 453 | 33 | 454–466 | 8,000 | 0.042521 | 0.063869 | −33.4% | no (10 mo) |
| **456** | **36** | **457–469** | **8,000** | **0.039186** | 0.060323 | **−35.0%** | ⭐ the published split |

⭐ **Eight of eight windows: the sign is the same, the order of magnitude is the same, and the
spread is 6.4 percentage points.** On four windows that share nothing with the published one, the
model's own index still buys **−28.6% to −33.7%** against its own unconditional baseline. ⇒ The
lookup's advantage is not an artifact of where the split boundary was drawn.

⭐ **B\* is stable at 8,000 or 16,000 across every window.** The ladder is flat-topped, not peaked:
at the published split, B = 8,000 / 16,000 / 32,000 give 0.039186 / 0.039287 / 0.039493 — a
0.8% spread over a 4× change in bin count. ⇒ #469 (L1, "the curve was still improving at 32,000")
is asking about a plateau, not a slope.

## 2 · ⛔ The finding that was not asked for: the published window is the EASIEST of the eight

Read the `CRPS(B=1)` column. The unconditional baseline — which knows nothing about the model —
falls monotonically as the window moves later: **0.086285 → 0.060323**, a **1.43× range**.

⇒ ⛔ **Months 457–469 are the quietest 13 months in the cube**, by a measure that cannot be
accused of model-dependence. The published absolute number **0.039186 is therefore the best
absolute CRPS obtainable from this cube**, and it sits on the most forgiving window available.

⚠️ And the published window is the most favourable on the **internal margin** as well: **−35.0%**
is the maximum of the eight. The next best is b=433 at −33.7%, so the flattery is ~1.3 pp against
the best alternative and ~6.4 pp against the worst.

⛔ **What this does NOT do:** it does not retract the **−15.3% / −17.6%** parametric margin. That
comparison was made on one window against a comparator scored on the *same* window, so the window's
easiness is common to both sides and largely cancels. ⇒ The margin stands; the **absolute** number
must never be transported to another window, and the fact that the window was the easiest available
must travel with it.

## 3 · ⛔ Why no "%-vs-0.046023" column exists here

`0.046023` ("best parametric cell") and `0.050758` ("delivered") were scored on months **457–469
and nowhere else**, and `family_free_cdf.py` states in writing that it does not know which
composition either was. Neither can be transported to months 444–456 — nobody ever scored the
parametric emission there. A %-against-0.046023 column on a different window would be a category
error, so it is not printed. The internal B=1 margin replaces it, and is weaker in exactly one way:
it measures *conditioning value*, not *superiority over a parametric head*.

## 4 · ⛔⛔ What this still does not establish, and it is the larger half

Every row above comes from **the same 49 months of the same cube from the same checkpoint**.

⇒ This is **split sensitivity, not independent replication.** It is blind to anything common to all
49 months: one model, one seed, one family (`nb`), one dropout rate (0.15), one era of the conflict
record. A held split says the result is not an artifact of the boundary. It says nothing about a
second checkpoint, and nothing about the noise floor — ⚠️ RP-2 measured σ̂ on a *different*
statistic (B-1a/B-1b), so no σ̂ exists for CRPS on this route and the 6.4 pp spread above cannot
currently be separated into "window effect" and "noise".

⛔ **And it is sb-only.** This cube has no target axis: `d4_mu_on_the_grid.py:151,154` slice
`SB = 0` out of a three-target head and discard the rest. The three-target half of #471 needs a new
forward sweep; **#487** holds the deferred decision about the cube writer itself.

## 5 · ⛔⛔ The gate fired on its first run, and the GATE was the defect

G-L3a aborted at `0.039186` against a declared target of `0.039327` and refused to score any other
split. Checking what it was copied from: `family_free_cdf.out` reports **0.039186 / 0.039287 /
0.048389 / 0.042003, argmin B=8,000** — the lifted scorer matches all five to six decimals, and
that script's own committed output reads `⛔ FAIL G-a ... got 0.039186`.

⇒ **The copy was exact; the anchor was wrong.** I had anchored a *copy* gate on **M67's ledger
value**, a number no implementation has ever produced. A copy gate's anchor is the committed
implementation. Amendment 1 moved it to 0.039186 / 0.039026, tightened the tolerance 5e-5 → **1e-6**
(an exact copy should be exact), and gated the fair estimator too.

⚠️ **M67's 0.039327 is reproduced by neither implementation** (gap −0.000141, **−0.36%**). That is
now printed beside the gate on every run as an open item of #460, never as a target. The abort was
the design working: had the gate been loose at 5e-5 it would have passed and the ledger discrepancy
would have stayed invisible for a second time.
