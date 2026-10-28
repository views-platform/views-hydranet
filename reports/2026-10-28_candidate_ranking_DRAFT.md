# ⚠️ DRAFT — a CANDIDATE ranking of the eleven synthetic forecasts

**Status: NOT the declared ranking.** This is a first pass for Simon to correct, written 2026-10-28
at his request (*"just draft it, we'll revisit it later and go into depth"*). ⛔ **Nothing may cite it,
no subset search may run against it, and no metric may be judged by it** until Simon declares a
ranking. The declared ranking is a judgement and is his; this exists so he has something to argue
with rather than a blank page.

**The population:** `reports/2026-10-27_metric_redteam/redteam_forecasts.py`, eleven forecasts over
13 months × 64,818 cells = 842,634 cell–months, each with a passing construction gate.

**The criterion I ranked on, stated so it can be rejected:** *operational value to a conflict
early-warning system* — how much a user learns about **where** violence will occur this month, then
**how much**, then **how honestly the uncertainty is expressed**. ⚠️ That weighting is a choice, not a
fact. Changing it reorders rows 2–7.

---

## The candidate order

| rank | forecast | why here | confidence |
|---|---|---|---|
| **1** | **oracle (raw truth)** | Perfect by construction. The only rank I would defend without argument. | ⭐ certain |
| **2** | **freq-right/mag-wrong** | Identifies **every** conflict cell exactly and no others, with the sizes wrong (mass 2.92× too high). If where-it-happens is the dominant decision output, this is extraordinary. ⚠️ **The single most contestable call in the table** — see §A. | ⚠️ low |
| **3** | **blurred-truth r=1** | The truth, smeared one cell, **monthly total preserved exactly**. Every event within one cell of correct ⇒ near-perfect at any aggregation coarser than a single cell. | ⚠️ medium |
| **4** | **reference (delivered)** | Our real model. Best non-oracle `crps_all` (0.0501) and FSS1 (0.630), AP 0.736. ⛔ Placed **below** two synthetic forecasts deliberately — see §A. | ⚠️ medium |
| **5** | **blurred-truth r=2** | Same virtue as r=1, degraded. Truth-derived, mass exact, 2-cell smear. | ⭐ fair |
| **6** | **blurred-truth r=4** | Same again, 4-cell smear. Still knows **which month**. | ⭐ fair |
| **7** | **climatology** | Knows the long-run geography of conflict and **nothing month-specific**. Genuinely informative (AP 0.440) but cannot be right about *this* month. ⚠️ Beats blur r=4 on AP (0.440 vs 0.250) — so this pair is a real fork, not an obvious call. | ⚠️ low |
| **8** | **persistence** | Last month repeated. Real information, weakly used (AP 0.221, FSS1 0.179). The dumb baseline. | ⭐ fair |
| **9** | **all-zero** | Predicts nothing, ever. **Useless but not misleading** — it raises no false alarm. ⛔ Ranked above the two below on the view that a confident wrong alarm costs an analyst more than silence. ⚠️ Uncomfortable consequence: this says predicting nothing beats two forecasts that try. | ⚠️ low |
| **10** | **spiky-1/30** | A month's entire mass on 97 cells. Actively misleading, though the cells it picks do correlate with truth (`placement_rho` 0.521). | ⭐ fair |
| **11** | **mag-right/freq-wrong** | The correct list of magnitudes, placed in the **wrong cells** (AP 0.0034 = chance). Maximally misleading: large confident numbers, all in the wrong place. | ⭐ fair |

---

## §A — The three forks I could not settle, and should not settle alone

**A-1 ⚠️ Is perfect placement with wrong sizes better than our model?** Rank 2 vs rank 4. It is the
upper bound on what placement alone buys, and the project's headline metric (AP) cannot tell it from
the oracle. If you rank it **below** our model, you are asserting that magnitude calibration is
load-bearing for the use case — which is a position the paper should state outright rather than imply.

**A-2 ⛔ Both `crps_events` and `placement_rho` rank the mass-preserving blur ABOVE our delivered
model** (13.946 vs 14.487; 0.683 vs 0.629). My draft agrees with them (rank 3 above rank 4), which
means **two metrics are vindicated, not refuted, by this pair.** If you disagree, those two columns
have a measured failure and that is a finding in its own right.

**A-3 ⚠️ Does month-specific-but-blurred beat month-agnostic-but-sharp?** Rank 6 vs rank 7. Blur r=4
knows which month; climatology does not, yet scores higher on AP. Whichever way you call it, the call
is the paper's position on what AP is for.

## §B — Two design points for the declared ranking, before it is written

1. ⭐ **It need not be a total order.** Declaring *"these two are incomparable"* is a legitimate and
   probably necessary move — it just means no metric set is required to order that pair, which makes
   the sufficiency criterion weaker and far more defensible. ⛔ A forced total order on eleven items
   invents 55 pairwise commitments, and I would not defend most of them.
2. ⛔ **The population is thin in one direction.** Nine of the eleven are built so every ensemble
   member is identical, so **nine of eleven have exactly zero spread**. ⇒ The population as it stands
   cannot test whether a metric set ranks *dispersion* correctly, which is the defect this programme
   has chased since July. **At least two more forecasts are needed** — same mean field, honest spread
   vs over-confident spread — and they need construction gates like the rest.
