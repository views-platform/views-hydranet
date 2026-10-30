# RP-2b — the battery's noise floors: FAO-02's locked MCR guardrail is swamped by seed noise 84×

**Date** 2026-10-28 · **Cost** ~6 min CPU, $0 · **All 9 gates PASS (after Amendment 1)** ·
**Script** `column_noise_floors.py`, committed **before** it ran (D-5) · **Output** `.out`

**C-351 slots.** Population = 842,634 cell-months (identical to M2's) · transform =
`ey = E_draws[gate × emitted_magnitude(mu, theta, family)]`, composed **per draw then averaged**
(C-386) · estimator = pooled sample sd, `ddof=1`, n=3, **2 df** · CV = σ̂/|mean| · not normalised.

---

## 1 · ⭐⭐⭐ The finding

**FAO-02 Appendix A.1 locks MCR with a non-inferiority margin of 1%**, judged as distance from 1.

| | |
|---|---|
| the margin the guardrail is judged on | **0.01** |
| **σ̂ from the seed alone** | **0.840477** |
| ⇒ | **the seed noise is 84× the margin** |
| the honest 4.97σ̂ bar against one unreplicated arm | **4.177**, i.e. **418× the margin** |

⇒ **A guardrail with a 1% margin cannot be evaluated on single runs whose seed noise is 84%.**
⛔ This is not a claim that FAO-02 is wrong — it is locked and stays locked. It is that **our
ability to test compliance with it is two orders of magnitude coarser than the test itself**, and
nothing in the apparatus says so.

⚠️ **Scoped, and the scope matters**: `truncated_nb` @ dropout 0.15, **expectation composition**,
not the delivered `nb` draws. ⇒ An **order-of-magnitude** statement, not a number for FAO-02.
⛔ It joins **C-389**, which found the frozen ruler computes only one of FAO-02's three locked
guardrails as defined. Now: of the one it does compute as defined, the margin is 84× under the noise.

## 2 · The profile — and a clean split

| | columns | CV range |
|---|---|---|
| **gate-only** | AP, Brier | **0.021 – 0.024** |
| **placement / shape** | FSS×7, hit_rate, Gini | 0.024 – 0.055 |
| ⛔ **magnitude calibration** | size_ratio **0.237**, pos_mcr **0.293**, MCR_field **0.356** | **10× noisier** |

⇒ ⭐⭐ **The battery's three magnitude-calibration columns are an order of magnitude noisier than
its placement columns.** The quantity this programme most wants to move — magnitude — is measured
by its three least resolvable columns.

## 3 · ⛔ Two columns are SATURATED, not stable

`AFSS` and `act_freq` read **exactly 1.000000** in all three replicates, σ̂ = 0. ⛔ That is a
**ceiling, not stability**: `ey = gate × emitted_magnitude` is strictly positive everywhere (a
sigmoid gate is never exactly 0; truncated_nb's `E[Y|Y>0] = mu/(1−p0) > 0`), so every cell is
"active" and both columns pin. Reporting them as the most stable columns would be the **C-376
shape** — a readout at its limit presented as a measurement. Detection added in Amendment 1 and
every reading now runs over the **14 live** columns.

## 4 · Predictions scored

- ⛔ **P4 REFUTED.** I predicted at least one Tier-A column would have CV > 1, as B-1b body-alone
  did. **None does** — the worst is MCR_field at 0.356. ⭐ **And the reason is informative**: B-1b
  body-alone scores the **ungated body** (`mu`), while these columns score the **gated composition**
  `gate × magnitude`, which multiplies in the seed-stable half. ⇒ The two results agree; the
  difference is the composition, not the instrument. **W-97.**
- ✅ **P5 CONFIRMED.** The two gate-only columns are exactly the two most stable live columns —
  **Brier 0.0213, AP 0.0241**. ⚠️ **But literal P5 named no saturation exclusion**, and without it
  the answer would have been `AFSS, act_freq` and P5 would read REFUTED. ⇒ Confirmed **under a
  correction made after the fact**, and recorded as such rather than claimed clean. **R-37.**

## 5 · ⛔ Tier B — NOT RUN, and the reason is a registered defect

`crps_all`, `crps_events`, `crps_none`, `QS99` — **the only four columns that read the ensemble
beyond its mean**, i.e. the battery's **only window onto dispersion** (RP-1/M72).

Measuring them needs real draws from `truncated_nb`. ⛔ **#441 is OPEN**: `TruncatedNBFamily.sample()`
disagrees with `.mean()` by up to **54× at small theta**, and its title states the fast path's
premise **"fails on 99% of the grid"**. ⇒ σ̂ would carry a known 54× defect across essentially the
whole population. **A noise floor contaminated by a sampler bug is worse than no noise floor.**

⭐ **Declared before the run, and it stands:** RP-1 found only **4 of 20** columns can *see*
dispersion. Those are exactly the **4 whose noise floor cannot be measured**. **The battery's one
window onto the defect this programme has chased since July is also its one unmeasurable window.**
⇒ Reported as **UNMEASURED** — ⛔ not as zero, and not as fine. Unblocking it means fixing **#441**.

## 6 · ⛔ Amendment 1 — two gates failed, and the GATES were wrong

**G-P0** asserted bit-exactness of a floating-point mean and got **4.547e-13**; **G-Pa** then
demanded exactly 0.0 movement and got **size_ratio 4.441e-16** — machine epsilon beside a value of
~1.07, *caused by* G-P0's own residual, not by a partition violation.

⛔⛔ **W-96: this is W-93 repeated, the same day.** M4's G-D asserted `spread == 0.0` exactly and
failed at ±8.2e-19 for the same reason; the lesson recorded there was *"assert the structural fact,
print the residual as noise with its scale, and log it rather than widening a tolerance quietly"* —
and the next gate was written the same wrong way. ⇒ The rule needs restating as a **constructive**
one: ⭐ **a gate on a floating-point quantity asserts a structural property (a sign, an exact
integer count, a ptp of zero over identical values) or a RELATIVE bound with its scale printed —
never bit-equality of a sum.**

**Amended to:** `G-P0a` every row has nonzero spread (**structural, exact**) · `G-P0b` the mean is
held to **float noise, relative 1.598e-16, printed with its scale** · `G-Pa` Tier-A moves at
float-noise scale only · ⭐ `G-Pc` **the discriminating gate**: the largest Tier-B move exceeds the
largest Tier-A move by **1.42e+16×** ⇒ the partition is **demonstrated, not assumed**.

⚠️ RP-1's original result held the mean bit-exactly because it **permuted values within a row**; an
additive offset cannot.

## 7 · ⛔ What this does NOT establish

- **Not comparable to M2's committed battery levels.** Those are the `nb` control's, scored on
  **draws**; these are `truncated_nb`, scored on the **expectation**. `act_freq` is 0.012225 there
  and 1.000000 here — a different object, not a worse model. ⛔ Only σ̂ and CV transfer, provisionally.
- **Not a noise floor for the 4 dispersion columns.** See §5.
- **Not tight.** n=3, 2 df, 95% CI **[0.52σ̂, 6.28σ̂]**.
- **Not a verdict on any arm.**
