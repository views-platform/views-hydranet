# 08 — The question battery, answered

Every question from `06_question_battery.md`, with its verdict, evidence, and what is *not*
established. Verdicts use the locked rule: **SUPPORTED** = 4/4 seeds agree in sign and |mean| >
seed sd; **CONTESTED** otherwise; no p-values, because a paired sign-flip at n=4 floors at 0.0625.

---

## Block 0 — gates

**Q0.1 — Does the model have any escalation skill to preserve?** → **YES, weak, and it rises**
rho 0.009 (h6) → 0.040 (h18) → **0.145** (h36) on the truth-fixed cohort. It clears the 0.05
unanswerable-threshold from h18 on, so C.1–C.3 are **answerable at long horizon and marginal at
h6–h12**. Not a null and not unanswerable — reported as the weak positive it is.

**Q0.2 — Is a single-step swap resolvable against donor noise?** → **SUPERSEDED, not answered**
The swap design was replaced by the **roll** design (cheaper, distribution-preserving, and it reuses
an instrument already validated in EXP-3b), so this gate never applied. The roll's own resolvability
is settled by its separation: **26/26 vs 0/26**.

---

## Block A — attribution: what actually moves the result?

**A.1 — GATE: input or state?** → **THE STATE, decisively (M60)**
Fraction of 26 origin-seed pairs whose gate field followed the displaced driver:

| driver | h2 | h4 | h6 | h12 | h24 | h36 |
|---|---|---|---|---|---|---|
| cell | **26/26** | **26/26** | **26/26** | **26/26** | 16/26 | 9/26 |
| input | **0/26** | 0/26 | 0/26 | 0/26 | 0/26 | 0/26 |

The input-rolled arm stays 0.79–0.93 correlated with the unrolled baseline at zero offset.
**Displacing what the model is fed does not move where it points — ever.**

**A.2 — BODY: same question, and is the answer DIFFERENT from the gate?** → **YES, DIFFERENT — the
one dissociation the battery predicted**

| driver | h2 | h4 | h6 | h12 | h24 | h36 |
|---|---|---|---|---|---|---|
| cell | **0/26** | 25/26 | 25/26 | 23/26 | 0/26 | 0/26 |
| input | **17/26** | 0/26 | 0/26 | 0/26 | 0/26 | 0/26 |

**At h2 the body follows the INPUT (17/26) while the gate already follows the cell (26/26).** From
h4 the body switches to the cell. So the body carries a **one-step direct dependence on the fed-back
magnitude that the gate does not have** — it reads the last number it emitted, once, then the state
takes over. This is the "body is input-driven while the gate is state-driven" split the battery
flagged as the most interesting possible outcome, and it is real but **confined to a single step**.

**A.3 — hidden or cell, for the gate?** → **CELL, on both routes**
Directly: hidden **0/26** at every horizon. By intervention (Wave 1, 4/4 seeds): AP@h36 cell
**+0.0591** vs hidden **+0.0186**, and hidden is CONTESTED at h18.

**A.4 — hidden or cell, for the body?** → **CELL.** hidden 0/26 on the body field too.
⚠️ Carries the C-292 caveat: `hs = o ⊙ tanh(cl)`, so hidden is a readout of cell and the split is
the caveated half throughout.

---

## Block B — what freezing means

**B.1 — Does the cell's advantage hold for the BODY, or is it gate-only?** → **Mostly gate-only**
`size_ratio`: **exactly 0 in all 16 arm-seeds** — the pre-registered falsifier did not fire; the
median conflict cell gets nothing in every arm. `crps_events`: **improves 4/4 seeds in every arm**
but small — cell −6.7/−6.1/−6.5/−4.1 % at h18, fading to **−0.6%** by h36.
**Verdict: a small replicated gain in the body's proper score, and no movement in its calibration.**

**B.2 — Is `all` better than `cell`, or redundant?** → **REDUNDANT (SUPPORTED)**
AP@h36 `all` +0.0578 vs `cell` +0.0591, inside a seed sd of 0.007. Dispersion −0.667 vs −0.659.
Freezing both halves buys nothing over freezing the cell. **M39's 89% attribution confirmed at n=4**;
its falsifier did not fire.

**B.3 — Does freezing ever HURT?** → **NOT CONSISTENTLY**
The only negatives are `hidden` at h18 on 2 of 4 seeds (−0.0090, −0.0002) — CONTESTED. No arm is
consistently worse than baseline on any head at any horizon. The clamp is **strictly non-harmful**
in this battery, which `05` pre-committed to treating as a finding rather than an absence.

---

## Block C — dynamics

**C.1 — Does the model express DIFFERENTIAL dynamics, or does the field move as one?** → **YES,
differential.** Dispersion of the predicted per-cell change is **1.197** at h36 against the
instrument's ~0.002 EPS floor — roughly **600×** the floor. The field does not move as one block.

**C.2 — Does freezing flatten the dynamics?** → **YES, and it is specifically the CELL (SUPPORTED)**
`cell` − none: **−0.4563** (h18), **−0.6593** (h36), 4/4 seeds. `hidden` is CONTESTED on `sb`; it
does flatten `ns`/`os` but always less than the cell (−0.34 vs −0.90 on `ns`).

**C.3 — Does the flattening cost direction skill?** → **NO on `sb`; it IMPROVES it on `ns`/`os`**

| target | rho at h36, cell − none | |
|---|---|---|
| `sb` | −0.0020 | CONTESTED — no cost |
| `ns` | **+0.0177** | SUPPORTED 4/4 |
| `os` | **+0.0598** | SUPPORTED 4/4 |

The clamp compresses the *amplitude* of predicted change and leaves — on two of three targets,
improves — the *ordering* of which places worsen. **The trade-off this question was built to catch
does not exist.** ⚠️ rho ≈ 0.14 is weak absolutely, so "no cost" partly reflects little to lose.

**C.4 — Does freezing buy CONTINUATION at the cost of ONSET?** → **NO. It improves BOTH, on all
three targets (12/12 SUPPORTED at 4/4)**

| target | h36 continuation | h36 onset |
|---|---|---|
| `sb` | +0.0614 | +0.0459 |
| `ns` | +0.1401 | +0.0357 |
| `os` | +0.1582 | +0.0579 |

The unclamped model already has real onset skill — AP 0.1211 against a 0.0079 base rate, **15×
chance**. ⚠️ **The falsifier was ambiguous** ("at least as large", no measure named): absolute gain
and skill-above-base favour continuation, relative gain favours onset. All three reported, none
chosen post hoc. **The defensible claim is "helps both", not a ranking.**

---

## Block D — robustness

**D.1 — Does it replicate across seeds?** → **YES for Wave 1** (all four seeds; every SUPPORTED
verdict above is 4/4). **Wave 2 is two seeds** (42, 43) — its 26/26 and 0/26 are unanimous but on
half the vehicles.

**D.2 — Does it hold for `ns` and `os`?** → **YES, and it is STRONGER there.** All twelve C.4 lines
SUPPORTED; the continuation effect is 2.3–2.6× larger; and C.3 flips from neutral to positive.

---

## The chain, in one paragraph

The ConvLSTM **cell state holds a learned spatial prior** — a map of where conflict will be. It is
not persistence (M55: 2.97× fair persistence at h36, gap widening) and it is not the input (M60:
input 0/26 at every horizon). Free-running **drains** it (M51). Clamping **hands it back** (M56),
which improves the gate on **both new and continuing conflict** across all three targets (M57, M59),
**compresses** how much the model says magnitudes will change without costing — and on `ns`/`os`
improving — **which** places it says will worsen (M58, M59), and **does nothing at all** for
magnitude calibration (B.1). The body differs from the gate in exactly one place: **a single-step
direct dependence on the last magnitude it emitted** (A.2).

## What the night did NOT answer

* **Anything about the body's amplitude.** `size_ratio` is 0 in every arm at every horizon. Nothing
  in this programme has moved it, and this battery did not try.
* **Attribution beyond h12.** The rolled pattern loses coherence (cell 16/26 at h24, 9/26 at h36),
  so A.1/A.2 are established for short-to-mid horizons only. A shift coprime with 180 would separate
  coherence-decay from residual aliasing.
* **Whether any of it is causal.** Every finding is an intervention's effect, not a mechanism proven
  to mediate it — `05b` §2's limitation stands for the whole battery.
* **Wave 2 at four seeds**, and `ns`/`os` for the scorer metrics (the wave scored `--targets=sb`).
