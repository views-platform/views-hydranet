# 01 — The questions, in the chair's words, with the answers

Written 2026-09-03 after Waves 1 and 2. This is the **plain-language** companion to
[`08_answers.md`](08_answers.md), which carries the same conclusions in the battery's own numbering
with full evidence and caveats. The questions below are the chair's, as asked.

---

## 1. "At every step into the horizon, what most influences the magnitude — t−1, hidden, or cell?"

**The cell state — except at the very first step, where it is t−1.**

At month 2 the body follows what it was just fed (**17 of 26** origin-seed pairs) and ignores the
cell (**0 of 26**). From month 4 that flips hard: the body follows the cell (**25, 25, 23 of 26** at
months 4, 6, 12) and never follows the input again. The body glances at the last number it emitted,
once, then runs on memory.

*Evidence:* M60 / A.2. Roll one driver 90 cells every step, ask which one the emitted body field
follows.

## 2. "Same question for the gate."

**The cell state, always. t−1 never.**

**26 of 26** at months 2, 4, 6 and 12. The input scores **0 of 26 at every horizon** — displace what
the model is fed and the gate does not budge; its field stays 0.79–0.93 correlated with the
undisturbed baseline. Hidden also scores 0 of 26. Unlike the body, the gate has no first-step
exception: it is on memory from the start.

*Evidence:* M60 / A.1.

## 3. "What does freezing hidden vs cell mean — for the gate?"

**The cell is the real lever; hidden is a weak shadow of it.**

Freezing the cell gains **+0.059 AP** at month 36 (4/4 seeds). Freezing hidden gains **+0.019**, and
at month 18 it is inconsistent — two of four seeds get *worse*. Freezing **both** gains +0.058, no
more than the cell alone, so the hidden half contributes nothing you do not already have.

*Evidence:* M56 / B.2, B.3.

## 4. "And for the body?"

**Almost nothing, in the way that matters.**

`size_ratio` — whether the typical conflict cell gets a right-sized number — is **exactly 0 in all
sixteen arm-and-seed combinations**. No freezing of anything moved it. The proper magnitude score
does improve, consistently: −6.7% at month 18 on 4/4 seeds, but fading to **−0.6%** by month 36.

**Freezing fixes *where*, not *how much*.**

*Evidence:* M58 / B.1. This corrects M55, which said "nothing" — the small gain is real and
replicated, it is only the calibration that never moves.

## 5. "Do we stop the model predicting upward or downward trends if we freeze?"

**It says change more quietly, but it does not lose track of which places change.**

Freezing compresses the spread of predicted change by about half (**−0.66** at month 36, 4/4 seeds)
— it hedges, predicting smaller swings. But its *ranking* of which places worsen is unharmed on
`sb`, and **improves** on `ns` (+0.018) and `os` (+0.060), 4/4 seeds each.

Quieter about size; no worse — sometimes better — about direction.

⚠️ The model's direction skill is weak in absolute terms to begin with (rho ≈ 0.14 at month 36), so
"no cost" partly reflects that there was little to lose.

*Evidence:* M58, M59 / C.2, C.3, Q0.1.

## 6. "Why does freezing help?"

**The memory holds a map of where conflict will be. Running on its own output drains that map.
Freezing pins it, so the model keeps firing in roughly the right places instead of falling silent.**

*Evidence:* M51 (the field collapses on both axes), M54 (move the state 90 cells, the forecast moves
90 cells intact), M60 (the cell drives placement, the input does not).

## 7. "Does it help for good reasons or bad reasons?"

**Good reasons.**

The worry was that the map is just "wherever conflict was last month" — persistence in disguise. It
is not. The frozen model beats a fairly-scored persistence by **2.97×** at month 36, and the gap
**widens** with horizon. It keeps **59%** of its month-1 skill where persistence keeps **40%** — it
degrades *more slowly than the observed field its own memory was built from*, which a persistence
fallback cannot do.

Nor is it a trick of only predicting conflict that was already there: it improves forecasts of
**brand-new** conflict as well, on all three targets, every seed.

*Evidence:* M55, M57, M59 / C.4.

---

## The one honest gap

**None of this touches how big the model says a conflict will be.** `size_ratio` has been 0 through
this entire programme, and nothing tested on 2026-09-02/03 moved it. Freezing is a placement fix;
the amount-ceiling is untouched and remains where M32/M45 left it.

## What is not established

* **Attribution beyond month 12** — the displaced pattern loses coherence (cell 16/26 at month 24,
  9/26 at month 36), so questions 1 and 2 are answered for short-to-mid horizons only.
* **Causality** — every answer above is an intervention's *effect*, not a mechanism proven to
  mediate it.
* **Wave 2 at four seeds** — questions 1 and 2 rest on two seeds; questions 3–5 on four.
