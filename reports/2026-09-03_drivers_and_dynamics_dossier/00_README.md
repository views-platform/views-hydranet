# Drivers & Dynamics — what moves the gate, what moves the body, what freezing costs

**Opened** 2026-09-03 · **Branch** `exp/silence-vs-fade` · **Status** Wave 1 running (16 arms, 4 seeds)

## Purpose

The predecessor dossier (`2026-09-02_silence_vs_fade`) established *what* the cell clamp does and
falsified two of my own claims doing it. It left three things open, and this dossier runs them at
**n=4 seeds** rather than the n=1 that every finding M51–M55 currently rests on.

| | |
|---|---|
| **M51** | free-running collapses on **both** axes — occurrence ×0.036, magnitude ×0.222 |
| **M53/M54** | the cell state is a **map**: roll it 90 cells, the forecast moves 90 cells intact (r≈0.90) and skill collapses 48× |
| **M55** | the clamp is **not** persistence (2.97× fair persistence at h36, gap widening) — but does **nothing** for the body (`size_ratio` 0→0) |
| **C-319** | every field statistic there is **blind to placement**; all survive a roll that destroys the forecast |

## The three open questions

1. **Attribution** — per horizon, per head: does the fed-back input, the hidden state, or the cell
   state move the result?
2. **Semantics of freezing** — what does freezing hidden *mean* versus freezing cell, for the gate
   and for the body separately?
3. **Dynamics** — does freezing preserve the *level* of magnitude while destroying *escalation*?
   And the gate's analogue: does it buy **continuation** at the cost of **onset**?

Question 3 holds the only result that could change the ship decision. If the clamp's AP gain were
entirely conflict that was already there, it would be close to worthless for the product — and
nothing measured before this dossier could tell the difference.

## Document index

| # | File | Status |
|---|------|--------|
| 00 | `00_README.md` | living |
| 03 | `03_harness_and_invariants.md` | **written** — the unattended launcher, the instruments, the red-team findings |
| 05 | `05_analysis_plan.md` | **LOCKED before Wave 1** |
| 06 | `06_question_battery.md` | absorbed from the predecessor dossier (original marked superseded) |
| 07 | `07_experiment_log.md` | append-only; carries the seed-42 interim |

## Standing rule adopted from C-319

Every question declares whether its measure is **INTERNAL** (a property of the emitted field) or
**TRUTH-REFERENCED**. An internal measure can never close a causal claim about a score. EXP-3 proved
it: alignment, occurrence and magnitude were all identical between a good forecast and one displaced
90 cells.

## Next actions

- [x] Wave 1 launched — 4 arms × 4 seeds, unattended, hardened after a red-team review
- [x] Instruments built and mutation-tested before the data landed (7/7, 7/7, 7/7)
- [ ] Cross-seed assembly once all 16 arms land (`tools/assemble.py`)
- [ ] Ledger rows, and the decision-rule verdict per question
- [ ] Wave 2 (roll attribution) **only** if its harness passes tests and mutations first
