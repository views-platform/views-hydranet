# Drivers & Dynamics — what moves the gate, what moves the body, what freezing costs

**Opened** 2026-09-03 · **Branch** `exp/silence-vs-fade` · **Status** Wave 1 COMPLETE (16/16) · D.2 complete · **M56–M59 logged**

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
| 01 | `01_plain_answers.md` | **the chair's questions, in his words, answered** |
| 03 | `03_harness_and_invariants.md` | **written** — the unattended launcher, the instruments, the red-team findings |
| 05 | `05_analysis_plan.md` | **LOCKED before Wave 1** |
| 06 | `06_question_battery.md` | absorbed from the predecessor dossier (original marked superseded) |
| 07 | `07_experiment_log.md` | append-only — seed-42 interim, Wave 1, D.2, Wave 2 |
| 08 | `08_answers.md` | the battery answered in its own numbering, with evidence and limits |

## Standing rule adopted from C-319

Every question declares whether its measure is **INTERNAL** (a property of the emitted field) or
**TRUTH-REFERENCED**. An internal measure can never close a causal claim about a score. EXP-3 proved
it: alignment, occurrence and magnitude were all identical between a good forecast and one displaced
90 cells.

## Next actions

- [x] Wave 1 launched — 4 arms × 4 seeds, unattended, hardened after a red-team review
- [x] Instruments built and mutation-tested before the data landed (7/7, 7/7, 7/7)
- [x] Cross-seed assembly — `results/FINDINGS{,_ns,_os}.md`
- [x] Ledger rows **M56–M59**, decision rule applied per question
- [x] D.2 — holds on `ns`/`os` at zero GPU; the `sb`-only caveat is lifted
- [ ] **Wave 2 (attribution)** — needs a new per-step roll seam in `hydranet_inference.py`;
      runs **only** if it passes tests and mutations first
- [ ] Branch `exp/silence-vs-fade` is **local only** — never pushed
