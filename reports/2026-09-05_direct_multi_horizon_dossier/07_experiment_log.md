# 07 — Experiment log

**Append-only.** Negatives and VOIDs in the same detail as wins.

---

## PROBE-A — stale-feedback probe (2026-09-06) — **carries no weight on this epic**

**Pre-registration:** `02b_design_review_rulings.md` proposed this as a gate on the epic.
**Status of that gate: WITHDRAWN before the numbers were read** — see below. The arms were finished
because they were already paid for.

**What ran:** `hold_last_real` (feed the origin's real field at every step, never the model's own
output) + `freeze_recurrent='cell'`, emit-only on the four existing L=300 artifacts. Compared against
the clamped control. 2 of 4 seeds completed before the driver stopped logging; the remaining two
produced controls only.

**Result — `AP@h18`, `sb`, 13 origins:**

| seed | clamped control | probe + clamp | Δ | act_ratio | probe, no clamp | Δ |
|---|---|---|---|---|---|---|
| 42 | 0.3622 | 0.3519 | **−0.0103** | ×1.97 | 0.2896 | −0.0725 |
| 43 | 0.3709 | 0.3528 | **−0.0182** | ×1.99 | 0.2805 | −0.0904 |
| | | **mean −0.0142**, sd 0.0055, **2/2 negative** | | | | |

**Verdict: this measures STALENESS, not the removal of feedback — my error, and the reason the gate
was withdrawn.** `hold_last_real` does not remove the bad input; it substitutes a *frozen* one — the
origin's map, held constant while predicting up to 36 months out. That is a third condition, and
plausibly worse than the model's own forecast, which at least attempts to move forward. So the
negative says nothing about a direct head.

**What it does say, and it is worth having:** feeding a stale-but-real field is worse than feeding
the model's own evolving prediction, by −0.014 with the clamp on and −0.081 without it. The model's
self-generated field, for all its faults, carries more usable information than a frozen observation.
That is a small positive result about the feedback loop, in a programme that has mostly produced
negatives about it.

**Also recorded:** firing roughly doubles (×1.97, ×1.99) while AP falls — the M45 signature again, on
an arm whose input is *real data*. Consistent with the pre-registration's corrected reading: firing
more is only good when placement improves.

**Method note for the register.** The first attempt at this probe drove `realism_arm_entry.py`
per-arm instead of the guarded `run_realism_arms.py`. The pipeline names the prediction directory
after the **artifact**, not the arm, so all three arms of a seed wrote the same path and each
silently overwrote its predecessor: 10 arms reported success and produced **zero** scores. The
guarded driver refuses to start on a leftover directory, asserts exactly one new directory appeared,
and scores before the next arm runs. **The guard existed and I bypassed it after reading the file it
is documented in.**
