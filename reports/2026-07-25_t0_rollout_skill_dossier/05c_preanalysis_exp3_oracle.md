# 05c — Pre-analysis plan: EXP-3, the teacher-forced oracle (bug-vs-ceiling gap)

**Pre-registered 2026-07-26, BEFORE running.** One variable: **feedback = the realized truth each step**
(`rollout_feedback='teacher_forced'`). Same frozen ruler, same zinb artifact (`…063927`), s44 INDICATIVE.

## Hypothesis (H-ORACLE)

Feeding back the *realized* `truth[o+t]` each step removes ALL exposure bias — the model always sees a real,
in-distribution input, so any remaining error is the **intrinsic one-step-conditioned ceiling** given the
conflict-history features. Define per horizon h:

    gap(h) = crps_<metric>(sample-feedback, h) − crps_<metric>(teacher-forced-oracle, h)

- **gap ≈ 0** ⇒ sample-feedback is already near the ceiling; the residual is *irreducible* (no more bug).
- **gap large & growing** ⇒ exposure bias still costs; there is headroom a better rollout could recover.

Per the method-review (C-222): the oracle is a **one-step-conditioned ceiling**, NOT a raw predictability
ceiling (if the one-step map is biased, the oracle inherits it); and gap = input-exposure-bias ⊕ induced
hidden-state drift (hedge, don't over-claim a clean decomposition).

## The change (a 3rd feedback mode, default-off)

`rollout_feedback='teacher_forced'`: in the AR loop, the step-t input is the REAL
`full_tensor[:, t, model_in_indices]` (the realized month-t data — the calibration window is historical, so
these are real) instead of the fed-back prediction/sample. No family draw needed; works for any head. `mean`
stays the byte-identical default; `sample` unchanged.

## Pre-registered predictions

- **P1 (magnitude ceiling confirmed):** on crps_events / size_ratio, the oracle is **≈ sample-feedback ≈
  climatology** — i.e. the magnitude gap is ~0. Feeding perfect inputs does NOT unlock event-size skill ⇒
  the amount-ceiling wall is intrinsic, not exposure bias.
- **P2 (occurrence headroom, maybe):** on AP (occurrence), the oracle may beat sample-feedback at longer h
  (perfect inputs preserve the occurrence signal the free rollout loses) — a nonzero gap here = exposure
  bias still costs *occurrence* skill, i.e. a better rollout could help locate conflict further out.
- **P3 (crps_all):** the oracle stays bounded and ≤ sample-feedback at every h (perfect feedback can't be
  worse than a sampled one).

## Pre-committed falsifiers

- **F-O1 (oracle NOT better than sample on ANY metric/horizon):** ⇒ sample-feedback already saturates the
  ceiling — the bloom fix is the whole story, no exposure-bias headroom remains. (Strengthens "it's a ceiling".)
- **F-O2 (oracle blooms too):** if even teacher-forced runs away ⇒ the instability is in the one-step map
  itself, not exposure bias — a deeper problem than H-SAMPLE assumed. (Would be very surprising.)
- **F-O3 (oracle hugely beats everything on magnitude, size_ratio→1):** ⇒ the magnitude failure IS
  recoverable with better inputs (NOT a hard ceiling) — would overturn the amount-ceiling-wall reading and
  re-open magnitude as a rollout/exposure problem.

## Method

1. Implement `teacher_forced` (config-validation + AR-loop branch; TDD: differs from mean/sample, works
   without a family, parity of mean unchanged). Full suite + ruff + determinism green.
2. Stealth-safe driver (verify+trap-restore floor md5; write to a FRESH dir — the in-place-overwrite scar).
3. Eval zinb `…063927` with `rollout_feedback=teacher_forced` → score at all h.
4. Compute gap(h) = sample − oracle for crps_all/events/none, size_ratio, AP; A/B table vs mean/sample/clim.
5. Log (`07`) vs F-O1..F-O3. Single-seed INDICATIVE.

## Decision rules

- **P1 holds + F-O1/F-O3 quiet ⇒** the bug-vs-ceiling verdict is settled: bloom = fixed bug; magnitude =
  intrinsic ceiling; occurrence has ~1yr skill. The rollout epic's core question is answered; remaining work
  is hardening (3-seed) + the product reframe (T=0 is the deliverable, long-horizon ≈ climatology).
- **P2 gap on AP ⇒** a scheduled-sampling/GTF retrain (teach recovery from own feedback) has measurable
  occurrence headroom — worth the rung-3 training cost.
- **F-O3 fires ⇒** re-open magnitude as recoverable; pivot back to the rollout, not the head.
