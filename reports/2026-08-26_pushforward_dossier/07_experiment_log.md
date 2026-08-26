# Experiment log — pushforward (#289)

## SMOKE-01 — does it run, and what does it cost? (2026-08-26)

**Pre-registration:** none. A smoke is a harness gate, not a scored result; it produces no
evidence about the hypothesis and nothing here may be cited as one.

**Arms:** `tinyzero_fortytwo` (control, `pushforward_weight=0.0`) and `pftinyzero_fortytwo`
(`0.1`, state attached). 2 lessons, seed 42, otherwise identical to the `fullzero_*` controls.
Config diff vs the control is exactly `{pushforward_weight, pushforward_detach_state,
total_lessons}` — verified by exec-and-diff, not by inspection.

**Result: PASS, both arms.** Trained and emitted a prediction cube, rc=0.

| | control | treatment | ratio |
|---|---|---|---|
| peak GPU | 4107 MiB | 5617 MiB | **×1.37** (69% of the 8188 MiB card) |
| training | 43 s | 87 s | **×2.02** |
| total runtime | 912 s | 1069 s | ×1.17 |
| emit + score | 869 s | 982 s | — (pushforward does not touch this path) |

**Projected 300-lesson arm:** control 2.03 h, treatment **3.87 h (×1.90)**. The control projection
lands within 12% of the incumbent's separately measured 1.82 h/arm, which is the sanity check on
the extrapolation.

### The finding that is about the harness, not the model

**The smoke's total-time ratio is ×1.17 and it is the wrong number.** A 2-lesson run is ~95% emit
and scoring, which the pushforward does not touch; a 300-lesson arm is almost all training. Quoting
×1.17 would have understated the GPU budget by 60% — the same shape of error as the architecture
bake-off's preflight cost projection, which was 17× low **and passed** (C-310).

Caught before it mattered, because training time was visible in the log while the run was still
going. Two changes so it cannot recur:

* `tools/train_time.py` extracts training wall-clock from the completed tqdm bar, and **refuses**
  rather than falling back to total runtime when the bar is absent (verified against a truncated
  log).
* `smoke_pf.sh` now reports the training ratio, prints the total ratio beside it as the trap it is,
  and states the projection in hours.

The CPU-side probe run during the training-loop audit said ×1.76. The real pipeline says ×2.02 —
**that probe was 13% low**, which is the standing lesson about cheap measurements holding even when
the cheap measurement is careful.

### Open, and load-bearing for the design

The existing 300-lesson `fullzero_*` controls were trained **before PR #303**. Reusing them saves
~8 h of GPU but assumes the merged changes are inert at `pushforward_weight=0.0`. There is an
argument and a test for that (`test_the_new_guard_is_byte_identical_when_frozen`, plus the
pushforward branch is never entered at weight 0), but no end-to-end evidence. Resolved in the
pre-registration, not here.
