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

### RETRACTION (same day): the training-time numbers above are WRONG

The table's `training 43 s -> 87 s (x2.02)` and everything derived from it are **retracted**. They
came from `train_time.py`, which scraped the training loop's tqdm bar out of the run log. It
matched *any* completed progress bar (`current == total`) and took the last one — and a real run
emits **~250** completed bars from the posterior sampler, so it was reading an emit bar, not a
training bar.

**Caught because a re-run disagreed with itself**, not by inspection: the second smoke reported
training `63 s -> 63 s` (x1.00) while peak memory still moved `3952 -> 5433 MiB` (x1.37). +37%
memory with +0% time is incoherent, which is what forced the check.

Filtering to the training loop's own `month/s` unit narrows it to two bars per run — the training
loop and the BN recalibration — with no reliable way to tell them apart, and on the second smoke
the control's bars were *slower* than the treatment's. **A scrape of a shared, noisy log is the
wrong instrument for this**, and `train_time.py` has been deleted rather than patched.

The irony is on the record: that tool was written *specifically* to avoid quoting a misleading
ratio, and it refused correctly when there was **no** completed bar. It had no defence against
there being **hundreds**.

### The corrected measurement

`tools/pf_cost.py` times the thing itself — one window at production shapes (`window_dim=32`,
`T=336`), forward plus backward, both conditions in one process so machine state is shared, median
of 3 repeats:

| | peak (torch-allocated) | fwd+bwd | spread |
|---|---|---|---|
| `pushforward_weight=0` | 2524 MiB | 3.22 s | 3.21–4.54 |
| `=0.1` | 3990 MiB | 6.55 s | 6.54–6.83 |
| `=0.1`, detached | 3990 MiB | 6.56 s | 6.51–6.61 |

**Training time ×2.03. Memory ×1.58 torch-allocated**, which is ×1.37 as whole-process peak
(`nvidia-smi` includes the ~900 MiB CUDA context, which does not scale). The detach fork is free.

**What survives from the smoke unchanged:** it PASSED — both arms trained, emitted a cube, rc=0 —
and the memory measurement, which reproduced across both runs (4107→5617 and 3952→5433 MiB, ×1.37
both times). Peak 5433 MiB is **66% of the 8188 MiB card**.

**Projected arm:** the incumbent is 1.82 h measured, of which ~0.3 h is emit. Treatment ≈
1.5 h × 2.03 + 0.3 h ≈ **3.4 h/arm** ⇒ 4 treatments + 1 recheck control ≈ **15.4 h**.

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

~~The CPU-side probe run during the training-loop audit said ×1.76. The real pipeline says ×2.02 —
that probe was 13% low.~~ **RETRACTED with the numbers above.** The direct probe was the reliable
instrument all along; the log scrape was not. Its ×1.76 came from a single un-repeated run, and
repeating it three times gives ×2.03. The lesson inverts: the cheap *direct* measurement held, and
the elaborate *indirect* one did not.

### Open, and load-bearing for the design

The existing 300-lesson `fullzero_*` controls were trained **before PR #303**. Reusing them saves
~8 h of GPU but assumes the merged changes are inert at `pushforward_weight=0.0`. There is an
argument and a test for that (`test_the_new_guard_is_byte_identical_when_frozen`, plus the
pushforward branch is never entered at weight 0), but no end-to-end evidence. Resolved in the
pre-registration, not here.
