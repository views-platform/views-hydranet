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


## EXP-01 — pushforward at w=0.1, seed 42 (2026-08-30) — **VOID**

**Pre-registration:** `05_analysis_plan.md` (LOCKED `bd1f1ec`), Amendment 1.

**Verdict: VOID.** F5 and F6 both fired. Not "pushforward is worse" — the arm did not test the
hypothesis.

| h | AP control | AP w=0.1 | ΔAP | **oracle** ctl → pf | Δoracle | act_ratio |
|---|---|---|---|---|---|---|
| 1 | 0.4779 | 0.4534 | −0.0245 | 0.4779 → 0.4534 | −0.0245 | 0.83× |
| 6 | 0.4008 | 0.3334 | −0.0673 | 0.4854 → 0.4593 | −0.0262 | 0.54× |
| 12 | 0.3770 | 0.2670 | −0.1100 | 0.4961 → 0.4600 | −0.0361 | 0.40× |
| 18 | 0.3298 | 0.2297 | −0.1001 | 0.4974 → 0.4657 | −0.0318 | 0.33× |
| 24 | 0.2967 | 0.1715 | −0.1252 | 0.4890 → 0.4658 | −0.0232 | 0.33× |
| 30 | 0.2631 | 0.1218 | −0.1413 | 0.4916 → 0.4634 | −0.0282 | 0.20× |
| 36 | 0.2208 | 0.0967 | −0.1241 | 0.4667 → 0.4365 | −0.0302 | 0.43× |

Retention `AP(h18)/AP(h1)`: 0.690 → 0.507. `crps_all` essentially unmoved (≤ +0.005).

**Why VOID and not NEGATIVE.** The oracle is scored teacher-forced — the model is fed real data at
every step, so no rollout is involved. It dropped at *every* horizon. F5 exists for precisely this:
an auxiliary loss that moves the ceiling changed the **model**, and the free-running collapse
cannot then be read as a rollout effect. F6 says the same thing at h1, which is nearly
teacher-forced.

**What it does establish, and it is worth keeping.** At w=0.1 the term is strong enough to trade
away one-step skill and make the model drastically more conservative — firing 0.20–0.33× as often
at long horizons — with AP falling alongside. §7 predicted the conservatism but predicted AP would
*rise* with fewer false positives. It fell. Read against **M45** (AP falls when the model
*over*-fires), this brackets the operating point: **moving firing in either direction from here
hurts.**

**Falsifiers:** F1 PASS (Amendment 1, max 1.48 sd), F2 PASS, F3 floor gate PASS, F4 PASS,
**F5 FIRED**, **F6 FIRED**, F7 n/a (VOID), F8 not evaluated.

**Next, per Amendment 2:** one arm at w=0.01, seed 42. Hard stop either way.

---

## EXP-02 — pushforward at w=0.01, 4 seeds (2026-08-31) — **UNDERPOWERED (negative direction)**

**Pre-registration:** `05_analysis_plan.md` LOCKED `bd1f1ec`, Amendments 1 and 2. All four arms
completed; queue reported `failed/skipped: none`.

### Primary endpoint — AP@h18, free-running, `sb`, calibration

| seed | control | pushforward | Δ |
|---|---|---|---|
| 42 | 0.3298 | 0.3026 | −0.0272 |
| 43 | 0.3318 | 0.3125 | −0.0194 |
| 44 | 0.3058 | 0.2798 | −0.0259 |
| 45 | 0.3352 | 0.3198 | −0.0154 |
| **mean** | **0.3257** | **0.3037** | **−0.0220** |

Exact one-sided permutation **in the observed direction**: **p = 0.0429** (floor 0.0143).
Paired 95% interval **[−0.0308, −0.0131]**.

### Verdict: UNDERPOWERED, per §4 as written

`|ΔAP| = 0.0220 < MDE = 0.024`, and the interval does **not** exclude −MDE. §4 requires
`ΔAP ≤ −MDE` **and** `p ≤ 0.05` for EFFECT NEGATIVE; the first condition fails by 0.004.

**This is deliberately not rounded up.** Four of four seeds negative and p=0.043 is a real signal,
and it would be easy to call it EFFECT NEGATIVE — but the MDE was declared before running precisely
so that this call could not be made afterwards. C-305 and C-306 are on the register for post-hoc
rule overrides. The honest statement is: **a replicated, significant negative effect whose magnitude
is not established to reach the threshold we pre-declared meaningful.**

### The mechanism — this is the part worth keeping

**The oracle does not move.** Teacher-forced AP@h18 across seeds: −0.0024, −0.0065, −0.0015,
+0.0019 — all inside σ=0.0134, F5 PASS 4/4. h1 is unchanged (+0.0012).

So the model is **equally good teacher-forced and worse free-running** — the exact opposite of what
the pushforward is designed to do. This is a clean negative for the method's core claim on this
vehicle, not a confounded one: at w=0.1 the same term damaged the model (EXP-01, VOID) and at 0.01
it leaves the model alone and still degrades the rollout.

### F7 does NOT fire — this is not an M45 artifact

| seed | act_ratio ctl → pf | ratio | ΔAP |
|---|---|---|---|
| 42 | 0.00699 → 0.01054 | 1.51× | −0.0272 |
| 43 | 0.01042 → 0.05733 | **5.50×** | −0.0194 |
| 44 | 0.01398 → 0.05329 | 3.81× | −0.0259 |
| 45 | 0.00586 → 0.00792 | 1.35× | −0.0154 |

`r(firing increase, ΔAP) = +0.008`. Firing rose by anywhere from 1.35× to 5.50× while the AP loss
stayed flat at −0.015 to −0.027. **The damage does not scale with the dose**, so M45's mechanism
does not explain this one and the two results are independent.

### Horizon curve

| h | ΔAP | p | act_ratio |
|---|---|---|---|
| 1 | +0.0012 | 0.371 | 1.16× |
| 6 | −0.0086 | 0.200 | 1.57× |
| 12 | −0.0188 | 0.086 | 2.38× |
| 18 | **−0.0220** | **0.043** | 3.47× |
| 24 | −0.0264 | 0.057 | 5.58× |
| 30 | −0.0213 | 0.100 | 8.36× |
| 36 | −0.0198 | 0.157 | 9.39× |

Damage is absent at h1 and appears from h12 on — consistent with a rollout effect rather than a
model effect, and consistent with F5.

### Two caveats, on the record

1. **The paired interval is a seed-level t-interval, not the pre-registered origin-block
   bootstrap.** `ap_diff_origin_block_ci` needs the prediction cubes, and the queue deletes each
   cube after scoring. The registered statistic was **not computable** after the fact. Recorded as
   a substitution rather than presented as the registered one; re-running it would require
   re-emitting eight arms.
2. **The controls predate PR #303.** Reuse was validated by a fresh control landing within 1.48 sd
   of its archived twin (Amendment 1), not by bit-reproduction — which **C-317** establishes is not
   available on this pipeline at all.

### Falsifiers

F1 PASS (max 1.48 sd, Amendment 1) · F2 PASS · F3 floor gate PASS 4/4 · F4 PASS · **F5 PASS 4/4** ·
**F6 PASS** · **F7 does not fire** (r=+0.008) · F8 not evaluated.

### What this closes, and what it does not

**Closes:** pushforward as a rollout remedy on this vehicle, at the only two weights that are
usable — 0.1 damages the model, 0.01 leaves it intact and makes the rollout worse. Per Amendment 2
there is no third weight.

**Does not close:** the horizon-of-the-loss idea. The reopen triggers registered in §8 stand — the
**detach fork** (`pushforward_detach_state=True`, the stateless reading, measured free in the
smoke) and a **longer unroll**. This result says a 2-step auxiliary term does not help; it does not
say a longer or differently-coupled one cannot.
