# Experiment log — does the emitted field blur across the rollout? (#301)

## EXP-01 — Stage 0/1, seed 42 (2026-08-31) — **H FALSIFIED**

**Pre-registration:** `05_analysis_plan.md`, LOCKED `4cf9fa8` with `results/` empty.
**Emit only, no training.** Two arms on one artifact: `identity` (free-running) and `use_real`
(real field fed back every step, so blur cannot accumulate by construction).

### Harness check first (§5) — PASSED exactly

At h1 both arms predict from real data, so their fields must be identical. **13 of 13 origins
byte-identical, max |difference| = `0.000e+00`**; by h36 they differ by 0.944. The arms are what
they claim.

### Result

| h | `moran_i` identity | `moran_i` use_real | Δ | `conc1pct` identity | `conc1pct` use_real |
|---|---|---|---|---|---|
| 1 | 0.6554 | 0.6554 | +0.0000 | 0.4999 | 0.4999 |
| 12 | 0.5408 | 0.6252 | −0.0844 | 0.4873 | 0.5017 |
| 18 | 0.5189 | 0.6237 | −0.1048 | 0.4475 | 0.4945 |
| 30 | 0.4842 | 0.6451 | −0.1609 | 0.3514 | 0.4792 |
| 36 | **0.4904** | **0.6492** | **−0.1588** | **0.3001** | **0.4570** |

`moran_i` under `identity`: **0.6554 → 0.4904**. Under `use_real`: 0.6554 → 0.6492 (flat).

### Verdict: **S1 FIRES — the hypothesis is dead. S5 FIRES — it is the other signature.**

The instrument was validated on synthetic fields before it saw data
(`tests/test_field_sharpness.py`), and that table is what makes this readable:

| | blur | displacement | thinning |
|---|---|---|---|
| `moran_i` | **rises** (0.64 → 0.97) | unchanged | **falls** (0.64 → 0.29) |

**Moran's I falls.** Blur raises it. **The field does not blur — it goes quiet.** `conc1pct` falls
too, and blur raises that as well, so both detectors agree against the hypothesis.

`use_real` is flat on both metrics, so **S2 does not fire**: this *is* accumulation through the
rollout and not a property of later calendar months. The control worked.

### The confound, stated rather than buried

`act_ratio` under `identity` collapses **0.3627 → 0.000246 across the horizon — a factor of
1,475** — while `use_real` holds at ~0.36. By h36 the gate field's maximum is 0.0596 (from 0.985)
and 0.11% of cells exceed 0.01. This is the #258 rollout collapse, already known.

**The Moran's I decline cannot be separated from that collapse by this design.** My own validation
says thinning lowers Moran's I, and there is a 1,475× thinning. The field retains measurable
relative structure (sd/mean 6.6 at h36 vs 9.45 at h1), so the metric is not degenerate — but
"the model stops firing" and "the field loses spatial organisation" are the same event here, and
nothing in this experiment tells them apart.

**So: the blur hypothesis is falsified, and the alternative reading is UNRESOLVED.**

### Stopped here, per the plan

Stage 2 (does the clamp slow it?) was pre-registered to run **only if Stage 1 survives**. It did
not. Asking whether the cell clamp slows the *quieting* is a different question about a different
mechanism, and it needs its own registration rather than being appended to a falsified one.

**Spend: ~25 minutes of a 1-hour budget, no training.**

### What this changes

**M48 still stands** — the cell anchor works, replicated on 4 seeds. What is now excluded is one
explanation for *why*. The clamp does not work by preventing the field from blurring, because the
field does not blur.

The prediction on record in §7 was that `moran_i` would **rise**. It fell. That is the third
prediction in this programme to be wrong (after the `conc1pct` direction and the `fss_ratio`
discriminator, both caught by the synthetic tests), which is the argument for keeping the
instrument-validation step rather than trusting the story.
