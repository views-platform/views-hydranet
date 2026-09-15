# Experiment log — does clamping the cell state stop the fade?

## EXP-01 (2026-09-01) — **CONFIRMED at both levels**

**Pre-registration:** `05_analysis_plan.md`, LOCKED `cb6b700` with `results/` empty.
**Emit only, no training.** Each comparison is **within seed**.

### F3 identity check — PASSED byte-identical

The clamped arm scored `AP@h18 = 0.3621885544392029`, exactly the archived seed-42 `cell` value from
M48. It is the same arm, now with per-step recording enabled.

### 3a — the field (seed 42): the collapse essentially stops

| step | firing, no clamp | firing, clamped | ratio | magnitude no clamp | clamped |
|---|---|---|---|---|---|
| 1 | 0.000612 | 0.000612 | 1.00× | 18.37 | 18.37 |
| 6 | 0.000148 | 0.000296 | 1.99× | 12.26 | 14.02 |
| 12 | 0.000032 | 0.000309 | 9.69× | 7.22 | 15.17 |
| 18 | 0.000009 | 0.000301 | 32.4× | 1.92 | 15.03 |
| 24 | 0.000003 | 0.000307 | 119× | 0.80 | 14.97 |
| 30 | 0.000001 | 0.000304 | **308×** | −0.74 | 15.71 |

**Unclamped: 1,547× collapse in firing over 35 steps. Clamped: 2×.** Magnitude on active cells
18.37 → −0.81 unclamped, 18.37 → 13.56 clamped.

The clamped arm is **flat from step 6 onward** — 0.000296, 0.000309, 0.000301, 0.000307, 0.000304.
It does not decay slowly; it stabilises.

**F1 does not fire.**

### 3b — the state (seed 43): the whole state is rescued, not just the clamped half

| step | baseline `max|h|` | hidden | cell | clamped `max|h|` | hidden | cell |
|---|---|---|---|---|---|---|
| 0 | 65.62 | 0.980 | 65.62 | 65.62 | 0.980 | 65.62 |
| 10 | 19.74 | 0.801 | 19.74 | 66.08 | 0.922 | 66.08 |
| 20 | 5.13 | 0.630 | 5.13 | 66.08 | 0.957 | 66.08 |
| 35 | **1.60** | **0.565** | 1.60 | **66.08** | **0.926** | 66.08 |

Baseline drains **41×**. Clamped holds (1.0×).

**F5 — the trap the plan registered — is answered, and this is the actual finding.** The *cell* half
being held is true by construction and proves nothing. The **hidden** half evolves freely under the
clamp, and it drains **0.98 → 0.56 (1.7×) without** the clamp versus **0.98 → 0.93 (1.05×) with**
it. **Holding the cell stabilises the half it does not touch.** C-292's note that
`hs = o ⊙ tanh(hl)` predicted this was possible; it is now measured rather than inferred.

**F2 does not fire.**

### Verdict

**The fade is real, and the clamp stops it.** The chain is now measured end to end at both levels:

```
clamp the cell  →  state stops draining (41× → 1.0×)
                →  emitted field stops collapsing (1547× → 2×)
                →  AP improves, 0.000 at h1 rising to +0.059 at h36  (M48)
```

This supplies the mechanism M43 and M49 each failed to provide — because both were looking at the
wrong property. M43 asked whether the state left its *range*; it does not, because collapsing toward
zero is inside the range. M49 asked whether the field *blurred*; it does not, because a vanishing
field has no structure to smear. **Neither asked about magnitude, which is what actually decays.**

It also explains the programme's four negative results in one line: M42 (ITF), M45 (truncated_nb),
M47 (pushforward) and M26–M33 (scheduled sampling) all changed *how the model decides to fire*, while
the signal driving that decision was draining to nothing.

### What this does NOT establish

**That the fade causes the AP loss.** Clamping stops the fade and improves AP, but both could follow
from a third property of holding the state. Establishing causation needs an intervention that stops
the fade *without* clamping — out of scope here, and stated so the correlation is not read as proof.

Also: seed 42 for the field, seed 43 for the state. The two levels are consistent but were measured
on sibling vehicles, not the same run.

### An anomaly, recorded not explained

Unclamped `mean_magnitude_on_active` goes **negative** by step 30 (−0.74, −0.81 by step 35) in a
log1p space where a fed count should not be negative. The clamped arm never does this. It may be a
separate defect in the feedback path; it is not load-bearing for anything above.

**Cost: ~15 minutes, emit only, no training.**

---

## CORRECTION (2026-09-02) — the magnitude claim above is wrong; the finding is sharper without it

Found by an `/expert-code-review` commissioned to design the next experiment, which read the CSVs
instead of the write-up. **The mechanism holds. One clause of its description does not.**

### What was wrong

`mean_magnitude_on_active` carries an explicit **`-1.0` UNDEFINED sentinel** when `n_active == 0`
(`views_hydranet/utils/hydranet_inference.py:527-533`). The column's own code comment warns:

> *"averaging the column would then mix empty fields with scattered ones — biasing the statistic
> downward exactly in the collapse regime this study is about."*

That is precisely what the table above does. `counts = expm1(field).clamp(min=0.0)`, so a mean of
clamped non-negatives **cannot be negative** — the −0.74 was the tell, and it was recorded as an
"anomaly, recorded not explained" rather than recognised as a sentinel.

| step | control: % records that are sentinel | raw mean (as published) | filtered mean |
|---|---|---|---|
| 1 | 0% | 18.37 | 18.37 |
| 18 | 79% | 1.92 | 12.82 |
| 30 | 97% | **−0.74** | 7.20 |
| 35 | 99% | **−0.81** | 13.50 (n=2 records) |

Worse than a wrong number: the table compared the **clamped arm's 15.71** — a real mean over
153/156 records — against the **control's −0.74**, which is 97% sentinel. Two different quantities
placed side by side as if they were one.

### What is actually true

| | control | clamped |
|---|---|---|
| active fraction, step 1 → 35 | 0.000612 → 0.0000004 (**1,547×**) | 0.000612 → 0.000315 (**2×**) |
| magnitude on cells that fired | 18.37 → 13.50 | 18.37 → 13.84 |

**Magnitude does not collapse, and the clamp does not restore it — the two arms end up at
essentially the same magnitude (13.5 vs 13.8).** The entire effect is **occurrence**.

`active_fraction` is `n_active / n_cells` and can never be a sentinel, so the 1,547× and 2× figures
— the load-bearing ones — are unaffected. The state numbers come from a separate capture path
(`state_*.pt`) and are untouched.

### The corrected mechanism

```
clamp the cell  →  state stops draining (41×  →  1.0×)
                →  the model keeps firing SOMEWHERE (1547× → 2× collapse in occurrence)
                →  AP rises, 0.000 at h1 to +0.059 at h36
```

**The clamp preserves *where* the model fires, not *how loudly*.** That is a cleaner claim than the
one it replaces, and it sits better with the rest of the programme: M32/M45 already established
that **placement is everything** — thinning events costs 3% of the oracle, scrambling their
locations costs 81%.

### What this changes about M43 and M49

The original said *"neither asked about magnitude, which is what actually decays."* **That sentence
is now wrong in its own right.** M43 asked about range, M49 about spatial structure, and this asked
about magnitude — and magnitude turns out not to be the answer either. The property that decays is
**occurrence**, and none of the three asked about it directly.

### Registered as C-318

A sentinel value averaged as a measurement, in a published ledger row, with the tell (a negative
mean of a non-negative quantity) visible in the output and written down as an unexplained anomaly
instead of chased.
