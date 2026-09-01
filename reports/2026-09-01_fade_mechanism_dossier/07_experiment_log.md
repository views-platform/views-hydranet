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
