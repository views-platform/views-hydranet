# 07 — Experiment log

Append-only. Newest at the bottom. Every entry links its pre-registration, names the one variable, and
states its verdict against the pre-committed decision rule.

**Negatives are written at the same length as wins.** An underpowered screen is recorded as
INCONCLUSIVE, never as a closure — that is **C-307**, already on the register.

_(no entries yet — S1 is the first)_

---

## S1 (#313) — the model's free-running error · **IT GOES SILENT** · 2026-09-04

**Pre-registration:** `05_analysis_plan.md` §5, committed `47d66af` before this ran.
**Rule lock:** `rule_md5 = e7cf96f4…` · **Vehicle:** `fullzero_fortytwo` (ε=0.0, 300 lessons, seed 42)
· **Support:** 13,110 cells × 13 origins, truth-referenced throughout (C-319).

### The measurement

| h | act_true | **FN rate** | **FP rate** | FN/FP | FN (hard) | mag err | CV(FN) |
|---|---|---|---|---|---|---|---|
| 1 | 0.00788 | **0.8104** | 0.001383 | 586× | 0.4311 | −0.53 | 0.028 |
| 6 | 0.00784 | **0.9428** | 0.000333 | 2,831× | 0.7187 | −1.08 | 0.009 |
| 12 | 0.00809 | **0.9846** | 0.000078 | 12,682× | 0.8700 | −1.42 | 0.007 |
| **18** | 0.00908 | **0.9959** | **0.000027** | **36,870×** | 0.9585 | −1.97 | **0.002** |
| 24 | 0.00933 | 0.9991 | 0.000009 | 117,315× | 0.9881 | −2.13 | 0.000 |
| 36 | 0.01044 | 0.9999 | 0.000001 | 674,733× | 0.9983 | *(n=3)* | 0.000 |

**FN** = expected fraction of TRUE events the model silences. **FP** = expected fraction of TRUE zeros
it fires on. **FN (hard)** = fraction of true events where *no draw* fired at all.

### Verdict against the pre-registered rule

**FN ≥ 2× FP by a factor of 36,870 at h18 ⇒ `occurrence_dropout`.** STOP-gate (a) passes with room to
spare: the dominant rate's coefficient of variation across the 13 origins is **0.002**, against a gate
of 0.5 — this is one of the most stable quantities this programme has ever measured.

**The model does not over-fire, jitter, or drift. It goes silent.** By h18 it has silenced **99.6%** of
true events while firing on **0.003%** of true zeros. That is not a perturbation to be modelled with
Gaussian noise; it is near-total extinction of occurrence. **Any design that adds dense noise to a
99% zero field would have been modelling the wrong failure entirely** — which is what transplanting
the paper's σ would have done.

### ⚠️ Two things S2 must not read off this table naively

**1. 81% of the silencing at h1 is NOT a rollout failure.** At h1 there is barely any rollout, and the
model already silences 81% of true events — that is its own conservatism (the gate is deliberately
timid; `act_ratio` at h1 is 0.39). The **rollout-induced** part is the growth on top:

| h | 6 | 12 | 18 | 24 | 36 |
|---|---|---|---|---|---|
| fraction of the events h1 *kept* that are lost by h | 0.698 | 0.919 | **0.978** | 0.995 | 0.9995 |

S2 must parameterise against the **incremental** figure, not the raw one. Matching h18's 0.996 would
train the model on an essentially empty input and "correct" a property that exists where there is no
recursion to correct.

**2. The magnitude column thins to nothing.** Cells active in *both* truth and forecast: 762 at h1,
**63 at h18**, **3 at h36**. The h36 magnitude median rests on **three cells and must not be used.**
Recorded explicitly because M50's original defect was a late-horizon magnitude claim resting on n=2 of
156 — the same shape, and it produced a retraction.

### Scope

One vehicle, one target (`sb`), one artifact (trained 2026-08-18, before today's code). S5's control is
retrained, so the exact rates may shift; what S2 depends on is the **shape** — FN ≫ FP by four to six
orders of magnitude, stable across origins — which is not a fragile finding. `--keep-cubes` was used
with a **single** arm, so C-321's contamination path (the flag skips the multi-arm guard) does not
apply; the single-arm precondition is asserted in `tools/emit_s1.sh`, not assumed.

**Instrument:** 18 tests, **15/15 mutations caught**, including the truth-month off-by-one, FN counting
`q` instead of `1−q`, the magnitude channel silently absorbing silenced cells, and the STOP-gate never
firing. Restored from file backups rather than `git checkout`, so no uncommitted work was at risk.
