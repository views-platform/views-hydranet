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

---

## S5/S6 (#317/#318) — **the noise arm is much worse, and it is the FIRING lever again** · 2026-09-05

**Pre-registration:** `05_analysis_plan.md` locked `47d66af` before S1 ran; amendment A1 (`2d9cd72`)
before S5 launched. Both arms retrained, ε=0, 300 lessons, seed 42, one variable
(`input_noise_dropout: 0.204`). Weight hashes differ (`cab486bc…` / `64a75eab…`).

### Gates, all passed before the number was read

| gate | result |
|---|---|
| SMOKE (2 lessons) | PASS |
| POTENCY, arm's own config, **trained checkpoint** (C-324/C-325) | **POTENT** — loss 265.80 → 106.27, rel 0.600 |
| **FG-A** — control ranks above chance (C-299) | **PASS** |
| FG-C — the effect exceeds what the setup resolves (reported, not binding per A1.2) | **PASS** — a 30% effect is 0.2305 AP; 3×MDE is 0.1800 |
| Weight-hash, before any score was read | **PASS** — arms genuinely distinct |

**Vehicle sanity:** control `AP@h18 = 0.3292` against the known ε=0 reference **0.3318** (M34–M37).

### Primary

| | control | noise | Δ |
|---|---|---|---|
| **AP@h18** (`sb`, free-running, 13 origins) | **0.3292** | **0.1329** | **−0.1963** |

**−0.1963 is 3.3× the ±0.06 band this design cannot see inside.** It is worse at every horizon
(h1 −0.118, h18 −0.196, h36 −0.118).

### The mechanism — F5 fired, and it is the lever again

| h | act_ratio control → noise | factor |
|---|---|---|
| 1 | 0.4067 → 0.8168 | ×2.0 |
| **18** | **0.0123 → 0.6903** | **×56** |
| 36 | 0.0005 → 0.5899 | ×1,180 |

**The model fires 56× more at h18.** `size_ratio` also rises (0.0000 → 0.0250) and `Brier` worsens at
every horizon, while `crps_events` barely moves — so this is occurrence, not magnitude.

**This is M45 for the sixth time**: *AP loss scales with how much the model FIRES.*

### ⛔ Why it fired more, when the augmentation only ever SILENCES

This is the part worth keeping, and it is a **design** error, not an implementation one.

The augmentation deletes events from the **input** while the **target keeps them**. So the task
becomes: *given a field with a fifth of its events missing, predict a field that still has them all.*
The optimal response to that task is **to invent occurrence** — and the model learned exactly that.

The "no target adjustment" decision came from `SanchezGonzalez2020`'s own supplement, which states it
*"happens implicitly when the loss is defined directly on next-step ground-truth"*. That sentence is
true, and it was verified against our loss structure — **but it does not survive the change from
their corruption to ours.** Their noise is small symmetric jitter on a dense field: zero-mean, so the
implicit target is unbiased. **Deletion is not zero-mean.** Silencing a fifth of the events makes the
input systematically sparser than the target, and the unbiased response to a systematically sparse
input is to over-predict.

S1 measured the right thing and the selection rule chose the right *family*. What was wrong is the
step from "the model silences events" to "so silence events in its input" — that mimics the
**symptom** the model produces, not the **conditions** it faces. The model's own errors arrive with
matching degraded targets downstream; a training corruption with intact targets does not.

### Verdict against the locked rule

`Δ ≤ 0` ⇒ **does not survive the screen.** The rule's own wording — *"INCONCLUSIVE, not 'input noise
does not work'"* — was written to stop a **null** being converted into a closure (C-307). This is not
a null: −0.1963 at 3.3× the band, worse at every horizon, with a coherent mechanism and a 56× firing
signal. So the honest split is:

* **Well supported:** *this design, at this rate, on this vehicle, is substantially harmful, and the
  reason is that it trains the model to invent occurrence.* The mechanism raises confidence rather
  than resting on the effect size alone.
* **NOT supported, and the rule rightly forbids it:** any claim about input-noise augmentation in
  general, about other rates, or about a variant that degrades the target alongside the input. One
  design, one rate, one seed, one vehicle.

A 4-seed confirmation is **not** worth buying: the pre-registered trigger for seeds was `Δ ≥ +0.02`,
and re-measuring a −0.196 more precisely buys nothing.

---

## S7 (#319) — disposition · 2026-09-05

**No ADR.** The design is harmful on this vehicle and the mechanism is understood; there is nothing
to promote.

**The code stays, default-off.** `input_noise_dropout` is merged, `None` by default, byte-identity
proven by test, 2,029 tests green, three independent mutation audits. Removing it would discard a
tested seam that the obvious follow-up needs — and the follow-up is specific: **degrade the target
alongside the input**, so the model is asked to predict what a degraded field implies rather than to
hallucinate what was deleted from it. That is the variant this result argues for, and it is *not*
what was tested here.

**What this epic bought, beyond the negative:**

* **M64** — the sixth confirmation that AP loss tracks firing, now with a *causal* account rather
  than a correlation: an augmentation that only ever silences made the model fire **56×** more,
  because silencing the input while keeping the target trains it to invent occurrence.
* **M63/S1** — a truth-referenced characterisation of the model's free-running error (FN 0.9959 vs
  FP 0.000027 at h18, CV 0.002 across origins) that did not exist before and is reusable by #309/#310.
* **C-328** — a Tier-2 defect class, and the second instance of an auxiliary forward inheriting the
  training input path.
* **`scripts/floor_gate.py` back in service** — first dossier to invoke it since August, and it
  passed both clauses.
* A method result worth more than the experiment: **three independent audit rounds each found
  defects the author's own checking had missed**, and the one that mattered — the BatchNorm leak —
  was found by *writing a test for an untested prose invariant*, not by mutation testing.

**Next**, per the library work that opened this epic: **#310, direct multi-horizon**. It is the road
`Aceituno2025_TemporalHorizons` argues for on proven grounds — long-horizon minima generalise, short
ones do not — and it sidesteps the exponential-gradient barrier rather than fighting it, because
predicting all horizons at once has no recursion to backpropagate through. Six failures on the
mitigate-the-recursion family is now a lot of evidence.

**Epic #311 closed.**

---

## ⛔ Correction to S5/S6, from `/code-review max` · 2026-09-05

**The arm ran a harsher schedule than `02_design.md` specifies.** The implementation dropped on the
first step of every segment; the design fits survival as `S(h) = (1-p)^(h-1)` — horizon 1 **clean** —
and deployment matches that, because inference feeds the real observation at the seed step.

| h | fitted survival | implemented |
|---|---|---|
| 1 | 1.0 | **0.796** |
| 18 | 0.02068 | 0.01646 |
| 36 | 0.00034 | 0.00027 |

A flat **20.4% over-silencing at every horizon** — four times the design's own 5% residual tolerance.

**What this does and does not change.** The **direction stands**: the mechanism is over-firing caused
by *intact targets*, and the schedule does not change that — the model is asked to predict events its
input lacks either way. What is **not** attributable is the effect size: −0.1963 is the loss at a
schedule ~20% harsher than p=0.204 as designed.

Fixed in code (the segment start is now clean, and a validator rejects `time_steps < 2`, where every
step would be a start and the augmentation could never apply). **A re-run was not bought:** the
pre-registered trigger for more GPU was `Δ ≥ +0.02`, and the measured Δ is −0.196.
