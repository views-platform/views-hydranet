# How to read the threshold-reliability table — and the only permitted way to state it

⛔ **This file is the communication half of the instrument.** The script answers *how good is the
predictive distribution*; this file fixes *how that answer may be said*, because over 5–6 November
2026 four statements about the same forecast were made and three were withdrawn — not for bad
arithmetic, but because each one was restated loosely in a later message.

---

> ### ⛔⛔ HOLD — added 2026-11-06 after `/code-review max`, before this file was ever used
>
> **Two of this document's load-bearing claims are not established.** Nothing below is deleted;
> read it with these two corrections in force.
>
> **1. `G-R1` does not do what §2 and §7 say it does.** §2 claims the identity makes a composite
> *"impossible"* and §7 row 1 credits it with closing that defect class. ⛔ It is an **algebraic
> tautology**: `brier_parts` defines `mcb = s_bar - s_rc` and `dsc = unc - s_rc`, so
> `mcb - dsc + unc` expands to `s_bar` identically, whatever `unc` is. Verified here — with `UNC`
> from a deliberately wrong population the gate returns **PASS at residual 0.00e+00** while `DSC`
> is **−0.0503** on the committed fixture, a negative discrimination.
> ⚠️ *Corrected later the same day: this read **−0.1622**, which no committed fixture reproduces —
> it came from an uncommitted scratch probe. The negative-DSC claim itself is reproduced by a test.* ⇒ The three numbers *are* computed correctly, but
> **nothing proves it**, and §7's claim that *"a guard that could not fire"* is closed is false.
>
> **2. §7's last two rows overstate what is protected.** *"Every gate has a test that makes it
> fire"* — three gates declared in this dossier's instruments (`G-B2`, `G-B4`, `G-P3`) **did not
> exist at all**, and `G-P2` was a tautology. All four were repaired on 2026-11-06, after this
> file claimed the class was closed.
>
> ⭐ **What stands unchanged:** §3's null-floor rule, §3b's *read the scale-free table, not the
> verdict column*, §3c's seed-conservative verdict, §4's closed vocabulary, §5's template and
> rules, and §6's limits. Those are the parts that earned their place.
>
> ⚠️ Separately: §5's `MCB/UNC` rule and §6's limits are sound, but **`NOT RESOLVABLE` now has two
> live definitions** — §4 locks it as *"across-seed sd exceeds the statistic"*, while the three
> later instruments also emit it for *too few pairs* and for *distance from chance*. This file was
> never amended when they landed. Reviewer's finding, not re-verified here.
>
> ---
>
> ### ⭐⭐ CLEARED — 2026-11-06, later the same day. Point 1 is fixed; point 2 stands in part.
>
> **1. `G-R1` now asserts the POPULATION, and §2/§7 may be read as written again — with one word
> changed.** The gate checks that `ybar` recomputed from the same `(n_k, s_k)` that produced
> `S_bar` is the reported `ybar`, and that `MCB >= 0` and `DSC >= 0`. Both inequalities are
> **theorems**: the forecast levels `p_k` and the constant `ybar` are each feasible isotonic
> candidates, and PAV returns the optimum over that set, so it cannot be beaten by either. ⇒ A
> mismatched `UNC` pushes `S_rc` above `UNC` and the second one fires. ⭐ The retraction is pinned
> by a test showing the **old** form passing at residual 0.00e+00 on exactly the input it was
> credited with catching. ⛔ **The word that changed:** the identity `S_bar = MCB − DSC + UNC` is a
> **transcription check**, not evidence — it is true by definition. Say *"the three parts are one
> population's, by two assertions that can fail"*. ⛔ Never say the identity *proves* anything.
>
> **2. Still true as written.** `G-B2`, `G-B4` and `G-P3` did not exist and `G-P2` was a tautology;
> all four were repaired. §7's claim that the class was closed was false when made, and the honest
> form is that the class is **actively policed and keeps producing instances** — `G-R1` was the
> fifth this week, found inside the instrument presented as the remedy.
>
> ⚠️ **`NOT RESOLVABLE`'s two definitions are NOT fixed.** Still open, still at the reviewer's
> weight.

## 1 · The question, and why one table answers all of it

A predictive distribution **is** the collection of probabilities `P(Y > t)` for every threshold
`t`. So *"how good is our predictive distribution"* is one question asked repeatedly:

> at threshold `t`, when the model says probability `p`, does it happen with frequency `p`?

| row | what it is | the old separate question it replaces |
|---|---|---|
| `t = 0` | **occurrence** | "does it fire in the right cells?" |
| `t = 1…25` | **the body** | "is the magnitude right?" |
| `t = 50…500` | **the tail** | "can we capture extremes?" |

⭐ These stop being separate measurements that have to be reconciled in prose. They are rows of one
table, from one function, on one population. **That is the whole point.**

---

## 2 · The three numbers, and the identity

The Brier score at each threshold splits exactly three ways (CORP, `Arnold2024` /
`Dimitriadis2021`):

```
S_bar  =  MCB  -  DSC  +  UNC        exactly, BY DEFINITION — a transcription check, not evidence; see §2's correction
```

| | name | what it means | who it is about |
|---|---|---|---|
| **UNC** | uncertainty | how hard the question is — `ybar(1-ybar)` | the world |
| **MCB** | miscalibration | **how wrong our probabilities are** | ⭐ us, the defect |
| **DSC** | discrimination | how much our ranking separates events from non-events | ⭐ us, the skill |

⛔ **What stops a composite here — CORRECTED 2026-11-06.** On 2026-11-06 a figure was withdrawn
for being a composite — an occurrence error multiplied by a magnitude error, reported as the
latter.

⛔⛔ This paragraph used to read *"the identity makes that impossible here: the three parts
**provably** sum to the score, so none of them can be a ratio of unlike things."* **That was
wrong.** `MCB` and `DSC` are DEFINED as `S_bar − S_rc` and `UNC − S_rc`, so the identity holds for
any `UNC` at all, including one from the wrong population. It is a **transcription check** — worth
having, because it is the only thing that catches a wrong scale factor on a reported part, and a
reader can still check the arithmetic in the printed row — but it is not evidence.

⭐ What actually stops a composite is two things that CAN fail: **`G-R5`**, which asserts the
forecast and the truth are aligned per origin before they are ever paired, and **`G-R1`'s two sign
theorems**, `MCB >= 0` and `DSC >= 0`, which a wrong base rate violates. Say *"the parts are one
population's, by assertions that can fail"*. ⛔ Never say the identity proves it.

---

## 3 · ⭐⭐ MCB is never read against zero

A PAV refit removes a little score even from a perfectly calibrated forecast, purely by fitting
noise. So `MCB > 0` is **not** evidence of anything. Every row therefore carries

> **`MCB_0`** — the MCB a *perfectly calibrated* forecast with our own group sizes would show, from
> 2000 Bernoulli resamples.

⛔ **This column is the one that would have prevented this week.** At `t = 100` the positives
collapse to a handful of cell-months, `MCB_0` becomes large, and a raw MCB that looks alarming is
revealed as noise. *A number without its floor is how "the top is too high" got said.*

The decision threshold is `MCB_0`'s **99.5th** percentile — Bonferroni over the ten rows, so the
false-positive rate **across the whole table** is 5%, not 5% per row. ⚠️ The 95th was tried first
and a perfectly calibrated synthetic forecast came back `MISCALIBRATED 1.1x` on the first seed
tried. `test_the_FALSE_POSITIVE_RATE_is_actually_controlled` pins the correction.

---

## 3b · ⭐⭐ Read the SCALE-FREE table, not the verdict column

⛔ **The verdict column measures our power to detect a defect, not the size of the defect.** The
first run made this mistake readable as a finding: `MCB / floor` fell `7.2 → 0.10` across the
table, which reads as *"the tail is fine"*. It is not what happened — the event count falls
**2,905 → 6**, so low thresholds are simply far easier to convict.

On the scale-free `MCB / UNC` the defect is roughly **flat at 0.9%–4.3% everywhere**.

So Table 2 prints three columns, all as fractions of `UNC`, which puts them on one scale:

| column | question it answers |
|---|---|
| `MCB/UNC` | **how wrong** we are |
| `MCB_0/UNC` | ⭐ **what noise looks like at this row's sample size.** At `t = 250` it is 11.9% of `UNC`, so that row's 4.3% `MCB` is nothing |
| `DSC/UNC` | ⭐⭐ **how much of the question our ranking actually answers** |

### ⭐⭐ And `DSC/UNC` is where the tail story actually is

It collapses from **33.8% at `t = 0` to 0.13% at `t = 250`**.

⇒ **The tail failure is a discrimination failure, not a calibration failure.** The probabilities up
there are not too high and not too low — they carry almost no information about *which* cells will
exceed the threshold. That is `recall is free, placement is everything`, measured on the
distribution for the first time, and it is consistent with the top-event table's scatter
(2,482 predicted as 11.1; 435 predicted as 0.125).

⛔ **This is a third thing, and it is why "too high" and "can't capture the tail" both sounded
true.** Neither was the right word.

---

## 3c · ⚠️ The verdict is seed-conservative, and the amendment is declared

On the first run `t = 0` came back `MISCALIBRATED 7.2x` on an `MCB` of `4.87e-05 ± 3.25e-05` — an
across-seed sd two thirds the size of the statistic, so the factor was anywhere in `2.4x…12x`
depending on which seeds you drew.

⇒ The factor is now computed from **`MCB − sd`**, the edge all five seeds support, so a stated
factor **can only move up with more seeds, never down**. Two rows (`t = 25`, `t = 50`) move from
`MISCALIBRATED` to `CALIBRATED` under the stricter rule.

⛔ Both verdicts are printed, the pre-registered one beside the headline, and the first run's output
is kept at `results/threshold_reliability_v1_prereg.out`. **The amendment was made after seeing the
data and says so** — it is not swapped in silently, and `test_the_conservative_factor_is_NEVER_LARGER_than_the_pre_registered_one`
pins that it can only ever weaken a claim.

---

## 4 · ⛔ The fixed vocabulary — exactly four tokens

Assigned by arithmetic, in this order. **No other sentence may be used to state a row.**

| token | assigned when | what may be said |
|---|---|---|
| `CALIBRATED` | `MCB − sd ≤ MCB_0`'s 99.5th pct | ⛔ **no defect is quotable for this row.** ⚠️ It means *not detectably miscalibrated at this sample size* — check `MCB_0/UNC` before reading it as *right* |
| `MISCALIBRATED Nx` | `MCB − sd` exceeds it by factor `N` | the probabilities at this threshold are wrong, by `N` times the noise floor — ⭐ a floor all five seeds clear |
| `NO FORECAST ISSUED` | every cell got `p = 0` | ⛔ **neither "too high" nor "too low" is sayable** — the model assigns no probability to exceeding `t` anywhere |
| `NOT RESOLVABLE` | across-seed sd of `MCB` exceeds `MCB` | ⛔ five seeds cannot separate this row from noise; **no verdict** |

### ⭐⭐ The token that fixes the specific confusion of 2026-11-06

`NO FORECAST ISSUED` exists because *"the top is too high"* and *"we can never capture the tail"*
sound like a contradiction and are not. They were about different axes:

- the **top prediction bin** (6% of cells where the model predicts most) over-states its magnitude;
- the **tail of the outcome** is under-predicted by 47×–3,480×.

A model that assigns probability zero to exceeding 100 everywhere is **not** over-predicting the
tail and **not** under-predicting it. It is silent. The token says so, and no later message can
turn silence into a direction.

---

## 5 · ⛔ The report template — copy it, do not paraphrase it

Any statement of this result, in chat, in a finding, or to a peer session, takes this form:

```
THRESHOLD RELIABILITY · <arm set> · h<N> · <target> · <date>
population   <N> cell-months, <M> seeds, months <a>..<b>
estimator    Brier, plug-in; CORP three-way split; MCB vs its 99.5th-pct null floor

  P(Y > 0   ) : <TOKEN>
  P(Y > 1   ) : <TOKEN>
  ...

how wrong       MCB/UNC <lo>%..<hi>% across live rows   (scale-free; the DEFECT)
our power       MCB/floor <lo>..<hi>                    (⛔ NOT the defect)
our signal      DSC/UNC <at t=0>% -> <at the top>%       (⭐ discrimination)
```

### Three rules on top of the template

1. ⛔ **A token is never upgraded or softened in prose.** `CALIBRATED` does not become "roughly
   fine"; `MISCALIBRATED 1.4x` does not become "a serious defect". The factor is the adjective.
2. ⛔ **No scalar without its decomposition beside it.** Both of this week's errors were scalars
   that had travelled away from their parts.
3. ⛔ **Never quote `MCB/floor` as the size of a defect.** It is a signal-to-noise ratio and it
   varies 70-fold across this table purely because the event count varies 500-fold. Quote
   `MCB/UNC` for size and `MCB/floor` only for confidence.
4. ⛔ **Every number names its population, transform, estimator and normalisation** (`C-351`). The
   template's header block is that, pre-filled.

---

## 6 · ⚠️ What this instrument cannot tell you

| | |
|---|---|
| ⛔ | It scores the **distribution**, not the **placement of named cells**. The cell where 2,482 was predicted as 11.1 enters the high-threshold rows as miscalibration; this instrument will not name it. That stays a separate look — and conflating the two is exactly what went wrong on 2026-11-06 |
| ⚠️ | `p` is quantised at 1/16 = 0.0625, against a base rate of 0.0034. ⇒ at `t = 0` the model **cannot** be calibrated on the 99.46% of cells where it says zero: it has no value between 0 and 0.0625 to say. The occupied-level count is printed per row so this is visible, not inferred |
| ⚠️ | h1 only, target `sb` only, control architecture only, one checkpoint |
| ⚠️ | It describes the **delivered composed emission**. Where `threshold_gate` closed, the body is unrecoverable, so this is the emission's calibration, not the head's |
| ⛔ | It does **not** say whether v1 (the rank-coupled body) would help, and it does not rank arms |
| ⛔ | `CALIBRATED` at `t ≥ 50` rests on 6–213 events. It means *undetectable at this sample size*, and `MCB_0/UNC` there runs 1.2%–13.6%. ⚠️ **It is not a tail result.** The tail statement this instrument does support is the `DSC/UNC` collapse, which does not depend on convicting a calibration error |

---

## 7 · Why this one is trustworthy when the last four were not

| failure, 5–6 Nov | what prevents it here |
|---|---|
| a composite reported as one of its factors | `G-R1`: finite, transcribes to 1e-12, and ⭐ `MCB >= 0` / `DSC >= 0` (theorems). ⛔ NOT a population check — that is `G-R5` |
| the statistic moved with its bin count | ⭐ **no free parameter** — `p` takes 17 values, so the table is exact |
| conditioned on the outcome | `G-R2`: permute the truth, every group size must be bit-identical |
| a single-seed number | `G-C2`: five seeds or nothing, sd printed in the same cell |
| a number quoted without its noise floor | `G-R4`: no MCB reaches the table without `MCB_0`; no verdict without both |
| a verdict restated loosely next message | §4's closed vocabulary and §5's template |
| a guard that could not fire | every gate has a test that makes it fire; `G-R4`'s first version could not, and that is recorded in its docstring |
| a verdict that depended on which seeds were drawn | §3c: the factor is computed from `MCB − sd`, so it can only move up |
| a power statistic read as a defect size | §3b: three columns on the `UNC` scale, and rule 3 in §5 |
