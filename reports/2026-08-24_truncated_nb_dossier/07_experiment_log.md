# Experiment log — removing the double-applied zero process

Pre-registration `05_analysis_plan.md`: **LOCKED `21d89e8`** with `tools/` empty; **AMENDMENT 1**
committed after the arms were scored and **disclosed as such** (see §"A defect in my own rule").

---

### EXP-01 · 2026-08-24 · **EFFECT (NEGATIVE) — the fix is catastrophic, and that is the finding**

| seed | control | `truncated_nb` | Δ AP@h18 | Δ AP@h1 | floor |
|--:|--:|--:|--:|--:|---|
| 42 | 0.3298 | **0.1113** | −0.2186 | −0.0212 | PASS |
| 43 | 0.3318 | **0.0970** | −0.2348 | −0.0277 | PASS |
| 44 | 0.3058 | **0.0497** | −0.2561 | −0.0273 | PASS |
| 45 | 0.3352 | **0.0943** | −0.2408 | −0.0351 | PASS |

**`p = 0.0143` — the exact `1/C(8,4)` floor, i.e. maximum separation: all four treatment arms fall
below all four controls.** Mean ΔAP = **−0.2376**, which is **7.5×** the 3×MDE bar of 0.0317, with
unanimous sign. §5 returns **EFFECT (NEGATIVE)**.

**All four arms PASSED the floor gate**, so this is not the `truncated_smoke` failure mode repeating —
these vehicles trained and had dynamic range. The result is real.

## The model is fine. The feedback loop is destroyed.

| seed | trunc free | trunc **oracle** | control free | control **oracle** | free/oracle: trunc vs ctrl |
|--:|--:|--:|--:|--:|--:|
| 42 | 0.1113 | 0.4720 | 0.3298 | 0.4974 | **0.236** vs 0.663 |
| 43 | 0.0970 | 0.4681 | 0.3318 | 0.5014 | **0.207** vs 0.662 |
| 44 | 0.0497 | 0.4626 | 0.3058 | 0.4910 | **0.107** vs 0.623 |
| 45 | 0.0943 | 0.4625 | 0.3352 | 0.4932 | **0.204** vs 0.680 |

**Handed a real field, `truncated_nb` is only ~5% below the control** (0.4625–0.4720 vs 0.4910–0.5014).
Handed its own field it retains **~0.19 of its ceiling** against the control's **~0.66**. The change did
not make a worse forecaster; it made a **catastrophically unstable rollout**. h1 barely moves (−0.02 to
−0.04), which says the same thing from the other end.

## F2 — the mechanism engaged, and overshot

Step-1 `act_ratio` (emitted actives ÷ true actives), where the input is entirely real and no feedback
has happened yet:

| seed | control h1 | trunc h1 | trunc h18 | trunc h36 |
|--:|--:|--:|--:|--:|
| 42 | 0.363 | **1.473** | 8.17 | 20.7 |
| 43 | 0.398 | **1.583** | 12.1 | 48.6 |
| 44 | 0.419 | **1.160** | 11.5 | 28.4 |
| 45 | 0.348 | **1.746** | 22.0 | 68.6 |

**F2 PASSES: the mechanism did exactly what M44 predicted it would.** Occurrence moved from ~0.36× of
truth to 1.16–1.75× — i.e. past the gate's own 1.28× over-prediction — because the body no longer
cancels any of the gate's fires. **And then it compounds**: ×8–22 by h18, ×21–69 by h36.

**This is the pre-registered case where the mechanism engaging and AP collapsing are both true**, and
§6 says that combination is interpretable: the fix is real, and it is harmful.

## §5 magnitude guardrails — every one moves the wrong way

Seed 42, h18: `crps_all` 0.134 → **0.520**; `size_ratio` 0.000 → **1.90**; `precision_at_k` 0.385 →
**0.166**; `n_false_pos` 952 → **1291**. Seed 43 h36 is worse still: `crps_all` 0.87 → **8.60**,
`size_ratio` 0.00 → **10.66**. **There is no trade to weigh here** — §5's "AP gain with a `crps_all`
regression is a trade" clause never engages, because AP falls too.

## What this establishes

**The double-counted zero was load-bearing.** It was not a defect being tolerated; it was an accidental
brake on the autoregressive loop, compensating for a gate that over-predicts occurrence by 28% (M44)
and localises poorly. Remove the brake and the rollout blooms — the C-113 failure mode, re-created by
removing a suppressor rather than by adding a driver.

**M44 is NOT overturned.** The zero process really is applied twice; that decomposition was measured
independently and stands. What is falsified is the inference *"therefore removing it helps"*. **A real
defect is not the same as a binding constraint** — the clearest statement this programme has of a
mistake it keeps making.

**It also vindicates M21**, whose confounded, floor-limited hint pointed exactly this way and which §3
recorded as "not an encouraging prior".

**And it completes a pattern across four independent interventions.** ITF (act_ratio ×1.6, ΔAP −0.019),
scheduled sampling (×5, −0.043), `truncated_nb` (×1170, −0.238) — the AP loss scales monotonically with
how much each makes the model fire. The only rollout intervention that ever *worked*, the cell-state
freeze (**+0.039**), is the only one that does not touch firing at all. **Raising recall in the rollout
is not merely unhelpful; it is actively harmful in proportion to the dose.**

## A defect in my own rule — disclosed, not quietly patched

§5's EFFECT clause required `p ≤ 0.05` from a permutation test I implemented as **one-sided for
`treatment > control`**. A strongly negative effect therefore scored **`p = 1.0`** and fell into
`NULL / UNDERPOWERED`. The first rendered verdict read *"NULL / UNDERPOWERED, p=1.0000"* for a −0.2376
effect at 7.5× the MDE bar — **the rule as implemented could not report the result the experiment
produced**, and my own `EFFECT (NEGATIVE)` branch was unreachable.

Found **after** seeing the numbers. AMENDMENT 1 fixes the test to run in the observed direction,
justified from the locked prose (§5's *"all four seeds agree in sign"* is meaningless if only one sign
can fire) rather than from the values. **No threshold moved.**

## Scope and registered false-negative mode

4 seeds, one architecture, `sb`, L=300, calibration. §7 stands: the gate is retrained alongside the
body, so this closes *"swapping the body fixes rollout skill"*, **not** *"the double-zero diagnosis was
wrong"*. The ZINB `self_zeroed` cross-check remains unbundled and unrun.

## Run notes

* Four arms, ~2.4 h each; `train+emit` 137–192 min vs 81–96 for `nb` controls, and **oracle 47–56 min
  vs 6–13** — the truncated sampler benchmarks **7.3×** slower than `nb` (322 vs 44 ms/call at
  180×180×3, k=4), contradicting `c07a352`'s "parity with nb". Training is unaffected because ε=0
  means no sampling in the loop; **emit and oracle are where it lands**.
* The `setsid` finisher **refused to assemble** — its 12 h ceiling expired before the queue finished,
  because I sized it from a ~2 h/arm estimate that turned out to be ~2.4 h. **It failed safe**: it
  wrote *"QUEUE_DONE never appeared — refusing to assemble a possibly stale verdict"* rather than
  emitting a verdict built from three arms. The guard worked; the estimate did not.
