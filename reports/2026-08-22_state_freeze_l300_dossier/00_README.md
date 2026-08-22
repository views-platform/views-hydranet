# State-freeze at L=300 — ANSWERED 2026-08-22

**Question.** **M8** says freezing recurrent state recovers gate AP@h18 `0.0070 → 0.0912`. It was
measured on a **40-lesson** vehicle that **M28** now classifies as a smoke test, and the pre-registered
confirmation was never run — making it the primary suspect in **#280**. **M8's *recovered* value is
3.6× below what an L=300 model scores free-running with no intervention at all.**

> ## It helps. It is the cell state. And the interval excludes zero.
>
> | h | `none` | `cell` | diff | 90% CI (paired) |
> |--:|--:|--:|--:|---|
> | 6 | 0.4071 | 0.4300 | **+0.0229** | [+0.0163, +0.0286] |
> | 18 | 0.3318 | 0.3709 | **+0.0391** | [+0.0297, +0.0469] |
> | 36 | 0.2287 | 0.2891 | **+0.0604** | [+0.0500, +0.0704] |
>
> **M8 is direction-confirmed, magnitude-retired** — its 13× recovery was a broken control, not a
> bigger effect. **`hidden` alone does nothing (−0.005); `cell` alone does everything.**
>
> **It is a SWITCH, and the anchor SATURATES at w≈0.1** (EXP-03): a 10% pull per step already buys
> **83%** of the hard clamp. On the **correct** paired interval (`cell@0.5` vs `cell`, MDE **0.0045**)
> the difference **includes zero at h18 and h36**, and at **h6 the hard clamp wins outright**
> (−0.0061, excludes zero). **`freeze_recurrent='cell'` ships; there is nothing to tune.**
>
> ⚠️ Two process defects are recorded against how that verdict was first reached — an overridden
> registered branch (**C-305**) and an MDE from the wrong contrast (**C-306**). The conclusion survived
> re-measurement; the reasoning that produced it did not.

## The methodological result may outlast the scientific one

The effect (+0.039) is **below** the programme's inherited MDE of 0.0541, so by the SS sweep's rule it
would read **UNDERPOWERED**. But that MDE prices a *between-seed* design. These arms are **naturally
paired** — same weights, same origins, same support, one flag differs.

**Paired MDE: 0.0086. Unpaired: 0.0541. 6.3× tighter on identical data**, and the same effect flips
from UNDERPOWERED to **EFFECT** under the same `3 × MDE` rule. That is **#281 answered by
demonstration**, and it cost no GPU time.

⚠️ It does **not** retroactively rescue the SS sweep: those arms differ by **seed as well as
treatment**, so they are not pairable. The lesson is for design, not for re-reading old results.

## ⚠️ Pre-registration status — read before citing anything here

| experiment | status |
|---|---|
| **EXP-01/02** (8-arm run + paired CI) | **NOT pre-registered.** Launched from a driver with no plan written; the decision rule was chosen after seeing data. See `05_analysis_plan.md` §8. |
| **EXP-03** (decay dial) | **Pre-registered.** Decision table committed at `da3156d` 10:58:07, first arm scored 19:29:15 — verifiable in git. |
| falsifiers | **EXP-03 only.** `freeze_table.py` first enters git *in the same commit as the score CSVs*, four hours after the overnight run — so for **EXP-01/02 the falsifiers are retrospective**. Corrected 2026-08-22 after review; the first version of this table claimed "all experiments". |

**What EXP-01/02 actually stand on:** a paired interval 4.5× its own MDE, and a control arm that
reproduces its published value — **not** pre-registration, and **not** pre-committed falsifiers.

⚠️ **The "switch, not a dial" verdict is under correction.** It quoted an MDE computed for a different
contrast (`cell` vs `none`); the right interval for `cell@0.5` vs `cell` is being measured. See
`05_analysis_plan.md` §5a.

## Documents

| file | what |
|---|---|
| `05_analysis_plan.md` | design, decision rules, falsifiers — **with per-section provenance stamps** |
| `DIAL_PAUSED.md` | **the pre-registration of record for EXP-03** — §5's decision table is quoted verbatim from it. Written as a live pause note; EXP-03 has since completed. |
| `07_experiment_log.md` | **EXP-01** the 8-arm run · **EXP-02** the paired interval · **EXP-03** the decay dial |
| `results/SUMMARY.md` | auto-assembled by `tools/freeze_table.py`; falsifiers printed **above** the table |
| `results/paired_ci.json` | the intervals |

## Tools

`tools/freeze_table.py` (6 tests) keys on the **filename**, which carries the seed — the CSV's `model`
column is just the arm name and does not. The first finisher keyed on `model`, collapsed both seeds
onto four rows, and rendered an empty comparison section. Same defect class as the `aggregate_seeds`
label collision fixed the day before, which is why this one is a tested module.

`scripts/ap_block_bootstrap.ap_diff_origin_block_ci` (9 tests) is the paired estimator. Its
load-bearing test bootstraps two arms **independently** and asserts the paired interval is tighter.

`tools/reemit_for_paired_ci.sh` exists because a paired bootstrap needs **both cubes at once** and the
driver is score-then-delete — and because both arms write the *same* pred-dir name, so `--keep-cubes`
alone would trip the driver's own contamination guard.

## ⚠️ What this does NOT say

* **Not a fix for the collapse.** +0.039 at h18 against an oracle ceiling near 0.50.
* **A rollout-time intervention**, not a trained one.
* **A frozen state is a static risk map by construction** — exactly **C-293**'s degenerate-forecast
  worry. That the effect *grows* with horizon is consistent with both "carries real information" and
  "static beats a degrading gate"; this design does not separate them.
* **One vehicle, AP only**, CI from one seed. The `crps_all` ARTIFACT verdict is untouched.
