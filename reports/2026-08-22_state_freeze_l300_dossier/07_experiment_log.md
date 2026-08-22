# Experiment log — state-freeze at L=300

---

### EXP-01 · does freezing recurrent state still help at 300 lessons? · 2026-08-22 00:49 → 02:23 · **YES — and it is the CELL state**

**Why.** **M8** claims freezing recurrent state recovers gate AP@h18 `0.0070 → 0.0912`. It was measured
on `truncated_smoke` — **40 lessons, one seed** — which **M28** now classifies as having no skill at any
horizon, and the pre-registered `violet_visitor` confirmation was never run. That makes M8 the primary
suspect in **#280**. The number that forced the re-run: **M8's *recovered* value (0.0912) is 3.6× BELOW
what an L=300 model scores free-running with no intervention at all** (0.3298, M34).

**Design.** 2 seeds × 4 arms (`none`/`hidden`/`cell`/`all`), emit-only on existing ε=0 L=300 weights.
93 min, **8/8 arms, no failures**.

#### Falsifiers — recorded before the numbers were read

* **h1 identical across all arms** (0.4774 seed 43, 0.4779 seed 42). There is no feedback at step 1, so
  freezing *cannot* move h1. If it had, the arms would not be what they claim.
* **Every `none` arm reproduces its published free-running value exactly** — seed 43 **0.3318**, seed 42
  **0.3298**, matching M34. The vehicle is what we think it is.

Both are enforced in `tools/freeze_table.py` and printed **above** the table, so a failure cannot be
read past.

#### Result

| arm | seed 43 h18 | seed 42 h18 | mean | vs `none` | h36 mean |
|---|--:|--:|--:|--:|--:|
| `none` | 0.3318 | 0.3298 | 0.3308 | — | 0.2248 |
| `hidden` | 0.3316 | 0.3209 | 0.3262 | **−0.005** | 0.2380 |
| `cell` | 0.3709 | 0.3622 | 0.3666 | **+0.036** | 0.2860 |
| `all` | 0.3743 | 0.3614 | 0.3678 | **+0.037** | 0.2834 |

#### Verdict for #280: M8's DIRECTION replicates; its MAGNITUDE framing does not transfer

At 40 lessons M8 read `0.0070 → 0.0912` — a **13×** recovery that looked spectacular only because the
control was broken. On a converged vehicle the same intervention buys **~+13% relative**. Both are
"state-freezing helps"; only one of them is a number you can quote.

**So M8 is not an artifact of the floor-limited vehicle — but its headline was.** #280 should record the
row as *direction-confirmed, magnitude-retired*, not as retired outright.

#### It sharpens C-292 rather than contradicting it

C-292 holds that `cell` and `hidden` are architecturally inseparable, because `hs = o ⊙ tanh(hl)` means
freezing cell constrains hidden by construction — so M8's "cell carries 89% of the effect" was
predetermined. These arms separate them **in the other direction**, which that argument does not
predict: **`hidden` alone does nothing (−0.005), `cell` alone does everything (+0.036), and `all` adds
nothing over `cell` (+0.001).** The effect lives entirely in the long-term state. C-292's objection was
to the *decomposition claim*; this is an *ablation ordering*, and it stands.

---

### EXP-02 · the paired interval · 2026-08-22 06:29 → 06:52 · **the effect is real, and #281 is answered by demonstration**

EXP-01 left the effect unresolvable by the programme's own bar: **+0.036 against an MDE of 0.0541**
inherited from the SS sweep's between-seed design. Under that rule this would read UNDERPOWERED.

But that MDE prices the wrong comparison. These two arms are **naturally paired** — same weights, same
origins, same support, one flag differs — which is exactly the construction **#281** argues for.
`ap_diff_origin_block_ci` draws origins **once per replicate and scores both arms on the same resampled
cell set**, carrying the correlation instead of assuming it away.

Cost: a paired bootstrap needs **both cubes at once** and the driver is score-then-delete, so two arms
were re-emitted (10 min). Both also write the *same* pred-dir name (keyed on the artifact), so
`--keep-cubes` alone would trip the driver's own contamination guard — each cube is moved aside between
runs.

| h | `cell` | `none` | diff | 90% CI | paired MDE | excludes 0 |
|--:|--:|--:|--:|---|--:|:--:|
| 6 | 0.4300 | 0.4071 | **+0.0229** | [+0.0163, +0.0286] | 0.0061 | ✅ |
| 18 | 0.3709 | 0.3318 | **+0.0391** | [+0.0297, +0.0469] | 0.0086 | ✅ |
| 36 | 0.2891 | 0.2287 | **+0.0604** | [+0.0500, +0.0704] | 0.0102 | ✅ |

`n_origins = 13`, `n_support = 170430`, 400 replicates, seed 0, seed-43 vehicle.

**The paired MDE at h18 is 0.0086 against the unpaired 0.0541 — 6.3× tighter, on identical data.** The
effect is **4.5× its own MDE**, so by the SS sweep's `|Δ| ≥ 3 × MDE` rule this is an **EFFECT**, where
the unpaired reading was UNDERPOWERED.

**That is #281's recommendation validated on real arms**, not in the abstract: the fix that costs no GPU
time was the right one to try first. It does **not** retroactively rescue the SS sweep — those arms
differ by *seed as well as* treatment, so they are not paired and cannot be re-analysed this way. The
lesson is for **design**, not for re-reading old results.

#### Scope

One seed for the interval (EXP-01's two seeds agree in sign and closely in size, but the CI is seed 43
only). One vehicle, `sb`, calibration partition, **AP only** — no CRPS claim, and the `crps_all` ARTIFACT
verdict is untouched. `cell` vs `all` is within noise of each other and this design does not separate
them.

#### What it does NOT say

Freezing state is a **rollout-time intervention**, not a fix for the collapse. It buys +0.039 at h18
against an oracle ceiling of ~0.50 — the gap remains large. And a frozen state is a *static* risk map by
construction, which is exactly the degenerate-forecast worry C-293 raised; that the effect *grows* with
horizon (+0.023 → +0.039 → +0.060) is consistent with both "carries real information" and "static beats
a degrading gate", and this design does not separate those.

---

### EXP-03 · is the freeze a switch or a dial? · 2026-08-22 19:23 → 19:55 · **a SWITCH — and it saturates at w≈0.1**

**Why.** EXP-01/02 measured only the two endpoints: `weight=0` (`none`) and `weight=1` (a hard clamp).
The freeze recovers 23% of the oracle gap and leaves **77% open**, so a hard clamp was the most extreme
setting of a dial nobody had turned. Four interior points, seed 43, ~6 min each, no training.

Registered before the run, in `DIAL_PAUSED.md`: **> 0.3709 ⇒ a dial** with an interior optimum;
**between 0.3318 and 0.3709 ⇒ monotone, a switch**; **< 0.3318 ⇒** the mechanism is not what we think.

#### Falsifier

**h1 is identical across all six arms** (0.47737082595880015). There is no feedback at step 1, so the
anchor cannot reach it — any movement would have meant the arms were not what they claim.

#### Result

| w | h1 | h6 | h18 | h36 | Δ@h18 | % of full-clamp gain |
|--:|--:|--:|--:|--:|--:|--:|
| 0.00 | 0.4774 | 0.4071 | 0.3318 | 0.2287 | — | 0% |
| **0.10** | 0.4774 | 0.4172 | 0.3643 | 0.2834 | **+0.0325** | **83%** |
| 0.25 | 0.4774 | 0.4200 | 0.3678 | 0.2852 | +0.0360 | 92% |
| 0.50 | 0.4774 | 0.4238 | 0.3716 | 0.2886 | +0.0398 | 102% |
| 0.75 | 0.4774 | 0.4247 | 0.3731 | 0.2916 | +0.0413 | 106% |
| 1.00 | 0.4774 | 0.4300 | 0.3709 | 0.2891 | +0.0391 | 100% |

**Everything at w ≥ 0.5 spans 0.0022 at h18, against a paired MDE of 0.0086 — indistinguishable.**
There is no interior optimum this design can resolve. **`freeze_recurrent='cell'` is the answer, and the
dial has nothing to give.**

#### The shape was none of the three registered outcomes

It is not monotone, not peaked, and not inverted: it is **sharply saturating**. A **10% pull per step
already buys 83%** of what a hard clamp buys.

**That changes the mechanism story.** The cell state does not catastrophically diverge under
free-running — it **drifts**, and a light restoring force is nearly as good as nailing it down. The
apparent rise at 0.75 (+0.0413, 106%) is inside the same 0.0022 band and must not be read as an optimum.

#### What it says about learning the parameter

Headroom above a crude constant is **~17% of a +0.039 effect ≈ 0.007 AP — below the paired MDE of
0.0086**. **A learned scalar has nothing measurable to win here.** Learning would only pay as a
*state-* or *horizon-dependent* function, which is a different and much larger piece of work (#290,
#291). Recorded so those issues inherit a measured prior rather than an assumption: the correction
needed is **small**, which makes it an easier target and a smaller prize at the same time.

#### Scope

**One seed.** EXP-01 showed the endpoint effect replicates across seeds 42/43, but this curve does not —
and it does not need to, because its conclusion is a *negative* (no resolvable interior optimum) and the
shipping decision is unchanged. `sb`, calibration partition, AP only.

#### Cost, and a correction carried from the paused run

Four arms in 32 min on an idle machine — **20 s/origin, ~3× FASTER than the overnight hard-freeze arms
managed (55 s/origin)**. The interior-weight path is not slow. The 240 s/origin measured on 2026-08-22
morning was **entirely** machine contention (a 10-core `library_rebuild` from another session; load
average 20.3), and the `torch.lerp` change made in response was a fix to a non-problem — kept because it
is strictly less work for identical maths, but it repaired nothing.

**Method rule this earns:** on a shared machine, `uptime` and `ps --sort=-pcpu` **before** any timing
claim. "Measure over an interval" is not enough when the interval's conditions are unknown.

---

