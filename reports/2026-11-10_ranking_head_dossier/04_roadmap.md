# 04 — Roadmap: from the loss to the one number

**Serves:** the *"how do we get there"* half — the gated sequence from an unwritten loss to the
finish line of `00 §⭐⭐`, cheapest-first, with a decision point at the end of every phase.

Written 2026-11-10. ⛔ **The governing constraint is the handover's:** *"Build the cheapest B that
can exist, and run it."* Every phase below is ordered so the cheapest thing that could change the
answer happens first.

---

## 1 The phases

### Phase 1 — the loss, and only the loss  ·  CPU, hours, nothing trained

**Build** the level-indexed listwise loss of `02 §3` with the dyadic transform of `02 §2`, as one
entry behind a default-off flag.

⛔ **The numerics are tested before anything trains.** These are the tests, and each needs a half
that fails against the defect — `03 §3.5` says why:

| # | the property | the defect it must refuse |
|---|---|---|
1 | `r = ceil(log2(1+y))` maps `y = 681 → r = 10`, `w = 1023` | raw counts: `w` is `inf` in float32 |
2 | `|R|` is the **a-priori** level set, fixed across folds | `|R|` read from observed levels (`C-423`) |
3 | an all-one-level list gives loss **exactly 0** | a non-zero loss on an unrankable list |
4 | the lowest level is the dropped final step | the zero block contributing gradient |
5 | an empty level contributes nothing and does not change `|R|` | level 9 being silently skipped |
6 | the criterion returns a **0-dim** tensor | a non-scalar that broadcasts silently (`03 §3.3`) |
7 | every weight is finite for every `y` in `[0, y_max]` | ⛔ `C-420`, the Tier-1 case |

**Gate to phase 2:** all seven green, plus the full suite and lint. ⛔ **If the loss cannot be made
to satisfy 3 and 4 simultaneously, that is a finding about Eq. 2 at our sparsity, and the reserve
option (`01 §3`, Blondel's differentiable sort) becomes the next thing tried** — not a verdict on B.

### Phase 2 — the seam  ·  CPU, hours

Answer `03 §3.1`: either the score fits the family shape at `n_params = 1`, or the dispatcher is
widened deliberately. Then **prove inertness**: with the flag off, the baseline is byte-identical
against `tests/distributions/test_parity_anchor.py`'s golden — asserted, not claimed in a docstring.

**Gate to phase 3:** the parity anchor passes with the new code present and the flag off.

### Phase 3 — ⭐⭐ the cheap readout that bears on the finish line  ·  CPU/GPU, one short train

**The question:** does a patch-trained ordering score sort the **global** grid?

`02 §4.3` names this as the one real cost of the patch-local list, and it is answerable long before
convergence. Train the ranking body briefly — far short of the 300-lesson plateau — emit the score
on the global grid, and measure **`ρ_event`** against the body's `+0.3127 ± 0.0713`.

⭐ **Why this is the right cheap probe:** it is in the units of the finish line. It cannot settle the
programme, but it is the only measurement available before convergence that could move the estimate
of whether convergence is worth buying.

**Read-out order:** `ρ_event` global · then `ρ_event` within-patch · then the gap between them. The
gap **is** the quantity `02 §4.3` worries about, and it is a number rather than an argument.

**Gate to phase 4:** a score field exists on the global grid and `ρ_event` is computable on it.
⚠️ **Not** a go/no-go on the value of `ρ_event` — phase 3 is an instrument check and an estimate,
and `00 §6`'s kill rule applies to it as to everything else here.

### Phase 4 — train to the plateau  ·  GPU, remote

⛔ **Remote, not local.** The local route is **closed on measurement**: both smoke arms were
OOM-killed on **host RAM** at ~21 GiB on a 31 GiB box, dying in the backtest pass after origin 1's
draw. This is not re-litigated.

Train the ranking body to the 300-lesson plateau at the vehicle's settings, one variable changed
from the control: the body's loss. **`windows_per_lesson: 3`, `window_dim: 32`, `batch_size: 1`** —
the training path of `02 §4.1`, unchanged.

⚠️ **The watchdog from `01 §1`:** Jagerman 2017's failure signature for mishandled target-side ties
is **overfitting** — validation decay while training performance still improves. Log both curves
and read them, because that is the one failure mode the literature told us to expect.

**Gate to phase 5:** the arm converged, its manifest records what it is, and the validation curve
was read.

### Phase 5 — the emission  ·  CPU, minutes, nothing retrained

Only now compose `gate × ranking body × empirical draw` per `02 §1`, and only now does the
`tie@0` prediction of `02 §5` become checkable: a continuous score should drive it to **0.000**,
where A's count-mean index put it at 0.635 at a loose threshold.

⚠️ `body_mean_dump_dir` writes a **mean**; a scale-free score is not a mean. The dump seam needs
either a draws extension or a score-aware variant — named in `03 §5` as an open box.

**Gate to the verdict:** `05_analysis_plan`'s pre-registered criterion, which does not yet exist.

---

## 2 Dependency graph

```
phase 1  loss + numerics  ──►  phase 2  seam + inertness  ──►  phase 3  ⭐ global ordering probe
                                                                   │
                                                                   ▼
                                                        phase 4  train to plateau (remote GPU)
                                                                   │
                                                                   ▼
                                                        phase 5  rank-coupled emission
                                                                   │
                                                                   ▼
                                               ⛔ the verdict — BLOCKED on 05 (see below)
```

**Off the critical path, in parallel, owned elsewhere:** `00 §5` item 1 (Simon — the primary
quantity) · item 2 (`baseline` — its seed dispersion) · item 3 (`library` — the Mohler 2020 audit).
⇒ Phases 1–4 do **not** wait on any of them. **Phase 5's verdict does.**

---

## 3 Decision points

| after | the question | what makes it answerable |
|---|---|---|
phase 1 | does Eq. 2 survive our sparsity numerically? | the seven tests. If 3 and 4 cannot both hold, the reserve option is next (`01 §3`) |
phase 2 | is the baseline provably untouched? | the parity anchor, with the flag off |
⭐ phase 3 | does a patch-trained score sort the global grid? | `ρ_event` global, and the patch↔global gap |
phase 4 | did it converge without the tie-overfitting signature? | the two curves |
phase 5 | did `tie@0` go to zero as `02 §5` predicts? | the `tie@0` column, recomputed on the trained score |
⛔ **the verdict** | is the ranking body better at ordering than the magnitude body? | **the pre-registered criterion of `05`, which does not exist yet** |

---

## 4 Milestones — and what "done" means

⭐ **Done is the handover's definition, not a win:** the measurement ran and it was **diagnostic** —
some possible outcome would have sent the decision the other way, and the measurement had the
resolution to tell those outcomes apart. **Three** outcomes, all of them done:

| | |
|---|---|
the ranking body orders better than the magnitude body | done |
the magnitude body holds | done |
⭐ **not resolvable at this `n`** | **done.** Under-power is a result, not a loss for B |

⛔ The third is non-negotiable. Without it an under-powered run reads identically to a negative one,
and the idea ends for the wrong reason — which is exactly what happened on 2026-11-08.

---

## 5 Cost, stated up front

| phase | compute | cost |
|---|---|---|
1 | CPU | hours of writing; seconds to run |
2 | CPU | hours |
3 | one short train + CPU scoring | ⭐ the cheapest thing that bears on the finish line |
4 | remote GPU, 300 lessons | the only expensive phase |
5 | CPU, minutes | nothing retrained |

⇒ **Phases 1–3 are all CPU-or-cheap and all precede the only expensive step.** That ordering is the
point: the expensive phase is bought with phase 3's number, not with an argument.
