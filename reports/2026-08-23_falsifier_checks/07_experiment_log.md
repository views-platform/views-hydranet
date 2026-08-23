# Experiment log — the three zero-GPU falsifier checks

Criteria in `05_analysis_plan.md`, committed **alone and first** (`6ec3c3c`, 2026-08-22 22:14:04).
That document labels itself a **decision memo, not a blind test** — exploration surfaced the values
before the thresholds were written, and #290/#291's expected outcomes were stated up front. Read it
before citing any verdict here.

---

### CHECK A · #290 Professor Forcing · **CLOSE — damage is front-loaded on all four seeds**

**Falsifiers first.** h1 **bit-identical** between free-running and oracle on every seed (there is no
feedback at step 1, so they cannot differ); `N` and `n_event` matched at all 7 horizons, so the
subtraction spans one support. **Both PASS.**

`gap(h) = oracle_AP(h) − free_AP(h)`, L=300, `sb`, n=4 seeds:

| seed | h1 | h6 | h12 | h18 | h24 | h30 | h36 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 42 | 0.0000 | 0.0847 | 0.1192 | 0.1676 | 0.1923 | 0.2285 | 0.2459 |
| 43 | 0.0000 | 0.0861 | 0.1323 | 0.1696 | 0.1988 | 0.2209 | 0.2499 |
| 44 | 0.0000 | 0.1073 | 0.1596 | 0.1852 | 0.2063 | 0.2395 | 0.2599 |
| 45 | 0.0000 | 0.0898 | 0.1330 | 0.1580 | 0.1766 | 0.2087 | 0.2269 |

| seed | rate_early (h1→h6) | rate_late (h6→h36) | **front_loading** | gap6/gap36 |
|---|--:|--:|--:|--:|
| 42 | 0.01693 | 0.00537 | **3.15×** | 34.4% |
| 43 | 0.01723 | 0.00546 | **3.16×** | 34.5% |
| 44 | 0.02146 | 0.00509 | **4.22×** | 41.3% |
| 45 | 0.01795 | 0.00457 | **3.93×** | 39.6% |

**Criterion: ≥ 2.0 ⇒ CLOSE. Worst seed 3.15×. VERDICT: CLOSE #290.**

The first five free-running steps lose AP at **3–4× the rate** of the remaining thirty, carrying
**34–41% of the entire h36 gap in 14% of the horizon**. Lamb et al. report Professor Forcing gives no
benefit below ~100 steps and that its benefit **scales with the importance of long-term dependencies**.
Our damage is a short-horizon phenomenon at a 36-step readout: **the method is aimed at the wrong part
of the curve.**

**Scope, stated not resolved.** Score CSVs exist only on {1,6,12,18,24,30,36}; **horizons 2–5 do not
exist anywhere in the repo**. This is a *block-level* answer and cannot resolve step 2 vs step 5.
Resolving it needs a re-emit, and the block-level margin (3.15× worst against a 2.0 bar, unanimous
across four seeds) does not depend on it.

---

### CHECK B · #291 Horizon Forcing · **CLOSE — the premise fails here, on both series**

**Falsifier first.** M27 must reproduce from the CSVs, not the ledger prose: T=0 300→600 **+0.02127**
(prose "+0.0213"), σ_T0 **0.0077** (prose 0.0077), Δretention **+0.00143** (prose "+0.0014"). **PASS.**

σ measured from the six L=160 ε=0 arms: **σ_T0 = 0.0077**, **σ_retention = 0.0458**.

| series | ΔT=0 (300→600) | ΔRetention | verdict |
|---|--:|--:|---|
| single-seed (as M26/M27 present it) | **+0.02127 = 2.76σ** | **+0.00143 = 0.03σ** | DECOUPLED |
| **multi-seed L=300, n=4** | **+0.02246 = 2.91σ** | **+0.00836 = 0.18σ** | DECOUPLED |

**Criterion: ΔT0 > 2σ AND ΔRetention < 1σ ⇒ decoupled ⇒ CLOSE. Both series agree. VERDICT: CLOSE #291.**

Zhuang et al. state the premise with its own scope condition — *"**in chaotic systems** controlling
long-term error necessitates controlling short-term error."* Conflict counts are not chaotic in the
Lyapunov sense, and here one-step skill moves **~2.9σ while robustness moves 0.18σ**. The method's own
justification does not apply to this vehicle.

**Ledger correction found by the mandatory multi-seed run.** M26/M27 quote L=300 from **one seed**; four
ε=0 seeds now exist (retention mean **0.6833 ± 0.0320**). The 300→600 retention step is therefore
**+0.0084**, not the **+0.0014** the ledger states. *(An exploration agent reported the mean as 0.6683
and the step as +0.023; recomputing from the CSVs gives 0.6833 and +0.0084. The agent's arithmetic was
wrong and the tool's is checked — this is why the check recomputes rather than quoting.)* Both figures
sit far below σ=0.0458, so **M26's plateau claim survives**; only its number needs correcting.

---

### CHECK C · #288 BPTT-SA · **the naive fix is a SILENT NO-OP — re-scope, do not close**

Not a threshold; a fact about code, and the one check that was **genuinely blind**. Executed rather than
reasoned about, because the answer depends on what the installed torch registers as a backward.

| operation | graph-connected | `grad_fn` | Σ\|grad\| |
|---|---|---|--:|
| `torch.poisson` | **True** | `PoissonBackward0` | **0.0** |
| `torch.bernoulli` | **True** | `BernoulliBackward0` | **0.0** |
| `_standard_gamma` *(one line above)* | True | `WhereBackward0` | **2.9179** |

**The gradient is severed** at `views_hydranet/distributions/nb_core.py:165` by `torch.poisson`, and on
gated compositions again at `composition.py:56` by `torch.bernoulli`. **And it fails silently:** both ops
are graph-connected with a registered backward that returns exactly zero. Removing the two `.detach()`
calls at `training_engine.py:363,365` would **not crash, not warn**, would pay the full memory cost of a
retained 36-step feedback graph, and would return a numerically identical model.

**That is the worst failure mode an experiment can have — it looks like it worked.**

**Two live arms, both cheap:**

1. **`ss_feedback='mean'` is differentiable end-to-end today** (`_family_target_log1p_mean` is pure
   analytic algebra on activated params). Removing the detach *there* is a genuine BPTT-SA arm and the
   clean falsifier for "was the detach load-bearing?"
2. **A reparameterised path exists by accident.** `_standard_gamma` is a hand-rolled Marsaglia-Tsang
   sampler — written for RNG determinism, but that *is* the reparameterisation trick. Its gradient is
   live (2.9179). So `lam` on `nb_core.py:164` is a differentiable stochastic rate, and feeding back
   `log1p(lam)` rather than `log1p(poisson(lam))` is **stochastic and differentiable**, two lines above
   the sever point.

**Mandatory for any such arm:** a graph assertion (`assert fb.requires_grad and fb.grad.abs().sum() > 0`).
`tests/train/test_feedback_parity.py` pins **values only** — its inputs do not require grad — so it would
pass a zero-gradient no-op unchanged.

**Carried risk.** Under autograd the Marsaglia-Tsang loop (up to 64 iterations, full-grid intermediates
per target per timestep, data-dependent count) is the binding memory constraint on a 4070 — *not* BPTT
depth, which is already paid because `h` is never detached.

---

## Cost

Three checks, **zero GPU**, ~20 minutes. They close two of five investigative issues and re-scope a
third, which was the information-theoretic case for running them before committing ~36 GPU-hours to
#287.

---

### CHECK D · #294 Generalized Teacher Forcing · 2026-08-22 23:08 → 23:45 · **correspondence is SUPERFICIAL**

Threshold registered in **GitHub issue #294 (2026-08-22)**, estimator in **AMENDMENT 1** (`e631f74`,
23:08:39) — both before any measurement. **Unlike Checks A and B, this one was genuinely blind.**

**The question.** Hess et al. derive `α = 1 − 1/σ_max` for the same interpolation our
`blend_recurrent_state` performs. **M41** measured our empirical optimum at **w ≈ 0.1**. If `w ≡ α`,
that implies `σ_max ≈ 1.11`. So the paper *predicts* our knob — a rare chance for an independently
derived theory to land on a hand-swept result.

#### Stage 1 — the analytic bound: INCONCLUSIVE, and structurally so

Recurrent convolution **operator** norms (power iteration on the conv as an operator at the true field
size, data-free; 0.48% drift between 50 and 200 iterations — converged):

| cell | Whi | Whf | Whc | Who |
|---|--:|--:|--:|--:|
| 1 | 1.95 | **3.61** | 1.92 | 2.40 |
| 2 | 1.97 | 2.91 | 1.87 | 1.76 |
| 3 | 1.81 | 2.61 | 2.13 | 1.69 |
| 4 | 2.25 | 2.34 | 2.13 | 1.73 |

Bound at M=1: **5.32**. AMENDMENT 1 registered that *"an upper bound above 1 licenses nothing"*, so this
is INCONCLUSIVE by the rule — **and the estimator turns out to be structurally incapable of ever
answering.** Its dominant term is `¼·M·‖Whf‖ + ¼·‖Whi‖ + ‖Whc‖`, which floors near **2.5 even as M → 0**.
The bound assumes every gate saturates simultaneously; with recurrent operator norms of 1.7–3.6 that
assumption is far too generous to be informative here. **Recorded rather than dropped: an estimator
registered in good faith that could not do the job.**

*(This stage was first run at a 180×720 field. The captured states show the true field is **180×180** —
africa, not global. A convolution's operator norm depends on field size, so it was re-run at 180×180;
the verdict is unchanged.)*

#### Stage 2 — the true Jacobian. **First attempt measured the WRONG PHASE.**

⚠️ **The first run of this stage is retracted.** The capture hook took the **first six** forward calls,
but the rollout loop is `for t in range(origin + time_steps)` (`hydranet_inference.py:913`) with
`origin = seq_len - 1`: it digests **335 steps of real history first**, *then* runs 36 autoregressive
steps. Calls 0–5 are the **teacher-forced warm-up**. Found independently by `/code-review medium` and by
reading the loop while preparing the ritual.

**Two claims fall with it:**

1. **σ_max = 1.60 characterised the warm-up, not the rollout.** The question was about the free-running
   regime.
2. **The headline observation was backwards.** The log previously read *"we watched the drift happen …
   the first direct observation of it"*, from `max|h|` rising 0.000 → 2.867. **That was the state filling
   from zero initialisation.** It is not drift, and the real direction is the opposite (below).

**Locating the phase boundary without hard-coding `origin`** (it is `seq_len - 1`, a data property, not
a config constant): the recurrent state is re-zeroed at the start of every posterior sample, so
`max|h| == 0` marks a sample boundary and the distance between boundaries is `origin + time_steps`.
Measured period = **371**, so with `time_steps = 36`, **`origin = 335`** and the autoregressive phase is
calls **335–370**.

#### Stage 2 (corrected) — 8 states spanning the true free-running phase

`--skip 335 --stride 5`, so the captures span the rollout rather than clustering at its start — σ_max is
a **supremum over the trajectory**, and sampling only the first steps would understate it.

| call | rollout step | max\|h\| | σ | drift |
|--:|--:|--:|--:|--:|
| 335 | 1 | 65.622 | 3.4120 | 0.00% |
| 340 | 6 | 56.621 | **7.7628** | 0.00% |
| 345 | 11 | 19.737 | 4.6363 | 0.00% |
| 350 | 16 | 8.186 | 1.8734 | 0.00% |
| 355 | 21 | 5.125 | 1.4820 | 0.00% |
| 360 | 26 | 3.017 | 1.4795 | 0.00% |
| 365 | 31 | 1.832 | 1.4769 | 0.00% |
| 370 | 36 | 1.598 | 1.4744 | 0.00% |

**σ_max = 7.7628**, every state converged to 0.00% drift.

#### The state COLLAPSES during free-running — the opposite of what was first reported

```
max|h|:  65.6 → 56.6 → 19.7 → 8.2 → 5.1 → 3.0 → 1.8 → 1.6
```

**A ~40× decay over 36 steps.** The recurrent state is *large* at the anchor — everything digested from
real observations — and then **drains away** as the model feeds on itself. This is a direct observation
of **M16**'s *"the gate keeps its shape and loses its nerve"*, and it is what the earlier claim got
exactly backwards.

It also reframes **M38/M39/M41**: the cell anchor helps not by *restraining* a growing state but by
**refilling a draining one**. A 10% pull per step (M41's saturation point) is enough because the state
is losing information, not accumulating error.

#### σ is strongly state-dependent, and a single number flattens that

**3.4–7.8 early in the rollout, when the state is large; ~1.47 once it has collapsed.** Quoting one
"σ_max" hides a 5× swing that tracks `max|h|` almost monotonically. Any future use of σ_max here should
say *which part of the rollout* it refers to.

#### Scope

One seed, one vehicle, **8 states spanning one origin's autoregressive phase** — σ_max is a supremum
over *all* states and this samples 8 of 36, at one origin of 13, for one posterior sample. Widening it
is cheap (`--n-states`, `--stride`) and, given the 5× swing across the rollout, **a denser sample would
likely raise σ_max further** — it would not move it toward [1.05, 1.20].

---

## ⚠️ POSTMORTEM — what these four checks are actually worth (C-307)

Added 2026-08-23 after the user pointed out a pattern that **predates this session**:

> *"We have dropped multiple things on the floor — and I keep telling you so — by doing quick smart
> tests to see if real implementation makes sense, then dropping real implementation because the test
> told us to. Then me later insisting that we try for real, and then it turns out that it in fact
> works. This happens so much."*

**Every check in this document is a proxy.** None of them tried the method being screened. A proxy's NO
is evidence against the real thing only in proportion to how tightly the two are coupled — and this
document originally recorded the verdicts and the falsifiers *on the checks*, but never the
**false-negative mode of the checks themselves**.

**The worst instance is CHECK D, and it is self-inflicted.** Issue #294's body states, in its own words,
*"GTF re-anchors every step; we anchor once … **these are not the same operator**"* — and the check then
compared their α against our w as though they were. Three independent reasons that comparison was
invalid are now recorded on the issue: σ_max was measured on a teacher-forced model when α governs a
GTF-trained one; the paper's adaptive variant does not use a fixed α at all; and the two weights
parameterise different operators. **"GTF's theory does not predict our result" is withdrawn.** The
number stands; the inference does not.

### What survives, and what does not

| | status |
|---|---|
| front-loading 3.15–4.22×, n=4, unanimous (A) | ✅ measurement stands |
| decoupling 2.76–2.91σ vs 0.03–0.18σ, both series (B) | ✅ measurement stands |
| `torch.poisson` severs the gradient, silently (C) | ✅ **fact about code — the one check that is not a proxy** |
| ~~σ_max = 1.60~~ (D) | ❌ **retracted — measured on the teacher-forced warm-up, not the rollout** |
| **σ_max = 7.7628** on 8 free-running states, converged (D, corrected) | ✅ measurement stands |
| ~~"we watched the drift happen", max\|h\| 0.000 → 2.867~~ | ❌ **retracted, and backwards** — that was the warm-up filling from zero |
| **the state COLLAPSES ~40× during free-running** (65.6 → 1.6) | ✅ the corrected observation, and it matches M16 |
| "Professor Forcing is aimed at the wrong part of the curve" | ⚠️ weakened — measured on the untreated model |
| "Horizon Forcing's premise fails ⇒ it would not help" | ⚠️ weakened — a wrong justification is not a wrong method |
| "the GTF correspondence is superficial" | ❌ **withdrawn** |

**CHECK C is the exception and it is worth naming why.** It is not a proxy: it establishes what the code
*does*, and un-detaching the feedback really would be a silent no-op. That check is safe to act on. The
other three screen a *method* through a *measurement of our current model*, which is a much weaker
instrument than the write-ups implied.

**CHECK D adds a second failure mode, worse than being a proxy: it measured the wrong part of the
process.** The probe was pointed at the teacher-forced warm-up while the question was about the
free-running rollout, and **nothing in the numbers looked wrong** — σ_max = 1.60 is a perfectly
plausible value, and a rising `max|h|` read as a satisfying confirmation of drift. It was caught by
reading the rollout loop, not by any guard. **A plausible number from the wrong measurement is harder to
catch than a wrong number**, because every downstream check passes: the iteration converged, the
falsifier was satisfied, and the write-up was internally consistent. Registered as **C-308**.

### Standing rule adopted (C-307)

An investigative issue closed on a proxy must carry **both**:

1. **the false-negative mode** — the specific way this screen could say no while the real method says
   yes; and
2. **a reopen trigger** — a concrete condition that would make revisiting correct.

**"CLOSED" means *screened out, here is what brings it back* — never *settled*.** Applied retroactively
to #290, #291 and #294 on 2026-08-23.

### The honest expected-value statement

These four checks cost ~35 minutes and no GPU, against ~36 GPU-hours for #287. That trade was worth
making. But the correct summary is **"three methods deprioritised, one code fact established"** — not
*"two ideas closed"*. The deprioritisation is a **prior update**, not a verdict, and the reopen triggers
above are deliberately low bars because the base rate of this programme reopening such decisions is
high.

