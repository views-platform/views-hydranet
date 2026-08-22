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

#### Stage 2 — the true Jacobian at production-path states

Six consecutive states captured by a **forward hook on a live free-running rollout** with **no
diagnostic flag set** (`tools/capture_states.py`, the same manager seam `freeze_arm_entry.py` uses), so
these are production-path states. `h_next` is assembled purely from the LSTM block
(`HydraBNrecurrentUnet_06_LSTM4.py:555`), so `∂h_next/∂h` is exactly the recurrent map's Jacobian and
the U-Net decoder does not enter it. Power iteration on `JᵀJ`, matrix-free (one jvp + one vjp per step).

| state | σ | drift (last 10 iters) | max\|h\| |
|---|--:|--:|--:|
| 0 | 1.5139 | 0.00% | 0.000 |
| 1 | 1.4388 | 0.02% | 0.995 |
| 2 | 1.4823 | 0.00% | 1.833 |
| 3 | **1.6000** | 0.00% | 1.928 |
| 4 | 1.5552 | 0.00% | 2.517 |
| 5 | 1.5016 | 0.00% | 2.867 |

**σ_max = 1.6000** (sup over states).

#### The falsifier fired once, and was obeyed

A first run at **60 iterations** returned σ_max = 1.6000 with one state drifting **1.11%** against the
registered **≤1%** convergence bar — **VOID**. The number was already on screen and would have been
reported unchanged. It was not read; the run was repeated at **250 iterations**, where every state
settles to ≤0.02%. Same answer, now earned. *(This is the third time today a registered rule has stood
between a convenient number and the write-up. **C-305** is what happens when one does not.)*

#### Verdict: CORRESPONDENCE SUPERFICIAL

```
α = 1 − 1/1.6000 = 0.3750        M41 measured w ≈ 0.10
```

The registered "striking" band was **σ_max ∈ [1.05, 1.20]**, i.e. the range in which the derived α would
have matched our measured optimum. **1.60 is outside it, and the predicted α is ~3.75× our measured
value.** The two methods share a functional form; **the paper's theory does not predict our result.**

**What this does and does not establish.** It does *not* show GTF would fail here — it shows the
**specific reason to expect it to work is absent**. The excitement was that an independently derived α
landed on a hand-swept optimum; it does not. Any future GTF work must be justified on other grounds and
should note that **#294's own differences 1–4** (GTF re-anchors every step, we anchor once; training vs
inference; shPLRNN vs ConvLSTM; chaotic vs not) remain untested.

**σ_max ≥ 1 is confirmed**, so the formula is at least *defined* for us — the third registered branch
("σ_max < 1 closes the issue") did not fire. The question was fair; the answer is no.

#### The finding worth keeping

**We watched the drift happen.** `max|h|` across six consecutive free-running steps:

```
0.000 → 0.995 → 1.833 → 1.928 → 2.517 → 2.867
```

M38/M39 inferred cell-state drift from its *consequences*; this is the first direct observation of it.
It is a **steady creep, not an explosion** — consistent with M41's saturation at w≈0.1, since a gentle
drift is precisely what a gentle restoring force can correct.

#### Scope

One seed, one vehicle, **six consecutive states from one origin** — σ_max is a supremum over *all*
states and this is a sample of six. Widening it is cheap (the hook takes `--n-states`) but would not
move 1.60 into [1.05, 1.20]; the gap is 3.75×, not marginal.
