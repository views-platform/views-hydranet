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
