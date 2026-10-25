# Pre-registration — the sharpness ceiling

**2026-10-25.** ⛔ Written **before** anything is computed. No pod, no money, no training, no forward
pass. ⛔ **This measurement gates `reports/2026-10-21_structure_penalty_arm_prereg.md`** and was
written because that arm's whole premise — *"the body carries only 0.391 of its due spatial structure
at one cell"* — silently assumes the achievable value is **1.0**. Nobody has checked that.

---

## ⛔ OUTCOME AND CORRECTIONS, 2026-10-25 — added after the run; **nothing below is deleted**

The measurement ran (**78.0 s**, 4.76 GiB peak RSS) and **Q-1 landed in band at `C_half(1) = 0.498`**.
A `/falsify` audit of the result then found **three errors in this document**, two of them hard. Every
original sentence is retained and annotated in place with the date and the reason it is now believed
wrong, following the retraction convention of `reports/2026-09-30_evaluation_learning_report.md`.

| # | where | what is wrong | severity |
|---|---|---|---|
| **X-1** | **§3** | ⛔ The decision rule compares **two different estimators**. `C(1)` is a raw variogram ratio (**amplitude**); B-1's 0.391 is variance-normalised (**shape**). Applying B-1's own estimator to the two halves gives **0.983–1.009 at every lag**, so the ceiling 0.391 is measured against is **~1.0**, not 0.498. §3 as written would have **falsely throttled** the arm | ⛔ hard |
| **X-2** | **§1**, half-to-full | ⛔ `C_full` is Spearman–Brown, whose premise — additive noise with variance ∝ 1/n — is a statement about **counts**. It is applied to `log(rate + eps)`. Own evidence: 0.665 on the logged field vs 0.262 on raw counts, **3.3× apart**. ⇒ **`C_full = 0.665` is withdrawn** | ⛔ hard |
| **X-3** | **§4** | ⚠️ W1/W2 are drafted **globally** over a sweep whose degenerate arms **Q-4 explicitly predicts**. They fired on `raw`/`contiguous` slots while the primary slot was clean, and the driver printed `VOID` then reported a verdict anyway | ⚠️ soft |

⭐ **Net effect on the arm: its premise SURVIVES, on better evidence than it had.** The ceiling is ~1.0,
so the body's 0.391 is a genuine ~61% shortfall. ⛔ The §5 directional caveat is untouched and still
applies to every number here, the ~1.0 included.

Audit trail: `reports/2026-10-25_sharpness_ceiling/falsify_ceiling_stubs.py` — 5 red-by-design tests,
one per finding. A test turning green means that defect is fixed.

---

## 0. The question, in one sentence

> **How much of the truth field's short-range spatial roughness is reproducible signal at all, and
> how much is noise that no model should ever reproduce?**

Call the answer `C(d)` — the reproducible fraction at lag `d`. ⭐ **`C(1)` is the ceiling the body's
0.391 should be compared against.** Comparing it to 1.0 is comparing it to a number that may be
unreachable in principle.

### Why now, and why it is first

The structure-penalty arm costs one pod (~$3) and a retrain. This costs minutes of laptop CPU. ⛔ **If
`C(1) ≈ 0.4`, the arm has nothing to gain and must not be run.** A free measurement that can
invalidate a paid one runs first. That is the whole argument.

### ⭐ Simon's framing, which is the reason this is posed as a ceiling and not as a target

> *"If you think of a time series instead of the spatial issue — we will likely never be able to have
> the same kind of jagged time series line as reality, **and nor should we**, because whether or not a
> certain event happened on one side of the 31st in a month or the other is probably aleatoric
> uncertainty, at least given the methods and especially the data we have now."*

The spatial statement is the same statement: whether a battle fell in cell `(i,j)` or its neighbour is,
at our resolution, partly a coin toss. ⇒ **A model that reproduced the truth's full roughness would be
fitting that coin toss.** So the quantity with meaning is not "how rough is the truth" but "how much of
that roughness repeats".

### ⛔ And the architectural fact that makes this a question about `mu` and nothing else

Simon asked whether we should expect individual draws to be sharp with smoothness emerging only in the
ensemble mean — the Gaussian-process reading, where a draw is as rough as the kernel and `mu` is
legitimately smooth.

⛔ **Our model cannot do that, and this was checked in the code rather than assumed.** The scored cube
draws `_standard_gamma` then `torch.poisson` **elementwise** (`distributions/nb_core.py:163-165`) and
`soft_gate` composes with `torch.bernoulli` **elementwise** (`distributions/composition.py:56`). Every
cell's draw is independent given `mu`. A spatially correlated draw exists in the codebase
(`utils/correlated_bernoulli.py`) but `hydranet_inference.py:1135-1136` states it is applied in the
**feedback path only, as a diagnostic** — *"the scored cube keeps independent sampling via
`to_cube_samples`"*.

⇒ **We have no kernel.** A single draw from our model IS rough, but it is rough as *white noise on a
smooth field*, while the truth's roughness is **clustered**. Independent per-cell noise cannot
impersonate spatial structure, and averaging 16 draws removes it. ⭐ **Therefore all spatial structure
must live in `mu`, and measuring it on `mu` — which is what B-1 does — is correct rather than a
category error.** The GP reading is right in general and unavailable to us in particular.

---

## 1. The measurement

Two independent halves of the in-sample record are two noisy observations of the same underlying
field. Write `A` and `B` for their per-cell base rates and `f` for the latent field:

```
gamma_A(d)     = gamma_f(d) + nugget_A
gamma_B(d)     = gamma_f(d) + nugget_B
gamma_cross(d) = gamma_f(d)            <- the two nuggets are independent, so they do not enter
```

⇒ **`C_half(d) = gamma_cross(d) / gamma_B(d) = gamma_f / (gamma_f + nugget)`** — the fraction of the
observed variogram at lag `d` that is reproducible. This is the spatial form of a split-half
reliability, and it has **no fitted parameter**, which is the property that let B-1 survive two
failed analyses around it (`d12b`, `d12c`).

### The three slots (C-351)

| slot | value |
|---|---|
| population | all **64,818** land cells, no admission filter, no event-rate selection. Zeros are data. In-sample months **121–456** (336 months, the window B-1's base rate uses), split in two |
| transform | ⛔ **swept, because `eps` is a free parameter and B-1's entire curve depends on it:** `log(rate + 1e-6)` (B-1's own value, for comparability), `log(rate + 1e-2)`, and the **raw rate** (no log). `1e-6` makes `log(0+eps) = -13.8` and 94% of cells are zero over 336 months, so the untransformed version is not optional |
| estimator | classical method-of-moments `gamma_hat(d)` and the matching cross-variogram `0.5*E[(A(s+d)-A(s))*(B(s+d)-B(s))]`, over all land pairs at separation `d`, computed by array shift. Distance in **grid cells**, never km. `d in {1,2,4,8,16,32}`, **E–W** and **isotropic** |

### Two split schemes, reported side by side — ⛔ not one

| scheme | definition | why it is a bound and in which direction |
|---|---|---|
| **interleaved** | odd months vs even months | ⇒ an **upper** bound. Adjacent months are temporally correlated, so the halves are more alike than independent replicates and `C` is **flattered** |
| **contiguous** | months 121–288 vs 289–456 | ⇒ a **lower** bound. 28 years of real non-stationarity (conflict moves) counts against reproducibility, so `C` is **penalised** |

⇒ **The pair brackets the answer.** ⛔ Neither is reported as the value, per
`feedback_sweep_before_reporting_a_scalar`.

### Half-to-full correction, derived here so it cannot be chosen later

Each half has 168 months, so if the noise is independent across months its variance is **2x** the
full-336 estimate's. With `r = gamma_f / nugget_full`:

```
C_half = r / (r + 2)   =>   r = 2*C_half / (1 - C_half)   =>   C_full = r / (r + 1)
```

⇒ `C_half = 0.50` implies `C_full = 0.667`. **Both are reported.** ⚠️ `C_full` rests on "noise
independent across months", which §5 names as **partly false**.

⛔ **RETRACTED 2026-10-25 (X-2) — kept because the derivation is CORRECT and the error is in its
APPLICATION.**

The algebra above is sound and reduces exactly to Spearman–Brown (`C_full = 2C/(1+C)`; verified). It is
also correct for a per-month **mean** as well as a sum, because what scales with `n` is the
signal-to-noise ratio either way. ⛔ **What is wrong is the scale it is applied on.** "Noise independent
across months with variance ∝ 1/n" is a statement about **counts**. The primary slot transforms by
`log(rate + 1e-6)`, and after that transform the noise is neither additive nor 1/n-scaling: a cell that
is zero in both halves is pinned at `log(eps) = -13.8` regardless of `n`, so its error variance is ~0,
not ∝ 1/n.

Measured — and it is **this document's own Q-4** that refutes it:

| scale | `C(1)` | → `C_full` |
|---|---|---|
| `log`, `eps=1e-6` | 0.4982 | **0.6650** |
| raw counts | 0.1509 | 0.2622 |

A 3.3× disagreement; both cannot describe one latent noise process. ⇒ **`C_full = 0.665` is withdrawn
and must not be quoted.** `C_half(1) = 0.498` stands — it needs no scaling argument, being a ratio
measured at a single `n`.

---

## 2. Pre-registered predictions

| # | quantity | prediction | confidence |
|---|---|---|---|
| **Q-1** ⭐ **the one that matters** | `C_half(1)`, interleaved, `eps=1e-6`, E–W | **0.50**, inside **[0.35, 0.65]** | **50%** |
| **Q-2** ⚪ *manipulation check* | `C(d)` across `d = 1 -> 32` | **monotonically increasing** — the nugget is uncorrelated, so its share is largest at the shortest lag | **90%** |
| **Q-3** | interleaved vs contiguous at `d = 1` | interleaved **higher** (it is the flattered bound) | 80% |
| **Q-4** | `C(1)` at `eps=1e-6` vs the **raw rate** | they differ by **> 0.10 absolute** — the zero atom at `log(1e-6)` dominates the logged version | 65% |
| **Q-5** | the number of cells positive in exactly ONE half | **> 25%** of the cells with a positive 336-month base rate | 60% |

⚠️ **Q-1's reasoning, stated so it can be judged wrong:** ~94% of land cells are zero across all 336
months, and of the ~3,790 with a positive base rate many carry only one or two events. A cell positive
in one half and zero in the other jumps between `log(eps) = -13.8` and a finite value — an enormous,
spatially uncorrelated contribution concentrated at every lag. **I expect that atom, not real
structure, to be a large share of the truth's variogram.** If so, 0.391 is much closer to the ceiling
than the structure-penalty arm assumes.

### Outcome, 2026-10-25

| # | predicted | measured | verdict |
|---|---|---|---|
| **Q-1** | 0.50, band [0.35, 0.65] | **0.498** | ✅ **in band** — ⛔ but see X-1: the quantity is right and §3 then compared it to a number measured a different way |
| **Q-2** | monotonically increasing | 18 of 36 slots violate; **the primary slot does not** | ⚪ **mixed** — see X-3 |
| **Q-3** | interleaved higher | sb 0.498 vs 0.246 · ns 0.484 vs 0.141 · os 0.405 vs 0.205 | ✅ **3/3** |
| **Q-4** | differ by > 0.10 | **0.347** (logged 0.498, raw 0.151) | ✅ — and this result is what falsifies X-2 |
| **Q-5** | > 25% | **37.9%** | ✅ |

⭐ **Q-1's reasoning above was wrong in an interesting way, and it is kept for that reason.** The zero
atom IS a large share of the truth's variogram — Q-4 confirms it at 0.347 — but it is a share of the
**amplitude**, and B-1 divides amplitude out. So the atom is real, and it does **not** lower the ceiling
that 0.391 is measured against. The inference "if so, 0.391 is much closer to the ceiling than the arm
assumes" does not follow, and that non-sequitur is X-1.

---

## 3. ⛔ Decision rules, committed before the number exists

| `C_half(1)` (interleaved, the flattered bound) | consequence for the structure-penalty arm |
|---|---|
| **< 0.45** | ⛔ **The arm's premise is substantially VOID.** The body at 0.391 is at or near the ceiling, there is no structure to recover, and the arm must **not** be run on this justification. ⭐ The cheapest negative in the project's history |
| **0.45 – 0.70** | ⚠️ **Partial headroom.** The arm may run, but its upside is capped at `C(1)`, **not 1.0** — and `P-4` ("d = 1 rises by >= 0.10 absolute") must be restated as a fraction of the ceiling before it is read |
| **> 0.70** | ✅ Real headroom. The arm is justified as written |

⛔ **These are not advisory.** They are the reason the measurement is being made and they are fixed now
so that the arm cannot be rescued by reinterpreting its own gate.

⛔⛔ **CORRECTION, 2026-10-25 (X-1) — THIS RULE IS A CATEGORY ERROR. The table above is kept; it must
not be applied.**

The rule caps the arm's upside at `C(1)` and compares it against B-1's **0.391**. Those two numbers come
from **different estimators**:

| | estimator | measures |
|---|---|---|
| **B-1** (gave 0.391) | `gamma_model(d)/gamma_truth(d)` **divided by the ratio of the two fields' variances** (the same-shape null) | **shape** |
| **`C(d)`** (gave 0.498) | `gamma_cross(d)/gamma_B(d)`, **not** variance-normalised | **amplitude** |

So the ceiling was computed again using **B-1's own estimator**, substituting half-A for "model" and
half-B for "truth" (sb, interleaved, `log(rate+1e-6)`, E–W, 64,818 land cells, months 121–456):

| d | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| `C = gX/gB` | 0.498 | 0.584 | 0.673 | 0.734 | 0.764 | 0.795 |
| **B-1 form** | **1.009** | **0.996** | **1.009** | **0.986** | **0.983** | **0.987** |

(`var(half A) = 6.5188`, `var(half B) = 6.3594`, ratio `1.0251`.)

⇒ **Under the estimator that produced 0.391, the ceiling is ~1.0 at every lag, within 1.7%.** The
variance normalisation cancels exactly the noise `C(1)` measures: two noisy halves of the truth have the
**same variogram shape** even though each is individually noisy.

⇒ **The row that fired — 0.45–0.70, "upside capped at `C(1)`" — does not apply.** `0.391` is read
against **~1.0**, which is what `reports/2026-10-21_structure_penalty_arm_prereg.md` assumed. That
assumption is now **measured rather than assumed**, and `P-4` needs no restatement.

⚠️ **What `C(1) = 0.498` does still mean**, because it is a real finding about the data: slightly under
half the **amplitude** of the truth's one-cell variogram is reproducible across independent halves. That
bears on noise in the truth field. It does not bound B-1.

---

## 4. ⛔ VOID conditions

| ID | VOID if | why |
|---|---|---|
| **W1** | `C(d)` is flat or **decreasing** in `d` (Q-2 fails) | an uncorrelated nugget MUST matter most at the shortest lag. A violation is an estimator bug, not a finding |
| **W2** | any `C(d) < 0` or `> 1` | the cross-variogram is not bounded by construction when the two fields are not co-located replicates; out of range means the pairing is wrong |
| **W3** | the fresh pull's cell set is not the **64,818**-cell land grid | a 13,110-cell support is `africa_me_legacy` and a different experiment (V4 of the control prereg) |
| **W4** | the fresh pull and the 2026-10-12 cache differ by more than vintage revision can explain | ⛔ the standing rule is **never validate on cached data — pull FRESH**. The pull is fresh; the cache is kept only as a cross-check and a disagreement is reported, not absorbed |

⚠️ **CORRECTION, 2026-10-25 (X-3) — W1 and W2 are kept, and their SCOPE is wrong.**

Both are written globally — W2 says "**any** `C(d) < 0` or `> 1`" — across a sweep that this very
document **predicts will contain degenerate arms**: Q-4 forecasts the raw transform diverging from the
logged one by > 0.10, and it diverged by 0.347.

What fired: `ged_os_best / contiguous / raw` gave `C(1) = -0.064` (E–W) and `-0.305` (isotropic), and 18
of 36 slots were non-monotone — **almost all of them `raw` or `contiguous`**. The **primary slot was
clean**: 0.498 → 0.584 → 0.673 → 0.734 → 0.764 → 0.795, strictly increasing, all within `[0, 1]`.

⛔ **And the driver resolved the contradiction silently**, printing `⛔ VOID` and then printing a §3
verdict regardless — a gate that cannot mean what it says.

⇒ **It should have read: W1/W2 void the measurement if the PRIMARY slot is non-monotone or out of range;
a swept diagnostic slot going degenerate is a finding, not a void.** Restated here rather than edited in
place, so the original drafting error stays visible. The general lesson is recorded in
`reports/2026-09-30_evaluation_learning_report.md`.

---

## 5. ⛔⛔ THE LIMITATION THAT DOMINATES THIS MEASUREMENT — the input is not ground truth

**Simon, 2026-10-25, recorded because it is load-bearing and exists nowhere else in the repo:**

> *"The data we get in is not ground truth. It is riddled with measurement errors, presenting as sharp
> ground truth. It is also aggregated temporally and spatially. So before we can ever really be honest
> with our model, we need to have input uncertainty put around the input data going in. And that's
> just a weakness that we don't have right now, because it is a lie. It's not a lie to tell our model
> that in reality conflict is sharp — it happens at a point in time and space where people, real
> people, are killed, and it matters exactly when in time and space it happens, even if there's too
> much aleatoric uncertainty around it to actually capture it. But even our event data is uncertain.
> The providers, UCDP and ACLED, are not particularly good or too keen at putting real uncertainty
> estimates around it, but we ourselves are making progress. ... The sharpness here is an artifact, and
> it would be more honest to have a more smeared and smooth risk surface also as input data, which
> complicates everything."*

### ⭐ What that does to this measurement, and it is DIRECTIONAL — which is what makes it usable

The cross-variogram cancels noise that is **independent between the two halves**. Event-level sampling
noise and month-of-occurrence binning are of that kind, and they cancel.

⛔ **Systematic provider error does not.** An event coded to a provincial centroid, a capital, a
rounded coordinate or a district seat lands in the **same cell in both halves**. It therefore survives
the cross-variogram and is counted as **real, reproducible structure**.

⇒ **`C(d)` is an UPPER BOUND on genuinely predictable structure, not an estimate of it.** And the two
outcomes are not symmetric:

| outcome | what it licenses |
|---|---|
| **`C(1)` comes out LOW** | ⭐ **conclusive.** The true ceiling is lower still, so the structure-penalty arm's premise fails by more than the number shows |
| **`C(1)` comes out HIGH** | ⚠️ **inconclusive about how much is provider habit.** It would license running the arm, but it would NOT establish that the recoverable structure is conflict rather than geocoding convention |

⛔ **This asymmetry is the single most important sentence in this plan** and must be carried into any
readout of it.

### What is NOT in scope here, noted and left alone

- **Input uncertainty on the covariates and the targets.** The honest object is a smeared risk surface
  with uncertainty attached, and we do not have one. ⛔ This plan does not build it, does not estimate
  it, and must not be read as a substitute for it.
- **A geocoding-artifact probe.** Excess mass at rounded coordinates or at administrative centroids
  would be measurable on the same pull and would separate "provider habit" from "conflict structure".
  ⛔ Not done here — it is a second measurement with its own plan.

---

## 6. ⛔ What this does NOT establish

- **It says nothing about the under-dispersion of counts**, which is the stated priority. `C(d)` is a
  property of the **mean field's spatial roughness**. The predictive distribution's width given `mu`
  is `theta` and tail shape, and no second-order spatial statistic touches it.
- **It does not say the structure penalty would work** if the ceiling is high. It removes one way for
  the arm to be pointless; it is not evidence for the arm.
- **It is a property of the TRUTH, not of any model.** No model artifact is read. The control now
  training is not involved and this measurement does not wait for it.
- **It cannot separate aleatoric irreducibility from provider artifact.** §5.

---

## 7. Cost

⚠️ To be replaced with the measured figure, per the standing rule that a cost is measured before it is
quoted. Reference point: D12's full sweep was **391 s** on this laptop with a 4-thread cap. This needs
~6 sweeps (2 schemes x {A, B, cross}) x 3 transforms. ⛔ No pod, no money, no GPU.

✅ **MEASURED, 2026-10-25: 78.0 s wall, 4.76 GiB peak RSS**, laptop CPU, `OMP_NUM_THREADS=4`, $0. The
estimate above is kept for provenance. It was pessimistic because `gamma_A`, `gamma_B` and
`gamma_cross` share **one** pass over the offsets rather than needing six sweeps.

## 8. Vocabulary

`reports/GLOSSARY.md` has no locked term for the reproducible fraction of a variogram. ⛔ The glossary
is edited rather than a synonym invented — **after** this runs, so the term enters with its measured
value rather than as a placeholder.

✅ **DONE 2026-10-25.** Two terms were needed, not one, and needing two **is** finding X-1:

| term | definition |
|---|---|
| **reproducible amplitude fraction** `C(d)` | `gamma_cross(d)/gamma_B(d)` between two independent half-samples of the truth. **Not** variance-normalised. Measured **0.498** at `d=1`. It bounds how much of the truth variogram's **size** is signal rather than sampling noise |
| **reproducible shape fraction** | the same split-half comparison under **B-1's** estimator — the variogram ratio **divided by the ratio of the two fields' variances**. Measured **0.983–1.009** across `d = 1…32`. ⭐ This is the one that bounds **B-1**, and therefore the one the 0.391 is read against |

⛔ **The two are not interchangeable and conflating them is X-1.** Any sentence of the form "the body
carries `X` of its due structure against a ceiling of `Y`" must state which estimator produced each.
