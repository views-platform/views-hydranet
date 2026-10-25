# Pre-registration — the sharpness ceiling

**2026-10-25.** ⛔ Written **before** anything is computed. No pod, no money, no training, no forward
pass. ⛔ **This measurement gates `reports/2026-10-21_structure_penalty_arm_prereg.md`** and was
written because that arm's whole premise — *"the body carries only 0.391 of its due spatial structure
at one cell"* — silently assumes the achievable value is **1.0**. Nobody has checked that.

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

---

## 3. ⛔ Decision rules, committed before the number exists

| `C_half(1)` (interleaved, the flattered bound) | consequence for the structure-penalty arm |
|---|---|
| **< 0.45** | ⛔ **The arm's premise is substantially VOID.** The body at 0.391 is at or near the ceiling, there is no structure to recover, and the arm must **not** be run on this justification. ⭐ The cheapest negative in the project's history |
| **0.45 – 0.70** | ⚠️ **Partial headroom.** The arm may run, but its upside is capped at `C(1)`, **not 1.0** — and `P-4` ("d = 1 rises by >= 0.10 absolute") must be restated as a fraction of the ceiling before it is read |
| **> 0.70** | ✅ Real headroom. The arm is justified as written |

⛔ **These are not advisory.** They are the reason the measurement is being made and they are fixed now
so that the arm cannot be rescued by reinterpreting its own gate.

---

## 4. ⛔ VOID conditions

| ID | VOID if | why |
|---|---|---|
| **W1** | `C(d)` is flat or **decreasing** in `d` (Q-2 fails) | an uncorrelated nugget MUST matter most at the shortest lag. A violation is an estimator bug, not a finding |
| **W2** | any `C(d) < 0` or `> 1` | the cross-variogram is not bounded by construction when the two fields are not co-located replicates; out of range means the pairing is wrong |
| **W3** | the fresh pull's cell set is not the **64,818**-cell land grid | a 13,110-cell support is `africa_me_legacy` and a different experiment (V4 of the control prereg) |
| **W4** | the fresh pull and the 2026-10-12 cache differ by more than vintage revision can explain | ⛔ the standing rule is **never validate on cached data — pull FRESH**. The pull is fresh; the cache is kept only as a cross-check and a disagreement is reported, not absorbed |

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

## 8. Vocabulary

`reports/GLOSSARY.md` has no locked term for the reproducible fraction of a variogram. ⛔ The glossary
is edited rather than a synonym invented — **after** this runs, so the term enters with its measured
value rather than as a placeholder.
