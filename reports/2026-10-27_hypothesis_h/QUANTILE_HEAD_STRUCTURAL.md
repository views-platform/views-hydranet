# The quantile head cannot represent the zero atom — and that is why it lost 11–35×

**Date** 2026-10-27 · **Closes epic #458** · **Registered as C-387 (Tier 2)** · No compute: this is
arithmetic plus four source reads.

**Provenance.** Simon's challenge: *"a quantile head gives you a bag of marginal quantiles, not a
predictive distribution … it's a bit weird you again and again go for something that looks like it's
cosplaying as probabilistic forecasting."* ⭐ Measuring it against the source made the objection
**structural** rather than stylistic, and produced a stronger argument than the one stated.

## The arithmetic

`midpoint_levels(k) = (i + 0.5) / k` — `quantile_head.py:34-41`. A **uniform** grid.
Measured zero fraction, score window, 842,634 cell-months: **0.996552**.

| K | levels above the zero atom |
|---|---|
| 9 | **0 of 9** |
| 50 — **what the 2026-07 run used** | **0 of 50** |
| 99 | **0 of 99** |
| **146** | 1 — the first K where *any* level clears the atom |
| 501 | needed to place one level above τ = 0.999 |
| 5001 | needed for τ = 0.9999 |

⇒ ⛔ At K=50 **every level sits below the atom, so the pinball optimum for all fifty is exactly 0.**

## The contradiction

`monotone_quantiles` — `quantile_head.py:60-61`:

```python
gaps = F.softplus(raw[..., 1:])
return torch.cat([base, base + torch.cumsum(gaps, dim=-1)], dim=-1)  # strictly increasing
```

`softplus > 0` always ⇒ the quantiles are **strictly increasing by construction**. They cannot be
tied at zero.

⇒ ⛔⛔ **The architecture forbids the objective's optimum.** The head is clamped between a loss whose
answer is fifty tied zeros and a structure guaranteeing fifty distinct ascending values. The excess
is forced upward, and the file names the consequence itself at `:67`: *"the cumulated softplus starts
at ~0.69/gap → top quantile pegs the count cap."*

0.69 × 49 = **33.8** in log1p space. `QUANTILE_EMIT_CEIL = 13.0` ⇒ **442,412** deaths in one
cell-month, **3.9×** the observed sb maximum of **113,000**.

⇒ ⭐⭐ **M1f/M1g's diagnosis is retired.** The *"top-quantile over-shoot on ~100 cells (the
mean-swamping fat tail)"* is **the forced monotone fan**, not a tail-modelling failure — and a bound
cannot fix a fan the architecture is required to produce. Hypothesis H was doomed twice: once on the
arithmetic of W-86 (36% measured, 96.2% required), once here.

## The second defect: a precondition the loss cannot enforce

`hurdle_quantiles_to_samples` — `quantile_head.py:110-118` — states:

> *"The magnitude head is trained on **POSITIVE cells only** (no zero mass to force a degenerate
> cliff); the gate `prob` = P(y>0) supplies the occurrence."*

Its composition `Q_mag((u − (1−p))/p)` is correct **only** if `F_mag` is conditional on `y>0`. But:

| | |
|---|---|
| `QuantileLoss.forward` `:152-157` | ends in a bare `pin.mean()` — **no mask, no weight** |
| `training_engine.py:924` | `target_j = y_reg[:, j, :, :]` — the **full** target |
| the only mechanism that could supply it | `body_supervision='pos_cells'`, **swept and rejected**: *"pos_cells lifts size-ratio ×11–60 but blows crps-all (ns 0.08→24.6)"* |

⇒ ⛔ **The emission function assumes a conditional body, and the configuration that would supply one
was already rejected on measurement.** Under the house default (`'all'`) the zero mass is represented
**twice** — once by the gate's `1−p` atom, again inside `F_mag` — and the mixture is not a
distribution.

## ⭐⭐⭐ One cause, two heads

| head | why it cannot hold the law |
|---|---|
| **NB** (ADR-073 §0) | `θ·log(1+μ/θ)` monotone increasing in `θ` ⇒ **no interior optimum** on an empty cell at any `μ` ⇒ `θ → 3.6e-06` is the objective's *correct* answer |
| **quantile** (here) | pinball's optimum on an empty cell is **all levels tied at zero** ⇒ forbidden by strict monotonicity ⇒ the fan inflates |

⇒ **Neither parameterisation can place an atom, and 99.66% of the mass is an atom.** The gate is
meant to supply it; all-cell supervision makes the body fight it.

## ⚠️ What is NOT claimed — three corrections to the challenge as stated

1. **The quantiles *are* a coherent univariate CDF.** Cumulative-softplus makes monotonicity
   structural, so *"no guarantee they describe a single coherent distribution"* is **wrong for this
   implementation**. The defect is that a coherent CDF still cannot contain an atom — which is worse.
2. **Scoring needs no interpolation.** The pinball-sum *is* a CRPS quadrature by the
   Gneiting–Raftery identity, as the loss's own docstring says. The unfaithful interpolation is at
   the **sampling** step — `:138`'s `q0 + w*(q1−q0)` straddling the atom. Right in effect, wrong step.
3. **Spatial dependence is unrepresented — equally by the NB head and by the family-free lookup.**
   It is a property of per-cell emission in general, so it does not discriminate between the
   options. It remains a real gap, and it belongs in the lookup's own scope.

## ⛔ The pattern, which is the part worth keeping

**Richer emission machinery was proposed while the record kept saying the defect is upstream of
emission.**

- M67: *"the real defect is upstream of every emission rule."*
- ADR-073 §0: *"No emission rule, no sampler fix and no rescale repairs it."*
- Seven interventions failed; four on one lever.
- The epic's own framing — *"this is the first epic that changes the head"* — treated a new head as
  though it were not still the emission side. **It is.**

⇒ ⭐ And the one thing that **wins**, by 15%, **touches no head at all.** The family-free lookup is a
recalibration of the model already trained. That is where the programme goes.

⚠️ It is still emit-time, so it is not exempt from the pattern above — it earns its place on a
measured win on the primary metric at zero training cost, not on being upstream.
