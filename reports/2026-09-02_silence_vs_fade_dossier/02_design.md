# 02 — Design: how to separate silence from fade without conditioning

**2026-09-02**

## 1. The two hypotheses, stated so they make different predictions

Let a cell's emitted forecast at horizon `h` be the composition of a **gate** `g` (the classifier's
`P(y>0)`) and a **body** `mu` (the family's count-space `E[Y|body]`).

* **C1 — SILENCE (the claim under test).** `g` collapses over the horizon; `mu` holds.
* **R1 — FADE / survivorship (the rival).** `mu` also falls. The apparent flatness in the existing
  readout is **selection**: the statistic conditions on cells that fired, and as the distribution
  shifts down only the upper tail keeps clearing the bar, so the conditional mean is propped up.
* **R2 — INSTRUMENT.** Neither is true of the model; the flat magnitude is an artifact of the
  feedback-stats recorder (the sentinel, the filtering, `n=2`). C-318 is proof this failure mode is
  live here, not hypothetical.

R1 is the one that matters, and it is defeated **by construction** if the statistic never conditions
on anything. That is the whole design.

## 2. The identity (the spine)

For each horizon `h`, over the `N` cells of the field:

```
        mean_cells( g·mu )   =   mean_cells( g )   ×   [ Σ g·mu / Σ g ]
        ─────────────────        ─────────────       ─────────────────
          EMITTED MASS            OCCURRENCE            MAGNITUDE
```

This is exact, not an approximation — it is the definition of a weighted mean. It splits the emitted
field into an occurrence factor and a magnitude factor **with no threshold, no cutoff, and no
"active cell" set**. Every cell contributes at every horizon. There is nothing for survivorship to
select on.

Reading the three curves over `h` answers the question directly:

| | OCCURRENCE | MAGNITUDE | verdict |
|---|---|---|---|
| C1 | collapses | flat | the model **falls silent** |
| R1 | collapses | also falls | the model **fades too** — claim dead |
| — | flat | — | the prior observation was an artifact (R2) |

## 3. Two independent instruments (both required)

### I1 — from the stored cubes (no new code)

`OCCURRENCE(h)` = mean over cells and posterior columns of the **gate cube** `by_*`. This is stored
directly and deterministically (`np.repeat` of `sigmoid(cls)`), so it is **exact and noiseless** — an
observation, not an estimate.

`EMITTED MASS(h)` = mean over cells and columns of `expm1(lr_*)`. Because the cube's composition is a
per-draw `Bernoulli(g)` mask on the body draw, each entry is `B·y` with `B ⊥ y`, so its column-mean
is an unbiased estimate of `g·mu`.

`MAGNITUDE(h)` = `MASS(h) / OCCURRENCE(h)` — a **ratio of sums over the whole field**, not a per-cell
division. This matters: per-cell, dividing by `g ~ 1e-4` is hopeless; aggregated over ~13k cells it
is well conditioned. See `03` §B.2.

### I2 — from the raw body params (small default-off dump)

`predict(..., return_params=True)` already returns the activated family params on the production
path (`hydranet_inference.py:1273`). `family.mean(params)` is `mu` **directly** — no division, no
sampling noise, no Bernoulli. Dumping that field per step is the second instrument.

I2 is not a luxury. It is needed for a reason I1 cannot cover, and for the identity check:

* **The subtle survivorship.** `MAGNITUDE` in §2 is **gate-weighted**. If the gate collapses
  *non-uniformly* — staying high only where `mu` is large — then the gate-weighted mean of `mu` can
  stay flat while the **unweighted** mean of `mu` falls. That is survivorship wearing a different
  hat, and only an unweighted `mean_cells(mu)` from I2 can see it. **Both must be reported.**
* **G1.** Two instruments computing the same quantity by different routes must agree, or neither is
  trusted (`05` F3).

## 4. Treatment and control

* **Treatment** — `feedback_transform="identity"`: free-running, byte-identical to `None`, asserted by
  `tests/test_feedback_transform_seam.py::test_F3_none_is_byte_identical_to_identity`. Recording is on.
* **Control** — `feedback_transform="use_real"`: the same model fed real observations. Under the
  control **neither** curve should collapse. If MAGNITUDE drifts here too, the readout is measuring
  something other than free-running degradation and the program halts (`05` F4).

The control is what makes this a comparison rather than a description, and it is free: the seed-42
`use_real` arm already exists in `reports/2026-08-31_sharpness_diagnostic_dossier/results/`.

## 5. Fixed-cohort arm (confirmatory)

The identity already defeats survivorship, so this is corroboration rather than the main test: take
the cell set ranked by `g` at `h=1`, freeze it, and track that **same set's** `mu` forward. A frozen
cohort cannot be re-selected, so if its `mu` holds while the whole-field `mu` holds, two different
selection regimes agree. If they disagree, selection is doing work and that is itself the finding.

## 6. What this design deliberately does not do

* It does **not** reuse `mean_magnitude_on_active` from the feedback-stats CSV. That statistic
  conditions on firing — it is the one R1 predicts will be misleading, and it is the one that already
  produced C-318. It is measured only to *explain* the earlier reading, never to support a claim.
* It does not retrain anything. Emit-only, on the four existing `fullzero_*` artifacts.
