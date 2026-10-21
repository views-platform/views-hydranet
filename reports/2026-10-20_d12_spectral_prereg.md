# D12 — pre-registration: at WHAT SPATIAL SCALE does the model's amplitude go missing?

**2026-10-20, round 6.** Written **before** anything is computed. Simon chose the instrument
("spectral power curve") after six adversarial rounds with the `opponent` seat and the ingestion of
the nowcasting and point-process literature. ⛔ **This verb writes the plan only. Nothing is run.**

Reads `2026-10-12_underfire_diagnosis/d4_result.npz` (verified on disk: 138.9 MB, `mu`/`theta`/`gate`
each `(4, 49, 64818)` float32, months 421–469, vehicle sha `4acbf4d3c2b457b0…`). **No forward pass,
no pod, no training, nothing fetched.** CPU only, thread-capped.

---

## 0. Why this, and the one thing it is NOT for

§9.3 measured over-smoothing with **lag-1** spatial autocorrelation: one number, one scale. §10.1's
own decision column now says the local half is **0.06** and the global half is **2.49×**. ⛔ What
neither measures is *at which spatial scale* the model's amplitude goes missing — and that is the
question `Ayzel2020_RainNet` and `Ravuri2021_PrecipitationNowcastingDGMR` both answer for their own
models, in units a reader outside this project can interpret:

- **C-3654** (RainNet): *"PSD shows significant loss at 16 km and below"* at a single 5-min step,
  propagating to 32 km under recursion.
- **C-3657** (DGMR): the U-Net baseline's *"effective resolution"* is **8 km at T+30** and **32 km at
  T+90** against a 1 km grid.

⚠️ **And `Cox1955` C-3610 says the scalar was the wrong form all along:** the dispersion index `I(x)`
across interval lengths is the diagnostic, because *"different types of non-randomness produce
distinct I(x) curve shapes"*. The curve, not the number, is the object. My own standing rule
(`feedback_sweep_before_reporting_a_scalar`) says the same thing; Cox wrote it in 1955 and §9.3 still
reported one number per head.

⛔ **What this is NOT for.** §10.11's struck clause — *"the latent field is right; the point process
on top is not"* — is withdrawn as **not identifiable**, and this instrument does **not** rescue it.
`Cox1955` C-3611 shows NB emission and latent variance are one object reparameterised, so no
second-order statistic can attribute blame between them. D12 measures **where in scale** the defect
sits, not **which component** owns it. Anyone reading a component attribution out of a D12 result is
repeating the error this prereg exists downstream of.

---

## 1. ⛔ TWO INSTRUMENTS WERE REJECTED, BOTH ON MEASUREMENTS MADE BEFORE WRITING THIS

Recording these so neither gets re-proposed, and so the reasons are checkable.

### 1a. A 2-D FFT power spectrum on the land grid — REJECTED, mask

Measured: **64,818 of 259,200 cells are land = 25.01% coverage.** A 2-D FFT needs a complete
rectangular field. Filling 75% of the domain with a constant injects broadband power at every
coastline, and the injected power is a **larger fraction of the total for the smoother field** — i.e.
it would bias the comparison *toward* finding more small-scale power in the model, which is the
direction of my own claim. ⛔ An instrument whose artifact points at the hypothesis is not usable
here regardless of how it comes out.

### 1b. A PSD on the largest clean all-land rectangle — REJECTED, it cannot resolve the gate

I computed the maximal axis-aligned all-land rectangle twice.

**Globally:** 80 × 128 = 10,240 cells, at **lat 27.2–66.8, lon 54.2–117.8** — Central Asia, Siberia
and western China. ⛔ It maximises *land* and minimises *conflict*: the comparison there would be
between two near-constant fields. The largest clean window is in the wrong place, which is worth
knowing on its own.

**Within Africa**, choosing the window on **geography alone** (bbox lat −35…38, lon −18…52 — a rule
fixed before any spectrum is computed, so the sub-domain is not selected on either field):
110 × 40 = 4,400 cells at lat −24.2–30.2, lon 14.2–33.8. **6.79%** of land cells.

The binding limit is the *smaller* dimension, 40 cells, so the largest resolvable scale is **~20
cells**. Against the correlation lengths implied by §9.3's own lag-1 figures (`ξ = −1/ln ρ`):

| field | ρ(lag 1) | ξ (cells) | needs ~3ξ | resolvable in a 40-cell window? |
|---|---|---|---|---|
| gate logit E–W | 0.9598 | **24.37** | 73.1 | ⛔ **NO** |
| gate logit N–S | 0.9574 | 22.97 | 68.9 | ⛔ **NO** |
| body `log mu` E–W | 0.8594 | 6.60 | 19.8 | ✅ marginal |
| body `log mu` N–S | 0.7636 | 3.71 | 11.1 | ✅ yes |
| truth `log` mean E–W | 0.5915 | 1.90 | 5.7 | ✅ yes |

⇒ **The FFT route fails for the gate on every available domain**, and is at the very edge of its
resolution for the body. ⚠️ **This also means we cannot state our defect in RainNet's or DGMR's
published units without a caveat** — their domains were complete rectangles and ours is a 25%-covered
sphere. That limitation is structural, not fixable by effort.

### 1c. The dispersion curve across block sizes — REJECTED, redundant and half-blind

Proposed in round 6 and withdrawn after two measurements. (i) Its trend is **confounded by, and in
fact largely identical to**, the correlation-length difference it was meant to be independent of — a
pure ξ difference of 12.8× (gate) manufactures a monotone ratio rise of 1.00 → 11.04 across k = 1…16
*with no emission defect at all* (reproduced independently from both seats). Since a wrong
correlation length **is** over-smoothing, the curve mostly re-measures §9.3. (ii) Saturation needs
k ≳ 3ξ, which on this grid gives **648 blocks** for the body and **36** for the gate — so it is
untestable for the gate for the same reason 1b is.

---

## 2. THE INSTRUMENT: the empirical variogram, binned by distance

The **structure function / semivariogram** `γ(d) = ½·E[(Z(s+d) − Z(s))²]`, estimated over all land
pairs at separation `d`, binned.

**Why it survives where 1a–1c fail.** It is the real-space dual of the power spectrum and answers the
same question — *at what scale is amplitude missing* — but it is computed **pair-by-pair on available
support**, so a 25% land mask is not an approximation, it is simply the set of pairs that exist. No
filling, no window, no rectangle, no block saturation. And it is the **direct generalisation of what
§9.3 already did**: lag-1 autocorrelation is `γ` at `d = 1`, so D12 is §9.3's own statistic swept over
its free parameter rather than a new quantity.

⚠️ **This inverts the emphasis of what I recommended and Simon chose.** He chose "spectral power by
length scale"; the FFT estimator of it is the part that fails here, not the question. D12 answers the
chosen question with a mask-free estimator and keeps the PSD as a **secondary corroboration for the
body only** (§1b's Africa window, at the edge of its resolution). ⛔ If the two disagree, the
variogram is primary and the PSD is reported as failing, because its limits were established above
before either ran.

### The three slots, per C-351

| slot | value |
|---|---|
| **population** | all **64,818 land cells**, no admission filter, no event-rate selection. ⛔ Zeros are data, not absence of data — this is the rule `d4_mu_on_the_grid.py:162`'s `keep = tm > 0` broke, producing the §9.2 artifact (2.2% → 22.4%). Reported in sample (421–456) and out of sample (457–469) separately. |
| **transform** | `log(mu + ε)`, `logit(gate)`, `log(truth mean + ε)` — the same three transforms §9.3 used, `ε = 1e-6`, so the numbers are comparable to it and to nothing else. |
| **estimator** | classical method-of-moments `γ̂(d)`, isotropic and then E–W / N–S separately (§9.3 found a 0.10 anisotropy on the body, so isotropic alone would hide it). Distance in **grid cells**, not km — the grid is not equal-area and converting would mix a projection into the statistic. |

### Fitted parameters, three not one

Fit `γ(d) = n + s·(1 − exp(−d/ξ))` per field: **nugget `n`**, **sill `s`**, **range `ξ`**. The
opponent's round-6 point stands — ratioing at matched `d` discards the shape information that does the
discriminating, so the comparison is **parameter by parameter**, not curve-ratio by curve-ratio.

---

## 3. PRE-COMMITTED PREDICTIONS, with numbers, before the run

Derived from D4's own measured quantities, all-land in-sample.

| # | quantity | prediction | basis |
|---|---|---|---|
| **P-1** | model `log mu` **range** vs truth | **longer, 2.9–3.5×** (ξ 6.60 vs 1.90 E–W) | §9.3's lag-1, converted |
| **P-2** | model `log mu` **sill** vs truth | **LOWER, ≈0.436×** (σ² 1.6417 vs 3.7640) | D4: sd 1.2813 vs 1.9401 |
| **P-3** | gate logit range vs truth | **longer, >10×** | ξ 24.37 vs 1.90 |
| **P-4** | truth **nugget** | **strictly positive**, and large enough to matter | the truth is counts; count noise is white ⇒ a nugget |
| **P-5** | model nugget | **≈0**, within fit error | `mu` and `gate` are analytic in the emitted parameters, no count noise |

⛔ **P-1 and P-2 together are the non-obvious prediction: the model's field is FLATTER *and*
SMOOTHER.** Not "smoother at the same amplitude". If that holds, "over-smoothing" is the wrong
single word for it — the field has less total variance *and* spreads what it has over longer
distances.

⚠️ **P-2 must NOT be read against the 2.49× concentration finding.** §10.9's `N_eff` was computed on
the per-cell **firing-probability** field over a **different population** with a **different
estimator**. Two different fields. ⛔ Combining them is the C-351 violation this dossier has now
committed seven times; the ratio of a sill to an `N_eff` is not a quantity.

---

## 4. ✅ ONE INTERNAL CONSISTENCY TEST THAT COSTS NOTHING

P-4's fitted nugget is **D6-C's estimator attenuation in variogram language**. D6-C measured, by
simulation through the truth's own estimator, that the estimator eats **0.154** (E–W) and **0.081**
(N–S) of lag-1 autocorrelation. A fitted nugget `n` against sill `s` implies an attenuation of
`n/(n+s)` at `d = 1`.

⇒ **Pre-registered agreement check: `n/(n+s)` for the truth should reproduce 0.154 / 0.081 to within
the variogram fit error.** Two instruments, two estimators, one quantity, computed independently. If
they disagree by more than the fit error, **one of them is wrong and D12's own sill comparison is
suspect** — which is a better outcome than either number standing alone.

---

## 5. ⛔ FALSIFIERS, committed now

| ID | fires if | consequence |
|---|---|---|
| **F1** | model `γ(d)` **exceeds** the truth's at any `d` where both are resolved | P-2's "uniformly flatter" reading dies; the defect is scale-dependent in amplitude, not just in range |
| **F2** | model range is **not** longer than the truth's | ⛔ §9.3's correlation-length reading dies, and with it the length-units restatement of the 0.06 |
| **F3** | the truth's `γ(d)` has **not plateaued** by the largest `d` with adequate pairs | the **sill** comparison is **VOID for lack of reach** — the same failure that killed 1b and 1c, and it must be declared rather than fitted through |
| **F4** | the nugget/sill agreement in §4 disagrees with D6-C by more than fit error | D12's sill comparison is **suspect**; report both and resolve before quoting either |
| **F5** | the fitted `ξ` for the gate exceeds the largest `d` with adequate pairs | the gate's **range** is unresolved; report as a bound, never as a value |

## 6. ⛔ VOID conditions

1. Any distance bin with **fewer than 1,000 pairs** is dropped, and if that drops the plateau region
   the run is VOID rather than reported on a truncated curve.
2. **Non-convergent** three-parameter fit, or a fitted nugget that is negative.
3. ⚠️ If the gate's fitted `ξ` lands within a factor of 2 of the largest resolvable `d`, the gate half
   is **VOID by §1b's own arithmetic** — committed here because the alternative is discovering it
   afterwards, which is exactly how **D5-A** was voided and what its guard was supposed to teach.
4. An anisotropy between E–W and N–S exceeding the §9.3 measurement by more than 2× suggests the
   binning is wrong, not the field.

## 7. Cost

⛔ **Measured, not estimated** — wall time and peak RSS printed by the script, per the standing cost
rule. `torch.set_num_threads` is irrelevant (numpy only), but the pair enumeration is `O(n²)` in land
cells at 64,818, which is **4.2 × 10⁹ pairs** — ⚠️ **this must be binned by cell-offset
multiplicity, not by enumerating pairs**, or it will not finish. The offset approach is `O(H·W)` per
lag and is the same trick that fixed a 120-second timeout while this prereg was being written.

**Nothing is run until Simon says so.** Open decisions remain his: the run itself, PR #420 and #419,
and whether the `library`'s self-excitation sub-folder gets extracted.

---

## 8. What D12 cannot do

- ⛔ It **cannot** locate the defect in a component (§0). `Cox1955` C-3611 forecloses that for every
  second-order statistic, not just this one.
- ⛔ It **cannot** say *why* the field is smooth. Four candidate mechanisms are live — BN recal on
  32×32 windows applied to 360×720, global `weight_decay` 0.1, recurrent state accumulation, and
  **conditional-mean averaging**, which the literature now favours and which `Mathieu2016` C-3634
  (enlarging the receptive field left sharpness at 24.7 → 24.7) and `Taylor2012`'s *"2-neighbourhood
  GMRF is sufficient on regular grids"* both support by demoting spatial reach. ⚠️ Our measurement is
  **teacher-forced**, so output feedback is excluded but **state** accumulation is not.
- ⛔ It **cannot** move the S1 verdict, and will not be read as bearing on it. S1 rests on §2/§3's
  `+0.006–0.012 AP` / `R² 0.10` bounds and C-152. Further: **AP is blind to spatial arrangement by
  construction** — it depends only on the ranking of scores against paired labels, which
  `scripts/ap_block_bootstrap.py:19` states in our own codebase (*"a global rank statistic over the
  pooled score vector"*), and which `Ravuri2021` C-3657 reports empirically for CSI. So D12 measures
  something the scoring framework cannot see, and that is a fact about the framework, not a licence to
  reopen it. ⛔ FAO-02 is LOCKED and changing it is Simon's decision alone.

---

# ⛔ OUTCOME — run 2026-10-21. The FIT is VOID; the CURVES answer the question and CORRECT the basis of this prereg

`d12_variogram.py` (committed `3468fe3` **before** it ran) and the post-hoc `d12b_curve_readout.py`.
Cost measured: **391.4 s / 2.68 GiB** at 4 threads for the sweep (22,612 half-plane offsets × 8
field-windows), **0.0 s / 0.03 GiB** for the readout. ⚠️ My pre-run projection was **14 min** against
**6.5 min** actual — conservative by 2.1×, recorded because the projection was printed before the run.

## 1. ✅ The estimator is sound — verified against §9.3 by a different route

For a field of variance `v`, `ρ(d) = 1 − γ(d)/v`, so `γ(1)` must reproduce §9.3's lag-1
autocorrelations, which were computed with `np.corrcoef` on shifted grids
(`d4_mu_on_the_grid.py:172`), not from a variogram.

| field | dir | D12 | §9.3 | diff |
|---|---|---|---|---|
| model `log mu` | E–W | 0.8583 | 0.8594 | −0.0011 |
| model `log mu` | N–S | 0.7609 | 0.7636 | −0.0027 |
| model gate logit | E–W | 0.9594 | 0.9598 | −0.0004 |
| model gate logit | N–S | 0.9567 | 0.9574 | −0.0007 |
| truth, window | E–W | 0.5842 | 0.5915 | −0.0073 |
| truth, 336-mo | E–W | 0.6378 | 0.6419 | −0.0041 |

⇒ **Worst disagreement 0.0090.** The measurement is right; what follows is a failure of the *model
fitted to it*, not of the instrument.

## 2. ⛔ F3 FIRED IN ALL FOUR CELLS — the fitted layer is VOID, by a condition written before the run

**No field plateaus by `d = 120`.** Model `log mu` E–W climbs 0.233 → 1.571 and the 336-month truth
2.679 → 7.240 across the full reach. §5's F3 says the consequence is that the sill comparison is
**"VOID for lack of reach — and it must be declared rather than fitted through"**, and since
`n + s(1 − exp(−d/ξ))` has only one sill to place, an inadmissible sill makes the fitted **nugget**
and **range** inadmissible with it.

⇒ **P-1, P-2, P-3, P-5, F2 and F4 are VOID, not missed.** The `⛔ MISSED` lines in
`d12_variogram.out` record an inadmissible fit and ⛔ **must not be read as falsifications** — e.g.
"P-3 gate/truth ξ ratio 2.599" is a ratio of two numbers that are not ranges.

⚠️ **And a second limit this prereg did not anticipate.** The curves are **non-monotone from
`d ≈ 9`** (in sample, E–W; `d ≈ 19` isotropic). A variogram cannot decrease. The pair population
changes with `d` on a 25%-land mask — by `d = 64` a large share of pairs span two continents — so the
large-`d` bins are not one population. ⛔ **That is C-351's population slot biting inside D12**, in the
one instrument chosen because the mask was *not* supposed to matter. The mask does not bias γ at a
given `d`; it changes *which* γ each `d` estimates.

## 3. ⭐ THE ANSWER, model-free: the amplitude goes missing at the SMALLEST scales

`γ_model(d)/γ_truth(d)` tends to the ratio of total variances if the two fields have the same shape,
so that ratio is the null. Divide by it and what is left is **shape alone**: 1.00 = the model carries
its full share of structure at that scale.

**In sample, E–W, against the 336-month base-rate truth** (variance null 0.222):

| | d1 | d2 | d4 | d8 | d16 | d32 |
|---|---|---|---|---|---|---|
| **body `log mu`** | **0.391** | 0.387 | 0.491 | 0.633 | 0.810 | 0.870 |
| **gate logit** | **0.112** | 0.182 | 0.314 | 0.504 | 0.711 | 0.822 |

Out of sample, E–W (null 0.396): body **0.205** → 0.776, gate **0.140** → 0.784.

⇒ **The deficit is monotone in scale and nearly closed by 32 cells.** At one cell the body carries
**39%** of its due share of structure and the gate **11%** — the gate is **3.5× worse than the body at
the smallest scale** and they converge as the scale grows. ⇒ This is the same shape
`Ayzel2020_RainNet` C-3654 reports (*"significant loss at 16 km and below"*) and
`Ravuri2021` C-3657's "effective resolution", in our own units.

## 4. ⛔ AND IT CORRECTS THIS PREREG'S OWN BASIS — the 12.8× was an extrapolation, not a measurement

§1b and §1c converted lag-1 autocorrelation to a correlation length with `ξ = −1/ln ρ`, giving
**24.37** cells for the gate against **1.90** for the truth ⇒ the **12.8×** and **3.5×** that were
quoted all programme. ⛔ **That conversion assumes the correlation function is exponential, and it is
not.** The exponential fitted at `d = 1` predicts `ρ_truth(8) = 0.029`; the measured value is
**0.316** — wrong by a factor of **11**, because the truth's decay is fast at one cell and slow after.

Model-free ranges instead — the smallest `d` with `ρ(d) ≤ τ`, with **τ swept** rather than chosen
(`feedback_sweep_before_reporting_a_scalar`), in sample, E–W, against the 336-month truth:

| τ | body | gate | truth | body ratio | gate ratio |
|---|---|---|---|---|---|
| 0.7 | 5 | 7 | 1 | **5.00×** | **7.00×** |
| 0.5 | 9 | 13 | 3 | **3.00×** | **4.33×** |
| 0.3 | 17 | 24 | 9 | **1.89×** | **2.67×** |

⇒ **The model's correlation length is longer than the truth's in every cell and at every threshold —
the SIGN is confirmed — but by 1.9–5.0× (body) and 2.0–7.0× (gate), not 3.5× and 12.8×.** The
ratio moves by 2.6× across the threshold sweep, so ⛔ **no single multiplier is quotable at all**;
the range is the result. Logged as `W-44`.

⚠️ **Does that reopen §1b and §1c?** Checked, not assumed: at τ = 0.5 the gate's range is ~13 cells,
so an FFT needs ~39 cells against the Africa window's ~20 — **both rejections stand, with the margin
shrinking from 73-vs-20 to 39-vs-20.** The instruments were rejected for the right reason on an
overstated number.

## 5. ⚠️ One anomaly, flagged and NOT claimed

The body's in-sample **N–S** `ρ(1) = 0.761` is **below** its own `ρ(2) = 0.824` — the only inversion
in 16 field-direction-window cells, and absent from the gate, from the truth and from the same field
out of sample. A field more unlike its immediate neighbour than its second neighbour is what
transposed-convolution checkerboarding and 32×32 patch stitching both produce. ⛔ **One cell, one
window, one head, and no second instrument** — recorded as an observation, with no mechanism claimed.

## 6. What D12 did and did not settle

| | |
|---|---|
| ✅ **settled** | the deficit's **scale profile** — worst at 1 cell, closed by ~32; the gate worse than the body below 8 cells; the model flatter than the truth at **every** resolved `d` (F1 never fired, 4/4 cells) |
| ⛔ **void** | every fitted parameter, hence the nugget/sill decomposition — so §4's free consistency check against D6-C's 0.154/0.081 **did not run**, and the nugget *is not* an estimate of the emission term |
| ⛔ **still foreclosed** | the component attribution (§0). `Cox1955` C-3611 is untouched by any of this, and C-353 now registers the class |
| 🔵 **open, and cheap** | a restricted-range fit (`d ≤ 8`, inside the monotone region) would make the nugget/sill test admissible. ⛔ **Post-hoc: it must be pre-registered as D12b before it runs, not chosen after seeing these curves** |
