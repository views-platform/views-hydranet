# D3 — the ceiling on a static base-rate channel, and what it found instead

**2026-10-20.** Free measurement, no pod, no training, nothing fetched. Run on artifacts already
on disk. Scripts: `2026-10-12_underfire_diagnosis/d3_ceiling.py`, `d3_followup.py`, `d3_clean.py`,
`slope_convention.py`, `slope_chase.py`, `smooth_control.py`, `d4_mu_on_the_grid.py`.

> ⛔ **READ §9 BEFORE QUOTING §4 OR §8. This report's own headline is falsified inside it.**
> §4 reported the body's magnitude slope as `b = 1.7301` and §8 built a "the two heads fail
> differently" conclusion on it. **§9 removes the limit that forced both** — it stores `mu` on all
> 64,818 land cells — and the slope **does not survive**: `b` = **0.92** in sample, **0.82** out of
> sample. 1.7301 is an artifact of the **100-cell set selected on observed positive rate**. The
> corrected finding is the opposite of §8's: **both heads fail the same way**, both emitting fields
> far smoother than the truth with roughly calibrated cell-level marginals. §4 and §8 are kept
> **unedited below except for pointers**, because the selection trap they fell into is the single
> most repeatable error in this dossier — it is the same trap as §2's tie artifact and §3's
> contamination, for the third time in one day.

## 0. Why it was run

`2026-10-12_underfire_next_step.md` §6 proposes **S1**: add one static channel per cell carrying its
own historical positive rate (and/or mean), ~$3 and ~2.5 h on one pod, readout on `(mu, theta)` and
`P(Y=0)` at h1.

Simon's objection, in his words: *"this is an LSTM architecture, so I don't understand how this
information is not already encoded in the hidden and cell state."* He added the framing that turned
out to matter — treat the field as a **doubly stochastic process** and the recurrent state may encode
the latent risk field well while encoding the marked point process on top of it badly.

D3 was pre-named by the `opponent` seat as the cheapest discriminator: regress the model's per-cell
mean error on the proposed channel, and read the **R² as S1's ceiling**. It cost minutes of CPU.

## 1. Design

| half | data | cells | months | in/out of sample |
|---|---|---|---|---|
| **D3-GATE** | `p2_result.npz` | **64,818** (all land) | 457–469, h=1 | **out of sample** |
| **D3-BODY** | `t3_result.npz` | 100 (top activity) | 421–456 | in sample (train period) |

Both artifacts come from the retrained vehicle `4acbf4d3c2b457b0…` — **not** the delivered weights
(C-317).

⛔ **C-350 is honoured.** Every channel is built strictly from months ≤ 456, the training partition
(`t3_scoring.py:45` records 421–456 as inside the train period; p2's 457–469 are out of sample). The
channel never reads 457–469.

**Free parameters swept, not chosen** (⛔ the standing rule, and it earned its keep twice here): two
channel forms (`pos_rate`, `log1p_mean`) × six history windows.

## 2. D3-GATE — the occurrence half has essentially nothing to gain

A logistic fit of the truth on the model's own gate logit, then the same fit with the channel added.
The gate logit is given a free slope and intercept, so what the channel adds is **beyond any
recalibration**. Fitted and scored on the same months **on purpose**: that makes it an upper bound,
and an upper bound is the only direction that can kill S1.

| | log-loss | AP |
|---|---|---|
| gate alone | 0.008194 | **0.563093** |
| + `pos_rate_301_456` | 0.008112 | 0.575582 (**+0.0125**) |
| + `pos_rate_121_456` | 0.008126 | 0.572129 (+0.0090) |
| + `log1p_mean_121_420` | 0.008151 | 0.569124 (+0.0060) |

**ΔAP +0.006 to +0.012 across all six sweeps** — positive, consistent, and a ceiling. Against a seed
sd of ~0.019 on this platform, it is **inside the noise**.

⛔ **One result in the first pass was an artifact, and the check that caught it is worth keeping.**
Per-cell over all 64,818 cells, the channel appeared to rank cells by their out-of-sample positive
rate *better than the model's own gate* — Spearman **+0.51 vs +0.20**. That reverses on inspection:
**98.6%** of land cells have an observed rate of exactly 0 and **96.0%** have a channel of exactly 0,
so the statistic is dominated by ties. Restricted to cells that have any history or any activity:

| cell set | n | spearman(channel, obs) | spearman(gate, obs) |
|---|---|---|---|
| all cells | 64,818 | +0.5095 | +0.2023 |
| channel > 0 | 2,608 | +0.5364 | **+0.7235** |
| obs > 0 | 912 | +0.5923 | **+0.8109** |

**The gate wins decisively wherever the question is meaningful.** Simon's premise is right for
occurrence: the recurrent state already holds the per-cell risk ordering, and holds it better than a
30-year base rate does.

## 3. D3-BODY — the channel's ceiling, after the contamination is removed

First pass put the best channel at **R² 0.3802** (`log1p_mean_301_456`). ⛔ **That number is
contaminated:** the window 301–456 overlaps the readout window 421–456, so the channel is partly
reading the quantity it is being asked to predict. Re-run with every window ending at 420:

| channel | span | R² raw | **R² after the two global scalars (§4)** |
|---|---|---|---|
| `pos_rate_121_420` | 300 | 0.0006 | **0.0065** |
| `log1p_mean_121_420` | 300 | 0.0529 | **0.0448** |
| `log1p_mean_301_420` | 120 | 0.1672 | **0.0991** |
| `log1p_mean_409_420` | 12 | 0.2993 | **0.1016** |
| `pos_rate_385_420` | 36 | 0.1164 | **0.0183** |
| *(contaminated, for contrast)* `log1p_mean_301_456` | 156 | — | *0.2108* |

**Two things the sweep decided that a single number would have got wrong:**

1. ⛔ **Decontaminating halves it.** 0.211 → **0.102**. Half the apparent signal was window overlap.
2. ⛔ **The form matters more than the idea.** `pos_rate` — which is the form §6 names **first** —
   tops out at **0.023**. `log1p_mean` reaches **0.102**. A 4× difference between two phrasings of
   the same sentence.

**S1's honest ceiling on the body is R² ≈ 0.10**, on a selected 100-cell set.

## 4. What it found instead — the body's slope defect

> ⛔ **SUPERSEDED IN ITS HEADLINE NUMBER BY §9, same day.** Everything in this section is correctly
> measured **on the 100-cell selected set** and is wrong as a statement about the body. `b = 1.7301`
> → **0.92 / 0.82** on the full grid. The ranking claim (spearman ≈ 0.69) **survives** and is in fact
> stronger on the grid (0.82 / 0.78); it is the *spread* claim that does not. Kept unedited.

The sign of every one of the twelve clean fits is **negative**: the higher a cell's history, the more
the model under-predicts it. That has two readings with opposite consequences — missing information,
or compressed range — so it was tested directly.

| | measured |
|---|---|
| spearman(`mu_model`, `mu_emp`) across cells | **0.6870** (independently reproduces M3's 0.687) |
| pearson on logs | 0.7174 |
| sd of `log mu_model` ÷ sd of `log mu_emp` | **0.4146** |
| `b` — slope of **truth on prediction**, `log mu_emp = a + b·log mu_model` | **1.7301** |
| `b'` — the other direction | 0.2974 |
| median `log(mu_model / mu_emp)` | **−0.5831** (predicts ~56% of the empirical mean) |

**The model ranks cells correctly and commits to 41% of the right spread.** It is not ignorant of
which cells are violent; it refuses to say how violent.

⛔ **This is the BODY's instance of the defect GLOSSARY §5b locked for the GATE on 2026-10-18.** Same
statistic in the same convention — truth regressed on prediction in a link space, `b > 1` meaning
truth moves faster than the prediction. The comparison:

| | `b` |
|---|---|
| gate, control kestrel | 1.146 / 1.177 / 1.139 |
| gate, arm kestrel | 1.419 / 1.251 / 1.456 |
| **body, this measurement** | **1.7301** |

⚠️ **Indicative, not like-for-like.** The gate's `b` is fitted on binned logits of a cell-month
Bernoulli; the body's is OLS on per-cell log means. Same kind of statistic, different estimator, so
the ordering is suggestive and the ratio is not a measurement. Both are **lower bounds** for the same
errors-in-variables reason §5b already states: noise in the predictor attenuates the slope toward 1.
Total-least-squares on the body gives **2.41**.

⛔ **An earlier version of this section concluded "one defect wearing two coats, not two findings."
§8 measured the like-for-like comparison the same day and it is wrong.** Kept here, struck, because
the reasoning that produced it is the reasoning `§5b`'s wording invites.

## 5. ⛔ Two global scalars do not fix it

If the compression were a single mis-scaling, a gain and an offset on `log mu` would remove it. Fitted
at their optimum:

| | rmse of per-cell log error | median |
|---|---|---|
| before | 1.0239 | +0.5831 |
| after the best two scalars | 0.7727 | −0.0122 |

**Variance removed: 15.9%.** The median goes to zero — the *level* error is a global constant and is
free to fix — but **84% of the per-cell error survives**, so the compression is per-cell, not global.

That is what an objective which rewards timidity produces, and T3 already measured that this objective
does: a **mean-preserving de-timidified** version of the model — same `mu`, `theta` raised until
`P(Y=0)` matches the observed zero rate — **loses** to the timid one. M6 is the same fact at the
parameter: emitted `theta` **0.1544** against an NLL-optimal **0.2327** at the model's own `mu`.

## 6. Verdict

⛔ **S1 as specified in §6 should not run.** On measurement, not on argument:

| §6's claim | measured |
|---|---|
| *"cannot know which cells are chronically violent"* | **false** — 0.687 body, 0.72–0.81 gate on cells that matter |
| a base-rate channel is the lever | ceiling **R² 0.10** body, **+0.01 AP** occurrence — and **0.02 / 0.006** in the form §6 actually names |
| readout at h1 on `(mu, theta, P(Y=0))` | h1 is where the information is already present; and C-228 measured this seam collapsing gate AP **0.31 → 0.13/0.20** with `crps_all` inert, i.e. **invisible to all three of S1's readouts** |

✅ **Simon's objection is sustained, with a mechanism it did not have before:** the recurrent state
holds the latent risk field; what fails is the **scale of the marks on top of it**. His doubly
stochastic framing predicted exactly this split, and the numbers in §2 and §4 are that split measured.

🔴 **The target is `b = 1.73` / sd ratio 0.41.** One number, large, unexplained, and now known to be
the same defect the gate carries — which means the `weight_decay` programme already pre-registered
for the gate's slope is the first place to look, not a new covariate.

## 7. What these numbers are not

- **One artifact**, `4acbf4d3c2b457b0…`, retrained, not the delivered weights (C-317). n=1.
- The body half is a **100-cell set selected on observed positive rate** — restricted range, which
  attenuates slopes and R² alike. It cuts both ways and is not corrected for.
- The body half's readout window is **in sample**. Only D3-GATE is out of sample.
- The D3-GATE ΔAP is fitted and scored on the same months: a **ceiling**, deliberately generous.
- No falsifier was pre-registered for D3 — it is a measurement that bounds a proposal, not an arm.
  Under C-307 that makes it **informative about a ceiling** and not a closure of anything.


## 8. The chase — and the correction it forced

`slope_chase.py` and `smooth_control.py`, same day, same artifacts. Written to find the mechanism
behind `b = 1.73`. It did not find it, and it falsified §4's closing sentence instead.

### 8.1 ⛔ The gate does NOT share the body's slope defect

§4 compared the body's per-cell OLS `b` against the gate's binned-logit `b` and flagged the estimator
mismatch, then drew a unifying conclusion anyway. Applying the **body's own estimator** to the gate,
on all 64,818 land cells at the 13 out-of-sample months:

| cell set | n | `b` | r | sd(pred) ÷ sd(truth) |
|---|---|---|---|---|
| all land cells | 64,818 | **0.3906** | +0.5678 | 1.4538 |
| cells with any event | 912 | **0.9444** | +0.7270 | 0.7698 |
| *(body, same estimator, §4)* | 100 | *1.7301* | *+0.7174* | *0.4146* |

`b ≤ 1` for the gate under this estimator — **over**-dispersed on all cells, essentially calibrated on
active ones. The body's compression is **the body's alone**, as far as anything here measures.
GLOSSARY §5c has been corrected in the same commit.

### 8.2 The scale discriminator ran, and tested the wrong head

Aggregating into `k×k` blocks separates three mechanisms: spatial smoothing relaxes `b` toward 1 as
`k` grows, global shrinkage leaves it flat, rank noise relaxes it while `r` climbs steeply.

| k | blocks | `b` | r |
|---|---|---|---|
| 1 | 912 | 0.9444 | +0.7270 |
| 2 | 514 | 0.7479 | +0.8364 |
| 4 | 267 | 0.7772 | +0.8932 |
| 8 | 129 | 0.8348 | +0.9280 |
| 16 | 65 | 0.9085 | +0.9573 |
| 32 | 32 | 0.9854 | +0.9876 |

`b` hovers at 0.75–0.99 at every scale while `r` climbs 0.73 → 0.99. For the gate that is the
**rank-noise** signature and there is no slope defect to explain.

⛔ **Stated plainly: the discriminator could not test the body, because `mu` is not saved on the full
grid by any artifact in this dossier.** The gate is the only head available at 360×720, and the gate
turns out not to have the defect. **The mechanism behind `b = 1.73` is still unidentified.** Closing it
needs one forward pass of the vehicle to write `mu` over the land grid — no training, no pod, but not
free either.

### 8.3 ✅ What the chase did find: the gate's field is spatially over-smoothed

| field | lag-1 autocorrelation (E–W / N–S) |
|---|---|
| **gate logit** | **+0.9644 / +0.9601** |
| truth logit, 13 months | +0.5371 / +0.5234 |

⚠ **That naive comparison is confounded** — 13 months of a rate quantised to 1/13 and positive in
1.4% of cells is a noisy field, and noise attenuates autocorrelation. Controlled by lengthening the
truth window:

| truth window | months | lag-1 (E–W / N–S) |
|---|---|---|
| 445–456 | 12 | +0.5594 / +0.5250 |
| 421–456 | 36 | +0.6200 / +0.6063 |
| 385–456 | 72 | +0.6472 / +0.6285 |
| 301–456 | 156 | +0.6743 / +0.6567 |
| 121–456 | **336** | **+0.6980 / +0.6797** |

The truth's autocorrelation rises and flattens — increments of 0.083, 0.027, 0.027, 0.024 per
roughly-doubled window — so it is heading for **~0.70–0.75**, not 0.96. Part of the original gap was
noise; **most of it survives the control.** The gate's field is genuinely smoother than the process it
predicts.

⛔ **So the two heads fail differently, and this is the sharper statement §4 should have made:**

| head | marginals | spatial structure |
|---|---|---|
| **gate** | roughly calibrated (`b` 0.94 on active cells) | **over-smoothed** (0.96 vs ~0.70) |
| **body** | **compressed** (`b` 1.73) | unmeasured |

> ⛔ **THIS TABLE IS WRONG IN ITS BODY ROW AND §9 MEASURED IT.** "Unmeasured" was the honest state
> when this was written and is the reason the forward pass was run. With `mu` on the grid the body
> row reads **`b` 0.92 / 0.82 — roughly calibrated** and **field over-smoothed, 0.86–0.93 against the
> same ~0.64 reference**. The two heads fail **the same way**. Kept struck rather than rewritten
> because "the heads differ" was itself a correction of "the heads are the same", and the sequence
> same → different → same, each step forced by a better measurement, is the useful record.

This is C-152 and EXP-3 in a new coordinate rather than a new finding: *"89% of the damage is cells in
the wrong places"*, with the marginals intact. It is also, unprompted, the shape Simon proposed — the
latent risk field survives, what sits on top of it does not.

## 9. D4 — `mu` on the grid, and the slope defect does not survive it

§8.2 closed with a named limit: the discriminator could only run on the gate, because no artifact
held `mu` on the full grid, and the gate turned out not to have the defect. `d4_mu_on_the_grid.py`
removes the limit — one forward pass of the same vehicle, the same lead-in and the same locked-dropout
protocol as `p2_gate_calibration.py`, collecting the `sb` body parameters **and** the gate for all
64,818 land cells at all 49 months 421–469 in one pass, so the two heads cannot drift apart.

**Cost, measured** (⛔ the standing rule — a number, not an estimate): **349.5 s wall, 3.95 GiB peak
RSS**, 4 posterior passes, output `d4_result.npz` 138.9 MB. Capped at 8 threads. Truth panel positive
rate **0.003499**, consistent with the 0.003448 on record.

### 9.1 ⛔ `b` = 1.73 is a selection artifact

| cell set | n | `b` | `r` | sd(`log mu`) | sd(`log truth`) | sd ratio |
|---|---|---|---|---|---|---|
| **in sample 421–456**, any event | 1,431 | **0.9234** | +0.8132 | 1.7140 | 1.9464 | 0.8806 |
| in sample, ≥2 events | 904 | **1.0049** | +0.8261 | 1.4069 | 1.7114 | 0.8221 |
| in sample, all land cells | 64,818 | 0.9032 | +0.5966 | 1.2813 | 1.9401 | 0.6605 |
| **out of sample 457–469**, any event | 912 | **0.8170** | +0.7718 | 1.6149 | 1.7096 | 0.9446 |
| out of sample, ≥2 events | 515 | **0.9670** | +0.7795 | 1.2311 | 1.5273 | 0.8061 |
| out of sample, all land cells | 64,818 | 0.4491 | +0.4746 | 1.7108 | 1.6189 | 1.0568 |
| **§4's 100-cell selected set** | 100 | **1.7301** | +0.7174 | — | — | **0.4146** |

**The slope is ≈1, and the entire difference is the sd ratio, not the correlation.** `b = r·sd(truth)/sd(mu)`:
on the selected set `0.7174 × (1/0.4146) = 1.730`; on the grid `0.8132 × (1/0.8806) = 0.923`. `r` is
*higher* on the grid. So §4 did not measure a compressed body — it measured a **compressed predictor
range**, which is what selecting the top 100 cells by observed positive rate produces: every selected
cell is one the model already calls high-risk, so `mu` barely varies across them, while the observed
rate keeps varying because at the top of a rate ranking it is partly noise.

✅ **The selection gradient is monotone, which is the internal check:** as the condition on the
outcome tightens, `n` falls 1,431 → 904 → 100 and `b` rises 0.9234 → 1.0049 → 1.7301. The headline
sat at the extreme of a gradient the design created.

⚠️ **Stated honestly: none of these rows is unbiased.** "Any event" and "≥2 events" also condition on
the outcome, just far more weakly; "all land cells" does not, but **98.6% of land cells never fire**,
so it is the tie-dominated statistic §2 already caught once — note its out-of-sample value of 0.4491
at an sd ratio **above 1**, which is 13 months of quantised truth, not a finding. The supported claim
is therefore **"`b` = 1.73 is not the body's slope"**, not "the body's slope is exactly 0.92".

### 9.2 The discriminator, now on the head that was supposed to have the defect

| `k` | blocks | `b` in sample | `r` | `b` out of sample | `r` |
|---|---|---|---|---|---|
| 1 | 1,431 / 912 | 0.9234 | +0.8132 | 0.8170 | +0.7718 |
| 2 | 731 / 514 | 1.0455 | +0.8733 | 0.9123 | +0.8377 |
| 4 | 346 / 267 | 1.1092 | +0.8864 | 0.9677 | +0.8713 |
| 8 | 162 / 129 | 1.2321 | +0.8999 | 1.0202 | +0.9284 |
| 16 | 78 / 65 | 1.4894 | +0.9392 | 1.1136 | +0.9557 |
| 32 | 33 / 32 | 1.6438 | +0.9381 | 1.2318 | +0.9740 |

⚠️ **`b` rising through 1 is not one of the three signatures `slope_chase.py` pre-registered** (S
smoothing → `b`→1 · G global shrinkage → `b` flat · R rank noise → `b`→1 with `r` climbing). The
discriminator therefore **does not discriminate here**, and its labels must not be applied. What it
does establish is a one-line fact about variance: across `k` = 1…32 in sample, **sd(`log mu`) is flat
— 1.7140 → 1.7045 — while sd(`log truth`) grows 1.9464 → 2.9867.** The model's amplitude does not
scale up when the grid is coarsened and the truth's does.

⇒ **The body has no large-scale amplitude structure.** It is not squashed cell by cell; it is missing
the regional highs and lows, which is a statement about the *field*, not the marginals.

### 9.3 ✅ Both heads emit a field far smoother than the truth

| field | lag-1 autocorrelation, in sample (E–W / N–S) | out of sample |
|---|---|---|
| **body, `log mu`** | **+0.8594 / +0.7636** | **+0.9266 / +0.9227** |
| **gate logit**, same pass | +0.9598 / +0.9574 | +0.9498 / +0.9464 |
| truth, same window | +0.5915 / +0.5780 | +0.5089 / +0.5028 |
| **truth, noise-controlled** — 336-month base rate (§8.3) | **+0.6419 / +0.6261** | same |

Against the §8.3 noise control — the one reference that is not attenuated by a short window — the
truth sits at **~0.64** and **both heads sit at 0.86–0.96.** The body ranks cells well: spearman(`mu`,
truth) **+0.8205** in sample and **+0.7808** out of sample, better than the 0.687 §4 reported on the
selected set.

### 9.4 The corrected finding

| head | cell-level marginals | its field |
|---|---|---|
| **gate** | roughly calibrated, `b` 0.39 all / 0.94 active | over-smoothed, 0.95–0.96 vs ~0.64 |
| **body** | roughly calibrated, `b` 0.82–0.92 | over-smoothed, 0.86–0.93 vs ~0.64 |

⛔ **The two heads fail the same way, and §8's "they fail differently" is withdrawn.** Both know which
cells are dangerous and neither can say *where within a region* — they put the right total mass on a
field that is too diffuse. Since the defect is shared across two heads that share only the trunk, it
is **upstream of both**: the recurrent state and the decoder, not either head's marginals.

**This sustains Simon's objection on its own terms and sharpens it.** His framing was that the hidden
and cell state may encode the latent risk field well and the marked point process on top of it badly.
Measured: the ranking is good (0.78–0.82) and the *spatial sharpness* is not (0.86–0.96 against 0.64).
⇒ The lever is the **trunk or the loss**, as he argued, and **not** a per-cell static input — which
§2 and §3 had already bounded at +0.01 AP and R² 0.10 from the other direction. Two independent routes
to the same answer, and §9 is the one that identifies *what* is wrong rather than only what a channel
could not fix.

⚠️ **What §9 does NOT establish.** Why the fields are smooth. Over-smoothing is now measured on both
heads and its *cause* is untested — decoder receptive field, the recurrent BatchNorm, weight decay on
the heads, and the NLL's own preference are all live and none is measured. ⛔ It also closes nothing
under C-307: one vehicle (`4acbf4d3c2b457b0…`, the retrained control, **not** the delivered weights —
C-317), n=1, `sb` only, no pre-registered falsifier. And the C-152 class statement still binds — *any*
proposal that improves per-cell marginals inherits that null unless it changes the joint or the
sampler, and §9 says the marginals were not the problem in the first place.
