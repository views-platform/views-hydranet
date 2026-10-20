# D3 — the ceiling on a static base-rate channel, and what it found instead

**2026-10-20.** Free measurement, no pod, no training, nothing fetched. Run on artifacts already
on disk. Scripts: `2026-10-12_underfire_diagnosis/d3_ceiling.py`, `d3_followup.py`, `d3_clean.py`,
`slope_convention.py`, `slope_chase.py`, `smooth_control.py`, `d4_mu_on_the_grid.py`.

> ⛔ **READ §10 FIRST. It audits §9, which audits §4 and §8. Three of this report's own claims are
> falsified inside it, and §10.1 is the status table for every one.**
>
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

## 10. D5–D7 — §9 audited against its own falsifier. One sub-claim survives shrunken, one is withdrawn.

**2026-10-20, same day.** Pre-registered in `2026-10-20_d5_oversmoothing_prereg.md` (commits e9bc4b4,
a00c975, b1b49a6 — **all before any of these ran**). Scripts `d5_why_smooth.py`,
`d6_estimator_null.py`, `d7_coarsening_sample.py`. Costs measured: **159.3 s / 2.99 GiB**,
**7.7 s / 2.95 GiB**, **2.4 s / 3.29 GiB**. D6 and D7 read `d4_result.npz` — no new forward pass.

The `opponent` seat attacked §9.3 (its §13, commit 885d6e8). ⛔ **The objection I had nominated as my
own weakest joint failed; a different one landed and took most of my effect size with it; and testing
it properly exposed a separate artifact in §9.2 that nobody had attacked.**

### 10.1 Status of every §9 sub-claim

| sub-claim | status |
|---|---|
| **§9.1** `b` = 1.73 is a selection artifact; the slope is ≈0.9 | ✅ **STANDS**, untouched |
| **§9.2** the body has no large-scale amplitude structure | ⛔ **WITHDRAWN** — an admission-rate artifact (§10.4) |
| **§9.3** both heads emit fields smoother than the truth | ✅ **STANDS** by its pre-registered falsifier, at **~¼ the effect size** (§10.3) |
| **§9.4** the defect is upstream of both heads | ✅ stands, **wording corrected** (§10.5) |

### 10.2 ✅ D6-A — the noise objection is dead, and more thoroughly than the seat said

Fit `ρ_obs(T) = ρ_S/(1 + v/T)` on **nine** truth windows, not three, leave-one-out:

| T (months) | 12 | 24 | 36 | 48 | 72 | 108 | 156 | 240 | 336 |
|---|---|---|---|---|---|---|---|---|---|
| observed (E–W/N–S mean) | 0.5172 | 0.5598 | 0.5848 | 0.5941 | 0.6036 | 0.6127 | 0.6266 | 0.6333 | **0.6340** |
| LOO prediction | 0.4970 | 0.5680 | 0.5874 | 0.5984 | 0.6100 | 0.6176 | 0.6208 | 0.6239 | 0.6262 |

Full fit **`ρ_S` = 0.6329, `v` = 2.8131**, max LOO error **0.020** (at T=12, the noisiest window; every
other window predicts to **≤0.009**). ⇒ **The 336-month reference is at its asymptote** — the observed
0.6340 is `−0.17%` *past* the fitted `ρ_S`, i.e. attenuation is indistinguishable from zero. The seat's
3-window fit gave 0.6405 / 3.4612 / 1.02%; nine windows agree in substance and identify it better.
**My §8.3 instinct was right and is now a measurement.**

### 10.3 ⛔ D6-C — the falsifier fires FOR §9.3, and takes ~¾ of the gap with it

The landing objection: `log mu` and the gate logit are continuous and **atom-free**; the truth is a raw
rate with **98.6% of its mass at exactly zero**. Lag-1 Pearson is invariant under *affine* maps, not
under `log` or `logit`, so the rows were never on one scale — and the truth **cannot** be put on the
model's scale, because `log(0)` does not exist for 98.6% of land cells.

The fix is to push the **model** through the **truth's** estimator. Per cell, 336 months of the
gated-NB generative model, the base-rate field built exactly as §9.3 builds the truth's, lag-1 of its
log, **300 replicates**. Done exactly, not by brute force: a sum of `k` i.i.d. `NB(θ, p)` is
`NB(kθ, p)`, so `k ~ Binomial(336, gate)` then `S ~ NB(kθ, p)`.

| | E–W | N–S |
|---|---|---|
| the model's field, measured directly (what §9.3 quoted) | **+0.8594** | **+0.7636** |
| the same field through the truth's estimator, mean of 300 | **+0.7053** | **+0.6824** |
| 95% posterior-predictive interval | [+0.6922, +0.7193] | [+0.6683, +0.6962] |
| **the truth, observed** | **+0.6419** | **+0.6261** |
| the estimator's own attenuation | **0.154** | **0.081** |
| what survives it | **0.063** | **0.056** |

> ⛔ **AND §9.3 WAS NOT A WELL-POSED QUESTION UNTIL THIS TEST EXISTED.** The `opponent` seat's
> strongest point, added after the first round and verified here independently. By the law of total
> variance `Var(Y) = Var(E[Y|X]) + E[Var(Y|X)]`, write a realisation as `Y_s = m_s + e_s` with
> `m = E[Y|X]`. If the conditional noise `e` is spatially uncorrelated with variance `σ²`, then for
> `s ≠ t` `Cov(Y_s, Y_t) = Cov(m_s, m_t)` while `Var(Y_s) = Var(m) + σ²`, so
>
> `ρ_Y(1) = ρ_m(1) · Var(m) / (Var(m) + σ²)  <  ρ_m(1)`   — **strictly.**
>
> **A conditional-mean field is smoother than its own realisation BY CONSTRUCTION.** So "is the
> model's field smoother than the truth's?" has the answer *yes* before any data is collected, for
> any model that emits a conditional mean. The only question with content is **"is it smoother than
> the decomposition requires?"** — and pushing the model through the truth's estimator is the one
> form that asks it, because the simulation injects the model's own `E[Var(Y|X)]` back in. ⇒ **D6-C
> is not a better comparator than §9.3's; it is the only version of the claim that means anything,
> and §9.3 as originally written was untestable rather than merely imprecise.**
>
> ⚠️ Exact on the untransformed field with white conditional noise — and D6-A's attenuation fit
> working to ≤0.009 at eight of nine windows is independent evidence that this field's sampling noise
> **is** near-white. The `log` of a 336-month mean makes it approximate, **which is precisely why the
> instrument has to be a simulation that applies the actual transform rather than an algebraic
> correction factor.**

**Pre-registered verdict: the truth falls OUTSIDE the interval on both axes ⇒ §9.3 STANDS.** ⛔ **And
the seat was right about the magnitude:** the estimator alone accounts for **0.154** of the E–W gap,
almost exactly the ~0.15 it predicted from a synthetic field. The honest gap is **0.06, not 0.22** —
roughly **a quarter** of what §9.3 reported. The direction survives; the size does not.

⛔ **CORRECTION to how the interval is named.** The 300 replicates carry **Monte Carlo variation in
the simulation only** — there is **no parameter-uncertainty component at all**, because this is one
training run and D6-C averaged over `d4`'s four dropout passes rather than propagating them. So it is
a **simulation interval**, not a posterior-predictive interval, and the earlier label overstated it.
It does not change the verdict: the truth sits **3.7** (E–W) and **3.0** (N–S) interval half-widths
**below** the lower edge, so a 3× wider interval keeps it outside.

✅ **0.063 / 0.056 is a FLOOR on three counts, all pointing the same way** (the first was mine, the
other two the seat's and both verified here):

1. The simulated field's atom is **larger** than the truth's (98.7% vs 94.2% silent) and the
   understatement is **monotone in the atom** ⇒ the model's row is attenuated more than the truth's.
2. The simulated support is not just sparser but **patchier** — 1.28% vs 5.85% of cells over 336
   months — so it is fewer, more concentrated places, which lowers the model's row further.
3. ⛔ **The model is over-dispersed, so the simulation injects more conditional noise than a
   well-calibrated version of the same model would.** `θ = 0.1544` gives `Var = μ + 6.477μ²` against
   the NLL-optimal `θ = 0.2327`'s `μ + 4.297μ²` — **1.51× the conditional variance the loss itself
   would choose** (`d45ce2c` establishes the premise). More injected noise ⇒ rougher simulated field
   ⇒ lower model row ⇒ **narrower measured gap.** Precisely: **0.06 is a floor on the smoothness
   excess attributable to the MEAN FIELD**, because part of the model's simulated roughness is its
   own over-dispersed noise rather than mean-field structure.

⚠️ **Two further limitations, one of which cuts against me and one for me.**
- **Against:** the simulated field is *less active than the truth* — 1.28% of cells ever fire in 336
  simulated months against **5.85%** observed, a factor of 4.6. The generative model is a poor fit to
  the truth's activity, which weakens this as a null. (It is also the under-firing this whole
  programme is about, showing up in a new place.)
- **For:** that mismatch makes the model's atom **larger** than the truth's, and the seat's own result
  is that the understatement is **monotone in the atom**. So the model's 0.7053 is itself attenuated
  *more* than the truth's 0.6419 is, and correcting for the mismatch would **widen** the surviving
  gap. The test is conservative in the direction of my conclusion.
- The simulation treats the model's 36-month mean parameters as stationary over 336 months; the
  truth's 336 months are not stationary. Stated, not corrected.

### 10.4 ⛔ D7 — §9.2 is WITHDRAWN. The seat found a tension; the tension was my artifact.

The seat's sharpest point was that §9.2 and §9.3 are in tension — §9.2 says too *little* large-scale
amplitude, §9.3 too *much* local correlation, and a blur produces the second and **the opposite** of
the first. Correct, and the resolution is not a reconciliation: **§9.2 is wrong.**

`d4_mu_on_the_grid.py`'s coarsening keeps only blocks with `tm > 0`, and a bigger block is more likely
to contain an event. So the **admitted population changes with `k`**:

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| blocks admitted / blocks with land | 2.2% | — | — | 11.5% | 18.1% | **22.4%** |

**A factor of 10.2.** §9.2's sd trend across `k` compares different populations at every step. Fix the
cell set once (cells with any event at `k`=1) so the population is nested, and the trend **reverses**:

| k | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| sd(`log mu`) | 1.7140 | 1.6858 | 1.7332 | 1.8274 | 1.8763 | 1.6918 |
| sd(`log truth`) | 1.9464 | 1.7940 | 1.7184 | 1.6379 | 1.5694 | 1.5669 |
| **ratio mu/truth** | **0.88** | 0.94 | 1.01 | 1.12 | **1.20** | 1.08 |

The truth's spread **falls** under coarsening and the model's is flat, so the ratio **rises above 1**.
⇒ The body does not lack coarse-scale amplitude; if anything it has slightly **too much**. §9.2's
"flat 1.714 → 1.705 while the truth grows 1.946 → 2.987" was measuring the `tm > 0` admission rate.

⛔ **This is the fifth instance of one error class in this dossier in one day, and the third inside
§9 alone** — §2's tie artifact, §3's contamination, §9.1's selection, §9.3's scale mismatch, and now
§9.2's admission rate. Every one is *a statistic computed on a set, or a scale, defined by the thing
being measured.* It is no longer a run of bad luck; it is the characteristic failure of this entire
measurement style, and the register entry is the deliverable (**C-351**).

✅ **With §9.2 gone there is no tension: one defect, not two.** The model's field is over-correlated
at every lag and its coarse-scale amplitude is fine.

### 10.5 What the lag profile does and does not narrow

`ρ(model)/ρ(truth)` rises monotonically and **never returns to 1**:

| lag | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|
| `log mu` / truth, E–W | 1.34 | 1.47 | 1.66 | 1.95 | 2.24 | 2.42 | 2.82 | **4.08** |
| gate logit / truth, E–W | 1.50 | 1.62 | 1.88 | 2.25 | 2.62 | 2.93 | 3.46 | **4.79** |

**No knee.** D5-B measured the **one-step** impulse reach with the trained weights — perturb one land
cell's `sb` input by +1 in `log1p` space and `|Δ log mu|` falls below 10% of peak at **10 cells** and
is **exactly 0 at ≥16**. A one-step blur of radius ~12 would return the ratio to 1 beyond 16 and it
does not. ⇒ **A one-step decoder blur is excluded as the sole cause.**

⛔ **AND THE SEAT WITHDREW ITS OWN DISCRIMINATOR, which strengthens this rather than weakening it.**
Its "knee at the radius, identical in both heads (same decoder depth)" treated the receptive field as
a **feed-forward decoder** property. The trunk is a **ConvLSTM**: effective spatial reach grows by
roughly one kernel radius **per carried step**, over far more steps than D5-B probes. So **recurrent
reach predicts exactly no knee** over any lag range shorter than the carried-state horizon ⇒ the
monotone, knee-free, never-returning-to-1 profile is the **predicted signature** of recurrent reach,
not merely consistent with it.

⚠️ **That is one candidate of five excluded, and I will not claim more.** The window is **49 recurrent
steps**, so the rising ratio *is what recurrent reach looks like*. And BN recal and global weight decay are both
scale-free, so this profile does not touch them either. Live: recurrent reach, BN recal, global weight
decay, and conditional-mean averaging. **Excluded: a single-step decoder blur.**

### 10.6 ⛔ D5-A is VOID, by a condition written before it ran

The untrained-architecture control **fired its own pre-registered VOID guard**. At random init the
field is essentially constant: `sd(log mu)` = **0.0017** and `sd(gate logit)` = **0.0016**, both far
below the committed 0.01 floor, with `mu` spanning 0.4798–0.5328 and `gate` 0.4993–0.5266 over all
64,818 cells. Its autocorrelations (+0.7745/+0.6146 and +0.5827/+0.3609) are the autocorrelation of
numerical noise on a constant and **mean nothing**.

**The architectural-vs-learned question is not answerable by an init measurement on this
architecture**, because this architecture at init emits a constant. The prereg named this exact
failure mode in advance (*"a random-init network through `softplus` is a real candidate for exactly
that"*), which is the only reason the +0.77 was not reported as "architectural". ✅ **The guard is the
result here** — and D5-B, written as the descriptive half, turned out to carry the architectural
question instead.

### 10.7 What §9 and §10 together now support

1. The body's cell-level slope is ≈0.9, **not** 1.73 (§9.1).
2. Both heads emit fields **measurably** smoother than the truth, by **0.06** on a like-for-like
   estimator, surviving a pre-registered two-sided falsifier (§10.3).
3. The coarse-scale amplitude story is **withdrawn** (§10.4).
4. The defect is shared by both heads ⇒ it lies in **anything they share** (§10.5).
5. A one-step decoder blur is **excluded**; four candidates remain, and the one the literature names
   — a pixel-wise proper-scoring objective emitting a conditional mean — is the only one that
   predicts the same signature on two heads with **different** losses.

⛔ Under **C-307** this is still an n=1 screen on one artifact (`4acbf4d3c2b457b0…`, the retrained
control, **not** the delivered weights — C-317), `sb` only. It bounds and narrows; it closes nothing.
And **nothing here revives S1**: that verdict rests on §2/§3's `+0.01 AP` and `R² 0.10` bounds and on
C-152, neither of which involves the truth comparator at all.

### 10.8 D8 — a by-product worth having, whose first reading was a confound

The `opponent` seat noticed that D6-C threw off a free statistic: the simulated field fires in
**1.28%** of cells over 336 months against the truth's **5.85%**, while the body's cell-level
marginals are roughly calibrated (`b` 0.82–0.92). **Right mass, too few places** — on a cumulative
support count, with no scale choice, no normalisation and no Moran's I, so it does not inherit the
C-299 floor-limited annotation the Moran's I series carries. It asked for the derivation rather than
the paragraph. `d8_support_vs_monthly.py` derives both halves **analytically** from the model's own
per-cell parameters — `P(y>0) = gate·(1 − p^θ)`, support `= 1 − (1 − P)^T` — so there is no
simulation noise at all.

**⛔ And deriving it properly reversed its headline.** The seat's paragraph inferred over-persistence
from a cumulative shortfall larger than the monthly one. Over 336 months that is what appears:

| | model | truth | shortfall |
|---|---|---|---|
| monthly `P(y>0)` | 0.000841 | 0.002094 | **2.49×** |
| cumulative support, 336 months | 0.012668 | 0.058471 | **4.62×** |
| ratio cumulative / monthly | | | **1.853** |

⛔ **But that cumulative column is confounded, and the confound pushes in exactly the direction the
reading claims.** The model's 421–456 parameters are held **stationary** over 336 months while the
truth's 336 months contain **28 years of real conflict relocation**. Nonstationarity inflates the
truth's support, inflates the cumulative shortfall, and therefore *manufactures* apparent
over-persistence. Match the windows and it **flips**:

| both sides over 36 months, 421–456 | model | truth | shortfall |
|---|---|---|---|
| monthly `P(y>0)` | 0.000841 | 0.003518 | **4.18×** |
| cumulative support | 0.006362 | 0.022077 | **3.47×** |
| ratio cumulative / monthly | | | **0.829** |

> ⛔ **AND THIS CONCLUSION IS ALSO WITHDRAWN — I read 0.829 against an implicit null of 1.0, and the
> correct null is nowhere near 1.** `1 − (1 − P)^T` is **concave and saturating** in `P`, so
> support-per-unit-firing falls as firing rises **at fixed spatial arrangement**. The two rows sit
> **4.18× apart in firing**, so part of the truth's lower support-per-month is the *map*, not the
> arrangement. The `opponent` seat caught this; §10.9 settles it without a null at all.

⚠️ **What survives D8 unconditionally:** on the matched window **the model under-fires by 4.18× per
cell-month and touches 3.47× too few cells.** Both are direct counts and neither depends on a null.
**Not quotable:** 1.853, 0.829, and any reading of either. 36 months is the model's own fitted
window, so the truth there is **in-sample** — that favours the model and is the price of removing the
relocation confound. ⚠️ The "event-months per ever-active cell" line in the script (truth 12.035,
model 22.300) is **the same quantity rearranged, not an independent check** — it returns 1.853
identically, verifying the algebra and nothing else; the script's original comment claiming otherwise
is corrected in place.

⛔ **D8-A is the sixth instance of C-351** (population slot — the rows covered different windows) and
⛔ **D8-B is the seventh** (**transform** slot — a nonlinear map applied to two rows at operating
points 4.18× apart, where the nonlinearity alone moves the statistic from 1.0 to 0.19–0.46). **The
slot added to the convention yesterday caught the very next measurement made after it was added.**
That is the first evidence the third slot is load-bearing rather than a tidy completion of a triple.

### 10.9 D9 — the sign, settled without a null, and it is over-concentration

`d9_concentration.py`. "Arrangement" is a property of the **shape** of the per-cell firing field, so
measure the shape. Both statistics below are **normalised by the total**, so the 4.18× level
difference **cancels exactly** — no saturating map, no synthetic field, no parameterisation choice,
no null.

| | model | truth (36 mo) | model/truth |
|---|---|---|---|
| share of firing mass in the top **0.1%** of cells | **0.4578** | 0.2455 | **1.87×** |
| share in the top 1% | 0.9860 | 0.8706 | 1.13× |
| Gini of the per-cell rate | 0.9963 | 0.9910 | 1.005 |
| **effective cells** (participation ratio `(Σp)²/Σp²`) | **215.0** | **516.9** | **0.416** |

⇒ ⛔ **The model's firing mass is MORE concentrated than the truth's, and the seat's direction is
right.** The model packs what little it fires into **215 effective cells against the truth's 517** —
**2.4× fewer places** — and puts **1.87×** as much of its mass in the top 0.1% of cells.

✅ **And it is robust across every truth window, which is the part the seat could not get from a
synthetic field:**

| truth window | 36 mo | 72 mo | 156 mo | 336 mo | **model** |
|---|---|---|---|---|---|
| Gini | 0.9910 | 0.9902 | 0.9877 | 0.9824 | **0.9963** |
| effective cells | 516.9 | 518.8 | 585.1 | 819.1 | **215.0** |
| top 1% share | 0.8706 | 0.8515 | 0.7992 | 0.7072 | **0.9860** |

The truth is **less** concentrated than the model at **every** window, and lengthening the window only
widens the gap. ⚠️ A longer window is less noisy *and* spans more regime change, and those push the
truth's concentration in opposite directions, so this is a **sensitivity sweep, not a clean bound** —
but the sign does not depend on the window anywhere in it.

**The seat's nulls, recomputed on the real field rather than a synthetic one** (its own suggestion):
the multiplicative null is **0.3910** exactly (its synthetic gave 0.4640). ⛔ The additive-logit null
is **not exactly computable on the real field** — **97.79%** of cells have an estimated rate of
exactly 0 and `logit(0) = −∞` — and the atom rule dominates it: `(k+0.5)/(T+1)` gives **0.2238** and
`(k+1)/(T+2)` gives **0.1857**, against the synthetic's 0.7709. ⚠️ **I therefore do not count the
logit nulls as corroboration** — they are an artifact of the atom choice, which is the transform slot
biting *inside the fix for a transform error*. The observed 0.829 sits above every null computed, but
**D9-B is the finding and D9-A's multiplicative null is its only trustworthy corroboration.**

### 10.10 The synthesis, corrected

⛔ **An earlier version of §10.8 said D8 "agrees with §10.3" because both showed the model spreading.
That reasoning was wrong even though the conclusion of agreement survives.** The two findings fit
together differently, and better:

> **The model emits a few large smooth blobs where the truth has many small sharp spots.**

Smoothness is a *local-gradient* property and concentration is a *global-distribution* property, so
there is no contradiction in being both: a single smooth blob is smooth **and** concentrated. §10.3
measures the local half (fields smoother than the variance decomposition requires, by 0.06) and §10.9
measures the global half (2.4× too few effective places). ⇒ **The latent risk field is roughly
right; the point process on top of it is not** — which is C-152/EXP-3's *"89% of the damage is cells
in the wrong places"* and is the shape Simon proposed from the architecture before any of it was
measured. **And concentration is a joint property, so this is the first direct quantification of
what C-152's class statement asserts.**

---

## Appendix A — C-351, drafted and ⛔ NOT REGISTERED, with the reason

⛔ **This entry is not in `technical_risk_register.md` and must not be assumed to be.** C-350 was
committed on `chore/register-c350-static-provenance` (**PR #419, open, unmerged**) and does **not**
exist on this branch — measured: `grep -c "C-350"` returns **0** here. The register is therefore
already forked across two branches, and its header carries hand-maintained counts plus a single
enormous `ID accounting` cell. Adding C-351 here would fork it a **third** way and guarantee a
conflict in exactly the field that records why the counts are what they are.

**So it is drafted here and registered after #419 merges.** `register-risk` is the intake; this is the
finding handed to it, not a bypass of it.

| field | value |
|---|---|
| **ID** | C-351 |
| **Tier** | **2** — not Tier 1: it corrupts *conclusions*, not data or forecasts, and every instance so far was caught. Not Tier 3: **seven** instances in one day, three inside one report section, and two of them reached a committed artifact. |
| **Source** | the D3/D4/D5/D6/D7/D8/D9 measurement pass, 2026-10-20, plus the `opponent` seat's §13/§14/§15 |
| **Trigger** | **The next time a per-cell or per-block statistic is computed on this grid and reported as a property of the model** — specifically: any slope, calibration, correlation or dispersion number computed on a cell set chosen by activity (`active_cells_top100`, `tm > 0`, `y > 0`), on a window that overlaps its own readout, or comparing a model field against a truth field through a non-affine transform. |
| **Location** | `reports/2026-10-12_underfire_diagnosis/vehicle.py:242` (`active_cells_top100`), `d4_mu_on_the_grid.py` (the `keep = tm > 0` coarsening and the lag-1 comparison), `d3_ceiling.py` (the first window choice), and §2/§3/§4/§8/§9.2/§9.3 of this report |

**Narrative.** Seven findings in one day were produced, reported or nearly reported from *a statistic
computed on a set, or a scale, defined by the thing being measured*:

| # | where | the artifact | what it did to the number |
|---|---|---|---|
| 1 | §2 | 98.6% of land cells tie at an observed rate of exactly 0 | made a channel out-rank the gate, 0.51 vs 0.20; reversed on meaningful cells |
| 2 | §3 | the channel's window overlapped its own readout (301–456 vs 421–456) | R² **0.3802 → 0.1016** |
| 3 | §9.1 | `active_cells_top100` selects the top 100 cells **by observed rate** | `b` **0.92 → 1.7301**, and it was the report's title |
| 4 | §10.3 | truth has an atom of 0.986 at zero; `log`/`logit` are not affine | **0.154** of a 0.22 gap, ~¾ of the effect |
| 5 | §10.4 | the coarsening admits only blocks with an event; admitted fraction **2.2% → 22.4%** | reversed the sign of the amplitude trend |
| 6 | §10.8 | the model's window (36 months, stationary) vs the truth's (336 months, with regime change) | reversed the sign of the persistence reading, 1.853 → **0.829** |
| 7 | §10.8 | `1−(1−P)^T` is concave, applied to two rows **4.18× apart in firing** | implicit null 1.0 against a true null of **0.19–0.46**; reversed the sign again |

**Three of the seven are in `§9`, which was itself written to correct `§4`.** Each was caught by a
*different* mechanism — restriction to meaningful cells, a window audit, a full-grid forward pass, an
adversarial peer, a nested-population recomputation, a window match, and a null-free re-measurement — and **no single habit would
have caught more than two.** That is the risk: the class is not one mistake with one remedy, and on this grid the
tempting set is *always* the active one, because 98.6% of cells carry no signal.

⛔ **Why a guard in the measurement code is the wrong remedy** (and the reason this is a register
entry rather than a patch): these are one-off diagnostic scripts, not production, and
`~/.claude/rules/design-preferences.md` prefers the bounded local change over a permanent guard in a
shared file. A `utils` validator asserting "your cell set was not chosen by the outcome" cannot be
written — selection is sometimes correct and sometimes fatal, and only the claim decides which.

**What is proposed instead, and it is cheap:** a reporting convention with **three** slots.

⛔ **The two-slot version failed on its own fifth instance, and the `opponent` seat found the hole.**
The first draft of this entry asked only for *the population and how it was chosen*. Written to that
rule, §9.3's truth row reads:

> *"lag-1 autocorrelation 0.6419 on all 64,818 land cells, population not selected."*

**Every word true, the population statement complete — and the number still wrong by 0.154**, because
the defect was the **scale**, not the set. Population discipline cannot catch a transform error.

| slot | states | the instance it catches |
|---|---|---|
| **population** | the set it was computed on **and the rule that chose it**, including the window | #1 ties, #2 window overlap, #3 selection, #5 admission rate, and D8-A's window mismatch (§10.8) |
| **transform** | the scale the statistic lives on, and whether the things being compared share it | **#4** — `log mu`, `logit p` and a raw rate with an atom at 0.986 are three different objects. ✅ **And #7**, a concave map read at two operating points 4.18× apart — made with this slot already in the convention, which is why it was caught in hours rather than days |
| **estimator** | how it was computed, and whether both sides went through the same one | #4 again, from the other side; it is what D6-C fixes |

**The count of catching mechanisms was one higher than the count of slots being guarded** — which is
the entry's own argument turned on the entry. *"b = 1.73 on 100 cells chosen by observed positive
rate, OLS of log truth on log prediction"* is self-refuting where *"b = 1.73"* is not. §10 is written
that way throughout — every row names its `n`, its condition and its estimator.

**Cross-refs.** **C-307** (a cheap screen's null is inconclusive, not negative) — the sibling about
how a *result* is read, where this is about how a *statistic* is built. **C-320** (a falsifier band
tighter than its own reference's noise) — same family, as instance 4 shows, where the comparator
rather than the set is ill-posed. **C-152/EXP-3** — the substantive finding these statistics were
chasing. **C-325** — why D5-A was pre-registered VOID rather than read.
