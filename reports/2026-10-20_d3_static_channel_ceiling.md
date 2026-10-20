# D3 — the ceiling on a static base-rate channel, and what it found instead

**2026-10-20.** Free measurement, no pod, no training, nothing fetched. Run on artifacts already
on disk. Scripts: `2026-10-12_underfire_diagnosis/d3_ceiling.py`, `d3_followup.py`, `d3_clean.py`,
`slope_convention.py`.

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

⛔ **Both halves of the model are under-dispersed, and the body looks worse than the gate.** That is
one defect wearing two coats, not two findings.

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
