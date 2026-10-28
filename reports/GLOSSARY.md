# LOCKED VOCABULARY — the only words I use

**Rule:** every concept has ONE name (left column). I use only that name. The "banned aliases" are words I
used before for the same thing — listed so you can map old messages onto the locked name. I never introduce
a new synonym. If I drift, you type **"drift"** and I stop and correct.

Last rebuilt: 2026-07-17 (v2, complete). **Amended 2026-10-16:** disambiguated the two "all"s in §3b (`body_supervision='all'` vs `body_nll_denominator='all_cells'`). **Amended 2026-10-15:** added **§3b the task-loss balance** (*the balancer*, *task-loss term*, *frozen balancer*, *the NLL denominator*, *truncated NB loss*) and scoped the stale `TruncatedNB` banned alias in §3. **Amended 2026-08-15:** added *the feedback realism gap* (the parent cause) and *the zero collapse* (its under-firing child); *the bloom* is now defined as the over-firing child of the same cause.

---

## 1. The parts of the model
The forecast for each cell each month = **gate × body** — except a **self-zeroed** count body (§2b),
which produces its own zeros and needs no gate.

| locked name | what it is | banned aliases (I will NOT use these) |
|---|---|---|
| **gate** | the part that predicts *whether* a cell has any conflict (a probability) | switch, onset, onset gate, classifier, the on/off |
| **body** | the part that predicts *how many* deaths | size-guesser, reg, regression head, magnitude head |
| **gated forecast** | the one forecast we use: **gate × body** | hurdle compose, hurdle_shrinkage, dense forecast, standard output |

The **body itself has two parts** (locked):

| locked name | what it is |
|---|---|
| **bulk** | the *typical* conflict magnitudes — the bottom ~97–99% of conflict cells |
| **tail** | the *extreme* conflict magnitudes — the top ~1–3% of conflict cells (the WALL lives here) |

## 1b. The data structures — input vs output (LOCKED)
Two big multi-dimensional arrays, deliberately named differently so input and output never blur. "tensor"
is BANNED as a name for either (everything is a torch tensor — it says nothing about *which*); use it only
generically ("a tensor", never "the tensor").

| locked name | what it is | banned aliases |
|---|---|---|
| **volume** | the **INPUT** spatiotemporal array fed to the model (features × grid × time), and its handling (`VolumeHandler`/`VolumeSampler`) | input cube, input tensor, the tensor |
| **sample cube** (or **cube**) | the **OUTPUT** posterior samples the model emits — `[T, H, W, n_reg, S]`, S = D×K draws per cell; what the ruler scores (`to_cube_samples`, `assert_cube_fits`, `max_posterior_cube_gb`) | output volume, prediction volume, output tensor, the tensor |

Mnemonic: **input = volume, output = (sample) cube.** Never call the output a "volume" (collides with the
input) and never call either "the tensor". Prefer the qualified "**sample cube**" in prose; bare "cube" is
fine once context is set.

## 2. How the body is TRAINED (a separate choice from §1)
| locked name | what it means | banned aliases |
|---|---|---|
| **all-cell** | body trained on *every* cell (including the empty ones) | dense, standard, no mask |
| **positives-only** | body trained *only* on cells that actually had conflict | hurdle, hurdle_shrinkage, hurdle-masked, masked body |
| **supervision window** | the temporal span around conflict where the **body** loss is supervised (ADR-065 amend. 2026-07-28): config `body_supervision ∈ {all, active}` + `onset_lead` (months before onset) + `cessation_lag` (months after cessation). `all` = all-cell; `active,0,0` = positives-only (per-step); `active,W,W` = full active-cell timelines. Sweep the radii; the body owns the boundary dynamics, the gate owns the deep structural zeros. | ⛔ **`body_mask` (RETIRED)**, ⛔ **`pos_cells` (→ `active,0,0`)**, ⛔ **`pos_timelines` (→ `active,W,W`)**, `active_window`, mask, hurdle_mask_mode |

Note: I previously blurred "positives-only" (a *training* choice) with "gated" (the *compose*). They are
separate. Until now the forecast has always been gated (gate × body), with the body trained either
**all-cell** or **positives-only**; the distributional-head work adds **self-zeroed** count bodies as the
alternative to a separate gate — see §2b.

## 2b. Where the zeros come from (a separate choice again)
Two ways a cell's predicted zero can arise:

| locked name | what it is | banned aliases |
|---|---|---|
| **gated zeros** | the zeros come from the separate **gate** (gate × body) — every model so far | use_gate, gate-on, hurdle |
| **self-zeroed** | **no separate gate** — the body's own output covers the zeros across all cells. A count body (NB/ZINB) puts probability mass on 0; a plain point body just regresses toward 0 (the default `standard`). | dense_nb, standard ("no gate"), zero_handling=none, use_gate=False |

## 2bb. The emit-time split (which family we DRAW from, vs which we TRAINED on) — ADR-073

Until 2026-10-21 these were one setting. `output_distribution` chose the family for **both** the
training loss and the emitted draw, so a body trained one way could only be drawn the same way.

| locked name | what it is | banned aliases |
|---|---|---|
| **the emit-time split** | training the body on one distribution family and **emitting** it from another: `output_distribution` is the family it was TRAINED on, `emit_distribution` the family it is DRAWN from. `None` (the default) means they are the same, which is every model to date. The pair it exists for is `nb` trained, `truncated_nb` emitted — the gate then owns occurrence and the body owns only magnitude. ⛔ The emitting family must declare the same `n_params`, because the head's channels mean what the trained family says they mean | emit family, emission family, train/emit decouple, dual family, `emit_family_core` (⛔ that is **§ADR-068** and is a *different* axis: which part of ONE family to emit) |

⚠️ **`gated forecast = gate × body` is unchanged by this** (§1). The split changes *what the body
is*, never how the gate and body combine — that is §2c.

⛔ **The TERM stays locked; the IDEA was rejected on 2026-10-22.** We still need the name to talk
about the axis, and `emit_distribution` still exists as a default-off config field — so keep using
this vocabulary. But **do not describe the emit-time split as a route to anything.** It is strictly
dominated: it inverts the emitted magnitude field's rank correlation with the gate (+0.49 → −0.16)
and pays 13.5× on the empty background to buy an event-cell gain that a family-free recalibration
takes outright at no background cost (M67; ADR-073 §0). ⇒ A sentence using this term should say
what was *measured*, never that it is the plan.

## 2c. The forecast-composition arms (how a trained gate + body become ONE scored forecast)
An **arm** is a specific rule for turning a trained **body** and the **gate** into the single scored
forecast. It is a *compose-time / score-time* choice, layered on top of the §2/§3 training choices — the
same trained heads can be composed several ways. Two independent knobs define an arm:

- **body source** — *which model's* parameters we use for the body: the plain **NB** `(μ,θ)`, the **ZINB**
  `(μ,θ,π)` (which additionally learns a structural zero parameter **π**), and later **gamma** / **lognormal** /
  a zero-inflated continuous. Same *form* of body, but trained under a different objective ⇒ different `μ`.
- **occurrence rule** — *how the zeros are produced*: **self** (a self-zeroed distribution's own π, no
  external gate — §2b), **soft gate** (prefix `gated_` — per draw `Bernoulli(gate) × body`), or **threshold
  gate** (prefix `th_gated_` — keep the *full* body where `gate ≥ τ`, zero it where `gate < τ`, for a **fixed
  a-priori** probability **τ**).

> ⛔ **HARD LOCK — `th_gated_NB`, NOT `masked_NB`.** The hard-threshold arm is **`th_gated_NB`**. The name
> **`masked_NB` is RETIRED** (ADR-068) and must **never** appear as a live term — not in code, configs, plot
> labels, scoring templates, drivers, dossier prose, or commit messages. It is permitted in exactly two
> places: (1) this glossary's *banned-aliases* column, and (2) ADR-068, each time explicitly labelled
> "RETIRED". Any other occurrence is **drift** — correct it on sight. *(This lock exists because the retired
> name resurfaced on 2026-07-24 and had to be scrubbed — `f0ac6cc`.)* The same `th_gated_<body>` rule holds
> for every body: `th_gated_gamma`, `th_gated_lognormal`, `th_gated_ZINBcore` — there is **no** `masked_*`
> spelling of any of them.

**`core` — the one tricky word (LOCKED, read this):** the positive body of a *zero-inflated* distribution
with its structural **π removed**, so it can be composed with an *external* gate instead of self-zeroing.
- `core` appears **only** on a body taken from a ZI model — e.g. **`ZINBcore`** = the `NB(μ,θ)` *inside* a
  trained ZINB, with π dropped. A body that has **no** structural π (NB, gamma, lognormal) **never** carries
  `core` — there is nothing to strip.
- So the **presence** of `core` is a signal: "a π was stripped here." Its **absence** means "no
  zero-inflation model was involved" — it does **NOT** mean "the full / non-core body." (This is the
  asymmetry to not misread: `gated_NB` has no `core` because NB has no π, not because it uses a fuller body.)
- ⚠️ **Never re-apply a core's own π:** `(1−π)μ × gate` **double-counts** the zeros (π zeros once, the gate
  again). `core` means π is *gone*; the gate is then the *only* zeroing mechanism.

**Naming pattern:** `[th_]gated_<bodymodel>[core]` for a *composed* forecast (`core` iff a ZI model's π was
stripped); the **bare distribution name** (`ZINB`, `ZIgamma`) for a *self-zeroed standalone*.

| locked name | the EXACT forecast it scores | banned aliases |
|---|---|---|
| **ZINB** | the self-zeroed standalone (the distribution of §3): NB body + structural-π zero spike; forecast `E[y]=(1−π)μ`, sampled with π-masking, **no external gate** | zero-inflated-NB-as-an-arm, gated ZINB, ZINB×gate |
| **gated_NB** | **soft**: per draw `Bernoulli(gate) × NB(μ,θ)`, with `(μ,θ)` from the **nb** model | hurdle_nb, hurdle, hurdle_shrinkage, gate×body |
| **th_gated_NB** | **hard**: full `NB(μ,θ)` body where `gate ≥ τ`, zeroed where `gate < τ`; `(μ,θ)` from the **nb** model; τ fixed a-priori | ⛔ **`masked_NB` (RETIRED — never live, see HARD LOCK above)**, masked_nb, masked, thresholded NB, hard_gated_NB |
| **gated_ZINBcore** | **soft**: per draw `Bernoulli(gate) × NB(μ,θ)`, with `(μ,θ)` from the **zinb** model, **π dropped** | gated ZINB, ZINB×gate, hurdle_zinb, gated_zinb |
| **gated_truncNB** | **soft**: per draw `Bernoulli(gate) × TruncNB(μ,θ)` — a **zero-truncated** NB body, so `P(body>0)=1` by construction and the emitted occurrence **is** the gate; `output_distribution='truncated_nb'` + `forecast_composition='soft_gate'` | soft_gate truncated_nb, truncated hurdle, hurdle_nb, gated_TNB, soft-gated truncated NB |

**Why `gated_NB` vs `gated_ZINBcore` is a real distinction, not a typo:** both gate a bare `NB(μ,θ)` core; the
*only* difference is the training that produced `(μ,θ)` — plain-NB (μ zero-diluted → **timid**) vs ZINB (π
absorbed the zeros in training → μ is the *conditional* magnitude, un-timid). `core` flags that the ZINB one
had a π we removed; the NB one never had one.

**Extending to new bodies (gamma / lognormal / a future ZI-continuous) — no new convention:** same two knobs,
same `core` rule. Continuous bodies (gamma, lognormal) have **no** zero mass, so they *must* be gated (there
is no self-zeroed standalone unless the distribution is ZI-wrapped):

| body model | self-zeroed standalone | soft-gated | threshold-gated |
|---|---|---|---|
| NB | — (NB is not self-zeroed) | `gated_NB` | `th_gated_NB` |
| ZINB | `ZINB` | `gated_ZINBcore` | `th_gated_ZINBcore` |
| truncated NB | ⛔ impossible (`P(Y=0)=0` by construction — no zero mechanism; ADR-069 refuses `self_zeroed`) | `gated_truncNB` | ⚠️ `th_gated_truncNB` — **degenerate**, see below |
| gamma | — (no zero mass) | `gated_gamma` | `th_gated_gamma` |
| lognormal | — (no zero mass) | `gated_lognormal` | `th_gated_lognormal` |
| ZI-gamma *(if ever built)* | `ZIgamma` | `gated_ZIgammacore` | `th_gated_ZIgammacore` |

**Why `th_gated_truncNB` is named but degenerate (measured, 2026-10-12):** a zero-truncated body has no zero
mass of its own, so under `threshold_gate` the *only* zeros come from cells below τ — every kept cell is
positive in every draw. Recomposing the delivered model's own output this way emits **`P(Y=0) = 0.0000`**
against an observed 0.2361 (M1). The pairing is therefore a named dead end, not an arm: a zero-truncated body
needs a **soft** gate, which is what `gated_truncNB` is. Equally, `gated_truncNB` makes the gate's
*calibration* load-bearing where `th_gated_NB` needed only its *ranking* — the emitted occurrence is the gate
itself.

## 3. The losses (how the body is scored during training)
| locked name | what it does | banned aliases |
|---|---|---|
| **MSE** | squared error; fits the **mean** | mean loss, squared loss |
| **MAE** | absolute error; fits the **median**. **Identical to "pinball at 0.5".** | pinball τ0.5, pinball-at-0.5, median loss, "no dial" |
| **the dial** | pinball loss at a level above 0.5 (0.6/0.7/0.8) — pushes guesses UP. | pinball, tau/τ, PinballBodyLoss, lifter |
| **count_mean** | MSE computed in count space (not log space) — the one that blows up | count-space MSE |
| **NB loss** | the negative-binomial likelihood body (the old floor) | hurdle_nb; ⚠️ **`TruncatedNB`** — banned ONLY as a name for *this* row (it meant the discredited legacy `truncated_nb_loss.py`). The live zero-truncated family is **truncated NB loss**, §3b. |
| **ZINB** | an **NB** body with an explicit extra spike of probability at exactly 0, sized to the data; always **self-zeroed** (§2b) | zero-inflated NB, zero_inflation |

"No dial" was never a loss — it meant "a plain loss (MSE or MAE), not the dial." I'll say **"plain MSE"** or
**"plain MAE"**, never "no dial".

**Additive terms (added AFTER the balancer, each with its own plain weight; byte-identical no-op at 0):**
| locked name | what it does | banned aliases |
|---|---|---|
| **the structure penalty** | ⭐ **LOCKED 2026-10-21.** A term penalising the mismatch between the *within-block dispersion* of the composed field and of the truth, summed over block sizes `k`. Weight `struct_penalty_weight`. ⛔ The gate is **detached** inside it, so gradients reach the **body only**. ⚠️ **NOT a proper scoring rule** — a `k×k` block sum of per-cell-`theta` NBs has no closed form, so this is moment-matching, in Mathieu's gradient-difference family. Pre-registered in `2026-10-21_structure_penalty_arm_prereg.md` | spatial loss, sharpness loss, multi-scale loss, GDL, block loss |
| **the qs99 regulariser** | the existing distribution-free pinball term on `mu`, weight `qs99_weight` | qs99 loss |
| **the decay penalty** | the existing gate-side term on recently-active-now-zero cells, weight `decay_gate_weight` | decay loss |
| **the parameter-prior ridge** | the existing family π/μ ridge, weight `pi_penalty_weight` | pi penalty, π-ridge |

**The gate's loss + knob:**
| locked name | what it does | banned aliases |
|---|---|---|
| **weighted_bce** | the gate's loss; **pos_weight** is its one knob | wBCE, BCE, weighted BCE |
| **pos_weight** | the gate's eagerness knob (higher = fires more). Foundation: 2 best-calibrated of {2,3,4,5} | pw, gate weight |
| **focal** | an alternative gate loss; knobs = **gamma** (focusing) + **alpha** (class weight) | focal loss |
| **per-target pos_weight** | letting sb / ns / os each have their own pos_weight (vs one shared) | — |

## 3b. The task-loss balance (how the SIX task losses become ONE number)

Six task losses are computed per lesson — a body loss and a gate loss for each of sb / ns / os — and
something must combine them into the scalar that is backpropagated. That combiner is **the
balancer**, and until 2026-10 we had no words for any of this.

| locked name | what it means | banned aliases |
|---|---|---|
| **the balancer** | `MultiTaskLoss` (`utils/mtloss.py`) — the combiner that turns the six **task-loss terms** into the one scalar. Implements Kendall/Gal/Cipolla homoscedastic-uncertainty weighting: a learnable `log_vars` per task, `coeff_j = 1/((is_regression_j + 1)·e^{v_j})`, plus a `v_j/2` regulariser. | MTL, multi-task loss, the weighting, mtloss |
| **task-loss term** | the value ONE task contributes, **after** the balancer's coefficient — `coeff_j · L_j + v_j/2`. Distinct from the raw per-task loss `L_j`. | task loss, loss component, the weighted loss |
| **frozen balancer** | `freeze_multitask_balancer: True` — `log_vars` are excluded from the optimizer and so stay at their **zero initialisation**, never learned. The objective then collapses to the **constant form** `total = 0.5·Σbody + 1.0·Σgate`. ⛔ Not "Kendall's weights frozen" — they were never learned; this is plain **uniform weighting**, and a regression task structurally cannot be weighted above 0.5. Every views-models config pins it `True`; the config **default is `False`**. | frozen weights, equal weighting (ambiguous — say *the constant form*), inert balancer, learned-then-frozen (⛔ false) |
| **the NLL denominator** | what a body NLL's per-cell losses are **divided by** — a separate axis from the **supervision window** (§2, which cells are supervised at all). Two values: **positives** (the count of `y>0` cells) and **all cells** (every cell in the window). Set by `weighted_nll_mean`'s `weight` argument (`distributions/nb_core.py:62-79`): `weight=None` ⇒ all cells; the `y>0` mask ⇒ positives. ⛔ It is today an undeclared consequence of `output_distribution` — `nb` passes `None`, `truncated_nb` passes the mask — so swapping the family silently swaps the denominator and rescales that **task-loss term** by ~1/(positive fraction). | reduction, the mean, normalisation, dense/sparse reduction |
| **truncated NB loss** | the **zero-truncated** NB likelihood body — NB renormalised by `1 − NB(0)`, so `P(Y=0) = 0` by construction. `output_distribution='truncated_nb'`. Composed only as **`gated_truncNB`** (§2c). ⚠️ Distinct from the discredited legacy `utils/truncated_nb_loss.py`; when that file is meant, say **the legacy truncated loss**. | TruncNB, zero-truncated NB, hurdle body, truncated_nb_loss (⇒ say *the legacy truncated loss*) |

⛔ **Two different "all"s — do not conflate them.** `body_supervision='all'` (§2, the locked
**all-cell** supervision window) says *which cells are supervised at all*. `body_nll_denominator='all_cells'`
(§3b) says *what the supervised cells' losses are divided by*. They are different axes on different
objects, and the near-identical spelling is a real trap. The relationship is one-directional and
enforced in code: **`all_cells` REQUIRES `body_supervision='all'`**, because under `'active'` the
supervision mask is itself a positive mask, so the denominator collapses back to the positive count
and the flag becomes bit-identical to `positives` (C-331). The reverse does not hold —
`body_supervision='all'` says nothing about the denominator, and the default denominator stays
`positives`.

**Why these words had to exist (measured 2026-10-13/15,
`reports/2026-10-13_composition_arm_outcome/`):** under the **frozen balancer** nothing reads the
**task-loss terms**, so there is no feedback path that can notice one of them changing scale. The
`gated_truncNB` arm changed the **NLL denominator** from **all cells** to **positives** as a
side-effect of changing the family, multiplying the body term by ~222 and making it **35.2×** the
gate term where the control's was **0.6×** — on a **shared trunk**, which is what makes the term
scale the allocation rule for shared capacity. No config-identity check could see it.

⛔ **A false claim to not repeat.** `utils/weighted_bce_loss.py:6-7`,
`utils/truncated_nb_loss.py:4-6` and views-models' `config_hyperparameters.py:43` all assert that
the gate and body NLLs' *"equal-weight sum is the joint hurdle-NB NLL"*. That is false on three
counts: the denominators differ (**positives** vs **all cells**), the weights are 0.5 and 1.0 rather
than equal, and `pos_weight = 2.0` makes the gate term a weighted rather than a proper Bernoulli
NLL. Under **Cragg (1971)** separability a constant on one part cannot move that part's own MLE — but
that holds only for **independently parameterised** parts, and our shared trunk is not that.

## 4. The scores (how we judge a finished model)
| locked name | what it measures | banned aliases |
|---|---|---|
| **AP** | gate quality: area under the precision–recall curve (higher better) | average precision |
| **Brier** | gate quality: probability accuracy (lower better) | switch score |
| **crps-all** | body quality: CRPS over *every* cell (lower better) | all-CRPS |
| **crps-events** | body quality: CRPS over cells that truly had conflict (lower better) | event-CRPS, pos-CRPS |
| **crps-none** | body quality: CRPS over cells that were truly zero (lower better) | zero-CRPS, neg-CRPS |
| **size-ratio** | **median** of (guess ÷ truth) on conflict cells; 1.0 = right-sized, <1 = too low | ratio_med |
| **pos-mcr** | **mean** of (guess ÷ truth) on conflict cells. NOTE: mean ≠ median; a few big cells can flatter it (why we once distrusted it). Different metric from size-ratio. | MCR, MCR_pos, mcr_pos |
| **pos-mcr**, continued | ⛔⛔ **ADDED 2026-10-28 (M5 / #482): `MCR` stays a BANNED alias here, and the reason is now measured.** `pos-mcr` is a **mean of per-cell ratios** over conflict cells; FAO-02's MCR is a **ratio of means** over **all** cell-months (next row). They are not the same statistic and neither bounds the other. ⭐ M3's I-6 measured the gap's mechanism: `pos-mcr`'s per-cell weight is **exactly `1/truth`**, confirmed to **7.6e-17** against a closed form, so the **same 100 units of mass move it 115.3× more when added to the smallest truths**. ⇒ `pos-mcr` is a statement about **one-event cells**; FAO-02's MCR is a statement about the **total**. ⛔ **A `pos_mcr` number does NOT discharge the MCR guardrail.** | — |
| **MCR** | **FAO-02's magnitude-calibration guardrail, LOCKED**: `MCR = ȳ̂ / ȳ` — the **ratio of mean predicted fatalities to mean observed fatalities, across ALL PRIO-GRID cell–month observations in the evaluation period** (Pre-Release Note 05, Appendix A.1, read 2026-10-28 — not quoted from memory). 1 = perfect aggregate calibration, <1 under-prediction, >1 over-prediction; performance is judged as **distance from 1**, non-inferiority margin **1%**. ⛔⛔ **The frozen ruler does NOT compute it.** `lodestar_score.py` emits `pos_mcr` (a mean of per-cell ratios over conflict cells) and `size_ratio` (the median of the same ratios) — **neither is this statistic**. `scripts/sharpness_scorecard.py` computes a **third** thing, a ratio of means restricted to `truth>0`. The red-team's `MCR_field` (M2/#479) is the FAO-02 form. ⚠️ `reports/2026-07-16_bulk_calibration_dossier/tools/bulk_score.py` **bans MCR by name** (*"the pooled-mean/MCR cancellation cannot occur; MCR is BANNED"*) — correctly, for its own per-cell question. Both are right about different questions; the error is letting one stand in for the other. | mcr (bare, for pos-mcr), MCR_pos, mcr_pos, magnitude calibration ratio (when the pooled form is not meant) |
| **QS99** | **FAO-02's tail-sanity guardrail, LOCKED**: `QS_0.99(q, y) = (1{y < q} − 0.99)·(q − y)`, i.e. the pinball loss at τ = 0.99 against the forecast's **predicted 99th percentile** `q`, computed at **every** cell–month (Pre-Release Note 05, Appendix A.1, read 2026-10-28). Expanded: `max(0.99·(y − q), 0.01·(q − y))`. Non-inferiority margin **1%**; chosen over QS95 as more extreme-sensitive and less correlated with CRPS. ⛔⛔ **The frozen ruler does NOT compute it** — `lodestar_score.py`'s output row has no QS99 column, confirmed by reading all 277 lines. ⭐ It IS computed, under this exact formula, in **three** places: `reports/2026-07-15_quantile_head_build_dossier/tools/t0_score.py:59-60`, `reports/2026-07-16_bulk_calibration_dossier/tools/bulk_score.py:111-113`, and the red-team `score()` of M2/M3. ⇒ **one definition, three implementations — a coverage gap, not a definitional conflict.** ⛔ **NOT the same thing as `qs99_weight`/`qs99_tau`** (`training_engine.py:958-965`), which is a **training regulariser**: the same pinball formula applied to **`mu`** rather than to a predicted 99th percentile, on **positive-masked cells only**. Same name, different quantity (§3's *the qs99 regulariser*). ⚠️ `scripts/tail_scorecard.py`'s `pin90` is τ = 0.9 and says so. | quantile score, QS, tail score, qs99 (when the training regulariser is meant) |
| **FSS** | **Fractions Skill Score — how well the forecast puts events in the right PLACES, at a stated spatial scale.** Both fields are binarised, each is convolved with an `n × n` uniform filter to give a local *fraction*, and the score is `1 − FBS / FBS_worst` on those fractions. ⛔ Here it is always **count-matched** (`matched_pred_thresh` picks the forecast threshold that makes the forecast's event count equal the truth's) and **pooled** — FBS summed over the window's monthly fields *before* the ratio. ⇒ It is a function of **rank order alone**: measured blind to magnitude at `0.000e+00`, and it saturates after one box-smoothing pass. It is therefore **B-1b**, the placement half of the pair, and says nothing about dispersion. ⛔ `n` is quoted with every value; the headline scalar is **`FSS(n=1)`**. | fractions score, FSS score, fraction skill, fractional skill score |
| **hit rate** | **The share of truly-active cells the forecast also marks active**, `H / E`. ⭐ **Identically equal to `FSS(n=1)`** under count-matching — derived, then measured to one ulp (`1.110e-16`) across 49 fields (`R-27`): at `n = 1` the filter is the identity and count-matching forces `sum(prd) = sum(obs) = E`, so FSS collapses to `H / E`. ⇒ The two names are **the same number**. Either may be used; ⛔ **never both in one comparison as if they were two pieces of evidence.** ⛔⛔ **PRECONDITION, added 2026-10-28 (M2 / #479, C-388): the identity holds only where count-matching actually works.** `matched_pred_thresh` silently returns **−1e-12** — selecting every cell — for any forecast with fewer nonzero cells than the truth has events, and there the identity fails by ~1.0: a forecast predicting **nothing** scores `hit_rate` **1.000000** against `FSS(1)` 0.001723. ⚠️ All 49 fields behind `R-27` satisfied the precondition, so the measurement is real and conditional. ⛔⛔ **AMENDED 2026-10-28 (M3 / #480): count-matching holds in only 75 of 143 measured forecast-months, and it fails in THREE ways** — (a) a zero k-th prediction ⇒ every cell selected; (b) a **TIED** k-th value, every duplicate clearing `kth − 1e-12`, which is **structural** because an ensemble mean of S integer draws is quantised to multiples of 1/S; (c) the `−1e-12` is **ABSOLUTE** on a field with no absolute scale. ⛔ **The delivered model is affected by (b) in 2 of 13 months**, so "sparser than the truth" is not the whole precondition. ⇒ **Never quote either name without reporting the realised event count against `n_events`.** | POD, probability of detection, recall, detection rate |
| **AFSS** | **The value FSS tends to at the largest scale**, Roberts & Lean Eq. (8), reached at `n = 2N − 1`. ⛔ **Under count-matching it is exactly `1.000000`**, because count-matching forces `f_M = f_o` — so **AFSS carries ZERO bias information here** and is a *control*, never a finding (`P-F3`). ⛔⛔ **PRECONDITION, added 2026-10-28 (M2 / #479, C-388): "under count-matching" is doing real work and was never checked.** Where count-matching holds, measured **0.999999** (climatology and the delivered model, 13 months × 360×720). Where it silently fails — any forecast sparser than the truth — AFSS returns **0.001724**, i.e. the control reads as a catastrophic score and `all-zero` and a correct-total spiky forecast are **identical to six decimals**. ⛔⛔ **AMENDED 2026-10-28 (M3 / #480): "under count-matching" held in only 52% of 143 measured forecast-months**, by three distinct mechanisms (see the `hit rate` row). AFSS's three regimes are therefore **1.000000** (matched), **0.999999–0.999668** (mildly over-selected, which is where the delivered model and climatology sit) and **0.001724** (all cells selected) — ⇒ an AFSS of 0.9999 is not "essentially 1", it is the signature of a partially broken threshold. ⚠️ On a 24-wide window `n = 47` is unavailable, so an AFSS reported for such a window is **derived, not observed**, and must say which. | asymptotic FSS, FSS asymptote, FSS limit |
| **no-skill bar** | **What a UNIFORM forecast scores**: `0.5 + f_o / 2` (Roberts & Lean p.83), where `f_o` is the observed event frequency over the scored domain. Measured on the body's windows, **on the scored 360×720 domain**: `f_o` **0.000880** in sample and **0.000862** out, so the bar is **0.500440** and **0.500431**. ⚠️ The published baseline first reported 0.501759 / 0.501724, from an `f_o` taken on **land cell-months only** — see the warning below. ⛔ **On this grid the bar is useless as a threshold** — a 360×720 domain drives `f_o → 0`, so the bar sits at ~0.5004 while FSS at a single cell is already 0.63. That is why `scale_min` (the smallest `n` reaching the bar) read **1** on all 49 fields, both windows and both predictors, and was **dropped** from the scoreboard (`C-376`, `W-73`). ⚠️ **`f_o` must be computed over the domain FSS actually scores**, not over a sub-population of it — an `f_o` taken on land cell-months while the score runs on the full grid is a different quantity. | uniform-forecast score, FSS floor, skill threshold, random-forecast score |
| **placement_rho** | **How well the forecast gets the ORDER of magnitudes right, among the cells where violence actually happened.** The Spearman rank correlation between predicted and observed magnitude over a **declared** population — `scripts/placement_rho.py`, shipped 2026-10-28 (#483). It exists because **no other column in the battery sees a magnitude error that preserves placement**: permuting the oracle's event magnitudes among themselves moves AP, Brier, FSS1, FSS9, AFSS and hit_rate by exactly **0.000e+00** at every permuted fraction, while `placement_rho` falls **1.000000 → −0.030036** monotonically (#444, answered by measurement). ⛔ **Quote the population and the pooling with every value** (§5e): `population` ∈ {events, union, all} × `pooling` ∈ {pooled, monthly}, and the **default and only validated** cell is `events`/`pooled`. ⛔⛔ **`population='all'` is BLIND to the very defect the column was built for** — the target defect scores **0.000000** on `events` and **0.999998** on `all`, because zeros tie with zeros and positives with positives. ⛔ **It is NOT a standalone selector**: the degenerate `spiky-1/30` field scores **0.521**, above persistence (0.428) and climatology (0.394). A constant input returns **0.0**, not nan — documented, because nan propagates into a verdict and 0.0 is the honest "no ranking skill". | placement correlation, rank correlation (bare), magnitude rho, spearman (bare), rho (bare) |
| **positive-draw fraction** | **What share of a cell's own posterior draws are above zero.** Per cell, `n_positive / S`. ⛔ It is a property of the **posterior's shape**, not of any point summary taken from it: it is what decides whether the mode is at zero, so a point map reads 0 wherever this is ≤ 0.5 regardless of how large the cell's mean is. Recoverable without the cube from any two of (mean, conditional mean, firing rate), since `mean = (n_positive/S) × E[y|y>0]`. Measured on the 2026-09-29 delivery (S=16): **3/16** median in the cells the model itself scores ≥100 fatalities (**C-345**). | pos-draw share, non-zero draw fraction, n_pos/S |
| **retention** | **How much of its month-1 ability the model still has later on.** The forecast runs 36 months ahead one month at a time. Month 1 gets real observed data as input; every month after that the model has to feed on its **own** previous guess, and errors compound. Retention = `AP(h) ÷ AP(h=1)`, free-running, same cube. Example: 0.3452 at month 18 against 0.4991 at month 1 = **0.69**, i.e. it kept 69% and lost 31% to its own output. 1.0 = lost nothing. ⚠️ It can exceed 1.0 without anything being wrong — more cells have conflict at later horizons (1343 events at h1, 1547 at h18), so AP can legitimately be higher later. **Absolute `AP(h18)` is the primary endpoint and retention is co-primary; the two must agree in sign**, because a ratio can move on its denominator. Interval: `scripts/ap_block_bootstrap.py --ratio`. | retention rate, skill retention, decay ratio |
| **the ceiling** | **What the model would score at a given month if its inputs were perfect.** A diagnostic in which we cheat: instead of letting the model feed on its own guess, we hand it the **real observed field** each month, so it can never wander off. Whatever it scores then is the best that month could possibly go for this model — hence *ceiling*. It is the score of **the oracle** arm. Measured 0.5072 at month 18 on the 600-lesson reference, against 0.3452 free-running: **the 0.16 gap is self-inflicted, not caused by the horizon** — fed good inputs the model stays as sharp at month 18 as at month 1. ⛔ **The ceiling is NOT the month-1 score.** Those are two different numbers and calling AP(h=1) 'the ceiling' is drift — the month-1 score is the **T=0** score (§7). It is the ceiling *of the feedback path*: it says what perfect occurrence and magnitude would buy, nothing about what is learnable from the data. | headroom, upper bound, oracle ceiling |
| **occurrence** | **How often the model fires, field-wide.** The mean of the **gate** over every cell at a given horizon — no threshold, no "active cell" set, every cell counted. It is an observation, not an estimate: the gate cube stores `sigmoid(cls)` directly. Free-running seed 42: `4.146e-03` at h1 → `1.492e-04` at h36 (**×0.036**). ⛔ **Not** the fraction of cells above some probability — that is a *conditioned* statistic and is what produced **C-318**. | active fraction, firing rate, fire rate |
| **body magnitude** | **How much the model would predict, given that it fires.** The count-space `E[Y|body]` (`mu`), reported two ways, and the pair must be named because they behave differently: **plain** = the unweighted mean of `mu` over every cell; **gate-weighted** = `Σ(gate·mu) / Σ(gate)`, the second factor of the identity `mean(gate·mu) = mean(gate) × [Σ gate·mu / Σ gate]`. Free-running seed 42 h1→h36: plain **×0.222**, gate-weighted **×0.0144**. ⛔ **Never** the mean over cells that fired — with 1 such cell at h36 that is not a measurement (**C-318**, twice). | mean magnitude on active, magnitude, size |
| **gate–body alignment** | **How much more the model expects where it is more likely to fire.** `alignment = gate-weighted body magnitude ÷ plain body magnitude`, identically `1 + Cov(gate, mu) / (mean(gate) · mean(mu))` — a **normalised covariance**, verified numerically. **1.0 means the gate says nothing about where the large values are.** Free-running seed 42: **66.6 at h1 → 4.3 at h36**; with the cell state clamped it is **flat, 66.6 → 69.3** (M51/M52). ⚠️ **It is a RATIO of two quantities that both fall**, so "preserved" means they fall *proportionally* — it does not mean the field is unchanged. ⛔ **Not sharpness** (§ the sharpness diagnostic measures spatial structure — FSS, Moran's I — and a field can be perfectly aligned and completely unstructured). ⛔ **Not a skill score**: it is a diagnostic of the emitted field, and restoring it is not by itself evidence of a better forecast. ⛔⛔ **MEASURED BLIND TO PLACEMENT (EXP-3, 2026-09-03).** It is an **internal** statistic — whether the gate and the body agree *with each other* — and says **nothing about whether those cells are the right ones**. Rolling the clamp anchor 90 cells leaves alignment at **61.5** (against the clamp's 69.3) while `AP@h18` collapses **0.362 → 0.0075**. The same holds for **occurrence** and **body magnitude**: all three are preserved by a spatial roll that destroys the forecast. **No statistic in this row, or the two above it, can distinguish a good forecast from the same forecast in the wrong place.** Judging placement needs a truth-referenced score — AP, or the sharpness diagnostic's FSS. | alignment (bare — acceptable short form in context only), concentration, placement quality, gate concentration, gate sharpness |
| **cell state** (`hl`) | **The model's long-term memory — the only part that persists across rollout steps.** Updated as `hl = f·hl + i·h̃l`, so it accumulates. Four blocks, the SECOND half of the packed state tensor. **It is what free-running drains** (M51), **it holds a spatial map of where conflict will be** (M54: displace it 90 cells and the forecast moves 90 cells intact), **it drives placement while the fed-back input does not** (M60: cell 26/26, input 0/26), and **holding it is the programme's one positive intervention** (M48/M56). | long-term memory, `hl`, c-state, the long half |
| **hidden state** (`hs`) | **The short-term readout — recomputed from the cell every single step** as `hs = o·tanh(hl)`. Four blocks, the FIRST half of the packed state tensor. It does two real jobs: it drives all four gates, and it is concatenated onto the U-Net input (`x = cat([x, hs_1..4])`), so it shapes what the encoder sees. ⛔ **It has NO memory of its own.** Whatever you put in is overwritten within one step. ⛔ **Therefore you cannot measure its importance by perturbing it** — the perturbation is regenerated from the *undisturbed* cell and self-heals (see [[ablation-of-a-regenerated-component]], **C-323**). M60's "hidden 0/26" is architecturally guaranteed, not a finding; M56's hidden-freeze arm jams the readout rather than isolating a component. **Verified against `HydraBNUNet06_LSTM4.forward` lines 568–603, not inferred from behaviour.** | short-term memory, `hs`, h-state, the short half |

| **noise floor** (σ̂) | **How much a number moves when NOTHING changes but the random seed.** Estimator **LOCKED**: pooled sample sd, `ddof=1`, over true seed replicates — identical configs, identical data, identical git sha, differing only in `torch_seed`/`np_seed` (`2026-10-18_weight_decay_arms_prereg.md:1051`). ⛔ **Quote n and df with every value**: at n=3 the 95% CI is **[0.52σ̂, 6.28σ̂]**, a **12.1-fold** span, and the honest bar against ONE unreplicated arm is **4.97σ̂** (≈3.0σ̂ at n=5, ≈2.5σ̂ at n=10) — derive every bound from the **upper** limit. ⛔ **It is a property of a named STATISTIC, never of 'the model'** — measured 2026-10-28 (M75), B-1b **body-alone** has **CV > 1 at 11 of 14 scales** (σ̂ exceeds its own mean; C-391) while B-1b **gated out-of-sample** has **CV 0.024–0.054** on the same three runs. ⇒ Two columns of one instrument, one unusable and one solid. ⛔ **And σ̂ is NOT flat in a scale parameter** — across the three profiles it falls, falls and **rises**, so a single scalar is wrong in both directions (D-4). ⛔⛔ **NOT the across-draw σ** of `e1_sigma.py` (MC-dropout spread *inside* one checkpoint, ~20× a different scale) and **NOT** the gate-slope σ̂ of the 2026-10-21 readout (C-390) — three different quantities, one word; the collision that produced **W-83**. | sigma, σ, run-to-run variance (bare), seed noise, error bar, the noise (bare), e1 sigma |

## 4b. The PAPER's method vocabulary (LOCKED 2026-10-28) — the synthetic population and the minimal set

⭐⭐⭐ The terminal objective of this programme is a paper: **synthetic forecasts with a known ranking,
used to find the minimal set of metrics that ranks them correctly every time.** These four words are its
method and are not interchangeable. Full statement: `2026-09-30_evaluation_learning_report.md` **§9**.

| term | what it means | ⛔ banned aliases |
|---|---|---|
| **the synthetic population** | **The eleven hand-built forecasts over the same 842,634 cell–months, each embodying one NAMED failure mode or virtue** — `reports/2026-10-27_metric_redteam/redteam_forecasts.py`. The oracle, all-zero, climatology, persistence, spiky-1/30, freq-right/mag-wrong, mag-right/freq-wrong, the delivered model, and three mass-preserving blurs. ⛔ **Append-only and gated**: a forecast enters only with a construction gate. | the red-team forecasts, the fake forecasts, the test cases, the degenerate forecasts |
| **construction gate** | **A check that a synthetic forecast IS what it is captioned as, which must PASS before it is allowed to be scored.** Twelve exist and all PASS — e.g. persistence byte-identical to the lagged truth (`array_equal True`), the blurs preserving their monthly total (max relative error 2.4e-16), `mag-right/freq-wrong` holding the sorted nonzero multiset exactly in all 13 months. ⛔ **Distinct from a VOID gate**, which asks whether a *measurement* is valid; a construction gate asks whether the *object* is. | sanity check, assertion, validation, void gate (a different thing) |
| **the declared ranking** | **The ground-truth ordering of the synthetic population, written down with a justification per adjacent pair, BEFORE any subset search runs.** ⛔⛔ **IT DOES NOT EXIST YET.** The only ordering constraint in the harness is `separation_matrix.py:47` **G-1 — every column must rank the oracle first**. ⇒ *"ranks correctly every time"* has no criterion yet, and declaring it is a judgement task, not a computation. | the known ranking (acceptable short form), the true order, the gold standard, ground truth (reserved — see §8) |
| **minimal sufficient battery** | **The smallest set of (column, population, transform, estimator) tuples that reproduces the declared ranking over the synthetic population every time.** ⛔ **A column is not a column without its parameter values** — `placement_rho` scores its own target defect 0.000000 or 0.999998 depending on one argument. ⛔ **The claim is a partial factorization with MEASURED redundancy, never an orthogonal basis** — `R_T ≈ R_A ×` (conditional magnitude ratio) is a real identity, but three degrees of freedom remain, not two (§8.3's retraction). | the minimal set (acceptable short form), the orthogonal basis, the sufficient statistic, the metric suite, the core metrics |


## 5. Behaviour words
| locked name | meaning | banned aliases |
|---|---|---|
| **timid** | guesses too small | under-fires, under-shoots, shrinks, timid-prophet |
| **the drag** | the pull toward zero from training on ~99.7% empty cells | all-zeros drag, zero-pull |
| **the feedback realism gap** | the model's emitted field is not distributed like real conflict history (sparse, **persistent**, integer, spatially coherent), so feeding it back puts the model off the input distribution it trained on, and the error compounds. **The parent cause, not a failure mode** — its children are *the bloom* (over-firing), *the zero collapse* (under-firing), and plain skill loss. `mean` feedback gives act_ratio ~96× at h36, `sample` gives 0.27; **both sit at gate AP ~0.01**. Compounds **steeply then saturates**: gate AP 0.298 → 0.028 by h6, then flat 0.006–0.008 to h36 (EXP-01) — so ~5 steps holds most of it. Share flowing through the recurrent state: **~23%, INDICATIVE** (EXP-02); the rest is the direct input→prediction path. **Which memory half carries it is NOT established** — `hs` is a readout of `hl`, so freezing the cell also constrains the hidden half (C-292). One seed, 40 lessons, one vehicle; the pre-registration requires a second vehicle before this is anything but indicative. | exposure bias, the distributional gap, autoregressive DGP warp, generated≠DGP, self-poisoning |
| **the zero collapse** | ⚠️ **quantitative examples below are smoke-era and the framing is asymmetric — see `postmortem_floor_limited_vehicle.md` (2026-08-17).** Across the six-model roster the two children are NOT symmetric in consequence: `violet_visitor` under-fires 8× and retains **best** (0.54); `pink_pirate` over-fires 3× and retains **worst** (0.02). Neither extreme is *the* failure, and commitment does not predict retention at all (24× span, no relationship). The *under-firing* child of the feedback realism gap: under `rollout_feedback=sample` the rollout goes quiet — act_ratio 1.41 → 0.27 and `size_ratio` 0.0000 from h6 — while `crps_all` *improves* (the zero-domination trap). The mirror of *the bloom*, not a separate mechanism. | timid rollout, rollout collapse, going quiet |
| **the bloom** | forecasts snowballing to infinity across the 36-month rollout. The *over-firing* child of **the feedback realism gap**. **Mitigated (ADR-070, 2026-07-27):** family heads default to `rollout_feedback=sample`, which bounds it 9/9 — but that trades it for *the zero collapse*, it does not remove the parent cause. T=0 is still the scored product. | C-113, autoregressive explosion, runaway |
| **rollout_feedback** | what the autoregressive loop feeds back each step: `mean` (the diffuse emit-mean E[y] — the bloom driver), `sample` (a sparse composition-aware family draw — the bloom mitigation, default for family heads), `teacher_forced` (the realized truth — oracle/diagnostic only) | ancestral-feedback |
| **T=0-neutral** | a change that cannot alter the h=1 / scored-T=0 output; the sample-on default is T=0-neutral (emit-mean/gate/params byte-identical, and the D×K cube too after the per-step sampler seeding) | T0-safe |
| **free-running** | **The model feeding on its own output** — from month 2 onward it has no real data to use, because the future has not happened, so its own previous guess becomes the next month's input. This is the real product and the thing every rollout number describes unless it says **the oracle**. `rollout_feedback` ∈ {`mean`, `sample`}; the `identity` arm. | autoregressive, self-feeding, closed-loop |
| **the oracle** | **The same model, but we hand it the truth each month instead of its own guess.** `rollout_feedback='teacher_forced'`, the `use_real` arm. It is a measuring instrument, never a forecast — you cannot know the future when forecasting it. Its whole purpose is to separate *'the model is bad at long horizons'* from *'the model is poisoned by its own output'*. Its score is **the ceiling**. | use_real, teacher-forced run, perfect-feedback |
| **lesson** | one training iteration over `windows_per_lesson` windows; `total_lessons` sets both the number of gradient steps **and** the denominator of the curriculum cooling slope (`curriculum.py:85`), so changing it stretches the difficulty schedule rather than truncating it. The LR schedule is *not* rescaled — it is stepped per lesson on a fixed inverse-sqrt law, so a longer run is a strict prefix-extension in LR. | epoch, curriculum step |
| **calibrated** | honest — a "10% chance" happens ~10% of the time (gate), or guesses are right-sized (body) | — |

## 5b. How a gate is MIScalibrated (two separate defects — LOCKED 2026-10-18)

`calibrated` (§5) is one word for a gate being honest. It being *dis*honest has **two independent
axes**, and the whole `pos_weight` programme was spent discovering that we had no words for them.
Fit `logit(observed rate) = a + b·logit(gate)` over gate bins. An honest gate gives `a = 0, b = 1`.

| locked name | meaning | banned aliases |
|---|---|---|
| **the calibration level** (`a`) | **Whether the gate is too high or too low overall.** The intercept of that fit. A pure level error shifts every probability the same amount in logit space and leaves the *ordering* untouched. ⛔ **This is the ONLY axis `pos_weight` can move**, and it moves it by exactly `log(w)`: at the weighted-BCE optimum `logit(s*) = log(w) + logit(p)`, derived, not approximate. | bias, offset, miscalibration (bare) |
| **the calibration slope** (`b`) | **Whether the gate's probabilities are spread too narrowly or too widely.** The slope of that fit. `b > 1` means truth moves faster than the gate does. ⛔ **`pos_weight` cannot move this at any value** — it has no slope degree of freedom. ⚠️ Measured `b` is a **lower bound** on the true compression: fitting on a noisy, binned predictor attenuates the slope toward 1. | calibration curve, reliability slope, dispersion |
| **under-dispersed** | **The gate's probabilities are squashed toward the middle: too high where risk is low, too low where risk is high.** `b > 1`. Measured on both kestrels — control `1.146/1.177/1.139`, arm `1.419/1.251/1.456`, `b > 1` in all six fits (2026-10-16 OUTCOME §2). ⛔ **This is a SLOPE defect and is NOT a bias** — the signed error crosses zero (control, at gate ≈ 0.30), so any statistic that takes the absolute value is structurally blind to it. That is what hid it for two readings. ⛔ **Not "the gate is miscalibrated"** — that sentence is true of both axes and names neither. | squashed, compressed, over-regularised gate, timid gate (⛔ **timid** is §5 and is about the BODY) |
| **over-dispersed** | The mirror: `b < 1`, probabilities too extreme. Not observed on this platform; named so the pair is sayable and so a remedy that overshoots can be reported in one word. | over-confident |
| **temperature scaling** | **The one-parameter fix that moves the slope**: report `sigmoid(logit / T)` with a single fitted `T`. Strictly monotone, so it leaves **AP exactly unchanged** — AP depends on the gate only through ranks. Seconds on CPU, no retrain. ⚠️ Fitting and evaluating it on the same data is how a recalibration gets a free pass; cross-validate by origin. | temp scaling, Platt (⛔ **Platt/vector scaling** is the TWO-parameter version, `a + b·logit`, and is a different remedy) |

⚠️ **Both axes are properties of a REPORTED number, not only of the weights.** The gate we measure is
`mean(sigmoid(logit))` over `D = 4` locked-dropout draws, so anything between the head and that mean
is in scope — see [[project_gate_underdispersion_candidates]].

## 5c. How a BODY is MIScalibrated in magnitude (LOCKED 2026-10-20, REVISED 2026-10-20)

§5b named the gate's two axes. The body has the **same two axes** and had no words for them, so the
one measurement that found them had to borrow the gate's. Fit `log(mu_empirical) = a + b·log(mu_model)`
over cells — truth on prediction, the same direction as §5b, so `b > 1` means the same thing in both.

> ⛔ **REVISED THE SAME DAY IT WAS LOCKED, by the measurement this entry said could not be made.**
> Every number below was first measured on the **100-cell selected set** of
> `2026-10-20_d3_static_channel_ceiling.md` §4, because `mu` was not stored on the full grid. §9 of
> that report stores it (`d4_mu_on_the_grid.py`, 349.5 s, 3.95 GiB, 64,818 land cells × 49 months)
> and the slope **does not survive the move off the selected set**: the headline `b = 1.7301`
> becomes **0.92** in sample and **0.82** out of sample. The selected set is `active_cells_top100`
> — **top 100 by OBSERVED positive rate in the readout window** — so it selects cells on the
> outcome, which is the one selection rule that biases a truth-on-prediction slope upward. The
> locked names below are kept; the values are corrected and the old ones are kept visible.

| locked name | meaning | banned aliases |
|---|---|---|
| **the magnitude level** | **Whether the body is too small or too large overall.** The intercept. Measured median `log(mu_model/mu_emp)` = **−0.5831** on the 100-cell set: the body predicts **~56%** of the empirical mean. ✅ **A global constant, and therefore free to fix** — two scalars drive the median to −0.0122. ⚠️ **This figure is a SELECTED-SET figure and has not been recomputed on the grid** (D4 stored `mu` but read out slope and spatial structure, not the level). ✅ **RECOMPUTED ON THE GRID 2026-10-21** (`2026-10-21_body_scoreboard.md` B-3, `body_scoreboard.py`, 64,818 land cells, the **gated forecast** `gate × mu` against the truth's window mean): the level is **NOT A SINGLE NUMBER** — on cells with truth > 0 in the 36-month window it is **−2.3368 (9.7%)**, and on the 336-month base-rate population **+0.1778 (119.5%)**. ⛔ **The sign flips with the population**, because selecting cells on a 36-month truth selects observations whose own sampling noise biases them up. ⇒ **−0.5831 is retired as a quotable scalar**; quote the level only with its population, and prefer the 336-month one. ⚠️ `mu` **alone** may not be compared to the truth's window mean at all — it is a *conditional* mean against an *unconditional* one (`W-52`). ⛔ **Not `size-ratio`** (§4), which is a median ratio on conflict cells and is a *score*; this is the intercept of a fit and is a *defect axis*. | bias, global under-prediction, timid (⛔ **timid** is §5 and is the behaviour, not the axis) |
| **the magnitude slope** (`b`) | **Whether the body's per-cell magnitudes are spread too narrowly or too widely across cells.** `b > 1` means truth moves faster than `mu` does. ⛔ **At cell level there is NO slope defect.** On all 64,818 land cells (D4): cells with any event `b` = **0.9234** in sample (n=1431) / **0.8170** out of sample (n=912); cells with ≥2 events **1.0049** / **0.9670**. ⛔ **The `b` = 1.7301 (sd ratio 0.4146) that this entry was locked with is a SELECTION ARTIFACT of the 100-cell set** and may not be quoted as the body's slope. ⚠ Every one of these is a **lower bound** for §5b's errors-in-variables reason. ⚠️ The **all-land-cells** row (0.9032 in sample, 0.4491 out of sample) is **tie-dominated** — 98.6% of land cells never fire — and is not the number to quote; use the event-bearing rows. ✅ **Like-for-like the gate agrees**: **0.3906** all cells / **0.9444** active. The gate's binned-logit 1.146–1.456 (§5b) is a **different estimator on different artifacts** and may not be quoted against these in either direction. | body dispersion, shrinkage, range compression, regression to the mean |
| **magnitude under-dispersion** | ⛔ **RETIRED AS A BODY DEFECT, 2026-10-20 — twice corrected in one day and now withdrawn.** First locked as a cell-level slope defect (`b` 1.73, §4) — a selection artifact. Then restated as a coarse-scale amplitude deficit, on `sd(log mu)` flat in `k` while the truth's grew — **also an artifact**: the coarsening admits only blocks with an event, and the admitted fraction moves **2.2% → 22.4%** across `k` = 1…32, so the trend compared different populations (§10.4, `d7_coarsening_sample.py`). On a **nested** population the truth's spread **falls** and the ratio mu/truth **rises above 1** (0.88 → 1.20). ⇒ **The body is not under-dispersed in magnitude at any scale**, and no locked term is needed for a defect that is not there. Kept as an entry only to stop the name being reused. ⛔ The earlier reading — "squashed toward the middle, too large in quiet cells and too small in violent ones, `b > 1`" — described the selected set. ⚠️ `b` rising through 1 is **not one of the three signatures `slope_chase.py` pre-registered** (S: `b`→1 · G: `b` flat · R: `b`→1 with `r` climbing), so the discriminator does **not** discriminate here; the variance deficit is what it establishes, and §3's autocorrelation is the independent confirmation. | the magnitude gap, the amount ceiling (⛔ **the amount ceiling** is the separate claim that magnitude is *irreducible*; this axis is a measured defect in the model, not a statement about the data) |

⛔ **THE TWO HEADS FAIL THE SAME WAY. This box has now said "same", then "different", then "same"
again — each move forced by a better measurement, and the record of the moves is kept on purpose.**

| head | per-cell slope `b` (event-bearing) | spatial lag-1 autocorrelation of its own field |
|---|---|---|
| **gate** | **0.39** all / **0.94** active | logit **0.9598 / 0.9574** in sample · **0.9498 / 0.9464** out of sample |
| **body** | **0.92** in sample / **0.82** out of sample | `log mu` **0.8594 / 0.7636** in sample · **0.9266 / 0.9227** out of sample |
| **the truth, same cells** | — | **0.5915 / 0.5780** in sample · **0.5089 / 0.5028** out of sample |
| **the truth, noise-controlled** (336-month base rate) | — | **0.6419 / 0.6261** |

⛔ **The autocorrelations above are NOT on one scale, and the raw gap is ~¾ artifact.** `log mu` and
the gate logit are continuous and atom-free; the truth is a raw rate with **98.6% of its mass at
exactly zero**, and lag-1 Pearson is invariant under *affine* maps, not under `log` or `logit`. Pushing
the model through **the truth's own estimator** (§10.3, 300 replicates) the model's field reads
**+0.7053 / +0.6824**, not 0.8594 / 0.7636 — the estimator alone eats **0.154 / 0.081**. The truth's
**+0.6419 / +0.6261** still falls **outside** the 95% **simulation** interval — ⚠️ a simulation
interval, *not* a posterior-predictive one: it carries Monte Carlo variation only, with **no
parameter-uncertainty component**, since this is one training run. The truth sits **3.7 / 3.0**
half-widths below the lower edge, so a 3× wider interval keeps the verdict. The finding stands, but
**the like-for-like gap is 0.06, not 0.22**, and 0.06 is a **floor** (the model's larger atom, its
patchier support, and its own 1.51× over-dispersed conditional variance all push the measured gap
down). ⛔ **And the comparison is only well posed in this form at all:** a conditional-mean field is
smoother than its own realisation *by construction* (`ρ_Y = ρ_m·Var(m)/(Var(m)+σ²) < ρ_m`), so the
question is never "smoother than the truth" but "smoother than the variance decomposition requires". ⛔ Quote 0.06. Never quote a model
autocorrelation against a truth autocorrelation without passing one through the other's estimator.

✅ **The global half, added 2026-10-20 (§10.9) — and it needs no estimator at all.** Share of firing
mass in the top 0.1% of cells: model **0.4578** against the truth's **0.2455**. Effective cells
(`(Σp)²/Σp²`): model **215.0** against a **debiased** **536.3** — **2.49× too few places**, robust at
every truth window from 36 to 336 months (2.46×–3.87×), and the matched 36-month window is the
**conservative** end. ⚠️ The truth's raw 516.9 is biased **down** by count-sampling noise
(`E[Σp̂²] = Σp² + Σp(1−p)/T`), i.e. against the finding; the model's row is analytic and needs no
correction. Both statistics are normalised by the total, so the model's 4.18×
under-firing **cancels exactly**: no scale choice, no null, no saturating map. ⇒ **The model emits a
few large smooth blobs where the truth has many small sharp spots.** Smoothness is local-gradient,
concentration is global-distribution, so a field can be both — and **concentration is a JOINT
property**, making this the first direct quantification of what C-152's class statement asserts.

**Both heads have roughly calibrated marginals at cell level and both emit a field measurably smoother
than the truth.** The body ranks cells well (spearman `mu` vs truth **+0.8205** in sample, **+0.7808**
out of sample). ⇒ The defect is **shared**, so it is upstream of both heads — in **anything they
share**, which is the trunk *and* the input channels, the data, the optimiser, the loss weighting, the
BatchNorm regime and the global weight decay. ⚠️ An earlier version of this line said "the trunk and
its recurrent state", which named one shared thing as if it were the only one and contradicted this
entry's own candidate list.

⛔ **Why this box moved twice.** (1) *"Same"* was written from two **different estimators** (body
per-cell OLS vs gate binned-logit) and was unsupported. (2) *"Different"* was written from the
like-for-like estimator but with the body measured **only on a set selected on the outcome**, and the
gate on the whole grid — a comparison between a biased slope and an unbiased one. (3) *"Same"* is now
written with **both heads on the same 64,818 cells from the same forward pass**, which is the first
version of this comparison in which neither side has a selection or an estimator advantage.
⚠️ One vehicle (`4acbf4d3c2b457b0…`, the retrained control, **not** the delivered weights — C-317),
n=1, `sb` only.

## 5d. The spatial-structure CEILING — two fractions, and they are not interchangeable (LOCKED 2026-10-25)

`2026-10-21_body_scoreboard.md` B-1 says the body *"carries 0.391 of its due spatial structure at one
cell"*. That sentence silently needs an **achievable value**, and until 2026-10-25 nobody had measured
one. `2026-10-25_sharpness_ceiling_prereg.md` measured it by splitting the 336 in-sample months into two
independent halves — two noisy observations of the same latent field — and comparing them.

⛔ **It produced TWO numbers, and conflating them was the finding.** Both are locked:

| locked name | definition | measured | banned aliases |
|---|---|---|---|
| **reproducible amplitude fraction** `C(d)` | `gamma_cross(d)/gamma_B(d)` between the two half-samples. **Not** variance-normalised | **0.498** at `d=1`; 0.795 at `d=32` | "the ceiling", "reproducible fraction" unqualified |
| **reproducible shape fraction** | the same split-half comparison under **B-1's own** estimator — the variogram ratio **divided by the ratio of the two fields' variances** (the same-shape null) | **0.983 – 1.009** across `d = 1…32` | "the ceiling" unqualified |

⛔⛔ **CORRECTED 2026-10-27b, AND THE LOCKED CONCLUSION IS REVERSED.** This read:

> ⭐ **Only the shape fraction bounds B-1**, because B-1 divides amplitude out. ⇒ **the ceiling the
> 0.391 is read against is ~1.0**, and the amplitude fraction 0.498 bounds something else.

**The first clause is sound and the conclusion does not follow.** B-1 does divide amplitude out, so a
bound on B-1 must indeed be shape-like. ⛔ But the **shape fraction as measured is degenerate**: it
compares one half of the truth against the other under B-1's formula, and two halves of one dataset
have the same roughness *and* the same variance **by construction**. ⇒ it is ~1.0 for **any** dataset,
white noise included. Measured: `[γ_A/γ_B] / [var_A/var_B]` reproduces the recorded 0.983–1.009 with a
variance ratio of **1.0251 at all six lags, constant to 0.00002**.

**A quantity that is ~1.0 regardless of the data cannot bound anything.** ⇒ **the ceiling B-1's 0.391
is read against is `0.498`** — the cross term, which is how well one half **predicts** the other, and
which is the only one of the two that a model has to beat.

| | `0.391` is… | headroom |
|---|---|---|
| against ~1.0 (withdrawn) | 39% of achievable | ~61% |
| **against 0.498 (locked)** | **78.5%** | **~21%** |

⛔ **Both terms stay locked and both keep their definitions** — the distinction between them was a real
finding and the conflation was a real error. What is withdrawn is *which one bounds B-1*. ⚠️ This
sentence had released a pod arm (`2026-10-25_sharpness_ceiling_prereg.md` X-1, now retracted); see the
amendment in `2026-10-26_structure_penalty_arm_outcome.md` for what it does to that arm's null.

> ⛔ **This box exists because the two were conflated once, in the document that measured them.** The
> prereg's §3 decision rule compared `C(1) = 0.498` (amplitude) against B-1's `0.391` (shape) and
> concluded the arm's upside was "capped at 0.498, not 1.0". That is a category error and it errs toward
> **falsely throttling** the arm. Corrected in that prereg as **X-1**, with the original text kept.
>
> ⚠️ **This is the THIRD recorded instance of the same error class in this glossary** — see §5c's *"Why
> this box moved twice"*, item (1): *"'Same' was written from two **different estimators** and was
> unsupported."* The rule that follows from three instances is in §5e.

⛔ **Standing caveat on BOTH fractions, which no estimator removes.** A split-half comparison cancels
noise that is **independent between the halves** (event-level sampling, month-binning). Systematic
provider geocoding lands in the **same cell in both halves**, survives, and is counted as reproducible
structure. ⇒ Every fraction above is an **UPPER BOUND**. A **low** value is conclusive; a **high** value
is not.

⛔ **`C_full` / "the 1/n-corrected ceiling" (0.665) is WITHDRAWN, not locked.** It applied
Spearman–Brown, whose premise (additive noise, variance ∝ 1/n) is a statement about **counts**, to a
`log(rate + eps)` field. The same correction gives 0.665 logged and 0.262 raw — 3.3× apart. Do not quote
it.

## 5e. ⛔ THE ESTIMATOR-NAMING RULE (LOCKED 2026-10-25, from three instances — a FOURTH added 2026-10-27)

Four separate measurements in this project have been wrong in the same way: **two numbers that share a
name were produced by different estimators and were treated as if they were one quantity.** ⛔ The fourth
is the costliest so far, because what it corrupted was not a number but a **plan**.

| # | where | what was compared |
|---|---|---|
| 1 | §5c item (1) | body per-cell OLS vs gate binned-logit, both called a "slope" |
| 2 | `feedback_never_mix_two_statistics` | a median-over-cells vs a pooled mean, which invented a phantom finding (B7) |
| 3 | §5d / X-1 | a variance-normalised variogram ratio vs a raw one, both called a "fraction of structure" |
| 4 | `e1_sigma.py` vs views-models#625 | an **across-draw** logit sd — MC-dropout noise *inside one checkpoint* — vs a **run-to-run `σ̂`** — variance *across independently seeded trainings*. Both are called "sigma". I read the existence of the first as the existence of the second, and reported a **free local job on 26 GB of seed replicates already on this disk** as an *external blocker* gating the whole structure programme (`W-83`) |

⇒ **THE RULE.** A sentence comparing two statistics must name, for **each side**: the **population**,
the **transform**, the **estimator**, *and* **whether it is normalised and by what**. The first three
were already required by C-351 (*"name the population with every number"*); **the fourth is new and is
what all three instances above actually missed.** A ratio and a normalised ratio are different
quantities even when both are dimensionless and both live in `[0, 1]`.

⛔ **AND THE TWO SIGMAS ARE NOW NAMED, because instance 4 will otherwise recur:**

| locked name | what it measures | banned aliases |
|---|---|---|
| **across-draw σ** | **Spread over MC-dropout draws of ONE checkpoint.** `e1_sigma.py`'s quantity: the sd of the gate logit over `D = 4` locked-dropout draws. ⛔ It is a property of **one trained model's own stochasticity** and bounds nothing about training. | sigma (bare), dropout sigma, logit sd |
| **run-to-run σ̂** | **Spread over independently SEEDED trainings of one config.** What a decision rule needs, what `C-372` and the scoreboard's rule 7 are about, and what views-models#625 asks for. Computable from `silver`/`bronze`/`cobalt_kestrel` (seeds 42/43/44) — ⚠️ `n = 3`, common complete population **origins 0–5**, family `truncated_nb`. | sigma-hat (acceptable short form), seed sigma, sigma (bare — ⛔ NEVER, this is instance 4) |

⇒ **"Sigma" bare is banned.** Say which one, every time.

## 6. The models we've actually run (described by their choices — codenames RETIRED)
I will describe models by their §2/§3 choices, not by codenames. The codenames below are retired; they map to:

| retired codename | its actual choices (the words I'll use instead) |
|---|---|
| **white_ranger** | the **baseline**: climatology (resample each cell's own history), all 3 targets, PGM. The thing to beat. |
| **A0p** | gated forecast, positives-only body, softplus, MAE, pos_weight 2 |
| **dense-mse** | all-cell body, MSE, softplus/relu (an OLD run whose forecast was body-alone, not gate×body) |
| **the dial grid** | gated + positives-only + MAE + the dial swept 0.5→0.8 (the F2 negative) |

## 7. Settled facts — do NOT re-open
- **the WALL** — the data cannot say *how big* a sudden jump will be (no feature predicts it). Proven.
- **BatchNorm fix** — the instability that collapsed ~40% of random starts; fixed, on by default. Proven.
- **T=0** — the first forecast month only; the *only* thing we score. The rollout past it (the bloom) is now **mitigated** at inference (ADR-070 `rollout_feedback=sample`), not merely ignored; T=0 remains the scored product.
- **the gate is calibrated** at pos_weight ~1–2 (its own question, largely solved).

## 8. The targets
| locked name | what it is |
|---|---|
| **sb** | state-based violence (e.g. government vs rebels) |
| **ns** | non-state violence (group vs group) |
| **os** | one-sided violence (against civilians) |

---
*If a needed word is missing, I ADD a row here — I never invent a synonym on the fly. You enforce with one word: "drift".*
