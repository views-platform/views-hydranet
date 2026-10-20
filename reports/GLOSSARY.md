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
| **positive-draw fraction** | **What share of a cell's own posterior draws are above zero.** Per cell, `n_positive / S`. ⛔ It is a property of the **posterior's shape**, not of any point summary taken from it: it is what decides whether the mode is at zero, so a point map reads 0 wherever this is ≤ 0.5 regardless of how large the cell's mean is. Recoverable without the cube from any two of (mean, conditional mean, firing rate), since `mean = (n_positive/S) × E[y|y>0]`. Measured on the 2026-09-29 delivery (S=16): **3/16** median in the cells the model itself scores ≥100 fatalities (**C-345**). | pos-draw share, non-zero draw fraction, n_pos/S |
| **retention** | **How much of its month-1 ability the model still has later on.** The forecast runs 36 months ahead one month at a time. Month 1 gets real observed data as input; every month after that the model has to feed on its **own** previous guess, and errors compound. Retention = `AP(h) ÷ AP(h=1)`, free-running, same cube. Example: 0.3452 at month 18 against 0.4991 at month 1 = **0.69**, i.e. it kept 69% and lost 31% to its own output. 1.0 = lost nothing. ⚠️ It can exceed 1.0 without anything being wrong — more cells have conflict at later horizons (1343 events at h1, 1547 at h18), so AP can legitimately be higher later. **Absolute `AP(h18)` is the primary endpoint and retention is co-primary; the two must agree in sign**, because a ratio can move on its denominator. Interval: `scripts/ap_block_bootstrap.py --ratio`. | retention rate, skill retention, decay ratio |
| **the ceiling** | **What the model would score at a given month if its inputs were perfect.** A diagnostic in which we cheat: instead of letting the model feed on its own guess, we hand it the **real observed field** each month, so it can never wander off. Whatever it scores then is the best that month could possibly go for this model — hence *ceiling*. It is the score of **the oracle** arm. Measured 0.5072 at month 18 on the 600-lesson reference, against 0.3452 free-running: **the 0.16 gap is self-inflicted, not caused by the horizon** — fed good inputs the model stays as sharp at month 18 as at month 1. ⛔ **The ceiling is NOT the month-1 score.** Those are two different numbers and calling AP(h=1) 'the ceiling' is drift — the month-1 score is the **T=0** score (§7). It is the ceiling *of the feedback path*: it says what perfect occurrence and magnitude would buy, nothing about what is learnable from the data. | headroom, upper bound, oracle ceiling |
| **occurrence** | **How often the model fires, field-wide.** The mean of the **gate** over every cell at a given horizon — no threshold, no "active cell" set, every cell counted. It is an observation, not an estimate: the gate cube stores `sigmoid(cls)` directly. Free-running seed 42: `4.146e-03` at h1 → `1.492e-04` at h36 (**×0.036**). ⛔ **Not** the fraction of cells above some probability — that is a *conditioned* statistic and is what produced **C-318**. | active fraction, firing rate, fire rate |
| **body magnitude** | **How much the model would predict, given that it fires.** The count-space `E[Y|body]` (`mu`), reported two ways, and the pair must be named because they behave differently: **plain** = the unweighted mean of `mu` over every cell; **gate-weighted** = `Σ(gate·mu) / Σ(gate)`, the second factor of the identity `mean(gate·mu) = mean(gate) × [Σ gate·mu / Σ gate]`. Free-running seed 42 h1→h36: plain **×0.222**, gate-weighted **×0.0144**. ⛔ **Never** the mean over cells that fired — with 1 such cell at h36 that is not a measurement (**C-318**, twice). | mean magnitude on active, magnitude, size |
| **gate–body alignment** | **How much more the model expects where it is more likely to fire.** `alignment = gate-weighted body magnitude ÷ plain body magnitude`, identically `1 + Cov(gate, mu) / (mean(gate) · mean(mu))` — a **normalised covariance**, verified numerically. **1.0 means the gate says nothing about where the large values are.** Free-running seed 42: **66.6 at h1 → 4.3 at h36**; with the cell state clamped it is **flat, 66.6 → 69.3** (M51/M52). ⚠️ **It is a RATIO of two quantities that both fall**, so "preserved" means they fall *proportionally* — it does not mean the field is unchanged. ⛔ **Not sharpness** (§ the sharpness diagnostic measures spatial structure — FSS, Moran's I — and a field can be perfectly aligned and completely unstructured). ⛔ **Not a skill score**: it is a diagnostic of the emitted field, and restoring it is not by itself evidence of a better forecast. ⛔⛔ **MEASURED BLIND TO PLACEMENT (EXP-3, 2026-09-03).** It is an **internal** statistic — whether the gate and the body agree *with each other* — and says **nothing about whether those cells are the right ones**. Rolling the clamp anchor 90 cells leaves alignment at **61.5** (against the clamp's 69.3) while `AP@h18` collapses **0.362 → 0.0075**. The same holds for **occurrence** and **body magnitude**: all three are preserved by a spatial roll that destroys the forecast. **No statistic in this row, or the two above it, can distinguish a good forecast from the same forecast in the wrong place.** Judging placement needs a truth-referenced score — AP, or the sharpness diagnostic's FSS. | alignment (bare — acceptable short form in context only), concentration, placement quality, gate concentration, gate sharpness |
| **cell state** (`hl`) | **The model's long-term memory — the only part that persists across rollout steps.** Updated as `hl = f·hl + i·h̃l`, so it accumulates. Four blocks, the SECOND half of the packed state tensor. **It is what free-running drains** (M51), **it holds a spatial map of where conflict will be** (M54: displace it 90 cells and the forecast moves 90 cells intact), **it drives placement while the fed-back input does not** (M60: cell 26/26, input 0/26), and **holding it is the programme's one positive intervention** (M48/M56). | long-term memory, `hl`, c-state, the long half |
| **hidden state** (`hs`) | **The short-term readout — recomputed from the cell every single step** as `hs = o·tanh(hl)`. Four blocks, the FIRST half of the packed state tensor. It does two real jobs: it drives all four gates, and it is concatenated onto the U-Net input (`x = cat([x, hs_1..4])`), so it shapes what the encoder sees. ⛔ **It has NO memory of its own.** Whatever you put in is overwritten within one step. ⛔ **Therefore you cannot measure its importance by perturbing it** — the perturbation is regenerated from the *undisturbed* cell and self-heals (see [[ablation-of-a-regenerated-component]], **C-323**). M60's "hidden 0/26" is architecturally guaranteed, not a finding; M56's hidden-freeze arm jams the readout rather than isolating a component. **Verified against `HydraBNUNet06_LSTM4.forward` lines 568–603, not inferred from behaviour.** | short-term memory, `hs`, h-state, the short half |

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
| **the magnitude level** | **Whether the body is too small or too large overall.** The intercept. Measured median `log(mu_model/mu_emp)` = **−0.5831** on the 100-cell set: the body predicts **~56%** of the empirical mean. ✅ **A global constant, and therefore free to fix** — two scalars drive the median to −0.0122. ⚠️ **This figure is a SELECTED-SET figure and has not been recomputed on the grid** (D4 stored `mu` but read out slope and spatial structure, not the level). Treat **−0.5831 as unverified off the 100-cell set** until it is recomputed. ⛔ **Not `size-ratio`** (§4), which is a median ratio on conflict cells and is a *score*; this is the intercept of a fit and is a *defect axis*. | bias, global under-prediction, timid (⛔ **timid** is §5 and is the behaviour, not the axis) |
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
