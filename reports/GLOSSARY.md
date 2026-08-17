# LOCKED VOCABULARY — the only words I use

**Rule:** every concept has ONE name (left column). I use only that name. The "banned aliases" are words I
used before for the same thing — listed so you can map old messages onto the locked name. I never introduce
a new synonym. If I drift, you type **"drift"** and I stop and correct.

Last rebuilt: 2026-07-17 (v2, complete). **Amended 2026-08-15:** added *the feedback realism gap* (the parent cause) and *the zero collapse* (its under-firing child); *the bloom* is now defined as the over-firing child of the same cause.

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
| gamma | — (no zero mass) | `gated_gamma` | `th_gated_gamma` |
| lognormal | — (no zero mass) | `gated_lognormal` | `th_gated_lognormal` |
| ZI-gamma *(if ever built)* | `ZIgamma` | `gated_ZIgammacore` | `th_gated_ZIgammacore` |

## 3. The losses (how the body is scored during training)
| locked name | what it does | banned aliases |
|---|---|---|
| **MSE** | squared error; fits the **mean** | mean loss, squared loss |
| **MAE** | absolute error; fits the **median**. **Identical to "pinball at 0.5".** | pinball τ0.5, pinball-at-0.5, median loss, "no dial" |
| **the dial** | pinball loss at a level above 0.5 (0.6/0.7/0.8) — pushes guesses UP. | pinball, tau/τ, PinballBodyLoss, lifter |
| **count_mean** | MSE computed in count space (not log space) — the one that blows up | count-space MSE |
| **NB loss** | the negative-binomial likelihood body (the old floor) | hurdle_nb, TruncatedNB |
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
| **calibrated** | honest — a "10% chance" happens ~10% of the time (gate), or guesses are right-sized (body) | — |

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
