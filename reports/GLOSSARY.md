# LOCKED VOCABULARY — the only words I use

**Rule:** every concept has ONE name (left column). I use only that name. The "banned aliases" are words I
used before for the same thing — listed so you can map old messages onto the locked name. I never introduce
a new synonym. If I drift, you type **"drift"** and I stop and correct.

Last rebuilt: 2026-07-17 (v2, complete).

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

## 2. How the body is TRAINED (a separate choice from §1)
| locked name | what it means | banned aliases |
|---|---|---|
| **all-cell** | body trained on *every* cell (including the empty ones) | dense, standard, no mask |
| **positives-only** | body trained *only* on cells that actually had conflict | hurdle, hurdle_shrinkage, hurdle-masked, masked body |

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
| **the bloom** | forecasts snowballing to infinity across the 36-month rollout (we ignore it; T=0 only) | C-113, autoregressive explosion, runaway |
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
- **T=0** — the first forecast month only; the *only* thing we score. Everything past it (the bloom) ignored.
- **the gate is calibrated** at pos_weight ~1–2 (its own question, largely solved).

## 8. The targets
| locked name | what it is |
|---|---|
| **sb** | state-based violence (e.g. government vs rebels) |
| **ns** | non-state violence (group vs group) |
| **os** | one-sided violence (against civilians) |

---
*If a needed word is missing, I ADD a row here — I never invent a synonym on the fly. You enforce with one word: "drift".*
