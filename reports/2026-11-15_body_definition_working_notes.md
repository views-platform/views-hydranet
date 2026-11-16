# Working notes behind ADR-079 (the definition of a body)

Moved out of the ADR when it was accepted on 2026-11-16. These are the decisions as they were
made, with Simon's words, plus findings, open questions and unknowns. The ADR is the decision;
this file is the record of how it was reached. Not everything here made it into the ADR — see T9
for what is still open.

## Notes

Everything known, thought or decided about the definition of a body, as of 2026-11-15. Each item
says what kind of statement it is:

- **Decided** — Simon decided it, on the date given.
- **Agreed** — discussed and agreed in conversation, not yet written as a decision.
- **Proposed** — put forward, not agreed.
- **Open** — a question nobody has answered.
- **Checked** — read in the code or the configs on 2026-11-15. Nothing here was trained or run
  unless it says so.

### T1 · The definition (Agreed 2026-11-15; brought up to date with D11–D16 the same day)

**A body is one named, complete answer to the question "what does the regression head compute, and
how is it trained?"**

A model has one body. All its regression targets (`sb`, `ns`, `os`) share it (D13).

Choosing a body fixes all of the following together. None of them can be set separately.

1. **Width** — how many raw numbers the head emits per cell, for each target.
2. **Meaning** — the map from those raw numbers to what they stand for, including the scale. The
   body's activation is this map, so which activation a body uses is part of the body (D16).
   Examples: a prediction in log1p space; a distribution's parameters in count space; a ranking
   score on the whole real line.
3. **Objective** — the loss that trains it.
4. **Forecast** — the body's own map from the raw numbers to a count forecast, and whether it can
   draw samples (yes or no).
5. **Own zeros** — whether the body produces its own zeros (yes or no).
6. **Required numbers** — the data facts it needs, such as `y_max`. Each is required, with no
   default, and the body refuses to build without it. A required number may be set per target
   (D13).

**Two more facts a body states.** Each is a yes or no. They are not extra choices: they follow from
the objective in item 3. They are stated so that settings outside the body can be checked against
them.

- **Needs raw counts** — whether the loss needs its target as raw counts (D11). The body does not
  own or name the transform that the data uses.
- **Must be trained on all cells** — whether the loss needs every cell, zeros included (D12). The
  ranking body: yes, because the zero cells are the bottom of its ranking.

**The test for "same body or different body":**

- If two setups differ in any of items 1 to 5, they are different bodies with different names.
- If they differ only in a number from item 6, they are the same body.

**What a body is not:**

- Not the gate, the network below the head, or the data.
- Not the transform applied to the data (D11).
- Not the choice of which cells it is trained on (`body_supervision`, D12).
- Not the choice of how gate and body are combined (`forecast_composition`), which distribution is
  reported from after training (`emit_distribution`), or what is fed back during rollout
  (`rollout_feedback`).

Those all stay separate settings. Each is checked against the facts the body states (items 2, 4
and 5, and the two yes-or-no facts above), and a setting the body cannot support is refused loudly.

**Rules beside the definition (D16).** They say how bodies are built, not what a body is:

- Every body uses softplus on every channel that must be positive. It is fixed in the code, with
  no config field
  (D6, D7; reasons in D17).
- The loss always gets the raw numbers and applies the activation itself where it needs to (D8).

### T2 · Decisions

| # | statement | kind |
|---|---|---|
| D1 | **Choosing a body is one choice.** Even if two bodies are very similar, it is not a mix and match of options. "You choose a body, and that is the body." | Decided 2026-11-15 |
| D2 | **A different setting that changes the kind of objective is a different body.** The listwise loss with `exp` and with `softplus` are two bodies. Reason given: duplication is cheaper than the wrong abstraction. | Decided 2026-11-15 |
| D3 | **`y_max` is a required parameter of a body, not a body.** It is a fact about the data. Required, no default. | Decided 2026-11-15 |
| D4 | **The definition stands for the ranking body even though its count forecast is not built.** "We are just not far enough in the construction, development and experimentation with the ranking head. It will/should at one point but it is simply not the thing we are trying to fix now." | Decided 2026-11-15 |
| D5 | **The unused bodies keep their code and tests and get no name now.** The six unused `output_distribution` values (`zinb`, `hurdle_nb`, `hurdle_lognormal`, `hurdle_shrinkage`, `dense_nb`, `quantile`) and the ten unused loss-registry entries are tested, working code that is not operational. Each earns a name when it is tried, one at a time. Saved models that use them must still load. | Decided 2026-11-15 (made for ADR-081; carried here) |
| D6 | **The last activation of the regression head is softplus.** There is no reason for anything else in what we run: the only alternative the config can name is `relu`, and a ReLU output head dies on rare targets (`C-178`). | Agreed 2026-11-15 |
| D7 | **Softplus is hardcoded, not a default and not a config field.** A rule beside the definition, not part of it (D16). | Decided 2026-11-15 (Simon's design; first logged as "proposed" by mistake) |
| D8 | **The loss is always handed the pre-activation output. Always.** A loss that needs the activated value applies it itself. A rule beside the definition, not part of it (D16). ⚠️ One known risk: the `mse` loss must then apply softplus itself; if that step is missing the model trains on one number and reports another with no error. Needs one test that fails when the step is missing. Nothing has been trained under this rule. | Decided 2026-11-15 (Simon's design; first logged as "proposed" by mistake) |
| D9 | **The softplus segment in ADR-081's composite string is redundant, for now.** | Agreed 2026-11-15 |
| D10 | **ADR-081 (the config field) is parked** until this ADR is done. | Decided 2026-11-15 |
| D11 | **The body states whether it needs its target as raw counts (yes or no). It does not own or name the transform.** Which transform the data uses stays a data setting. Reason: softplus is internal to the body, but the transform is a decision about the data, shared with the network's inputs and the pipeline downstream; and Simon is not sure log1p is the right transform, so it must not be locked in the way softplus is. | Decided 2026-11-15 |
| D12 | **Which cells the body is trained on (`body_supervision`) is not part of the body.** It is a separate setting outside the body, and a paused research question whose code stays. The body states one fact it is checked against: "must be trained on all cells, yes or no" (the ranking body: yes, because the zero cells are the bottom of its ranking). A conflicting setting is refused loudly. Item 5 ("can the body produce zeros") is a different thing, is part of the body, and does not change. | Decided 2026-11-15 |
| D13 | **One body per model.** All regression targets (`sb`, `ns`, `os`) have the same body. A body's required numbers and hyperparameters may be set per target (the gate's `pos_weight` is the existing example of a per-target number). There is a future use case where different targets have different bodies; it is not now and is not to be designed for. | Decided 2026-11-15 |
| D14 | **The quantile head is deleted, not kept.** This changes D5 for `quantile` only; the other five unused values are untouched. Simon's reasons, stated 2026-11-15: it is too different from the other heads and too bothersome for its expected utility, and a quantile head is not a promising avenue for zero-inflated, heavy-tailed data. He states he refused its addition several times. If it is ever wanted again, it is designed from scratch at that time. ⚠️ The code is purged once the body definition is written, not before. Not deleted yet (9 package files, 14 test files mention it; 0 configs use it — checked 2026-11-15). | Decided 2026-11-15 |
| D15 | **No quantile head is reconsidered until multi-number heads are settled.** A parametric body emits more than one number per cell (width above 1). Today all of those numbers come out of one shared convolution. ADR-081 §2 notes they could instead come from separate convolutions or from decoder branches that split earlier (issue #600). Before anyone thinks again about a quantile head, whose width is variable, there must be a clear discussion, a clear decision, and clear experiments on how the parametric bodies with width above 1 get their numbers. Not to be solved by implementing one more kind of head. | Decided 2026-11-15 |
| D16 | **The split between the definition and the building rules.** *Inside the definition:* which activation a body uses, because item 2 says a body fixes the map from its raw numbers to their meaning, and the activation is that map. *Beside the definition, in this same ADR, as rules for how bodies are built:* (a) every body uses softplus on every channel that must be positive (D6, D7). A channel that must lie between 0 and 1 gets a sigmoid instead (`zinb`, `mixture_nb`; checked). Reworded 2026-11-16 from "count channels", because the plain regression body and the ranking body do not output counts; (b) the loss always gets the raw numbers (D8). | Decided 2026-11-15 |
| D17 | **Why softplus, and not ReLU or leaky ReLU.** *ReLU:* tried, and it failed. The head's raw numbers drifted negative on 100% of cells, including event cells; ReLU then outputs exactly 0 and passes back no gradient, so the head never recovers (`C-178`; read in ADR-063 on 2026-11-15, the run is from July 2026 and was not re-run). ReLU also re-creates the tie at zero for the ranking body (ranking dossier `03` §5, read). *Leaky ReLU:* it can output negative numbers, and a count cannot be negative, nor can the parameters of a count distribution. This is reasoning, not a measurement: as far as is known nobody has run leaky ReLU here. *Softplus:* always above zero, always passes back a gradient, unbounded above. | Decided 2026-11-15 |
| D18 | **This ADR replaces part of ADR-063.** Replaced: the ReLU default for `standard`, and `reg_activation` as an override. Not replaced: softplus for the hurdle bodies, and ADR-063's reasoning against ReLU. A note saying so was added to ADR-063's status line and decision statement on 2026-11-15. | Decided 2026-11-15 |
| D19 | **What "own zeros" (item 5) means.** A body makes its own zeros when it can itself put weight on zero. Simon, 2026-11-15: most bodies do; the only live one that does not is `truncated_nb`, with the justification of the hurdle-model papers. It is **not** the code's `self_zeroed` flag (only `zinb`; means "built to be used with no gate"). The first draft of the §2 table filled this column from that flag and said "no" for all five bodies; that was Claude's error, found by the audit of the written ADR and corrected. Library, corrected 2026-11-16: Claude's first search reported one hurdle paper and no supporting claim. That was wrong: the library's search index file (`papers/_index/library.db`) does not exist, so the search returned noise. The library session found ten papers on hurdle, zero-inflated and zero-truncated models. The citation is `Mullahy1986_ModifiedCountDataModels`, claim C-809 (claim text checked in the library's claim file by Claude), with `Cragg1971_LimitedDependentVariables`, claim C-790, for the original two-part argument (C-790 read from the library session's message, not checked). | Decided 2026-11-15 |

### T3 · The smells found (Checked)

| # | smell | where |
|---|---|---|
| 1 | A config states a loss it does not train — 27 of 37 live configs | `choose_loss`, `utils/utils.py:220` |
| 2 | `output_distribution` means two things: a distribution for four values, a head shape for six | the field itself |
| 3 | `reg_activation` is set in all 37 configs and read in only 10 | `HydraBNrecurrentUnet_06_LSTM4.py:118-127` |
| 4 | A string prefix (`startswith("hurdle")`) picks an activation | same, `:124-127` |
| 5 | The head emits the same output twice, activated and raw | `ModelOutput.reg` / `.reg_latent` |
| 6 | `needs_latent` answers two unrelated questions (see T6) | `training_engine.py:712`, `config_initializer.py:1563` |
| 7 | The softplus segment in ADR-081's composite string is redundant | ADR-081 |
| 8 | The activation is applied in three different places | the head ladder; `family.activate`; inside `TruncatedNBLoss` and `DenseNBLoss` |
| 9 | Decision-relevant defaults: `"standard"`, `"exp"`, `"self_zeroed"` | `training_engine.py:1337`, `:1339`, `:1401`; `utils/utils.py:142`; `config_initializer.py:220` |
| 10 | Eight loss hyperparameters that no live body reads | config |

Also seen: the five `*_counter` configs set `loss_reg_y_max`, which `mse` never reads.

### T4 · The two root causes (Agreed 2026-11-15)

**Root cause 1 — the body is split, so its parts must be told about each other.** A body is one
decision, but the code stores it as separate parts in separate places: width and activation in the
architecture file, the loss in `utils/`, the forecast map in inference. Nothing owns all of them.
Config fields act as the glue (`loss_reg`, `output_distribution`, `reg_activation`); separate fields
can disagree, and the code settles a disagreement by precedence, silently. Flags act as the glue
inside the code (`needs_latent`). Where no field or flag exists, the code guesses from a string or
a default.

**Root cause 2 — new things were added beside old things, never instead of them.** The distribution
families (ADR-067) were added under a byte-identical rule: "those fall through to the untouched
legacy path below (byte-identical)" (head comment, `HydraBNrecurrentUnet_06_LSTM4.py:100-102`). So
two generations run side by side, and the old generation's fields stay in every config and go dead
whenever a family is named.

Simon's words for it: "we made the body into a decision tree even though that was not actually
viable or useful or helped us. And we did it organically, letting old branches rot."

**Where ADR-081 sits:** it fixed the glue, not the split. It merged three fields into one string,
so a config can no longer disagree with itself, but the string still describes the parts because
the parts still exist separately underneath.

**The evidence that mix-and-match was never used:** the three fields can express 12 × 10 × 3 = 360
combinations. The 37 live configs use 5.

### T5 · Two candidate shapes for the code (neither decided)

**Shape A — an argument on the loss (Simon, 2026-11-15).** The head's activation is softplus,
hardcoded. The loss always receives the pre-activation output. A loss that needs the activated
value is constructed with `output_activation="softplus"`; the argument takes `"softplus"` or `None`.
Relu and leaky-relu are conceivable and are not implemented. A loss that must never see the
activated value does not have the argument at all.

**Shape B — one object per body (Proposed by Claude, 2026-11-15; not agreed).** The object needed
already exists: `DistributionFamily` (`distributions/base.py`, ADR-067) already owns `n_params`,
`activate`, `nll`, `mean`, `sample`, `self_zeroed` and `initial_raw_bias`. The four distribution
bodies already use it. The `mse` body and the listwise body would become two more such objects.
The activation and the loss are then methods on the same object and cannot be paired wrongly.
Cost: the class is called `DistributionFamily` and is governed by ADR-067, and `mse` and the
listwise loss are not distributions — so either it is renamed (which touches ADR-067) or its name
becomes untrue for two of its implementations.

**Open:** which shape. Simon said of Shape B's direction: "I think we are moving ahead."

### T6 · Facts about the code that the definition must survive (Checked)

**Which losses read the pre-activation output today** (`needs_latent = True`):
`ListwiseLevelLoss` (`listwise_level_loss.py:193`), `TobitLoss`, `TruncatedNBLoss`, `DenseNBLoss`.
The four distribution families and `mse` read the activated output.

**`mse` must score the activated output.** If it scored the raw output it would train
`raw ≈ y` while the model reports `softplus(raw)`. Under D8 this means the `mse` body applies
softplus inside its loss.

**`needs_latent` does two jobs.** Job 1: "where does my input come from." Job 2: "do I handle zeros
in my own loss" — read at `training_engine.py:712` (`apply_body_mask = body_supervision == "active"
and not use_latent`) and at `config_initializer.py:1563` (`validate_body_supervision_latent`,
`C-193`, which refuses `body_supervision='active'` for such a loss). Under D8, job 1 disappears and
job 2 needs its own name. Deleting the flag outright would turn the body mask back on for the
listwise and tobit losses.

**The model output already carries both values:** `ModelOutput.reg` (activated) and `.reg_latent`
(raw).

**The distribution bodies have no single activation.** `nb` and `truncated_nb`: softplus on both
channels. `zinb`: softplus, softplus, sigmoid. `mixture_nb`: sigmoid on the weight, softplus on
`mu1`, `theta1`, `theta2`, and `mu2 = mu1 + softplus(delta)`. A probability cannot come out of a
softplus, so D6/D7 ("softplus, hardcoded") is true per count channel and not per head for these.

**The `mse` body's output is a prediction in log1p space; a distribution body's outputs are
parameters in count space** (`hydranet_inference.py:1030`; `negative_binomial.py:7-10`).

**Settings outside the body, and how the 37 live configs set them:**

| setting | live values | what it needs from the body |
|---|---|---|
| `forecast_composition` (ADR-069) | 10 `self_zeroed`, 12 `soft_gate`, 15 `threshold_gate` | does the body make its own zeros |
| `emit_distribution` (ADR-073) | set in 0 of 37 | the body's parameter layout |
| `rollout_feedback` (ADR-070) | `sample` in all 37 | can the body draw samples |
| `body_supervision` (ADR-065) | `all` in all 37 | does the body handle zeros in its loss |
| `n_head_samples` | 4 in 27, 1 in 10 | can the body draw samples |
| `body_nll_denominator` | set in 0 of 37 | meaningful only for `truncated_nb` |
| `struct_penalty_weight`, `qs99_weight`, `pushforward_weight` | set in 0 of 37 | each is a term added beside the body's loss |

The same body (`nb`) is live under two different combinations with the gate.

**The gate does not read the body.** Nothing in the package calls `prob_positive`; the gate heads'
width does not depend on the body. The only link is the combination rule.

**The network below the head does not depend on the body.** The input width is fixed by
`input_channels == 3*output_channels + static` and the body's width sizes only the regression heads
(`HydraBNrecurrentUnet_06_LSTM4.py:131-155`).

**Where the transform does real work, and where it is a round trip.** The pipeline applies log1p once to the count series, and that one series is both the network's input (past months) and the target. The transform matters for the input. For the target alone it is transform-then-undo inside the loss. The body's count forecast is then log1p'd again on the way out (`hydranet_inference.py:1030-1062`), because it is fed back as the next month's input during rollout (`training_engine.py:550-557`) and the pipeline downstream undoes it. The round trip is not exact in 32-bit numbers (`04_roadmap.md` phase 1 note 1).

**The gate is not affected by the transform as configured today.** Its target is "count above a threshold" with threshold 0 (checked in `calm_lander` only), and a count is above zero exactly when its log1p is. Not checked: the order of threshold and transform if the threshold were not 0.

### T7 · The ranking body (Checked against `reports/2026-11-10_ranking_head_dossier/`)

- Its score is scale-free by design; only the order is read (`02_design.md` §3.4).
- Nothing downstream may read the score as a count (`03_harness_and_invariants.md` §1c item 4).
- Its real forecast is planned and not built: gate × ranking score × a draw from the observed
  counts at that rank (`02_design.md` §1; `04_roadmap.md` phase 5).
- `reg_activation: 'softplus'` on a ranking arm was decided by Simon on 2026-11-10, because ReLU
  re-creates the tie at zero (`03` §5).
- ⚠️ **Where the code reads the score as a count today, logged here because it was logged nowhere
  else:** the forecast pass-through at `hydranet_inference.py:1030` ("Standard: identity (reg is
  already the log1p-space point prediction)") and the feedback at `training_engine.py:827`
  (`fed = t1_pred`). Under D4 this is a known stage of the work, not a fault in the definition.
- The listwise transform defaults to `exp` in two places (`config_initializer.py:220`,
  `utils/utils.py:142`). The loss's own docstring (`listwise_level_loss.py:45-58`, citing Bai 2023)
  calls `softplus` the regression-compatible choice and `exp` the incompatible one. Under D2 these
  are two bodies.

### T8 · Falsification audit of the definition, 2026-11-15

Claim tested: "choosing a body is a well defined and contained choice in setting up a HydraNet."
Seven probes, read-only. **"Contained" mostly held** (the gate and the network below the head do
not depend on the body). Five findings; only the first has been discussed with Simon.

| # | finding | status |
|---|---|---|
| F1 | The ranking body has no count forecast (item 4) | **Withdrawn as a fault in the definition** — D4 |
| F2 | **Missing fact: what the body needs from the target.** Distribution bodies and the ranking body undo the transform inside the loss to get raw counts (`count_target_bridge.py`); the `mse` body takes the target as it arrives. | **Resolved by D11.** The definition gains one stated fact: "needs raw counts, yes or no". ⚠️ Gap left in the code: the check that protects the distribution bodies (`config_initializer.py:1348`, `C-198`) does not cover the ranking body, which does the same undo (`listwise_level_loss.py:241`). |
| F3 | I reported item 5 as two facts. Simon separated them: (A) can the body produce zeros — part of the body, item 5, an open research question; (B) which cells the body trains on — not the body. | **Resolved by D12.** Item 5 unchanged; `body_supervision` joins the settings outside the body; the body states "must be trained on all cells, yes or no". Today that fact is job 2 of `needs_latent` (T6). |
| F4 | **Per target was unstated.** The code has one body for all three targets (one `output_distribution`, one `loss_reg`, three heads of the same width and activation); the definition said "for a target". | **Resolved by D13.** One body per model; numbers may differ per target. |
| F5 | **A number can change the width.** For the unused quantile body, `n_quantiles` sets item 1, so the same-or-different test gives two answers for it. | **Resolved by D14.** The quantile head is to be deleted, so no body has a number that sets its width. |

### T8b · Gaps in the code that must be addressed (not part of the definition)

| # | gap | logged |
|---|---|---|
| G1 | **The ranking body assumes log1p targets and nothing checks it.** `ListwiseLevelLoss` undoes log1p inside the loss (`listwise_level_loss.py:241`) to get raw counts for its rating levels. The config check that refuses a non-log1p target (`config_initializer.py:1348`, `C-198`) asks "is this a distribution body?", so it passes the ranking body without looking. No other check was found in the config validators or the loss file. With another target transform the loss would recover wrong counts, assign wrong rating levels, and train on the wrong ranking with no error. Not triggered today: every current config uses log1p. Under D11 the check should ask "does this body need raw counts?". Simon, 2026-11-15: "we will need to address this." | Here only. ⚠️ Not in the risk register, and this section is deleted when the ADR is done. |

### T9 · Open questions and unknowns

- **Open:** Shape A or Shape B (T5).
- **Open:** none of F1–F5 (T8). The quantile head's code is still to be deleted (D14).
- **Open:** if Shape B, what the class is called and what happens to ADR-067's name for it.
- **Open, for the ranking programme (Simon, 2026-11-16; reasoning only, nothing measured):**
  *raw score or softplus for the later draw from the empirical counts.* If the draw uses only each
  cell's rank, raw and softplus give the same result, because softplus keeps the order. If it uses
  how far apart two scores are, softplus squeezes the low scores together near zero and the raw
  score is the better thing to read. The raw score stays available next to the activated one, so
  this can be decided when the draw is designed.
- **Open, for the ranking programme (Simon, 2026-11-16):** *does ranking throw away how different
  two counts are?* Separate from the activation; it comes from the loss. The loss is not purely
  ordinal: it sorts counts into levels by powers of two (checked: 14 levels for `y_max` 10,000), so
  confusing far-apart levels costs more than confusing neighbours. But nothing trains the gaps
  between output scores to mean anything; whether they carry signal is an empirical question. And
  if counts come back from the empirical distribution by rank, the size differences come from the
  data, so the model cannot say "this month is three times worse than last month" except through
  how many cells the gate fires. Simon's related observation: the count bodies (negative binomial)
  already could not tell large from not large. ⚠️ Logged here only; this section is deleted when the
  ADR is done, so it needs a permanent home in the ranking dossier, which only Simon adds rows to.
- 🔴 **Checked 2026-11-15:** all ten live `*_counter` and `*_sorter` configs set
  `rollout_feedback: 'sample'` with `output_distribution: 'standard'`. The refusal of that pair is
  real: `tests/distributions/test_rollout_feedback.py::test_rollout_feedback_sample_requires_family`
  passes. No such config was run end to end.
- ⚠️ **Checked 2026-11-15:** 0 configs set `loss_reg_listwise_transform`, so all five ranking
  configs get the default `exp`. Under D2 `exp` and `softplus` are two bodies, so the live one is
  picked by a default.
- **Open:** what job 2 of `needs_latent` is called, and whether it is the same fact as
  `self_zeroed` (T6, F3).
- 🔴 **Unknown, not run:** `calm_counter` and `calm_sorter` both set `rollout_feedback: 'sample'`
  with `output_distribution: 'standard'`. `hydranet_inference.py:297-303` raises a `ValueError` for
  that pair. Both sides were read; nothing was run. It is not known whether those two models can
  currently produce a forecast.
- **Unknown:** how the dossier's arm `eager_sorter` (control `calm_lander`) relates to the
  `calm_sorter` / `calm_counter` pair.
- **Checked, relevant to ADR-081 later:** `training_engine.py:1337` and `:1401` read
  `config.get("output_distribution", "standard")`, so deleting that field makes both resolve to "no
  distribution" with nothing raised. With pydantic `extra="allow"`, an extra key that collides with
  a computed field is dropped from `model_dump()` (pydantic 2.13.4, measured 2026-11-15).

### T10 · Platform doctrine this ADR must answer to

- views-hydranet ADR-003, Law 1 Fail Loud: "Prohibited: Magic default values." Law 2 Zero-Magic:
  "No hardcoded strings, magic indices, or positional assumptions." Law 5: "If the config is
  silent, the data remains raw."
- views-baseline ADR-021: "A parameter cannot be both 'reproducibility-critical' and 'silently
  defaulted'."
- pipeline-core ADR-040 No-Sniffing: "All meaningful semantics must be explicitly declared.
  Inference of semantics across component boundaries is forbidden."
- Simon's design preferences: correctness → clarity → simplicity → testability → maintainability →
  extensibility → elegance; duplication before the wrong abstraction; explicit state; the least
  machinery that solves the actual problem.
- Simon, 2026-11-15: "Hydranet was once built so that a data scientist or researcher could follow
  through the code."
