# ADR-079: One Field Names the Body

**Status:** Accepted
**Date:** 2026-11-13 · accepted 2026-11-14
**Deciders:** Simon Polichinel von der Maase
**Informed:** views-hydranet contributors; views-models roster owners

---

## Summary (read this first — self-contained)

*What this section gives you: the whole decision in one page, with the words it uses defined
first.*

**The words used in this document.** Each HydraNet target has **two** output heads. The
classification head — the **gate** — predicts whether the next value is above zero; it is trained
with binary cross-entropy and this ADR does not change it. The regression head — the **body** —
predicts how large that value is, and it is the whole subject of this ADR. **Emit** means what the
model draws forecasts from: either a distribution whose parameters the body produces, or nothing at
all, in which case the body's single number *is* the forecast. **Head width** means how many
numbers the body's last layer produces per grid cell.

A HydraNet regression head is currently described by **two** config fields — `loss_reg` and
`output_distribution` — plus a third, `reg_activation`, that sometimes applies and sometimes does
not. The objective that actually trains the body is **inferred** from the combination, by statement
order inside `choose_loss`, and the inference discards one of the fields without saying so.

**The measured consequence: 27 of 37 live configs state a loss they do not train.** This is not a
latent hazard awaiting a trigger. It is the fleet's normal condition today.

This ADR replaces all three fields with **one**, whose value is drawn from a **closed, declared
vocabulary**. Each value names the loss, the activation and the emit distribution together. A
combination that is not in the vocabulary cannot be written down, so there is no pair of fields to
cross-check against each other. There is exactly one refusal left: a value that is not in the
vocabulary is rejected when the config is built.

⇒ The change **removes** a branch from `choose_loss` and **removes** three config fields, adding
one. It adds a vocabulary table and a rename table. No new layers, no new abstractions, no
compatibility framework.

---

## 1. Context

*What this section shows: one function decides what trains the body, it reads two fields in a
fixed order, and 27 of the 37 live configs state a loss that order throws away.*

**Problem.** One function, `choose_loss`, turns a config into the loss object that training will
minimise. These are the first two branches it tries, at `views_hydranet/utils/utils.py:220-222`:

```python
if _family is not None:
    criterion_reg = FamilyLoss(_family).to(device)
elif config["loss_reg"] in ("hurdle_nb", "dense_nb"):
    ...
```

`_family` is set earlier from `output_distribution`. So if the config names a distribution, the
first branch is taken and `loss_reg` is never read at all — the field is not overridden, not
warned about, not logged. It is skipped. The module comment states the rule outright
(`:192-194`): *"if output_distribution names a registered distribution family, the reg loss is the
family NLL — loss_reg is irrelevant for a family."* The `loss_reg` validator
(`config_initializer.py:844-854`) derives its accepted set from `LOSS_REG_REGISTRY.keys()` and never
consults `output_distribution`. Validator and executor therefore disagree by construction: the
validator accepts a value the executor will discard.

**Census, checked 2026-11-13** across `views-models/models/*/configs/config_hyperparameters.py`:

| `loss_reg` | `output_distribution` | configs | trains |
|---|---|---:|---|
| `mse` | `nb` | 15 | NB likelihood |
| `mse` | `truncated_nb` | 7 | truncated-NB likelihood |
| `mse` | `mixture_nb` | 5 | mixture-NB likelihood |
| `mse` | `standard` | 5 | MSE |
| `listwise_level` | `standard` | 5 | listwise ranking |

**Assumption that is no longer true.** ADR-067 introduced the family path as a *strangler fig* — a
temporary seam, with legacy configs falling through unchanged. With 27 of 37 configs on the
strangled path, the scaffolding has become the structure, and the field it strangles (`loss_reg`)
is dead weight that still validates.

**Urgency.** The ranking experiment requires a body trained by a ranking objective. Expressing that
today means either accepting a config whose stated loss is discarded, or stripping the distribution
off the head to reach the loss — which then forfeits draws and gate composition for reasons that
have nothing to do with ranking. The defect is now blocking work, not merely untidy.

**Constitutional basis.** ADR-003, *Authority of Declarations Over Inference*: *"All meaningful
semantics must be explicitly declared. Inference of semantics across component boundaries is
forbidden."* The present design infers the body's objective from a field that names its emission.
This ADR is the application of ADR-003 to the body.

---

## 2. Decision

*What this section states: the single field, the five values it may take, how today's configs
translate into them, and what the change does not cover.*

**Statement:** *We will name the regression body with a single config field drawn from a closed,
declared vocabulary, so that the objective a config trains is read, not inferred.*

### The three dimensions a value encodes

*What this part gives you: the three things every value names, and how the values are spelled.*

1. **Loss** — what is minimised.
2. **Activation** — the map from raw head output into the valid range.
3. **Emit distribution** — what is drawn from at forecast time, or nothing.

**Naming format:** `<loss>_<activation>_<emit>`. **Every value is spelled out in full, and this
is settled, not a preference to be revisited:** a name that has to be looked up somewhere else is
what produced the 27 lying configs. The spelling repeats itself on the family rows — a distribution
already fixes its own loss and activation — and that redundancy is accepted deliberately, with its
cost stated in §4. The format is a reading aid for people, not a grammar for code: the values are a
fixed list that is matched whole, never split on underscores.

### The declared vocabulary

*What this part gives you: the five values that may be written today, and the rule for adding a
sixth.*

Exactly the combinations that are live in the fleet today. (Other combinations have been trained
here in the past — `zinb` and the quantile head among them — and have no row, because no live
config uses them. Membership is *currently in use*, not *ever tried*.) The repository's registry of
implemented losses is deliberately wider than this table; see §4 for what that means for the ten
losses with no row.

| value | width | activation | loss | emits |
|---|---:|---|---|---|
| `mse_softplus_point` | 1 | softplus | squared error | a point, no draws |
| `listwise_softplus_point` | 1 | softplus | listwise ranking | a point, no draws |
| `nll_softplus_nb` | 2 | softplus | NB likelihood | NB draws |
| `nll_softplus_truncated_nb` | 2 | softplus | truncated-NB likelihood | truncated-NB draws |
| `nll_softplus_mixture_nb` | 5 | softplus | mixture-NB likelihood | mixture draws |

**Adding a row is a deliberate act with an owner and a stated reason. It is not a config edit.**

### Renaming the existing configs

*What this part shows: how each of today's five field combinations becomes one value, and why no
model changes what it computes.*

All 37 configs in views-models are rewritten once, by hand. Five combinations are in use today, and
each becomes one value.

| the config says today | what it has always trained | becomes |
|---|---|---|
| `mse` + `standard` + `softplus` | squared error | `mse_softplus_point` |
| `listwise_level` + `standard` + `softplus` | listwise ranking | `listwise_softplus_point` |
| `mse` + `nb` | the NB likelihood | `nll_softplus_nb` |
| `mse` + `truncated_nb` | the truncated-NB likelihood | `nll_softplus_truncated_nb` |
| `mse` + `mixture_nb` | the mixture-NB likelihood | `nll_softplus_mixture_nb` |

**No model changes what it computes.** Read the middle column: in the last three rows the word
`mse` was never used. Those models have always trained the likelihood of their distribution, and
`mse` sat in the config unread. The new name on the right says what the model was already doing.

One further field survives the rename, named here so it is not a surprise:
`body_nll_denominator`. It applies to `truncated_nb` and to nothing else, it chooses what the
likelihood averages over, and it keeps its own name and its own validator. All seven live
`truncated_nb` configs leave it at its default, `positives` (checked 2026-11-14), so it does not
affect the rename above.

The last three rows do not mention an activation because those configs never used one. When a
distribution is named, the model takes the activation from the distribution and ignores the
`reg_activation` field, so that field is simply deleted.

Anything not in this list is refused when the config is built. The case to watch is a `standard`
config with no activation set: it trains with relu, it has no name in the vocabulary, and so it
stops with an error instead of being renamed into something untrue.

*Both checked 2026-11-14. The relu default and the distribution override are in
`HydraBNrecurrentUnet_06_LSTM4.py`, lines 119 to 127. All ten live `standard` configs set
`reg_activation: 'softplus'`: `calm`, `keen`, `mellow`, `nimble` and `quiet`, each as a `_counter`
and a `_sorter`.*

### Why `width` is a column in that table and not a field in the config

*What this part shows: the head's width is the emit distribution's parameter count, so putting it
in the config would create a second place to record a number that is already fixed.*

The body ends in one convolution. The number of channels that convolution produces — the head
width — is the number of values the loss then receives for each grid cell.

A distribution needs one value per parameter. The negative binomial has two parameters, `mu` and
`theta`, so its head is two channels wide and both channels go through softplus
(`distributions/negative_binomial.py:33-34`). A point prediction has one value and needs one
channel. The five-channel mixture row is two NB components — four parameters — plus the one weight
that mixes them (`distributions/mixture_negative_binomial.py:45-46`). ⇒ **The
width is therefore not a choice anyone makes. It is the emit distribution's parameter count, read
off the distribution**: `reg_out_ch = self._family.n_params`
(`architectures/HydraBNrecurrentUnet_06_LSTM4.py:153`).

This is why the table states the width as information and the config does not state it at all.
Putting it in the config would create a second place to record a number that is already fixed
elsewhere, and two places that record the same number can disagree — which is the same defect this
ADR removes, relocated to a new field.

> **Aside, deliberately not pursued here.** "One parameter per channel of one shared convolution"
> is how the model is built today, not the only way to build it. The parameters could come from
> separate convolutions, or from decoder branches that diverge earlier, which may well be better —
> today both NB parameters are asked of the same weights at the very last layer. Whether to change
> that is **issue #600** and is out of scope for this ADR. What is in scope is the *count*: however
> the parameters are produced, how many there are is the distribution's property, never the config
> author's.

### Worked example — the ranking experiment

*What this section shows: the experiment that forced this ADR, and the one field that expresses
it.*

Two models. Same trunk, same two heads, same gate trained with binary cross-entropy on all
observations. Only the body differs.

| | arm 1 — control | arm 2 — ranking |
|---|---|---|
| `body` | `nll_softplus_nb` | `listwise_softplus_point` |
| what trains the body | NB likelihood | listwise ranking |
| head width (derived) | 2 | 1 |

The experiment asks which of those two bodies puts cells in the better order. Nothing else is
meant to vary.

**That pair cannot be written today.** There are two ways to try, and each changes more than the
body.

The first way is to write `loss_reg: 'listwise_level'` beside `output_distribution: 'nb'`. The
validator accepts it. `choose_loss` then builds the NB likelihood and never constructs the ranking
loss at all (`utils.py:220`). Arm 2 would train as a second copy of arm 1, and the experiment would
read as a clean null. This is how the defect was found.

The second way is to write `output_distribution: 'standard'`, which does reach the ranking loss. It
also strips the emit distribution off the head. Arm 2 then differs from arm 1 in three places — the
loss, the emission, and what the composition validator accepts — for a question about one.

**One difference survives this ADR.** A ranking body has no emit distribution, and
`forecast_composition: 'threshold_gate'` is refused for a body that has none. (Checked 2026-11-13
by constructing `HydraNetConfig` on both specifications.) So the two arms still differ in
composition as well as in body. This ADR makes the body difference a single stated field; it does
not reduce the whole experiment to a single field. Whether a point body should be allowed the
hard-threshold gate is a composition question and belongs to ADR-069.

### Where the counts come from, once the body may be trained to rank

*What this section answers: the objection that a rank-trained body no longer produces fatalities.
The objection is correct, and counts are recovered by a later step that this ADR does not govern.*

A reader should stop at arm 2 and object. If the body is trained by a ranking loss, the number it
produces is not a magnitude. Its units are arbitrary, and its only meaning is the order it puts
cells in. This system exists to deliver counts, so where did the counts go?

Three facts answer that, in order.

**First, until now the body has estimated magnitude** — how large the next value is, in fatalities.
Every row in the vocabulary except the listwise one does that.

**Second, the ranking arm asks a different question of the same head.** Trained differently, is it
better at putting cells in the right order? Its output is an ordering, not an amount.

**Third, counts are recovered afterwards, by a step applied to the finished forecast.** Take the
body's number, use it to rank cells, then draw each cell's count from a **non-parametric empirical
distribution** of observed counts. That step's decisive property is that it does not care how the
number was produced — point or distribution, likelihood or ranking objective, any row in the
vocabulary. It consumes one number per cell and nothing else about the head.

Because that step is indifferent to the head, it is not a property of the head, and so no `body`
value can name it. That is why it is a separate step and not a sixth row.

**This ADR's only job in that story is to let the difference between arm 1 and arm 2 be stated**
rather than inferred from two fields. It takes no position on which objective is right, and it does
not turn a ranking output into counts.

⛔ **No decision record governs the recovery step today.** ADR-073 §5 describes it as a route the
programme intends to take. It has never been specified as a design, and no ADR owns it. Nothing in
this ADR depends on it, and nothing in it constrains this ADR — but a reader who asks where the
counts went is owed a document, and there is not one. Recorded here as an acknowledged gap.

### Why `listwise_softplus_nb` is not a row

*What this section shows: the vocabulary admits only combinations that have been trained, and
this one has not.*

`listwise_softplus_nb` would be a body trained by ranking that still emits NB draws. It is absent,
and the reason it is absent shows what the vocabulary is for.

The listwise loss gives each cell **one** score and compares those scores across cells. An NB head
produces **two** numbers per cell. To rank such a head, something must first reduce two numbers to
one — the distribution's mean, one of the two channels, some quantile. That reduction is a
modelling decision. It has never been made here, and nothing has been trained with it.

So the row is missing not because the combination is forbidden, but because no evidence supports it
yet. Adding it would mean choosing the reduction, training it, and writing down the result — which
is exactly what *"adding a row is a deliberate act with an owner and a stated reason"* requires.

One warning, because this row looks like an answer to the previous section and is not. A
rank-trained head that also emits NB draws would be a head doing two jobs at once, where the
recovery step above touches no head at all. The absent row is an open design question about heads,
not the counts question in disguise.

### In-scope

- The regression body: `loss_reg`, `output_distribution`, `reg_activation`.
- `choose_loss`'s regression path and the validators for those three fields.
- The rename table from the three old fields to the new value, including saved artifacts (§2).

### Out-of-scope

- ⛔ **Head width.** Derived from the emit distribution's parameter count, already correct, and it
  **must not become a config field** — see *"Why `width` is a column in that table and not a field
  in the config"* above. Whether those parameters should share one convolution is #600, separately.
- ⛔ **The classification head.** `loss_class` and the gate's activation are coupled the same way
  and deserve the same treatment, but the gate is not blocking anything and is not changed here.
  Leaving it asymmetrical is a known cost, recorded in §4.
- ⛔ **`forecast_composition` and `gate_threshold`.** Unchanged. They continue to ask whether the
  body has an emit distribution; after this ADR they ask the `body` value instead of
  `output_distribution`.
- ⛔ **Any head × loss compatibility matrix.** A general matrix with a legality table is the correct
  design *once several combinations are known to work*. Today there are five. A closed vocabulary
  is strictly smaller and has no legality table at all.

---

## 3. Rationale & Integrity Impact

*What this section argues: why one closed vocabulary is better than the smaller fix and better
than two free fields, and what the change does to this repository's fail-loud rule.*

**Logic — why this over the obvious alternative.** The smaller-looking fix is to make the family
override conditional: *the family wins unless `loss_reg` names a non-family loss*. It is rejected
on three grounds. It adds a second precedence rule where the complaint was that one rule was
already invisible. It leaves all 27 configs still stating a loss they do not train, so the record
keeps lying. And it makes the answer to *"what does this config train?"* depend on which loss was
named — the reader must now know the rule **and** the loss taxonomy. The vocabulary version deletes
the branch and the question.

**Why a closed vocabulary rather than two free fields.** A reasonable objection is that loss and
emit distribution are two separate ideas and should therefore be two separate fields, with
something to check that the pair makes sense. The objection fails because **the two are not
independent in the first place**, and the reason is counting.

The emit distribution fixes the head width, as described in §2. The head width then fixes which
losses can consume that head's output, because a loss must do something with every number it is
handed. A two-channel NB head hands its loss a `mu` and a `theta`, and the NB likelihood is the
thing that consumes exactly that pair; squared error and the listwise loss each take one number per
cell, so neither can be applied to that head without first inventing a reduction (§2, *"Why
`listwise_softplus_nb` is not a row"*). **The coupling is arithmetic, not stylistic.**

⇒ Two free fields would therefore need a legality table whose job is to refuse combinations that
arithmetic already refuses. A closed vocabulary refuses them by not containing them, and needs no
table.

**Fortress State.** Reproducibility improves directly: a saved config states the objective that
trained the artifact, so provenance no longer requires knowing a precedence rule and a registry.
Numerical behaviour is unchanged — each vocabulary value constructs the identical criterion object
that its old combination of fields constructed.

**Fail-Loud.** The present failure is the opposite of loud: `C-324`, the inert-knob signature — a
value registered, dispatched and silently discarded, discovered only by a result that looks clean.

This ADR removes one instance of that failure, not the whole class. ⚠️ **Other fields can still be
ignored.** `body_nll_denominator` means nothing outside `truncated_nb`, and the listwise
hyperparameters mean nothing outside the listwise body. The difference is that those fields
**raise** when they would be ignored rather than passing silently —
`validate_body_nll_denominator_pair` (`config_initializer.py:49-94`) refuses the inert pair and
cites `C-324` by name. ⇒ The rule this ADR follows is therefore not "one field only"; it is **a
field that would not be read must stop the run, not be skipped.**

An unknown `body` value fails at config construction, before a GPU is allocated. What that error
must say is specified in #602.

**Relation to ADR-073.** ADR-073 holds two separate things, and this ADR touches only the first.

1. *Emit need not equal train.* ADR-073 established that the distribution a model emits from need
   not be the one it was trained on, shipped that capability default-off, measured one arm and
   declined it, leaving the axis open. This ADR neither reopens nor closes that question — it makes
   the answer **visible in the name**, because a value whose loss and emit differ says so on its
   face, where today it is inferable only from several fields plus a precedence rule.
2. *The empirical count step.* ADR-073 §5 also describes the post-processing route that recovers
   counts. ⛔ **It is downstream of everything this ADR names and is governed by no ADR at all.**
   §2, *"Where the counts come from"*, states it and records the gap. Out of scope here.

---

## 4. Consequences

*What this section lists: what improves, what it costs, what is left for an owner to decide, and
what is refused outright.*

### ✅ Positive

- [x] Simplification of `choose_loss` — one branch deleted; the function becomes the registry
      dispatch its own docstring already claims it is.
- [x] Simplification of the config schema — three fields become one.
- [x] One instance of `C-324` removed — the silent discard of `loss_reg` by the family branch.
      The class itself survives in other fields and is handled by making them raise, as §3 sets
      out.
- [x] Improved traceability for anyone reading a delivered model's sidecar: the recorded objective
      is the trained objective.
- [x] The ranking experiment's two bodies become two values of one field. The loss each arm trains
      is stated in the config instead of being discarded by `choose_loss`.

### ⚠️ Negative

- [ ] **Breaking change requiring migration of 37 roster configs** in views-models. Mechanical —
      three fields become one, by the fixed table in §2 — and behaviour-identical, but it is a real
      commit in a repository this ADR does not own.
- [ ] **Saved artifacts carry the old fields.** Every delivered checkpoint's sidecar holds
      `loss_reg`, `output_distribution` and `reg_activation`. New code must either refuse them or
      translate them, using the same three-field table as the configs. That table is the one piece
      of machinery this design cannot avoid, and it must be chosen on day one, not scheduled.
      ⭐ The sidecar makes this determinate: for a legacy head it records the activation function's
      actual name, and for a distribution family it records `None` because the family supplies the
      activation (`train/train_model.py:116-128`, checked 2026-11-14). So no sidecar has to be
      guessed at.
- [ ] **The values are verbose, and for family rows the activation segment is not the whole
      truth.** `nll_softplus_nb` is longer than `nb`, and for a family row the loss and the
      activation repeat what the distribution already fixes. The segment is also imprecise for a
      multi-parameter family, because such a family does not have *one* activation: the mixture
      applies `sigmoid` to its weight, `softplus` to two channels, and builds its second mean as
      `mu1 + softplus(delta)` (`distributions/mixture_negative_binomial.py:48-60`, checked
      2026-11-14). So `nll_softplus_mixture_nb` names the commonest of those maps rather than all
      of them, and the same would hold for `zinb` (`softplus, softplus, sigmoid`). For the point
      rows the segment is exact and load-bearing, because relu and softplus are both reachable
      there. ⇒ **This is the accepted price of the long spelling, not an open question.** The
      spelling is decided in §2 and the reason is the whole point of this ADR: a name that has to
      be looked up elsewhere is what produced the 27 lying configs.
- [ ] **The config is left asymmetrical** until the classification head receives the same
      treatment. Recorded, not fixed.
- [ ] **Ten of the twelve registry losses have no declared row** — `count_mean`, `mae`, `huber`,
      `shrinkage`, `lognormal_nll`, `pareto`, `tobit`, `quantile`, `hurdle_nb`, `dense_nb`. Only
      `mse` and `listwise_level` are named by a live config (`LOSS_REG_REGISTRY` has twelve keys,
      checked 2026-11-15). **Decided 2026-11-15: all ten stay, and none of them gets a row now.**
      They are tested, working implementations that are simply not operational. A row is earned by
      trying a loss in a real configuration and writing down the result, one loss at a time — the
      same rule §2 states for adding any row. ⇒ The registry is the shelf of implemented losses;
      the vocabulary is the list currently on offer. **The two are not meant to match, and an entry
      with no row must not be read as dead code to delete.**

### Explicitly refused

- ⛔ **A generic `loss_kwargs` dictionary** for per-loss hyperparameters. An open bag of keys is
  precisely the failure this ADR removes — a key nobody reads, silently ignored, discovered on a
  paid run. Loss hyperparameters (`loss_reg_y_max`, `loss_reg_listwise_transform`, `loss_reg_sigma`,
  `loss_reg_theta_init`, `n_quantiles`, `body_nll_denominator`) remain **named schema fields with
  validators**, each read only by the `body` values that need it.

---

## 5. Validation

*What this section specifies: what must come out identical after the change, and the tests that
prove it — beginning with one that must be written before any code moves.*

**Invariants — what must come out the same afterwards.** This change is meant to alter how a model
is *described*, never what it computes. Each statement below is something that must hold both
before and after, so that a disagreement is a bug in the migration and not a new result.

1. For each of the five vocabulary values, `choose_loss` constructs an object of the identical
   type, with identical parameters, to the one its old combination of fields constructed.
2. Every delivered artifact remains loadable, producing the same criterion and the same emission
   behaviour, via the rename table.
   ⚠️ Including its **activation**: the sidecar's `reg_activation` must be read, not assumed. It
   is always recorded, so there is nothing to infer — see §4.
3. `forecast_composition` and `gate_threshold` behaviour is unchanged for every existing config.

**Tests (ADR-005).**

- ⛔ **Red, and it must be written and committed BEFORE any code moves:** a table-driven
  characterization test asserting which criterion object each of today's five live field
  combinations produces.

  ⚠️ **A partial test already exists, and it is not enough.** `tests/distributions/test_loss_wiring.py`
  pins two cases (checked 2026-11-14): `choose_loss` returns a `FamilyLoss` for `nb` and for
  `zinb`, and does not return one for `standard` with `mse`. It does not cover `truncated_nb`,
  `mixture_nb` or `listwise_level`. ⇒ And the gap that matters most: **no test anywhere asserts
  that `loss_reg` is discarded when a distribution is named.** Both of its cases use a `loss_reg`
  that agrees with the outcome, so the precedence itself is unpinned and nothing fails if it
  changes.

  Extending that file to all five live combinations — including one case where `loss_reg` names a
  non-family loss beside a family — is the first step of this ADR and the one that makes the rest
  of it safe.
- **Green:** the same table in the new spelling, asserting identical criterion types.
- **Green:** an unknown `body` value is rejected at `HydraNetConfig` construction, and the error
  lists every declared value. The listed set must be exactly the set that runs — a value that
  prints but does not train is the defect being prevented. Specified in #602.
- **Green:** the three-field rename is exercised on every row of the rename table, and
  `(mse, standard, unset)` and `(mse, standard, 'relu')` are both refused rather than renamed to
  `mse_softplus_point`.
- **Beige:** a saved sidecar in the old spelling loads through the rename table and yields the
  same criterion as before.

**Failure mode that would trigger reconsideration.** If the declared set grows past roughly ten
values, or if two rows are ever needed that differ only in one dimension with the others held
fixed, the closed vocabulary has become a combinatorial table written out by hand. At that point
the orthogonal design — separate fields plus an explicit compatibility table — is the correct
successor, and this ADR should be superseded rather than extended.

---

## 6. Implementation Notes

*What this section lists: every file that changes, the order the work must happen in, and the
references.*

**Location.**

- `views_hydranet/utils/config_initializer.py` — the `body` field, its vocabulary, the rename
  table, and removal of the `loss_reg` / `output_distribution` / `reg_activation` fields.
- `views_hydranet/utils/utils.py` — `choose_loss` loses the `_family` branch (`:220-221`) and
  becomes a single registry lookup.
- `views_hydranet/architectures/HydraBNrecurrentUnet_06_LSTM4.py` — reads width from the `body`
  value's family instead of `output_distribution`. Width stays derived.
- `views_hydranet/utils/hydranet_inference.py` — asks the `body` value whether there is an emit
  distribution.
- `views_hydranet/train/train_model.py` — writes the sidecar. It currently persists
  `output_distribution` and `reg_activation` per head type (`:116-128`); it must persist the
  `body` value instead.
- `docs/CICs/HydraNetConfig.md` — the field count and the contract.
- views-models: all 37 configs, three lines each becoming one. **Not this repository's to apply.**

**Order of work — do not reorder.** (1) characterization test on current code; (2) the field and
the vocabulary, with the branch deleted and the test rewritten to the new spelling; (3) the roster
migration.

**References.** Issue #601 (this change), #602 (the rejection message for an undeclared `body`),
#600 (head-parameter separation), #599 (`aggregate_method`
narrowing — same class: a validator vocabulary wider than what executes). ADR-003 (declarations over
inference), ADR-005 (testing), ADR-067 (the strangler fig this completes), ADR-069
(`forecast_composition`), ADR-073 (emit need not equal train). Risk register: `C-324`.
