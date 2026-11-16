# ADR-079: The Definition of a Body

**Status:** Accepted 2026-11-16. Simon read and approved the definition, the same-or-different test
and the two building rules (section 2). He did not read the supporting sections; they were checked
by three independent readers and against the code. A wrong sentence in them is corrected without
reopening the decision. The working notes behind this ADR are kept in
`reports/2026-11-15_body_definition_working_notes.md`.
**Date:** 2026-11-15
**Deciders:** Simon Polichinel von der Maase
**Informed:** views-hydranet contributors; views-models roster owners

**Related:** ADR-080 (the definition of a gate) · ADR-081 (one config field names the body — not to be
implemented until this ADR is done) · ADR-063 (regression-head activation) · ADR-067 (distribution families) ·
ADR-069 (`forecast_composition`) · ADR-070 (`rollout_feedback`) · ADR-073 (`emit_distribution`)

---

## Summary (read this first — self-contained)

This document defines one term: the **body** of a HydraNet model.

A HydraNet model forecasts fatalities for three targets. For each target it has two output parts.
One estimates whether there will be any fatalities; this document calls it the gate. The other
estimates how many; it is the regression head. **The body is the complete specification of the
regression heads: what they output, what the outputs mean, and how they are trained.**

The decision has three parts:

1. A body is chosen whole. Its parts cannot be set one by one.
2. A model has one body, shared by its three targets.
3. Two setups (model configurations) have the same body when everything about their regression heads is
   the same, or when the only difference is a number the body requires from the data, such as an
   upper bound on the counts.

Two rules for building bodies are decided with it. First, every output of the body gets softplus,
except an output that must lie between 0 and 1, which gets a sigmoid. There is for now no clear
use case for any other activation on the body. Second, the loss always receives the body's outputs
from before the activation, so that the activation and the loss that depends on it are written in
one place and cannot be combined wrongly.

The document also decides that one unused kind of head, the quantile head, is to be deleted from
the code. The deletion itself is done afterwards, as a separate step.

The document does not decide how a config names a body, and it changes no code by itself.

---

## 1. Context

This section says why a definition is needed: today a body is stored as separate pieces that can
disagree, and when they disagree nothing reports it.

### Words used in this document

- **Target** — one of the three kinds of fatalities the model forecasts: from state-based
  conflict (`sb`), non-state conflict (`ns`) and one-sided violence (`os`).
- **Setup** — one model configuration: everything that is chosen when a model is built and
  trained.
- **Gate** — the part of the model that estimates, per grid cell and target, the probability that
  the count is above zero. Its definition is a separate record (ADR-080).
- **Regression head** — the part of the model that estimates, per grid cell and target, how many
  fatalities. There are three, one per target, and all three are built the same way.
- **Body** — the term this document defines. In short: the way the three regression heads are
  built and trained.
- **Outputs before activation** — the numbers the regression head produces before any function
  such as softplus is applied to them. One cell and one target can have more than one such number;
  each is called a **channel**.
- **Activation** — the function applied to those numbers, for example softplus.
- **log1p** — the transform `log(1 + count)`. The data pipeline applies it to the fatality counts
  before the model sees them.
- **Config** — the settings file of one model. There are 37 HydraNet configs in use today.

### The problem

Choosing a body is one decision. The code stores it as four separate pieces, in three different
files: how many channels the head has, which activation it uses, which loss trains it, and how a
forecast is computed from it. No single place holds all four.

Three config settings are meant to keep the pieces consistent. They are called `loss_reg` (the
loss), `output_distribution` (the count distribution, if any) and `reg_activation` (the
activation). Because they are three separate settings, they can contradict each other. When they
do, the code applies a fixed order of priority: a named count distribution overrides the loss
setting. It reports nothing when this happens.

This is what that produces today. Every item was read in the code and in the 37 configs on
2026-11-15.

- 27 of the 37 configs name a loss that is not the loss they are trained with. They set
  `loss_reg` to mean squared error and also name a count distribution. The code then trains the
  distribution's likelihood and ignores `loss_reg`.
- `output_distribution` has ten allowed values. Four name a distribution. The other six name a
  way of building the head. So one setting answers two different questions.
- `reg_activation` is set in all 37 configs. The code reads it for only 10 of them.
- For some configs the activation is picked by checking whether the text of
  `output_distribution` begins with the letters "hurdle".
- The head returns its output twice, once before and once after the activation, and a single
  yes-or-no switch in the code decides two unrelated things: which of the two a loss reads, and
  whether the loss must be trained on every grid cell.

### The two causes

1. **The body is stored in pieces.** Because of that, extra settings and switches are needed to
   keep the pieces consistent, and each of them can be set wrongly.
2. **New ways of building a body were added next to the old ones, and the old ones were never
   removed.** The count distributions were added under a rule that existing behaviour must not
   change at all. So the older settings are still present in every config, and they have no effect
   whenever a count distribution is named.

### Only 5 of 360 combinations are used

`loss_reg` has 12 allowed values, `output_distribution` has 10 and `reg_activation` has 3. That
allows 12 × 10 × 3 = 360 combinations. The 37 configs use 5 of them.

### Earlier fixes

Earlier fixes changed the settings and left the pieces separate. The most recent one (ADR-081)
replaces the three settings with one. It is paused, because it was written before there was a
definition of what that one setting selects.

---

## 2. Decision

This section gives the definition of a body, the test for whether two setups are the same body,
two rules for building bodies, and the five bodies in use today. One further decision, to delete
the unused quantile head, is in section 4.

### The definition

**A body is one complete specification of what the regression heads compute and how they are
trained. It has a name, and it is chosen as a whole.**

A model has one body. Its three regression heads all follow it.

A body fixes the six things below. For items 1 to 5 it fixes the thing itself, and none of them
can be set separately from the others. For item 6 it fixes which numbers are required. Their
values are given per model.

1. **Width** — how many channels each regression head has, that is, how many numbers it outputs
   per grid cell.
2. **Meaning** — what those numbers stand for once the activation has been applied, and on what
   scale. The activation is part of this, so the activation is part of the body.
3. **Objective** — the loss that trains the body.
4. **Forecast** — how the body's forecast is computed from the outputs, and whether the body
   can draw random samples of the count (yes or no).
5. **Own zeros** — whether the body can itself forecast "no fatalities" for a cell, without the
   gate (yes or no).
6. **Required numbers** — numbers the body needs that are taken from the data and not chosen by a
   person. Each must be given. None has a default, and the body cannot be built without it. The
   value of a required number may differ between the three targets.

How a body gets its name is not decided here. That belongs to ADR-081.

**What "own zeros" means.** A body makes its own zeros if, used without the gate, it can say "no
fatalities in this cell". For each kind of body:

- *A count distribution, such as the negative binomial.* Yes. It gives the count zero a
  probability above zero, so a sample drawn from it can be zero.
- *The mean-squared-error body.* Yes. It predicts one number per cell. With softplus that number
  is never exactly zero, but it can be as close to zero as needed, and this document counts that
  as saying "no fatalities".
- *The zero-truncated negative binomial.* No. It gives the count zero a probability of exactly
  zero, so it describes only the positive counts and always needs the gate. This is the hurdle
  model (Mullahy 1986; the two-part idea is from Cragg 1971): one part decides zero or positive,
  and a zero-truncated count distribution describes the positive counts.
- *The ranking body.* Not decided yet. Its output is a score, and how a count forecast is made
  from that score has not been built, so the question cannot be answered now.

### Two facts every body states

Each is a yes or a no. Neither is an extra choice, because each is determined by the loss.
They are written down so that settings outside the body can be checked against them.

- **Converts the target back to counts** — the data arrives transformed with log1p. Some losses
  need the fatality counts themselves and undo the transform inside the loss. The body states
  whether its loss does this.
- **Must be trained on all cells** — whether the loss needs every grid cell, including the cells
  with zero fatalities.

### The test: same body or different body

- If two setups differ in any of items 1 to 5, they are different bodies and have different names.
- If two setups differ only in a required number (item 6), they are the same body.

Example of different bodies. The ranking body is trained with a ranking loss: a loss that rewards
giving higher scores to the cells with more fatalities. That loss receives the score from before
the head's activation (Rule 2 below) and applies a function of its own to it. That function can be
an exponential or softplus, and the choice changes what the loss computes. It is separate from the
head's activation, which is softplus in both cases and is applied to the score the model reports.
The exponential version and the softplus version are different objectives (item 3), so they are
two bodies.

Example of the same body. The ranking loss needs an upper bound on the counts, called `y_max`,
taken from the data. Two setups that differ only in `y_max` are the same body.

Two bodies may be almost identical. That is accepted. Repeating some code is cheaper than
combining two things that later turn out to differ.

### Two rules for building bodies

These are not part of the definition. They say how every body is built.

**Rule 1. Every output of the body gets softplus, except an output that must lie between 0 and 1,
which gets a sigmoid.** "Output of the body" means every channel of every regression head. The
rule is about the body only. It says nothing about the gate or about the layers inside the
network. For now there is no clear use case for an output of the body that needs anything other
than these two activations. The activation is written into the code. It is not a default that a
config can change, and there is no config setting for it. Softplus is needed wherever a number
must be positive: a count, a mean, a dispersion.

| body | channels | activation |
|---|---|---|
| mean squared error | one: the prediction | softplus |
| ranking, exponential version | one: the score | softplus |
| negative binomial | two: mean, dispersion | softplus on both |
| zero-truncated negative binomial | two: mean, dispersion | softplus on both |
| mixture of two negative binomials | five: a weight, the first mean, the first dispersion, an increment, the second dispersion | sigmoid on the weight, softplus on the other four; the second mean is the first mean plus the increment |

For the ranking body the score does not have to be positive, because only its order is used. It
gets softplus under the same rule, and softplus does not change the order: a higher number before softplus always
gives a higher number after it. ReLU would change the order, because it turns every negative
number into the same zero.

**Rule 2. The loss always receives the body's outputs from before the activation.** A loss that
needs the values after the activation applies the activation itself. The reason: the activation
and the loss that depends on it are then written in one place, so the two cannot be combined
wrongly. The gate is already built this way (ADR-064, the record on the gate's activation).

*Corrected 2026-11-16.* This rule first ended with the sentence "It also means the regression
head returns its output once and not twice." That sentence was wrong and is removed. About
twenty places in the code need the values after the activation: the forecast, the feedback during
a multi-month forecast, the sampling and two penalty terms. So the model keeps reporting the
activated values as well. The rule itself is unchanged.

### The five bodies in use today

Each row is one body, with the same names as in the table under Rule 1. The count is the number
of configs, out of 37.

| body | configs | width | meaning after activation | objective | forecast | samples | own zeros | converts target to counts | all cells required | required numbers |
|---|---|---|---|---|---|---|---|---|---|---|
| mean squared error | 5 | 1 | a prediction on the scale of the transformed target | mean squared error | the prediction itself, on the transformed scale | no | yes | no | no | none |
| ranking, exponential version | 5 | 1 | a score; only its order is used | ranking loss with the exponential | not built yet | no | not decided | yes | yes | `y_max` |
| negative binomial | 15 | 2 | mean and dispersion of the count | negative binomial likelihood | the distribution's mean | yes | yes | yes | no | none |
| zero-truncated negative binomial | 7 | 2 | mean and dispersion of the count | zero-truncated negative binomial likelihood | the distribution's mean | yes | no | yes | no | none |
| mixture of two negative binomials | 5 | 5 | a weight, two means, two dispersions; the second mean is the first mean plus an increment | mixture likelihood | the distribution's mean | yes | yes | yes | no | none |

The ranking row has two cells that are not filled. It has no count forecast yet, and whether it
makes its own zeros is not decided. These are stated exceptions to items 4 and 5. Building that forecast is
planned work in the ranking programme, the ongoing research on whether a ranking loss orders
cells better than a count likelihood. It is not part of this decision.

The ranking row is the exponential version because the five ranking configs do not choose a
version and the code's default is the exponential. A body should not be selected by a default.
Section 5 names this as a condition that fails today.

### In-scope

- The definition of a body and the same-or-different test.
- The decision to delete the unused quantile head (section 4).
- The two building rules.
- The two facts every body states.
- One body per model.

### Out-of-scope

Each item below stays a separate setting or a separate decision.

- **The gate.** Defined in ADR-080.
- **How a config names a body.** ADR-081, paused.
- **The transform applied to the data.** The body states whether its loss converts the target back
  to counts. It does not choose the transform. Whether log1p is the right transform is an open
  question, so this document does not fix it.
- **Which grid cells the body is trained on.** A separate setting (`body_supervision`, ADR-065).
  The research question behind it, whether to train the body only on cells with fatalities, is
  paused. Its code stays.
- **How gate and body are combined into the forecast.** A separate setting
  (`forecast_composition`, ADR-069).
- **Which distribution the forecast is reported from.** A separate setting (`emit_distribution`,
  ADR-073).
- **What is fed back into the model during a multi-month forecast.** A separate setting
  (`rollout_feedback`, ADR-070).
- **Different bodies for different targets.** There may be a use for this later. It is not
  designed for now.
- **How a body with several channels produces them.** Today one final layer produces all of them.
  Producing them from separate layers is a possible change (GitHub issue #600).

Four of the items above are config settings that depend on the body: which cells it is trained
on, how gate and body are combined, which distribution is reported from, and what is fed back.
This document calls them the four neighbouring settings. Each is checked against what the body
states. If one of them asks for something the body cannot do, building the config must stop with
an error.

---

## 3. Rationale & Integrity Impact

This section gives the reasons for each part of the decision, and says which earlier records it
changes.

**Why a body is chosen whole.** The three old settings allow 360 combinations and 5 are used. For
every unused combination the code must either support it or reject it. Today it does neither: it
follows the order of priority and reports nothing. A model that is trained with a different loss
than its config names is a scientific error that nobody is told about.

**Why one body per model.** The code builds the three regression heads identically today, and no
current work needs them to differ. Allowing a different body per target would add choices that
nobody uses.

**Why softplus, and not ReLU.** ReLU was tried and it failed. The outputs before activation became
negative on every grid cell, including cells with fatalities. ReLU turns a negative number into
exactly zero and gives no gradient there, so training could not move the outputs back. This is
entry C-178 in the risk register and the subject of ADR-063. The run was in July 2026 and was not
repeated for this document.

**Why not leaky ReLU.** Leaky ReLU can give negative numbers. A count cannot be negative, and
neither can the mean or the dispersion of a count distribution. This is a reason and not a
measurement: leaky ReLU has not been run in this project.

**What softplus gives.** Its result is always above zero, although it can be extremely close to
zero. It gives a gradient everywhere. It has no upper limit.

All 37 configs already set softplus. Writing it into the code removes a choice that no config
makes.

**Why the loss receives the outputs from before the activation.** The head then no longer needs
to return its output twice, and the switch described in section 1 no longer has to decide which of
the two a loss reads. The activation and the loss that depends on it are written in the same place,
so they cannot be combined wrongly.

**Why the transform is not part of the body, although the activation is.** Both are functions
applied to numbers, so they could be treated alike. They differ in what they affect. The
activation affects only the regression head's outputs. The transform is applied to the data, and
that transformed data is also the model's input and is used by the steps after the model. So the
transform cannot be chosen by the body alone.

**Earlier records this document changes.**

- **ADR-063 (the activation of the regression head).** Two parts of it no longer apply: that the
  mean-squared-error body uses ReLU unless a config says otherwise, and that a config setting can
  override the activation. Two parts still apply: that hurdle bodies use softplus, and its
  reasoning against ReLU. (A hurdle body is one that describes only the positive counts and relies
  on the gate for the zeros. The zero-truncated negative binomial is the one in use.)
- **ADR-067 (the count distributions).** Not changed. A count distribution still defines its own
  activation, within Rule 1.
- **ADR-081 (one config setting names the body).** Paused. It must be read again against this
  definition before anything in it is built.

---

## 4. Consequences

This section lists what improves, what it costs, and what this document rules out.

### Positive

- The question "which body does this model have" has one answer, and the answer covers everything
  about the regression heads.
- The four neighbouring settings can be checked against what the body states, where today the code
  compares text values.
- Once the code follows the definition, a config cannot name one loss and be trained with another.
- The ranking programme compares two models: one trained with mean squared error and one with the
  ranking loss. Both output one number per cell. With this definition the two differ in the
  body and in nothing else.

### Negative

- Two almost identical bodies are two bodies, so some code is written twice on purpose.
- Rule 2 creates one new possible error. Under Rule 2 the mean-squared-error loss must apply
  softplus itself. If that step is missing, the model is trained on one number and reports a
  different one, and nothing reports it. A test must cover this. The other losses are not
  affected: a count distribution already applies its own activation, and the ranking loss already
  reads the outputs from before the activation.
- The code does not follow this definition yet. Until it does, the code behaves as ADR-063 and the
  three config settings describe.
- No model has been trained with the two building rules in place.

### Explicitly refused

- **Setting the parts of a body separately.**
- **A configurable activation.** No config setting, no default, no ReLU and no leaky ReLU.
- **The quantile head.** The code contains a further kind of regression head that no config uses:
  one that outputs a set of quantiles of the count. It is to be deleted and not kept. It differs
  too much from the other heads to be worth its cost, and a quantile head is not a promising
  approach for data with many zeros and a heavy tail. Its code is removed after this document is
  finished, as a separate step. A quantile head is not to be
  reconsidered until there has been a clear discussion, a clear decision and clear experiments on
  how a body with several channels should produce them.
- **Giving names to the unused bodies now.** The code contains further bodies that no config uses:
  a zero-inflated negative binomial and four older variants (three hurdle bodies, and one negative
  binomial trained on all cells with its own loss). Their code and tests
  stay. A body gets a name when it is first tried, one body at a time. Models saved with them must
  still load.

---

## 5. Validation

This section lists five conditions that must hold for the definition to be working. Conditions 3,
4 and 5 have automated tests since 2026-11-16. Conditions 1 and 2 have none. One condition fails
today in the code: condition 1, for the ranking body. Condition 3 now holds in the code, but ten
configs in the models repository still ask for what it refuses; see section 6.

1. Each body in use can be written as one row of the table of the five bodies in section 2. No
   cell is filled from a default in the code. A cell may say "not built yet" or "not decided" only
   where section 2 says so in words. **Fails today for the ranking body:** its version, exponential
   or softplus, comes from a default.
2. Changing a required number, and nothing else, changes neither the loss that is built nor the
   width of the head.
3. If one of the four neighbouring settings asks for something the body cannot do, building the
   config stops with an error. The case to cover first: feeding back random samples during a
   multi-month forecast, for a body that cannot draw samples. Since 2026-11-16 this case is
   refused when the config is built. Before that it was caught only when a forecast run started.
4. A test fails if the mean-squared-error loss is given the outputs from before the activation and
   does not apply softplus. (Exists since 2026-11-16.)
5. A body whose loss converts the target back to counts refuses data that was not transformed with
   log1p, because it would convert it back wrongly. This check exists for the count distributions
   (risk register entry C-198) and, since 2026-11-16, for the ranking body.

What would reopen this decision: a body that someone wants to use and that cannot be written as a
row of the table of the five bodies.

---

## 6. Implementation Notes

This section records what is known about turning the definition into code. It contains no
decisions. The items are independent of each other.

- **How the code is organised: Option B, decided by Simon on 2026-11-16, and built the same day**
  (GitHub epic #616). One object per body. Reason: it is the definition written as code. A body
  is chosen whole, and only Option B makes that true in the code.
- **What was built.** One file, `views_hydranet/body/bodies.py`, holds an object for each of the
  five bodies in use. During training the body receives the outputs from before the activation,
  applies the activation once, and computes the loss. Training is unchanged: for all five bodies
  the real training loop reaches bit-for-bit identical weights with and without the body objects
  (measured on one CPU; not measured on a GPU). Two findings shaped it. First, the result of
  softplus depends on how the numbers are laid out in memory, for the value and for the gradient,
  so the body repeats the model's own sequence of operations exactly. Second, if the loss applied
  the activation a second time, the gradient changed in its last digits in the one case where two
  loss terms share a target, so the activation is applied once and everything reads that result.
- **What was not built, on purpose.** The count distributions keep their existing class
  (`DistributionFamily`); a distribution body wraps it, so nothing was renamed and ADR-067 is
  untouched. The model's layers and its choice of activation are unchanged. The older path through
  the training loop remains, for the unused bodies; it can only be removed by removing those
  bodies. The softplus version of the ranking loss has no object, because no config uses it.
- **The two options that were considered:**
  Option A: every loss receives the outputs from before the activation, and a loss that needs the
  activated values is given an argument that tells it to apply softplus.
  Option B: one object per body. The code already has such an object for the count distributions
  (the class `DistributionFamily`). It holds the width, the activation, the likelihood, the mean,
  the sampling and whether it makes its own zeros. The mean-squared-error body and the ranking body would become
  two more such objects. The drawback: that class is named for distributions and is governed by
  ADR-067, and those two bodies are not distributions, so the class would have to be renamed or
  its name would be wrong for them.
- **The switch from section 1** (`needs_latent` in the code). For the five bodies in use its second
  job, "must be trained on all cells", is now a fact stated by the body object. The switch itself
  remains on the older path, for the unused bodies.
- **The check that was missing is added** (condition 5 in section 5). A ranking-loss config whose
  target is not transformed with log1p is now refused when the config is built.
- **Ten configs ask for something their body cannot do, today.** The five mean-squared-error
  configs and the five ranking configs ask for random samples to be fed back during a multi-month
  forecast. Neither body can draw samples. With the released package these configs can be read
  and their models trained, and the run then stops when a forecast starts. With the code as of
  2026-11-16, not yet released, they are refused when the config is read. The fix belongs in the
  models repository and was handed over there (views-models issue #737). None of the ten configs
  was run for this document.
- **The five ranking configs do not choose a version of the ranking loss.** They get the code's
  default, the exponential version. By this document the exponential and the softplus versions are two
  bodies, so the body of those five models is picked by a default.
- **The quantile head was removed from the code on 2026-11-16.** Its module and its test file are
  deleted, its branches are removed from the head, the loss factory, the training loop, the model
  saving and the forecast code, and the config setting `n_quantiles` is gone. The config now has
  106 settings. No config and no saved model used it.
- **Wrong record number in existing files.** One test file and the new `body` folder in the
  package say "ADR-079" where they mean the record that is now ADR-081.

---
