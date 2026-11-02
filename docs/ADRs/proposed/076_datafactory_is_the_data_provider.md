# ADR-076: HydraNet models get their data from views-datafactory

**Status:** Proposed
**Date:** 2026-11-01
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Related:** ADR-077 (global land coverage — the change this one makes reachable), ADR-009 (configuration validation)

## Summary (read this first — self-contained)

A model gets its input data from a **provider**. The platform has two: **viewser**, the older
service, and **views-datafactory**, the newer one.

HydraNet models take their data from **views-datafactory**. viewser is not a target for a new model
or for a migration.

Two reasons, and they stand independently. The covariates viewser supplies that are not conflict
counts are not trusted. And working at global scale requires the datafactory in any case, so a model
left on viewser cannot follow the fleet.

This ADR also fixes what a provider migration must prove before it ships: a **parity comparison on
freshly pulled data, whose residual is decomposed and explained**, judged against falsifiers
registered before the comparison is run. Exact agreement is not the bar, because the two providers
differ by deliberate preprocessing.

**This states intent, not accomplishment.** Most HydraNet models are on the datafactory and a
minority are not. That gap is real, and §5 names the check that counts it rather than leaving it to
recollection.

---

## 1. Context

**The decision was being made per model.** Each migration carried its own reasoning, recorded
against whichever model happened to move. The fleet-wide rule was never stated, so every migration
re-derived it, and the remaining unmigrated models were visible only to whoever remembered them.

**The mechanism is already built and must not be rebuilt.** The pipeline's data loader inspects what
a model's configuration declares and dispatches on it: a viewser queryset goes to viewser, and a
datafactory descriptor goes to the datafactory. The point of change for any one model is therefore a
**single function in that model's own configuration** — which is why the provider is configuration
rather than architecture, and why this ADR states the rule rather than any model's current setting.

**Exact agreement is the wrong bar, and this was learned the expensive way.** The providers differ
by intentional preprocessing, including whether events with imprecise geolocation are filtered and
which data vintage is served. Byte-equality is therefore unachievable, and a bar that cannot be met
is a bar that gets quietly lowered.

**A cached pull cannot validate a provider.** A cache can be arbitrarily out of date, and staleness
is indistinguishable from a genuine provider difference. The first migration's cache was a year
behind the remote, and the discrepancy would have been read as a defect in the new provider.

---

## 2. Decision

**Statement.** *We will take HydraNet model input from views-datafactory, and require a provider
migration to prove a decomposed parity result on freshly pulled data before it ships.*

### 2.1 The provider

The declared provider for a HydraNet model is **views-datafactory**. viewser is not a target for a
new model or for a migration. Each model declares its provider in its own configuration; this ADR
states what that declaration is expected to say.

### 2.2 What a migration must prove

A provider migration is justified by a comparison against the incumbent that:

1. runs on **freshly pulled** data;
2. compares the cell population, the per-cell agreement, and the extreme values;
3. **decomposes any residual and attributes it to a named cause**;
4. is judged against falsifiers **registered before the comparison is run**.

### 2.3 A parity claim from cached data is void

A comparison computed against a cache does not answer the question. Either pull fresh, or report
that the question was not answered.

### 2.4 Scope

**In scope:** which provider a HydraNet model targets, and what a migration must prove.

**Out of scope:** which cells a model forecasts, which is ADR-077 and has its own rationale; which
models have already migrated, which is configuration; the retirement of viewser as a platform
service, which is not this repository's to decide; and feature naming across providers, which the
output-schema ADR governs.

---

## 3. Rationale & Integrity Impact

| alternative | why it loses |
|---|---|
| **stay on viewser and take the non-conflict covariates from elsewhere** | those covariates are the part not trusted, so sourcing them separately relocates the problem. Global scale needs the datafactory regardless, so this defers a migration without avoiding one |
| **start new models from an already-migrated one rather than migrating in place** | discards the migrating model's experimental history and configuration lineage. The point of change is one function, so migrating in place costs less than re-establishing provenance |
| **require exact agreement as the acceptance bar** | unachievable and undesirable, since the providers differ by deliberate preprocessing. An honest decomposed residual is a bar that can be met and defended |
| **validate against the cached pull** | staleness cannot be told apart from a provider difference |
| **keep deciding per model as each comes up** | the present arrangement: it leaves the fleet rule unstated and the remainder uncounted |

**Reproducibility.** The next migration inherits a stated standard rather than an example to
imitate, and the fresh-pull rule removes a class of false provider defect.

**Fail-loud.** Registering falsifiers before the comparison prevents a migration being justified
after the fact by whichever comparison looked acceptable. The check in §5 makes the unmigrated
remainder a counted fact rather than an assumption.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] The rule is stated once, so each remaining migration inherits it.
- [ ] The unmigrated remainder becomes countable rather than assumed empty.
- [ ] The parity standard is reusable: a bar rather than a precedent.
- [ ] States the prerequisite that ADR-077 depends on.

### ⚠️ Negative (Costs)

- [ ] **It declares an intent the fleet does not yet meet**, and a reader may mistake the
      declaration for the state. §5's check exists for exactly that reason.
- [ ] A migration is now more expensive: fresh pull, decomposed residual, falsifiers registered in
      advance. The cheaper alternative is a migration nobody can defend.
- [ ] Models remaining on viewser are formally off-target, which may force a decision about retiring
      rather than migrating them. That decision is not taken here.

---

## 5. Validation

**Invariants.**
- A model's provider is readable from its own configuration and from nowhere else.
- No ADR states any model's provider. This one states the target; configurations state the actuals.

**Tests.**
- *Reporting check:* read every HydraNet model's declared provider and report each one that is not
  the datafactory. It **reports rather than fails** while the migration is incomplete — a failing
  test on an accepted gap gets disabled, and a disabled test is indistinguishable from a passing
  one. It becomes an assertion when the count reaches zero.
- *Red:* a parity claim computed from cached data is rejected by the parity harness.
- *Beige:* a model declaring an unrecognised provider is refused at configuration validation rather
  than defaulting to either one.

**Not yet enforced.** The reporting check does not exist. Until it does, the remainder is knowable
only by reading configurations by hand, which is the condition this ADR is meant to end.

**What would reopen this.** If the datafactory's non-conflict covariates turned out to carry the
same defects as viewser's, the first reason would fall away and only the scale argument would
remain — making this a consequence of ADR-077 rather than an independent decision.

---

## 6. Implementation Notes

- **Location.** The rule is honoured in each model's configuration and counted by the check in §5.
  The dispatch mechanism it relies on belongs to the pipeline and is not modified.
- **References.** ADR-009, ADR-032, ADR-077. Cross-repository: the pipeline's data-loader contract
  and the datafactory consumer contract.
- **Evidence.** The datafactory migration dossier under `reports/` holds the parity harness, the
  registered falsifiers and the decomposed residual for the first migration.
