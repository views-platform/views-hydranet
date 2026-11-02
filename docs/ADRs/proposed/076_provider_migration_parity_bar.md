# ADR-076: A provider migration must prove decomposed parity on freshly pulled data

**Status:** Proposed
**Date:** 2026-11-02
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Required by:** ADR-077 (global land coverage — reachable only on the current provider)
**Related:** ADR-009 (configuration validation), ADR-032 (output schema and feature naming)

## Summary (read this first — self-contained)

A model gets its input from a **provider**. The platform recognises three: viewser, the older
service; views-datafactory, the newer one; and a synthetic source used for testing.

**HydraNet models take their input from views-datafactory, and that migration is complete** — every
HydraNet model declares it, and the last ones moved in August 2026. That part is settled and is
recorded here as context, not as a decision still to take.

What is *not* settled is the bar a **future** migration must clear. This ADR sets it: a provider
change must be justified by a parity comparison against the incumbent that runs on **freshly
pulled** data and whose residual is **decomposed and explained**, judged against falsifiers
registered before the comparison is run.

Exact agreement is explicitly **not** the bar, and cannot be: the providers differ by deliberate
preprocessing — one of them reassigns imprecisely located events to a neighbouring cell where the
other leaves them unhandled, and they serve different data vintages. A bar that cannot be met is a
bar that gets quietly lowered.

And a cached pull cannot be used to establish parity at all, because a stale cache is
indistinguishable from a genuine provider difference.

---

## 1. Context

**What is already true.** Every HydraNet model declares views-datafactory as its source. The
migration ran model by model and finished in August 2026. The pipeline's loader inspects what a
model's configuration declares and dispatches on it, so the point of change for any one model is a
single function in that model's own configuration.

**Why the move was made.** Two reasons were recorded at the time, and they are of different kinds.
The first is a maintainer's **judgement** rather than a measurement: the covariates viewser supplies
that are not conflict counts are not trusted. Nothing measures this, and it is recorded as a
judgement so that it is not later mistaken for a finding. The second is structural: working at
global coverage requires the newer provider, because coverage can only be declared on its
descriptor at all (ADR-077).

**What the first migration taught, and why it needs writing down.** It was assumed that a migration
could be judged by exact agreement with the incumbent. It cannot. The two providers differ by
intentional preprocessing and by vintage, so a residual is expected, and the useful question is
whether it can be **attributed**. The first migration passed on exactly that basis: a decomposed,
explained residual with pre-registered falsifiers, none of which fired.

It also taught the fresh-pull rule the hard way. That migration's cache was a year behind the
remote, and the discrepancy would have read as a defect in the new provider.

**Why now.** The rule exists only as the precedent of one migration, in a dossier. The next
migration — of any model, in either direction — would re-derive it or skip it.

---

## 2. Decision

**Statement.** *We will require a provider migration to prove a decomposed parity result on freshly
pulled data before it ships.*

### 2.1 What a migration must prove

A provider change is justified by a comparison against the incumbent that:

1. runs on **freshly pulled** data;
2. compares the cell population, the per-cell agreement, and the extreme values;
3. **decomposes any residual and attributes it to a named cause**;
4. is judged against falsifiers **registered before the comparison is run**.

### 2.2 Exact agreement is not the bar

A residual is expected and is not a failure. An **unattributed** residual is.

### 2.3 A parity claim from cached data is void

Either pull fresh, or report that the question was not answered.

### 2.4 Scope

**In scope:** what a provider migration must prove.

**Out of scope:** which provider a model declares, which is configuration and, for HydraNet, already
settled; the retirement of viewser as a platform service, which is not this repository's to decide;
coverage, which is ADR-077; and feature naming across providers, which the output-schema ADR governs.

---

## 3. Rationale & Integrity Impact

| alternative | why it loses |
|---|---|
| **require exact agreement** | unachievable, because the providers differ by deliberate preprocessing and vintage. A bar that cannot be met is met by lowering it |
| **accept a residual without attributing it** | indistinguishable from accepting a defect. Attribution is the whole content of the check |
| **validate against a cached pull** | a stale cache and a provider difference produce the same observation |
| **register falsifiers after looking** | lets a migration be justified by whichever comparison happened to look acceptable |
| **leave the bar as the precedent of one migration** | the present arrangement: the standard exists in a dossier, and the next migration re-derives or skips it |
| **state it as a per-model rule** | the provider is configuration and differs per model; the bar does not |

**Reproducibility.** The next migration inherits a stated standard rather than an example to
imitate, and the fresh-pull rule removes a class of false provider defect.

**Fail-loud.** Registering falsifiers in advance is what makes the check capable of failing.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] The standard is stated once and is reusable: a bar rather than a precedent.
- [ ] The fresh-pull rule closes a failure mode that has already occurred once.
- [ ] Separating the judgement from the measurement in §1 means a later reader can weigh the
      judgement as a judgement.

### ⚠️ Negative (Costs)

- [ ] A migration is more expensive: a fresh pull, a decomposed residual, and falsifiers registered
      before looking.
- [ ] The bar is stated in prose and is not machine-checkable. Whether a residual has been
      *attributed* is a judgement, and this ADR does not remove it.
- [ ] It constrains a direction nobody currently intends to travel. If no further migration ever
      happens, the cost was writing it down.

---

## 5. Validation

**Invariants.**
- A model's provider is declared in its own configuration and nowhere else. No ADR states any
  model's provider.

**Tests.**
- *Assertion:* every HydraNet model declares views-datafactory. This is an assertion rather than a
  report, because the count of exceptions is **zero** today — there is no accepted gap to tolerate.
- *Red:* a parity claim computed from cached data is rejected by the parity harness.
- *Beige:* a model declaring an unrecognised provider is refused at configuration validation rather
  than defaulting to any of the three.

**Not yet enforced.** The assertion does not exist. Note also that reading a model's provider
means *executing* its configuration function and inspecting what it returns — it is not a static
read, and can raise if a client library is absent.

**What would reopen this.** If a provider migration were ever required under time pressure that made
a fresh pull impossible, §2.3 would be the rule under strain, and the honest response is to report
the question as unanswered rather than to relax the rule.

---

## 6. Implementation Notes

- **Location.** The assertion belongs with the configuration conformance tests. The parity harness
  from the first migration is the reusable instrument.
- **Evidence.** `reports/2026-11-02_adr_claim_verification/00_CLAIMS.md` §3 holds every factual
  claim in this ADR with its source, including the migration census and the preprocessing
  differences. The first migration's parity harness, registered falsifiers and decomposed residual
  are in the datafactory migration dossier under `reports/`.

---

## References

**This repository:** ADR-009 (configuration validation) · ADR-032 (output schema) · ADR-077 (global
land coverage).

**Other repositories** — the contracts this ADR works within:

- [views-pipeline-core ADR-015 — model-specific input querysets](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/ADRs/015_model_specific_inputdata_querysets.md). Establishes that a model's input declaration lives in that model's own configuration, which is why the provider is configuration and this ADR states only the bar.
- [views-pipeline-core ADR-011 — separation of configs](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/ADRs/011_seperation_of_configs.md). The configuration boundary the point of change sits on.
- [views-pipeline-core ADR-023 — input drift detection](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/ADRs/023_input_drift_detection.md). The adjacent authority for §2.3: a stale input is a detectable condition rather than an acceptable one.
