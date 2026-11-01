# ADR-077: HydraNet models forecast global land coverage

**Status:** Proposed
**Date:** 2026-11-01 (recording a decision taken 2026-09-19)
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Requires:** ADR-076 (views-datafactory as the provider — the prerequisite) · **Supersedes (in part):** ADR-071, which recorded the superseded scope for one model

## Summary (read this first — self-contained)

A HydraNet model forecasts over a set of grid cells. Historically that set was **Africa and the
Middle East**. Since 2026-09-19 it is **global land** — every PRIO-GRID land cell, excluding water.

This ADR records that decision. It is being written **six weeks after the fact**, which is the
reason it exists: the change was made fleet-wide in config, and nothing in the architectural record
said so. The only ADR that mentioned spatial scope at all recorded the *old* value, so for six
weeks the governance record and the running system disagreed and nothing flagged it.

The driver is a delivery requirement, not a preference. The consumer deliverable's coverage is a
**cut of global land**, so a model restricted to Africa and the Middle East cannot produce it at
all. The regional scope was always a de-risking stage, never the target.

⛔ **Consequence worth stating plainly, because it has already cost us:** any data or result
produced under the regional scope is **not comparable** to one produced under global land, and the
two are distinguishable only by cell count. A regionally-scoped cached pull survived the flip and
was silently reused weeks later.

## 1. Context

**The problem this ADR fixes is its own absence.** The scope changed across the fleet on
2026-09-19. The change is recorded in each model's config and in a runbook issue. It was **not**
recorded in any ADR. The one ADR that named a spatial scope named the superseded one.

⇒ So the architectural record asserted a scope the fleet had abandoned, and the divergence went
unflagged for six weeks. Under ADR-000, that is an unresolved intent-versus-fact divergence: either
a decision was never implemented, or the code was implemented wrong. Here it is neither — the code
is right and the record was never written.

**Why global land, as recorded at the time.** The consumer delivery's coverage is produced by
cutting a subset out of global land. A model covering only a region cannot supply that cut.
Two outputs depend on the wider scope: calibration-partition runs used for the maintainer's own
analysis, and true forward forecasts pooled by an ensemble and delivered to the consumer.

**Why the regional scope existed.** It was the de-risking stage — a smaller, cheaper grid on which
to make the architecture work locally before committing to the full grid on rented hardware. It was
always intended to be left behind, which is precisely why nobody wrote an ADR when it was: the
change felt like completing a plan rather than taking a decision.

⚠️ **OWED — the one part not recoverable from the record.** Whether there is a *scientific*
rationale in addition to the delivery requirement — for example that training on the global grid
improves the model, or that a region-restricted grid biased what it learned — is not written down
anywhere I can find. Simon has it. Until it is added, this ADR claims only the delivery rationale.
**Stated as a gap rather than guessed at** (ADR-000 R7's spirit applied to rationale).

**Assumption that is no longer true.** It was assumed that scope was a per-model experimental
setting, low-stakes and freely varied. It is not: scope determines whether a model can serve the
deliverable at all, and it silently determines whether two results may be compared.

---

## 2. Decision

**Statement.** *We will forecast over global land coverage — every PRIO-GRID land cell — and we
will treat spatial scope as a declared, verified property of a run rather than an incidental
config value.*

### 2.1 The scope rule

**Global land is the coverage for HydraNet models.** Each model declares its coverage in its own
config; this ADR states what that declaration is expected to say. A regional scope is a
**de-risking stage**, valid only while explicitly declared as such, and never a target.

⇒ This document names no model, and states no cell count. The count is a mechanical consequence of
the coverage definition and belongs in config and in the tooling that checks it (ADR-000's
volatility rule).

### 2.2 Scope is part of a result's identity

A result's coverage must be recoverable from the result, not from recollection of which config ran.

⇒ Coverage joins the set of facts an emitted artifact declares about itself (ADR-074). A result
whose coverage cannot be established is a result that cannot be compared.

### 2.3 Cross-scope comparison is invalid

Results produced under different coverage are **not comparable** — not merely noisier. The cell
population differs, so every spatially-pooled statistic is of a different population.

⛔ **And the two are distinguishable only by counting cells**, which is why this failed in practice:
a regionally-scoped cached pull survived the flip, was copied onward by a builder that propagates
one model's cached data to every arm it builds, and was consumed weeks later as though global. The
scope was inferable from a row count and from nothing else.

### 2.4 In-scope / out-of-scope

**In scope:** the coverage HydraNet models target, and the requirement that a run's coverage be
declared and verifiable.

**Out of scope:**
- **The data provider.** ADR-076. It is a prerequisite for this decision, not part of it.
- **The consumer delivery's own coverage.** That is a cut taken downstream, governed by the
  delivery declaration in the models repository.
- **Per-model migration state.** Config, enumerable by the check in §5.
- **Purging regionally-scoped data already on disk.** An operational task, tracked separately.

---

## 3. Rationale & Integrity Impact

| alternative | why rejected |
|---|---|
| **stay regional and extend only for delivery runs** | the deliverable's coverage is a cut of global land, so delivery runs would use a different population from every calibration run behind them — making the validated model and the served model different objects |
| **keep both scopes as a supported axis** | doubles every comparison's precondition and makes "is this comparable?" a question asked per artifact. The failure in §2.3 happened with only one scope nominally live |
| **treat scope as an ordinary experimental knob** | this is the status quo assumption, and it is what let a regional pull propagate silently. Scope decides whether results may be compared at all, which is not a knob |
| **leave the decision unrecorded, since the configs already say it** | this is what actually happened, and it is the reason this ADR exists. Config states the actual; nothing stated the intent, so the stale copy in the architectural tier had no counterweight |

**Fortress state.** Reproducibility improves directly: coverage becomes a declared, verifiable
property of a result instead of a fact about which config happened to run. Combined with ADR-074,
a result's comparability becomes checkable rather than remembered.

**Fail-loud.** §2.2 makes an unverifiable coverage a defect rather than an inconvenience. The check
in §5 converts "is anything still regional?" from a memory question into an enumeration.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] The fleet's actual coverage is finally stated in the tier that governs it.
- [ ] Closes a six-week intent-versus-fact divergence, and removes the only ADR text asserting the
      superseded scope.
- [ ] Makes cross-scope incomparability an explicit rule rather than a thing learned by being burned.
- [ ] Gives the provider decision (ADR-076) its stated purpose — the scale-up it was a prerequisite for.

### ⚠️ Negative (Costs)

- [ ] **Roughly five times the cells**, hence proportionally more compute, memory and storage per
      run. Already paid since 2026-09-19; recorded here so the cost is attributed to a decision
      rather than discovered in a bill.
- [ ] Results produced before the flip are **not comparable** to results after it. That boundary
      runs through our own experimental record and is not fixable retroactively.
- [ ] Regionally-scoped artifacts may still exist on disk and remain valid-looking. Purging them is
      a separate operational task; this ADR only makes them formally off-target.
- [ ] The de-risking affordance is lost: there is no longer a sanctioned smaller grid to debug on
      unless one is explicitly declared as a de-risking stage under §2.1.

---

## 5. Validation

**Invariants.**
- Coverage is readable from a model's own config, and from an emitted result's own declaration
  (ADR-074).
- No ADR states a cell count. This one states the coverage; config and tooling state the counts.

**Tests.**
- ⭐ **Enumeration check:** a test that reads every HydraNet model's declared coverage and reports
  any that is not global land. ⚠️ **Reports rather than fails** while any de-risking model is
  legitimately declared — a failing test on an accepted exception gets disabled, and a disabled
  test is how the divergence returns.
- *Red:* a result whose declared coverage differs from another's is refused by any comparison tool,
  rather than pooled.
- *Beige:* a run whose declared coverage and actual cell population disagree fails loudly, rather
  than proceeding with the population it happens to have.

⚠️ **Honest gap (ADR-000 R7):** none of these tests exists yet. The coverage declaration they would
check is specified by ADR-074 and is also not yet built. ⇒ **This ADR is, today, intent with no
enforcement** — exactly the condition that let its predecessor go stale. Stated plainly rather than
implied, and the dependency is the reason ADR-074 is sequenced first.

**Failure mode that would trigger reconsideration.** If the deliverable's coverage ever stopped
being a cut of global land, the delivery rationale would evaporate and only the scientific
rationale — currently the owed gap in §1 — would remain to carry the decision.

---

## 6. Implementation Notes

- **Location.** Declared in each model's config; checked by the enumeration test in §5; recorded
  per result by ADR-074's emission manifest.
- **Sequencing.** ADR-076 is the prerequisite (the provider makes the scope reachable); ADR-074 is
  the prerequisite for §2.2 (a result cannot declare its coverage until it declares anything).
- **Relationship to ADR-071.** ADR-071 recorded the superseded scope for one model, bundled with
  the provider decision. It should be marked superseded, not deleted — its parity evidence remains
  a useful historical record. ⛔ Its scope claim must stop reading as current.
- **References.** ADR-000 (ADR standard: volatility, intent-vs-fact, R7), ADR-074 (emission
  provenance), ADR-076 (data provider), ADR-032 (output schema). Cross-repo: the delivery
  declaration and the pod runner contract in the models repository.
- **Evidence.** The flip is recorded in each model's queryset config with its date, and in the
  runbook issue that sequenced it. Cell counts and row counts live there and in the purge record
  under `reports/`, not here.
