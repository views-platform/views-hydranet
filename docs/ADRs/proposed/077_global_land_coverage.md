# ADR-077: HydraNet models forecast global land

**Status:** Proposed
**Date:** 2026-11-01 (recording a change made 2026-09-19)
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Requires:** ADR-076 (views-datafactory as the provider)
**Related:** ADR-074 (an emitted artifact records what it is)

## Summary (read this first — self-contained)

A model forecasts over a set of grid cells — its **coverage**. HydraNet coverage was Africa and the
Middle East. Since 2026-09-19 it is **global land**: every land cell of the grid, excluding water.

The driver is a delivery requirement rather than a preference. The coverage the consumer deliverable
is cut from is a **subset of global land**, so a model restricted to one region cannot produce it at
all. The regional coverage was a de-risking stage — a smaller, cheaper grid on which to make the
architecture work before committing to the full one — and was always meant to be left behind.

**Results produced under different coverage are not comparable.** The cell population differs, so
every statistic pooled over cells is of a different population. The two are also hard to tell apart:
coverage is visible in a result only as a cell count, which is why a regionally scoped data cache
survived the change and was reused weeks later as though it were global.

---

## 1. Context

**The coverage changed and was not recorded as a decision.** Each model's configuration states its
coverage, and a runbook sequenced the change across the fleet. No ADR said so, and the only ADR that
named a coverage named the superseded one. So the architectural record described a coverage the
fleet had abandoned, and nothing reported the disagreement.

**Why global land.** The consumer deliverable's coverage is produced by cutting a subset out of
global land. A model covering one region cannot supply that cut. Two outputs depend on the wider
coverage: the calibration runs used for the maintainer's own analysis, and the forward forecasts
pooled for delivery.

**Why a regional coverage existed.** It was the de-risking stage, and that is why nobody wrote an
ADR when it ended: the change felt like completing a plan rather than taking a decision.

**Coverage is not an ordinary experimental setting.** It was treated as one — low-stakes, freely
varied. It is not. It determines whether a model can serve the deliverable, and it silently
determines whether two results may be compared.

**Open question, not answered here.** Whether there is a *scientific* reason for the wider coverage
in addition to the delivery requirement — that training on the global grid improves the model, or
that a region-restricted grid biased what it learned — is not recorded anywhere. Until it is, this
ADR rests on the delivery rationale alone.

---

## 2. Decision

**Statement.** *We will forecast over global land, and treat coverage as a declared and verifiable
property of a run rather than an incidental configuration value.*

### 2.1 The coverage

**Global land is the coverage for a HydraNet model.** Each model declares its coverage in its own
configuration; this ADR states what that declaration is expected to say. A regional coverage is a
de-risking stage, valid only while declared as one, and never a target.

This ADR states a coverage, not a cell count. The count follows from the coverage definition and
belongs with the configuration and the tooling that checks it.

### 2.2 Coverage is part of a result's identity

A result's coverage must be recoverable from the result, not from recollection of which
configuration produced it. Coverage is therefore among the facts an emitted artifact declares
(ADR-074).

### 2.3 Results of different coverage are not comparable

They are of different cell populations, so a pooled statistic from one is not an estimate of the same
quantity as a pooled statistic from the other. A comparison across coverage is invalid rather than
noisy.

### 2.4 Scope

**In scope:** the coverage HydraNet models target, and the requirement that a run's coverage be
declared and verifiable.

**Out of scope:** the data provider, which is ADR-076 and a prerequisite rather than part of this;
the consumer deliverable's own cut, which the models repository governs; which models have changed,
which is configuration; and removing regionally scoped data already on disk, which is an operational
task.

---

## 3. Rationale & Integrity Impact

| alternative | why it loses |
|---|---|
| **stay regional and widen only for delivery runs** | the deliverable is cut from global land, so delivery runs would use a different cell population from every calibration run behind them. The validated model and the served model would be different objects |
| **support both coverages as an ongoing axis** | makes "are these comparable?" a question to be asked of every artifact. The silent reuse described in §2.3 happened with only one coverage nominally live |
| **treat coverage as an ordinary experimental knob** | the assumption that allowed a regionally scoped cache to propagate unnoticed. Coverage decides whether results may be compared at all, which is not a knob |
| **leave it unrecorded, since the configurations already say it** | what actually happened, and the reason this ADR exists. A configuration states the actual; nothing stated the intent, so a stale declaration in the architectural record had nothing to contradict it |

**Reproducibility.** Coverage becomes a declared, verifiable property of a result rather than a fact
about which configuration happened to run, so comparability becomes checkable.

**Fail-loud.** §2.2 makes an unverifiable coverage a defect rather than an inconvenience, and §5
turns "is anything still regional?" into a count.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] The fleet's coverage is stated in the tier that governs it.
- [ ] Removes the only architectural text asserting the superseded coverage.
- [ ] Makes cross-coverage incomparability a stated rule rather than something learned by being
      caught by it.
- [ ] Gives the provider decision its stated purpose: the scale it was a prerequisite for.

### ⚠️ Negative (Costs)

- [ ] **Several times the cells**, and proportionally more compute, memory and storage per run.
      Already being paid since the change; recorded here so the cost belongs to a decision rather
      than appearing in a bill.
- [ ] Results from before the change are **not comparable** to results after it. That boundary runs
      through the experimental record and cannot be repaired retroactively.
- [ ] Regionally scoped artifacts may remain on disk and still look valid. This ADR makes them
      formally off-target; removing them is separate work.
- [ ] The de-risking affordance is lost. There is no sanctioned smaller grid to debug on unless one
      is explicitly declared as a de-risking stage under §2.1.

---

## 5. Validation

**Invariants.**
- Coverage is readable from a model's own configuration, and from an emitted result's own
  declaration.
- No ADR states a cell count. This one states the coverage.

**Tests.**
- *Reporting check:* read every HydraNet model's declared coverage and report each one that is not
  global land. It **reports rather than fails** while any de-risking model is legitimately declared,
  because a failing test on an accepted exception gets disabled.
- *Red:* a result whose declared coverage differs from another's is refused by any comparison tool
  rather than pooled with it.
- *Beige:* a run whose declared coverage and actual cell population disagree fails loudly rather
  than proceeding with the population it happens to have.

**Not yet enforced.** None of these tests exists, and the coverage declaration they would read is
specified by ADR-074 and not yet built. This ADR is intent without enforcement, which is the
condition that let its predecessor go stale — and the reason ADR-074 comes first.

**What would reopen this.** If the deliverable's coverage ceased to be a cut of global land, the
delivery rationale would fall away and only the unrecorded scientific rationale would remain to
carry the decision.

---

## 6. Implementation Notes

- **Location.** Declared in each model's configuration, counted by the check in §5, and recorded per
  result by ADR-074's manifest.
- **Sequencing.** ADR-076 is the prerequisite that makes the coverage reachable; ADR-074 is the
  prerequisite for §2.2, since a result cannot declare its coverage until it declares anything.
- **Evidence.** The change is recorded in each model's configuration with its date, and in the
  runbook that sequenced it. Cell counts and row counts live there and in the purge record under
  `reports/`.

---

## References

**This repository:** ADR-009 (configuration validation) · ADR-032 (output schema) · ADR-074 (an
emitted artifact records what it is) · ADR-076 (views-datafactory as the provider).

**Other repositories** — the contracts this ADR works within:

- [views-models ADR-019 — delivery declaration](https://github.com/views-platform/views-models/blob/main/docs/ADRs/019_delivery_declaration.md). Where the consumer deliverable's own coverage is declared — the cut referred to in §1, and the reason the wider coverage is required.
- [views-pipeline-core ADR-015 — model-specific input querysets](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/ADRs/015_model_specific_inputdata_querysets.md). Establishes that coverage is declared per model in its own configuration, which is why this ADR states a coverage and not a cell count.
