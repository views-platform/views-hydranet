# ADR-077: HydraNet models forecast global land

**Status:** Accepted
**Date:** 2026-11-01 (recording a change made 2026-09-19) · accepted 2026-11-02
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Related:** ADR-074 (an emitted artifact records what it is)
**Rests on a fact, not an ADR:** every model declares the same data source, and a coverage declaration exists only on that source's descriptor

## Summary (read this first — self-contained)

A model forecasts over a set of grid cells — its **coverage**. Subnational HydraNet coverage was
Africa, then Africa and the Middle East. It is now **global land**: every land cell of the grid,
excluding water.

**Coverage has always been set by what was funded, not by preference.** The original VIEWS ERC
grants named **Africa** as the spatial extent of the subnational forecast — a choice shaped in
large part by the hardware, software and compute available in 2017–18. A later, smaller UN ESCWA
grant extended it to **the Middle East**. More recent grants carry **global coverage as a
deliverable**. This ADR records the coverage those commitments require.

**The data backend and the coverage are one chain, not two coincidences.** The datafactory was
built *in order to* make global coverage possible; the older backend could not reach it. So the
move to the datafactory and the move to global are the same programme at two stages. The
datafactory deliberately continued to support Africa and the Middle East — largely **as a parity
check**, to confirm it could reproduce the forecasts the old backend produced on the extent we
already knew, before the extent itself changed.

**Results produced under different coverage are not comparable.** The cell population differs, so
every statistic pooled over cells is of a different population. The two are also hard to tell apart:
coverage is visible in a result only as a cell count, which is why a regionally scoped data cache
can survive such a change and be reused afterwards as though it were current.

---

## 1. Context

**Coverage is a funded commitment, and it has moved twice.** Africa was the extent the original
ERC grants named, and the compute available when that work began is part of why. The Middle East
was added under a later UN ESCWA grant. Global coverage is a deliverable of more recent grants.
Each step widened what the project had undertaken to produce, so coverage is not a parameter the
project is free to choose per experiment.

**Reaching global required a new data backend, which is why one was built.** The older backend
could not serve the global extent. The datafactory was built to make it reachable, and a coverage
declaration exists only on the datafactory's own descriptor — so until a model sources from it,
coverage cannot be stated at all.

⇒ The two moves are therefore sequenced rather than independent: source first, extent second. The
datafactory kept serving Africa and the Middle East through the first stage on purpose, as the
**parity check** — reproducing the old backend's forecasts on the extent already known, before the
extent changed underneath the comparison.

**The change happened and was not recorded as a decision.** Each model's configuration states its
coverage and a runbook sequenced the change across the fleet, but no ADR said so, and the only ADR
that named a coverage named the superseded one. So the architectural record described an extent the
project had already left, and nothing reported the disagreement.

**Why it went unrecorded is itself informative.** Widening an extent the grants had already widened
reads as completing a plan rather than as taking a decision — which is exactly when a record is
least likely to be written and most likely to be needed later.

**Coverage is not an ordinary experimental setting.** It determines whether a model can serve the
deliverable at all, and it silently determines whether two results may be compared.

**What this ADR does not claim.** The reason for global coverage is a **funded commitment**, not a
measured finding. Whether training on the global grid also *improves* the model, or whether a
region-restricted grid biased what it learned, is a separate empirical question that nothing here
answers and no measurement in the project settles.

---

## 2. Decision

**Statement.** *We will forecast over global land, and treat coverage as a declared and verifiable
property of a run rather than an incidental configuration value.*

### 2.1 The coverage

**Global land is the coverage for a subnational HydraNet model.** Each model declares its coverage
in its own configuration; this ADR states what that declaration is expected to say. A narrower
extent is valid only while explicitly declared, and only for a stated purpose — a parity check
against a previous backend, or a deliberately cheap test bed — never as a default.

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

**Out of scope:** which data source a model declares, which is configuration and already settled;
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
| **leave it unrecorded, since the configurations already say it** | a configuration states what a model does; nothing would state what it is supposed to do. A stale declaration elsewhere in the architectural record would then have nothing to contradict it |

**Reproducibility.** Coverage becomes a declared, verifiable property of a result rather than a fact
about which configuration happened to run, so comparability becomes checkable.

**Fail-loud.** §2.2 makes an unverifiable coverage a defect rather than an inconvenience, and §5
turns "is anything still regional?" into a count.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] The fleet's coverage is stated in the tier that governs it.
- [ ] States the coverage in the tier that governs it, so a stale declaration elsewhere has
      something to contradict it.
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
- [ ] A narrower extent is no longer available by default, so anything that relied on a smaller
      grid being cheap — a parity check, a quick local test — must now declare it and say why.

---

## 5. Validation

**Invariants.**
- Coverage is readable from a model's own configuration, and from an emitted result's own
  declaration.
- No ADR states a cell count. This one states the coverage.

**Tests.**
- *Reporting check:* read every HydraNet model's declared coverage and report each one that is not
  global land. It **reports rather than fails** while any narrower extent is legitimately declared,
  because a failing test on an accepted exception gets disabled.
- *Red:* a result whose declared coverage differs from another's is refused by any comparison tool
  rather than pooled with it.
- *Beige:* a run whose declared coverage and actual cell population disagree fails loudly rather
  than proceeding with the population it happens to have.

**Partly enforced as of 2026-11-02.** ADR-074's manifest exists and records each run's declared
coverage, so a result now carries its own extent.

The reporting check is `scripts/coverage_report.py`, with the reader it shares with the manifest
tested in `tests/test_coverage_declaration.py`. It is scoped by the configurations present on disk
rather than by a roster, and it reads them without executing them, so one unreadable model cannot
end the census. A coverage that cannot be resolved to a declaration is reported as unresolved with
its source expression, never as absent.

One part of it is stronger than this ADR asked for: the check **asserts** that every model of our
own algorithm declares global land, because the condition stated above is met — every such model
declares it and the exception count is zero. The remaining declarations are reported, not required.

**Still not enforced.** Two of the three tests above.

- No comparison tool reads the recorded coverage, so nothing yet refuses to pool results of
  differing extent. The declaration is available to such a tool; none consumes it.
- A run whose declared coverage and actual cell population disagree does **not** fail. Both numbers
  are now recorded side by side in the manifest — the declaration, and the measured per-horizon row
  count — so a disagreement is visible to a reader. Nothing acts on it.

**And a limit of the reporting check itself.** It reads a declaration; it cannot tell whether a
declared extent is the one the project undertook to produce, or whether a narrower one has a stated
purpose. Both remain judgements for a reader, which is why the check names what it finds rather
than ruling on it.

**What would reopen this.** Coverage follows what has been undertaken, so a change in what is
funded is what would move it — in either direction. A narrower extent returning as a commitment
would be a new decision rather than a relaxation of this one, and would carry the same requirement
that it be declared rather than assumed.

---

## 6. Implementation Notes

- **Location.** Declared in each model's configuration, counted by the check in §5, and recorded per
  result by ADR-074's manifest.
- **Sequencing.** The data source is what makes a coverage declaration possible at all, and it is
  already in place fleet-wide; ADR-074 is the
  prerequisite for §2.2, since a result cannot declare its coverage until it declares anything.
- **Evidence.** `reports/2026-11-02_adr_claim_verification/00_CLAIMS.md` §4 holds every factual
  claim in this ADR with its source, including the coverage census and the delivery-cut relation.
  The change is also recorded in each model's configuration with its date, and in the
  runbook that sequenced it. Cell counts and row counts live there and in the purge record under
  `reports/`.

---

## References

**This repository:** ADR-009 (configuration validation) · ADR-032 (output schema) · ADR-074 (an
emitted artifact records what it is).

**Other repositories** — the contracts this ADR works within:

- [views-models ADR-019 — delivery declaration](https://github.com/views-platform/views-models/blob/main/docs/ADRs/019_delivery_declaration.md). Where the consumer deliverable's own coverage is declared — the cut referred to in §1, and the reason the wider coverage is required.
- [views-pipeline-core ADR-015 — model-specific input querysets](https://github.com/views-platform/views-pipeline-core/blob/main/documentation/ADRs/015_model_specific_inputdata_querysets.md). Establishes that coverage is declared per model in its own configuration, which is why this ADR states a coverage and not a cell count.
