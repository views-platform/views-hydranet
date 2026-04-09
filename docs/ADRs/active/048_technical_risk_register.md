# ADR 048: Technical Risk Register

| ADR Info            | Details                              |
|---------------------|--------------------------------------|
| Subject             | Governance and Risk Management       |
| ADR Number          | 048                                  |
| Status              | Accepted                             |
| Author              | Simon / Claude                       |
| Date                | 08.04.2026                           |

## 1. Context

The views-hydranet codebase accumulated 47 ADRs, 14 CICs, and extensive test infrastructure over its development lifecycle. During routine repo-assimilation audits, structural risks are identified — coupling hotspots, untested invariants, implicit assumptions, serialization fragilities. These findings were previously scattered across investigation reports, post-mortems, and conversation logs with no canonical location or tracking.

Without a formal register, risks discovered in one audit cycle are forgotten or re-discovered in the next. The register closes this loop by providing a single, append-only ledger of structural concerns with severity tiers, trigger conditions, and resolution tracking.

## 2. Decision

**The project maintains a Technical Risk Register at `reports/technical_risk_register.md` as a first-class governance artifact.**

### In-Scope

- Structural risks identified during repo-assimilation, expert reviews, tech debt audits, and falsification audits
- Severity classification via a 4-tier system
- Trigger conditions that describe when each risk becomes actionable
- Resolution tracking with dates and summaries

### Out-of-Scope

- Bug tracking (use GitHub Issues)
- Feature requests
- Performance benchmarks (use `reports/` investigation files)

## 3. Format

Each concern entry contains:

| Field | Description |
|-------|-------------|
| ID | Sequential: `C-xx` for concerns, `D-xx` for disagreements |
| Tier | 1 (Critical) through 4 (Low) — see tier definitions in register |
| Source | Origin of the finding: `repo-assimilation`, `expert-review`, `tech-debt-audit`, `falsification-audit`, `incident` |
| Trigger | The specific circumstance under which the risk becomes actionable |
| Location | File path(s) and line numbers |
| Description | Plain-language explanation of the risk and its consequences |

## 4. Lifecycle

1. **Opened:** During any structured audit (repo-assimilation, expert review, tech debt cleanup, falsification). The auditor assigns an ID, tier, trigger, and source.
2. **Reviewed:** During subsequent audits, existing concerns are re-evaluated. Tier may be adjusted if context changes.
3. **Resolved:** When the underlying code is changed to eliminate the risk. The concern moves to the "Resolved Concerns" section with a resolution date and summary. Resolved concerns are never deleted.

## 5. Tier Definitions

| Tier | Severity | Description |
|------|----------|-------------|
| 1 | Critical | Silent data corruption or model output correctness risk. Requires immediate attention. |
| 2 | High | Structural fragility that will cause failures under realistic change scenarios. |
| 3 | Medium | Maintainability or coupling issues that increase cost of change. |
| 4 | Low | Code quality concerns that do not affect correctness or reliability. |

## 6. Consequences

### Positive

- Risks discovered during audits are captured and tracked, not forgotten
- Tier classification helps prioritize remediation work
- Trigger conditions make risks actionable rather than abstract
- Resolution tracking provides an audit trail

### Negative

- Register maintenance is a manual process; stale entries are possible
- Tier assignments are subjective and may drift between auditors
- The register is a markdown file, not a database; querying and filtering require manual effort

## 7. References

- Register file: `reports/technical_risk_register.md`
- Initial seeding: repo-assimilation audit (2026-04-08), 14 concerns identified
- Related: ADR-004 (Rules for Evolution and Stability), ADR-005 (Testing as Critical Infrastructure)
