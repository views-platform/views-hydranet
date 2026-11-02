# ADR-076: Moved — the provider-migration parity bar is pipeline-core's

**Status:** Moved out of this repository, 2026-11-02
**Where it went:** [views-pipeline-core#585](https://github.com/views-platform/views-pipeline-core/issues/585)

## Why this number is spent

ADR-076 was drafted here as *"A provider migration must prove decomposed parity on freshly pulled
data"* — the bar a data-source change must clear before it is trusted.

On review it said nothing about HydraNet: not its architecture, not its heads, not its emission,
not its configuration schema. Any model in the platform switching data sources needs the same
rule, and the rule sits between two ADRs **views-pipeline-core** already owns — its ADR-015 (a
model's input is declared in its own configuration) and ADR-023 (input drift detection).

⇒ So it was proposed to that repository instead of being kept in this one.

**This is not a superseded decision.** Nothing replaced it here, and nothing here depends on it.
ADR-072 and ADR-077 previously named it as a prerequisite; both now state the thing they actually
rely on — that every model declares the same data source — as a **fact about the configurations**
rather than as a reference to a decision.

## The number is not reused

ADR numbers are identifiers, not positions in a sequence. 076 stays spent so that any reference to
it, in a commit, an issue or another repository, still resolves to something that explains itself.
Later ADRs are **not** renumbered to close the gap.

## What was kept

The evidence, which is where it belongs:

- `reports/2026-07-28_datafactory_migration_dossier/` — the parity harness, the registered
  falsifiers and the decomposed residual from the one migration that has run.
- `reports/2026-11-02_adr_claim_verification/00_CLAIMS.md` §3 — the verified facts behind the
  claims the drafted ADR made, including the migration census and the preprocessing differences
  between the two sources.
