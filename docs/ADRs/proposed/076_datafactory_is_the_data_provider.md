# ADR-076: views-datafactory is the data provider for HydraNet models

**Status:** Proposed
**Date:** 2026-11-01
**Deciders:** Simon Polichinel von der Maase
**Informed:** HydraNet maintainers · views-models maintainers
**Supersedes (in part):** ADR-071, which recorded this decision for a single model and bundled it with a separate scope decision · **Related:** ADR-077 (global land coverage — the decision this one unblocks)

## Summary (read this first — self-contained)

A HydraNet model gets its input data from a **provider**. Two exist: **viewser**, the older
platform service, and **views-datafactory**, the newer one.

This ADR decides that **views-datafactory is the provider for HydraNet models**, and that viewser
is no longer the target for new or migrated ones.

Two reasons, and they are independent. First, the non-conflict layer that viewser supplies — the
covariates that are not conflict counts — is not trusted. Second, running at global scale requires
the datafactory regardless, so a model that stays on viewser cannot follow the fleet.

⛔ **This ADR declares intent, not accomplishment.** At the time of writing the fleet is split:
most HydraNet models are on the datafactory and a minority are still on viewser. That gap is real
and is meant to be visible — see §5, which names the check that enumerates it. An ADR that claimed
the migration was finished would be a worse document than one that states the target and admits
the remainder.

## 1. Context

**The problem.** Provider choice was being made per model, as each migration came up, with the
reasoning recorded against whichever model happened to move. That produced a decision document
titled after one model (ADR-071), which left the fleet-wide rule unstated and let a per-model
config value sit in the architectural tier where it did not belong.

**What is already built, and must not be rebuilt.** The pipeline's data loader already inspects a
model's queryset declaration and dispatches on it: a viewser queryset object goes to viewser, and a
datafactory descriptor goes to the datafactory fetcher or to the frame-native input path. The swap
seam for any one model is therefore **a single config function** — the queryset generator in that
model's own config package.

⇒ This matters for the shape of the decision: because the seam is one config function per model,
the provider is **config, not architecture**. What belongs in this tier is the *rule* about which
provider to choose and what must be proven before choosing it — not any model's current setting.

**What was learned from the first migration.** A parity exercise was run against viewser truth on a
freshly pulled dataset, with pre-registered falsifiers. It passed, with a residual that was
decomposed and explained rather than eliminated. Two findings from it are load-bearing here and
are carried forward as rules in §2.

**Assumption that is no longer true.** It was assumed a provider migration could be judged by
byte-parity against the incumbent. It cannot: the two providers differ by intentional
preprocessing, including geographic-imprecision filtering and data vintage, so byte-parity is both
unachievable and the wrong bar.

**Urgency.** The provider is a prerequisite for scope (ADR-077). Leaving the rule unstated means
each remaining migration re-derives it, and the unmigrated remainder stays invisible.

---

## 2. Decision

**Statement.** *We will use views-datafactory as the data provider for HydraNet models, and we will
require a provider cutover to prove decomposed parity on a freshly pulled dataset before it ships.*

### 2.1 The provider rule

A HydraNet model declares its provider in its own config, through the queryset generator its config
package exposes. **The declared provider for HydraNet models is views-datafactory.** viewser is not
a target for new models or for migrations.

⇒ The ADR states which provider; **each model's config states that it uses it.** This document
names no model and no region, by ADR-000's volatility rule.

### 2.2 The acceptance bar for a cutover — decomposed parity, not byte-parity

A cutover must be justified against the incumbent by a parity exercise that:

1. runs on a **freshly pulled** dataset, never a cached one;
2. compares the **cell set**, the per-cell agreement and the extreme values;
3. **decomposes and explains** any residual, attributing it to a named cause;
4. is judged against **falsifiers pre-registered before the comparison**.

**Byte-parity is explicitly rejected as the bar** (§3).

### 2.3 Never validate a provider on a cached pull

A cached pull can be arbitrarily stale, and staleness is indistinguishable from a provider
difference. ⇒ A parity claim made against cached data is void. Pull fresh, or report that the
question was not answered.

⚠️ This rule earned itself: the first migration's cache was a year short of the remote, and the
discrepancy would have been read as a provider defect.

### 2.4 In-scope / out-of-scope

**In scope:** which provider HydraNet models target, and what must be proven to move one.

**Out of scope, deliberately:**
- **Spatial scope and coverage.** Which cells a model forecasts is a separate decision with its own
  rationale — ADR-077. Bundling the two is the defect this ADR exists to correct.
- **Which models have migrated.** That is config, enumerable by the check in §5.
- **The retirement of viewser as a platform service.** Not ours to decide; this ADR governs only
  what HydraNet models target.
- **Feature naming and renaming** across providers — governed by the output-schema ADR.

---

## 3. Rationale & Integrity Impact

| alternative | why rejected |
|---|---|
| **stay on viewser and graft the non-conflict layer from elsewhere** | the non-conflict layer is the thing not trusted, so grafting moves the problem rather than solving it — and global scale-up needs the datafactory regardless, so this defers a migration without avoiding it |
| **branch new models off an already-migrated one instead of migrating in place** | discards a model's experimental history and config lineage. The seam is one config function, so migrating in place is cheaper than re-establishing provenance |
| **byte-parity as the acceptance bar** | impossible and undesirable: the providers differ by intentional preprocessing and vintage. A bar that cannot be met gets quietly lowered, which is worse than a bar that is honest about residuals |
| **validate against the cached pull** | staleness is indistinguishable from a provider difference — §2.3 |
| **decide the provider per model as each comes up** | this is the status quo, and it is what put a per-model config value in the architectural tier and left the fleet rule unstated |

**Fortress state.** Reproducibility improves because the parity bar is now a rule rather than a
precedent: the next cutover has a stated standard instead of an exemplar to imitate. The
fresh-pull rule removes a class of false provider defect.

**Fail-loud.** §2.2's falsifiers are pre-registered, so a cutover cannot be justified after the
fact by whichever comparison happened to look acceptable. §5's check makes the unmigrated remainder
an enumerated fact rather than an assumption.

---

## 4. Consequences

### ✅ Positive (Benefits)

- [ ] The fleet rule is stated once, so each remaining migration inherits it instead of re-deriving it.
- [ ] The unmigrated remainder becomes **visible and countable** rather than assumed-empty.
- [ ] The parity standard is reusable: a bar, not a precedent.
- [ ] Removes a per-model config value from the architectural tier, which is what let its stale
      copy sit unchallenged (see ADR-000's volatility rule).
- [ ] Unblocks ADR-077 by stating the prerequisite explicitly.

### ⚠️ Negative (Costs)

- [ ] **Declares an intent the fleet does not yet meet.** A reader may mistake the declaration for
      the state — which is precisely why ADR-000 holds that an ADR is authoritative over intent and
      never over fact, and why §5's check exists.
- [ ] A cutover is now more expensive: fresh pull, decomposed residual, pre-registered falsifiers.
      Accepted — the alternative is a cheaper cutover nobody can defend.
- [ ] Models remaining on viewser are now formally off-target, which may force a decision about
      retiring rather than migrating them. That decision is not taken here.

---

## 5. Validation

**Invariants.**
- A model's provider is readable from its own config and from nowhere else.
- No ADR states any model's provider. This one states the target; configs state the actuals.

**Tests.**
- ⭐ **Enumeration check (the one that makes the gap visible):** a test that reads every HydraNet
  model's queryset declaration and reports every model whose provider is not the datafactory.
  ⚠️ **It must report, not fail**, while the migration is incomplete — a failing test on a known,
  accepted gap gets disabled, and a disabled test is how the gap goes invisible again. It converts
  to a failing assertion when the count reaches zero.
- *Red:* a parity claim computed from a cached pull is rejected by the parity harness.
- *Beige:* a model declaring an unknown provider fails loudly at config validation, rather than
  silently defaulting to either provider.

⚠️ **Honest gap (ADR-000 R7):** the enumeration check **does not exist yet**. It is specified here
and owed. Until it lands, the unmigrated remainder is knowable only by grepping configs by hand,
which is the condition this ADR is trying to end. Stated rather than implied.

**Failure mode that would trigger reconsideration.** If the datafactory's non-conflict layer turns
out to carry the same defects as viewser's, the first reason in §1 evaporates and only the
scale-up argument remains — which would make this a pure consequence of ADR-077 rather than an
independent decision.

---

## 6. Implementation Notes

- **Location.** The rule is enforced in each model's config package (the queryset generator) and
  checked by the enumeration test in §5. The dispatch mechanism it relies on is pipeline-core's and
  is not modified.
- **Relationship to ADR-071.** ADR-071 recorded this decision for one model, with that model's
  region bundled in. This ADR takes the general decision and the parity rules; ADR-077 takes the
  scope decision. ADR-071 should be marked superseded rather than deleted — its parity evidence for
  the first migration is a historical record worth keeping.
- **References.** ADR-000 (ADR standard, incl. the volatility rule and intent-vs-fact), ADR-009
  (boundary contracts and configuration validation), ADR-032 (output schema and feature naming),
  ADR-077 (global land coverage). Cross-repo: the pipeline-core data-loader contract and the
  datafactory consumer contract.
- **Evidence.** The datafactory migration dossier under `reports/` holds the parity harness, the
  pre-registered falsifiers and the decomposed residual for the first migration. Numbers live
  there, not here.
