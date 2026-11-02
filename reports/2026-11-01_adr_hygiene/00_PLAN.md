# ADR hygiene — what ADRs are for, why ours are rotting, and the remediation

**Date:** 2026-11-01 · **Branch:** `adr/emission-provenance-and-selection` · **Status:** plan, nothing executed

---

# ⛔ Lead with the real hole — and first, a correction

## An ADR is authoritative over INTENT, never over FACT

This distinction is the foundation of everything below, and an earlier draft of this plan got it
wrong. Stated by Simon, and it is correct:

| | says | kind of statement |
|---|---|---|
| **ADR** | what the system *should* do | a **norm** |
| **code** | what the system *does* | a **fact** |

These are not two competing descriptions of one thing. A norm cannot be "out of date about
behaviour", because it never described behaviour. The code is the only account of what the code
does — and an ADR is the only account of what it was *supposed* to do.

⛔ **The error to avoid, which this plan's first draft made.**
`templates/000_use_of_adrs.md` says *"If code and ADRs disagree, the ADR is the source of truth —
or a new ADR is required."* Read **descriptively** ("the ADR tells you what the code does") that
licenses a reader to revert working code to a stale declaration. Read **normatively** ("the ADR
tells you what the code owes") it is correct — and the skipped clause *"or a new ADR is required"*
is the tell: it only makes sense if a divergence is an **open question**, not a tie to be broken.

⇒ ⛔ **No reader may conclude from an ADR that the code conforms.** It may conclude only that the
code *ought* to. Verifying it means reading the code.

⇒ A divergence is an **unresolved finding** with exactly two possible causes, and it must be
**flagged, never silently resolved**:

| cause | meaning |
|---|---|
| (a) | a decision was taken and never implemented |
| (b) | the code was implemented wrong |

⚠️ Keep one thing from the error: a careful reader *did* take the descriptive reading. ⇒ ADR-000
should **state which kind of authority it holds**, precisely because the other reading was
available.

## ⭐ The bridge, and therefore the real hole

> **ADR** (intent) → **test** (the machine-checked claim) → **code** (behaviour)

A test is the only thing that turns declared intent into verified intent. So:

> **An ADR with no test is intent that nothing verifies.**

That is exactly ADR-071's situation. It declares a model's data scope as Africa/ME (13,110 cells);
the config has said global land (64,818) since 2026-09-19; no test compares them; **nothing flagged
it for six weeks.**

⇒ The danger is not that the constitution mandates the purged data — it does not. The danger is
that **intent and reality diverged silently and indefinitely.**

⭐ And this is the same disease as the artifacts, one layer up. Artifacts carried no manifest, so
nothing could check what they were. ADRs name no test, so nothing checks whether they are honoured.
**Declarations that nothing verifies** — ADR-003's own prohibition, and it reaches the documents
that contain it. This is also the "guard that cannot fire" family already in the risk register,
appearing at the governance layer.

---

# 1. What an ADR is for — its unique responsibility

Every artifact in this repo answers a different question. Only one answers *why*.

| artifact | answers |
|---|---|
| code | what it does |
| tests | what must stay true |
| config | what the current settings are |
| reports / ledgers / dossiers | what we measured |
| **ADR** | **why it is this way, and what we gave up** |

Nothing else holds the *why*. That is the ADR's monopoly, and the whole reason the tier exists:
it stops us re-litigating settled questions and reversing decisions whose cost we have forgotten.

The constitution is already explicit about the boundary, and we are on the wrong side of it:

> *"An ADR records **a decision**, not a discussion or a design proposal."*
> *"An ADR answers: **'Why is the system the way it is?'** — not just 'How does it work?'"*
> *"An ADR is **not**: a full design document · a tutorial or user guide · a speculative roadmap ·
> a substitute for code comments · a place to argue indefinitely."*

And `active/000_standard_for_adrs.md` §1.6 and §2:

> *"describe **Semantic Intent** … rather than **Mechanical Format**"*
> *"New developers can understand the 'Physics' of the system **without reading every line of
> source code**."*

⇒ An ADR carrying 56 line-number citations inverts that last promise: you now need the source open
to read the ADR.

---

# 2. The broad principle — volatility tiers, and the rule that follows

This is the part worth getting right, because the smells are all one violation.

| tier | artifact | changes when | half-life |
|---|---|---|---|
| 1 constitutional | ADR 000–009 | the project's philosophy changes | years / never |
| 2 architectural | ADR 010+ | a structural choice is revisited | years |
| 3 contract | CICs, test invariants | an interface changes | months–years |
| 4 code | modules, functions | behaviour changes | weeks |
| 5 config | model configs | an experiment changes | **days** |
| 6 evidence | reports, ledgers, dossiers | every measurement | **hours** |
| 7 artifact | predictions on disk | every run | per run |

## ⭐ The rule

> **A document may reference *downward* in volatility — toward things more stable than itself.
> It may never depend *upward*, on something that changes faster than it does.**

Because a document **rots at the rate of its least stable reference**, not its own.

This is not a new rule. It is the **Stable Dependencies Principle**, which this repository already
holds as `active/002_topology_and_dependency_rules.md` — applied to code and never to documentation.

## The rule is predictive, which is how we know it is the right one

| ADR | least stable reference | predicted half-life | observed |
|---|---|---|---|
| **071** | a config value (tier 5) | days–weeks | ⛔ **stale in 6 weeks**, and wrong in the dangerous direction |
| **072** | 56 line numbers (tier 4) | one refactor | ⚠️ unverified; will rot on the next touch of those files |
| **075** (mine) | τ=0.20 (tier 5) + a measurement (tier 6) | days | ⚠️ already fragile on arrival |
| **074** (mine) | 14 line numbers (tier 4) | one refactor | ⚠️ same |

⇒ The staleness is **structural, not sloppy.** It was predictable from the references alone. That
is the broader finding: we do not need to catch stale ADRs, we need to stop ADRs from depending on
fast-moving things in the first place.

---

# 3. The operational rules — what to look for

Derived from §2, and each one testable by reading a single sentence:

| # | rule | ✅ allowed | ⛔ forbidden |
|---|---|---|---|
| R1 | **Names, not addresses** | `VolumeHandler`, `compose_samples`, a module name | `volume_handler.py:314-321` |
| R2 | **Fields, not values** | *"composition is configurable via `gate_threshold`"* | *"τ=0.20"*, *"τ=0.5"* |
| R3 | **Classes of thing, not instances** | *"the floor model"*, *"a datafactory-backed model"* | `violet_visitor`, `amber_kestrel` |
| R4 | **Decisions, not mechanics** | *"we chose X over Y because Z"* | *"Primer"*, *"Worked example"*, *"How it actually works"* |
| R5 | **Evidence by pointer, not by value** | *"measured in `reports/<dossier>/`"* | *"99.54% of cells"*, *"842,634 cell-months"* |
| R6 | **Scope by rule, not by roster** | *"PGM land coverage is declared per model in config"* | *"13,110 cells"*, `africa_me_legacy` |
| **R7** | **Name the test, or admit there is none** | *"enforced by `test_<name>`"* · *"⚠️ not yet enforced by any test"* | silence — which reads as "enforced" and is how intent diverges unnoticed |

⭐ **R7 is the one that would have caught ADR-071.** It follows from §0's bridge: an ADR that
cannot name the test enforcing it has not finished being written. Stating *"no test yet"* is a
perfectly acceptable answer — an honest gap is checkable, an unstated one is not.

⚠️ **R1 has a legitimate counter-pressure and it must be honoured.** `active/000` §1.8 *requires* a
`## References` section for cross-repo contracts, and ADR-000 lists *"traceability to code"* as
something the template enforces. ⇒ Traceability is wanted; **addresses** are not. Reference the
named thing, not its current coordinates.

**Exemptions, stated so the rules are usable:**
- `archive/` is **frozen history** — exempt from all of it. A superseded ADR naming a deleted model
  is a correct historical record.
- `templates/` describe the generic form — exempt.
- An ADR may name the ADR it supersedes, and may name a *version* (`0.1.8`) — those are stable.

---

# 4. Remediation — ordered

## Phase 0 · Mine first (074, 075)

I wrote the problem into the two ADRs I produced an hour ago: 3 model names, 34 line-number
citations, τ=0.20, 99.54%, 842,634, 2.09 GiB, 27 GiB. Both are **Proposed**, so editable without a
supersede.

- Strip to R1–R6. Replace every citation with a named reference; move every measurement to a
  pointer at `reports/2026-11-01_body_output_blast_radius/00_REPORT.md`.
- ⚠️ **Keep what is load-bearing as a *decision*:** that `forecast_composition` is read by the
  rollout feedback is the *reason* for 075's central rule. State it as a named fact about the
  feedback path, not as four line numbers.
- Re-run `generate_readme.py`, amend the commit.

## Phase 1 · ADR-071 — the dangerous one

Its violation is in its **title**, and its §2 contains *"Locked decisions … recommended
defaults"* — defaults are config, not architecture.

⛔ **This is not a hygiene edit. The decision itself is miscategorised**, so it needs a judgement
from Simon:

| option | what it means |
|---|---|
| **A — rewrite as the rule it should have been** (recommended) | *"A model's data provider and spatial scope are declared in its own config; a provider cutover requires proven parity before it ships."* Keeps the real decision, drops the instance. It is still Proposed, so this is an edit — but it changes what is decided, so it is Simon's call, not mine. |
| **B — supersede** | leave 071 as a historical record of a specific migration, mark it Superseded, write the rule as a new ADR. Safer against the never-delete rule; costs a number. |

⇒ Either way the **Africa/ME scope claim must stop being authoritative today**, because of §0.
A one-line status change (`Superseded` / `Deprecated`) removes the authority immediately and can
land before the rewrite is agreed.

## Phase 2 · ADR-072 — split decision from tutorial

56 citations, 5 model names, and sections the constitution names as explicitly not-an-ADR
(*"Primer"*, *"Worked example"*, *"How it actually works (with the real code)"*). The decision is
in §4; everything else is a guide.

- Keep: context, decision, alternatives, consequences, validation.
- Move the primer/worked-example/mechanics to a dossier or `docs/`, and link it.

## Phase 3 · The smaller ones

| ADR | fix | status |
|---|---|---|
| 070 | τ=0.5 → cite ADR-068's decision, not the number (R2) | Active — hygiene edit |
| 057, 053 | model names used as examples → name the class (R3) | Proposed — edit |
| 049 | the only Active model-naming ADR; low priority (R3) | Active — hygiene edit |
| 061 | a full config quoted as validation evidence → pointer (R5) | Active — defensible; review, may stand |

## Phase 4 · The gate — so this cannot recur

⛔ **Minimum machinery, and I am naming the trap:** Simon's own design preferences cite a config
validator that grew 303 → 1,167 lines and 17 → 111 guards, each one then on the path of every later
change. This must be **one test with two checks**, not a framework. If it needs a third rule, that
is a signal to stop and think, not to add it.

| check | mechanisable? |
|---|---|
| R1 — no `\.py:\d+` in `active/` or `proposed/` | ✅ yes, one regex, `archive/` and `templates/` exempt |
| R6 — no known-retired scope literal (`africa_me_legacy`) | ✅ yes, a short literal denylist |
| R3 — no model names | ⛔ **no.** Model names are two-word snake_case and so is much legitimate prose; a regex would be mostly false positives. **This stays a human review item on the ADR checklist.** Saying otherwise would be pretending. |
| R2, R4, R5 | ⛔ not reliably. Judgement. |

⇒ So the gate catches **two** of six rules. That is honest and still worth having: R1 is the
highest-volume smell (70 citations across four ADRs) and R6 is the one that is actively dangerous.

## Phase 5 · Where the rules live — a decision for Simon

The rules in §3 are themselves architecture, so they belong in an ADR, not in this report.

⇒ **Recommended: amend `active/000_standard_for_adrs.md`** rather than writing ADR-076. It is the
ADR about ADRs; a separate one would fork the standard.
⚠️ But 000 is Accepted and constitutional, so amending it is Simon's decision, not mine.

**Three things to add**, in priority order:

| # | addition | why |
|---|---|---|
| 1 | **An ADR is authoritative over intent, not over fact.** A reader may not conclude the code conforms; divergence is an open finding with the (a)/(b) causes, to be flagged and never silently resolved | the descriptive misreading was available to a careful reader and was actually made — §0 |
| 2 | **R7 — every ADR names the test that enforces it, or states plainly that none exists** | the only mechanism that converts declared intent into verified intent |
| 3 | **The volatility rule** — reference downward, never upward; R1–R6 | stops the rot structurally rather than by vigilance |

⇒ ⭐ Note the ordering: **1 and 2 matter more than the hygiene rules.** R1–R6 stop ADRs going
stale; 1 and 2 stop a stale ADR doing damage. A tidy ADR nobody verifies is still intent nobody
verifies.

⚠️ **Noted, not planned:** ADR-000's lifecycle gives `Proposed` no time bound. 071 sat Proposed and
rotted. Six of our seven Proposed ADRs carry smells versus one of fifty-three Active — the Proposed
tier is where volatility accumulates and nothing strips it. A staleness rule for Proposed may be
the real root enabler, but it is a separate decision and **out of scope here.**

---

# 5. Verification

| check | how |
|---|---|
| no ADR claims a scope the config contradicts | grep the live corpus for scope literals; reconcile each against the config that owns it |
| 074/075 carry no addresses or measurements | the Phase 4 regex, plus a read |
| the Africa divergence is resolved, not just tidied | 071 either no longer asserts the scope, or records explicitly that the scope moved and which side was wrong — cause (a) or (b), named |
| no ADR declares intent with no enforcement | every live ADR names its test or says it has none (R7); the count of "none yet" is recorded, not hidden |
| the index is true | `generate_readme.py` re-run; entries match titles |
| nothing silently lost | every removed measurement appears in the dossier it points to — checked by following each pointer |
| gates | `ruff check .` clean; `pytest` 0 failed; `test_falsification_repo_clean.py` green on a committed tree |

# 6. Risks

1. ⛔ **071 is an unresolved intent-vs-reality divergence, and it has sat unflagged for six weeks.**
   It does not mandate the purged data (§0) — but it does declare a scope the fleet abandoned, with
   nothing recording which side is wrong. The same-day cost is that a reader taking the descriptive
   reading would "fix" the code toward it; a one-line status change removes that reading today.
2. ⚠️ **Rewriting an ADR's decision is not hygiene.** Phase 1 option A changes what was decided.
   Flagged as Simon's call precisely because the never-delete rule exists to stop me doing that
   quietly.
3. ⚠️ **The gate could become the disease.** Two checks, two exemptions. Watch its line count.
4. ⚠️ **I am the author of two of the seven offenders**, having written them after reading the
   plan that said to be careful. The rules in §3 were derived *after* my own violation, so they are
   not a standard I met and then applied — they are one I failed and then wrote down.
