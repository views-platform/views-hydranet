# 00 — README · the ranking head

**Serves:** the whole of Simon's question — *"if we changed the 'magnitude head' to a 'ranking head'
do we think we could rank better?"* — and holds the programme's destination so a fresh context
cannot lose it.

**Opened** 2026-11-10 · **living document** · guarded by `tests/test_ranking_head_destination.py`

---

## ⭐⭐ THE FINISH LINE — the one number this programme exists to produce

> **A `trained` ranking head's globally-measured `ρ_event` (`rho_event`), against the body's
> `+0.3127 ± 0.0713`.**

`ρ_event` = Spearman correlation with truth on **event cells** (`open_m & truth > 0`), measured on
the **global** 360×720 grid — not within a training patch. Benchmarks, measured over 7 folds and 10
delivered arms:

| | `ρ_event` | what it is |
|---|---|---|
| the **body** today (NB magnitude head) | **+0.3127 ± 0.0713** | ⭐ the thing we are trying to beat |
| the **gate** today (occurrence head) | +0.4418 ± 0.0583 | the benchmark to reach — **not** the thing being improved |

⛔ **The gate is not the target.** It appears in this dossier in exactly three licensed roles: it
composes the forecast (`ADR-069`), it is the ordering benchmark above, and it defines the firing
set. A fourth use is drift, and the guard test counts it.

⚠️ The programme ends when that number exists for a trained ranking head — in any of the handover's
**three** outcomes: it beats the body · the body holds · **not resolvable at this `n`**. The third
is a result, not a loss.

---

## 1 Purpose

Today's model emits **`gate × magnitude body`** — an occurrence probability times a negative-binomial
count. This programme designs **`gate × ranking body × nonparametric empirical draw`**: the body
stops predicting *how big* and predicts *which is bigger*, and the magnitude comes from the
empirical distribution of what actually happened at that rank.

Call them **A** (today) and **B** (proposed). Epic #579 established that the question *"is the
rank-coupled draw measurable?"* resolves to **answer 3 — the question moves to the body's ranking.**
This dossier is the design of the thing that would move it: a head whose **training objective** is
ordering. Every instrument built so far re-scored a frozen cube, and no frozen-cube test can answer
a question about a training objective.

## 2 Relationship to prior work

| | |
|---|---|
| **#578** | the handover that scoped this: `reports/2026-11-09_HANDOVER_ranking_body.md`. ⭐ Its §2 definition of done and §3 "do not do these" **govern this dossier** |
| **#579** | `reports/2026-11-10_rank_coupling_verdict/` — the τ sweep, the `MCB − DSC + UNC` attribution, and `λ*`. Supplies §1.3 of `02_design` and the `tie@0` measurement |
| **#577** | *"how far up the magnitude range is the conditional mean a rankable target"* — the same axis from the magnitude side. Cross-reference, not merged |
| `ADR-067` | the distribution-family subsystem and its strangler-fig registry seam — **the route a new head takes in** |
| `ADR-069` | `forecast_composition` — where `gate` and body become one forecast |
| `ADR-073` (proposed) | what a model emits from need not be what it was trained on — the nearest existing precedent for a train/emit split |
| **exits to** | a proposed ADR under `docs/ADRs/proposed/` if the design validates; this dossier is the staging area, not the decision |

## 3 Document index

| # | file | role | status |
|---|---|---|---|
| 00 | `00_README.md` | this spine — purpose, destination, index, next actions, conventions | **living** |
| 01 | `01_literature.md` | what we take from each of the 14 papers; the gaps that remain | seeded |
| 02 | `02_design.md` | ⭐ **the maths** — the loss, the label transform, the ranked list, the emission | seeded |
| 03 | `03_harness_and_invariants.md` | the guardrails as *discovered*, the gaps, the pre-flight checklist | seeded, checklist **RED** |
| 04 | `04_roadmap.md` | gated phases from loss numerics to the first run | seeded |
| 05 | `05_analysis_plan.md` | ⛔ **STUB — blocked.** Two blockers named inside, one owner each | **blocked** |
| 07 | `07_experiment_log.md` | append-only ledger, negatives first-class. **No entries yet** | empty |

⛔ **There is no `06_glossary`.** The canonical dossier set has one, and this dossier deliberately
omits it: `reports/GLOSSARY.md` is *"LOCKED VOCABULARY — the only words I use"*, and a dossier-local
glossary would be a **second** vocabulary — exactly what the drift rule exists to prevent. The 15
new terms went into the locked file instead, as §3c and §4d, stamped `LOCKED 2026-11-10`.

## 4 The harness at a glance

⭐ **Most of it already exists** — detail and evidence in `03`. Reusable as-is: the `ADR-067`
registry seam (a new family is one file + one entry, pinned by a test), a **byte-exact parity
anchor** over all six legacy distributions, `lock_entropy` wired at three production sites, disk
guards, `run_queue.sh` one-arm-at-a-time scheduling, `setsid` discipline, the non-finite halt, the
retrain-free `body_mean_dump_dir` seam, and 2,443 tests under the `ADR-005` taxonomy.

⛔ **Three gaps gate this programme's first run**, all in `04` phase 1: a new loss *shape* needs its
own branch at the `isinstance` dispatch (the registry alone does not route it); **no registered loss
couples cells to each other** so a listwise term would be the first of its kind; and **nothing
asserts the criterion returned a scalar**.

## 5 Next actions — ⛔ a CLOSED budget of 5

⚠️ **Adding a row here is a request to Simon, not an edit.**
`tests/test_ranking_head_destination.py::test_the_next_actions_list_is_the_approved_length` fails if
the count moves, which is how an added row becomes visible rather than quiet.

- [ ] **1 · Simon:** approve a primary superiority quantity for `05`. ⚠️ `MCB` is **contaminated** —
      it was measured before it was proposed (`MCB_A` 0.138 vs `MCB_B` 2.952), which is what
      `C-3531` forbids. A quantity whose value on a *trained* head is unknown is still available.
- [ ] **2 · `baseline` session:** seed-to-seed dispersion of whatever `#1` names, `ddof=1` declared.
      Sets the margin. Handover §2 constraint 4: *the margin is in units of measured dispersion,
      never percent.* ⛔ Not a go/no-go.
- [ ] **3 · `library` session:** close the audit on `Mohler2020_CrimeRankHotspots` so `01` may take
      claims from it. Until then it is cited as *ingested, audit pending*.
- [x] **4 · next session:** implement the loss — `04` phase 1. Zhu Eq. 2 with the `02 §2` label
      transform, behind a default-off registry entry, with the numerics tested first.
      ⭐ **DONE 2026-11-10:** `views_hydranet/utils/listwise_level_loss.py`, registry key
      `listwise_level`, config field `loss_reg_y_max`. `tests/test_listwise_level_loss.py` — 23
      tests, the seven properties of `04` phase 1 each with a refusing half. ⚠️ Nothing is trained
      by it; it is reachable only by naming it in `loss_reg`, which no config does.
- [ ] **5 · next session:** the patch→global ordering probe — `04` phase 3. The one cheap readout
      that bears on the finish line before anything is trained to convergence.

## 6 Conventions

**Numbering and dating.** Numbered docs; `00` is living, the rest are point-in-time and dated
in-header. On close the whole directory moves to `reports/archived/`.

**Git.** `reports/` is gitignored but its files are tracked, so every file here is force-added **by
name** — never by directory (`C-395`). ⛔ A recursive `grep` under `reports/` silently under-reports
(`C-407`); use `command grep`.

⛔⛔ **The kill rule.** **No sentence in this dossier may conclude against B without citing a
measurement from a trained ranking head.** None exists, so there can be no such sentence. Up-front
work *informs*; it gets **no veto**. This is Simon's standing instruction — *"any idea we propose,
if you throw it around long enough, you end up refuting it and killing it. 100% of the time."* A
retracted past claim may be quoted, but only on a line that marks it withdrawn. The guard test
enforces both halves.

⛔ **Scope.** Every document opens with a `**Serves:**` line naming the part of the question it
answers. Anything noticed in passing goes to `reports/technical_risk_register.md` or to a
*"noted, not fixed"* line — **never into the work.**

**Evidence.** Nothing enters this dossier that is not backed by an artifact on disk whose inputs are
also on disk. A `.out` file is a note until both halves are there.

**Risks** fold into the risk register, not this dossier. **The decision** graduates to
`docs/ADRs/proposed/`.
