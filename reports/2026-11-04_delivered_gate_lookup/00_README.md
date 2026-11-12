# E-1 — the empirical-lookup distribution on the DELIVERED forecast, conditioned on the gate

**Opened** 2026-11-04 · **Epic** #559 · **Tracking** #569 · **Status** pre-registered, no instrument
yet

---

## 1 · Purpose

One question: **does a predictive distribution built from what actually happened beat the delivered
negative binomial at h1, when cells are sorted by a conditioning variable that has values?**

That is Simon's research direction — *"preserve the ranking, but draw from a non-parametric
distribution that was closely fitted to the empirical, instead of trying to impose parametric
assumptions"* — tested on the one conditioning variable nobody has tried.

## 2 · Relationship to prior work

| artifact | what it gives this dossier |
|---|---|
| **`RP-3`** / epic **#468** (closed by #560) | the direction. Its status read *"CLOSED BY FALSIFICATION, not by measurement […] The arithmetic was never wrong; a label was."* |
| **`M89`** — `reports/2026-10-29_delivered_forecast/` | ⛔ **the adverse prior.** Ran this method on the delivered forecast and it lost: best 0.054232 against the emission's own 0.052109. ⭐ And measured **why**: the composed index has 1,321 distinct values and is exactly 0.0 on 99.6567% of cells |
| **L1–L5** — `reports/2026-10-2*_rp3_l*/` | the instruments, which this dossier **imports rather than rewrites**. ⛔ All five FINDING documents are retracted as forecast results (`W-107`, `C-399`) |
| **`reports/2026-11-03_nonparametric_distribution_programme.md`** | the literature and design record, 2,847 lines. ⇒ **This dossier has no `01_literature.md` or `02_design.md`; that document is them.** ⛔ Not executable as written — its own notes record contradictory solution/dead rules, colliding label namespaces and an improper primary score |
| **`reports/2026-10-30_upsample_arm_dossier/`** | the delivered artifacts this experiment reads, and `exp03_pairs.json`, which supplies `G-0`'s anchor |
| **#530** | `crps-concentration` / `crps-effective-n`, and the reason **no h36 number is produced** |

## 3 · Document index

| # | file | status |
|---|---|---|
| 00 | `00_README.md` | **living** — this file |
| — | `PREREG.md` | ⛔ **LOCKED 2026-11-04**, committed before any instrument. The verdict rule |
| 03 | `03_harness_and_invariants.md` | **living** — what is reused, what is frozen, what the frictions are |
| 05 | `05_analysis_plan.md` | stable — points at `PREREG.md`, restates the endpoints |
| 07 | `07_experiment_log.md` | **append-only** — empty at creation |
| — | `tools/` | ⛔ **empty at the pre-registration commit.** Instruments arrive with #563 |
| — | `results/` | empty. Captured `.out` files arrive with their instruments |

ℹ️ `01_literature.md` and `02_design.md` are deliberately absent — see §2.
ℹ️ Git does not track empty directories, so `tools/` and `results/` appear in the repository with
their first file. They exist on disk now.

## 4 · Harness at a glance

⭐⭐ **Nothing new is written for CRPS, binning, bootstrap or ranking.** All of it already exists,
frozen and self-tested; the marginal code is a **loader** and a **driver**. Full table with frozen
markers in `03_harness_and_invariants.md`.

⛔ The load-bearing risk is therefore **not** the scorer. It is a one-step misalignment between the
gate, the truth and the draws, which would produce a plausible wrong number that survives every
statistical check. `G-0`, `G-4`, `G-5` and `G-8` exist for that.

## 5 · Current state and next actions

- [x] #560 S0 — housekeeping; the input is protected (#557 closed as *delete nothing*)
- [x] #561 S1 — five terms locked in `reports/GLOSSARY.md` §4 and the new §4c
- [ ] #562 S2 — **this dossier and `PREREG.md`** ⬅️ in progress
- [ ] #563 S3 — the loader and its gates
- [ ] #564 S4 — the baseline
- [ ] #565 S5 — the lookup
- [ ] #566 S6 — the comparison and the verdict
- [ ] #567 S7 — the finding, the ledgers, the queue
- [ ] #568 S8 — the shape parameter in the body dump (independent of E-1)

## 6 · Conventions

Numbered docs, dated in-header; `00` living, `07` append-only. ⛔ `reports/` is gitignored, so
everything here is **`git add -f` with named files, never a directory** (`C-395`), and
`git show --stat HEAD` after every commit. ⛔ **`command grep`** for anything under `reports/` — the
shell's `grep` honours `.gitignore` and silently under-reports. ⛔ Measurement scripts are
**committed before they run** (`D-5`), with their gates declared in the module docstring. ⛔ A copy
gate's anchor is the committed implementation, never a ledger row (`W-98`). Every number names its
population, transform, estimator and normalisation (`C-351`, `GLOSSARY.md` §5e). On close the
directory moves to `reports/archived/`.
