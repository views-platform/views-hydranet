# Silence vs Fade — is the free-running collapse occurrence-only?

**Opened** 2026-09-02 · **Branch** `exp/silence-vs-fade` · **Status** EXP-1 COMPLETE — **C1 FALSIFIED** (seed 42; replication pending)

## Purpose

One sentence is under test, because it has been asserted twice, published once in the ledger, and
retracted once already:

> *During free-running the model does not make **smaller** forecasts. It makes **fewer** of them.
> The ones it still makes are the same size as at month one.*

If true this is a core diagnostic: it says the failure is **placement/occurrence**, it is consistent
with M32/M45 (placement is everything, magnitude is capped), and it supplies the mechanism sentence
that M48/M50 currently rest on. If false, M50's mechanism is wrong for the second time and the
programme's reading of every firing intervention (M42/M45/M47) needs revisiting.

The chair's standing objection is the reason this dossier exists: *"some of your conclusions change
with the hour."* So the burden here is not to find a number that agrees — it is to **try to break the
claim** with an instrument that is independent of the one already shown to be faulty.

## Why the existing evidence does not settle it

| | |
|---|---|
| **Instrument already failed once** | The first M50 averaged an in-band `-1.0` UNDEFINED sentinel as a magnitude and published `18.4 → -0.8` (**C-318, Tier 2**). |
| **The correction uses the same file** | The surviving claim is the *same* CSV, merely filtered. A second, independent instrument has never been applied. |
| **n = 2** | The late-horizon magnitude rests on **2 surviving records out of 156**. |
| **The rival was never tested** | **Survivorship**: if the whole predictive distribution shifts down, only upper-tail cells clear the firing threshold, so "mean magnitude among active cells" can look flat while everything shrinks. |

## What the harness audit changed (read this before designing anything)

The instrument this dossier was opened to use **does not exist as described**. See `03` §B.1. In
short: `compose_mean` (multiplicative, `gate * mean`) governs the **autoregressive feedback**;
`compose_samples` (a **per-draw Bernoulli mask**) governs the **written cube**. So the proposed
"free decomposition" `mu = expm1(lr)/gate` recovers a *masked draw*, not the body mean, and its
variance blows up as `1/gate` — precisely in the late-horizon regime the claim is about.

What the audit found instead is **better**: an exact algebraic identity that splits the emitted field
into occurrence and magnitude with **no threshold and no conditioning**, so survivorship cannot
operate on it. That identity is the spine of the design (`02` §2).

## Document index

| # | File | Status |
|---|------|--------|
| 00 | `00_README.md` | living |
| 02 | `02_design.md` | **written** — the identity, the two instruments, the rivals |
| 03 | `03_harness_and_invariants.md` | **written** — crown jewel; includes the premise correction |
| 04 | `04_roadmap.md` | **written** — gated phases |
| 05 | `05_analysis_plan.md` | **pre-registration — LOCK before any run** |
| 07 | `07_experiment_log.md` | seeded, empty |

`01_literature` and `06_glossary` are deliberately not seeded: this program introduces no new method
and no new vocabulary. It uses `reports/GLOSSARY.md` unchanged.

## Result (2026-09-02)

**The claim is false.** Free-running does not merely make the model quieter — it makes it quieter
*and* smaller. h1 → h36: occurrence ×0.036 (28× fewer), plain body magnitude ×0.222 (4.5× smaller).
Both halves collapse. Full entry and falsifier verdicts in [`07`](07_experiment_log.md).

The unlooked-for finding is a third one: the **alignment** between where the model fires and where
it predicts large values decays 66.6× → 4.3×, so the cells that keep firing are disproportionately
the small ones. That, not survivorship upward, is why the gate-weighted magnitude falls 69× while
the plain mean falls 4.5×.

**Why the old claim looked true:** at h36 the conditioned "active cell" set holds **1 cell** at
τ=0.1 and **0** at every higher τ. The statistic that produced it had no support.

## Next actions

- [x] **LOCK** `05` before a single GPU second — `db20868`, amended `8e3b99c`
- [x] Build the `mu`-field dump (`03` §C.1), default-off, parity-tested, 15/15 mutations caught
- [x] G1 — **F3 fired and is left fired**; shown by measurement to be unsatisfiable (it fires on the
      known-good control). Chair authorised proceeding with it on the record.
- [x] EXP-1 seed 42, treatment `identity` + control `use_real`
- [ ] **Replicate on seed 43** — 05 §7 makes replication a condition of the finding
- [ ] Retract M50's mechanism sentence; correct the #262 comment that rests on it
- [ ] Register: the grid-orientation trap, and the unsatisfiable-band defect in F3

## Conventions

Dated docs, `00` living, append-only `07`, git-tracked via `git add -f`, archived to
`reports/archived/` on close. Vocabulary is `reports/GLOSSARY.md` only.
