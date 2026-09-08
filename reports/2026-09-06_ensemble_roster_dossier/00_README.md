# Ensemble roster selection — choosing the eight on evidence

**Opened:** 2026-09-06 · **Status:** RUN — 8 arms emitted + scored, maps for sb/ns/os, M66–M70 logged
**Chair's instruction:** *"a clean slate… and then the eight best models we think we can create for
the ensemble, and nothing else."*

## Purpose

`rusty_bucket` pools eight HydraNets into one forecast for `sb`, `ns` and `os`. The eight were
chosen when the roster was locked (views-hydranet#246) on the evidence then available. Three things
have changed since, and each of them bears on whether these are still the right eight:

1. **The cell-state clamp shipped** (ADR-027 §2.1, 2026-09-05): +0.0591 AP@h36, 4/4 seeds. It is
   going on all eight. **But it was measured on four `fullzero_*` test vehicles, not on these
   models**, and it changes what is fed back — which is the exact mechanism that makes the gate
   compositions diverge. **Every prior comparison between roster members was made without it.**
2. **The composition comparison was read off the wrong metric.** `gated_NB` and `th_gated_NB` were
   called "statistically tied" from `crps_all`, which on a 99.94%-zero field is dominated by true
   zeros — the same blindness that produced the h36 artifact. The full table disagrees: at h36,
   threshold has **higher AP (0.1726 vs 0.1623)**, fires more, and leaks more. Numerically the two
   compositions are not close at all: on a realistic gate field they disagree on **32% of cells**
   and their mean emitted value differs **79×**.
3. **Ensemble members are chosen for the wrong property.** Every comparison to date ranks members by
   *score*. An ensemble gains from members that fail **differently**. Two members with identical
   scores may be perfectly redundant or perfectly complementary, and nothing measured so far
   distinguishes those cases.

## The question this dossier answers

**Which eight configurations, pooled, produce the best ensemble** — measured on the pooled forecast,
not on member scores.

## What is NOT in question

- The eight *trained artifacts* exist and are used as-is. **No retraining.** Composition and the
  clamp are emit-time settings, so the whole design space below is reachable from the artifacts on
  disk.
- `mixture_nb` stays a candidate. It was **significantly better than plain NB on all three seeds**
  (CI excludes 0, p ≤ 0.003); it was "not promoted" only because the gain was **below the
  pre-registered 5% bar for replacing the ship candidate** — a far higher bar than earning an
  ensemble seat. Its governance gap (uncommitted code, no ADR) is real and tracked separately.

## Answers (2026-09-08 · `sb` unless stated · one origin, **one artifact per config, no seed
replication** — adjacent ranks sit inside C-119's ~20% training variance)

**The three pre-registered questions**

- **Q1 / F1 — the clamp transfers, and by more than it was sold for. F1 does not fire.**
  ΔAP@h36 **+0.14 to +0.19** on both test models and both compositions, against ADR-027 §2.1's
  +0.0591 on `fullzero_*` vehicles. `pink_pirate` unclamped is effectively dead from h18
  (AP 0.008 vs 0.268). ΔAP@h1 is **exactly 0.0000** — the VOID condition passing as a positive
  control (**M71**).
- **Q2 — the composition is a firing-rate dial, not a skill lever.** `crps_events` and
  `n_false_pos` are unchanged (ratio 1.00); `act_ratio` and `crps_none` move 1.3–4.3×. It buys
  nothing on placement or event magnitude and costs leak in proportion to firing (**M72**).
- **Q3 / F2 — the pool beats its best member at every horizon** (h18 AP 0.2203 vs 0.1804).
  **F2 does not fire.** But subsampling the same pool to 16 draws collapses it *below* the best
  member, and AP is monotone in S — the gain is **sample count, not member disagreement**
  (**M66**).

**What the questions did not anticipate**

- **The roster is not diverse.** Gate errors correlate **0.96–0.99** on all 28 pairs; giving every
  model a soft gate compresses AP@h18 into **0.228–0.286** across eight architectures (**M67**).
- **τ=0.5 is a delivery-shape decision, not a quality one** — it costs nothing if the gate ships
  alongside the fatality field, and destroys ranking if a single field ships. `rusty_bucket` pools
  the composed cubes, so the pooled product is the single-field case (**M68**, and read it with
  **M72**).
- **A multi-τ roster cannot work**: E_τ~U(0,1)[(gate ≥ τ)] = gate, so averaging thresholds
  reconstructs the soft gate from below and never crosses it (**M69**).
- **Point estimates**: the head's own `gate × mu` beats the mean of 16 draws by **+58% to +159%**
  at identical total mass; the `median` collapse is an empty map. Filed as views-hydranet#337
  (**M70**).

**Acceptance and deliverables**

- **K1 PASS 8/8** (`recall_mass` h36 **0.016–0.057** against a 0.01 floor — `bold_comet`
  clears it by only 1.6×), **K2 PASS 8/8**
  (`crps_none` *falls* h1→h36, ratio 0.41–0.85), **K3 VOID** — the comparator produced no
  scores (views-models#445), so **the roster has not been checked against a baseline** (**M73**).
- **27 forecast maps**, nine per target for `sb`/`ns`/`os`, re-emit verified non-destructive
  (**M74**).
- **views-datafactory#484** — two `lr_os_best` values of 27,413 and 32,505 in priogrid 149451
  (Darfur, 2025-10 and 2025-12) with a **2** in the month between and a next-largest of 1,100
  anywhere in the file. `os` magnitude for month 552 is provisional until resolved.

## Documents

| # | file | status |
|---|---|---|
| 00 | `00_README.md` | living |
| 03 | `03_harness_and_invariants.md` | **written 2026-09-08** — and names the pre-registered gates that did NOT run |
| 04 | `04_run_plan.md` | the 9-run validation plan |
| 05 | `05_analysis_plan.md` | **pre-registered before the first emit** |
| 06 | `06_acceptance_criteria.md` | K1/K2/K3 — **pre-registered 2026-09-07, before the runs** |
| 07 | `07_experiment_log.md` | **M66–M74 logged 2026-09-08** |
