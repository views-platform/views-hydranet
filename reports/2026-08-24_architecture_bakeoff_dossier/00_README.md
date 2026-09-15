# Dossier — architecture bake-off for SPATIAL PRECISION (2026-08-24)

## Purpose

Six new architectures compete against the incumbent `HydraBNrecurrentUnet_06_LSTM4` on the one axis
our own measurements say the oracle gap lives on: **occurrence placement**. Two seeds each at
**L=300**, paired against the **existing** `fullzero_fortytwo`/`fortythree` controls (no baseline
retraining). Evaluation is the **full gate AND body battery** at h1/6/12/18/24/30/36 plus the oracle
ceiling — not AP alone.

## Why placement, and why now

* **M45 (2026-08-24)** — four independent interventions form a dose-response: ITF (act_ratio ×1.6,
  ΔAP −0.019), scheduled sampling (×5, −0.043), `truncated_nb` (×1170, −0.238). AP loss scales with how
  much each makes the model **fire**. The only rollout intervention that ever worked — the cell-state
  freeze, **+0.039** — is the only one that leaves firing alone.
* **Feedback-realism arms on the same converged vehicle** — `thin_0.75` (a real field with 25% of events
  DELETED) keeps **97%** of the oracle (0.4807 vs 0.4974); `spatial_scramble` costs **81%** (0.0925);
  real occurrence + the model's own magnitudes reaches **0.4888**. So **recall is nearly free,
  magnitude is worth ~2%, and placement is the whole thing.**

Every lever we have pulled so far acted on *how much* the model emits. None acted on *where*. The
architecture is the untouched surface, and the incumbent was chosen without exhausting the space.

## Relationship to prior work

| prior | how this relates |
|---|---|
| **ADR-061** static top-skip | the seam candidate (2) reuses — **retired for STATIC content** by the v2 CoordConv negative (C-228/C-230). This program re-tests the seam with **dynamic** content, which the retirement's stated mechanism does not cover. |
| **C-230** "raw-concat-wrong-primitive" | candidates (2) and (3) are the head-to-head that settles whether that verdict was about the primitive or the content. |
| **project_coordinate_grounding** (CLOSED) | CoordConv is a clean 3-seed negative; explicit coordinates are **not** re-tested. |
| **#258 / #262** | the parent rollout-collapse programme. |
| **M8 / M38 / M39 / M41** | the state-freeze line: state is implicated, and candidate (6) asks whether the recurrent memory is simply too small. |

## Document index

| # | file | status |
|---|---|---|
| 00 | `README` | living |
| 01 | `literature` | written — claims extracted 2026-08-24 |
| 02 | `design` | written — the six candidates |
| 03 | `harness_and_invariants` | **written — the gating document** |
| 04 | `roadmap` | written — phased and gated |
| 05 | `analysis_plan` | **not yet — blocked on 03's pre-flight being green** |
| 06 | `glossary` | written |
| 07 | `experiment_log` | empty (append-only) |

## Harness at a glance

The repo already carries a strong standing harness (sequential queue with `flock`, disk preflight,
HEAD-drift abort, **skip-if-scored resumability**, verify-after-every-arm with its exit code checked,
cube-deletion interlock, floor gate with an md5-pinned threshold, paired origin-block CIs that refuse
on support mismatch). **`03` audits it in full.** What is missing for *this* program:

1. an **architecture registry** (`choose_model` is a hardcoded `if/else`) — the OCP seam;
2. the queue's arm-identity tuple does **not** include `model` — a resumed queue could reuse an arm
   built on a different architecture;
3. no **preflight smoke** per architecture;
4. no **postflight setup audit** (artifacts + cross-arm consistency) after each arm;
5. no **baseline byte-identity** test for the registry refactor;
6. the device gate **warns but does not raise** — a CPU fallback would burn the 12.5 h timeout.

## Current state & next actions

- [ ] build the six architectures + per-architecture unit tests
- [ ] registry seam + baseline byte-identity test
- [ ] extend the queue's identity tuple to `model`, with a test
- [ ] preflight smoke harness (2 lessons × 6 architectures) + hard CUDA assert
- [ ] postflight setup audit, validated against today's completed arms as a positive control
- [ ] **then** `preregister` → `05_analysis_plan`
- [ ] smoke all six; project per-architecture cost; only then launch the 12-arm queue

## Conventions

Numbered docs, dated in-header, `git add -f` (the `reports/` tree is gitignored). Archive to
`reports/archived/` on close.
