# 04 — Roadmap

**Date:** 2026-08-15 · **Epic:** #263 · **Tracking:** #272

Each story ends with a **gate**: a checkable condition, not a judgement call. Do not start N+1 until N's gate
is green. **If a story exceeds 1.5× its estimate: stop, log the partial in `07`, escalate.**

| # | Issue | Story | Gate | Est |
|:--:|:--:|---|---|:--:|
| S0 | #264 | Dossier + absorb DRAFT + lock `05` & `SCOPE.md` | `05` carries `LOCKED <date>`; `SCOPE.md` ≥18 exclusions; DRAFT superseded | 1.5h |
| S1 | #265 | Exact CRPS-gap decomposition from archived CSVs | max `\|residual\|` < 1e-9 on every archived row; provisional finding in `07` | 1.5h |
| S2 | #266 | Partition & provenance audit | `partition_audit.json` per arm, all `leak: false`, `rollout_feedback: 'sample'`, truth sha matches | 2.5h |
| S3 | #267 | FAO-02 climatology + skill score | climatology passes unchanged through `_metric_row`; S>1; `crpss` raises on a 1-sample ref | 3.5h |
| S4 | #268 | MDE, C-252 memory assertion, pin the CI-support bug | `mde_h*/MDE.md` state a number; the bug is `xfail(strict=True)` + registered, **not fixed** | 2.5h |
| S5 | #269 | C-224 Taillardat index (DIAGNOSTIC-only) | 9 numbers in one table; tests #17–#20 green; **≤120+120 lines** | 3.5h |
| S6 | #270 | Re-score surviving cubes + verdict | `rescore.csv` complete; verdict token addressing `crps_all` **and** AP | 4.0h |
| S7 | #271 | Close-out | register integrity green; promote-vs-park recorded; #262/#249 cross-linked | 1.0h |

## Dependency graph

```
S0 ──┬─> S1 ──┐
     └─> S2 ──┴─> S3 ──┬─> S4 ──┐
                       └─> S5 ──┴─> S6 ──> S7
```

## Sequencing rationale

- **S1 first** — it delivers ~75% of the headline answer for ~8% of the effort, from archived CSVs with no
  cubes. If everything downstream stalls, a defensible partial answer already exists.
- **S2 before S3** — a leak finding would invalidate everything downstream. Fail early and cheaply: the audit
  reads `identifiers.npz` only, never a 2.5 GB cube.
- **S5 second-to-last** — it is the entry most likely to expand, so it is scheduled where a hard stop costs the
  least, and its gate is a line count rather than a judgement.

## Decision points

| When | Decision | Recommendation |
|---|---|---|
| S0 | C-218 as a pytest behind `VIEWS_MODELS_ROOT`, or runtime fail-loud only? | **Runtime only** — a pytest would hardcode a cross-repo path (the C-247 sin) and skip in CI anyway |
| S5 | Does `diag_Tu` separate the tail-differing pair? | If no: record it, do **not** extend the method |
| S7 | Promote `02_design` to a proposed ADR, or park? | **Park** — promotion normally follows a *validated* design; a second independent use (#249) is the natural trigger |
