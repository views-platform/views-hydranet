# 05 — Analysis plan (pre-registered)

# **LOCKED 2026-08-15**

> Locked **before any number is computed**. Nothing below may be changed after S1 runs. If something here turns
> out to be wrong, that is a logged finding in `07_experiment_log.md` and a *new* pre-registration — not an edit
> to this file.

---

## The question

**Was `gated_NB`'s h36 "win" over climatology real, or a zero-driven artifact?**

The V2 scoreboard reported `gated_NB` h36 `crps_all` **0.877** vs `light_strider` **0.960** — a 8.7%
improvement on the FAO-02 primary metric — while simultaneously being **worse on AP** (0.162 vs 0.195), with
`size_ratio = 0.0000` and `mcr_all = 0.0239`.

## Hypothesis

**H1.** The `crps_all` gap is carried predominantly by true-zero cells (confident zeros on a 99% empty field),
not by event skill. Formally: `zero_share_of_gap > 0.5`.

**H0 (the thing that would make it a real win).** The gap is carried by event cells and accompanied by better
occurrence ranking: `crpss_vs_clim ≥ 0.05` with `ΔAP > 0`.

## The one variable

None — this is not an intervention experiment. It is a **measurement-instrument** programme: the same stored
predictions, scored on a ruler with a reference forecast, a skill score, and a decomposition it did not have
before.

---

## Method

### Substrate
- **Arms:** the surviving 2026-08-12/13 cubes (13 origins, months 457–504, S = 16, cube `(471960, 16)`):
  `violet_visitor` (gated_NB soft_gate, seed 42 — the ship candidate), `bright_starship` (seed 43),
  `blazing_meteor` (th_gated τ=0.5), and the three mixture arms.
- **Reference:** the **in-code FAO-02 empirical conflictology baseline** built in S3.
- **Truth:** the pinned `v2_ruler.V2_TRUTH` / `V2_TRUTH_SHA256`.
- **Support:** identical `(origin, cell)` set across all arms at every horizon — the cross-arm intersection of
  `_support_keys`, matching the V2 scoreboard's 170,430 cells.

### Parameters — locked
| Parameter | Value |
|---|---|
| **Horizons** | `(1, 6, 12, 18, 24, 30, 36)` — **7, not 36.** Bounds runtime; pre-registered so it is not a post-hoc choice. |
| **Targets** | `sb`, `ns`, `os`. **Headline target: `sb`.** |
| **Climatology window** | 36 months, strictly `[m0 − 36, m0 − 1]` — **pre-origin only.** |
| **Climatology draws** | `S = 16` (matches the arms' cube width). |
| **Climatology seed** | `0`. |
| **FAO-02 superiority** | CRPSS ≥ **0.05** vs the reference. |
| **FAO-02 non-inferiority** | guardrails ≥ **0.01** better. |
| **CI** | 90%, **origin-block bootstrap** (never iid over cells), P = 13. |
| **Taillardat q** | **{0.99, 0.995, 0.999} only. No optimisation over q.** |
| **Min exceedances** | `m ≥ 50`, else `nan` + a reason string. |

### Metrics reported per (arm, target, h)
`crps_all` · `crps_none` · `crps_events` · `AP` · `Brier` · `crpss_vs_clim` · `zero_share_of_gap` ·
`act_ratio` · `size_ratio` · `mcr_all` · `n_origins` · `diag_Tu` (DIAGNOSTIC).

**The headline is never a bare `crps_all`.** The assembler raises on a row missing the split, `AP`,
`crpss_vs_clim`, or `zero_share_of_gap` (C-219 as code, not norm).

---

## Pre-registered predictions

| # | Prediction |
|---|---|
| **P1** | `zero_share_of_gap` at sb/h36 lands in **(0.70, 0.80)** — the archived-CSV arithmetic gives 0.759. |
| **P2** | The decomposition identity reconciles to `<1e-9` on **every** archived V2 row. |
| **P3** | `ΔAP < 0` at h36 (the model ranks events *worse* than climatology). |
| **P4** | The all-zero degenerate forecast scores `crpss < 0` while its raw `crps_all` looks unremarkable. |
| **P5** | The in-code climatology's `crps_all` **will not** reproduce `light_strider`'s 0.960 — different object. Only the **sign and decomposition** of the gap are comparable. *Pre-declared so this is not read as a bug at read time.* |
| **P6** | At P = 13 the MDE will be large enough that the h36 CRPSS difference is **not** separable from 0 at 90%. |

## Falsifiers (pre-committed)

| # | Falsifier — fires if… | Consequence |
|---|---|---|
| **F1** | the decomposition identity does **not** reconcile (`residual > 1e-9`) on archived rows | the archived CSVs are internally inconsistent ⇒ **stop**; the V2 scoreboard's numbers cannot be trusted at all, which is a bigger finding than this epic |
| **F2** | `partition_audit` finds `min_emitted_month <= train_max` for any arm | that arm is **in-sample** ⇒ drop it; if it is the headline arm, the epic's question is unanswerable on this substrate |
| **F3** | any arm's cube is `(N, 1)` or `(N,)` | it is not a predictive distribution; CRPS on it is MAE ⇒ drop the arm (C-220) |
| **F4** | the climatology is **not** byte-identical under permutation of post-origin truth | leakage ⇒ the reference is invalid; **stop** and fix before any score |
| **F5** | `rollout_feedback != 'sample'` for a scored arm without `diagnostic_only=True` | it is a broken-by-construction rollout (C-218) ⇒ label it diagnostic, never "deployed skill" |
| **F6** | S5 exceeds the 120-line cap | **STOP**, log the partial, escalate — do not push through |

---

## Decision rule — pre-committed

Evaluated at **sb, h36**:

```
ARTIFACT      iff  zero_share_of_gap > 0.5  AND  ΔAP < 0
REAL          iff  crpss_vs_clim >= 0.05    AND  ΔAP > 0  AND  the 90% CI excludes 0
UNDECIDABLE   otherwise
```

**`diag_Tu` appears in no branch of this rule.** It is reported and never selected on.

The verdict token is written to `07_experiment_log.md` in S6. It is determined by this rule applied to the
numbers, not by judgement at read time.

---

## Skepticism ledger

1. **P = 13 is the binding inferential sample**, not the ~170k cells (C-254 — Giacomini–White's asymptotics
   are in the number of forecast periods). The 13 origins are also *adjacent* months, so origin blocks fix
   within-origin but not between-origin correlation. **Report the MDE; do not try to fix it.** Calling this
   "the Giacomini–White test" would be overclaiming — `gw_stratified` is a bootstrap on the mean loss
   differential, not a GW conditional-predictive-ability regression.
2. **The reference is a different object from `light_strider`.** See P5. A reader comparing our climatology's
   absolute number to 0.960 will be comparing two different baselines.
3. **The archived CSVs are aggregates.** No per-cell losses survive, so S1's decomposition is exact arithmetic
   on means — it can attribute the gap, but it cannot produce a CI. The CI comes from S6 on the surviving cubes.
4. **The surviving arms are 160-lesson ensemble members**, not the 300-lesson V2 scoreboard arms. The
   substrate is comparable (same partition, support, S) but not identical. State this in the verdict.
5. **Single-seed reads are INDICATIVE.** The repo rule is ≥3 seeds to rank. Two gated_NB seeds survive
   (42, 43); use both where possible and label accordingly.
6. **Meta-pattern 8 (DRAFT §3): invalid knowledge from a bug.** A wrong implementation silently produces a
   confident verdict — this happened twice in this programme (the buggy zero-truncated sampler flipped a
   verdict from "bloom" to "collapse"; `emit_family_core`'s half-wiring overturned an F2). **The countermeasure
   is process, not a metric:** S2's provenance audit runs *before* any cube is scored, and every new pure
   function ships with a hand-computed test.

---

## What this plan does NOT decide

Whether to *act* on the verdict. If the h36 win is an artifact, that fact is the deliverable — the remedy is
out of scope (`SCOPE.md` #1).
