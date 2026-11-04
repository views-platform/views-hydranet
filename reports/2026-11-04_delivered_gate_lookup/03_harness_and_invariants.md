# 03 — Harness and invariants

**2026-11-04.** What already exists, what is frozen, what the frictions are, and the command behind
every number this dossier asserts.

---

## A · Invariant taxonomy

### A1 · HARD — must not change, and this experiment does not change them

| invariant | where |
|---|---|
| `lodestar_score.py` is the **frozen** T=0 ruler — *"FROZEN once the self-test passes; a change is a new version and re-scores all"* | `reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py:1-5` |
| `binned_cdf.py` is **effectively frozen** — *"NOT re-derived, NOT tidied and NOT improved — a lifted scorer that has been 'cleaned up' is a different scorer"*; its AST is gated by `G-S1` | `reports/2026-10-28_rp3_l3/binned_cdf.py:4-18` |
| `l3a_split_sensitivity.py`'s inline duplicate is **deliberate** and must not be deduplicated | `reports/2026-10-28_rp3_l3/l3a_split_sensitivity.py` |
| FAO-02 governs scoring and selection; twCRPS, LogScore and PIT are **rejected** | `reference_fao02_locked_eval_framework` |
| `pod_run_model.sh` is **never-modify**; views-models is **read-only** for this session | standing |
| `rollout_ruler_core.py` is **size-capped at 120 lines** from its cap marker to EOF; exceeding it is a STOP condition | `scripts/rollout_ruler_core.py:780-783` |

### A2 · DELIBERATELY CHANGED by this experiment — exactly one thing

| | |
|---|---|
| **the conditioning index** | from the **composed**, post-`threshold_gate` index (`M89`) to the **gate**. ⛔ Nothing else moves: same lifted scorer, same equal-count binning, same truth, same cell-months on both sides |

### A3 · RESPECT WHILE CHANGING

- The delivered artifacts are **read-only inputs**. ⛔ Nothing in `views-models/models/*/data/` is
  written, moved or deleted.
- `crps_concentration`'s refusal of a negative per-cell vector is **respected, not worked around** —
  the plug-in vector is passed, and the headline says which estimator it is.
- The epic produces a **diagnostic**. No promotion is proposed and FAO-02 is untouched.

---

## B · The standing harness — what already exists, audited 2026-11-04

⭐⭐ **Nothing new is written for CRPS, binning, bootstrap or ranking.**

| reuse | from | gives | frozen? |
|---|---|---|---|
| `binned_cdf_crps(x_fit, y_fit, x_sco, y_sco, B)`, `self_test()`, `pair_sum_sorted(v)` | `reports/2026-10-28_rp3_l3/binned_cdf.py:26,36,103` | pooled plug-in + fair CRPS of the equal-count lookup, **exact, no sampling**; mean members per occupied bin; fraction of score cells falling in an **empty** fit bin | ⛔ **yes**, AST-gated |
| `equal_count_edges(xf_sorted, B)`, `score_cells(...) -> {plug, fair, bin, bin_n, bin_npos, bin_mean, occ}` | `reports/2026-10-29_rp3_l2/l2_scorer.py:25,38` | the **per-cell** plug-in and fair vectors plus per-bin diagnostics. Its existing callers assert its mean against `binned_cdf_crps` to **1e-12** | effectively |
| `crps_ensemble(Y, z) -> (plug[N], fair[N])`; `ranks(x, rule)`; `auc(score, pos)`; `ff_sweep(idx, rule, tie)` | `reports/2026-10-29_delivered_forecast/ar_score.py:138,151,166,370` | the emission's per-cell CRPS, tie-aware ranks, vectorised AUC. ⭐ `ff_sweep` is already **parameterised by the index**, which is why this experiment is a driver and not a rewrite | lifted, pinned by test |
| `gw_stratified_test(loss_a, loss_b, stratum, origin_ids, *, n_boot, seed, ci, resample="origin")` | `reports/2026-07-29_v2_scoreboard_dossier/tools/gw_stratified.py:78` | mean difference, % reduction, **origin-blocked** interval, bootstrap p. ⭐ Takes **bare per-cell loss vectors** — exactly our shape | — |
| `crps_concentration(crps, truth, *, edges, where)`; `require_headline_columns(row, *, where)`; `HEADLINE_COLUMNS` (8) | `scripts/rollout_ruler_core.py:413,531,70` | `F3`'s instrument and the mandatory reporting columns | shipped #537 |
| `average_precision(y_bin, score)` | `reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py:36` | `G-0`'s anchor | ⛔ **FROZEN** |
| `placement_rho(pred, truth, population="events", pooling="pooled")` | `scripts/placement_rho.py:96` | `F4`'s event-cell Spearman | — |

---

## C · ⚠️ Three frictions — real, none blocking, all stated before the run

**1. ⛔ `crps_concentration` structurally refuses the fair estimator.**
`scripts/rollout_ruler_core.py:465-474`, verbatim: *"the fair estimator can be negative per cell and
must not be passed here."* ⇒ `require_headline_columns` **cannot accept a fair-CRPS headline row
today.** Decision: the **plug-in** per-cell vector feeds `crps_concentration`; the headline pooled
score is **fair**; both are printed and the estimator is named on every line. ⚠️ This is a wart in a
guard shipped 2026-11-04 (#537) — **noted, not fixed here.** Scope.

**2. ⚠️ `fair-crps` exists only under `reports/`, in ten files, covered by no pytest.**
Counted 2026-11-04 by grepping the `2m(m−1)` denominator; the ten are named in `GLOSSARY.md` §4.
⛔ None in `scripts/`, `views_hydranet/` or `tests/`. ⇒ We **reuse in place and do not consolidate**
(WET). #442 is open on which estimator FAO-02 makes primary.

**3. ⛔ No CMA, CID or Kendall implementation exists anywhere.**
The 2026-11-03 document prescribes CMA with a DeLong-type test for exactly this comparison and it is
unimplemented. ⇒ `F4` uses **Spearman** and **AP**, and `PREREG.md` §6 states that Spearman cannot
attain its maximum on a non-continuous outcome (`Walz2026` **C-3605**), so **F4 is a guardrail, not
an endpoint.** Building CMA is out of scope.

---

## D · PRE-FLIGHT CHECKLIST — every item verified 2026-11-04, with its command

| ✅ | fact | command |
|---|---|---|
| ✅ | the delivery is intact | `sha256sum -c gate_endpoints.sha256` → **OK** |
| ✅ | arm identity | `cat MANIFEST` → `model: ctlfullzero_fortytwo`, `git: 651e625`, `lessons: 300`, `mode: production` |
| ✅ | **13 origins**, h1 months **457…469**, h36 **492…504**, all three targets | read `gate_endpoints/SLICE_MANIFEST.json`; 13 dirs on disk **and** 13 keys in the manifest |
| ✅ | gate shape **[64818, 16]** float32, range `1.83e-06`…`0.9954` over draws | `np.load(origin_0/by_sb_best/y_pred_h1.npy)` |
| ✅ | **the gate has support: 64,732 distinct values of 64,818, exactly 0.0 on NONE** | `np.unique(y.mean(axis=1)).size`; `(g==0).sum()` → **0** |
| ✅ | truth is **raw counts**, max **113,395** at `priogrid 148759, month 504` | `pq.read_table(... ['month_id','priogrid_id','lr_sb_best'])`, then `y.max()` and the selecting mask |
| ✅ | h1 window 457–469: **842,634** cell-months, **2,905** events (**0.3448%**), max **2,482** | boolean mask on `month_id`, then `.sum()` / `.mean()` / `.max()` |
| ✅ | fit 457–462: **388,908** / **1,333** events / max **2,482**; score 463–469: **453,726** / **1,572** / max **681** | same, per window |
| ✅ | ⭐ **fit max > score max**, so the record-event ceiling does not bind | the two maxima above |
| ✅ | the gate's `unit` **is** `np.unique(parquet.priogrid_id)`, same order, ascending | `np.array_equal(gu, pg)` → **True** |
| ✅ | the draws are **[2333448, 16]**, **unit-major stride 36**, h1 = `[0::36]`, same unit order | `np.array_equal(u[t==457], np.unique(u))` → **True**; `np.diff(flatnonzero(sel))` → 36 |
| ✅ | the draws are in **count space** — h1 max **728.0001** against truth max 552 ⇒ **no `expm1`** | `np.load(y_pred.npy, mmap_mode='r')[0::36]`, then `.max()` |
| ✅ | **99.4569%** of h1 cells have all 16 draws exactly zero | `(h1.max(axis=1)==0).mean()` |
| ✅ | `G-0`'s anchor exists and is committed | `exp03_pairs.json`, record `{"seed":"fortytwo","h":1}` → `ap_ctl = 0.5667056384496834`, `n = 842634`, `n_org = 13` |
| ✅ | disk headroom | `df -h` → **38 GB free, 96% used** ⇒ stream one origin at a time; each `y_pred.npy` is 149 MB |
| ⬜ | `binned_cdf.self_test()` passes in this environment | #565 |
| ⬜ | `score_cells` mean == `binned_cdf_crps` to 1e-12 on fold 1 | #565 |

---

## E · Rules of engagement

- ⛔ **`D-5`:** every instrument is committed **before** it runs, with its gates in the module
  docstring under `⛔ THE GATES, declared before the run:`, printing `PASS`/`FAIL` and aborting
  `VOID:` on a load-bearing failure.
- ⛔ **A guard must be shown to FIRE**, not merely to pass. Each gate gets a deliberately broken
  input in a test.
- ⛔ **`D-4`:** the free parameter is swept and the **whole** sweep reported before any scalar is
  quoted. `C-393` records the failure this prevents.
- ⛔ **`C-395`:** `git add -f` with named files, never a directory. ⚠️ `__pycache__` already exists
  under `reports/2026-10-28_rp3_l3/` and `reports/2026-10-29_rp3_l2/` and must never be force-added.
- ⛔ **`C-347`:** `python <path>` puts the script's own directory on `sys.path[0]`, not the repo root,
  so `views_hydranet` can resolve to the installed wheel instead of the working tree. Insert the repo
  root explicitly and name the hazard in a comment.
- ⛔ Interpreter: `conda run --no-capture-output -n views-hydranet-env`. Without
  `--no-capture-output` a long job looks hung.
- ⛔ **Nothing is deleted** — not by a predicate, not at all. A `find … -exec rm -rf` destroyed 698
  directories earlier in this programme because the predicate matched the parent.
