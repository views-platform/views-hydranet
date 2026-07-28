# 07 — Experiment log (append-only; negatives first-class)

Every run + outcome, each linked to its pre-registration (05 or a later `preregister`) and its
verdict-vs-falsifiers. No run yet — the pre-flight checklist (03 §D) is RED.

| date | experiment | pre-reg | one variable | result loc | verdict vs falsifiers | decision |
|------|-----------|---------|--------------|------------|-----------------------|----------|
| 2026-07-28 | **Tier-A parity (FRESH pull)** | 05 (LOCKED) | data provider viewser→datafactory (conflict targets) | job tmp `fresh_africa_me_legacy_121_504.parquet`, `parity_fresh.py`, `parity_residual.py` | **PASS — no falsifier fired** | trust the datafactory conflict data; proceed to queryset swap + P2 |
| 2026-07-28 | **E2 — Tier-A on the PLATFORM-fetched parquet (S3)** | 05 (LOCKED) | violet queryset swapped → pipeline-core `get_data` fetch | `tools/tier_a_parity.py`; violet `data/raw/calibration_datafactory_df.parquet` | **PASS** (identical to E1: exact 99.98/99.99/99.99%, maxima ✓, drift −0.35/+1.25/+0.06%, cell-set identical, coverage 121–504) | the platform datafactory dispatch produces correct data for violet; 2-lesson smoke trains green |

### E3 — S5 v2 foundation baseline: gated_NB ×3 seeds on datafactory truth (2026-07-28)
Arm **gated_NB** (nb family, soft_gate, 40 lessons) × seeds {42,43,44}, trained on datafactory data,
scored on the **v2 truth** (frozen lodestar functions, new truth). Results: `results/v2_baseline/`
(`SUMMARY.md` + `score_nb_{42,43,44}.txt`). **crps-all mean sb 0.144 / ns 0.084 / os 0.030**
(v1 ref 0.137/0.083/0.028; +4.8/+0.4/+9.4%), seed-stable. **size_ratio=0** (body timid), **tail dead**,
AP ~0.31/0.23/0.10 (occurrence intact). **Tier-B = PASS on reproduction + structure**; the
"beats white_ranger" leg is OPEN (no white_ranger cube on v2 truth — separate baseline run). Floor md5
restored; no OOM. **Decision:** v2 baseline established ⇒ future experiments extend from it; unblocks
S6–S8 (population) and S9 (frame-native). ⚠️ near-miss avoided: the "floor" (`hurdle_nb`) is the
discredited legacy path — trained gated_NB (nb family) instead per the scope-lock.

### E2 — S3 platform fetch + smoke (2026-07-28)
With violet's queryset swapped to the datafactory descriptor (ADR-071), pipeline-core `get_data`
fetched `calibration_datafactory_df.parquet` **fresh** (schema-identical to viewser: `lr_*_best, c_id,
row, col`; 384 mo × 13,110 cells). A 2-lesson calibration smoke (floor md5 `6c28bdb…` trap-restored)
trained green (rc=0) — datafactory data flows through DataSniffer→training. Tier-A on the fetched
parquet = **PASS**. This validates the swap end-to-end at the data layer.

### E1 — Tier-A parity on the FRESH pull (2026-07-28) — PASS
**Pull:** `load_dataset(region="africa_me_legacy", start=121, end=504, features=[ged_sb/ns/os_best, gaul0_code], data_dir=REMOTE.zarr_url)` — fresh remote, `last_valid_month_id=559`.
**Results vs the four pre-registered falsifiers (05):**
- **F-A1 (cell set differs)** — did NOT fire. Cell set **identical** (13,110; 0 only-vv / 0 only-fr); aligned frame index-identical.
- **F-A2 (unexplained residual)** — did NOT fire. sb residual = 0.0235% of cells; **small-count-dominated** (38% ±1, 65% ≤3, median 2), **temporally diffuse** (322/384 months, median 3/mo), country-concentration consistent with vintage revisions clustering in high-conflict countries (top-1 gaul0 23.5%). Attributable to L1(nokgi source-col) + L2(vintage) + L3(geocoding). No new difference class.
- **F-A3 (coverage <504)** — did NOT fire. Fresh reaches 559; pull covers full 121–504 (vs stale cache's 492).
- **F-A4 (maxima differ)** — did NOT fire. **Identical maxima** all three targets (sb 113395, ns 2000, os 178459).
- **Exact-match** 99.977/99.995/99.991%; **corr** 0.99995/0.98944/0.99998; **total drift** sb −0.346% / ns +1.246% / os +0.062%.

**Residual fully explained (user clarification 2026-07-28):** the reassignment part (L3) is the datafactory's **KGI (Known Geographical Imprecision) handling** — viewser's `_nokgi` column left KGI *unhandled* (legacy stub); the datafactory handles it. That is the intended improvement, not drift. The non-conserving part is L2 vintage revisions. Nothing unexplained.
**Caveat (accepted, not a blocker):** ns has the largest total drift (+1.25%) / lowest corr (0.989) — ns is the rarest target, so vintage revisions move it proportionally more; watch at re-baseline. *(user: noted + accepted.)*
**Decision:** Tier-A PASS ⇒ the datafactory `africa_me_legacy` conflict data is trustworthy. Advance to the violet queryset swap (P1) and v2 re-foundation (P2).

*(The earlier cached-pull diff was indicative prior-art, NOT this gate; it under-covered by 12 months — see `02_design.md`.)*
