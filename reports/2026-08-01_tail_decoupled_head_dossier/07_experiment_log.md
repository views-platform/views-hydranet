# 07 — Experiment log (append-only)

Every run + outcome, **including negatives/postmortems**, newest at the bottom. Each entry links its
pre-registration (`05` / a `NN_preanalysis`) and states its verdict against the pre-committed
falsifiers. Negatives get the fuller postmortem and are recorded as prominently as wins.

### EXP-01 · S4 single-tile overfit smoke (faithfulness gate) · 2026-08-01 · SUCCESS (GREEN)
- **Plan (pre-reg):** the S4 STOP-gate (`04_roadmap` M4 / issue #234) — can the mixture head fit ONE heavy cell with a **live component 2**, distinguishing "can't train it" from "no tail signal"?
- **Variable:** direct gradient fit of the 5 mixture params (isolating the head from the spatial pipeline) to 3 real heavy sb cells, vs a single-NB fit. Seeded, CPU. Driver: scratch `single_tile_overfit.py`.
- **Data note (verify-don't-trust):** `lr_sb_best` is **RAW count space** (integer, max **113,395** fatalities — a genuine ξ≈0.8 tail), NOT log1p; family target = `log1p(count)`. component-2 `μ2` init from the cell's own 90th-pct (data-informed init, legitimate for an overfit probe).
- **Readout:** every cell → **comp-2 LIVE** (`w<0.995`, `μ2≫μ1`) AND mixture **beats NB**:
  - cell 123511 (max 3005, 34 ev): `w=0.97 μ1=1.9 μ2=434`; NB 0.738 → MIX 0.732 (Δ0.007)
  - cell 181875 (max 1413, 101 ev): `w=0.28 μ1=323 μ2=934`; NB 2.378 → MIX 2.329 (Δ0.048)
  - cell 176832 (max 547, 95 ev): `w=0.97 μ1=2.2 μ2=80`; NB 1.411 → MIX 1.397 (Δ0.014)
- **Verdict vs gate:** GREEN — the head trains and uses component 2 on real heavy data ⇒ **not** a build failure; proceed to S5/S6. **Honest hint (NOT the verdict):** the gain over NB is *modest*, and *smallest* on the heaviest cell — a preview that the 2-NB light tail strains against ξ≈0.8, consistent with the pre-registered light-tail scope. The decisive answer awaits the stratified-GW verdict (S7).
- **Decision:** S4 STOP-gate cleared → proceed to S5 (pre-registration).

<!-- template (from rnd-dossier/references/templates.md):
### EXP-NN · <short title> · <date> · <SUCCESS|FALSIFIED|INCONCLUSIVE>
- **Plan (pre-reg):** <link>
- **Variable:** <the one thing changed>
- **Driver / artifact / results:** <script · artifact ts · results/log>
- **Readout:** <fast probe> → <full metrics vs the locked baseline>
- **Verdict vs falsifiers (plan §5):** <which fired / none> ⇒ <verdict>
- **Decision:** <next step per plan §7>
-->

### EXP-02 · S6/S7 mixture-vs-NB 6×300 + full falsifier scorecard · 2026-08-01→02 · **NULL** (within-family uncracked; heavy-tail UNtested)
- **Plan (pre-reg):** `05_analysis_plan.md` 🔒 (F1–F4; PRIMARY = ≥5% stratified-CRPS reduction on the ex-ante high-risk stratum, origin-block-bootstrap 95% CI excludes 0, ≥2/3 seeds).
- **Variable:** `output_distribution` ∈ {nb, mixture_nb} × seeds 42/43/44 × 300 lessons; else identical (soft_gate, sample-feedback, K=8, no-coords floor). **Retrain-both** (the v2 gated_NB config was uncommitted scratch, now lost — so both arms retrained under ONE reconstructed config for a provably clean 1-variable comparison). 6 trainings + 6 emits on the frozen v2 datafactory truth.
- **Driver / artifacts:** `scratch/marathon.sh` (setsid daemon) per arm: config trap-restore → TRAIN (proc1) → EMIT (fresh proc2). Cubes: nb {42:`…205045`, 43:`…052811`, 44:`…124321`}, mixture_nb {42:`…001300`, 43:`…091128`, 44:`…171855`}. Results: `results/gw_verdict.json`, `gw_results.json`, `f2f3_results.json`.
- **Execution scars (all recovered, no data lost):**
  - **nb-43 emit blocked mid-run** by a cross-repo skew: views-evaluation `native_evaluator._validate_config` (its fail-loud epic, 03:42) hard-rejects the legacy `'targets'` key that pipeline-core `configuration.py:543` still injected. Fixed upstream (pipeline-core PR #381, 06:52). **Verified moot for our verdict**: our ruler re-scores the raw `y_pred.npy` cubes with the frozen `crps_ensemble` and never consumes views-evaluation's metrics; cubes are byte-shape-identical old-vs-new-eval `(471960,64)`. nb-43 re-emitted from its saved artifact via `--artifact_name` (pipeline-core save-timestamp bug confirmed fixed; loaded weights verified ≠ latest mixture-44). views-frames pinned 1.8.0 across all 6 emits.
  - **Two latent scorer bugs** in `gw_stratified.py`, exposed only on the real parquet (unit tests used synthetic fixtures): `parents[2]`→`parents[3]` (repo-root path for the `lodestar_score` import) and `exante_stratum` now `reset_index()` first (the truth parquet keys `month_id`/`priogrid_id` in the **index**). Neither changes the scoring math.
- **Readout — PRIMARY (GW, sb h=1, ex-ante high-risk stratum, 13 origins, 6302 cells):** mixture beats NB **significantly on all 3 seeds** (CI excludes 0, p_boot ≤ 0.003) but every reduction is **sub-threshold**: seed42 2.14% / seed43 1.51% / seed44 3.45% (< 5%). Across horizons the edge is **h=1 only** — sb h=18/36 go slightly negative (−0.75%, −0.99%); ns/os the same shape.
- **Readout — guardrails:**
  - **F2 (size_ratio inflation):** does NOT fire. size_ratio moved *down* nb→mix (sb h1 s44 0.414→0.234), so the small win is **calibration/sharpening, not magnitude inflation** — the "heavy-tail" component did not make event magnitudes bigger.
  - **F3 (crps_none bloom):** does NOT fire. Mixture is *tighter* on true-zero cells at every horizon; leaked mass (`mcr_none`) sb h1 s44 0.088→0.046. Sample-feedback (ADR-070) held; no explosion re-armed.
  - **F4 (dead component-2 / w→1):** does NOT fire — **directly measured** (see the w|active probe, `scratch/w_probe.py`). The training forensic's `μ̄→1.0` was a **mislabel** (C-255: the param-health plotter hardcodes the ZINB μ/θ/π layout, so for the 5-param mixture row-3 "μ̄" is actually field-mean `w`, dominated by the ~99.3% zeros). On **active (truth>0)** cells, component-2 is alive and seed/origin-stable: median `w|act` 0.71–0.92, `min→0.003`, **½–⅔ of active cells have w<0.9**, 5–34% have w<0.5, median `1−w` 0.08–0.31, and **median μ2:μ1 ≈ 14×–690×**. Channel layout verified target-major (`out_reg=concat[reg1,reg2,reg3]`, slice `j*n_params`, activate `[w,μ1,θ1,μ2,θ2]`); alignment cross-checked — probe active-cell counts match the truth parquet exactly (sb 101/ns 47/os 69 @ month 469).
- **Verdict vs falsifiers:** PRIMARY **NULL** (significant but sub-5%, ≥5%-in-≥2/3 rule fails 0/3); F2/F3/F4 all **clean**. So this is not an inflation/bloom/collapse artifact — it is a genuine, well-behaved, **capacity-used** null.
- **Postmortem — the *meaningful* null ("active-but-insufficient"):** the mixture is not a dead second component reverting to NB — it **genuinely engages** a mean-decoupled tail (14×–690× larger μ2 on the heavy cells) and is well-behaved (no inflation, no bloom). Yet magnitude skill stays capped (sub-5%, h=1-only). ⇒ **mean-decoupling within the NB family is NOT the missing ingredient — tail SHAPE is.** A 2-NB mixture is asymptotically light-tailed and cannot reach ξ≈0.8 no matter how large μ2 grows. This **sharpens** the amount-ceiling story ([[project_amount_ceiling_wall]]): the wall is not "the model won't decouple the tail mean" (it will) — it is the exponential tail of the NB family itself. **The heavy-tail head (GPD/PIG) is now the single, pointed, sole-remaining magnitude lever** — and the pre-registered honest scope holds: this NULL does **not** prove the wall is real, it proves *within-family light-tail flexibility, even fully used, is insufficient*.
- **Decision (S8):** close the magnitude axis at the within-family boundary with the honest scope above. gated_NB remains the ship candidate; **mixture_nb NOT promoted** (no ADR); the family code stays in-tree, uncommitted. Byproducts registered: **C-255** (forensic mislabeling, Tier 2). Next lever if magnitude is ever reopened: a genuinely heavy-tailed head (its own epic).
