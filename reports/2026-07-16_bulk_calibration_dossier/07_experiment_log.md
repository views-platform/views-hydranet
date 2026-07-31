# 07 — Experiment Log (append-only; negatives first-class)

Every entry links its pre-registration (`05` or a `NN_preanalysis_*`) and states the verdict vs the
pre-committed falsifiers. Negatives get a full postmortem, recorded as prominently as a win. Newest at
bottom.

### P0 · Bulk-calibration metric built + baseline anchor · 2026-07-16 · CONTEXT (+ a surfaced finding)
- **Plan:** `05` / metric `03 §D`. Tool `tools/bulk_score.py` (T=0-only, positives-only, bulk-only,
  per-cell `ratio_med`, cut 97/98/99 from positive TRAINING truths). Retrain-free; ran on the dense-mse
  pw2 predictions `predictions_calibration_20260716_103054`.
- **Metric validated:** guardrails reproduce t0_score (sb CRPS 0.140, Brier 0.0119); cuts sb 211/300/545.
- **⭐ ANCHOR + FINDING — the baseline body is DEAD, not timid:** bulk `ratio_med = 0.000` on ALL targets,
  ALL cuts. Verified: **97% of positive cells get E[y] EXACTLY 0** (frac>0.1 = 1.7%, frac>1 = 0.2%); the
  0.043 among the 23 alive cells. So dense-mse predicts literally nothing on ~all positive cells, and its
  CRPS "win" over white_ranger is **entirely the dead-body-wins-the-99.7%-zeros artifact** — the metric
  exposed what CRPS hid.
- **Cause (2 stacked):** (1) **dead-ReLU** — output_distribution='standard' ⇒ reg_activation=ReLU
  (C-178 / [[project_body_loss_not_the_lever]]); pre-activation drifts negative ⇒ ReLU≡0, zero gradient,
  unrecoverable ⇒ *exactly* 0. (2) **all-cell MSE** pulls the body toward the ~0 mean of a 99.7%-zero
  target.
- **Implication for the plan (refinement, not a silent change):** a DEAD body cannot be "lifted" by any
  dial — there is no gradient, and all-cell MSE fights us. The mechanism needs a **revival step first**:
  (a) `reg_activation='softplus'` (alive gradient, C-178 fix) and (b) train the body on POSITIVE/active
  cells (hurdle-compose with the frozen gate) so the zeros don't drag it to 0 — THEN the winsorize+dial
  calibrates the revived positive body. This makes the real baseline "dense-mse softplus, positives-only"
  and reframes A0. Owed to the user before amending `05`.
- **CONFIRMED (not a regression, not circular):** our run's saved config = `reg_activation=relu`. `standard`
  DELIBERATELY defaults to ReLU ("byte-identical to pre-#100"); softplus is the default only for `hurdle*`.
  The T=0 screen's alive dense-mse explicitly set `reg_activation='softplus'` — we omitted it. Fix = the
  1-line flag. No mystery.
- **Decision:** `05` AMENDED (2026-07-16 changelog): A0 dead-ReLU anchor · **A0′ softplus revive** ·
  A1 +winsorize · A2 +τ-dial; honest CRPS bar = white_ranger. → P1 (implement winsorize + τ-pinball; revive
  is config-only). Metric + anchor LOCKED and correct.

---

### P1 · Mechanism implemented (TDD, default-off) · 2026-07-16 · DONE
- **New:** `views_hydranet/utils/pinball_body_loss.py` — `PinballBodyLoss(tau, cap)`: winsorize target at
  `cap` (log1p) + asymmetric pinball at `tau` (the dial: 0.5=median, >0.5 lifts). Registered in
  `LOSS_REG_REGISTRY["pinball"]` (OCP; params `loss_reg_tau`/`loss_reg_cap`); config fields added
  (`loss_reg_tau` ∈(0,1), `loss_reg_cap` >0). Default-OFF (only built when `loss_reg='pinball'` ⇒ baseline
  byte-identical).
- **Tests (`tests/test_pinball_body_loss.py`, 6/6 green):** minimiser = τ-quantile of the WINSORIZED target;
  higher τ fits a higher magnitude (the dial lifts); cap neutralises an extreme; finite gradient; τ=0.5 ⇒
  MAE; invalid τ rejected; config accepts `pinball`. Regression: `test_config_validation` 37 green (no
  breakage). Lint: 11 E501 (cosmetic, ship-it debt). Code UNCOMMITTED (standing rule); dossier git-tracked.
- **Run config (A-arms, for P2/P3):** `output_distribution='hurdle_shrinkage'` (gate × point body),
  `reg_activation='softplus'` (revive), `hurdle_threshold=0` (body on positives), frozen gate
  (`loss_class='weighted_bce'`, `pos_weight=2`). A0′ `loss_reg='mae'` (revived timid baseline); A2
  `loss_reg='pinball'`, `loss_reg_cap≈5.7` (=log1p(300)≈sb 98th pct), `loss_reg_tau` swept {0.5..0.8}.
- **Pre-flight (`03 §E`):** metric ✅ · loss+tests ✅ · OCP+default-off ✅ · pre-registered ✅ (05 amended)
  · gate frozen/tail parked ✅. **GREEN → ready for P2 (2-lesson smoke, GPU).**

---

### P2 · 2-lesson GPU smoke (A2 mechanism) · 2026-07-16 · ✅ PASS
- **Config (A2, smoke):** `output_distribution='hurdle_shrinkage'` (gate × point body,
  `hurdle_point_expected_log1p`), `reg_activation='softplus'` (revive), `hurdle_threshold=0`
  (per_step positives mask ⇒ body on positive cells only), `loss_reg='pinball'` `tau=0.7` `cap=5.7`,
  `loss_class='weighted_bce'` `pos_weight=2`, seed 42, `total_lessons=2`. STEALTH: violet_visitor config
  patched in place (floor md5 `6c28bdb1…`), trap-restore driver `scratchpad/run_smoke_A2.sh`,
  `python main.py -r calibration -t` (train-only — fastest mechanism check).
- **Gate (what "pass" means):** trains without crash · reg loss finite (no NaN/inf) · the winsorized
  τ-pinball is the actual body criterion (dial active) · artifact + sidecar persist
  `output_distribution=hurdle_shrinkage`, `reg_activation=softplus`.
- **✅ VERDICT — PASS (no falsifier fired).** Both lessons trained (`Done training`, exit 0, ~1.5 min GPU);
  NO NaN/inf anywhere; `PinballBodyLoss(tau=0.7, cap=5.7, needs_latent=False)` confirmed the built
  criterion (import check: finite loss + grad on a positive log1p batch); sidecar
  `calibration_model_20260716_213743.pt.config.json` = `{output_distribution: hurdle_shrinkage,
  reg_activation: softplus}`. Floor config restored (md5 `6c28bdb1…` OK). Mechanism is live and stable.
- **⚠️ INFRA FINDING (surfaced, not caused by us) — #144 grid-flip is live; the floor config is STALE.**
  First launch died in `data_sniffer._check_obligatory_columns` (`Missing: ['priogrid_gid']`) — the current
  calibration parquet + a fresh viewser fetch both carry **`priogrid_id`**, not the legacy `priogrid_gid`.
  The #144 fix (`1f707d3`) made `data_fetcher`/`mcr_readout` grid-name-agnostic but did NOT wire the fix into
  `data_sniffer` (still reads the hardcoded config `identity_cols`). So ANY run from the on-disk floor
  (`config_hyperparameters.py`, priogrid_gid) now fails ingestion. Smoke fix: set `id_col`/`identity_cols`/
  `index_names` → `priogrid_id` (a pure column RENAME — same PRIO-GRID cells/values, so the eventual A/B is
  unaffected; all A-arms run on `priogrid_id`). **Register-worthy: `data_sniffer` obligatory-columns check is
  not grid-name-agnostic (#144 gap).**

---

### P3 · First-seed A/B batch (winsorize + τ-dial ladder) · 2026-07-16 · 🛑 F2 — REJECTED
- **Plan:** `05` §4 + the 2026-07-16 P3-realization changelog (recorded before launch). Full τ-sweep
  (user-approved, ~3.5h). Seed 42, 40 lessons, `hurdle_shrinkage`+`softplus`+`hurdle_threshold=0` (positives
  body), `weighted_bce` `pw2` gate, `priogrid_id`. train+eval → T=0 predictions for `tools/bulk_score.py`.
- **Ladder (single-variable):** A0p `pinball τ0.5 no-cap` (revived timid baseline) → A1 `τ0.5 cap5.7`
  (+winsorize only, F3 control) → A2t6/A2t7/A2t8 `τ{0.6,0.7,0.8} cap5.7` (+dial). A0 (all-cell dead-ReLU,
  `ratio_med` 0.000) = banked P0 anchor (not re-run — pure #144 rename).
- **Pre-flight:** all 5 configs pydantic-validate; 43 tests green (pinball + config). **Source fix:** pinball
  registry `params` → `["loss_reg_tau"]` (cap now OPTIONAL — uncapped τ-pinball is valid; A0p needs it).
  Driver `scratchpad/run_p3_batch.sh` (stealth patch-per-arm, trap-restore, arm→predictions manifest).
  Disk 64 GB free (>10 guard + ~7 GB batch). Started 21:47.
- **Readout (pending):** bulk `ratio_med` per arm (primary, band [0.7,1.3]) + Δ vs A0 (bootstrap CI) +
  guardrails Brier/CRPS/QS99 vs white_ranger. Falsifiers F1 / F3 / F4 / F2 all live.

- **✅ COMPLETE (5/5 arms exit 0, floor restored md5 OK). VERDICT — F2 FIRES: hypothesis REJECTED.**
  Headline `cut98 ratio_med` / T=0 CRPS (white_ranger bar sb .276 / ns .108 / os .039):

  | arm (τ) | sb rm / CRPS | ns rm / CRPS | os rm / CRPS |
  |---|---|---|---|
  | A0p .5 no-cap | .131 / .285 | .105 / .757 | .035 / 1.73 |
  | A1 .5 +cap    | .136 / .297 | .117 / .964 | .036 / 1.95 |
  | A2 .6         | .200 / .432 | .331 / 2.77 | .058 / 2.01 |
  | A2 .7         | .325 / 1.14 | **.767 / 7.53** | .070 / 1.47 |
  | A2 .8         | **.780 / 5.14** | **1.468 / 19.3** | .123 / 2.50 |

  **What the falsifiers say:**
  - **Mechanism CONFIRMED, attribution CLEAN.** `ratio_med` rises monotonically with τ on all 3 targets ⇒
    the dial lifts the bulk. **F3 does NOT fire:** A1 (cap alone) stays timid (sb .136 vs A0p .131; ns .117
    vs .105; os .036 vs .035 — all Δ<0.02, well under the 0.3 confound bar) ⇒ the τ-dial, NOT the cap, is
    the lever. The winsorize is inert on its own, as predicted.
  - **F1 partial:** the dial lifts sb (.78@τ.8) and ns (.77@τ.7, 1.47@τ.8) INTO/PAST the [0.7,1.3] band, but
    **os never lifts** (max .123 @τ.8) — os stays timid at every τ (the rarest / amount-ceiling-worst target).
  - **🛑 F2 FIRES HARD — the lift is degenerate.** Wherever the dial reaches the band it **destroys the
    guardrails**: sb τ.8 CRPS **5.14 = 18× white_ranger** (QS99 1.03 vs .11); ns τ.7 CRPS **7.53 = 70×**;
    ns τ.8 CRPS 19.3. The cap does NOT rescue it (it winsorizes the TARGET; the blow-up is in the PREDICTION).
  - **Decisive diagnostic — the dial rescales, it does NOT calibrate.** As τ rises, `within2x_rescaled`
    is FLAT-to-worse (sb 16.8→14.9) and `spearman` is FLAT (sb .246→.253). The lift comes entirely from the
    ratio-spread exploding (sb p90 4.1→52.9; **ns p90 30.7→566.8**): the dial over-fires a subset of cells by
    50–500×. It moves the median without making any cell's prediction better — because the body still can't
    *rank/size* individual cells (the amount-ceiling WALL, [[project_amount_ceiling_wall]]).
- **POSTMORTEM (negative, first-class).** H ("a winsorized τ-pinball is a clean bulk-magnitude knob analogous
  to the gate's pos_weight") is **FALSE**. A *global* point-loss dial cannot calibrate a *heterogeneous* bulk:
  cells have wildly different true magnitudes, one τ lifts them all, so the cells it can't identify get
  over-fired → spread/CRPS/QS99 explode. The gate's pos_weight works because occurrence is ~homogeneous
  (a probability); magnitude is not. The winsorize was necessary-but-insufficient (caps target-driven, not
  prediction-driven, explosion). This is the *same* failure family as count_mean
  ([[project_count_mean_fails_oos]]) — lift-the-magnitude → blow-up — and it **reinforces the volatility
  program's conclusion** ([[project_volatility_ceiling_predictable]]): the bulk needs **per-cell conditional
  magnitude/spread** (a distributional head), NOT a global scalar dial. Best guardrail-safe operating point
  is the timid baseline (A0p/A1, `ratio_med` ~0.13, CRPS ≈ white_ranger for sb only). Pre-registered F2
  fallback = "back off τ / tighten cap" → τ-backoff just returns to timid; a tighter cap won't fix
  prediction-side over-fire (argued). ⇒ redirect per 05 §7 toward the **mixture / per-cell distributional**
  path. Tools/predictions banked (5 arms, `scratchpad/score_*.txt`, manifest `p3_manifest.txt`).
