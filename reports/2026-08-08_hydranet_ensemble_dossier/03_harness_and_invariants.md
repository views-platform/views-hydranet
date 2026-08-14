# 03 — Harness & invariants (2026-08-08)

The crown jewel: what makes this ensemble program safe to run. **~80% already exists** — this is an
integration+config program, not a new-mechanism build.

## A. Invariant taxonomy
**1. Hard invariants (never break):**
- Fail-loud / no silent clamp; NaN/Inf guards in emit (`assert_cube_fits`, sampler finiteness).
- Reproducibility gate (seed/entropy lock; per-`(pass,step)` sub-generator seeding — ADR-070). *(bitwise GPU
  reproducibility NOT guaranteed — 3-seed spread is the uncertainty unit.)*
- **NEVER validate on stale/cached data** — a fresh datafactory pull or it's a finding (datafactory_migration
  03; the cache was 12 months short once).
- Evaluation comparability: the **frozen v2 truth parquet** + the frozen v2 horizon ruler primitives
  (`crps_ensemble`/`AP`/`Brier`) — identical months/cells/truth across every arm.
- Members must be mutually consistent (same partition, region, scale, sample count) or `concat` pools apples
  and oranges (F4).

**2. Deliberately changed by this program:**
- The 8 committed member configs move from **legacy heads** (tobit/shrinkage/hurdle_shrinkage) → the **v2
  `gated_NB` foundation** (nb/th_gated/mixture family heads). This is the intended replacement (S1/S3), not a
  regression.
- `rusty_bucket` (or a successor ensemble) is repointed from `temporary_*` stand-ins → the 8 real members (S4).
- 3 members move viewser → datafactory (S2).

**3. Respect while changing:**
- The ZINB bloom (sample-feedback ADR-070 stabilises the gated arms; ZINB self-zeroed still blooms — keep it
  OUT). F2 re-arms the bloom detector.
- The magnitude ceiling (ξ=0) — do not claim magnitude skill from the ensemble.
- The stable baseline configs stay byte-identical until deliberately reconfigured (transient-mutation discipline
  proven in the smoke).

## B. Standing harness — AUDITED (reuse, don't reinvent)
| Mechanism | Status | Location |
|---|---|---|
| Reproducibility (seed/entropy, sub-gen seeding) | ✅ exists | `infrastructure/reproducibility_gate.py`; ADR-070 |
| Config validators (family/composition/feedback fail-loud) | ✅ exists | `views_hydranet/utils/config_initializer.py` (K>1⇒family; gate_threshold iff threshold_gate) |
| D×K sampler + cube memory guard | ✅ exists | `hydranet_inference.py:generate_posterior_samples`; `disk_guard.assert_cube_fits` |
| **Locked baseline + GW readout** | ✅ exists | `reports/2026-07-29_v2_scoreboard_dossier/tools/{gw_stratified.py,score_v2_horizons.py}`; `light_strider` climatology row |
| **Frozen v2 truth** | ✅ exists | `reports/2026-07-28_datafactory_migration_dossier/tools/v2_truth/calibration_datafactory_df.parquet` |
| Ensemble concat pooling + contract guards | ✅ exists | pipeline-core `PredictionFrameEnsembleManager` (`_aggregate_prediction_frames` sample-axis concat; `_assert_expected_sample_count`; C-85 stale-cache peek-guard) |
| ADR-015 sample-count contract | ✅ exists | `tests/test_ensemble_configs.py` (`expected_models`/`expected_samples_per_model`) |
| **Transient config-mutation driver + D×K-cube verify** | ✅ built (smoke) | `scratchpad/{smoke_run.sh,smoke_mutate.py,check_cube.py}` — per-model md5 floor trap-restore, disk preflight, clear-predictions (C-85), setsid, manifest+sentinel |
| Fresh-pull discipline + Tier-A parity | ✅ exists | `reports/2026-07-28_.../tools/tier_a_parity.py`; ADR-071 |
| Run discipline (one-heavy-job, setsid, notify, trap-restore) | ✅ proven | 2026-08-04 smoke (14 trains autonomous, floors restored, nothing committed) |

## C. New harness this program needs (gaps to build — the real S1–S4 work)
- **S1:** the committed `gated_NB` foundation config (the lost v2 scratch), 2-lesson smoke-verified.
- **S2:** 3 viewser→datafactory queryset swaps + `views-datafactory` req + **Tier-A PASS on a fresh pull** each.
- **S3:** the 8 member configs set to the pre-registered roster (family, composition, seed, `S`) on the v2
  foundation; per-member 2-lesson smoke.
- **S4:** repoint an 8-member `concat` ensemble at the real dirs + **reconcile the D×K-vs-`n_posterior_samples`
  contract wrinkle** — the config-time `test_ensemble_configs` reads `n_posterior_samples` (=D), but the
  runtime produces **D×K**; `expected_samples_per_model` must equal the produced D×K, and the CI contract must
  be made consistent (the smoke sidestepped this by not running that CI test).
- **F3 memory:** `rusty_bucket` OOM'd ~28.6 GB at 8×128 — pick `S` so 8×`S` pools within the RTX 4070; the
  smoke's 7×16=112 pooled fine.

## D. Pre-flight checklist (must be GREEN before S5, the 300-lesson run)
- [ ] Roster + `S` LOCKED in `05` (needs-decision resolved) — **blocker**
- [ ] `gated_NB` foundation config committed + 2-lesson smoke green (S1)
- [ ] 3 viewser models on datafactory; **Tier-A PASS on a fresh pull** each (S2)
- [ ] 8 member configs match the locked roster; per-member smoke green; `config_initializer` validators pass (S3)
- [ ] 8-member `concat` ensemble points at real dirs; sample-count contract reconciled + green; pools ≤ hardware (S4)
- [ ] GW scorer + frozen truth wired for ensemble-vs-member + ensemble-vs-`light_strider`
- [ ] F1–F4 pre-registered (05 LOCKED); honest ξ=0 scope recorded
- [ ] full suite + lint green in both repos; nothing stale on disk

**`status` must refuse "ready to run" until this is green.** The plumbing smoke (2026-08-04) already exercised
the mechanics (config-flip→family-head→D×K→composition→emit→concat-pool) end-to-end — the remaining work is the
real *config* (S1–S4), not new *mechanism*.

## E. Rules of engagement
One variable at a time; pre-register then run; cheap readout (2-lesson smoke) before the 300-lesson run;
falsifier honesty (a fired F-gate kills the claim — no ad-hoc rescue); improvements from representation, not
masking; **STOP-gates:** S2 Tier-A must PASS on a fresh pull per model before S3 reconfigures it; S6 must not
fire the F2 bloom before S7 ships.
