# 03 — Harness & invariants (the crown jewel)

**Audited:** 2026-07-25, grounded in the actual code (file:line below), not templated. Headline: **~70% of
the harness already exists**; the free-running skill curve is a **GPU-free re-score of data on disk**. The
real build is a per-horizon loader + a teacher-forced-oracle flag.

## A. Invariant taxonomy

### Hard invariants — never break
- **Identical support** (the lodestar's cardinal rule; killed the "different-months" scar): every model +
  baseline scored on the *same* (origin, cell) set at each horizon, from the *same* truth parquet, *same*
  scoring functions. Extended here to the **horizon axis**: the scored (origin,cell) set must be fixed
  across h so the skill-vs-h curves are mutually comparable (G4).
- **h=1 reproduces the lodestar T=0 number** — the faithfulness sanity check; if the per-horizon loader's
  h=1 row ≠ the frozen ruler's T=0 row, the loader is wrong. STOP. (Mirrors the gated_ZINBcore self_zeroed
  sanity row that just caught/confirmed fidelity.)
- **Frozen scorer reuse** — `crps_ensemble`, `average_precision`, `brier` from
  `reports/2026-07-17_lodestar_eval_dossier/tools/lodestar_score.py` are used *verbatim*; the ruler adds a
  loader + a horizon loop + baselines, never a new metric implementation.
- **Determinism** (S2 #121): seeded `torch.Generator` for any re-inference (the oracle run); the ruler
  itself is pure re-score (deterministic).
- **Stealth protocol**: the violet_visitor floor config is never committed/pushed (md5
  `6c28bdb1390fc413d43b2d74d87251f8`); every re-inference driver trap-restores it.
- **Fail-loud** — no silent CPU fallback for the oracle re-run; NaN/Inf guards on the rollout.

### Deliberately changed by this program (behind a flag)
- **The AR feedback source.** Today the rollout feeds back the emit-mean (`fb = t1_pred`,
  `hydranet_inference.py:432`) — the bloom driver. This program adds, **behind a default-off flag**, a
  *teacher-forced* feedback source (feed back the realized `full_tensor[:, t, reg_indices]`) to measure the
  intrinsic predictability ceiling. Baseline (flag off) stays byte-identical.
- **What is scored.** Today only T=0 is scored (loader `sel = t == m0`, `lodestar_score.py:~92`). This
  program scores all persisted horizons. The T=0 number is unchanged (it is the h=1 row).

### Respect while changing
- The emit path / D×K sampler (ADR-067) — untouched; the ruler consumes its persisted output.
- The composition axis (ADR-069) — τ / gated arms are *feedback variants* we will score, not modify.
- The `feedback_clamp_log1p` rail — inert (C-216); leave as-is; not a lever here.

## B. Standing harness — what ALREADY EXISTS (verified)

| Mechanism | Status | Evidence |
|---|---|---|
| **Frozen scoring core** | ✅ reuse verbatim | `lodestar_score.py:26` `crps_ensemble`, `:36` `average_precision`, `:45` `brier` |
| **All 36 horizons persisted** | ✅ **verified on disk** | an `origin_*/…/identifiers.npz` has **36 distinct months** in `time`; `y_pred.npy` is `(36×n_cells, S)`. The lodestar loader (`:92` `sel = t == m0`) keeps only the first month and **discards h>1**. Free-running rollout at every horizon is already saved. |
| **Rollout carries realized future** | ✅ | `predict()` reads `full_tensor[:, t+1, reg_indices]` as step-truth (`hydranet_inference.py:411,478`) — the future is in the eval tensor. |
| **Climatology baseline** | ✅ reuse dir | white_ranger (`ConflictologyModel`, per-cell history resample) — its lodestar prediction dir exists; horizon-independent, so its T=0 samples reuse at every h. |
| **Determinism gate** | ✅ | S2 #121 seeded generator + regression test. |
| **Config trap-restore** | ✅ pattern | every re-inference driver verifies + restores the floor md5. |
| **Timestamped artifacts / bg + notify** | ✅ | eval writes `predictions_calibration_<ts>/origin_*`; drivers run background with completion notify. |
| **Negative-result discipline** | ✅ | `07_experiment_log` append-only; postmortems first-class (cf. gated_ZINBcore). |

## C. New harness this program needs (the real build — gaps)

- **G1 — per-horizon loader (`gather_all_horizons`).** Generalize `gather_t0`: instead of `sel = t == m0`,
  keep *every* month per origin and index by **horizon h = month − origin_month**. Returns
  `{(origin, h, unit): (count_samples[S], switch_prob)}`. Pure Python, no GPU. **TDD: h=1 slice must equal
  the current `gather_t0` output byte-for-byte** (faithfulness). *This gap alone unlocks the free-running
  skill curve from data already on disk.*
- **G2 — teacher-forced-oracle rollout.** One feedback-source switch in `predict()` (`:432`) behind a
  default-off config flag (`rollout_feedback ∈ {predicted (default), teacher_forced}`): when
  `teacher_forced`, `fb = full_tensor[:, t, reg_indices]` (the realized value) instead of `t1_pred`. Then a
  *small* re-inference to persist oracle `origin_*` dirs for the arms we care about. GPU, but bounded
  (eval-only, the arms already trained). Parity: flag off ⇒ byte-identical to today.
- **G3 — per-horizon baselines.** (a) climatology: reuse white_ranger samples at every h (horizon-flat);
  (b) persistence: `truth[origin]` broadcast to all h, scored vs `truth[origin+h]` — pure Python from the
  truth parquet. Both on the identical (origin,cell) support (G4).
- **G4 — identical-support origin set.** Pin the origin set to those with a full 36-month realized future in
  the truth parquet (origin_month + 36 ≤ max_truth_month). Fix the scored (origin,cell) set across all
  horizons (the most-constraining horizon defines it) so curves are comparable. Log the N dropped and why
  (no silent truncation).

## D. Pre-flight checklist (green before the FIRST scored read)

- [ ] G1 loader implemented + unit-tested; **h=1 == lodestar T=0 (byte-exact)** — *blocker*.
- [ ] G4 origin set pinned + logged (count, dropped-with-reason).
- [ ] G3 baselines (climatology reuse + persistence) constructed on the same support.
- [ ] The frozen scorer reused verbatim (no re-implemented metric).
- [ ] `05_analysis_plan` pre-registered (predictions + falsifiers vs the baselines) — *before looking*.
- [ ] (For the oracle read only) G2 flag implemented, default-off, parity-proven, full suite + lint green,
      determinism gate green, floor trap-restore in the driver.

## E. Rules of engagement

One variable at a time (feedback source is the variable; the scorer is frozen). Pre-register, then run.
**Cheap readout before expensive:** the free-running curve (GPU-free re-score) is the fast probe; the
oracle re-run is the expensive follow-up, run only after the free-running curve motivates it.
**Behavior-neutral by construction:** skill must come from honest scoring, never from clamps/caps hiding the
symptom. **Falsifier honesty:** a fired falsifier kills the hypothesis (see `05`).

## F. The two caveats carried as first-class framing

1. **STABILITY ≠ SKILL.** Bounded + sparse ≠ accurate. This ruler exists precisely to stop us conflating
   them. Every rollout-fix claim must cite a skill number from this ruler, not a trajectory-bound.
2. **The feature/world-model ceiling.** The model sees only conflict-history features, so long-horizon
   *skill* may be capped no matter the rollout method. The **teacher-forced-oracle curve measures that
   ceiling directly** — if the oracle also decays to the climatology baseline by some horizon h*, then h*
   is an intrinsic predictability limit and *no* feedback trick beats it past h*. This is a possible (and
   important) outcome, not a failure of the program.
