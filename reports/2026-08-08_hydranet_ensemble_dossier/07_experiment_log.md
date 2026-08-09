# 07 — Experiment log (append-only)

Every run + outcome, **including negatives/postmortems**. Each entry links its pre-registration (05 or a
per-experiment prereg) and its verdict vs the pre-committed falsifiers. No success-only drift.

---

## EXP-00 — Plumbing smoke (pre-S0 de-risk) · 2026-08-04
- **Pre-registration:** none (mechanics de-risk, not a scientific claim; no metric asserted).
- **One variable:** n/a — end-to-end plumbing exercise.
- **Setup:** 14 trains (7 dirs × 2 seeds), roster-mix families (gated_NB/th_gated/mixture), 40 lessons,
  D×K=4×4=16, transient config-mutation (trap-restored), `scratchpad/smoke_run.sh` setsid harness; then the
  7-member `concat` pool via `rusty_bucket` repointed (transient).
- **Readout:** 14/14 trains emitted a **(N,16)** finite cube; ensemble pooled **(471960, 112)** finite = 7×16;
  all floors + rusty_bucket restored (git clean); nothing committed; disk clean.
- **Verdict vs falsifiers:** n/a (no F-gates — plumbing only). **PLUMBING PASS.** Covers all 3 family heads +
  both data providers (datafactory + viewser) + concat pooling + contract guards.
- **Decision:** the S1→S5 mechanics are sound; proceed to S0 pre-registration. Deltas for the real run:
  heavy_freighter (global grid) untested in the smoke; the D×K-vs-`n_posterior_samples` contract wrinkle to
  reconcile at S4.

---

## EXP-01 — S2 datafactory freshness / Tier-A (transitivity path) · 2026-08-08
- **Pre-registration:** F1–F4 (05); Tier-A falsifiers F-A1..F-A4 (migration dossier 05). Rigor path chosen:
  **fresh datafactory pull + transitivity** (the 3 viewser models' queryset is byte-identical to violet's
  pre-migration queryset → same dataset violet already Tier-A-PASSed vs viewser 2026-07-28).
- **One variable:** the datafactory source freshness (is it still live + stable since 2026-07-28?).
- **Setup:** FRESH datafactory pull (`africa_me_legacy`, months 121–504, remote `last_valid=559`, NOT cached) vs
  the frozen v2 truth; Tier-A scorecard (`tools/s2_df_freshness_check.py` reusing `tier_a_parity.py`).
- **Readout:** cell-set identical (0 diff), index identical, coverage 121–504 (384 mo), **exact-match 100.000%**
  on lr_sb/ns/os_best (corr 1.0, maxima identical, 0.000% drift, 0 mismatches).
- **Verdict vs falsifiers:** F-A1..F-A4 all **quiet** → **Tier-A PASS**; the datafactory source is STABLE (no
  vintage drift since the v2 freeze). Combined with violet's identical-queryset Tier-A PASS (df-vs-viewser),
  **transitivity holds** → migrating the 3 models preserves the viewser conflict truth.
- **Decision:** S2 validated. The migration itself = copy violet's model-name-agnostic datafactory
  `config_queryset.py` verbatim to pink_pirate/blue_stranger/purple_alien + add `views-datafactory>=1.9.0,<2.0.0`
  to each `requirements.txt`. Applying + committing to views-models is the next action (user-gated).

---

## EXP-02 — S5 africa ensemble run LAUNCHED · 2026-08-09 04:07
- **Pre-registration:** 05 (LOCKED + 2026-08-09 amendment: 160 lessons, window-constrained).
- **One variable:** ensembling (8 roster members `concat`-pooled) vs each member.
- **Setup:** 8 roster members (gated_NB 42/43/44, th_gated_NB 45/46, mixture_NB 42/43/44), **160 lessons**,
  D×K=4×4=16, all `africa_me_legacy` datafactory (violet migrated; heavy_freighter global→africa, global config
  banked at `tools/heavy_freighter_global_config.py`). Driver `scratchpad/s5_run.sh` (setsid, manifest-resumable,
  disk-preflight): train+emit each (keep the 16-cube) → pool the 8 into rusty_bucket (expect 128 draws).
  Configs live in worktree `_s3_worktree` (uncommitted; S3 PR pending). `/falsify` pre-launch caught the missing
  driver + a 2.5× time-estimate error before this run. ETA ~19h (fits the 25–26h compute window).
- **Readout / verdict:** **RUN COMPLETE 2026-08-09 19:21** (`pool_rc=0 poolok=1`). 8/8 members trained (160L,
  ~1.1h each) + emitted 16-cubes; ensemble pooled **(471960, 128)** finite = 8×16 draws. No failures.
  **Incident:** an unattended-upgrades reboot ~5.5h into the FIRST launch (04:07) wiped it (0 output, /tmp
  cleared); relaunched 09:46, machine stable thereafter, completed in ~9.6h. Artifacts + cubes live in the
  **worktree** `_s3_worktree` (models/*/data/generated + ensembles/rusty_bucket/data/generated;
  gitignored, survive on /home). **Plumbing verdict: PASS** (the full 8→128 ensemble runs end-to-end at africa
  scale). **Skill verdict (F1–F4) PENDING** — needs S6 scoring (GW vs best member + vs `light_strider`), which
  needs the S4 D×K-vs-`n_posterior_samples` reconciliation. That is the next session's first job.

---

## EXP-03 — S6 scoring: ensemble vs its 8 members (v2 ruler) · 2026-08-09
- **Pre-registration:** 05 (F1–F4). Scores the EXP-02 cubes on the frozen v2 datafactory truth via
  `score_v2_horizons.score_horizons_v2` (crps_all/AP/crps_none @ h=1/18/36, per target). The scorer reads the
  cube's real sample axis, so the D×K-vs-`n_posterior_samples` wrinkle does NOT affect scoring (it was only a
  config-time CI concern). Registry: ENSEMBLE (128 draws, `frac(samples>0)` occurrence — no gate channel) + 8
  members (16 draws, gate-head occurrence). Common support intersected across all 9 arms. Tool
  `scratchpad/score_ensemble.py`; results `_s3_worktree/ensemble_scoreboard.csv`.
- **⚠️ SCOPE CAVEATS:** 160 lessons (amended, window-constrained) — NOT the 300-lesson production numbers;
  single seed per member (no per-member seed spread); no origin-bootstrap / GW significance test yet;
  `light_strider` climatology not scored (baseline cubes absent).

### Readout (crps_all, lower=better; ensemble vs BEST single member)
| target | ENS h1 | best-mem h1 | ENS h18 | ENS h36 | note |
|---|---|---|---|---|---|
| sb | 0.1348 | **0.1299** (violet) | 0.1342 | 0.8746 | ens 2nd of 9; converges by h18 |
| ns | 0.0813 | **0.0798** (violet) | 0.0375 | 0.0273 | same shape |
| os | 0.0274 | **0.0272** (violet) | 0.0336 | 0.0672 | same shape |

- **crps_all — NULL (robustness, not a win).** The ensemble does NOT beat its single best member (violet, seed
  42) on crps_all at any horizon; it lands **robustly near-best (2nd of 9)**, beating 7/8 members, tied at long
  horizon. Textbook equal-weight posterior pooling: buys robustness (reliably near-best without picking the
  winner a priori), not a new frontier. **Consistent with the honest ξ=0 scope.** *(Pre-registered P1 "ensemble
  ≤ best member @ h1" — FALSIFIED: ensemble is marginally worse than the best member.)*
- **crps_none (bloom) — ✅ CLEAN, a real ensemble benefit.** ENSEMBLE has the **lowest leaked mass of all 9 arms**
  (sb h1 0.0016 < every member; ns/os likewise). Pooling dilutes any single member's off-support mass. **F2
  does NOT fire** — no bloom.
- **AP (occurrence) — ⚠️ THE ACTIONABLE FINDING.** ENSEMBLE AP is **materially worse than every member** (sb h1
  0.316 vs members 0.38–0.47; far worse at h18/36). **Root cause: the `concat` pool carried the magnitude
  samples but DROPPED THE GATE** — the pooled cube has `lr_` only, **no `by_` occurrence channel** (verified:
  the ensemble dir has no `by_*` sub-dirs). So the ensemble has no calibrated gate; its occurrence could only be
  scored as `frac(pooled samples>0)`, a far coarser ranker than the members' gate-heads. **As currently wired,
  the ensemble throws away the members' calibrated occurrence gates** — sacrificing "occurrence calibration,"
  one of its headline pre-registered benefits. This is a **plumbing/design gap (S4)**, not a model failure.

### Verdict vs pre-registered falsifiers
- **F1** (ensemble adds nothing over best member on crps_all): the crps_all *lift* premise **fails** — no win over
  the best member. But the honest-scope value (robustness + bloom-suppression) holds; not a hard kill.
- **F2** (bloom re-armed): **does NOT fire** — ensemble is the cleanest on crps_none.
- **F3** (OOM at 8×S): **did NOT fire** — 8×16=128 pooled fine.
- **F4** (member inconsistency): **did NOT fire** — all 8 shared support, scored cleanly.

### Decision
The plumbing works and the ensemble is robust + bloom-free, but (a) it does not beat its best member on
crps_all, and (b) **the pooling drops the occurrence gate**, crippling AP. Before any ship/no-ship call, **chase
the gate-pooling gap** — it determines whether the ensemble's occurrence story is recoverable. GW significance +
`light_strider` scoring deferred until the gate question is settled (an ungated ensemble's AP isn't worth a
formal test). Members' sb-h1 crps_all ~0.13 already beats the v2 climatology reference ~0.19 (indicative, P2).

### UPDATE 2026-08-10 — gate-pooling gap CONFIRMED as the AP cause, and FIXED
- **Root cause (code-confirmed):** `PredictionFrameEnsembleManager._build_context` sets
  `ctx.targets = c.get("targets", c.get("regression_targets", []))` (prediction_frame_ensemble.py:373), and the
  HydraNet ensembles (rusty_bucket, golden_hour, stellar_horizon) declare **only `regression_targets`** (the 3
  `lr_`) — **no classification targets**. The pool loops `for target in ctx.targets`, so it concats the 3
  magnitude channels and **never touches `by_`**. `_aggregate` is generic (`np.concatenate(axis=1)`), so the
  framework CAN pool `by_` (it's per-sample `(N,16)`, same shape) — it simply is never asked to.
- **Fix + test (no retrain):** added a `"targets"` list (`lr_*` + `by_*`) to rusty_bucket's meta and re-pooled
  the cached member cubes → the ensemble now emits a pooled gate `by_sb_best (471960,128)`. Re-scored with the
  gate:

  | target | AP ungated h1 | **AP gated h1** | best-member h1 |
  |---|---|---|---|
  | sb | 0.316 | **0.456** | 0.474 |
  | ns | 0.177 | **0.355** | 0.404 |
  | os | 0.135 | **0.225** | 0.267 |

  AP recovered from crippled → **near-best** at h1 (2nd of 9, like crps_all). The collapse was 100% the dropped
  channel; **zero code change** needed.
- **Corrected verdict:** the ensemble is robustly **near-best on BOTH crps_all AND AP** (doesn't beat the
  standout member, violet s42) with the **cleanest bloom** — the honest equal-weight-pooling result (robustness,
  not a frontier). ξ=0 scope holds; no overclaim.
- **Two follow-ups:** (1) the gate-pooling gap is a **silent-correctness bug in the ensemble framework** — every
  HydraNet ensemble's occurrence (AP) is understated until the gate is pooled by default → **registered** (see
  risk register). (2) violet s42 dominates both axes but is **single-seed** — "genuinely best vs seed luck" is
  undecidable without the multi-seed run.

---

*(next: fix the gate-pooling framework bug properly; multi-seed decision; then GW significance + light_strider)*
