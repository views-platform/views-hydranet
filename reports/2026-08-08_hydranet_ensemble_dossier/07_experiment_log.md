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

*(next: S5 verdict once the sentinel lands → S6 scoring)*
