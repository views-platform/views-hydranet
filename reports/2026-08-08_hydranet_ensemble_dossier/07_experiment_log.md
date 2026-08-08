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

*(next entry: S5 member runs / S6 ensemble score, once 05 is LOCKED)*
