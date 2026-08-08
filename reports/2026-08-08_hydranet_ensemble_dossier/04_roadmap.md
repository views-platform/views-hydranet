# 04 — Roadmap (2026-08-08)

Phased, gated sequencing for Epic #242. Stories live in views-hydranet (#243–251).

```
S0 (#243 pre-reg) ┐
S1 (#244 found.)  ┤
S1.5 (#255 hygiene ✅ DONE) │
S2 (#245 migrate) ┼─→ S3 (#246 roster) ─→ S4 (#247 wire) ─→ S5 (#248 run) ─→ S6 (#249 score) ─→ S7 (#250 disp.)
```

| Story | What | Gate / dependency | Status |
|---|---|---|---|
| **S0** #243 | dossier + **LOCK** pre-registration (roster, `S`, scoring, F1–F4, honest scope) | blocks S3; S6 binds to F1–F4 | **in progress** (scaffolded; LOCK pends roster/`S` decision) |
| **S1** #244 | reconstruct + bank the `gated_NB` foundation config; 2-lesson smoke | blocks S3 | next |
| **S1.5** #255 | fleet config hygiene + parity unblock | — | ✅ **DONE** (PR #335) |
| **S2** #245 | migrate 3 viewser → datafactory + **Tier-A PASS on a fresh pull** | **STOP-gate**: Tier-A per model before S3 | pending |
| **S3** #246 | 8 members → roster on v2 foundation; per-member smoke | needs S0+S1+S2 | pending |
| **S4** #247 | wire 8-member `concat` ensemble; **reconcile D×K-vs-`n_posterior_samples` contract**; K/memory standardize | needs S3 | pending |
| **S5** #248 | run 8 members × 300 lessons + pool (GPU, setsid harness) | needs S3+S4 | pending (GPU) |
| **S6** #249 | score ensemble vs best member vs `light_strider` (GW + v2 ruler); verdict vs F1–F4 | **STOP-gate**: F2 bloom must not fire before S7 | pending |
| **S7** #250 | disposition + ADRs + close #146/#203; `promote` dossier | needs S6 | pending |

## Milestones / decision points
- **M0 (now):** S0 LOCK — the roster + `S` decision. **Blocks everything downstream.**
- **M1:** foundation banked (S1) + fleet on datafactory (S2 Tier-A green) → the fleet is roster-ready.
- **M2:** 8-member ensemble runs end-to-end at `S` within hardware (S4 pre-flight) → cleared for the 300-lesson run.
- **M3:** S6 verdict vs F1–F4 → ship ensemble / ship best member / quarantine + re-pool / re-pick `S`.

## Cheap-readout discipline
Every config change gets a **2-lesson smoke** (the proven `scratchpad/smoke_run.sh` harness) before the
300-lesson run — never burn a long job to learn what seconds tell us. The 2026-08-04 plumbing smoke already
cleared the mechanics.
