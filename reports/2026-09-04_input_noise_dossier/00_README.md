# Input-noise: make the training inputs look like what the model feeds itself

**Opened** 2026-09-04 · **Branch** `exp/silence-vs-fade` · **Status** S5/S6 COMPLETE — **Δ AP@h18 = −0.1963, the noise arm is much worse** (M64) ·
**Epic #311** · **Tracking #320** · **This story: #312**

## Purpose

Five attempts to close the training/deployment gap have failed. The fifth, BPTT-SA (#308), closed
today with `AP@h18` **0.3064 → 0.1997** and firing **×3.0** (M62/M63). This programme tries the one
thing in that family we have never tried: **corrupt the training inputs so the model stops depending
on them being perfect** — and, because the pushforward loss is already built and has never been run,
carries it as a same-family comparator at zero implementation cost.

| inherited finding | what it constrains here |
|---|---|
| **M45** — AP loss scales with how much the model FIRES | any augmentation that increases firing is expected to cost AP; the noise must not manufacture occurrence |
| **M50** — active fraction **0.000612**; fed magnitude goes *negative* in log1p space | the data is ~99.94% zero and the space has a floor; naive additive noise breaks both |
| **M62/M63** — BPTT-SA hurt, and the estimator was *not* the reason | the objective, not the approximation, was wrong; this programme changes the *inputs* instead |
| **`Aceituno2025_TemporalHorizons`** — gradient scales `O(e^{λT})` with training horizon | explains all five failures as one proven barrier; noise sidesteps it rather than fighting it |

## Relationship to prior work

Supersedes nothing. Depends on the #308 dossier for its instruments (`run_realism_arms.py`,
`score_arms.sh`, `read_screen.py`, the potency-gate pattern) and on
`reports/2026-08-16_feedback_realism_dossier/` for `--body-mean-dump`. The pushforward arm's own
pre-registration lives in `reports/2026-08-26_pushforward_dossier/05_analysis_plan.md` and is reused
rather than rewritten.

## Document index

| # | File | Status |
|---|---|---|
| 00 | `00_README.md` | living |
| 01 | `01_literature.md` | seeded (S0) |
| 02 | `02_design.md` | **stub — S2 (#314) fills it from S1's numbers** |
| 03 | `03_harness_and_invariants.md` | seeded (S0) — the gate set this dossier opts into |
| 04 | `04_roadmap.md` | points at #320; not duplicated |
| 05 | `05_analysis_plan.md` | **LOCKED at S0, before S1 produces numbers** |
| 07 | `07_experiment_log.md` | empty, append-only |

**No `06_glossary`.** The standing rule is one fixed vocabulary in `reports/GLOSSARY.md`; if a word is
missing it gets edited there, never re-coined per dossier.

## Harness at a glance

Most of the harness already exists — the honest finding is that **the gaps are wiring, not
invention**. Several good gates are CI-tested and invoked by *no recent dossier*. See `03`.

## Conventions

- `05_analysis_plan.md` is **never edited in place** after locking; changes are appended as dated
  `## AMENDMENT An` blocks stating when they were written relative to data existing.
- `07_experiment_log.md` is append-only; negatives are written at the same length as wins.
- Risks go to `reports/technical_risk_register.md` via `register-risk`, not into this dossier.

## Next actions

- [x] S0 — dossier scaffolded; harness audited; plan locked *(this story, #312)*
- [x] S1 #313 — measured: **FN 0.9959 vs FP 0.000027 at h18 (36,870×), CV 0.002** ⇒ design `occurrence_dropout`, STOP-gate (a) passes
- [x] S2 #314 — design + implement
- [x] S3 #315 — adversarial audit — 3 rounds; found a real BN bug (C-328)
- [x] S4 #316 — smoke + potency at a trained checkpoint — PASS (rel 0.600)
- [x] S5 #317 — 2 arms × 300 lessons (pushforward dropped, A1) — FG-A PASS
- [x] S6 #318 — **control 0.3292 → noise 0.1329, Δ = −0.1963; act_ratio ×56**
- [ ] S7 #319 — disposition
