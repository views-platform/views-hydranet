# Bulk Magnitude Calibration @ T=0 — dossier spine

**Opened:** 2026-07-16 · **Status:** INIT (scaffold + harness audit done; pre-registration next).

## 1. Purpose
Fix the **downward magnitude bias of the body** on the **bulk** (bottom ~97–99% of non-zero cells) at
**T=0**, in isolation — freeze the gate, park the extreme tail. The front-runner config (dense-mse +
weighted-BCE, `pos_weight=2`) beats the conflictology baseline on CRPS but is a **timid prophet**
(`ratio_med` 0.05–0.11 — predicts ~5–11% of truth on positive cells). Hypothesised fix (MVP): an
**outlier-robust per-cell winsorize/cap of the target** (the *stabilizer* — removes the infinite-variance
tail's blow-up pressure) **+ a tunable magnitude dial** (moderate-τ log-space pinball; τ is the knob — the
*lifter*) that raises predicted body magnitude toward truth without exploding.

## 2. Relationship to prior work / ADRs
- Builds on: [[project_densemse_beats_baseline]] (dense-mse beats white_ranger on CRPS; loses QS99 via
  under-firing), [[project_body_knob_quest]] (survey: the winsorized-target angle is the untried door;
  every prior body loss went timid or exploded), [[project_volatility_ceiling_predictable]] (bulk
  spread/level predictable; tail = risk not value), [[project_amount_ceiling_wall]] (extreme value
  irreducible → tail parked), [[project_gate_loss_finding]] (`pos_weight` = the *gate* knob, proven).
- Baseline of record: **`white_ranger`** (ConflictologyModel, PGM, viewser) — NOT hurdle_nb (broken).
- Will become: a proposed ADR for a body-magnitude loss if validated (via `promote`).

## 3. Document index
| # | doc | status |
|---|---|---|
| 00 | README (this) | living |
| 01 | literature | seeded (findings; `/library` later) |
| 02 | design | drafted |
| 03 | harness_and_invariants | **drafted — the crown jewel is the bulk-calibration METRIC** |
| 04 | roadmap | drafted |
| 05 | analysis_plan | **pending `preregister`** |
| 06 | glossary | seeded |
| 07 | experiment_log | empty (append-only) |

## 4. Harness at a glance
~70% already exists (config-validator OCP flags, seed lock, config trap-restore, one-heavy-job discipline,
timestamped predictions, `t0_score.py` retrain-free T=0 scorer, locked white_ranger baseline). **The new
harness this program MUST build first = the locked bulk-calibration metric** (T=0-only · positives-only ·
bulk-only · per-cell `ratio_med` not pooled-MCR · guardrails held · same-seed A/B · ≥3 seeds · validation
graduation). See `03`. This measurement is the thing that must be rigorously right (past failures were all
measurement lies: pooled-MCR ×3, rollout-pooling ×1).

## 5. Current state & next actions
**Status: 🛑 CLOSED — F2 (hypothesis REJECTED, 2026-07-17). The global body-magnitude dial does not exist.**
- [x] scaffold + harness audit (`init`)
- [x] **pre-registered** — `05_analysis_plan.md` LOCKED (+ 2 changelogs: P0 revive-first, P3 arm realization)
- [x] **P0** — metric `tools/bulk_score.py` built + validated; anchor found the baseline body DEAD
  (`ratio_med` 0.000, dead-ReLU + all-cell zero-pull)
- [x] **P1** — `PinballBodyLoss` (winsorize + τ-dial) implemented, OCP/default-off, 6 unit tests green
- [x] **P2** — 2-lesson GPU smoke PASS (trains, finite, dial-active, revived hurdle_shrinkage+softplus)
- [x] **P3** — first-seed τ-sweep A/B (5 arms, seed 42, 40L). **VERDICT: F2 FIRES.** The dial lifts
  `ratio_med` (F3 clean — cap alone stays timid) but every band-hit explodes CRPS (sb 18×, ns 70× white_ranger);
  it **rescales, does not calibrate** (spearman/within2x flat; ns ratio-spread p90 30→567). Results in
  `results/score_*.txt` + `07` postmortem. Root: a *global* point-dial can't calibrate a *heterogeneous* bulk.
- **⇒ REDIRECT (user-approved 2026-07-17):** the bulk needs a **per-cell distributional head**, not a scalar
  knob — see `reports/2026-07-15_quantile_head_build_dossier/` (quantile head, banked VIABLE) +
  `reports/2026-07-15_volatility_ceiling_dossier/` (per-cell spread IS predictable). NOT promoted to an ADR
  (negative). **Banked side-findings:** dead→revived body fix (softplus + positives-only); dense-mse's CRPS
  "win" = dead-body-wins-zeros artifact (revived-honest only ties white_ranger on sb); **#144 grid-flip live**
  (parquet/viewser → `priogrid_id`; `data_sniffer` + this scorer weren't grid-agnostic — register-worthy).

## 6. Conventions
Numbered dated docs; `00` living; git-tracked via `git add -f`; archived on `promote`. Anti-circle: one
variable at a time; pre-register then run; cheap readout before expensive; a fired falsifier kills the
hypothesis (no ad-hoc rescue). `conda run -n views-hydranet-env`. views-models config edits stealth
(trap-restore, never push).
