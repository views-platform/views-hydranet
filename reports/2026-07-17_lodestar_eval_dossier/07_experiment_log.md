# 07 — Experiment Log (append-only; negatives first-class)

Every entry links its pre-registration (`05`) and states the verdict vs the pre-committed falsifiers.
This is a MEASUREMENT build — the "result" is a trustworthy table + a frozen ruler, and any place the
ruler surprises us (small support, coverage gaps, a family that can't be scored) is logged as prominently
as a clean number. Newest at bottom.

---

### Ruler built + validated + FROZEN · white_ranger re-aligned · grid launched · 2026-07-17
- **Ruler** `tools/lodestar_score.py`: self-test PASS (CRPS energy-form vs brute force, event/zero split,
  AP, Brier all reproduce hand-computed values). End-to-end PASS on real predictions (white_ranger vs
  hydranet A0p): common support N=170,430, months 457–469, one truth parquet, sane finite metrics. **FROZEN.**
- **white_ranger re-run** on current partition → calibration T=0 months **457–469** (13 origins, all 3
  targets), exit 0. **Now aligned with hydranet** (the original 445-vs-457 misalignment is fixed).
  Preds: `white_ranger/.../predictions_calibration_20260717_110149`.
- **⚠️ HONEST INTERIM (A0p vs white_ranger, properly aligned — NOT the grid winner yet):** on **sb**, the
  simple climatology baseline BEATS hydranet A0p on switch AP (0.334 vs 0.283), Brier, all-CRPS (0.19 vs
  0.28), zero-CRPS (0.042 vs 0.133), size ratio (0.39 vs 0.13), and **ties event-CRPS** (19.2 vs 19.7).
  This **overturns** the earlier "hydranet beats baseline on event-CRPS" read — that was on mismatched
  months (F2-class error the ruler now prevents). hydranet's switch only leads on **ns** (AP 0.273 vs 0.223).
  Full: `results/validation_A0p_vs_white_ranger.csv`.
- **Grid launched** (`ls_manifest.txt`): pos_weight {1,2,4} × seed {42,43,44} = 9 runs, best-known
  architecture, train+eval → scoreable hurdle predictions. The grid may beat A0p (esp. on the switch, which
  pos_weight directly moves). Verdict deferred to the full grid score → the lodestar table.

---

### Plan pivot: positives-only grid KILLED → all-cell + MSE grid · 2026-07-17
- The v1 positives-only grid was **killed** (1 of 9 runs done) at the user's instruction: the positives-only
  body's raw output looked broken (a diffuse blob — the body never learns the empty background and leans
  entirely on the gate). Machine confirmed clean after kill (floor md5 OK, GPU idle).
- **Vocabulary LOCKED** (`reports/GLOSSARY.md` v2): gate (not switch); one gated forecast (no "dense
  forecast"); body = bulk + tail; MAE ≡ pinball-at-0.5 (same thing, my past renaming caused the confusion).
- **New plan (05 v2):** gated forecast, **all-cell body, MSE, softplus, BatchNorm fix on**; pos_weight
  {2,3,4,5} × seed {42,43,44} = 12 runs. Answers 3 questions: gate calibration vs pos_weight (AP+Brier);
  body wobble across seeds; gated forecast vs white_ranger. Config verified: `hurdle_shrinkage` + no
  `hurdle_threshold` + `mse` = gated forecast with all-cell body (`training_engine.py:371-372`).
- **Ruler:** added **pos-mcr** (mean of guess÷truth on conflict cells; distinct from size-ratio = median).
  Self-test re-run. Frozen again.
- Next: 2-lesson smoke (bloom check) → 12-cell grid → score.

### Smoke PASS (bloom check) + grid launched · 2026-07-17
- **2-lesson smoke** (all-cell body, gated forecast, MSE, softplus): trained + **evaluated cleanly, exit 0,
  ZERO "infinity" — NO BLOOM.** The gate suppresses the rollout as predicted; F2 does not fire. Floor
  restored (md5 OK). ⇒ the all-cell gated forecast is safe to score at T=0.
- **12-run grid launched** (`found_manifest.txt`): pos_weight {2,3,4,5} × seed {42,43,44}, 40 lessons,
  train+eval. Scored next on the frozen ruler vs white_ranger → the lodestar table (answers Q1 gate
  calibration, Q2 body seed-wobble, Q3 gated forecast vs baseline).

---

### FOUNDATION RESULT — 12-cell grid scored on the frozen ruler · 2026-07-17 · VERDICT
Pre-registration: `05` v2. All 12 runs exit 0; floor restored. Scored vs white_ranger, T=0, N=170,430,
months 457–469, identical cells. Full: `results/lodestar_full.csv` (39 rows) + `results/lodestar_table.csv`.

**Headline (lower CRPS/Brier better; AP higher better; size-ratio 1.0 = right-sized):**
| target | model | AP | Brier | crps-all | crps-events | crps-none | size-ratio |
|---|---|---|---|---|---|---|---|
| sb | white_ranger | 0.334 | 0.0065 | 0.191 | 19.23 | 0.042 | 0.39 |
| sb | gated all-cell pw2 | 0.323 | 0.0103 | **0.137** | **17.13** | 0.005 | 0.019 |
| ns | white_ranger | 0.223 | 0.0031 | 0.088 | **22.50** | 0.010 | 0.15 |
| ns | gated all-cell pw2 | 0.277 | 0.0039 | **0.083** | 23.90 | 0.001 | 0.003 |
| os | white_ranger | 0.158 | 0.0039 | 0.030 | **5.99** | 0.005 | 0.039 |
| os | gated all-cell pw2 | 0.120 | 0.0045 | **0.028** | 6.55 | 0.001 | 0.002 |

**Verdict vs pre-registered predictions:**
- **P1 (gate):** AP FLAT across pos_weight ✔; Brier moves ✔ — but Brier gets **WORSE** as pos_weight rises,
  so pos_weight **2 is the best-calibrated** of {2,3,4,5}. The "2 is too low" hunch is **NOT supported** (test
  below 2 next if we care). vs baseline: baseline gate better on sb/os (AP+Brier); **our gate wins ns AP**.
- **P2 (seed wobble):** CONFIRMED small — crps-events spans ~16.3–18.4 on sb (~12%), tighter on ns/os. The
  BatchNorm fix settled the wobble. Body is seed-stable.
- **P3 (baseline beats/ties on sb):** **FALSIFIED (good surprise).** The all-cell gated forecast BEATS the
  baseline on **crps-all all 3 targets**, and on **crps-events for sb** (17.1 < 19.2). The all-cell change
  fixed the zero-smear that sank the positives-only A0p (crps-none 0.005 vs A0p's 0.133).
- **P4 (timid):** CONFIRMED — size-ratio 0.02–0.03 (vs baseline 0.39). The body wins CRPS partly by being
  conservative on a mostly-zero target. crps-events is mixed: grid wins sb, baseline wins ns/os.

**Falsifiers:** none fired. F1 (ruler) — self-test + end-to-end passed. F2 (bloom) — smoke clean. F3
(months) — all 457–469. F4 (coverage) — all cover N=170,430. **Measurement is valid.**

**FOUNDATION (what we now stand on, honestly):** the **all-cell + MSE gated forecast** is a real, seed-stable
model that **beats white_ranger on the headline score (crps-all) on all 3 targets**, with **pos_weight 2**
the best-calibrated gate — but the **body is timid** (size-ratio ~0.02). Timidity is the thing to improve.
This table is the lodestar; future ideas are judged here, same frozen ruler, same cells.

---

### Gate follow-ups G1 (focal vs weighted_bce) + G2 (per-target pos_weight) launched · 2026-07-17
Body held fixed at the foundation (all-cell MSE gated forecast, softplus, BN-fix). Judged on the frozen ruler.
- **G1 (config-only, running):** gate = **focal** (gamma 2, alpha {0.25, 0.75}) × seed {42,43,44} = 6 runs,
  vs the weighted_bce pw2 foundation. Question: is weighted_bce actually better than focal *here* (old sweep
  said focal worse, but on a different setup/eval — treated as a prior, not fact).
- **G2 (needs code, built + tested, chained after G1):** per-target pos_weight — sb/ns/os each get their own
  gate eagerness. **Code (TDD, 41 tests green):** `loss_class_pos_weight` now accepts a list; `choose_loss`
  builds a per-target list of `weighted_bce` losses; the training loop applies `criterion_class[j]` per
  target (mirrors the per-target reg dict). Config: list length must == #classification_targets, each >0.
  Values [sb2, ns5, os2] (foundation-informed: ns's AP rose with pos_weight, sb/os prefer low) × seed
  {42,43,44} = 3 runs + a 2-lesson smoke. Question: does per-target beat the best shared (pw2)?

---

### G1 VERDICT — weighted_bce > focal (2026-07-18)
Gate loss swept (body fixed all-cell MSE); scored on the frozen ruler, 3 seeds each. Results
`results/g1_focal_vs_bce.csv`. **weighted_bce WINS AP on all 3 targets** (non-overlapping seed spreads —
sb bce 0.30–0.35 vs focal 0.20–0.28; ns 0.25–0.30 vs 0.15–0.26; os 0.11–0.13 vs 0.06–0.08). focal α0.25's
only edge = marginally better Brier on sb (0.0086 vs 0.0103) at a large AP cost; α0.75 worse on both. Body
scores ~identical (clean isolation). **The old "focal worse" prior HOLDS on the frozen ruler. Keep
weighted_bce; focal is not a lever.**

### G2 VERDICT — per-target pos_weight WORKS (separability holds) (2026-07-18)
Per-target code (TDD, 41 tests) built; smoke clean (per-target training loop + no bloom); per-target
[sb2, ns5, os2] × 3 seeds vs shared pw2 & pw5. Results `results/g2_per_target.csv`. **Each target reaches
its assigned pos_weight's behavior:** ns (5) → AP **0.291** (up from 0.277 at shared pw2, ≈ shared pw5)
while sb/os (2) keep their good pw2 Brier (0.010 / 0.0046) instead of pw5's worse Brier (0.016 / 0.008).
So per-target is the **best of both** — ns AP boost + sb/os calibration in one model. **Honest:** the ns gain
is modest (+0.014 AP) with overlapping seed spreads; a validated, ~free lever (mainly helps ns), not a
game-changer. **User's instinct confirmed: the targets want different pos_weights.** Per-target pos_weight
code is a keeper (uncommitted).
