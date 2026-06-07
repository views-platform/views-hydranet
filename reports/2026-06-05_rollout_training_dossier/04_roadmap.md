# 04 — Roadmap (implementation + experiment sequencing)

**Date:** 2026-06-06 · **Status:** seeded · **Dossier:** [00_README](00_README.md)
**Depends on:** [02_design](02_design.md) (B1 pushforward), [03_harness_and_invariants](03_harness_and_invariants.md) (the gates), [05_analysis_plan](05_analysis_plan.md) (the pre-registered MVP).

Turns the design into an **ordered, gated** build. Sequencing principles (`03 §4`): **one variable at a time**; **cheapest informative experiment first**; **pre-register before running** (`05`); each step gates the next (tests → fast readout → eval). Nothing trains until the harness pre-flight (`03 §5`) is green.

---

## 1. Phases

| Phase | Goal | Key steps | Exit gate | Depends on |
|------:|------|-----------|-----------|-----------|
| **P0 — Decide** | settle the B1 knobs | confirm **K=12**; the stability-term form (pushforward: feed the model's own one-step-prior prediction) + its **annealing schedule**; the gradient-clip norm; the feedback space (log1p) | knobs recorded (here + `05`) | — |
| **P1 — Harness build (no model, no training)** | the `03 §3` gaps, TDD | (a) pushforward path behind `rollout_horizon` + **parity test** (K=1 byte-identical); (b) **feedback-gradient-liveness test**; (c) annealed-weight + **uncontaminated-CRPS** check; (d) gradient clipping; (e) full-36 boundedness readout; (f) calibration readout (PIT/coverage/MCR/zero-rate) | `03 §5` pre-flight blockers green; full suite + ruff green | P0 |
| **P2 — B1 wired (behind the flag)** | training can run the rollout objective; baseline untouched | wire the pushforward `L_stability` into `training_engine._process_sequence` behind `rollout_horizon` (default 1) | **parity gate**: K=1 byte-identical to today; suite green | P1 |
| **P3 — First experiment (MVP)** | does B1 bound the runaway *without* breaking calibration? | train **violet/seed-42, ACTIVE balancer**, `rollout_horizon=12` (per `05 §0.1`); fast readout first | `diagnose_io_gain` attractor **in-range across all 36** (vs the `…233938` exploder), then eval: CRPS / MCR / calibration vs `…051634` (pre-registered in `05`) | P2 |
| **P4 — Iterate / escalate** | reach a verdict | per `05` decision rules: **F1 fires → B2 GTF** (un-detach, soft-mix, α-bound, clip); **F2 fires → ZITD layering** (C-129); vary K; **multi-seed** confirm (C-112) | each step pre-registered; logged in `07` | P3 |
| **P5 — Decide & graduate** | commit or fall back | if B1 wins on active: **ADR-058** (`docs/ADRs/proposed/`) + **resolves C-124** + roll to golden_hour; if not: postmortem → B2 GTF or the ZITD layer | ADR proposed *or* documented negative result; dossier archived | P4 |

## 2. Dependency graph

```
[C-121 guard #76 — DONE] ─┐
P0 decide ─▶ P1 harness build (TDD: parity · grad-liveness · CRPS-quarantine · clip · 36-readout · calibration)
                          └─▶ P2 B1 wired (flag, K=1 parity) ─▶ P3 MVP (violet/42, ACTIVE, K=12)
                                                                      │
                                                                      ├─▶ P4  F1→B2 GTF · F2→ZITD layer · vary K · multi-seed
                                                                      └─▶ P5  win → ADR-058 (+resolve C-124) · else → postmortem → B2/ZITD
ZITD dossier (C-129) ───────────────────────────────────────────────────▶ layered AFTER Axis B (P4/F2), not concurrent
```

## 3. The first experiment (MVP) — why this shape
Smallest change that tests the core claim, per `05`. **One model** (violet/seed-42, the clean exploder), **active balancer** (the production setting + the thing that explodes — so bounding it is the real win and **resolves C-124**), **K=12** (reaches the step-12 onset), **pushforward** (flat memory, the cheap B1 before the heavier B2 GTF). Readout is the **retrain-free `diagnose_io_gain`** first (≈30 s: in-range across 36?), *then* a full eval for CRPS/MCR/**calibration** (the F2/C-126 guard). Falsifiers pre-registered in `05 §4`.

## 4. Decision points (the open choices)

| Choice | First cut | Decide at | Note |
|--------|-----------|-----------|------|
| `rollout_horizon` K | **12** | P0→P4 | reaches the blow-up onset; raise toward 36 if F3 (tail divergence) fires |
| stability mechanism | **B1 pushforward** (detach across steps, backprop last) | P3→P4 | **B2 GTF** is the fallback if F1 fires (`02 §7.4`) |
| stability-term weight | **annealed, small** | P0/P1 | R1 — CRPS stays the uncontaminated headline |
| balancer arm | **active (primary)** | P3 (`05 §0.1`) | frozen = control; R6 satisfied |
| feedback space | **log1p** (where `‖J‖₂>1` was measured) | P1 | the interpolation/perturbation lives here |
| rollout × ZITD | **Axis B first** | P4/F2 (C-129) | ZITD softplus head layered only if calibration still needs it |

## 5. Relationship to other work
- **The C-121 guard (#76, DONE):** the prerequisite — already merged; P1 extends it to the full-36 readout.
- **The C-111 balancer (verdict in):** R6 satisfied; the **active** arm doubles as the C-124 resolution test.
- **ZITD dossier (C-129):** sequenced **after** Axis B (P4/F2 layering), not concurrent — both touch `training_engine` + the feedback.
- **B2 GTF / Professor Forcing (`02 §4.2`):** the escalation ladder if pushforward under-delivers.

## 6. Milestones / definition of done
- **M1 (P1):** `03 §3` harness gaps built + tests green — *the pre-flight blockers cleared*.
- **M2 (P2):** B1 wired behind `rollout_horizon`; **baseline byte-identical with K=1**; suite green.
- **M3 (P3):** violet/42 active + B1 trained; `diagnose_io_gain` in-range across 36; first CRPS/MCR/calibration vs baselines in `07`.
- **M4 (P4):** verdict path taken (B1 holds · or → B2 GTF · or → ZITD layer); multi-seed confirm.
- **M5 (P5):** ADR-058 proposed (adopt, + C-124 resolved) **or** postmortem (fall back) — dossier archived.

> Cost note: **P0–P2 are CPU/test work (no GPU)** — the real "before you experiment" build. The first GPU cost is **P3** (one ~80-min violet train + 30 s readout + one ~40-min eval). P4 ablations are the main GPU spend — one model at a time, n ≤ 64, pre-registered, GPU-enforced driver (`03 §4`).
