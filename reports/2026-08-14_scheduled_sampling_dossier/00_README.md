# Scheduled-sampling dose-response dossier (2026-08-14)

**Question:** does closing the train/rollout exposure gap (scheduled sampling, ADR-056) flatten the
`truncated_nb` activation bloom (act_ratio 1.6×→44× across h1→h36, #258)? Tested as a **dose-response**
in `ss_epsilon_max` ∈ {0, 0.1, 0.25, 0.5}, one variable, seed 42, 40L, on africa datafactory truth.

**Why this order (register → verify → pre-register → run):** past experiments here generated invalid
knowledge from wrong/half-wired implementations. So before enabling SS we (a) `/register-risk`'d the
findings (C-259..C-262 + C-246), (b) fixed + **proved** the SS piping correct (train↔inference feedback
byte-parity test, coupling/order validators), (c) fixed a 160× sampler-perf blocker, (d) pre-registered
(`05_analysis_plan.md`, LOCKED), then run. Fixes committed `c07a352` on `feat/truncated-nb-family`.

## Index
- `05_analysis_plan.md` — 🔒 LOCKED pre-registration (hypothesis, predictions P1–P3, falsifiers F1/F2/F-DEGEN).
- `07_experiment_log.md` — append-only outcomes (negatives first-class).
- `results/` — per-arm score CSVs + the horizon curve.

## Harness / guardrails (audit)
- **SS piping == inference exposure:** `tests/train/test_feedback_parity.py` (byte-equal) — the gate.
- **Fail-loud config:** `validate_scheduled_sampling_params` (C-259 coupling + mean-under-gate; C-260 order).
- Fresh datafactory pull (never stale); `diagnostic_visualizations=False`; setsid + floor trap-restore.
- Scorer: `reports/2026-07-29_v2_scoreboard_dossier/tools/{score_v2_horizons,activation_metrics}.py`.
- Vehicle: `views-models/models/truncated_smoke` (scratch clone of violet; NOT a roster member).

## Status
LOCKED + running the 4-arm ε sweep (2026-08-14).
