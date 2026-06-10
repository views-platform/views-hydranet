# Results Log — HydraNet (FAO Pre-Release Note 05 metrics)

**Every run logged here, one row. Append-only.** Metrics follow FAO PRN-05 (Topic A); this is the
standing comparison table going forward. (The narrative history lives in `RESULTS_LEDGER.md`; *this* is
the clean per-run table.)

## Metrics (FAO PRN-05, Topic A) — what each column means
| Column | Role (FAO) | Better | Definition |
|--------|------------|--------|------------|
| **CRPS** | **PRIMARY ranking metric** | lower | Continuous Ranked Probability Score (proper score on the posterior samples). **Superiority = CRPS ≥ 5% lower than baseline.** |
| **QS99** | guardrail — *tail sanity* | lower | 99th-percentile quantile score; catches *timid* models that under-predict severe events. |
| **Brier** | guardrail — *hurdle/onset calibration* | lower | Calibration of P(y>0) (conflict onset). |
| **MCR** | guardrail — *magnitude calibration* | **→ 1** | `mean(ŷ) / mean(y)`. <1 under-predicts, >1 over-predicts. |
| **Bounded?** | **our C-113 sanity gate** (not FAO) | bounded | Does the 36-step free-running rollout stay in range, or `expm1`-explode? (✓ / 💥) |
| **Note** | — | — | Qualitative read; config nuance; anything wandb doesn't capture. |

- **Guardrail non-inferiority:** QS99 & Brier ≥ 1% better than baseline; MCR's `|MCR−1|` at least marginally closer-to-1 than baseline.
- **Eligibility (FAO Topic C):** Eligible iff **CRPS ≥5% better than baseline AND all 3 guardrails non-inferior.** Fail any → Ineligible. *(Our hard pre-gate: `Bounded?=💥` ⇒ auto-Ineligible — an exploding model cannot be evaluated honestly.)*
- **Coverage:** full dataset (not extremes-only — avoids the forecaster's dilemma). Metrics reported **per target: `lr_sb` / `lr_ns` / `lr_os`**.
- *(twCRPS = supplementary diagnostic only; Log Score rejected — per FAO Topic A.)*
- **Baseline:** *TBD — set once a clean (bounded) reference run lands.*

## Run table
| # | Date | Config (variation axes) | CRPS ↓ (sb/ns/os) | QS99 ↓ (sb/ns/os) | Brier ↓ (sb/ns/os) | MCR →1 (sb/ns/os) | Bounded? | Eligible? | Note |
|---|------|--------------------------|-------------------|-------------------|--------------------|-------------------|----------|-----------|------|
| 1 | 2026-06-10 | **violet** · hurdle (`lognormal_nll`, σ=0.9, `hurdle_threshold=0`) **+ SS** (`ss_epsilon_max=0.5`) · 40 lessons · active balancer · seed 42 · artifact `…_20260610_010843` | **eval crashed** | **eval crashed** | **eval crashed** | **≈3.4e33 / 3.1e33 / 7.5e33** † | **💥** | **Ineligible** | **R4** — cheap rollout probe (one var vs Arm-1: SS on). Eval **FAILED — "Input contains infinity"**, so the FAO pipeline produced **no** CRPS/QS99/Brier. † MCR is from our `mcr_readout` (full-rollout); pipeline crashed before computing metrics. Step-1 magnitude un-collapsed but not better (sb 0.21→0.088). **Completes the SS bracket: 0.25/0.5/1.0 all fail.** Current-state "before" for the C-111 fix. |
