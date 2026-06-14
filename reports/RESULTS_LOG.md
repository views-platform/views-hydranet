# Results Log — HydraNet (FAO Pre-Release Note 05 metrics)

> ⚠️ **C-119 confound (2026-06-15).** Every single-run comparison logged here **before** the determinism
> fix (commit `daab1c1`) is **confounded**: training was non-deterministic at a fixed seed (~20% run-to-run
> variance; root cause C-119 — `reports/postmortem_training_nondeterminism_init_rng_drift.md`). The no-coords
> baseline alone swung FULL MCR sb 2.99–3.69 run-to-run. **Treat all pre-fix deltas — including the
> coordinate #110 verdict and any FAO eligibility calls — as not trustworthy.** Re-run on the deterministic
> pipeline (post-`daab1c1`) before drawing any comparative conclusion.

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


### overnight S1_seed42 — hurdle_nb (theta_init=1.0, pos_weight=10, seed=42, 40 lessons) — 2026-06-11
- train+eval exit=0; predictions=predictions_calibration_20260611_011210
```
### RUN: /home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260611_011210
  lr_sb_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.411 [95%CI 0.32,0.52]  CRPS=15.5983 twCRPS=15.5983  median-ratio=0.26 (IQR 0.01-1.43)  n_pos=1320
    FULL   (all steps, incl. rollout):       MCR=2.547 [95%CI 2.02,2.96]  CRPS=41.1703 twCRPS=41.1703  median-ratio=9.08 (IQR 2.95-24.38)  n_pos=53847
  lr_ns_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.404 [95%CI 0.31,0.53]  CRPS=22.7994 twCRPS=22.7994  median-ratio=0.27 (IQR 0.01-2.34)  n_pos=587
    FULL   (all steps, incl. rollout):       MCR=3.377 [95%CI 3.22,3.54]  CRPS=43.2323 twCRPS=43.2323  median-ratio=10.33 (IQR 3.02-27.92)  n_pos=17686
  lr_os_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.451 [95%CI 0.34,0.58]  CRPS=6.5555 twCRPS=6.5555  median-ratio=0.05 (IQR 0.00-0.73)  n_pos=706
    FULL   (all steps, incl. rollout):       MCR=4.582 [95%CI 4.40,4.78]  CRPS=37.8836 twCRPS=37.8836  median-ratio=14.60 (IQR 3.77-47.79)  n_pos=28328
```

### overnight S1_seed4 — hurdle_nb (theta_init=1.0, pos_weight=10, seed=4, 40 lessons) — 2026-06-11
- train+eval exit=0; predictions=predictions_calibration_20260611_013849
```
### RUN: /home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260611_013849
  lr_sb_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.577 [95%CI 0.44,0.72]  CRPS=16.4861 twCRPS=16.4861  median-ratio=0.54 (IQR 0.03-2.33)  n_pos=1320
    FULL   (all steps, incl. rollout):       MCR=5.460 [95%CI 4.36,6.35]  CRPS=87.4036 twCRPS=87.4036  median-ratio=20.56 (IQR 6.69-52.48)  n_pos=53847
  lr_ns_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.396 [95%CI 0.31,0.54]  CRPS=22.5614 twCRPS=22.5614  median-ratio=0.29 (IQR 0.01-2.37)  n_pos=587
    FULL   (all steps, incl. rollout):       MCR=5.439 [95%CI 5.19,5.70]  CRPS=70.2049 twCRPS=70.2049  median-ratio=17.36 (IQR 5.16-45.06)  n_pos=17686
  lr_os_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.697 [95%CI 0.53,0.90]  CRPS=7.0620 twCRPS=7.0620  median-ratio=0.22 (IQR 0.01-1.67)  n_pos=706
    FULL   (all steps, incl. rollout):       MCR=7.422 [95%CI 7.12,7.69]  CRPS=61.1308 twCRPS=61.1308  median-ratio=25.87 (IQR 6.84-77.01)  n_pos=28328
```

### overnight S2_seed42 — hurdle_nb (theta_init=1.0, pos_weight=25, seed=42, 40 lessons) — 2026-06-11
- train+eval exit=0; predictions=predictions_calibration_20260611_020548
```
### RUN: /home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260611_020548
  lr_sb_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.519 [95%CI 0.41,0.66]  CRPS=16.0166 twCRPS=16.0166  median-ratio=0.54 (IQR 0.03-2.32)  n_pos=1320
    FULL   (all steps, incl. rollout):       MCR=2.363 [95%CI 1.88,2.74]  CRPS=38.6380 twCRPS=38.6380  median-ratio=8.47 (IQR 2.85-22.39)  n_pos=53847
  lr_ns_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.315 [95%CI 0.24,0.42]  CRPS=22.0398 twCRPS=22.0398  median-ratio=0.15 (IQR 0.00-1.50)  n_pos=587
    FULL   (all steps, incl. rollout):       MCR=2.898 [95%CI 2.76,3.04]  CRPS=36.8955 twCRPS=36.8955  median-ratio=8.71 (IQR 2.41-24.15)  n_pos=17686
  lr_os_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.752 [95%CI 0.57,0.98]  CRPS=7.5174 twCRPS=7.5174  median-ratio=0.16 (IQR 0.00-1.61)  n_pos=706
    FULL   (all steps, incl. rollout):       MCR=7.718 [95%CI 7.40,8.03]  CRPS=64.7791 twCRPS=64.7791  median-ratio=23.90 (IQR 6.46-82.54)  n_pos=28328
```

### overnight S2_seed4 — hurdle_nb (theta_init=1.0, pos_weight=25, seed=4, 40 lessons) — 2026-06-11
- train+eval exit=0; predictions=predictions_calibration_20260611_023311
```
### RUN: /home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260611_023311
  lr_sb_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.759 [95%CI 0.59,0.97]  CRPS=16.9666 twCRPS=16.9666  median-ratio=1.11 (IQR 0.19-3.64)  n_pos=1320
    FULL   (all steps, incl. rollout):       MCR=5.631 [95%CI 4.45,6.54]  CRPS=89.3496 twCRPS=89.3496  median-ratio=21.29 (IQR 7.31-52.86)  n_pos=53847
  lr_ns_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.400 [95%CI 0.31,0.54]  CRPS=22.3149 twCRPS=22.3149  median-ratio=0.35 (IQR 0.01-2.29)  n_pos=587
    FULL   (all steps, incl. rollout):       MCR=5.779 [95%CI 5.51,6.05]  CRPS=74.4510 twCRPS=74.4510  median-ratio=18.51 (IQR 5.88-46.69)  n_pos=17686
  lr_os_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=1.356 [95%CI 1.04,1.72]  CRPS=9.5096 twCRPS=9.5096  median-ratio=0.95 (IQR 0.03-4.63)  n_pos=706
    FULL   (all steps, incl. rollout):       MCR=13.357 [95%CI 12.84,13.88]  CRPS=115.3394 twCRPS=115.3394  median-ratio=46.56 (IQR 14.28-136.14)  n_pos=28328
```

### overnight S3_seed42 — hurdle_nb (theta_init=0.3, pos_weight=10, seed=42, 40 lessons) — 2026-06-11
- train+eval exit=0; predictions=predictions_calibration_20260611_030028
```
### RUN: /home/simon/Documents/scripts/views_platform/views-models/models/violet_visitor/data/generated/predictions_calibration_20260611_030028
  lr_sb_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.542 [95%CI 0.42,0.69]  CRPS=16.3636 twCRPS=16.3636  median-ratio=0.38 (IQR 0.02-2.00)  n_pos=1320
    FULL   (all steps, incl. rollout):       MCR=3.306 [95%CI 2.65,3.84]  CRPS=53.1349 twCRPS=53.1349  median-ratio=11.56 (IQR 3.76-32.39)  n_pos=53847
  lr_ns_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.285 [95%CI 0.22,0.38]  CRPS=21.7133 twCRPS=21.7133  median-ratio=0.06 (IQR 0.00-1.04)  n_pos=587
    FULL   (all steps, incl. rollout):       MCR=2.984 [95%CI 2.84,3.13]  CRPS=37.2525 twCRPS=37.2525  median-ratio=8.73 (IQR 2.18-25.08)  n_pos=17686
  lr_os_best:
    STEP-1 (teacher-forced, magnitude axis): MCR=0.552 [95%CI 0.41,0.72]  CRPS=6.6001 twCRPS=6.6001  median-ratio=0.15 (IQR 0.02-1.18)  n_pos=706
    FULL   (all steps, incl. rollout):       MCR=6.631 [95%CI 6.37,6.89]  CRPS=54.9811 twCRPS=54.9811  median-ratio=20.02 (IQR 5.27-69.59)  n_pos=28328
```

### overnight S3_seed4 — hurdle_nb (theta_init=0.3, pos_weight=10, seed=4, 40 lessons) — 2026-06-11
- train+eval exit=120; predictions=predictions_calibration_20260611_032845
```
```
