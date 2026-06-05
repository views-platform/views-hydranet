# S0 Baseline Metrics — Reference Card

**Date:** 2026-05-29
**Config:** `config_hp_s0_baseline.py` (production baseline, no new features)
**Run type:** Single calibration run, `n_posterior_samples=3`, `evaluation_mode=stochastic`

## Key Metrics (step-wise, primary reference for pass/fail comparison)

| Target | CRPS | CRPS_mean | QS_sample | QS_sample_mean | MCR_sample | MCR_sample_mean |
|--------|------|-----------|-----------|----------------|------------|-----------------|
| lr_sb_best | 0.291 | 0.310 | 0.132 | 0.153 | 2.046 | 1.885 |
| lr_ns_best | 0.104 | 0.140 | 0.033 | 0.068 | 4.881 | 2.675 |
| lr_os_best | 0.255 | 0.237 | 0.054 | 0.035 | 6.658 | 11.293 |

## Classification (step-wise Brier scores)

| Target | Brier_cls_sample | Brier_cls_sample_mean |
|--------|------------------|----------------------|
| by_sb_best | 0.00923 | 0.00817 |
| by_ns_best | 0.00288 | 0.00333 |
| by_os_best | 0.00486 | 0.00423 |

## Pass/Fail Thresholds

A subsequent step PASSES if:
1. Evaluation completes without crash or infinity
2. CRPS values are within ~2× of baseline (i.e., no metric exceeds ~0.6)
3. MCR values are within ~3× of baseline (i.e., no MCR exceeds ~35)
4. Brier scores are within ~2× of baseline (i.e., no Brier exceeds ~0.02)

These are generous bounds — we're looking for catastrophic degradation, not fine-tuning.

## Full wandb Run Summary

### month-wise
| Metric | Value |
|--------|-------|
| by_ns_best/Brier_cls_sample | 0.00213 |
| by_ns_best/Brier_cls_sample_mean | 0.00334 |
| by_os_best/Brier_cls_sample | 0.00525 |
| by_os_best/Brier_cls_sample_mean | 0.00426 |
| by_sb_best/Brier_cls_sample | 0.00851 |
| by_sb_best/Brier_cls_sample_mean | 0.00833 |
| lr_ns_best/CRPS | 0.11172 |
| lr_ns_best/CRPS_mean | 0.14014 |
| lr_ns_best/MCR_sample | 3.81794 |
| lr_ns_best/MCR_sample_mean | 3.34888 |
| lr_ns_best/QS_sample | 0.04109 |
| lr_ns_best/QS_sample_mean | 0.06849 |
| lr_os_best/CRPS | 0.27952 |
| lr_os_best/CRPS_mean | 0.24323 |
| lr_os_best/MCR_sample | 4.58887 |
| lr_os_best/MCR_sample_mean | 12.02263 |
| lr_os_best/QS_sample | 0.07725 |
| lr_os_best/QS_sample_mean | 0.04017 |
| lr_sb_best/CRPS | 0.24892 |
| lr_sb_best/CRPS_mean | 0.31943 |
| lr_sb_best/MCR_sample | 3.02924 |
| lr_sb_best/MCR_sample_mean | 2.00018 |
| lr_sb_best/QS_sample | 0.09047 |
| lr_sb_best/QS_sample_mean | 0.16551 |

### step-wise
| Metric | Value |
|--------|-------|
| by_ns_best/Brier_cls_sample | 0.00288 |
| by_ns_best/Brier_cls_sample_mean | 0.00333 |
| by_os_best/Brier_cls_sample | 0.00486 |
| by_os_best/Brier_cls_sample_mean | 0.00423 |
| by_sb_best/Brier_cls_sample | 0.00923 |
| by_sb_best/Brier_cls_sample_mean | 0.00817 |
| lr_ns_best/CRPS | 0.10423 |
| lr_ns_best/CRPS_mean | 0.13979 |
| lr_ns_best/MCR_sample | 4.88061 |
| lr_ns_best/MCR_sample_mean | 2.67450 |
| lr_ns_best/QS_sample | 0.03271 |
| lr_ns_best/QS_sample_mean | 0.06808 |
| lr_os_best/CRPS | 0.25530 |
| lr_os_best/CRPS_mean | 0.23690 |
| lr_os_best/MCR_sample | 6.65764 |
| lr_os_best/MCR_sample_mean | 11.29340 |
| lr_os_best/QS_sample | 0.05441 |
| lr_os_best/QS_sample_mean | 0.03483 |
| lr_sb_best/CRPS | 0.29066 |
| lr_sb_best/CRPS_mean | 0.31008 |
| lr_sb_best/MCR_sample | 2.04618 |
| lr_sb_best/MCR_sample_mean | 1.88521 |
| lr_sb_best/QS_sample | 0.13211 |
| lr_sb_best/QS_sample_mean | 0.15344 |

### time-series-wise
| Metric | Value |
|--------|-------|
| by_ns_best/Brier_cls_sample | 0.00311 |
| by_ns_best/Brier_cls_sample_mean | 0.00333 |
| by_os_best/Brier_cls_sample | 0.00446 |
| by_os_best/Brier_cls_sample_mean | 0.00423 |
| by_sb_best/Brier_cls_sample | 0.00832 |
| by_sb_best/Brier_cls_sample_mean | 0.00817 |
| lr_ns_best/CRPS | 0.12547 |
| lr_ns_best/CRPS_mean | 0.13979 |
| lr_ns_best/MCR_sample | 2.79398 |
| lr_ns_best/MCR_sample_mean | 2.25497 |
| lr_ns_best/QS_sample | 0.05422 |
| lr_ns_best/QS_sample_mean | 0.06808 |
| lr_os_best/CRPS | 0.24195 |
| lr_os_best/CRPS_mean | 0.23690 |
| lr_os_best/MCR_sample | 9.51564 |
| lr_os_best/MCR_sample_mean | 10.93041 |
| lr_os_best/QS_sample | 0.03926 |
| lr_os_best/QS_sample_mean | 0.03483 |
| lr_sb_best/CRPS | 0.28687 |
| lr_sb_best/CRPS_mean | 0.31008 |
| lr_sb_best/MCR_sample | 2.04534 |
| lr_sb_best/MCR_sample_mean | 1.74950 |
| lr_sb_best/QS_sample | 0.13015 |
| lr_sb_best/QS_sample_mean | 0.15344 |
