"""S7 (#200): counted bloom verdict from the 18 banked S6 arms.

Corrected signature (matches EXP-2 + rollout_diagnostics, NOT the buggy s6_score_one auto-verdict):
  bloom fingerprint = FIELD-WIDE wrong-zero mass, read as
    (a) crps_none(h) — CRPS on the zero cells (EXP-2's signature); a bloom drives it up as the
        rollout smears magnitude onto cells that should be 0.
    (b) M_mean(h)     — field-mean emitted log1p magnitude; a bloom drives the whole field out of range.
  NOT M_max (a single outlier cell) and NOT crps_all(36)/crps_all(1) (crps_events is truth-driven at
  long h — identical for mean & sample — so crps_all degradation is a horizon effect, not a bloom).

Per (arm, seed): compare mean-feedback vs sample-feedback on max_h crps_none and max_h M_mean, worst
target. "Bounded" = max crps_none < 0.5 AND max M_mean < 5 (log1p; e^5≈148/cell avg is already far
out of range for 99.7%-zero data). Count how many of the 9 mean arms bloom and 9 sample arms bound.
"""

from __future__ import annotations

import pandas as pd

RES = "/home/simon/Documents/scripts/views_platform/views-hydranet/reports/2026-07-25_t0_rollout_skill_dossier/results/bloomverify"
ARMS = ["gated_NB", "th_gated_NB", "ZINB"]
SEEDS = [42, 43, 44]
TGTS = ["sb", "ns", "os"]
CRPS_NONE_CAP = 0.5
M_MEAN_CAP = 5.0


def arm_stats(label):
    c = pd.read_csv(f"{RES}/bloomverify_{label}.csv")
    t = pd.read_csv(f"{RES}/traj_{label}.csv")
    # worst-target max over horizons
    max_cn = max(c[c.target == tg]["crps_none"].max() for tg in TGTS)
    max_mm = max(t[t.target == tg]["M_mean"].max() for tg in TGTS)
    h1_all = {tg: float(c[(c.target == tg) & (c.h == 1)]["crps_all"].iloc[0]) for tg in TGTS}
    return max_cn, max_mm, h1_all


def bounded(max_cn, max_mm):
    return (max_cn < CRPS_NONE_CAP) and (max_mm < M_MEAN_CAP)


print("Arm            seed | MEAN: maxCRPSnone  maxMmean  -> verdict | SAMPLE: maxCRPSnone maxMmean -> verdict", flush=True)
print("-" * 108, flush=True)
mean_blooms = sample_bounded = 0
t0_deltas = []
for arm in ARMS:
    for seed in SEEDS:
        mcn, mmm, mh1 = arm_stats(f"{arm}_{seed}_mean")
        scn, smm, sh1 = arm_stats(f"{arm}_{seed}_sample")
        mv = "bounded" if bounded(mcn, mmm) else "BLOOMS"
        sv = "bounded" if bounded(scn, smm) else "BLOOMS"
        if mv == "BLOOMS":
            mean_blooms += 1
        if sv == "bounded":
            sample_bounded += 1
        # T=0-neutrality check: h=1 crps_all should match mean vs sample (seed step is pre-feedback)
        for tg in TGTS:
            t0_deltas.append(abs(mh1[tg] - sh1[tg]))
        print(
            f"{arm:14s} {seed}  |       {mcn:8.3f}  {mmm:8.2f}  -> {mv:7s} |         {scn:8.3f}  {smm:8.2f}  -> {sv:7s}",
            flush=True,
        )

print("-" * 108, flush=True)
print(f"COUNTED VERDICT: mean-feedback BLOOMS in {mean_blooms}/9 arms;  "
      f"sample-feedback BOUNDED in {sample_bounded}/9 arms.", flush=True)
verdict = (
    "BLOOM FIXED by sample-feedback (productionized default) across all arms/seeds"
    if mean_blooms == 9 and sample_bounded == 9
    else "MIXED / not clean — inspect"
)
print(f"  => {verdict}", flush=True)
import numpy as np  # noqa: E402

td = np.array(t0_deltas)
print(f"\nT=0-neutrality (h=1 crps_all |mean-sample|): max={td.max():.4f} mean={td.mean():.4f} "
      f"median={np.median(td):.4f}  (n={len(td)} arm-targets)", flush=True)
print("  (small + sign-varying => benign D×K RNG-ordering at the pre-feedback seed step, not a "
      "feedback leak; large + systematic => possible F-B2 T=0 leak to verify)", flush=True)
