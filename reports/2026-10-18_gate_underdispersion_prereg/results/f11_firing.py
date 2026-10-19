"""F11 - the firing-rate gate on the COMPOSED forecast under the R1 corrected gate, at h=1.

WHY THIS NEEDS NO GPU (correcting RND_PLAN section 5, which says the kestrels need a re-run):
the h=1 gate posterior survives for BOTH kestrels in a hand-rescued `gate_h1` slice (13 origins
x 3 targets x (64818, 16), every value finite per its own SLICE_MANIFEST), and the composed
draws survive complete in `*_lr.tar.zst`. Section 5's "needs a re-run" is about the
36-horizon `by_*` archive, which for amber is 9% truncated. At h=1 nothing is missing.

THE ESTIMATOR, and it is exact in expectation rather than an approximation.
Under `soft_gate`, compose_samples draws `Bernoulli(gate) * body_draw` per pass, so for cell c and
pass d the composed indicator z = 1{composed > 0} is Bernoulli with

    E[z_cd] = g_cd * p_cd           p = P(body draw > 0), untouched by a gate recalibration

The R1-corrected gate is g' = sigmoid(logit(g)/T). So

    E[ z_cd * (g'_cd / g_cd) ] = g'_cd * p_cd = E[ z'_cd ]

i.e. the observed composed indicator, REWEIGHTED by the gate ratio, is an unbiased estimator of the
corrected composed firing probability. No per-cell inversion, no clipping, no imputation, and it is
valid whether g' is above or below g. The body is never needed, which is why the truncated `by_*`
archive does not block this.

TRUTH is this session's own FRESH datafactory pull (truth_pull.py), which reproduced the Phase-1
cache exactly on all three targets.

SWEEP RULE: act_ratio depends on the free parameter T, so T is swept over its measured fold range
and the family is reported, never a single number.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

S = Path(__file__).parent
HN = Path("/home/simon/Documents/scripts/views_platform/views-hydranet")
MODELS = Path("/home/simon/Documents/scripts/views_platform/views-models/models")
sys.path.insert(0, str(HN / "reports/2026-07-29_v2_scoreboard_dossier/tools"))
import activation_metrics as am  # noqa: E402  the committed instrument, unit-tested

TARGETS = {"sb": "ged_sb_best", "ns": "ged_ns_best", "os": "ged_os_best"}
BASE_ORIGIN, N_ORIGINS = 456, 13
EPS = 1e-9

# RESULTS.md section 7: T mean / min / max across the 13 origin folds. The spread IS the sweep.
ARMS = {
    "amber": dict(
        root=MODELS / "amber_kestrel/data/generated/calibration_delivery_20261013",
        lr="predictions_calibration_20261013_010739_lr.tar.zst",
        stem="predictions_calibration_20261013_010739",
        T=(0.8674, 0.8708, 0.8747),
    ),
    "copper": dict(
        root=MODELS / "copper_kestrel/data/generated/calibration_delivery_20261016",
        lr="predictions_calibration_20261016_120217_lr.tar.zst",
        stem="predictions_calibration_20261016_120217",
        T=(0.9155, 0.9155, 0.9155),  # only the mean is on record for copper
    ),
}


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def r1(g, T):
    return 1.0 / (1.0 + np.exp(-logit(g) / T))


def truth_rates():
    df = pd.read_parquet(S / "truth_fresh_457_469.parquet").reset_index()
    out = {}
    for k, col in TARGETS.items():
        sub = df[(df.month_id >= BASE_ORIGIN + 1) & (df.month_id <= BASE_ORIGIN + N_ORIGINS)]
        out[k] = float((sub[col].to_numpy() > 0).mean())
    return out


def extract(arm, cfg):
    dest = S / f"lr_{arm}"
    if dest.exists():
        return dest
    dest.mkdir(parents=True)
    subprocess.run(["tar", "--zstd", "-xf", str(cfg["root"] / "draws" / cfg["lr"]),
                    "-C", str(dest)], check=True)
    return dest


def run(arm):
    cfg = ARMS[arm]
    lr_root = extract(arm, cfg) / cfg["stem"]
    # amber's rescue nests one level deeper than copper's; resolve rather than assume
    gate_root = cfg["root"] / "gate_h1/gate_h1"
    if not (gate_root / "origin_0").is_dir():
        gate_root = cfg["root"] / "gate_h1"
    if not (gate_root / "origin_0").is_dir():
        raise SystemExit(f"VOID: no gate_h1 slice under {cfg['root']}")
    Ts = sorted(set(cfg["T"]))
    res = {}
    for tgt in TARGETS:
        sum_z = 0.0
        sum_g = 0.0
        n = 0
        sum_zw = {T: 0.0 for T in Ts}
        gt_frac = 0.0            # share of cell-draws where g' > g
        max_w = 0.0
        raw_cubes = []
        for o in range(N_ORIGINS):
            month = BASE_ORIGIN + 1 + o
            lrd = lr_root / f"origin_{o}" / f"lr_{tgt}_best"
            z_ids = np.load(lrd / "identifiers.npz")
            y = np.load(lrd / "y_pred.npy", mmap_mode="r")
            sel = z_ids["time"] == month
            if sel.sum() == 0:
                raise SystemExit(f"VOID: origin_{o} {tgt} has no month {month}")
            units_lr = z_ids["unit"][sel]
            cube = np.asarray(y[sel], dtype=np.float32)

            gd = gate_root / f"origin_{o}" / f"by_{tgt}_best"
            g_ids = np.load(gd / "identifiers_h1.npz")
            if not np.all(g_ids["time"] == month):
                raise SystemExit(f"VOID: gate_h1 origin_{o} {tgt} is not month {month}")
            g = np.load(gd / "y_pred_h1.npy")

            # join on unit explicitly; never assume two archives share an ordering
            oz = np.argsort(units_lr, kind="stable")
            og = np.argsort(g_ids["unit"], kind="stable")
            if not np.array_equal(units_lr[oz], g_ids["unit"][og]):
                raise SystemExit(f"VOID: cell sets differ, origin_{o} {tgt}")
            cube = cube[oz]
            g = g[og]
            if cube.shape != g.shape:
                raise SystemExit(f"VOID: shape {cube.shape} vs gate {g.shape}")
            if not np.isfinite(g).all():
                raise SystemExit(f"VOID: non-finite gate, origin_{o} {tgt}")

            z = (cube > 0).astype(np.float64)
            sum_z += z.sum()
            sum_g += g.astype(np.float64).sum()
            n += z.size
            for T in Ts:
                w = r1(g.astype(np.float64), T) / np.clip(g.astype(np.float64), EPS, None)
                sum_zw[T] += float((z * w).sum())
                if T == cfg["T"][1]:
                    gt_frac += float((w > 1.0).sum())
                    max_w = max(max_w, float(w.max()))
            raw_cubes.append(cube)
        res[tgt] = dict(
            n=n, act_pred_raw=sum_z / n, mean_gate=sum_g / n,
            p_body_implied=sum_z / sum_g,
            act_pred_r1={str(T): sum_zw[T] / n for T in Ts},
            share_gprime_above_g=gt_frac / n, max_weight=max_w,
            # cross-check against the committed instrument on the same raw cube
            act_pred_instrument=am.activation_frequency(
                np.concatenate(raw_cubes), np.zeros(1))["act_pred"],
        )
        del raw_cubes
    return res


if __name__ == "__main__":
    at = truth_rates()
    out = {"act_true_fresh": at}
    for arm in sys.argv[1:] or ["amber"]:
        out[arm] = run(arm)
    print(json.dumps(out, indent=2))
    (S / f"f11_{'_'.join(sys.argv[1:] or ['amber'])}.json").write_text(json.dumps(out, indent=2))
