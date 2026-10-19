"""P-3 / P-4 / P-5 / P-6 for the pos_weight arm, both sides measured the same way.

Methods follow reports/2026-10-16_pos_weight_prereg.md §4 and reuse the shapes r1_readout.py
established for the composition arm:
  P-3  gate AP, pooled and paired by month, from each arm's own h=1 gate slice
  P-4  CRPS from each arm's own metricframes (the pipeline scorer's own output, 'mean' group)
  P-5  emitted positive fraction, 100-cell set, h=1, pooled over 13 origins, from the lr draws
  P-6  act_ratio at h18 = P(emitted>0 over draws) / P(truth>0), all land cells, pooled
⛔ One statistic per column; never mixed across sides.
"""
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HN = Path("/home/simon/Documents/scripts/views_platform/views-hydranet")
VM = Path("/home/simon/Documents/scripts/views_platform/views-models")
DIAG = HN / "reports/2026-10-12_underfire_diagnosis"
CELLS100 = Path("/tmp/claude-1000/-home-simon-Documents-scripts-views-platform-views-hydranet/"
                "f7a7ca8c-4eb9-45a0-8923-271b8be2a39c/scratchpad/t1_fits.parquet")
BASE_ORIGIN, N_ORIGINS = 456, 13

ARMS = {
    "amber (control, w=2)": dict(
        root=VM / "models/amber_kestrel/data/generated/calibration_delivery_20261013",
        gate="gate_h1/gate_h1", lr="predictions_calibration_20261013_010739_lr.tar.zst",
        stem="predictions_calibration_20261013_010739"),
    "copper (arm, w=1)": dict(
        root=VM / "models/copper_kestrel/data/generated/calibration_delivery_20261016",
        gate="gate_h1", lr="predictions_calibration_20261016_120217_lr.tar.zst",
        stem="predictions_calibration_20261016_120217"),
}

z = np.load(DIAG / "p2_result.npz")
land, truth = z["land_cells"], z["truth"]          # truth: (13, 64818) sb>0 at months 457..469
order = {u: i for i, u in enumerate(land)}
cells100 = pd.read_parquet(CELLS100).priogrid_id.to_numpy()
sys.path.insert(0, str(DIAG))
import vehicle  # noqa: E402
df = vehicle.pull((121, 504), DIAG / "pull_land_all3_121_504.parquet")
truth_by_month = {m: g.set_index("priogrid_id")["ged_sb_best"] for m, g in df.groupby("month_id")}

out = {}
for name, A in ARMS.items():
    root, res = A["root"], {}

    # ---- P-3: gate AP from the arm's own h=1 slice
    g_all = np.zeros_like(truth, dtype=np.float64)
    man = json.loads((root / A["gate"] / "SLICE_MANIFEST.json").read_text())
    for n in range(N_ORIGINS):
        d = root / A["gate"] / f"origin_{n}" / "by_sb_best"
        g = np.load(d / "y_pred_h1.npy")
        unit = np.asarray(np.load(d / "identifiers_h1.npz")["unit"]).ravel()
        assert man[f"origin_{n}"]["by_sb_best"]["h1_month"] == BASE_ORIGIN + 1 + n
        assert set(unit) == set(land)
        g_all[n] = g.mean(1)[np.argsort([order[u] for u in unit])]
    assert np.isfinite(g_all).all()
    res["ap_pooled"] = average_precision_score(truth.ravel(), g_all.ravel())
    res["ap_months"] = np.array([average_precision_score(truth[m], g_all[m])
                                 for m in range(N_ORIGINS)])

    # ---- P-4: CRPS from the arm's own metricframes
    crps = {}
    for tgt in ("lr_sb_best", "lr_ns_best", "lr_os_best"):
        d = next(p for p in (root / "metricframes").rglob(f"metricframe_{tgt}") if p.is_dir())
        v = np.load(d / "values.npy").ravel()
        ids = np.load(d / "identifiers.npz", allow_pickle=True)
        sel = (ids["metric"] == "CRPS") & (ids["group_id"] == "mean")
        crps[tgt] = float(v[sel][0])
    res["crps"] = crps

    # ---- P-5 and P-6 from the emitted lr draws
    arch = root / "draws" / A["lr"]
    pos_h1, act_h18_pred, act_h18_true = [], [], []
    for n in range(N_ORIGINS):
        member = f"{A['stem']}/origin_{n}/lr_sb_best"
        subprocess.run(["tar", "--use-compress-program=unzstd", "-xf", str(arch),
                        f"{member}/y_pred.npy", f"{member}/identifiers.npz"],
                       cwd="/tmp/lrwork", check=True)
        p = Path("/tmp/lrwork") / member
        y = np.load(p / "y_pred.npy")                       # (64818*36, 16) emitted draws
        ids = np.load(p / "identifiers.npz", allow_pickle=True)
        t, u = np.asarray(ids["time"]).ravel(), np.asarray(ids["unit"]).ravel()
        h1m = BASE_ORIGIN + 1 + n
        m1 = (t == h1m) & np.isin(u, cells100)
        pos_h1.append((y[m1] > 0).mean())                   # emitted positive fraction, 100 cells
        h18m = h1m + 17                                     # the 18th forecast month
        m18 = (t == h18m)
        act_h18_pred.append((y[m18] > 0).mean())
        tv = truth_by_month[h18m].reindex(u[m18]).to_numpy()
        act_h18_true.append((tv > 0).mean())
        for f in (p / "y_pred.npy", p / "identifiers.npz"):
            f.unlink()
    res["pos_h1_100"] = float(np.mean(pos_h1))
    res["act_pred_h18"] = float(np.mean(act_h18_pred))
    res["act_true_h18"] = float(np.mean(act_h18_true))
    res["act_ratio_h18"] = res["act_pred_h18"] / res["act_true_h18"]
    out[name] = res
    print(f"[{name}] done", flush=True)

c, a = out["copper (arm, w=1)"], out["amber (control, w=2)"]
print("\n" + "=" * 92)
print("P-3 — GATE AP, 64,818 land cells x 13 h=1 months (842,634 cell-months)")
print("=" * 92)
print(f"  control (amber)  {a['ap_pooled']:.4f}   per-month sd {a['ap_months'].std(ddof=1):.4f}")
print(f"  arm     (copper) {c['ap_pooled']:.4f}   per-month sd {c['ap_months'].std(ddof=1):.4f}")
print(f"  pooled delta     {c['ap_pooled'] - a['ap_pooled']:+.4f}")
d = c["ap_months"] - a["ap_months"]
se = d.std(ddof=1) / np.sqrt(N_ORIGINS)
print(f"  PAIRED BY MONTH  mean delta {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  "
      f"95% CI [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]  "
      f"({int((d > 0).sum())}/13 favour the arm)")
print(f"  P-3 (unchanged within +/-0.010): {'HELD' if abs(d.mean()) <= 0.010 else 'MISSED'}")
print(f"  F2 (paired AP below -0.010)    : {'FIRES (KILL)' if d.mean() < -0.010 else 'does not fire'}")

print("\n" + "=" * 92)
print("P-4 — CRPS, each arm's own metricframes, 'mean' group")
print("=" * 92)
for tgt in ("lr_sb_best", "lr_ns_best", "lr_os_best"):
    ca, cc = a["crps"][tgt], c["crps"][tgt]
    print(f"  {tgt:<12} control {ca:.6f}   arm {cc:.6f}   "
          f"change {100 * (cc - ca) / ca:+.2f}%  ({'worse' if cc > ca else 'better'})")
worst = max(100 * (c['crps'][t] - a['crps'][t]) / a['crps'][t] for t in a['crps'])
print(f"  P-4 (does not worsen)        : {'HELD' if worst <= 0 else 'MISSED'}  (worst {worst:+.2f}%)")
print(f"  F3 (worsens by > 5%)         : {'FIRES' if worst > 5 else 'does not fire'}")

print("\n" + "=" * 92)
print("P-5 — EMITTED positive fraction, 100-cell set, h=1, pooled over 13 origins")
print("=" * 92)
print(f"  control (amber)  {a['pos_h1_100']:.4f}")
print(f"  arm     (copper) {c['pos_h1_100']:.4f}")
print(f"  P-5 (stays in [0.55, 0.85]): {'HELD' if 0.55 <= c['pos_h1_100'] <= 0.85 else 'MISSED'}")
print(f"  F5 (leaves the band)       : {'FIRES (VOID)' if not (0.55 <= c['pos_h1_100'] <= 0.85) else 'does not fire'}")

print("\n" + "=" * 92)
print("P-6 — act_ratio at h18 = P(emitted>0 over draws) / P(truth>0), all land cells")
print("=" * 92)
print(f"  control (amber)  act_pred {a['act_pred_h18']:.6f}  act_true {a['act_true_h18']:.6f}  "
      f"ratio {a['act_ratio_h18']:.4f}")
print(f"  arm     (copper) act_pred {c['act_pred_h18']:.6f}  act_true {c['act_true_h18']:.6f}  "
      f"ratio {c['act_ratio_h18']:.4f}")
print(f"  P-6 (<= control): {'HELD' if c['act_ratio_h18'] <= a['act_ratio_h18'] else 'MISSED'}")
print(f"  F4 (> control)  : {'FIRES (a finding about the knob)' if c['act_ratio_h18'] > a['act_ratio_h18'] else 'does not fire'}")
np.savez(Path(sys.argv[0]).parent / "readout_result.npz",
         **{f"{k}_{m}": v for k, r in out.items() for m, v in r.items() if not isinstance(v, dict)})
