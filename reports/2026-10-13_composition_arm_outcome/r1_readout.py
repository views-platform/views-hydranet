"""R1 — the pre-registered readout of the composition arm, h=1.

Reads the arm's OWN emitted posterior, never a reconstruction of it. Judges exactly the
statistics `reports/2026-10-12_composition_arm_prereg.md` committed to, against references
recomputed from disk rather than quoted.

⛔ ONE statistic per column, never mixed across sides (the defect that generated the phantom B7).
"""
import io
import json
import hashlib
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ARM = Path("/home/simon/Documents/scripts/views_platform/views-models/models/amber_kestrel"
           "/data/generated/calibration_delivery_20261013")
DIAG = Path("reports/2026-10-12_underfire_diagnosis")
SCRATCH = ("/tmp/claude-1000/-home-simon-Documents-scripts-views-platform-views-hydranet/"
           "f7a7ca8c-4eb9-45a0-8923-271b8be2a39c/scratchpad")
BASE_ORIGIN, N_ORIGINS = 456, 13

# ---------- F6 / F6c: the artefact describes itself, and the sha is a value READ ----------
prov = dict(l.split(":", 1) for l in
            (x for x in (ARM / "artifact/PROVENANCE").read_text().splitlines() if ":" in x))
prov = {k.strip(): v.strip() for k, v in prov.items()}
pt = ARM / "artifact" / prov["artefact"]
computed = hashlib.sha256(pt.read_bytes()).hexdigest()
cfg = json.loads((pt.with_suffix(".pt.config.json")).read_text()) if pt.with_suffix(
    ".pt.config.json").exists() else json.loads(Path(str(pt) + ".config.json").read_text())

print("=" * 90)
print("R1 — composition arm readout, h=1 (prereg 2026-10-12, amendments 1-3)")
print("=" * 90)
print(f"  artefact      {prov['artefact']}")
print(f"  sha256        {computed[:16]}…  recorded {prov['sha256'][:16]}…  "
      f"{'MATCH' if computed == prov['sha256'] else '⛔ MISMATCH -> F6 VOID'}")
print(f"  lessons       {prov['lessons']}   mode {prov['mode']}   region {prov['region'].split(' ')[0]}")
print(f"  git           {prov['git']}")
# ⛔ The .pt.config.json sidecar carries only 12 summary keys and does NOT record
#    forecast_composition. Absence there is a property of the sidecar, not of the run — so the
#    composition is read from the config AT THE RECORDED SHA, which is F6c's own mechanism
#    (a value read, not a promise), and then CHECKED against the emitted draws below.
print(f"  sidecar says  output_distribution={cfg.get('output_distribution')!r} "
      f"(12 keys; it does not record forecast_composition)")
import subprocess  # noqa: E402
MODELS = "/home/simon/Documents/scripts/views_platform/views-models"
src = subprocess.run(["git", "-C", MODELS, "show",
                      f"{prov['git']}:models/amber_kestrel/configs/config_hyperparameters.py"],
                     capture_output=True, text=True)
if src.returncode != 0:
    raise SystemExit(f"⛔ F6c VOID: sha {prov['git']} does not resolve in views-models.")
decl = {k: (k + "': '") for k in ()}
got = {}
for key in ("output_distribution", "forecast_composition"):
    hits = [l.split(":", 1)[1].strip().strip(",").strip("'\"")
            for l in src.stdout.splitlines() if f"'{key}'" in l and ":" in l.split("#")[0]]
    got[key] = hits[0] if hits else "<absent>"
has_tau = any("'gate_threshold'" in l.split("#")[0] for l in src.stdout.splitlines())
print(f"  config @ {prov['git']}  output_distribution={got['output_distribution']!r} "
      f"forecast_composition={got['forecast_composition']!r} "
      f"gate_threshold={'present' if has_tau else '<absent>'}")
if cfg.get("output_distribution") != "truncated_nb" or got["forecast_composition"] != "soft_gate":
    raise SystemExit("⛔ VOID: the pod did not train/emit the arm's composition.")
if has_tau:
    raise SystemExit("⛔ VOID: gate_threshold present under soft_gate.")

# ---------- the control's F1 reference, recomputed from disk ----------
z = np.load(DIAG / "p2_result.npz")
land, truth = z["land_cells"], z["truth"]          # (64818,) ; (13,64818) sb>0 at months 457..469
ctrl_gate = z["gate"].mean(0)                      # posterior-mean gate over D=4
ctrl_ap = average_precision_score(truth.ravel(), ctrl_gate.ravel())
ctrl_months = np.array([average_precision_score(truth[m], ctrl_gate[m]) for m in range(N_ORIGINS)])

# ---------- the arm's gate posterior at h=1, read from its own archive ----------
order = {u: i for i, u in enumerate(land)}
arm_gate = np.zeros_like(ctrl_gate)
man = json.loads((ARM / "gate_h1/gate_h1/SLICE_MANIFEST.json").read_text())
for n in range(N_ORIGINS):
    d = ARM / f"gate_h1/gate_h1/origin_{n}/by_sb_best"
    g = np.load(d / "y_pred_h1.npy")
    ids = np.load(d / "identifiers_h1.npz")
    unit = np.asarray(ids["unit"]).ravel()
    if man[f"origin_{n}"]["by_sb_best"]["h1_month"] != BASE_ORIGIN + 1 + n:
        raise SystemExit(f"⛔ VOID: origin_{n} h1 month mismatch.")
    if len(unit) != len(land) or set(unit) != set(land):
        raise SystemExit(f"⛔ VOID: origin_{n} cell set differs from the control's land set.")
    arm_gate[n] = g.mean(1)[np.argsort([order[u] for u in unit])]   # posterior mean over 16 draws
if not np.isfinite(arm_gate).all():
    raise SystemExit("⛔ VOID: non-finite gate.")

arm_ap = average_precision_score(truth.ravel(), arm_gate.ravel())
arm_months = np.array([average_precision_score(truth[m], arm_gate[m]) for m in range(N_ORIGINS)])

print()
print("-" * 90)
print("F1 / P-3 — POOLED GATE AP, all 64,818 land cells x 13 h=1 months (842,634 cell-months)")
print("-" * 90)
print(f"  control  {ctrl_ap:.4f}   per-month sd {ctrl_months.std(ddof=1):.4f}")
print(f"  arm      {arm_ap:.4f}   per-month sd {arm_months.std(ddof=1):.4f}")
print(f"  delta    {arm_ap - ctrl_ap:+.4f}")
d = arm_months - ctrl_months
se = d.std(ddof=1) / np.sqrt(N_ORIGINS)
print(f"  paired by month: mean delta {d.mean():+.4f}  sd {d.std(ddof=1):.4f}  "
      f"95% CI [{d.mean() - 1.96 * se:+.4f}, {d.mean() + 1.96 * se:+.4f}]  "
      f"({int((d > 0).sum())}/13 months favour the arm)")
print(f"  F1 threshold {ctrl_ap - 0.15:.4f}  ->  "
      f"{'⛔ F1 FIRES (KILL)' if arm_ap < ctrl_ap - 0.15 else 'F1 does not fire'}")
print(f"  P-3 (|delta| <= 0.05)  ->  {'HELD' if abs(arm_ap - ctrl_ap) <= 0.05 else 'MISSED'}")

# ---------- P-1 / P-2 / F2 — the emitted forecast on the 100-cell set ----------
cells = pd.read_parquet(f"{SCRATCH}/t1_fits.parquet").priogrid_id.to_numpy()
import zstandard  # noqa: E402
with open(ARM / "draws/predictions_calibration_20261013_010739_lr.tar.zst", "rb") as fh:
    raw = zstandard.ZstdDecompressor().stream_reader(fh).read()
emit = {}
with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
    root = tf.getnames()[0].split("/")[0]
    o100 = {u: i for i, u in enumerate(cells)}
    for n in range(N_ORIGINS):
        dd = f"{root}/origin_{n}/lr_sb_best"
        y = np.load(io.BytesIO(tf.extractfile(f"{dd}/y_pred.npy").read()))
        ids = np.load(io.BytesIO(tf.extractfile(f"{dd}/identifiers.npz").read()))
        t_, u_ = np.asarray(ids["time"]).ravel(), np.asarray(ids["unit"]).ravel()
        h1 = t_.min()
        if h1 != BASE_ORIGIN + 1 + n:
            raise SystemExit(f"⛔ VOID: lr origin_{n} h1 is {h1}.")
        sel = (t_ == h1) & np.isin(u_, cells)
        rows, units = y[sel], u_[sel]
        if len(rows) != len(cells):
            raise SystemExit(f"⛔ VOID: lr origin_{n} has {len(rows)} of {len(cells)} cells.")
        emit[n] = rows[np.argsort([o100[u] for u in units])]
del raw

stack = np.concatenate([emit[n] for n in range(N_ORIGINS)], axis=0)    # (1300, 16)
pooled_pos = float((stack > 0).mean())
median_pos = float(np.median((stack > 0).mean(axis=1)))
obs_idx = np.isin(land, cells)
obs = float(truth[:, obs_idx].mean())

# ---------- the composition, MEASURED rather than declared ----------
# truncated_nb has P(body = 0) = 0 by construction. So under `self_zeroed` the emitted draws
# would be ~100% positive; under `soft_gate` the emitted positive fraction IS the gate. This
# separates the two hypotheses by ~0.3 and does not depend on any config file.
print()
print("-" * 90)
print("COMPOSITION, MEASURED on the emitted draws (not read from a config)")
print("-" * 90)
gsub = arm_gate[:, obs_idx].ravel()
print(f"  emitted positive fraction, 100-cell set   {pooled_pos:.4f}")
print(f"  mean gate on the same cell-months         {gsub.mean():.4f}")
print(f"  self_zeroed would give                    1.0000  (P(trunc body = 0) = 0)")
if pooled_pos > 0.97:
    raise SystemExit("⛔ VOID: emitted draws are ~all positive — composed as self_zeroed, not soft_gate.")
print(f"  -> consistent with soft_gate, |emitted - gate| = {abs(pooled_pos - gsub.mean()):.4f}")

print()
print("-" * 90)
print("P-1 / P-2 / F2 — EMITTED positive fraction, 100-cell set, h=1, pooled over 13 origins")
print("-" * 90)
print(f"  {'':28}{'pooled mean':>14}{'median over cells':>20}")
print(f"  {'arm (this run)':28}{pooled_pos:>14.4f}{median_pos:>20.4f}")
print(f"  {'control (archive, P1)':28}{0.3791:>14.4f}{'—':>20}")
print(f"  {'observed':28}{obs:>14.4f}{'—':>20}")
print(f"  P-1 interval [0.62, 0.76] on the pooled mean  ->  "
      f"{'HELD' if 0.62 <= pooled_pos <= 0.76 else 'MISSED'}")
print(f"  P-2 composed P(Y=0) ~ 0.31  ->  measured {1 - pooled_pos:.4f}")
print(f"  F2 (> 0.85)  ->  {'⛔ F2 FIRES' if pooled_pos > 0.85 else 'F2 does not fire'}")
gap_closed = (pooled_pos - 0.3791) / (obs - 0.3791)
print(f"  gap to observed closed: {100 * gap_closed:.1f}%  "
      f"(control {0.3791:.4f} -> arm {pooled_pos:.4f} -> observed {obs:.4f})")

np.savez(Path("reports/2026-10-13_composition_arm_outcome/r1_result.npz"),
         arm_gate=arm_gate, ctrl_gate=ctrl_gate, truth=truth, land=land,
         arm_ap=arm_ap, ctrl_ap=ctrl_ap, arm_months=arm_months, ctrl_months=ctrl_months,
         emitted=stack, cells=cells, sha=computed)
print("\nwrote r1_result.npz")
