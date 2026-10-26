"""Does the OUTPUT keep moving while the LOSS stops moving? Measured on one fixed window.

Simon's observation: "the more the loss plateaus, the more I run it, the different kind of the
output still gets." The `library` seat's discriminator (DAmour2020 + Maddox2020): movement in a
direction the loss does not constrain is EXPECTED in any overparameterised model, and is a defect
only when it is functionally meaningful on behaviour the loss is not pinning down. So the
comparison must be made on FORECAST steps, not only on reconstruction.

⛔ A biopsy pair is only comparable if the two files carry the SAME truth. 900 of the arm's biopsy
files hold 858 distinct truth arrays; this reads the largest group that shares one, byte-identical.
Any other pairing is two different patches of the world and measures the curriculum, not the model.

Each snapshot carries 6 steps: 3 conditioning ("_in") and 3 forecast ("_out"). They are reported
SEPARATELY. ⚠️ Whether the training loss covers all six steps is NOT established here, so no claim
is made that the "_out" steps are unsupervised — only that they are the forecast.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ARMD = Path("/home/simon/views-diagnostics/hf_arm_20261026")
IN, OUT = slice(0, 3), slice(3, 6)


def lesson_of(p):
    return int(p.name.split("lesson_")[1].split("_win")[0])


groups = {}
for p in ARMD.rglob("*forensic_(lesson_*_win_*).npz"):
    groups.setdefault(hash(np.load(p)["y_reg"].tobytes()), []).append(p)
files = sorted(max(groups.values(), key=len), key=lesson_of)
lessons = [lesson_of(p) for p in files]

print("=" * 100)
print("OUTPUT DRIFT ON A FIXED WINDOW — arm, one byte-identical truth array")
print("=" * 100)
print(f"  {len(files)} snapshots, lessons {lessons[0]}..{lessons[-1]}")
z0 = np.load(files[0])
print(f"  each: {z0['y_hat_reg'].shape} = (steps, H, W, targets); "
      f"3 conditioning + 3 forecast; truth identical in all {len(files)} by construction\n")

fields = {"body": [], "gate": [], "forecast": []}
key = {"body": "y_hat_reg", "gate": "y_hat_cls", "forecast": "y_hat_forecast"}
for p in files:
    z = np.load(p)
    for k, c in key.items():
        fields[k].append(z[c].astype(np.float64))
for k in fields:
    fields[k] = np.stack(fields[k])

t = pd.read_csv(list(ARMD.rglob("trajectory.csv"))[0]).set_index("lesson")
body_c = [c for c in t.columns if c.startswith("loss/body/")]
gate_c = [c for c in t.columns if c.startswith("loss/gate/")]
t["total"] = 0.5 * t[body_c].sum(1) + 1.0 * t[gate_c].sum(1)
#: the trajectory is noisy (sd ~15 on a level of ~45), so the loss at a lesson is a 9-lesson
#: centred mean - comparing two single lessons would measure the noise, not the trend.
sm = t["total"].rolling(9, center=True, min_periods=1).mean()

print(f"  {'lesson':>7s} {'loss(sm9)':>10s} | "
      + " ".join(f"{k + ' ' + w:>14s}" for k in ("body", "forecast") for w in ("IN", "OUT")))
for i, ls in enumerate(lessons):
    row = [f"{fields[k][i][s].mean():14.4f}" for k in ("body", "forecast") for s in (IN, OUT)]
    print(f"  {ls:7d} {sm.get(ls, float('nan')):10.3f} | " + " ".join(row))

print("\n--- TRAVEL PER LESSON between consecutive snapshots.")
print("--- ⛔ The snapshot gaps are IRREGULAR (1 to 20 lessons), so raw step-to-step travel is")
print("--- not comparable across pairs and every figure below is divided by the gap.")
print("--- ⛔ The forecast field's mean is near zero (0.0000-0.1049), so a %-of-mean denominator")
print("--- explodes. CORRELATION between consecutive snapshots is reported instead: it is")
print("--- scale-free, and 1.0 means the field did not change shape at all.")
hdr = ("dles", "loss %/les", "body r", "fcst r", "body 1-r/les", "fcst 1-r/les")
print(f"  {'lessons':>12s} " + " ".join(f"{h:>13s}" for h in hdr))
rows = []
for i in range(1, len(files)):
    la, lb = lessons[i - 1], lessons[i]
    g = lb - la
    dl = abs(sm.get(lb, np.nan) - sm.get(la, np.nan)) / sm.get(la, np.nan) * 100 / g
    out = [g, dl]
    for k in ("body", "forecast"):
        a, b = fields[k][i - 1].ravel(), fields[k][i].ravel()
        out.append(float(np.corrcoef(a, b)[0, 1]))
    out += [(1 - out[2]) / g, (1 - out[3]) / g]
    rows.append((la, lb, out))
    print(f"  {la:5d}->{lb:<5d} " + " ".join(f"{v:13.4f}" for v in out))

print("\n" + "=" * 100)
print("  Per-lesson means. 'loss %/les' is how much the smoothed total moves per lesson;")
print("  '1-r/les' is how much of the output field's SHAPE changes per lesson (0 = frozen).")
for lab, lo, hi in (("lessons  94-149", 0, 150), ("lessons 200-299", 200, 999)):
    grp = [r[2] for r in rows if lo <= r[0] < hi]
    if grp:
        m = np.nanmean(np.array(grp), 0)
        print(f"  {lab} (n={len(grp):2d}, mean gap {m[0]:4.1f}):  loss {m[1]:6.3f}%/lesson   "
              f"body r {m[2]:.4f}  fcst r {m[3]:.4f}   "
              f"body shape-change {m[4]:.4f}/les  fcst {m[5]:.4f}/les")
print("\n  ⚠️ body IN and body OUT are near-identical in every row above: this arm's body field is")
print("     almost constant across the 6 steps, so the conditioning/forecast split does NOT")
print("     discriminate on this vehicle. The library seat's test needs a vehicle where it does.")

print("\n" + "=" * 100)
print("CONTROL 1 — gap-matched. Late snapshots sit further apart (6.5 vs 4.8 lessons), which")
print("would lower their correlation on its own. Restricted to pairs 4-6 lessons apart:")
for lab, lo, hi in (("lessons  94-149", 0, 150), ("lessons 200-299", 200, 999)):
    g = [r[2] for r in rows if lo <= r[0] < hi and 4 <= r[2][0] <= 6]
    if g:
        m = np.nanmean(np.array(g), 0)
        print(f"  {lab} (n={len(g)}, gaps 4-6):  body r {m[2]:+.4f}   forecast r {m[3]:+.4f}")

print("\nCONTROL 2 — is the loss actually plateaued? NET progress vs GROSS movement,")
print("smoothed total, lessons 200-299:")
w = sm.loc[200:299].dropna()
net = (w.iloc[-1] - w.iloc[0]) / w.iloc[0] * 100 / (w.index[-1] - w.index[0])
gross = (w.diff().abs() / w.shift()).dropna().mean() * 100
print(f"  net   {net:+.4f}% per lesson  ({w.iloc[0]:.2f} -> {w.iloc[-1]:.2f} over "
      f"{w.index[-1] - w.index[0]} lessons)")
print(f"  gross {gross:+.4f}% per lesson of absolute movement")
print(f"  => {100 * (1 - abs(net) / gross):.1f}% of the loss's movement is back-and-forth, "
      "not progress.")
print("\n⛔ NOT RULED OUT: inference runs MC-dropout ON (ADR-059), so two snapshots differ by")
print("   dropout noise as well as by weights, and no noise floor for THIS quantity exists.")
print("   Evidence it is not the whole story: two pairs one lesson apart give body r 0.9701")
print("   (94->95) and 0.2155 (166->167). A constant noise floor cannot produce both.")
