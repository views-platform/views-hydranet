"""E2 - refit the calibration slope on sigmoid(mean logit) instead of mean(sigmoid).

PRE-REGISTERED DECISION TABLE (not a prediction - all three outcomes are covered, PREREG §4):
  b -> 1      C1 uncancelled
  the Jensen average was the defect
  b stays >1  the compression is in the NETWORK, not the reporting path
  b < 1       training absorbed the Jensen effect and R2 OVERSHOOTS -> R2 IS DROPPED

Reported under BOTH estimators (§7): binned n>=1000 (the published fact) and bin-free ML
(the adopted instrument, retargeted to 1.1875 on the full grid).
"""
import numpy as np

EV = ("/home/simon/Documents/scripts/views_platform/views-hydranet/reports/"
      "2026-10-16_pos_weight_arm_outcome/evidence")
E = np.array([0, .01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01])
EPS = 1e-9


def lg(x):
    return np.log(np.clip(x, EPS, 1 - EPS) / (1 - np.clip(x, EPS, 1 - EPS)))


def binned(g, y, minn=1000):
    rows = []
    for lo, hi in zip(E[:-1], E[1:]):
        m = (g >= lo) & (g < hi)
        if m.sum() < minn or not 0 < y[m].mean() < 1:
            continue
        rows.append((g[m].mean(), y[m].mean()))
    if len(rows) < 2:
        return None, 0
    x = lg(np.array([r[0] for r in rows]))
    yv = lg(np.array([r[1] for r in rows]))
    X = np.vstack([np.ones_like(x), x]).T
    return np.linalg.lstsq(X, yv, rcond=None)[0][1], len(rows)


def ml_slope(x, yy, iters=80):
    a, b = 0.0, 1.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * x)))
        W = p * (1 - p)
        r = yy - p
        H = np.array([[W.sum(), (W * x).sum()], [(W * x).sum(), (W * x * x).sum()]])
        try:
            d = np.linalg.solve(H + 1e-9 * np.eye(2), np.array([r.sum(), (r * x).sum()]))
        except np.linalg.LinAlgError:
            break
        a += d[0]
        b += d[1]
        if np.abs(d).max() < 1e-11:
            break
    return a, b


print("E2 - mean(sigmoid)  vs  sigmoid(mean logit)")
print(f"{'arm':<9} {'statistic':<22} {'binned b (bins)':>18} {'bin-free b':>12}")
for nm in ("amber", "copper"):
    z = np.load(f"{EV}/p2_result_{nm}.npz")
    G = z["gate"].astype(np.float64)
    y = z["truth"].ravel().astype(np.float64)
    gp = G.mean(0).ravel()                          # mean of sigmoid  (as published)
    gl = 1 / (1 + np.exp(-lg(G).mean(0).ravel()))    # sigmoid of mean logit
    for tag, g in (("mean(sigmoid)  [pub]", gp), ("sigmoid(mean logit)", gl)):
        bb, nb = binned(g, y)
        _, bf = ml_slope(lg(g), y)
        bbs = "no fit" if bb is None else f"{bb:.4f} ({nb})"
        print(f"{nm:<9} {tag:<22} {bbs:>18} {bf:>12.4f}")
print()
print("VERDICT per the pre-registered decision table:")
for nm in ("amber", "copper"):
    z = np.load(f"{EV}/p2_result_{nm}.npz")
    G = z["gate"].astype(np.float64)
    y = z["truth"].ravel().astype(np.float64)
    gl = 1 / (1 + np.exp(-lg(G).mean(0).ravel()))
    _, bf = ml_slope(lg(gl), y)
    v = "b -> 1 (C1 uncancelled)" if abs(bf - 1) < 0.05 else (
        "b STAYS >1 -> compression is in the NETWORK" if bf > 1 else
        "b < 1 -> R2 OVERSHOOTS, R2 IS DROPPED")
    print(f"  {nm:<8} bin-free b = {bf:.4f}  ->  {v}")
