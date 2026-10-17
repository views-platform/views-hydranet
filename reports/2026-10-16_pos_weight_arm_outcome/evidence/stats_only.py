"""Apply P2's statistics block, copied verbatim from p2_gate_calibration.py, to a stored npz.

Purpose: does the statistics path reproduce the published 0.0054 / 1.0365 from the arrays the
2026-10-12 run saved? If yes, the binning/branch-test code is validated independently of torch.
"""
import sys
import numpy as np

z = np.load(sys.argv[1], allow_pickle=True)
gate, truth = z["gate"], z["truth"]
gate_mean = gate.mean(0)
s = gate_mean.ravel().astype(np.float64)
y = truth.ravel()

def implied_w(sig, p):
    sig = np.clip(sig, 1e-9, 1 - 1e-9)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return sig * (1 - p) / (p * (1 - sig))

edges = np.array([0.0, .01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01])
rows = []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (s >= lo) & (s < hi)
    if m.sum() == 0:
        continue
    sg, p = s[m].mean(), y[m].mean()
    w = implied_w(sg, p) if 0 < p < 1 else np.nan
    rows.append((lo, hi, int(m.sum()), sg, p, 2 * p / (1 + p), w))
big = [r for r in rows if r[2] >= 1000 and 0 < r[4] < 1]
obs = np.array([r[4] for r in big]); sg = np.array([r[3] for r in big])
err1 = np.median(np.abs(sg - obs))
err2 = np.median(np.abs(sg - np.array([r[5] for r in big])))
w_med = float(np.median([r[6] for r in big]))
print(f"  {len(s):,} cell-months · positive rate {y.mean():.6f} · bins n>=1000: {len(big)}")
print(f"  median |gate - p|        {err1:.4f}")
print(f"  median |gate - 2p/(1+p)| {err2:.4f}")
print(f"  median w_implied         {w_med:.4f}")
