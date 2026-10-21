"""D7 — is §9.2's amplitude claim an ADMISSION-RATE artifact?

§9.2 read `sd(log truth)` growing 1.946 -> 2.987 across k = 1..32 while `sd(log mu)` stayed flat,
and concluded the body "has no large-scale amplitude structure". But the coarsening in
`d4_mu_on_the_grid.py` keeps only blocks with `tm > 0` -- blocks containing at least one event --
and a bigger block is more likely to contain one. So the admitted POPULATION changes with k, and a
trend in sd across k may be a trend in which blocks are being measured.

This script quantifies the confound and then removes it by recomputing the sd trend on a FIXED set
of blocks. It also tabulates the model/truth lag ratio so D6-B's "no knee" reading is a number
rather than an eyeball.

Reads d4_result.npz. No forward pass. Free.
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, TARGETS, grid_rowcol, pull  # noqa: E402

SP = Path(__file__).parent
EPS = 1e-6
H, W = 360, 720
IS_LO, IS_HI = 421, 456
t0 = time.time()

z = np.load(SP / "d4_result.npz")
land_cells, kept = z["land_cells"], z["months"]
lr, lc = grid_rowcol(land_cells)
row, col = lr - 1, lc - 1
MU, GA = z["mu"].mean(0), z["gate"].mean(0)
j = np.array([list(kept).index(m) for m in range(IS_LO, IS_HI + 1)])
mu_w = MU[j].mean(0).astype(np.float64)

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= IS_LO) & (df.month_id <= IS_HI)]
y_w = w.groupby("priogrid_id")[TARGETS[SB]].mean().reindex(land_cells).fillna(0.0).to_numpy()

print("=" * 78)
print("D7-A — HOW MUCH DOES THE ADMITTED POPULATION CHANGE WITH k?")
print("=" * 78)
print("   k   blocks w/ land   blocks w/ event   admitted %   sd(log mu)  sd(log truth)")
rows = []
for k in (1, 2, 4, 8, 16, 32):
    bid = (row // k) * (W // k + 1) + (col // k)
    uniq, inv = np.unique(bid, return_inverse=True)
    pm = np.zeros(len(uniq)); tm = np.zeros(len(uniq)); cnt = np.zeros(len(uniq))
    np.add.at(pm, inv, mu_w); np.add.at(tm, inv, y_w); np.add.at(cnt, inv, 1.0)
    pm, tm = pm / cnt, tm / cnt
    keep = tm > 0
    sx = np.log(pm[keep] + EPS).std()
    sy = np.log(tm[keep] + EPS).std()
    rows.append((k, len(uniq), keep.sum(), 100 * keep.mean(), sx, sy))
    print("  %3d   %13d   %15d   %9.1f%%   %10.4f   %12.4f"
          % (k, len(uniq), keep.sum(), 100 * keep.mean(), sx, sy))

print("\n  => the admitted fraction moves from %.1f%% to %.1f%% (a factor of %.1f)."
      % (rows[0][3], rows[-1][3], rows[-1][3] / rows[0][3]))
print("     §9.2's sd trend across k is therefore computed on DIFFERENT POPULATIONS at each k.")

print("\n" + "=" * 78)
print("D7-B — THE SAME TREND ON A FIXED POPULATION")
print("=" * 78)
print("  Fix the cell set ONCE (cells with any event at k=1), aggregate only those cells into")
print("  k x k blocks, and admit every block that contains one. The population is then nested.")
print("\n   k   blocks   sd(log mu)  sd(log truth)   ratio mu/truth")
base_keep = y_w > 0
for k in (1, 2, 4, 8, 16, 32):
    bid = (row // k) * (W // k + 1) + (col // k)
    b = bid[base_keep]
    uniq, inv = np.unique(b, return_inverse=True)
    pm = np.zeros(len(uniq)); tm = np.zeros(len(uniq)); cnt = np.zeros(len(uniq))
    np.add.at(pm, inv, mu_w[base_keep]); np.add.at(tm, inv, y_w[base_keep])
    np.add.at(cnt, inv, 1.0)
    pm, tm = pm / cnt, tm / cnt
    sx, sy = np.log(pm + EPS).std(), np.log(tm + EPS).std()
    print("  %3d   %6d   %10.4f   %12.4f   %14.4f" % (k, len(uniq), sx, sy, sx / sy))

print("\n" + "=" * 78)
print("D7-C — THE MODEL/TRUTH LAG RATIO.  A fixed blur radius => the ratio returns to 1")
print("       beyond that radius.  A scale-free cause => the ratio does not.")
print("=" * 78)


def lag(v, L):
    g = np.full((H, W), np.nan)
    g[row, col] = v
    o = []
    for a, b in ((g[:, :-L], g[:, L:]), (g[:-L, :], g[L:, :])):
        kk = np.isfinite(a) & np.isfinite(b)
        o.append(np.corrcoef(a[kk], b[kk])[0, 1])
    return np.array(o)


lmu = np.log(mu_w + EPS)
gt_w = np.clip(GA[j].mean(0).astype(np.float64), EPS, 1 - EPS)
lgt = np.log(gt_w / (1 - gt_w))
ltr = np.log(df[df.month_id <= 456].groupby("priogrid_id")[TARGETS[SB]].mean()
             .reindex(land_cells).fillna(0.0).to_numpy() + EPS)
print("\n   lag    mu/truth E-W   mu/truth N-S   gate/truth E-W   gate/truth N-S")
for L in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
    t = lag(ltr, L)
    m = lag(lmu, L) / t
    g = lag(lgt, L) / t
    print("   %3d    %12.2f   %12.2f   %14.2f   %14.2f" % (L, m[0], m[1], g[0], g[1]))
print("\n  D5-B measured the ONE-STEP impulse reach: below 10%% of peak at 10 cells, exactly 0")
print("  at >=16. ⚠️ But the window is 49 RECURRENT steps, so the effective multi-step reach is")
print("  far larger than one step's -- the absence of a knee near 10-16 rules out a ONE-STEP")
print("  receptive-field story, NOT a recurrent one.")

print("\n" + "=" * 78)
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
