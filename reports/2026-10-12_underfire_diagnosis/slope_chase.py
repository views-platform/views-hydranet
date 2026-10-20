"""Chasing b = 1.73 — where does the dynamic-range compression come from?

D3 (2026-10-20_d3_static_channel_ceiling.md §4) measured the body ranking cells at spearman
0.687 while committing to 41% of the right spread.  Three mechanisms could do that, and they
imply different remedies:

  S  SPATIAL OVER-SMOOTHING.  The U-Net decoder's convolutions average neighbouring cells, so a
     sharp per-cell field comes out blurred.  Blurring compresses across-cell spread by
     construction.  Remedy: architecture (the decoder, or the top-skip seam).
  G  GLOBAL SHRINKAGE.  Weight decay / a timidity-rewarding objective pull every head output
     toward its mean, at every scale equally.  Remedy: weight_decay or the loss -- and the
     weight_decay arms are ALREADY pre-registered.
  R  RANK NOISE.  The model simply mis-ranks, and the attenuated slope is measurement error in
     the predictor.  Remedy: none of the above; the slope is not the defect.

DISCRIMINATOR.  Aggregate prediction and truth into k x k spatial blocks and recompute the slope.
  * Under S, blurring is a LOCAL operation: coarsening averages away the very neighbours that
    were averaged in, so b -> 1 as k grows.
  * Under G, shrinkage is scale-free: b stays put.
  * Under R, coarsening averages out the noise, so b falls toward 1 AND the correlation rises
    sharply -- distinguishable from S by watching r, not just b.

Run on the GATE, because the gate is the only head available on the full 360x720 grid
(p2_result.npz, 64,818 land cells x 13 OUT-OF-SAMPLE months 457-469).  The gate shares the whole
trunk and decoder with the body and differs only in its final conv, so a mechanism found here is
a mechanism the body is exposed to.

Also reports the gate's slope under the BODY's estimator, which the glossary 5c entry flags as
the missing like-for-like comparison.
"""
from pathlib import Path

import numpy as np
from scipy.special import logit
from scipy.stats import spearmanr

D = Path("reports/2026-10-12_underfire_diagnosis")
EPS = 1e-6
p2 = np.load(D / "p2_result.npz")
gate = p2["gate"].astype(np.float64).mean(0)      # [13, 64818] averaged over the 4 locked-dropout draws
truth = p2["truth"]                               # [13, 64818]
land = p2["land_cells"]
row = (land - 1) // 720
col = (land - 1) % 720
print("gate", gate.shape, "cells", len(land), "months", truth.shape[0])
print("observed positive rate %.6f" % truth.mean())

def slope(pred, obs, link):
    """b: regress TRUTH on PREDICTION in the link space (GLOSSARY 5b/5c convention)."""
    x, y = link(pred), link(obs)
    k = np.isfinite(x) & np.isfinite(y)
    x, y = x[k], y[k]
    b, a = np.polyfit(x, y, 1)
    return b, np.corrcoef(x, y)[0, 1], x.std(), y.std(), k.sum()

L = lambda v: logit(np.clip(v, 1e-4, 1 - 1e-4))

print("\n" + "=" * 78)
print("1. THE GATE'S SLOPE UNDER THE BODY'S ESTIMATOR (the missing like-for-like)")
print("=" * 78)
obs_rate = truth.mean(0)
pred_rate = gate.mean(0)
for lab, m in [("all land cells", np.ones(len(land), bool)),
               ("cells with any event", obs_rate > 0)]:
    b, r, sx, sy, n = slope(pred_rate[m], obs_rate[m], L)
    print("  %-22s n %6d   b %7.4f   r %+.4f   sd(pred) %.4f  sd(truth) %.4f  ratio %.4f"
          % (lab, n, b, r, sx, sy, sx / sy))
print("  (body, same estimator family, from D3 section 4:  b 1.7301  r +0.7174  sd ratio 0.4146)")

print("\n" + "=" * 78)
print("2. DISCRIMINATOR — does the slope relax as the grid is COARSENED?")
print("=" * 78)
print("   S spatial smoothing => b falls toward 1.   G global shrinkage => b flat.")
print("   R rank noise        => b falls AND r rises steeply.\n")
print("  %4s %8s %9s %9s %9s %9s" % ("k", "blocks", "b", "r", "sd(pred)", "sd(truth)"))
for k in (1, 2, 4, 8, 16, 32):
    br, bc = row // k, col // k
    key = br.astype(np.int64) * 10_000 + bc
    uq, inv = np.unique(key, return_inverse=True)
    n = len(uq)
    # mean over the cells of each block, then over the 13 months
    pm = np.zeros(n); tm = np.zeros(n); cnt = np.zeros(n)
    np.add.at(pm, inv, pred_rate)
    np.add.at(tm, inv, obs_rate)
    np.add.at(cnt, inv, 1.0)
    pm, tm = pm / cnt, tm / cnt
    keep = tm > 0                      # blocks with no event at all carry no slope information
    b, r, sx, sy, nn = slope(pm[keep], tm[keep], L)
    print("  %4d %8d %9.4f %+9.4f %9.4f %9.4f" % (k, nn, b, r, sx, sy))

print("\n" + "=" * 78)
print("3. IS THE PREDICTED FIELD SMOOTHER THAN THE TRUTH?  (rook lag-1 autocorrelation)")
print("=" * 78)
H, W = 360, 720
def to_grid(v):
    g = np.full((H, W), np.nan)
    g[row, col] = v
    return g
def lag1(g):
    out = []
    for a, b_ in ((g[:, :-1], g[:, 1:]), (g[:-1, :], g[1:, :])):
        k = np.isfinite(a) & np.isfinite(b_)
        out.append(np.corrcoef(a[k], b_[k])[0, 1])
    return out
for lab, v in [("gate (mean over 13 months)", pred_rate),
               ("observed rate (13 months)", obs_rate),
               ("gate, logit", L(pred_rate)),
               ("observed rate, logit", L(obs_rate))]:
    e, s = lag1(to_grid(v))
    print("  %-30s east-west %+.4f   north-south %+.4f" % (lab, e, s))
print("\n  a predicted field MORE autocorrelated than the truth is the signature of S.")
