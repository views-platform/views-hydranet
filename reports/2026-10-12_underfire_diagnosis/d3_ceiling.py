"""D3 — the CEILING on what a static per-cell base-rate channel (S1) can buy.

Question: does a per-cell historical base rate carry occurrence/magnitude information
the trained model does NOT already have?

Design.  Nothing is trained and no pod is used.  Two halves:

  D3-GATE  all 64,818 land cells x 13 OUT-OF-SAMPLE h=1 months (457-469), from
           p2_result.npz.  Fit a logistic model of the truth on the model's own gate
           logit, then add the channel, and read the IMPROVEMENT.  Fitted and scored on
           the same months on purpose: that makes the number an upper bound (a CEILING),
           which is the only direction that can kill S1.

  D3-BODY  the 100-cell set x 36 months (421-456, in-sample), from t3_result.npz.
           Regress log(mu_model / mu_empirical) on the channel.  R^2 is the ceiling.

C-350 is honoured: the channel is built STRICTLY from months <= 456, the training
partition (t3_scoring.py:45 records 421-456 as inside the TRAIN period; p2's 457-469 are
out-of-sample).  The channel never sees 457-469.

Free parameters are SWEPT, not chosen: two channel definitions x three history windows.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss

D = Path("reports/2026-10-12_underfire_diagnosis")
EPS = 1e-6

p2 = np.load(D / "p2_result.npz")
t3 = np.load(D / "t3_result.npz")
gate = p2["gate"].astype(np.float64)        # [4 posterior passes, 13 months, 64818 cells]
truth = p2["truth"]                         # [13, 64818] bool
land = p2["land_cells"]                     # [64818]
print("p2 artifact sha", str(p2["sha"]), "verdict", str(p2["verdict"]))
print("gate", gate.shape, "truth", truth.shape, "land", land.shape)
print("observed positive rate on the scored months %.6f" % truth.mean())

df = pd.read_parquet(D / "pull_land_all3_121_504.parquet").reset_index()
print("pull", df.shape, "months", int(df.month_id.min()), "-", int(df.month_id.max()))

# ---------------- the channel, training partition only ----------------
WINDOWS = [(121, 456), (121, 420), (301, 456)]
CHANNELS = {}
for lo, hi in WINDOWS:
    w = df[(df.month_id >= lo) & (df.month_id <= hi)]
    g = w.groupby("priogrid_id")["ged_sb_best"]
    rate = g.apply(lambda s: float((s > 0).mean()))
    mean = g.apply(lambda s: float(np.log1p(s).mean()))
    CHANNELS[f"pos_rate_{lo}_{hi}"] = rate.reindex(land).to_numpy(np.float64)
    CHANNELS[f"log1p_mean_{lo}_{hi}"] = mean.reindex(land).to_numpy(np.float64)
for k, v in CHANNELS.items():
    assert np.isfinite(v).all(), k
    print("  channel %-22s min %.4f  median %.4f  max %.4f  nonzero %.4f"
          % (k, v.min(), np.median(v), v.max(), (v > 0).mean()))

# ---------------- D3-GATE ----------------
print("\n" + "=" * 78)
print("D3-GATE — 64,818 land cells x 13 OUT-OF-SAMPLE months (457-469)")
print("=" * 78)
g_mean = gate.mean(0)                                   # average the 4 MC-dropout passes
y = truth.reshape(-1).astype(np.int8)
gl = logit(np.clip(g_mean, EPS, 1 - EPS)).reshape(-1)

base = LogisticRegression(max_iter=2000).fit(gl[:, None], y)
pb = base.predict_proba(gl[:, None])[:, 1]
ll_b, ap_b = log_loss(y, pb), average_precision_score(y, pb)
print("  gate alone                     log-loss %.6f   AP %.6f" % (ll_b, ap_b))
print("  %-24s %12s %12s %12s %12s" % ("+ channel", "log-loss", "dLL", "AP", "dAP"))
for name, ch in CHANNELS.items():
    x = np.repeat(ch[None, :], truth.shape[0], 0).reshape(-1)
    xs = (x - x.mean()) / (x.std() + EPS)
    m = LogisticRegression(max_iter=2000).fit(np.c_[gl, xs], y)
    p = m.predict_proba(np.c_[gl, xs])[:, 1]
    ll, ap = log_loss(y, p), average_precision_score(y, p)
    print("  %-24s %12.6f %12.6f %12.6f %+12.6f" % (name, ll, ll - ll_b, ap, ap - ap_b))

# channel ALONE, for reference: how much occurrence signal it holds at all
print("\n  for reference — the channel ALONE (no gate):")
for name, ch in CHANNELS.items():
    x = np.repeat(ch[None, :], truth.shape[0], 0).reshape(-1)
    print("    %-24s AP %.6f" % (name, average_precision_score(y, x)))
print("    %-24s AP %.6f" % ("gate alone", ap_b))

# per-cell aggregate view (coarser, but no cell-month pseudo-replication)
print("\n  per-cell view (observed rate over the 13 months vs the gate's own mean):")
obs_rate = truth.mean(0)
pred_rate = g_mean.mean(0)
from scipy.stats import spearmanr
for name, ch in CHANNELS.items():
    r_ch = spearmanr(ch, obs_rate).statistic
    r_gt = spearmanr(pred_rate, obs_rate).statistic
    # partial: does the channel explain the gate's residual?
    res = logit(np.clip(obs_rate, EPS, 1 - EPS)) - logit(np.clip(pred_rate, EPS, 1 - EPS))
    r_res = spearmanr(ch, res).statistic
    print("    %-24s spearman(channel,obs) %+.4f   spearman(channel,gate residual) %+.4f"
          % (name, r_ch, r_res))
print("    %-24s spearman(gate,obs)    %+.4f" % ("gate", r_gt))

# ---------------- D3-BODY ----------------
print("\n" + "=" * 78)
print("D3-BODY — the 100-cell set, mu_model vs mu_empirical on 421-456")
print("=" * 78)
cells = t3["cells"]
mu_ts = t3["MDL"][..., 0]            # [36, 100]
Y = t3["SCOREY"]                     # [100, 36]
mu_model = mu_ts.mean(0)             # per cell
mu_emp = Y.mean(1)
ratio = np.log((mu_model + EPS) / (mu_emp + EPS))
print("  cells %d   log-ratio: median %+.4f  sd %.4f  (negative = model under-predicts)"
      % (len(cells), np.median(ratio), ratio.std()))
pos = {int(c): i for i, c in enumerate(land)}
idx = np.array([pos[int(c)] for c in cells])
for name, ch in CHANNELS.items():
    c = ch[idx]
    cs = (c - c.mean()) / (c.std() + EPS)
    beta, a = np.polyfit(cs, ratio, 1)
    pred = a + beta * cs
    r2 = 1 - ((ratio - pred) ** 2).sum() / ((ratio - ratio.mean()) ** 2).sum()
    print("    %-24s R^2 %.4f   slope %+.4f   spearman %+.4f"
          % (name, r2, beta, spearmanr(c, ratio).statistic))
print("\n  for reference — does the channel predict mu_empirical itself?")
for name, ch in CHANNELS.items():
    c = ch[idx]
    print("    %-24s spearman(channel, mu_emp) %+.4f   spearman(mu_model, mu_emp) %+.4f"
          % (name, spearmanr(c, mu_emp).statistic, spearmanr(mu_model, mu_emp).statistic))
