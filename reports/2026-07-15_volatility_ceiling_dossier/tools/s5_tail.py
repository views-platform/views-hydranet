"""S5 STRETCH — tail-family adequacy (EVT/GPD diagnostic).

Peaks-over-threshold: fit a Generalized Pareto to exceedances of the conflict-count target, estimate the
tail index xi (heaviness), and test whether the LIGHT families the head could use (geometric-ish NB with
theta~=0.98, or lognormal) structurally under-cover the extreme tail. Also: is xi state-varying (can a
head predict tail HEAVINESS, not just whether-volatile)? Diagnostic only — informs the head spec; the lab
demoted a GPD *head* before, so we probe whether the FAMILY matters, we don't commit to one.

Usage: s5_tail.py <parquet> <out_json>
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd


def gpd_xi(exceed):
    from scipy.stats import genpareto
    if exceed.size < 50:
        return {"n": int(exceed.size), "xi": None}
    c, loc, scale = genpareto.fit(exceed, floc=0.0)
    return {"n": int(exceed.size), "xi": float(c), "scale": float(scale)}


def tail_prob_nb(mu, theta, T):
    from scipy.stats import nbinom
    p = theta / (theta + mu)
    return float(nbinom.sf(T, theta, p))  # P(Y > T)


def tail_prob_lognorm(vals, T):
    from scipy.stats import lognorm
    v = vals[vals > 0]
    s, loc, scale = lognorm.fit(v, floc=0.0)
    return float(lognorm.sf(T, s, loc, scale))


def main():
    parquet, out_json = sys.argv[1], sys.argv[2]
    df = pd.read_parquet(parquet).reset_index()
    months = np.sort(df["month_id"].unique())
    m2i = {m: i for i, m in enumerate(months)}
    cells = np.sort(df["priogrid_gid"].unique())
    c2i = {c: i for i, c in enumerate(cells)}
    sb = np.zeros((len(cells), len(months)))
    sb[df["priogrid_gid"].map(c2i).values, df["month_id"].map(m2i).values] = df["lr_sb_best"].values
    absdiff = np.abs(sb - np.roll(sb, 1, 1))
    recent_vol = np.zeros_like(sb)
    for t in range(1, sb.shape[1]):
        lo = max(0, t - 12)
        recent_vol[:, t] = absdiff[:, lo:t].mean(axis=1)

    y = sb.reshape(-1)
    pos = y[y > 0]
    res = {"n_positive": int(pos.size), "max": float(y.max()), "mean_pos": float(pos.mean())}

    # ---- GPD tail index at several thresholds ----
    res["gpd"] = {}
    for u in (5, 10, 25, 50):
        ex = pos[pos > u] - u
        res["gpd"][f"u={u}"] = gpd_xi(ex)
    print("GPD tail index xi (heavier tail = larger xi; xi>0.5 => infinite variance):")
    for k, v in res["gpd"].items():
        print(f"  {k:7s} n={v['n']:6d} xi={v['xi']}")

    # ---- light-family under-coverage at extreme thresholds ----
    mu = float(pos.mean())
    theta = 0.98  # production global theta (S4)
    res["tail_coverage"] = {}
    print("\nP(Y>T): empirical vs light families (fit on positives):")
    for T in (100, 1000, 10000):
        emp = float(np.mean(pos > T))
        nb = tail_prob_nb(mu, theta, T)
        ln = tail_prob_lognorm(pos, T)
        res["tail_coverage"][f"T={T}"] = {"empirical": emp, "nb_theta0.98": nb, "lognormal": ln,
                                          "nb_undercover_x": emp / nb if nb > 0 else float("inf"),
                                          "lognorm_undercover_x": emp / ln if ln > 0 else float("inf")}
        print(f"  T={T:6d}: emp={emp:.2e}  NB(θ.98)={nb:.2e} ({emp/nb:.0f}x under)  "
              f"logN={ln:.2e} ({emp/ln:.0f}x under)" if nb > 0 and ln > 0 else
              f"  T={T:6d}: emp={emp:.2e} NB={nb:.2e} logN={ln:.2e}")

    # ---- is xi state-varying? (tertiles of recent_vol among positive-target cells) ----
    rv = recent_vol.reshape(-1)[y > 0]
    res["xi_by_state"] = {}
    print("\nxi by recent-volatility tertile (state-varying tail heaviness?):")
    qs = np.quantile(rv, [1 / 3, 2 / 3])
    for lbl, lo, hi in [("low", -np.inf, qs[0]), ("mid", qs[0], qs[1]), ("high", qs[1], np.inf)]:
        grp = pos[(rv > lo) & (rv <= hi)]
        ex = grp[grp > 10] - 10
        g = gpd_xi(ex)
        res["xi_by_state"][lbl] = g
        print(f"  {lbl:4s} n={g['n']:6d} xi={g['xi']}")

    with open(out_json, "w") as fh:
        json.dump(res, fh, indent=2, default=float)
    print("\nwrote", out_json)


if __name__ == "__main__":
    main()
