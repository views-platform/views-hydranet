"""TIER 2 - the remedy comparison. Post-hoc, monotone, cross-validated by ORIGIN.

Runs on the STORED gate/truth arrays only: no model, no forward pass, no volume build.
Fit on 12 origins, evaluate on the held-out 1, 13 folds (PREREG §4).

  R1  temperature scaling   logit/T, T fitted by NLL on the fit folds        1 param
  R2  sigmoid(mean logit)   measured in E2; carried for completeness         0 params
  R3  Platt / vector        a + b*logit, fitted by ML on the fit folds       2 params
  R4  isotonic              non-parametric ceiling on monotone recalibration

FALSIFIERS (PREREG §5):
  F2  R1 held-out slope outside 1.00 +/- 0.05        -> temperature is not the matching fix
  F7  R1 reaches b~1 while AP and sharpness are unchanged within the ~0.019 seed sd -> H3 COSMETIC
  F8  R4 beats R1 held-out by >20% on |gate-obs|     -> not a pure slope; OUTCOME §2 incomplete
  F9  R1's T varies >15% across the 13 folds         -> no well-defined global temperature
  F10 rank identity fails under R1 or R3             -> implementation bug. AP movement alone is
      NOT sufficient; assert argsort equality plus a new-tie count.

NOTE ON F7, recorded before running: AP is rank-based and R1/R3 are strictly monotone, so AP is
unchanged BY CONSTRUCTION (PREREG §3.2's structural argument). F7's AP limb is therefore a theorem,
not a measurement, and the informative falsifiers here are F2, F8 and F9. Reported as such.
"""
import sys
from pathlib import Path

import numpy as np

EV = Path("/home/simon/Documents/scripts/views_platform/views-hydranet/reports/"
          "2026-10-16_pos_weight_arm_outcome/evidence")
E = np.array([0, .01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01])
SEED_SD_AP = 0.019          # 2026-10-16_pos_weight_prereg.md:175
EPS = 1e-9


def lg(x):
    x = np.clip(x, EPS, 1 - EPS)
    return np.log(x / (1 - x))


def sig(z):
    return 1 / (1 + np.exp(-z))


def ml_fit(x, y, iters=100):
    """Platt: ML fit of logit P(y=1) = a + b*x. Returns (a, b)."""
    a, b = 0.0, 1.0
    for _ in range(iters):
        p = sig(a + b * x)
        W = p * (1 - p)
        r = y - p
        H = np.array([[W.sum(), (W * x).sum()], [(W * x).sum(), (W * x * x).sum()]])
        d = np.linalg.solve(H + 1e-9 * np.eye(2), np.array([r.sum(), (r * x).sum()]))
        a += d[0]
        b += d[1]
        if np.abs(d).max() < 1e-11:
            break
    return a, b


def fit_T(x, y):
    """Temperature: 1 param, logit/T, by Newton on the NLL."""
    t = 1.0
    for _ in range(100):
        z = x / t
        p = sig(z)
        g = ((p - y) * (-x / t ** 2)).sum()
        h = ((p * (1 - p)) * (x / t ** 2) ** 2 + (p - y) * (2 * x / t ** 3)).sum()
        if abs(h) < 1e-12:
            break
        step = g / h
        t_new = max(0.05, min(20.0, t - step))
        if abs(t_new - t) < 1e-12:
            t = t_new
            break
        t = t_new
    return t


def pava(x, y):
    """Isotonic fit. sklearn's PAVA is C-optimised; the hand-rolled version was O(n^2) via
    np.delete in a loop and did not finish on 778k points per fold."""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    ir.fit(x, y)
    return ir, None


def iso_apply(ir, _unused, q):
    return ir.predict(q)


def binned_abs(g, y, minn=1000):
    out = []
    for lo, hi in zip(E[:-1], E[1:]):
        m = (g >= lo) & (g < hi)
        if m.sum() < minn or not 0 < y[m].mean() < 1:
            continue
        out.append(abs(g[m].mean() - y[m].mean()))
    return float(np.median(out)) if out else np.nan


def ap(g, y):
    """Average precision, ties grouped (the rank-axis metric; monotone maps cannot change it)."""
    o = np.argsort(-g, kind="stable")
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    n_pos = ys.sum()
    return float((prec * ys).sum() / n_pos) if n_pos else np.nan


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "amber"
    z = np.load(EV / f"p2_result_{arm}.npz")
    G = z["gate"].astype(np.float64).mean(0)      # (13, n_land) posterior-mean gate
    Y = z["truth"].astype(np.float64)             # (13, n_land)
    n_or = G.shape[0]
    print(f"TIER 2 - {arm}   {n_or} origins x {G.shape[1]:,} cells   "
          f"positives {int(Y.sum()):,}")

    rows, Ts = [], []
    for o in range(n_or):
        fit = np.arange(n_or) != o
        xf, yf = lg(G[fit].ravel()), Y[fit].ravel()
        xh, yh = lg(G[o]), Y[o]
        T = fit_T(xf, yf)
        a3, b3 = ml_fit(xf, yf)
        ir, _ = pava(G[fit].ravel(), yf)
        held = {
            "raw": sig(xh),
            "R1": sig(xh / T),
            "R3": sig(a3 + b3 * xh),
            "R4": iso_apply(ir, None, G[o]),
        }
        r = {"origin": o, "T": T, "a3": a3, "b3": b3}
        for k, p in held.items():
            _, bb = ml_fit(lg(p), yh)
            r[f"b_{k}"] = bb
            r[f"err_{k}"] = binned_abs(p, yh, minn=200)
            r[f"ap_{k}"] = ap(p, yh)
            r[f"sh_{k}"] = float(p.std())
        # F10: rank identity under the monotone maps
        o_raw = np.argsort(held["raw"], kind="stable")
        r["rank_R1"] = bool(np.array_equal(o_raw, np.argsort(held["R1"], kind="stable")))
        r["rank_R3"] = bool(np.array_equal(o_raw, np.argsort(held["R3"], kind="stable")))
        r["ties_R1"] = int(len(held["raw"]) - len(np.unique(held["R1"])))
        rows.append(r)
        Ts.append(T)

    def col(k):
        v = np.array([r[k] for r in rows], dtype=np.float64)
        return v[np.isfinite(v)]

    print(f"\n{'':<6}{'b held-out':>13}{'|gate-obs|':>13}{'AP':>11}{'sharpness':>12}")
    for k in ("raw", "R1", "R3", "R4"):
        print(f"{k:<6}{col(f'b_{k}').mean():>13.4f}{col(f'err_{k}').mean():>13.5f}"
              f"{col(f'ap_{k}').mean():>11.5f}{col(f'sh_{k}').mean():>12.5f}")

    T = np.array(Ts)
    print(f"\nR1 temperature: mean {T.mean():.4f}  min {T.min():.4f}  max {T.max():.4f}  "
          f"spread {(T.max()-T.min())/T.mean()*100:.1f}%")

    b1 = col("b_R1").mean()
    e1, e4 = col("err_R1").mean(), col("err_R4").mean()
    dap = abs(col("ap_R1").mean() - col("ap_raw").mean())
    dsh = abs(col("sh_R1").mean() - col("sh_raw").mean())
    print("\n=== FALSIFIERS ===")
    print(f"F2  |b_R1 - 1| = {abs(b1-1):.4f} vs 0.05        "
          f"-> {'FIRES: temperature is not the matching fix' if abs(b1-1) > 0.05 else 'pass'}")
    spread = (T.max() - T.min()) / T.mean()
    print(f"F9  T spread = {spread*100:.1f}% vs 15%     "
          f"-> {'FIRES: no well-defined global T' if spread > 0.15 else 'pass'}")
    print(f"F8  R4 vs R1 on |gate-obs|: {e4:.5f} vs {e1:.5f} "
          f"({(e1-e4)/e1*100:+.1f}%) -> "
          f"{'FIRES: NOT a pure slope' if e4 < 0.8*e1 else 'pass: the defect IS a slope'}")
    print(f"F7  dAP = {dap:.5f} (seed sd {SEED_SD_AP})  d_sharpness = {dsh:.5f}")
    if abs(b1 - 1) <= 0.05 and dap < SEED_SD_AP:
        print("    ⛔ FIRES -> H3: the calibration gain is COSMETIC. R1 fixes the number and "
              "the forecast does not move. Must be reported as such.")
    else:
        print("    does not fire as stated")
    print(f"F10 rank identity R1 {all(r['rank_R1'] for r in rows)}  "
          f"R3 {all(r['rank_R3'] for r in rows)}  "
          f"new ties under R1 (mean) {np.mean([r['ties_R1'] for r in rows]):.0f}")


if __name__ == "__main__":
    main()
