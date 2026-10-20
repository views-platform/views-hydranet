"""D9 — settle the D8-B sign WITHOUT a null, and show why D8-B could not.

D8-B compared (cumulative support shortfall)/(monthly shortfall) = 0.829 against an IMPLICIT null of
1.0. The `opponent` seat showed that null is wrong: cumulative support `1-(1-P)^T` is CONCAVE and
SATURATING in P, so support-per-unit-firing falls as firing rises AT FIXED SPATIAL ARRANGEMENT. The
two rows sit 4.18x apart in firing, so part of the truth's lower support/monthly is the map, not the
arrangement. Its synthetic-field nulls put the correct value at 0.46 (multiplicative in P) to 0.77
(additive in logit) -- and 0.829 sits ABOVE both, reversing my conclusion back toward concentration.

This script does three things.

  A  The seat's own suggested test, on the REAL per-cell field instead of a synthetic one. The
     multiplicative null is exactly computable (c*0 = 0). The logit null is NOT: 97.8% of cells have
     an estimated rate of exactly 0 and logit(0) = -inf, so it requires a choice about the atom --
     and the choice moves the answer. That is the transform slot again, inside the fix for a
     transform error.

  B  ⛔ THE INSTRUMENT THAT NEEDS NO NULL AT ALL. "Arrangement" is a property of the SHAPE of the
     per-cell firing field, so measure the shape directly: the share of total firing mass in the top
     k% of cells, and the Gini. Both are normalised by the total, so the 4.18x level difference
     CANCELS EXACTLY -- no saturating map, no null, no synthetic field, no parameterisation choice.
     More concentrated model => over-persistence. Less => it spreads.

  C  The noise bound on B. The truth's per-cell rate over 36 months is quantised to 1/36 and zero in
     97.8% of cells; estimation noise INFLATES measured concentration, biasing the truth's
     concentration UP and so biasing B AGAINST the concentration conclusion. Bounded by recomputing
     the truth's shape on longer windows.

Reads d4_result.npz. No forward pass. Free.
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, TARGETS, pull  # noqa: E402

SP = Path(__file__).parent
IS_LO, IS_HI = 421, 456
T = IS_HI - IS_LO + 1
t0 = time.time()

z = np.load(SP / "d4_result.npz")
land_cells, kept = z["land_cells"], z["months"]
j = np.array([list(kept).index(m) for m in range(IS_LO, IS_HI + 1)])
mu = z["mu"].mean(0)[j].mean(0).astype(np.float64)
th = z["theta"].mean(0)[j].mean(0).astype(np.float64)
ga = np.clip(z["gate"].mean(0)[j].mean(0).astype(np.float64), 0.0, 1.0)
p_model = ga * (1.0 - np.power(th / (th + mu), th))       # per-cell monthly P(y>0)

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
piv = df.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB]).reindex(land_cells)


def truth_p(lo, hi):
    cols = [m for m in range(lo, hi + 1) if m in piv.columns]
    Y = np.nan_to_num(piv[cols].to_numpy(np.float64), nan=0.0)
    return (Y > 0).mean(1), Y


p_truth, Y36 = truth_p(IS_LO, IS_HI)
supp = lambda p, t=T: (1.0 - np.power(1.0 - p, t)).mean()
m_month, t_month = p_model.mean(), p_truth.mean()
m_supp, t_supp = supp(p_model), supp(p_truth)
obs_ratio = (t_supp / m_supp) / (t_month / m_month)
print(f"model monthly {m_month:.6f}  support {m_supp:.6f}")
print(f"truth monthly {t_month:.6f}  support {t_supp:.6f}")
print(f"firing shortfall {t_month / m_month:.3f}x   OBSERVED ratio {obs_ratio:.4f}")

print("\n" + "=" * 78)
print("D9-A — THE SEAT'S NULLS, ON THE REAL FIELD INSTEAD OF A SYNTHETIC ONE")
print("=" * 78)
c = m_month / t_month
n_supp = supp(c * p_truth)
print(f"  multiplicative in P:  P -> {c:.6f}*P   (exact; no atom problem)")
print(f"    null support {n_supp:.6f}   null ratio {(t_supp / n_supp) / (t_month / m_month):.4f}"
      f"   (seat's synthetic: 0.4640)")

print("\n  additive in logit: ⛔ NOT exactly computable on the real field.")
nz = (p_truth > 0).mean()
print(f"    {100 * (1 - nz):.2f}% of cells have an estimated rate of EXACTLY 0, and logit(0) = -inf.")
print("    It needs a choice about the atom, and the choice moves the answer:")
for name, phat in (("(k+0.5)/(T+1)", (Y36 > 0).sum(1) + 0.5),
                   ("(k+1)/(T+2)", (Y36 > 0).sum(1) + 1.0)):
    den = T + 1.0 if "0.5" in name else T + 2.0
    q = phat / den
    lo_, hi_ = -40.0, 10.0
    for _ in range(200):                      # bisect the logit shift to match the model's mean P
        mid = 0.5 * (lo_ + hi_)
        v = (1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + mid)))).mean()
        lo_, hi_ = (mid, hi_) if v < m_month else (lo_, mid)
    d = 0.5 * (lo_ + hi_)
    pn = 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + d)))
    print(f"    atom rule {name:16} shift {d:+.3f}  null support {supp(pn):.6f}"
          f"  null ratio {(t_supp / supp(pn)) / (t_month / m_month):.4f}")
print("    (seat's synthetic logit null: 0.7709)")

print("\n" + "=" * 78)
print("D9-B — ⛔ NO NULL NEEDED: measure the ARRANGEMENT directly. Level cancels exactly.")
print("=" * 78)


def shape(p):
    s = np.sort(p)[::-1]
    tot = s.sum()
    out = {f"top{k}%": s[: max(1, int(round(len(s) * k / 100)))].sum() / tot
           for k in (0.1, 1, 5, 10)}
    x = np.sort(p)
    n = len(x)
    out["gini"] = (2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())
    out["eff_cells"] = (tot ** 2) / (p ** 2).sum()        # participation ratio
    return out


sm, st = shape(p_model), shape(p_truth)
print(f"  {'statistic':14}{'model':>12}{'truth':>12}{'model/truth':>14}")
for k in ("top0.1%", "top1%", "top5%", "top10%", "gini"):
    print(f"  {k:14}{sm[k]:12.4f}{st[k]:12.4f}{sm[k] / st[k]:14.3f}")
print(f"  {'eff. cells':14}{sm['eff_cells']:12.1f}{st['eff_cells']:12.1f}"
      f"{sm['eff_cells'] / st['eff_cells']:14.3f}")
more = sm["gini"] > st["gini"]
print(f"\n  => the model's firing mass is {'MORE' if more else 'LESS'} concentrated than the truth's")
print(f"     (gini {sm['gini']:.4f} vs {st['gini']:.4f}; effective cells "
      f"{sm['eff_cells']:.0f} vs {st['eff_cells']:.0f}).")
print(f"     {'OVER-PERSISTENCE / over-concentration.' if more else 'IT SPREADS MORE.'}")

print("\n" + "=" * 78)
print("D9-C — THE NOISE BOUND. Estimation noise inflates the TRUTH's measured concentration,")
print("       so it biases D9-B against the concentration conclusion. How much?")
print("=" * 78)
print(f"  {'truth window':22}{'gini':>10}{'eff cells':>12}{'top1%':>10}")
for lo, hi, lab in ((IS_LO, IS_HI, "36 mo (matched)"), (385, 456, "72 mo"),
                    (301, 456, "156 mo"), (121, 456, "336 mo")):
    pt, _ = truth_p(lo, hi)
    sh = shape(pt)
    print(f"  {lab:22}{sh['gini']:10.4f}{sh['eff_cells']:12.1f}{sh['top1%']:10.4f}")
print(f"  {'MODEL':22}{sm['gini']:10.4f}{sm['eff_cells']:12.1f}{sm['top1%']:10.4f}")
print("\n  ⚠️  a longer truth window is less noisy but spans more regime change, which spreads the")
print("      truth's own footprint -- the two effects push the truth's concentration opposite ways,")
print("      so this is a sensitivity sweep and not a clean bound.")

print("\n" + "=" * 78)
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
