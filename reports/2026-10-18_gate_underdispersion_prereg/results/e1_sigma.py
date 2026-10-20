"""E1 - across-draw logit sd (sigma) and the implied MC-dropout temperature T.

PRE-REGISTERED as a MEASUREMENT, not a test (PREREG §4): the sigma floors were derived by
inverting published b values both seats had already read, so agreement is consistency, not
confirmation. One-sided floors: sigma >= 0.85 (baseline), >= 1.20 (arm); b is a LOWER bound on T.

Per §3.5 the summary statistic for a varying quantity is itself a free parameter, so it is swept.
"""
import numpy as np

EV = ("/home/simon/Documents/scripts/views_platform/views-hydranet/reports/"
      "2026-10-16_pos_weight_arm_outcome/evidence")
EPS = 1e-9


def lg(x):
    return np.log(np.clip(x, EPS, 1 - EPS) / (1 - np.clip(x, EPS, 1 - EPS)))


def T_of(s):
    return np.sqrt(1 + np.pi * s ** 2 / 8)


print("E1 - across-draw logit sd, D=4 locked-dropout draws")
print("PRE-REGISTERED FLOORS:  baseline sigma >= 0.85   arm sigma >= 1.20")
print("(one-sided: sigma ABOVE the floor is CONSISTENT with the chain, not falsifying)")
print()
for nm, floor, b_pub in (("amber  (w=2, baseline)", 0.85, 1.146),
                         ("copper (w=1, arm)", 1.20, 1.419)):
    z = np.load(f"{EV}/p2_result_{nm.split()[0]}.npz")
    g = z["gate"].astype(np.float64)              # (4, 13, 64818)
    sat = int((g <= EPS).sum() + (g >= 1 - EPS).sum())
    L = lg(g)                                     # per-draw logits
    sd = L.std(axis=0, ddof=1)                    # per cell-month, 3 dof
    print(f"=== {nm} ===")
    print(f"  cells {sd.size:,}   saturated/clipped draws: {sat}")
    print(f"  per-cell sigma: min {sd.min():.3f}  p25 {np.percentile(sd,25):.3f} "
          f" median {np.median(sd):.3f}  p75 {np.percentile(sd,75):.3f}  max {sd.max():.3f}")
    print(f"  {'summary':<10} {'sigma':>8} {'T=sqrt(1+pi s^2/8)':>20} {'vs floor':>10} "
          f"{'T vs b_pub':>12}")
    for tag, s in (("mean", sd.mean()), ("median", np.median(sd)),
                   ("RMS", np.sqrt((sd ** 2).mean())), ("p75", np.percentile(sd, 75))):
        print(f"  {tag:<10} {s:>8.3f} {T_of(s):>20.3f} "
              f"{'PASS' if s >= floor else 'BELOW':>10} {T_of(s)/b_pub:>12.2f}x")
    print()
