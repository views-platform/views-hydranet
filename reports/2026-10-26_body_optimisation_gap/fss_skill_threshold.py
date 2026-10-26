"""Roberts & Lean (2008, MWR 136:78-97, DOI 10.1175/2007MWR2123.1) applied to OUR grid.

⛔ PREDICTION PRE-COMMITTED FROM THE PAPER, NOT FROM A MEASUREMENT. The paper states, p. 83:

    FSS_random  = f_o
    FSS_uniform = 0.5 + f_o/2        <- the "target skill"; halfway random-to-perfect
    "If the domain is large, f_o -> 0 and FSS_uniform -> 0.5; then, for a displacement
     distance D, scale_min = 2D."

Our f_o on all land cell-months is 0.003448, so we sit IN that f_o -> 0 limit. The claim under
test is therefore not "FSS detects displacement" (weak, and B-1 passed a test that weak) but the
QUANTITATIVE one: the neighbourhood at which FSS crosses 0.5 + f_o/2 equals TWICE the
displacement, in cells. An instrument that reports the displacement in the units of the defect
is a ruler; B-1 was not one.

P-A  ideal band, 128x128, displaced by D: scale_min == 2D (+-1 cell for the odd-n grid).
P-B  the paper's Eq. 8 asymptote AFSS = 2 f_o f_M / (f_o^2 + f_M^2) == 1 for our implementation,
     because scripts/sharpness_scorecard.py:92 matched_pred_thresh FORCES f_M = f_o. If P-B holds,
     our FSS carries no bias information at all and is a pure placement instrument -- which is
     what it is wanted for, but it must be stated rather than assumed.

⚠️ ONE DEVIATION FROM THE PAPER, DECLARED: `fss()` uses uniform_filter(mode="constant"), which
pads OUTSIDE the grid with zeros. The paper's fractions are domain-interior. Swept here as a free
parameter (constant vs nearest) rather than assumed harmless.
"""

import sys

import numpy as np

sys.path.insert(0, "scripts")
from sharpness_scorecard import fss, matched_pred_thresh  # noqa: E402

F_O_LAND = 0.003448          # observed positive rate, all land cell-months (project memory)
ODD = list(range(1, 42, 2))  # neighbourhood lengths; FSS is defined for odd n


def bar(f_o: float) -> float:
    return 0.5 + f_o / 2.0


def scale_min(scores: dict[int, float], target: float) -> int | None:
    """Smallest n whose FSS reaches the target skill. None = never reached."""
    for n in sorted(scores):
        if np.isfinite(scores[n]) and scores[n] >= target:
            return n
    return None


def ideal_band(side: int, disp: int) -> tuple[np.ndarray, np.ndarray]:
    """The paper's Fig. 4: a 1-cell-wide band of events, forecast displaced by `disp`."""
    truth = np.zeros((side, side))
    pred = np.zeros((side, side))
    truth[:, side // 2] = 1.0
    pred[:, side // 2 + disp] = 1.0
    return pred, truth


def main() -> None:
    print("=" * 92)
    print("P-A — IDEAL BAND, 128x128.  paper: scale_min = 2D as f_o -> 0")
    print("=" * 92)
    side = 128
    f_o = 1.0 / side
    print(f"  f_o = 1/{side} = {f_o:.6f}   FSS_random = {f_o:.6f}   "
          f"FSS_uniform = {bar(f_o):.6f}")
    print(f"  {'D':>4s} {'scale_min':>10s} {'2D':>5s} {'verdict':>9s}   FSS at n=2D-1 / 2D+1")
    for d in (1, 2, 3, 5, 8, 11):
        s = fss(*ideal_band(side, d), ODD, pred_thresh=0.5, truth_thresh=0.5)
        got = scale_min(s, bar(f_o))
        ok = "PASS" if got is not None and abs(got - 2 * d) <= 1 else "FAIL"
        lo, hi = 2 * d - 1, 2 * d + 1
        print(f"  {d:4d} {str(got):>10s} {2*d:5d} {ok:>9s}   "
              f"{s.get(lo, float('nan')):.4f} / {s.get(hi, float('nan')):.4f}")

    print()
    print("=" * 92)
    print("P-B — does matched_pred_thresh force zero bias (AFSS = 1, paper Eq. 8)?")
    print("=" * 92)
    rng = np.random.default_rng(11)
    truth = (rng.random((128, 128)) < 0.02) * rng.gamma(2.0, 3.0, (128, 128))
    pred = rng.gamma(1.5, 2.0, (128, 128))          # dense, no zeros: a maximally biased field
    th = matched_pred_thresh(pred, truth)
    f_o_m = float((truth > 0).mean())
    f_m = float((pred > th).mean())
    afss = 2 * f_o_m * f_m / (f_o_m**2 + f_m**2)
    print(f"  f_o = {f_o_m:.6f}   f_M = {f_m:.6f}   AFSS = {afss:.6f}")
    print(f"  => {'CONFIRMED' if abs(afss - 1.0) < 1e-6 else 'REFUTED'}: our FSS is bias-free by "
          "construction, so it reads PLACEMENT ONLY.")

    print()
    print("=" * 92)
    print("DECLARED DEVIATION — zero-padding. scale_min for D=3 under both pad modes")
    print("=" * 92)
    import scipy.ndimage as ndi
    for side_sweep in (32, 64, 128, 180):
        p, t = ideal_band(side_sweep, 3)
        row = []
        for mode in ("constant", "nearest"):
            orig = ndi.uniform_filter
            s = {}
            for n in ODD:
                Pf = orig((p > 0.5).astype(float), size=n, mode=mode)
                Of = orig((t > 0.5).astype(float), size=n, mode=mode)
                w = np.mean(Pf**2) + np.mean(Of**2)
                s[n] = 1.0 - np.mean((Pf - Of) ** 2) / w if w > 0 else float("nan")
            row.append(scale_min(s, bar(1.0 / side_sweep)))
        print(f"  side={side_sweep:4d}   constant={str(row[0]):>5s}   nearest={str(row[1]):>5s}"
              f"   (2D = 6)")

    print()
    print("=" * 92)
    print(f"OUR BAR.  f_o = {F_O_LAND:.6f} on all land cell-months")
    print("=" * 92)
    print(f"  FSS_random  = f_o       = {F_O_LAND:.6f}")
    print(f"  FSS_uniform = 0.5+f_o/2 = {bar(F_O_LAND):.6f}   <- the target skill to clear")
    print("  and at that crossing the paper gives hit rate = 0.5, CSI = 0.33.")


if __name__ == "__main__":
    main()
