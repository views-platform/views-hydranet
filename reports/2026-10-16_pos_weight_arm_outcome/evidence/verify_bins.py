"""The bin-support check — the comparability issue the readout did not flag.

P-1 and P-2 are medians over calibration bins with n >= 1000. The two arms do NOT have the same
such bins: the arm fires less, so its [0.20, 0.30) bin falls below the cut and drops out. The
published medians are therefore taken over different supports. This recomputes both on the COMMON
bins, which strengthens the kill (2.39x rather than 1.96x) but moves the evidence's interpolated
"honest pos_weight" from ~1.55 to ~1.41. See OUTCOME.md §4.2.
"""

import numpy as np

BINS = [0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]


def table(path: str, tag: str) -> list[tuple[float, float, float]]:
    d = np.load(path, allow_pickle=True)
    gm = d["gate"].mean(axis=0).ravel()
    yy = (d["truth"].ravel() > 0).astype(float)
    idx = np.digitize(gm, BINS) - 1
    print(f"\n{tag}")
    print(f"  {'bin':<14}{'n':>9}{'mean gate':>11}{'obs p':>9}{'|g-p|':>9}{'impl w':>9}  used?")
    used = []
    for b in range(len(BINS) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        s, p = float(gm[m].mean()), float(yy[m].mean())
        iw = s * (1 - p) / (p * (1 - s)) if 0 < p < 1 else float("nan")
        flag = "YES" if n >= 1000 else "no (n<1000)"
        print(
            f"  [{BINS[b]:.2f},{BINS[b + 1]:.2f}){n:>9}{s:>11.4f}{p:>9.4f}"
            f"{abs(s - p):>9.4f}{iw:>9.4f}  {flag}"
        )
        if n >= 1000:
            used.append((BINS[b], abs(s - p), iw))
    return used


ctrl = table("p2_result_amber.npz", "CONTROL (trained w=2)")
arm = table("p2_result_copper.npz", "ARM (trained w=1)")

print("\n=== the bins each published median is taken over ===")
print("  control:", [f"{b:.2f}" for b, _, _ in ctrl])
print("  arm    :", [f"{b:.2f}" for b, _, _ in arm])
common = sorted({b for b, _, _ in ctrl} & {b for b, _, _ in arm})
print("  COMMON :", [f"{b:.2f}" for b in common])

cd_ = {b: (e, w) for b, e, w in ctrl}
ad = {b: (e, w) for b, e, w in arm}
ce = [cd_[b][0] for b in common]
ae = [ad[b][0] for b in common]
cw = [cd_[b][1] for b in common]
aw = [ad[b][1] for b in common]

print("\n=== restricted to the COMMON bins (apples to apples) ===")
ratio = np.median(ae) / np.median(ce)
print(
    f"  median |g-p|   control {np.median(ce):.4f}   arm {np.median(ae):.4f}   ratio {ratio:.2f}x"
)
print(f"  median impl w  control {np.median(cw):.4f}   arm {np.median(aw):.4f}")
honest = 1 + (1.0 - np.median(aw)) / (np.median(cw) - np.median(aw))
print(f"  interpolated honest pos_weight on this support: {honest:.2f}")
print("\n  (as published, over each arm's OWN bins: 0.0154 / 0.0303, 1.3090 / 0.6252, ~1.55)")
