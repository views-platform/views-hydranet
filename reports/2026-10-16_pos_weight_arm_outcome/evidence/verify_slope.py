"""The signed-error and slope analysis — the evidence that the defect is a ROTATION.

Written by the views-hydranet session to verify the views-models session's deep-review finding
that the gate is UNDER-DISPERSED (a slope error) rather than carrying a second downward bias (a
level error). Everything printed here reproduced that finding to the digit.

P-1 as pre-registered takes the ABSOLUTE error, so it is structurally blind to the sign flip this
prints — which is why neither session saw it until the bin-support correction forced a second look.
See OUTCOME.md §2.
"""

import numpy as np

BINS = [0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]


def bin_table(path: str) -> list[tuple]:
    d = np.load(path, allow_pickle=True)
    gm = d["gate"].mean(axis=0).ravel()
    yy = (d["truth"].ravel() > 0).astype(float)
    idx = np.digitize(gm, BINS) - 1
    out = []
    for b in range(len(BINS) - 1):
        m = idx == b
        n = int(m.sum())
        if n == 0:
            continue
        s, p = float(gm[m].mean()), float(yy[m].mean())
        iw = s * (1 - p) / (p * (1 - s)) if 0 < p < 1 else np.nan
        out.append((BINS[b], BINS[b + 1], n, s, p, s - p, iw))
    return out


ctrl, arm = bin_table("p2_result_amber.npz"), bin_table("p2_result_copper.npz")
ad = {r[0]: r for r in arm}

print("=== SIGNED error (gate - observed p). P-1 takes |.| and cannot see this ===")
print(f"{'bin':<15}{'n ctrl':>9}{'ctrl signed':>13}{'n arm':>8}{'arm signed':>13}")
for lo, hi, n, _s, _p, e, _w in ctrl:
    a = ad.get(lo)
    an = a[2] if a else 0
    ae = a[5] if a else float("nan")
    print(f"[{lo:.2f},{hi:.2f}){n:>9}{e:>+13.4f}{an:>8}{ae:>+13.4f}")

print("\n=== sign sequences ===")
for tag, R in (("control", ctrl), ("arm", arm)):
    print(f"   {tag:<8}", " ".join("+" if r[5] > 0 else "-" for r in R))
for (lo1, _h1, _n1, _s1, _p1, e1, _w1), (lo2, *_rest) in zip(ctrl, ctrl[1:]):
    e2 = [r[5] for r in ctrl if r[0] == lo2][0]
    if e1 * e2 < 0:
        print(
            f"   ⛔ control CROSSES ZERO between [{lo1:.2f}..) and [{lo2:.2f}..): "
            f"{e1:+.4f} -> {e2:+.4f}"
        )

print("\n=== implied_w is NOT a constant, so there is nothing to interpolate to ===")
for tag, R in (("control", ctrl), ("arm", arm)):
    w = [r[6] for r in R if np.isfinite(r[6])]
    crosses = "YES" if min(w) < 1 < max(w) else "no"
    print(
        f"   {tag:<8} min {min(w):.3f}  max {max(w):.3f}  "
        f"range {max(w) / min(w):.1f}x  crosses 1.0? {crosses}"
    )

print("\n=== logit(p) = a + b*logit(gate).  An honest gate gives a=0, b=1 ===")


def logit(x: np.ndarray) -> np.ndarray:
    return np.log(x / (1 - x))


for tag, R in (("control", ctrl), ("arm", arm)):
    usable = [r for r in R if 0 < r[4] < 1]
    for lbl, sub in (("n>=1000", [r for r in usable if r[2] >= 1000]), ("all bins", usable)):
        b, a = np.polyfit(
            logit(np.array([r[3] for r in sub])), logit(np.array([r[4] for r in sub])), 1
        )
        print(f"   {tag:<8} {lbl:<11} a={a:+.3f}  b={b:.3f}")
    wt = np.sqrt(np.array([r[2] for r in usable], float))
    b, a = np.polyfit(
        logit(np.array([r[3] for r in usable])), logit(np.array([r[4] for r in usable])), 1, w=wt
    )
    print(f"   {tag:<8} {'n-weighted':<11} a={a:+.3f}  b={b:.3f}   (sqrt-n weights)")

print("\n   b > 1 everywhere => the gate's logits are COMPRESSED: truth moves faster than")
print("   the gate does. pos_weight shifts the LEVEL (a); the defect is in the SLOPE (b).")
