"""Independent re-derivation of the pos_weight arm's headline numbers.

Written by the views-hydranet session, from the stored arrays only, WITHOUT reading the
views-models session's statistics block — so agreement is a reproduction rather than a re-run of
the same code. Confirms P-1, P-2, the weight shas, and the `pos_weight` field trap
(EVIDENCE.md §0).
"""

import numpy as np

BINS = [0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.01]


def stats(path: str, tag: str) -> tuple[float, float]:
    d = np.load(path, allow_pickle=True)
    print(f"\n--- {tag}  ({path})")
    print("   keys:", sorted(d.files))
    print("   npz 'pos_weight' field:", d["pos_weight"], "<- TRAP: hardcoded, meaningless here")
    print("   sha field:", str(d["sha"])[:24])
    gate, truth = d["gate"], d["truth"]
    print("   gate shape:", gate.shape, " truth shape:", truth.shape)
    gm = gate.mean(axis=0).ravel()
    yy = (truth.ravel() > 0).astype(float)
    assert gm.shape == yy.shape, (gm.shape, yy.shape)
    idx = np.digitize(gm, BINS) - 1
    cal, imp = [], []
    for b in range(len(BINS) - 1):
        m = idx == b
        if int(m.sum()) < 1000:
            continue
        s, p = float(gm[m].mean()), float(yy[m].mean())
        cal.append(abs(s - p))
        if 0 < p < 1:
            imp.append(s * (1 - p) / (p * (1 - s)))
    print(f"   bins n>=1000: {len(cal)}")
    print(f"   MEDIAN |gate - p|  = {np.median(cal):.4f}")
    print(f"   MEDIAN implied_w   = {np.median(imp):.4f}")
    return float(np.median(cal)), float(np.median(imp))


c_cal, c_iw = stats("p2_result_amber.npz", "CONTROL amber_kestrel (trained w=2)")
a_cal, a_iw = stats("p2_result_copper.npz", "ARM     copper_kestrel (trained w=1)")

print("\n" + "=" * 66)
print(f"P-1 calibration: control {c_cal:.4f}  arm {a_cal:.4f}   ratio {a_cal / c_cal:.2f}x")
flip = "YES" if (c_iw - 1) * (a_iw - 1) < 0 else "NO"
print(f"P-2 implied_w  : control {c_iw:.4f}  arm {a_iw:.4f}   sign flip? {flip}")
print("EVIDENCE.md claims  P-1 0.0154 / 0.0303 ; P-2 1.3090 / 0.6252")
print("=" * 66)

r = np.load("readout_result.npz", allow_pickle=True)
print("\nreadout_result.npz:")
for k in sorted(r.files):
    v = r[k]
    shown = v if v.size <= 4 else str(v.ravel()[:4]) + "..."
    print(f"   {k}: {shown}")
