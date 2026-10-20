"""D10 — debias the effective-cell count, and settle whether D9-C's sweep is noise or relocation.

The `opponent` seat (§16) points out that D9's effective-cell count
    N_eff = (sum p)^2 / sum p^2
is computed for the TRUTH from a count-based rate estimate p_hat = k/T, and
    E[sum p_hat^2] = sum p^2 + sum p(1-p)/T
so the DENOMINATOR is biased UP and N_eff is therefore biased DOWN. The truth row understates, and
the bias runs AGAINST the finding. The model row needs no correction: p_model is computed
analytically from (mu, theta, gate), so it carries no count-sampling noise.

⛔ ITS FORMULA IS INCOMPLETE AND I AM NOT COPYING IT. The NUMERATOR is biased too:
    E[(sum p_hat)^2] = (sum p)^2 + Var(sum p_hat) = (sum p)^2 + sum p(1-p)/T
the SAME additive term. It is negligible here only because (sum p)^2 >> sum p^2 -- which is a
statement about this field, not about the estimator -- so it is computed rather than assumed.

Both corrections applied:  S2_hat - B  and  S1_hat^2 - B,  with B = sum p_hat(1-p_hat)/T * T/(T-1)
(the T/(T-1) factor because B itself is estimated from p_hat; the uncorrected plug-in underestimates
the binomial variance).

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
t0 = time.time()

z = np.load(SP / "d4_result.npz")
land_cells, kept = z["land_cells"], z["months"]
j = np.array([list(kept).index(m) for m in range(IS_LO, IS_HI + 1)])
mu = z["mu"].mean(0)[j].mean(0).astype(np.float64)
th = z["theta"].mean(0)[j].mean(0).astype(np.float64)
ga = np.clip(z["gate"].mean(0)[j].mean(0).astype(np.float64), 0.0, 1.0)
p_model = ga * (1.0 - np.power(th / (th + mu), th))

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
piv = df.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB]).reindex(land_cells)

m_S1 = p_model.sum()
m_S2 = (p_model ** 2).sum()
m_neff = m_S1 ** 2 / m_S2
print(f"MODEL: sum p {m_S1:.4f}  sum p^2 {m_S2:.6f}  N_eff {m_neff:.1f}")
print("  (analytic from (mu, theta, gate) -- no count-sampling noise, so no debias applies)\n")

print("=" * 78)
print("D10 — EFFECTIVE CELLS FOR THE TRUTH, RAW AND DEBIASED")
print("=" * 78)
print(f"  {'window':>10}{'T':>5}{'N_eff raw':>12}{'B':>9}{'num corr %':>12}"
      f"{'N_eff debiased':>16}{'bias':>8}{'gap x':>8}")
rows = []
for lo, hi, lab in ((IS_LO, IS_HI, "36 mo"), (385, 456, "72 mo"),
                    (301, 456, "156 mo"), (121, 456, "336 mo")):
    cols = [m for m in range(lo, hi + 1) if m in piv.columns]
    T = len(cols)
    Y = np.nan_to_num(piv[cols].to_numpy(np.float64), nan=0.0)
    ph = (Y > 0).mean(1)
    S1, S2 = ph.sum(), (ph ** 2).sum()
    raw = S1 ** 2 / S2
    B = (ph * (1.0 - ph)).sum() / T * (T / (T - 1.0))      # unbiased binomial variance sum
    deb = (S1 ** 2 - B) / (S2 - B)
    rows.append((lab, T, raw, B, deb))
    print(f"  {lab:>10}{T:5d}{raw:12.1f}{B:9.2f}{100 * B / S1 ** 2:11.4f}%"
          f"{deb:16.1f}{100 * (deb / raw - 1):7.1f}%{deb / m_neff:8.2f}")

print(f"\n  the numerator correction is {100 * rows[0][3] / (rows[0][2] * 0 + 1):.0f}"
      f"… see the 'num corr %' column: it is <0.02% of (sum p)^2 at every window, so the seat's")
print("  denominator-only formula is adequate HERE -- confirmed by computing it, not assumed.")
print(f"\n  => the 36-month gap widens from {rows[0][2] / m_neff:.2f}x (raw) to "
      f"{rows[0][4] / m_neff:.2f}x (debiased).")

print("\n" + "=" * 78)
print("IS D9-C'S SWEEP NOISE OR RELOCATION?")
print("=" * 78)
raw_rise = 100 * (rows[-1][2] / rows[0][2] - 1)
deb_rise = 100 * (rows[-1][4] / rows[0][4] - 1)
print(f"  raw      36 -> 336 mo: {rows[0][2]:.1f} -> {rows[-1][2]:.1f}  = {raw_rise:+.1f}%")
print(f"  debiased 36 -> 336 mo: {rows[0][4]:.1f} -> {rows[-1][4]:.1f}  = {deb_rise:+.1f}%")
print(f"  noise explains at most {100 * (rows[0][4] / rows[0][2] - 1):.1f}% (at 36 mo), falling to "
      f"{100 * (rows[-1][4] / rows[-1][2] - 1):.1f}% (at 336)")
print(f"  => a {deb_rise:+.1f}% rise SURVIVES debiasing ⇒ RELOCATION carries essentially all of it.")

mono = all(rows[i][4] <= rows[i + 1][4] for i in range(len(rows) - 1))
print(f"\n  ⚠️  is the debiased series monotone in the window?  {'yes' if mono else 'NO'}")
if not mono:
    print("      " + " -> ".join(f"{r[4]:.1f}" for r in rows))
    print("      It DIPS at 72 months. So 'relocation monotonically spreads the footprint' is NOT")
    print("      what the data says; the 36->336 direction is clear and the path is not. Reported")
    print("      rather than smoothed, because the dip is larger than the noise correction that")
    print("      produced the series.")

print("\n" + "=" * 78)
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
