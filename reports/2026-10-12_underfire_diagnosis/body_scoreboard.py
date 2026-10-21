"""The body's scoreboard — B-1, B-2, B-3 on the control artifact.

Defined in `reports/2026-10-21_body_scoreboard.md`, which is committed and must be read first.
⛔ This script computes a RULER's baselines, not a result. Per that file's §1 cap: three numbers,
existing instruments only, and anything interesting it turns up is logged and NOT chased.

  B-1  the scale profile      gamma_model(d)/gamma_truth(d) over the same-shape null, from the
                              variogram curves D12 already computed and committed (d12_result.npz).
  B-2  the magnitude slope    OLS of log truth on log mu, on BOTH pinned populations.
  B-3  the magnitude level    median(log mu - log truth), closing the gap GLOSSARY.md:263 records
                              as unverified off the 100-cell set.

⛔ WHY BOTH POPULATIONS FOR B-2 AND B-3.  `b` read 1.7301 on the 100 cells with the highest
observed rate and 0.92 on the grid — C-351 instance #3, and it was a report title. The
sensitivity IS the result, so both are reported and neither is called the real one.
97.8% of land cells have a window truth of exactly 0, so log(0 + eps) = -13.8 dominates
the all-land version ⇒ ⛔ it is a COMPARATOR (identical definition for every arm), never
an unbiased estimate of anything.

Reads `d4_result.npz`, `d12_result.npz`, `d12_fits.csv` and the cached land pull. Seconds, no pod.
Cost printed, per the standing rule.
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, TARGETS, pull  # noqa: E402

SP = Path(__file__).parent
EPS = 1e-6
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
SHOW = (1, 2, 4, 8, 16, 32)
t_start = time.time()

npz = np.load(SP / "d4_result.npz")
MU, GATE = npz["mu"], npz["gate"]
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {m: i for i, m in enumerate(months)}
mu_pm = MU.mean(0).astype(np.float64)
gt_pm = GATE.mean(0).astype(np.float64)

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64))
base336 = (df[df.month_id <= 456].groupby("priogrid_id")[TARGETS[SB]].mean()
           .reindex(land_cells).to_numpy(np.float64))

curves = np.load(SP / "d12_result.npz")
var = {}
for line in (SP / "d12_fits.csv").read_text().splitlines()[1:]:
    head, _ok, _n, _s, _xi, _nos, _dm, _bins, v = line.rsplit(",", 8)
    wn, rest = head.split(",", 1)
    fn, dn = rest.rsplit(",", 1)
    var[(wn, fn.strip('"'), dn)] = float(v)

print("=" * 96)
print("THE BODY'S SCOREBOARD — baselines on the CONTROL artifact")
print("=" * 96)
is_cols = [mcol[m] for m in range(IS_LO, IS_HI + 1)]
zero_is = float((Y[:, is_cols].mean(1) == 0).mean())
zero_336 = float((base336 == 0).mean())
print("  vehicle sha %s…   land cells %s" % (str(npz["sha"])[:16], f"{len(land_cells):,}"))
print("  land cells with truth exactly 0:  in-sample window %.4f   336-month base rate %.4f"
      % (zero_is, zero_336))

# ---------------------------------------------------------------- B-1
print("\n" + "-" * 96)
print("B-1  the scale profile — gamma_model(d)/gamma_truth(d) over the same-shape null")
print("-" * 96)
print("     1.00 = the model carries its full share of spatial structure at that scale")
TRUTH_F = "truth log mean, 336-mo base rate"
for wname in ("in sample", "out of sample"):
    for d in ("E-W", "isotropic"):
        td = curves[f"{wname}|{TRUTH_F}|{d}|dist"]
        tg = curves[f"{wname}|{TRUTH_F}|{d}|gam"]
        tv = var[(wname, TRUTH_F, d)]
        md = curves[f"{wname}|model log mu|{d}|dist"]
        mg = curves[f"{wname}|model log mu|{d}|gam"]
        mv = var[(wname, "model log mu", d)]
        null = mv / tv
        vals = [(mg[np.argmin(np.abs(md - k))] / tg[np.argmin(np.abs(td - k))]) / null
                for k in SHOW]
        print("     %-14s %-10s " % (wname, d)
              + "  ".join(f"d{s}={v:.3f}" for s, v in zip(SHOW, vals)))

# ---------------------------------------------------------------- B-2 and B-3
print("\n" + "-" * 96)
print("B-2  the magnitude slope b  |  B-3  the magnitude level (median log mu - log truth)")
print("-" * 96)
print("     %-13s %-26s %-21s %7s %7s %8s %8s %8s"
      % ("window", "predictor", "population", "n", "b", "r", "level", "implied"))
for wname, (lo, hi) in (("in sample", (IS_LO, IS_HI)), ("out of sample", (OS_LO, OS_HI))):
    j = [mcol[m] for m in range(lo, hi + 1)]
    mu_w, gt_w, y_w = mu_pm[j].mean(0), gt_pm[j].mean(0), Y[:, j].mean(1)
    # ⛔ TWO PREDICTORS, and the distinction is the whole correctness of B-3. `mu` is the body's
    # CONDITIONAL mean (the magnitude given an event); the truth's window mean is UNCONDITIONAL
    # (it averages over months with no event). Comparing them is a C-351 estimator violation and
    # it manufactures apparent over-prediction. The composed forecast `gate x mu` — GLOSSARY §2's
    # "gated forecast" — is the unconditional object and is the one comparable to the truth.
    for predname, pred in (("gated forecast (gate x mu)", gt_w * mu_w), ("body alone (mu)", mu_w)):
        for pname, keep in (("all land", np.ones(len(land_cells), bool)),
                            ("truth > 0 in window", y_w > 0),
                            ("336-mo base rate > 0", base336 > 0)):
            x, y = np.log(pred[keep] + EPS), np.log(y_w[keep] + EPS)
            b, _a = np.polyfit(x, y, 1)
            r = float(np.corrcoef(x, y)[0, 1])
            level = float(np.median(x - y))
            print("     %-13s %-26s %-21s %7d %7.4f %+8.4f %+8.4f %7.1f%%"
                  % (wname, predname, pname, keep.sum(), b, r, level, 100 * np.exp(level)))

print("""
     ⛔ READING RULES (§4 of the scoreboard file)
       - b and the level on 'all land' are COMPARATORS, not estimates: 97.8% of cells have a window
         truth of exactly 0, so log(eps) = -13.8 dominates. Quote DIFFERENCES between arms measured
         identically, never the level of either number as a property of the model.
       - the 'truth > 0' and '336-mo base rate > 0' populations are defined by the TRUTH alone, so
         they are identical for every arm and cannot flatter one.
       - GLOSSARY.md:263 records the level as -0.5831 on the 100-cell set and flags it
         UNVERIFIED on the grid. The figures above settle that, and they are NOT comparable to
         -0.5831 — different population, so a different number is expected, not a
         contradiction.""")

print("\n" + "-" * 96)
print("RETAINED, not recomputed here: crps_events (FAO-02 comparability; its insensitivity is a")
print("property of the metric) and size-ratio (on every headline row, locked in GLOSSARY §4).")
print("-" * 96)

print("\n" + "=" * 96)
print("COST, measured")
print("=" * 96)
print("  wall time %.2f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
