"""Is the body's magnitude level a real ~10x shortfall, or a selection effect? — the flag check.

Simon asked for this directly. `2026-10-21_body_scoreboard.md` §6 flagged that B-3's level FLIPS
SIGN across truth-defined populations — 9.7% on cells with truth > 0 in the 36-month window, 119.5%
on cells with a positive 336-month base rate — and that if the second reading holds, the body's
SPREAD is not the problem and the loss arm is only about spatial structure. That is a narrower
claim than the design notes make, so it has to be settled before the arm is designed.

TWO THINGS ARE CONFOUNDED IN THAT SIGN FLIP and this separates them:

  1  THE POPULATION. One is selected by activity in the SAME 36 months the comparison uses, which
     selects observations whose own sampling noise biases them UP (C-351 instances #1 and #3), so
     the prediction/truth ratio is biased DOWN. The other is selected over 336 months.

  2  ⛔ THE TRUTH'S OWN WINDOW. The 336-month population is compared against the 36-month window
     truth, so a cell can be in that population and still have truth EXACTLY 0 in the comparison
     window — in which case log(0 + eps) = -13.8 drives both the median and the regression. If that
     share is large, the 119.5% reading is an ATOM artifact, not a reading about the body.
     That is D8-A's window mismatch wearing a different hat.

THE SEPARATION, and it needs no new instrument. Hold the POPULATION fixed at the 1,431 cells with
truth > 0 in the window, and change only the TRUTH ESTIMATOR: compare the gated forecast against
(a) those cells' 36-month window mean, and (b) their 336-month base rate — a less noisy estimate of
the same per-month rate on the same cells. Selection is then identical on both sides and only the
estimator moves. ⇒ If the shortfall survives (b), it is the body. If it closes, it was selection.

⛔ PRE-STATED BEFORE RUNNING, so this is a test and not an exercise: I expect the shortfall to
SHRINK but to stay large — level somewhere in 20-50%, i.e. still a 2-5x deficit, not a 10x one and
not closed. If it closes past ~80% I was wrong about the body being timid at all.

Reads `d4_result.npz` and the cached land pull. Seconds. Cost printed.
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, TARGETS, emitted_magnitude, pull  # noqa: E402

SP = Path(__file__).parent
EPS = 1e-6
IS_LO, IS_HI = 421, 456
t_start = time.time()

def _family_of(npz, name):
    """Which family wrote this npz. Loud fallback: only the frozen nb control lacks the key."""
    if "family" in npz.files:
        return str(npz["family"])
    print(f"  \u26a0 {name} carries no `family` key (it predates Amendment 4) \u21d2 assuming 'nb'. "
          "Its provenance file establishes this; any NEW artifact records it.")
    return "nb"

#: Optional filename suffix, so a SECOND vehicle can be measured without overwriting the first.
#: ⛔ Default "" keeps every existing filename byte-for-byte: the control's `d4_result.npz` was
#: produced by a checkpoint that NO LONGER EXISTS on disk (see
#: `d4_result_CONTROL_nb_4acbf4d3.PROVENANCE.md`), so an unconditional `out_path` made reading out
#: a second artifact destroy the only copy of the first. Found by a readiness falsification audit.
TAG = ""
for _a in sys.argv[1:]:
    if _a.startswith("--tag="):
        TAG = _a.split("=", 1)[1]


npz = np.load(SP / f"d4_result{TAG}.npz")
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {m: i for i, m in enumerate(months)}
FAMILY = _family_of(npz, f"d4_result{TAG}.npz")
mu_pm = emitted_magnitude(npz["mu"].astype(np.float64),
                          npz["theta"].astype(np.float64), FAMILY).mean(0)
gt_pm = npz["gate"].mean(0).astype(np.float64)

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64))
base336 = (df[df.month_id <= 456].groupby("priogrid_id")[TARGETS[SB]].mean()
           .reindex(land_cells).to_numpy(np.float64))

j = [mcol[m] for m in range(IS_LO, IS_HI + 1)]
pred = gt_pm[j].mean(0) * mu_pm[j].mean(0)          # the gated forecast, per month
y_w = Y[:, j].mean(1)                                # 36-month window truth, per month

print("=" * 96)
print("THE FLAG: is the ~10x magnitude shortfall real, or selection?")
print("=" * 96)

# ---- part 1: is the 119.5% reading an atom artifact?
pop336 = base336 > 0
zero_in_window = pop336 & (y_w == 0)
print("\n  1. THE 336-MONTH POPULATION, CHECKED FOR THE ATOM")
print("     cells with base336 > 0                     %6d" % pop336.sum())
print("     of those, truth EXACTLY 0 in 421-456       %6d  = %.1f%%"
      % (zero_in_window.sum(), 100 * zero_in_window.sum() / pop336.sum()))
print("     ⇒ %s"
      % ("⛔ ATOM-DOMINATED: the median and the regression are driven by log(eps) = -13.8, so "
         "119.5%% is NOT a reading about the body"
         if zero_in_window.sum() / pop336.sum() > 0.5 else
         "✅ a minority: the reading is not atom-driven"))

# ---- part 2: hold the population fixed, change only the truth's estimator
keep = y_w > 0
print("\n  2. POPULATION FIXED (%d cells, truth > 0 in window); ONLY THE TRUTH ESTIMATOR MOVES"
      % keep.sum())
print("     %-34s %8s %9s %9s %9s" % ("truth estimator", "b", "r", "level", "implied"))
for tname, truth in (("36-month window mean", y_w), ("336-month base rate", base336)):
    x, y = np.log(pred[keep] + EPS), np.log(truth[keep] + EPS)
    b, _a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    level = float(np.median(x - y))
    print("     %-34s %8.4f %+9.4f %+9.4f %8.1f%%"
          % (tname, b, r, level, 100 * np.exp(level)))

# ---- part 3: and the same on the ONE population where neither side is selected or atom-driven
both = (base336 > 0) & (y_w > 0)
print("\n  3. BOTH-POSITIVE population (%d cells): no atom on either side" % both.sum())
print("     %-34s %8s %9s %9s %9s" % ("truth estimator", "b", "r", "level", "implied"))
for tname, truth in (("36-month window mean", y_w), ("336-month base rate", base336)):
    x, y = np.log(pred[both] + EPS), np.log(truth[both] + EPS)
    b, _a = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    level = float(np.median(x - y))
    print("     %-34s %8.4f %+9.4f %+9.4f %8.1f%%"
          % (tname, b, r, level, 100 * np.exp(level)))

print("""
  ⛔ WHAT THIS CANNOT SETTLE. The predictor is a 36-month model window and the 336-month base rate
  spans a regime change (D8-A's confound, §10.8). Comparing the two assumes the model's 36-month
  mean stands in for a 336-month one, which is the stationarity assumption D8-A violated.
  ⇒ Part 2's second row is a BOUND, not an estimate, and the direction of the bound is what this
  check is for.""")

print("\n" + "=" * 96)
print("COST, measured")
print("=" * 96)
print("  wall time %.2f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
