"""M-1 and M-2 of `../2026-10-26_scoreboard_update_plan.md`: B-1b's baseline on B-1a's population.

⛔ A RULER's baseline, not a result. The plan is committed and decided: Simon chose to SPLIT B-1
in place rather than add a fourth number, so B-1 becomes a pair —

  B-1a  dispersion   gamma_model(d)/gamma_truth(d), the existing variogram ratio. UNCHANGED.
  B-1b  placement    count-matched pooled FSS, the instrument the 2026-10-26 battery adopted.

⛔ The FSS numbers in `../2026-10-26_fss_battery_outcome.md` DO NOT GO ON THE SCOREBOARD. They were
measured on an 18-field, 24x24, in-sample window with a median of 1 event per field and
f_o = 0.009452 — not this population, not its base rate, not its bar. This script produces the
numbers that may go on it, and the plan's §2 forbids applying the edit before they exist.

WHAT IS MEASURED
  M-1  pooled FSS on B-1a's exact population: all 64,818 land cells, sb, NO admission filter, NO
       activity selection, in sample 421-456 and out of sample 457-469, NEVER pooled together.
  M-2  f_o on that population, so the skill bar 0.5 + f_o/2 (Roberts & Lean 2008 p.83) is THIS
       population's bar and not a borrowed one.
  M-4  whether scale_min is stable, read as the spread of the per-month scale_min across the
       window's fields. The 24x24 battery window overshot the ideal-band ruler by 1-3 cells and
       nothing established whether that grows on the real grid.

  ⛔ M-3 (sigma-hat, the noise floor) is NOT here and cannot be. It needs the seed replicates —
  views-models#625. Until it exists B-1b is a baseline and carries no decision rule.

THREE DECLARED ESTIMATOR CHOICES (C-351). Each differs from B-1a and is declared, not hidden.

  1. FIELD = ONE MONTH, pooled within the window. B-1a reads a single window-mean field; a
     placement readout needs the events themselves, so the field is a month and the FBS is summed
     over the window's months BEFORE the ratio. The two halves share the POPULATION and the
     WINDOWS, which is what makes them readable side by side; they do not share the field.
  2. TRUTH = `truth > 0` in that month. B-1a's truth is the 336-month base rate (its noise
     control); binarising a base rate gives a static mask with no placement content.
  3. PREDICTOR = the BODY ALONE (`mu`), primary, because B-1a's field is the body and the pair has
     to be read on one object. The GATED FORECAST (`gate x mu`) is reported alongside — it is the
     real forecast and B-3's primary. ⛔ FSS is a function of RANK ORDER alone (proven exactly,
     outcome §2), and those two fields have different rank orders, so this choice is load-bearing
     in a way no transform choice is.

OCEAN. The grid is 360x720 and 94.4% of it is not land. Production `fss` means over the whole
grid — and FSS's ratio is sum(num)/sum(den), so cells where BOTH fields are zero cancel exactly
and contribute nothing. What does contribute is the band where a wide window reaches off the
coast: that is Roberts & Lean's own `mode="constant"` domain treatment, kept rather than patched.

Reuses the PRODUCTION `fss` and `matched_pred_thresh` (`scripts/sharpness_scorecard.py:78,92`) and
asserts a single pooled field equals production exactly (gate C-1). Cost printed, per the standing
rule. Local CPU, no pod, no money.
"""

import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reports" / "2026-10-12_underfire_diagnosis"))

from vehicle import (  # noqa: E402
    GRID_COLS,
    GRID_ROWS,
    SB,
    TARGETS,
    emitted_magnitude,
    grid_rowcol,
    pull,
)

from scripts.sharpness_scorecard import fss, matched_pred_thresh  # noqa: E402

D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
ODD = list(range(1, 82, 2))          # every odd n to 81; scale_min is swept, never read off a grid
SHOW = (1, 3, 5, 9, 17, 33, 65)      # the scales the plan's Change 1 commits to printing
t_start = time.time()


def bar(f_o: float) -> float:
    """Roberts & Lean 2008 p.83: a UNIFORM forecast scores 0.5 + f_o/2. Below it = no skill."""
    return 0.5 + f_o / 2.0


def scale_min(scores: dict[int, float], target: float) -> int | None:
    """Smallest n whose pooled FSS reaches the bar. None = never reached in the swept range."""
    for n in sorted(scores):
        if np.isfinite(scores[n]) and scores[n] >= target:
            return n
    return None


def to_grid(values: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """Land values onto the full 360x720 field. Ocean = 0.0, which FSS's ratio cancels."""
    g = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float64)
    g[rr, cc] = values
    return g


def pooled_fss(preds, truths, scales):
    """FBS pooled over fields BEFORE the ratio. One field MUST equal production `fss` (C-1)."""
    num = dict.fromkeys(scales, 0.0)
    den = dict.fromkeys(scales, 0.0)
    for p, t in zip(preds, truths, strict=True):
        prd = (p > matched_pred_thresh(p, t)).astype(np.float64)
        obs = (t > 0.0).astype(np.float64)
        for n in scales:
            pf = uniform_filter(prd, size=n, mode="constant")
            of = uniform_filter(obs, size=n, mode="constant")
            num[n] += float(np.mean((pf - of) ** 2))
            den[n] += float(np.mean(pf**2) + np.mean(of**2))
    return {n: (1.0 - num[n] / den[n] if den[n] > 0 else float("nan")) for n in scales}


# ------------------------------------------------------------------ the vehicle
npz = np.load(D4 / "d4_result.npz")
if "family" in npz.files:
    FAMILY = str(npz["family"])
else:
    FAMILY = "nb"
    print("  ⚠ d4_result.npz carries no `family` key (it predates Amendment 4) ⇒ 'nb'. "
          "Its PROVENANCE file establishes this; emitted_magnitude is the identity under nb.")
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}
MAG = emitted_magnitude(npz["mu"].astype(np.float64), npz["theta"].astype(np.float64),
                        FAMILY)
MU_PM = MAG.mean(0)
#: ⛔ THE GATED FORECAST IS E[gate x mu], NOT E[gate] x E[mu]. Averaging the two halves over the
#: four MC-dropout draws and multiplying afterwards drops Cov(gate, mu), and the covariance is not
#: zero by construction: both heads sit on one shared trunk and see the SAME dropout mask on every
#: draw. Found by `/code-review max` on PR #457 and measured by `b1b_composition_check.py`:
#: Cov > 0 in 51.9% of land cell-months and < 0 in 48.1%, mean relative difference +1.96% per cell,
#: total emitted mass +0.43%. ⭐ FSS barely notices, because `matched_pred_thresh` is a
#: count-matched quantile and FSS is therefore blind to any monotone distortion of the field —
#: measured at -0.0007 (in sample) and +0.0017 (out of sample) at n=1, with the sign flipping
#: between windows, so it is not even a bias. ⚠️ It is composed correctly ANYWAY, because the next
#: statistic to read this field may be magnitude-sensitive rather than rank-based, and then the
#: 1.96% is not negligible. `views_hydranet/utils/hydranet_inference.py:995` carries the same
#: warning about the same quantity.
GATED_PM = (npz["gate"].astype(np.float64) * MAG).mean(0)

rows, cols = grid_rowcol(land_cells)
RR, CC = rows - 1, cols - 1

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64))

print("=" * 98)
print("B-1b — THE PLACEMENT HALF OF B-1.  baseline on the CONTROL artifact")
print("=" * 98)
print("  plan ../2026-10-26_scoreboard_update_plan.md (M-1, M-2, M-4) · decision: SPLIT in place")
print(f"  vehicle sha {str(npz['sha'])[:16]}…  family {FAMILY}  land cells {len(land_cells):,}"
      f"  grid {GRID_ROWS}x{GRID_COLS}")
print(f"  target {TARGETS[SB]}  ·  predictors: body alone (mu) PRIMARY, gated forecast alongside")

# ------------------------------------------------------------------ C-1, the reuse gate
_j = mcol[IS_LO]
_p = to_grid(MU_PM[_j], RR, CC)
_t = to_grid(Y[:, _j], RR, CC)
_prod = fss(_p, _t, SHOW, matched_pred_thresh(_p, _t))
_pool = pooled_fss([_p], [_t], SHOW)
_gap = max(abs(_prod[n] - _pool[n]) for n in SHOW)
print(f"\n  C-1 one pooled field vs PRODUCTION fss, max |diff| over n={SHOW}: {_gap:.3e}")
if _gap != 0.0:
    raise SystemExit("VOID: the pooled estimator does not reduce to production `fss` on 1 field.")

# ------------------------------------------------------------------ M-1, M-2, M-4
for wname, (lo, hi) in (("in sample", (IS_LO, IS_HI)), ("out of sample", (OS_LO, OS_HI))):
    j = [mcol[m] for m in range(lo, hi + 1)]
    truths = [to_grid(Y[:, k], RR, CC) for k in j]
    f_o = float((Y[:, j] > 0).mean())                        # M-2, on THIS population
    the_bar = bar(f_o)
    print("\n" + "-" * 98)
    print(f"{wname}  months {lo}-{hi}  ({len(j)} monthly fields)")
    print("-" * 98)
    print(f"  M-2  f_o on all land cell-months in the window = {f_o:.6f}"
          f"   ⇒ bar 0.5 + f_o/2 = {the_bar:.6f}"
          f"   (mean events/field {float((Y[:, j] > 0).sum()) / len(j):.1f})")

    for predname, field in (("body alone (mu)", MU_PM), ("gated forecast (gate x mu)",
                                                         GATED_PM)):
        preds = [to_grid(field[k], RR, CC) for k in j]
        sc = pooled_fss(preds, truths, ODD)
        print(f"\n  {predname}")
        print("    M-1  pooled FSS   "
              + "  ".join(f"n{n}={sc[n]:.4f}" for n in SHOW))
        smin = scale_min(sc, the_bar)
        print("    M-1  scale_min (smallest n reaching the bar) = "
              + (f"{smin}" if smin is not None else f"NEVER within n<={ODD[-1]}"))
        per = [scale_min(pooled_fss([p], [t], ODD), the_bar) for p, t in zip(preds, truths)]
        got = [s for s in per if s is not None]
        print(f"    M-4  per-field scale_min: {len(got)}/{len(per)} fields reach the bar"
              + (f"  min {min(got)}  median {int(np.median(got))}  max {max(got)}" if got else ""))
        if got:
            print(f"         per-field values: {per}")

print("\n" + "=" * 98)
print(f"cost  {time.time() - t_start:.2f} s  /  "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB peak RSS")
print("⚠ M-3 (sigma-hat) is absent and blocked on views-models#625. Until it exists these are")
print("  BASELINES and B-1b carries NO decision rule.")
