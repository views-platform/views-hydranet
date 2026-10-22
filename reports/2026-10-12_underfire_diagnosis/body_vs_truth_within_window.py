"""Is the body DARKEST where the conflict is? — the within-window check.

Simon read it off three training-forensic plots: in all three, the body's field looks darkest
exactly where the truth has events and brightest where the map is empty. That contradicts what
`body_scoreboard.py` measured this afternoon on the same vehicle, where the body's relationship to
the truth on the global land grid came out POSITIVE (b 1.3095, r +0.5835 on truth > 0 cells).

⛔ BOTH CAN BE TRUE AT ONCE, and that is the whole point of this check. The global correlation is
dominated by BETWEEN-REGION variation — the conflict belt has both more events and a larger body
than the Sahara does, which makes r positive no matter what the body does locally. The plots show
32x32 WINDOWS, where the regional level is held nearly constant and only local variation is
visible. A field can be positively correlated between regions and negatively correlated inside
them. That is Simpson's paradox and it is not exotic; it is the default when a strong regional
confounder is present.

THE TEST, with no fitted parameter. Partition the land grid into k x k blocks. Inside each block
that contains BOTH event and non-event cells, compare the body's median on the event cells against
its median on the non-event cells. The block is its own control, so the regional level cancels.

  ⇒ positive difference  = the body is LARGER where the conflict is  (Simon's reading is wrong)
  ⇒ negative difference  = the body is SMALLER where the conflict is (Simon's reading is right)

⛔ k IS SWEPT (8, 16, 32, 64), because a single block size is a free parameter and this repo has
withdrawn three claims that crossed the value they were testing as their bin count moved.

⛔ PRE-STATED BEFORE RUNNING, so this is a test and not an exercise. I expect the within-block
difference to be NEGATIVE or indistinguishable from zero, and the within-block correlation to be
much weaker than the global +0.5835 — i.e. I expect Simon to be right and my own grid number to be
a between-region artifact. If the within-block difference is clearly POSITIVE at every k, then the
plots are showing something specific to training windows that the delivered artifact does not do,
and I report that instead.

⚠️ WHAT THIS CANNOT SETTLE. The plots are single-month 32x32 curriculum windows seen DURING
training, on whichever target that forensic run plotted. This reads the DELIVERED artifact, the
`sb` target only, averaged over the in-sample window. A disagreement would not mean either is
wrong — it would mean the effect is specific to training-time windows, which is a different and
also interesting finding.

Population: all 64,818 land cells, months 421-456, no admission filter.
Transform: none on the body (it is compared to itself within a block); events are truth > 0.
Estimator: median within block, paired difference across blocks, reported with sign counts.
"""
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vehicle import SB, TARGETS, emitted_magnitude, grid_rowcol, pull  # noqa: E402

SP = Path(__file__).parent
IS_LO, IS_HI = 421, 456
BLOCKS = (8, 16, 32, 64)
t_start = time.time()

npz = np.load(SP / "d4_result_tnb.npz")
family = str(npz["family"])
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {m: i for i, m in enumerate(months)}
j = [mcol[m] for m in range(IS_LO, IS_HI + 1)]

body = emitted_magnitude(npz["mu"].astype(np.float64),
                         npz["theta"].astype(np.float64), family).mean(0)[j].mean(0)
gate = npz["gate"].astype(np.float64).mean(0)[j].mean(0)

df = pull((121, 504), SP / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= IS_LO) & (df.month_id <= IS_HI)]
truth = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
         .reindex(land_cells).to_numpy(np.float64).mean(1))

row, col = grid_rowcol(land_cells.astype(np.int64))
event = truth > 0

print("=" * 96)
print("IS THE BODY DARKEST WHERE THE CONFLICT IS? — within-window, the block is its own control")
print("=" * 96)
print(f"  vehicle        {str(npz['sha'])[:16]}…  family {family}")
print(f"  population     {len(land_cells):,} land cells, months {IS_LO}-{IS_HI}, target {TARGETS[SB]}")
print(f"  event cells    {int(event.sum()):,}  ({100 * event.mean():.2f}%)")

lg_b, lg_t = np.log(body + 1e-6), np.log(truth + 1e-6)
print("\n  THE GLOBAL NUMBER, for contrast (this is what the scoreboard reports)")
print("    pearson(log body, log truth), all land            %+.4f" % np.corrcoef(lg_b, lg_t)[0, 1])
print("    pearson(log body, log truth), event cells only    %+.4f"
      % np.corrcoef(lg_b[event], lg_t[event])[0, 1])

print("\n  THE WITHIN-BLOCK TEST")
print("    %-4s %8s %12s %12s %10s %10s %9s" % (
    "k", "blocks", "med body on", "med body on", "median", "blocks", "sign"))
print("    %-4s %8s %12s %12s %10s %10s %9s" % (
    "", "used", "event cells", "empty cells", "diff", "negative", "test p"))
for k in BLOCKS:
    bid = ((row - 1) // k) * (720 // k + 1) + (col - 1) // k
    diffs, ev_all, em_all = [], [], []
    for b in np.unique(bid):
        m = bid == b
        e, n = m & event, m & ~event
        if e.sum() < 1 or n.sum() < 1:
            continue
        me, mn = float(np.median(body[e])), float(np.median(body[n]))
        diffs.append(me - mn)
        ev_all.append(me)
        em_all.append(mn)
    d = np.asarray(diffs)
    neg = int((d < 0).sum())
    # exact two-sided sign test against "no difference"
    from scipy.stats import binomtest
    p = binomtest(neg, len(d), 0.5).pvalue if len(d) else float("nan")
    print("    %-4d %8d %12.4f %12.4f %+10.4f %9d%% %9.2e" % (
        k, len(d), float(np.median(ev_all)), float(np.median(em_all)),
        float(np.median(d)), round(100 * neg / len(d)), p))

print("""
  READING. A negative median difference at every k, with most blocks negative, means the body is
  systematically SMALLER on the cells where conflict actually happened than on the empty cells
  beside them — and the positive global correlation is then a between-region artifact.""")

print("\n" + "=" * 96)
print("COST, measured")
print("=" * 96)
print("  wall time %.2f s" % (time.time() - t_start))
print("  peak RSS  %.2f GiB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
