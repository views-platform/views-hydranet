"""L3c (#471) — does the family-free lookup work on `ns` and `os`, or only on `sb`?

The second half of L3, and the half nothing on disk could answer until now: every cube in this
repository stores ONE target. `l3b_three_target_cube.py` wrote a three-target cube from the exact
vehicle the published `sb` number came off (sha `4acbf4d3…`, family `nb`, dropout 0.15), and its
G-2 proved the `sb` slice **bit-identical** to `d4_result.npz`. ⇒ Everything below is measured on
the same vehicle, the same months, the same draws and the same dropout masks as the headline.

⛔ NO CROSS-FAMILY CAVEAT APPLIES HERE. That is new, and it is the whole value of the checkpoint
turning out to exist (C-394): the three targets are three heads of ONE model on ONE pass, so a
difference between them cannot be a family effect, a dropout effect, a seed effect or a vintage
effect. It can only be the target.

⛔ AND NO PARAMETRIC COMPARATOR EXISTS FOR `ns` OR `os`. M67's `0.046023` and `0.050758` are `sb`
numbers on months 457–469. Nothing was ever scored for the other two targets. ⇒ The margin reported
is the same INTERNAL one L3a used: CRPS at the best bin count against **B = 1**, the unconditional
empirical CDF of the same fit window, with the model's index never consulted. It transports across
targets for the same reason it transported across windows — both halves move together.

C-351 SLOTS, declared:
  · population  per target, fit = months 421–456 x 64,818 land cells = 2,333,448 cell-months;
                score = months 457–469 x 64,818 = 842,634. ⭐ M67's own split, unchanged, so the
                `sb` column is directly comparable to the published ladder.
  · transform   count space, no log. The index is in count units too.
  · estimator   CRPS of the bin's empirical distribution, exact via the sorted identity; BOTH the
                plug-in 1/n^2 and the fair 1/(n(n-1)) forms reported (#442 is undecided).
  · normalised  no. Absolute CRPS in count units, plus the internal B=1 margin as a percentage.
  ⛔ The three targets have very different base rates, so their ABSOLUTE CRPS levels are NOT
    comparable to each other. Positive rates are printed per target for exactly that reason, and
    the cross-target claim is made on the MARGIN alone.

⛔ FREE PARAMETERS: bin count B over {1, 50, 250, 1k, 2k, 4k, 8k, 16k, 32k} — M67's ladder plus
B=1, the internal comparator. Binning rule and out-of-range clipping are inherited unchanged from
`family_free_cdf.py` via `binned_cdf.py`, because G-S2 requires this route to reproduce that
script's published number exactly.

⛔ THE GATES, declared before the run:
  G-S0  the sorted-pairwise identity against brute force on foreign input (`binned_cdf.self_test`).
  G-S1  ⭐ SOURCE IDENTITY. `binned_cdf.binned_cdf_crps` and the inline copy in
        `l3a_split_sensitivity.py` must parse to the SAME abstract syntax tree, docstrings
        excluded. Two scripts quoting one statistic must be running one implementation; this
        asserts it mechanically rather than by my word.
  G-S2  ⭐⭐ THE ANCHOR. The `sb` column at B = 8,000 must reproduce `family_free_cdf.out`'s own
        0.039186 (plug-in) and 0.039026 (fair) to 1e-6. ⚠️ Anchored on the COMMITTED
        IMPLEMENTATION, never on M67's ledger value 0.039327, which no implementation has ever
        produced — that was W-98, and the lesson is applied here rather than repeated.
  G-S3  the cube's stored sha equals `d4_result.npz`'s, so G-S2's anchor is legitimate.
  G-S4  NON-VACUITY: the three targets do not produce the same CRPS curve.
  G-S5  COVERAGE: for every (target, B), the fraction of score cell-months landing in an empty fit
        bin must be < 1% — those score as nan and are dropped, silently changing the population.

Run: python reports/2026-10-28_rp3_l3/l3c_three_target_scores.py
"""

import ast
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
SP = Path(__file__).parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(D4))
sys.path.insert(0, str(SP))

from binned_cdf import binned_cdf_crps, self_test  # noqa: E402
from vehicle import emitted_magnitude, pull  # noqa: E402

IS_LO, IS_HI = 421, 456
OS_LO, OS_HI = 457, 469
B_SWEEP = (1, 50, 250, 1_000, 2_000, 4_000, 8_000, 16_000, 32_000)
COMMITTED_B8000_PLUGIN = 0.039186
COMMITTED_B8000_FAIR = 0.039026
LEDGER_M67 = 0.039327              #: ⛔ NOT A TARGET. Never reproduced by any implementation.
PARAMETRIC_SB_ONLY = 0.046023      #: ⛔ an `sb` number on months 457-469. Never quoted for ns/os.

t_start = time.time()

# ── G-S0 ──────────────────────────────────────────────────────────────────────────────────
self_test()
print("  PASS  G-S0  sorted pairwise identity == brute force, foreign input")


# ── G-S1: the two copies of the scorer must parse identically ─────────────────────────────
def _fn_ast(src: str, name: str) -> str:
    """Dump the named function's AST with its docstring removed, so comments and prose differ
    freely while any difference in EXECUTABLE code is caught."""
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
            return ast.dump(ast.Module(body=body, type_ignores=[]))
    raise SystemExit(f"⛔ G-S1 VOID: no function named {name!r} found")


a_src = (SP / "binned_cdf.py").read_text()
b_src = (SP / "l3a_split_sensitivity.py").read_text()
g_s1 = _fn_ast(a_src, "binned_cdf_crps") == _fn_ast(b_src, "binned_cdf_crps")
print(f"  {'PASS' if g_s1 else '⛔ FAIL'}  G-S1  binned_cdf.py and l3a_split_sensitivity.py hold "
      f"the SAME scorer (AST, docstring excluded)")
if not g_s1:
    raise SystemExit("⛔⛔ ABORT — two implementations of one statistic have diverged.")

# ── the three-target cube ─────────────────────────────────────────────────────────────────
cube_path = SP / "l3b_cube3.npz"
npz = np.load(cube_path)
FAMILY = str(npz["family"])
land_cells, months = npz["land_cells"], list(npz["months"])
mcol = {int(m): i for i, m in enumerate(months)}
cube_targets = [str(x) for x in npz["targets"]]
MU, TH, GT = npz["mu"], npz["theta"], npz["gate"]

ref_sha = str(np.load(D4 / "d4_result.npz")["sha"])
g_s3 = str(npz["sha"]) == ref_sha
print(f"  {'PASS' if g_s3 else '⛔ FAIL'}  G-S3  cube sha == d4_result.npz's sha "
      f"({str(npz['sha'])[:16]}…) ⇒ the sb anchor is legitimate")
if not g_s3:
    raise SystemExit("⛔⛔ ABORT — G-S2's anchor does not apply to this cube.")

print()
print("=" * 99)
print("L3c (#471) — THE FAMILY-FREE LOOKUP ON ALL THREE TARGETS, one vehicle, one pass")
print("=" * 99)
print(f"  cube          {cube_path.name}  sha {str(npz['sha'])[:16]}…  family {FAMILY}  "
      f"dropout {npz['dropout_rate']}  draws {MU.shape[1]}")
print(f"  targets       {cube_targets}  (axis 0 of {MU.shape})")
print(f"  split         fit {IS_LO}-{IS_HI}, score {OS_LO}-{OS_HI} — M67's own, unchanged")
print("  ⛔ no cross-family caveat: three heads of ONE model on ONE pass, same dropout masks")
print()

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
fit_m = [mcol[m] for m in range(IS_LO, IS_HI + 1) if m in mcol]
sco_m = [mcol[m] for m in range(OS_LO, OS_HI + 1) if m in mcol]

summary = []
curves: dict[str, dict[int, tuple[float, float, float, float]]] = {}
worst_void = 0.0
for j, tgt in enumerate(cube_targets):
    mag = emitted_magnitude(MU[j].astype(np.float64), TH[j].astype(np.float64), FAMILY)
    #: ⛔ C-386: E[gate x mu], per draw THEN averaged — never E[gate] x E[mu].
    index = (GT[j].astype(np.float64) * mag).mean(0)
    Y = (w.pivot(index="priogrid_id", columns="month_id", values=tgt)
         .reindex(land_cells)[months].to_numpy(np.float64)).T
    x_fit, y_fit = index[fit_m].ravel(), Y[fit_m].ravel()
    x_sco, y_sco = index[sco_m].ravel(), Y[sco_m].ravel()
    curves[tgt] = {B: binned_cdf_crps(x_fit, y_fit, x_sco, y_sco, B) for B in B_SWEEP}

    pos = float((y_sco > 0).mean())
    print(f"  {tgt}   score-window positive fraction {pos:.6f}   "
          f"index range {index.min():.3e} .. {index.max():.3f}")
    print("    %6s %11s %11s %11s %10s %9s"
          % ("bins", "plug-in", "fair", "vs B=1", "n/bin", "void%"))
    base = curves[tgt][1][0]
    best = min(curves[tgt], key=lambda k: curves[tgt][k][0])
    for B in B_SWEEP:
        p, f_, o, vd = curves[tgt][B]
        worst_void = max(worst_void, vd)
        print("    %6d %11.6f %11.6f %+10.1f%% %10.1f %8.3f%%%s"
              % (B, p, f_, 100.0 * (p / base - 1.0), o, 100.0 * vd,
                 "  <- best" if B == best else ""))
    print()
    summary.append((tgt, pos, best, curves[tgt][best][0], curves[tgt][best][1], base,
                    100.0 * (curves[tgt][best][0] / base - 1.0)))

# ── the gates that need the numbers ───────────────────────────────────────────────────────
sb = cube_targets[0]
g_s2 = (
    abs(curves[sb][8_000][0] - COMMITTED_B8000_PLUGIN) < 1e-6
    and abs(curves[sb][8_000][1] - COMMITTED_B8000_FAIR) < 1e-6
)
g_s4 = len({round(curves[t][8_000][0], 9) for t in cube_targets}) == 3
g_s5 = worst_void < 0.01

print("=" * 99)
print("  THE ANSWER — the internal margin, per target")
print("=" * 99)
print("  %-14s %10s %7s %11s %11s %11s %11s"
      % ("target", "pos rate", "B*", "CRPS(B*)", "fair", "CRPS(B=1)", "margin"))
for tgt, pos, best, c_p, c_f, c_base, marg in summary:
    print("  %-14s %10.6f %7d %11.6f %11.6f %11.6f %+10.1f%%"
          % (tgt, pos, best, c_p, c_f, c_base, marg))
print()
print(f"  {'PASS' if g_s2 else '⛔ FAIL'}  G-S2  the sb column at B=8,000 reproduces the "
      f"committed "
      f"{COMMITTED_B8000_PLUGIN:.6f} / {COMMITTED_B8000_FAIR:.6f} to 1e-6: "
      f"got {curves[sb][8_000][0]:.6f} / {curves[sb][8_000][1]:.6f}")
print(f"           ⚠️ NOT gated: M67's ledger value {LEDGER_M67:.6f} is reproduced by NO "
      f"implementation (gap {curves[sb][8_000][0] - LEDGER_M67:+.6f}, "
      f"{100.0 * (curves[sb][8_000][0] / LEDGER_M67 - 1.0):+.2f}%) — open item of #460, W-98.")
print(f"  {'PASS' if g_s4 else '⛔ FAIL'}  G-S4  NON-VACUITY: the three targets give "
      f"{len({round(curves[t][8_000][0], 9) for t in cube_targets})} distinct CRPS values at "
      f"B=8,000")
print(f"  {'PASS' if g_s5 else '⛔ FAIL'}  G-S5  COVERAGE: worst void fraction over all "
      f"(target, B) is {100.0 * worst_void:.4f}%, bar 1%")

print()
print("=" * 99)
print("  HOW TO READ THIS — printed, not left to the reader")
print("=" * 99)
print("  ⛔ The three ABSOLUTE CRPS levels are NOT comparable to each other. The targets have")
print("     different base rates (printed above), and a rarer target scores lower for free.")
print("     The cross-target claim is the MARGIN column and nothing else.")
print(f"  ⛔ {PARAMETRIC_SB_ONLY} and 0.050758 are `sb` numbers on months 457-469. There is NO")
print("     parametric comparator for ns or os, so none is printed. The B=1 margin replaces it")
print("     and measures a weaker thing: conditioning value, not superiority over a")
print("     parametric head.")
print("  ⭐ What a held margin on all three DOES establish: the lookup is reading something real")
print("     in the trunk's ranking, not something peculiar to the sb head.")
print("  ⛔ ONE SPLIT, ONE CHECKPOINT, ONE SEED. L3a swept the split for sb only. Nothing here is")
print("     a replication, and no σ̂ exists for CRPS on this route (C-394: the vehicle is on disk,")
print("     but there is still no second seed in the `nb` @ 0.15 cell —")
print("     reproducible ≠ resolvable).")
print()
ok = g_s1 and g_s2 and g_s3 and g_s4 and g_s5
print("  ALL GATES PASS" if ok else "  ⛔⛔ A GATE FAILED — the numbers above are NOT usable")
print("  wall time %.1f s   peak RSS %.2f GiB"
      % (time.time() - t_start, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2))
raise SystemExit(0 if ok else 1)
