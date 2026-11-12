"""M3b — what does a DRAW-DERIVED gate cost the battery's gate leg at S = 256?

⭐ WHY THIS EXISTS. M2/M3 scored eleven forecasts that each supplied their own gate array `p`.
A pure COUNT generator — conflictology, or views-baseline's `EmpiricalKDEMixtureConflictology`
α-mixture (views-baseline#116) — has no gate. But `AP` and `Brier` are the battery's only two
columns that read `p` (M3 Finding A), so scoring such a generator at all requires DEFINING one.

The precedent in this harness is the generator's own occurrence probability: `f_climatology`
uses the fit window's positive rate (`redteam_forecasts.py:188`). For a count generator that is
`p = (cs > 0).mean(1)`. ⛔ At S = 256 that is quantised to 1/256 = 0.003906, which is COARSER
than this population's base rate of 0.003448 — so the estimate's step size exceeds the
prevalence it is ranking. This measures what that costs, rather than arguing it.

⇒ Asked and answered for views-baseline#109, whose α-sweep must score a count generator on
this battery, on this population, at this S.

**C-351 slots.** Population = 13 months 457-469 x 64,818 land cells = 842,634 cell-months,
`ged_sb_best` counts, identical ordering to M2's. Transform = none (count space, no
normalisation). Estimator = `p_hat = Binomial(S, p) / S`, i.e. the exact distribution of
`(cs > 0).mean(1)` for a generator whose per-cell occurrence probability is `p`; mean and
sample sd (`ddof=1`) over 5 seeds. `AP`/`Brier` are IMPORTED from the frozen ruler.

**G-0, and it is the load-bearing gate.** Before perturbing anything, the continuous delivered
gate must reproduce M2's committed `separation_matrix.npz` reference row, `AP = 0.736228`,
exactly. That one check confirms the truth assembly, the population, the row ordering and the
frozen primitive all at once. If it fails, every number below is void.

⚠️ PROVENANCE, stated rather than implied. The first run of this probe happened in a scratchpad
BEFORE the script was committed, which inverts the dossier's D-5 discipline. This committed copy
was then run again from scratch and reproduced every figure identically; the `.out` beside it is
from this version. The ordering defect is recorded here because it is not visible in the output.

⚠️ SCOPE LIMIT — the one that matters, and it is not the obvious one. This probe perturbs the
delivered cube's GATE, so it isolates the cost of ESTIMATING an occurrence probability from S
draws. It does NOT model the fact that the delivered cube's own `(cs > 0).mean(1)` would sit
BELOW its gate, because a `gate x NB` draw can be zero when the gate fires. ⇒ For a generator
that has a separate gate the two quantities differ and this probe does not bound that gap. For a
count generator there IS no separate gate — `P(Y > 0)` is its occurrence probability — so the
probe is apt for exactly the case it was built for, and silent about the other.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "reports" / "2026-07-17_lodestar_eval_dossier" / "tools"))
from lodestar_score import average_precision, brier  # noqa: E402

#: ⚠️ C-347: this import runs `redteam_forecasts` in full (~80 s, ~8 GiB) to reuse ITS truth
#: assembly rather than re-deriving the population here. That is deliberate — a second copy of
#: the pivot/reindex is exactly how two artifacts come to disagree about which cells they scored.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from redteam_forecasts import GT, TRUTH, sco_m  # noqa: E402

M2_REFERENCE_AP = 0.736228  #: the committed separation_matrix.npz row, to 6 dp as printed
SEEDS = 5

YBIN = (TRUTH > 0).astype(np.float64)
p_cont = GT[:, sco_m, :].mean(0).ravel()  # the delivered gate, averaged over its 4 MC draws
N = p_cont.size

print()
print("=" * 94)
print("M3b — a draw-derived gate on the battery's gate leg")
print("=" * 94)
print(f"  population  N = {N:,}   positives {int(YBIN.sum()):,} "
      f"(base rate {YBIN.mean():.6f})")
print(f"  quantum     1/256 = {1 / 256:.6f}  ⇒ COARSER than the base rate")

ap_cont, br_cont = average_precision(YBIN, p_cont), brier(YBIN, p_cont)
ok = abs(ap_cont - M2_REFERENCE_AP) < 5e-7
print()
print("  G-0  the continuous delivered gate reproduces M2's committed reference row")
print(f"       AP = {ap_cont:.6f}  vs committed {M2_REFERENCE_AP:.6f}   "
      f"{'PASS' if ok else 'FAIL'}")
if not ok:
    raise SystemExit(
        f"VOID: G-0 failed. AP {ap_cont!r} != M2's committed {M2_REFERENCE_AP!r}. The population, "
        "the row ordering or the frozen primitive has moved, so every figure below would be a "
        "measurement of something other than what this probe claims. Nothing is reported."
    )

print()
print(f"  p_hat = Binomial(S, p)/S, mean +/- sample sd over {SEEDS} seeds")
print()
print("  S          AP                      Brier       distinct p   dAP vs continuous")
print("  " + "-" * 78)
for S in (256, 1024, 4096, 16384):
    aps, brs, nuniq = [], [], []
    for seed in range(SEEDS):
        rng = np.random.default_rng(1000 + seed)
        p_hat = rng.binomial(S, p_cont) / S
        aps.append(average_precision(YBIN, p_hat))
        brs.append(brier(YBIN, p_hat))
        nuniq.append(len(np.unique(p_hat)))
    a, b = np.array(aps), np.array(brs)
    print(f"  {S:<9}  {a.mean():.6f} +/- {a.std(ddof=1):.6f}   {b.mean():.6f}    "
          f"{int(np.mean(nuniq)):>7,}   {a.mean() - ap_cont:+.6f}")
print(f"  {'continuous':<9}  {ap_cont:.6f}               {br_cont:.6f}    "
      f"{len(np.unique(p_cont)):>7,}   {0.0:+.6f}")
print()
print("  ⇒ AP carries a SYSTEMATIC downward bias at S = 256, several times the seed sd; Brier")
print("    is nearly untouched. The bias is common to every forecast scored at the same S, so")
print("    it cancels in a paired difference and does NOT cancel in an absolute level.")
