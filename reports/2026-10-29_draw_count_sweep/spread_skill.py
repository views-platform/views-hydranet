"""Q2, amendment — the SPREAD–SKILL diagnostic `draw_sweep.py` promised and did not deliver.

WHY THIS EXISTS, AND IT IS A DEFECT IN MY OWN PRIOR SCRIPT
---------------------------------------------------------
`draw_sweep.py`'s docstring promised *"a spread–error relation (binned predicted ensemble
sd vs realised |error|)"*. ⛔ **It computed a GLOBAL `sd/RMSE` ratio instead and never
binned anything.** That ratio is confounded, and badly:

    RMSE on this population is **3.674** against a truth mean of **0.060**, because a
    handful of extreme events — truth max **2482** — dominate the squared error. The
    single largest event contributes more than half of it.

⇒ `sd/RMSE = 0.03` therefore conflates *"the ensemble is too narrow"* with *"truth has
outliers no forecast could hit"*. **That is a `W-51` violation inside my own metric**, in
a script whose whole subject is a statistic nobody had checked. It is logged, not
quietly patched over.

WHAT THE SWEEP DID ESTABLISH, and this script takes as given
------------------------------------------------------------
  * ⭐ **The ensemble is CONVERGED in the draw count.** sd moves 0.113698 (K=256) ->
    0.115288 (K=1024), **1.4% apart**, so K=256 is the converged operating point and
    this diagnostic needs only that point rather than the whole sweep.
  * ⭐ The delivered 16-member ensemble is **38% narrower** than the model it is drawn
    from (0.071330 vs 0.115288).
  * ⭐ The fair estimator is flat to **0.65%** across 4 -> 4,096 members while the plug-in
    moves **14%** of the score — Ferro 2014, confirmed on our own ensemble.

WHAT IS MEASURED  (C-351 slots)
-------------------------------
  population      the same 842,634 cell-months (months 457-469 x 64,818 land cells).
  truth           `ged_sb_best` counts, no transform.
  estimator       both, for the copy gate only; the diagnostic itself is not a CRPS.
  inference mode  ⛔ **TEACHER-FORCED** (`vehicle.py:301-307`). `C-399`.
  draw count      K=256 per MC pass -> **1,024 members**, the converged point above.
  composition     `threshold_gate` tau=0.20 (what ships) and `soft_gate`, paired.
  free parameter  the bin count over {5, 10, 20, 50} equal-count bins of predicted
                  spread (D-4), plus the zero-spread mass reported separately because
                  most cells are a point mass and would make equal-count bins degenerate.

THE DIAGNOSTIC
--------------
For a correctly dispersed ensemble, `E[(y - mean)^2] = sd^2`, so **within any bin of
predicted spread the realised RMSE should equal the predicted sd** and the ratio should
be 1. Reported per bin, with MAE beside RMSE because RMSE inside a bin is still the
outlier-sensitive one.

⚠️ This is a **DIAGNOSTIC, not a FAO-02 column.** FAO-02 governs scoring and selection and
rejects PIT; a spread–skill table and an interval coverage count answer a mechanism
question and are not proposed as selection metrics. ⛔ Using any of it to drive a
promotion decision needs Simon's amendment to FAO-02 first.

PRE-REGISTERED PREDICTIONS — declared here, committed before this runs (D-5)
----------------------------------------------------------------------------
  P-5  the zero-spread mass exceeds 99% — `threshold_gate` hard-closes 99.55% of
       cell-month-passes, so most cells are a point mass at 0.
  P-6  ⭐⭐ THE ONE THAT MATTERS. Among cells that DO have spread, the binned
       sd/RMSE ratio is far below 1 in every bin — i.e. the under-dispersion is real and
       not an artifact of the global ratio's outlier domination.
  P-7  Spearman(predicted sd, |error|) is POSITIVE and substantial. ⚠️ Declared because
       the alternative is a worse finding than under-dispersion: if the spread carries no
       information about error magnitude, the ensemble is not merely narrow, it is
       uninformative, and no rescaling of it would help.
  P-8  coverage of the ensemble's own [min, max] range is far below the ~99.8% a
       calibrated 1,024-member ensemble would give.

GATES, declared before the run
------------------------------
  G-6 ⭐ COPY. K=256 `threshold_gate` must reproduce `draw_sweep.out`'s plug-in
      **0.047376** and fair **0.047351** exactly — same seed `489+256`, same chunk
      `20_000_000//1024`, and the same composition order, because `soft_gate`'s Bernoulli
      consumes the generator and therefore shifts every later draw. The anchor is the
      committed implementation (`W-98`).
  G-7 STRUCTURAL. Binning cannot change a total, so the pooled sd and RMSE recovered from
      the bins must equal the unbinned ones. Asserted on the exact count and as a relative
      residual printed with its scale (`W-96`).
  G-8 Ranks with average ties on a hand case: `[5,5,5,1] -> [2,2,2,0]`.
"""

from __future__ import annotations

import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

T0 = time.time()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
D4 = REPO / "reports" / "2026-10-12_underfire_diagnosis"
sys.path.insert(0, str(D4))

from vehicle import SB, TARGETS, pull  # noqa: E402

from views_hydranet.distributions.composition import compose_samples  # noqa: E402
from views_hydranet.distributions.nb_core import NBCore  # noqa: E402

torch.set_num_threads(8)

OS_LO, OS_HI = 457, 469
GATE_TAU = 0.20
K = 256
SEED = 489
COMPS = ("threshold_gate", "soft_gate")
B_SWEEP = (5, 10, 20, 50)
G6 = {"threshold_gate": (0.047376, 0.047351), "soft_gate": (0.050041, 0.050024)}


def crps_ensemble(Y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row plug-in and fair CRPS. Copied from ar_score.py, the committed anchor."""
    Ys = np.sort(Y, axis=1)
    m = Ys.shape[1]
    e1 = np.abs(Ys - z[:, None]).mean(1)
    w = 2.0 * np.arange(m) - (m - 1)
    A = 2.0 * (Ys * w).sum(1)
    return e1 - A / (2.0 * m * m), e1 - A / (2.0 * m * (m - 1))


def ranks_avg(x: np.ndarray) -> np.ndarray:
    """Ranks 0..n-1 with tied values pooled to their mean rank."""
    o = np.argsort(x, kind="stable")
    r = np.empty(x.size, dtype=np.float64)
    xs = x[o]
    starts = np.flatnonzero(np.concatenate(([True], xs[1:] != xs[:-1])))
    stops = np.concatenate((starts[1:], [x.size]))
    r[o] = np.repeat((starts + stops - 1) / 2.0, stops - starts)
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = ranks_avg(a), ranks_avg(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


hand = ranks_avg(np.array([5.0, 5.0, 5.0, 1.0]))
assert np.allclose(hand, [2.0, 2.0, 2.0, 0.0]), hand
print("PASS  G-8  average-tie ranks on the hand case [5,5,5,1] -> [2,2,2,0]")

npz = np.load(D4 / "d4_result.npz")
land_cells, months = npz["land_cells"], [int(m) for m in npz["months"]]
mcol = {m: i for i, m in enumerate(months)}
MU, TH, GT = (npz[k].astype(np.float64) for k in ("mu", "theta", "gate"))

df = pull((121, 504), D4 / "pull_land_all3_121_504.parquet")
w = df[(df.month_id >= months[0]) & (df.month_id <= months[-1])]
Y = (w.pivot(index="priogrid_id", columns="month_id", values=TARGETS[SB])
     .reindex(land_cells)[months].to_numpy(np.float64)).T

sco_m = [mcol[m] for m in range(OS_LO, OS_HI + 1)]
y = Y[sco_m].ravel()
N, D = y.size, MU.shape[0]
mu_f = MU[:, sco_m].reshape(D, -1)
th_f = TH[:, sco_m].reshape(D, -1)
gt_f = GT[:, sco_m].reshape(D, -1)
m_tot = D * K
chunk = 20_000_000 // m_tot          # identical to draw_sweep.py, required by G-6

print("=" * 110)
print("Q2 AMENDMENT — THE SPREAD–SKILL DIAGNOSTIC, AT THE CONVERGED DRAW COUNT")
print("=" * 110)
print(f"  population      {len(sco_m)} months {OS_LO}-{OS_HI} x {len(land_cells):,} = {N:,}")
print(f"  truth           ged_sb_best counts, no transform. mean {y.mean():.6f} max {y.max():.0f}")
print(f"  draws           K={K} per MC pass x {D} passes = {m_tot:,} members (converged: sd is")
print("                  1.4% from its K=1024 value)")
print("  inference mode  ⛔ TEACHER-FORCED (vehicle.py:301-307)")

store = {c: {k: np.empty(N) for k in ("sd", "mean", "lo", "hi", "q05", "q95")} for c in COMPS}
crps = {c: [0.0, 0.0] for c in COMPS}
gen = torch.Generator().manual_seed(SEED + K)
for lo in range(0, N, chunk):
    hi = min(lo + chunk, N)
    masks = {c: [] for c in COMPS}
    for d in range(D):
        mu_d = torch.as_tensor(mu_f[d, lo:hi], dtype=torch.float32)
        th_d = torch.as_tensor(th_f[d, lo:hi], dtype=torch.float32)
        gt_d = torch.as_tensor(gt_f[d, lo:hi], dtype=torch.float32)
        s = NBCore.sample(mu_d, th_d, K, gen)
        for c in COMPS:                        # order matters: soft_gate consumes the RNG
            masks[c].append(compose_samples(s, gt_d, c, GATE_TAU, gen))
    for c in COMPS:
        E = torch.cat(masks[c], dim=1).numpy().astype(np.float64)
        assert np.isfinite(E).all(), "non-finite draw; draw_sweep.out recorded zero of these"
        Es = np.sort(E, axis=1)
        pl, fr = crps_ensemble(E, y[lo:hi])
        crps[c][0] += float(pl.sum())
        crps[c][1] += float(fr.sum())
        store[c]["sd"][lo:hi] = E.std(axis=1, ddof=1)
        store[c]["mean"][lo:hi] = E.mean(axis=1)
        store[c]["lo"][lo:hi] = Es[:, 0]
        store[c]["hi"][lo:hi] = Es[:, -1]
        store[c]["q05"][lo:hi] = Es[:, int(0.05 * m_tot)]
        store[c]["q95"][lo:hi] = Es[:, int(0.95 * m_tot) - 1]
        del E, Es
    del masks
print(f"\n  drawn in {time.time() - T0:.1f} s")

for c in COMPS:
    p, f = crps[c][0] / N, crps[c][1] / N
    ep, ef = G6[c]
    ok = abs(p - ep) < 1e-6 and abs(f - ef) < 1e-6
    print(f"  {'PASS' if ok else '⛔ FAIL'}  G-6  COPY {c}: got {p:.6f} / {f:.6f} "
          f"(committed {ep} / {ef})")
    if not ok:
        print("\n⛔⛔ G-6 FAILED — this is not the same ensemble draw_sweep.out scored.")
        sys.exit(1)

print("\n" + "=" * 110)
print("1. THE POINT-MASS PROBLEM — reported separately, because it would make bins degenerate")
print("=" * 110)
for c in COMPS:
    sd = store[c]["sd"]
    z = sd == 0.0
    print(f"  {c:<16s} zero-spread cells {z.mean():.6f}  ({int(z.sum()):,} of {N:,})"
          f"   ⇒ the ensemble is a point mass on {100.0 * z.mean():.2f}% of the population")
    print(f"  {'':<16s} of those, truth is nonzero on {int((y[z] > 0).sum()):,} "
          f"— events the ensemble assigns ZERO spread and (where mean=0) zero probability")

print("\n" + "=" * 110)
print("2. ⭐⭐ SPREAD–SKILL, among cells WITH spread. A calibrated ensemble gives sd/RMSE = 1")
print("=" * 110)
for c in COMPS:
    sd, mean = store[c]["sd"], store[c]["mean"]
    nz = sd > 0.0
    s, e = sd[nz], np.abs(mean[nz] - y[nz])
    pooled_sd = float(np.sqrt((s ** 2).mean()))
    pooled_rmse = float(np.sqrt((e ** 2).mean()))
    print(f"\n  {c}   {int(nz.sum()):,} cells with spread")
    print(f"    pooled: predicted sd {pooled_sd:.6f}   realised RMSE {pooled_rmse:.6f}"
          f"   ratio {pooled_sd / pooled_rmse:.4f}   MAE {e.mean():.6f}")
    print(f"    ⭐ Spearman(predicted sd, |error|) = {spearman(s, e):+.6f}")
    order = np.argsort(s, kind="stable")
    for B in B_SWEEP:
        edges = (np.arange(B + 1) * s.size / B).astype(np.int64)
        rec_sd = np.empty(B)
        rec_n = np.empty(B, dtype=np.int64)
        print(f"    B={B:<3d} " + "".join(f"{v:>12s}" for v in
                                          ("bin", "n", "pred sd", "RMSE", "ratio", "MAE")))
        for b in range(B):
            idx = order[edges[b]:edges[b + 1]]
            bs, be = s[idx], e[idx]
            r_sd = float(np.sqrt((bs ** 2).mean()))
            r_rm = float(np.sqrt((be ** 2).mean()))
            rec_sd[b], rec_n[b] = r_sd, idx.size
            if B <= 10 or b in (0, B // 2, B - 1):
                print(f"         {b:>12d}{idx.size:>12d}{r_sd:>12.6f}{r_rm:>12.6f}"
                      f"{r_sd / max(r_rm, 1e-12):>12.4f}{be.mean():>12.6f}")
        rebuilt = float(np.sqrt((rec_sd ** 2 * rec_n).sum() / rec_n.sum()))
        rel = abs(rebuilt - pooled_sd) / pooled_sd
        assert rec_n.sum() == s.size, (rec_n.sum(), s.size)
        print(f"         G-7  bins recover the pooled sd: {rebuilt:.6f} vs {pooled_sd:.6f}"
              f"   relative {rel:.2e}  (n exact: {rec_n.sum():,})")

print("\n" + "=" * 110)
print("3. INTERVAL COVERAGE — ⚠️ a diagnostic, NOT a FAO-02 column, and NOT a PIT histogram")
print("=" * 110)
for c in COMPS:
    st = store[c]
    full = ((y >= st["lo"]) & (y <= st["hi"])).mean()
    c90 = ((y >= st["q05"]) & (y <= st["q95"])).mean()
    ev = y > 0
    print(f"  {c:<16s} truth inside [min,max] {full:.6f}  inside central 90% {c90:.6f}")
    print(f"  {'':<16s} on the {int(ev.sum()):,} EVENT cells: [min,max] "
          f"{((y[ev] >= st['lo'][ev]) & (y[ev] <= st['hi'][ev])).mean():.6f}"
          f"   central 90% {((y[ev] >= st['q05'][ev]) & (y[ev] <= st['q95'][ev])).mean():.6f}")

print("\n" + "=" * 110)
print("4. ⚠️ LIMITS")
print("=" * 110)
print("  · TEACHER-FORCED. The emission's dispersion given the trunk's mu/theta (C-399).")
print("  · A diagnostic, not a FAO-02 column. Selection use needs Simon's amendment first.")
print("  · Coverage on the full population is dominated by true zeros, which any ensemble")
print("    containing 0 covers trivially — the EVENT-cell row is the informative one.")
print("  · n=1 checkpoint, sb only, no sigma-hat on any number here.")
print(f"\n  wall time {time.time() - T0:.1f} s   peak RSS "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB")
