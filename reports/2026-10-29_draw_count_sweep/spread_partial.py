"""Q2, amendment 2 — is the spread–error correlation anything more than the MEAN?

WHY THIS EXISTS, AND IT IS A DEFECT IN MY OWN REPORTING
-------------------------------------------------------
`spread_skill.out` measured **Spearman(predicted sd, |error|) = +0.728114**
(`threshold_gate`) and **+0.932179** (`soft_gate`), and I reported it to Simon as
*"the model knows where it is uncertain"* and *"the one lever this opens"*.

⛔ **That reading is confounded, and I did not check it before reporting it.** Two
mechanisms make the correlation large with no conditional information at all:

  1. **sd is a monotone function of the mean.** For the NB, `var = mu + mu^2/theta`, so a
     cell with a large `mu` necessarily has a large predicted sd. Nothing is learned.
  2. ⛔⛔ **On a cell whose truth is zero, `|error| = |mean - 0| = mean` EXACTLY.** So on
     every zero-truth cell the statistic degenerates to `Spearman(sd, mean)` — the
     mechanical link of (1) — and zero-truth cells are **52%** of `threshold_gate`'s
     spread-bearing cells and **82%** of `soft_gate`'s.

⇒ The pooled number may be a restatement of *"the model knows where the mean is high"*,
which was never in doubt. ⚠️ And note the direction this cuts: it predicts `soft_gate`'s
**higher** correlation is a symptom of its larger zero-truth share, not of better
information.

WHAT THIS SCRIPT MEASURES
-------------------------
The same draw, three ways the pooled statistic cannot distinguish:

  A · `Spearman(sd, ensemble mean)` — how much of sd IS the mean. The confound, sized.
  B · `Spearman(sd, |error|)` **within equal-count bins of the ensemble mean**, n-weighted
      across bins. Holding the mean fixed removes mechanism (1).
  C · B again on **event cells only** (`truth > 0`), which removes mechanism (2) because
      `|error|` is no longer identically the mean.

C is the decisive one. ⭐ If it survives, the spread carries information beyond the mean
and Simon's conditional-heteroskedasticity framing has something to work on. ⛔ If it
collapses, the "one lever" claim is withdrawn by me.

WHAT IS MEASURED  (C-351 slots) — identical to `spread_skill.py`, same draw
--------------------------------------------------------------------------
  population      842,634 cell-months (months 457-469 x 64,818 land cells).
  truth           `ged_sb_best` counts, no transform.
  estimator       not a CRPS. Spearman rank correlation, average ties.
  inference mode  ⛔ **TEACHER-FORCED** (`vehicle.py:301-307`). `C-399`.
  draw count      K=256 per MC pass -> 1,024 members (the converged point).
  composition     `threshold_gate` tau=0.20 (what ships) and `soft_gate`, paired.
  free parameter  the conditioning bin count over {5, 10, 20, 50} (D-4).

⚠️ A **DIAGNOSTIC, not a FAO-02 column**, exactly as `spread_skill.py`'s was.

PRE-REGISTERED PREDICTIONS — declared here, committed before this runs (D-5)
----------------------------------------------------------------------------
  P-9   `Spearman(sd, ensemble mean)` exceeds **+0.95** for both compositions. The link is
        mechanical, so this is a sanity check on my own reasoning, not a discovery.
  P-10  ⭐⭐ THE ONE THAT MATTERS. The within-mean-bin, events-only Spearman (C) falls
        **below +0.20** for both compositions — i.e. the pooled +0.73/+0.93 was the mean,
        and the spread carries little information beyond it.
  P-11  The `soft_gate` > `threshold_gate` ordering **reverses or narrows to under half**
        its pooled gap once conditioned, because `soft_gate`'s advantage is its larger
        zero-truth share where `|error|` is identically the mean.

GATES, declared before the run
------------------------------
  G-9  ⭐ COPY. The pooled Spearman must reproduce `spread_skill.out` EXACTLY:
       **+0.728114** / **+0.932179**, from the same seed `489+256`, the same chunk
       `20_000_000//1024`, and the same composition order (`soft_gate`'s Bernoulli
       consumes the generator, so order shifts every later draw). The anchor is the
       committed implementation and its captured output (`W-98`).
  G-10 ⭐⭐ POSITIVE **AND** NEGATIVE CONTROL on the conditional estimator itself, on
       synthetic data where the answer is known by construction:
         · negative: sd and |error| both scale with the mean but are independent given it
           -> the conditional statistic must be within 0.05 of 0, while the POOLED one
           must still exceed +0.8. This is the confound reproduced in a test tube.
         · positive: sd drives |error| within each mean level -> conditional > +0.95.
       ⛔ If the negative control does not collapse, the estimator cannot detect the
       confound and nothing below it is readable.
  G-11 STRUCTURAL. The conditioning bins must partition the sample exactly (counts sum,
       asserted on the integer).
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
G9 = {"threshold_gate": 0.728114, "soft_gate": 0.932179}


def ranks_avg(x: np.ndarray) -> np.ndarray:
    """Ranks 0..n-1 with tied values pooled to their mean rank. Copied from spread_skill.py."""
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


def partial_spearman(cond: np.ndarray, x: np.ndarray, z: np.ndarray,
                     B: int) -> tuple[float, int, int]:
    """Spearman(x, z) within B equal-count bins of `cond`, n-weighted across bins.

    Returns (weighted rho, bins used, cells in those bins). A bin where `x` or `z` is
    constant has no rank variance and yields nan; those bins are dropped and counted,
    because a degenerate bin is a real outcome on a point-mass-heavy population, not an
    error to silence.
    """
    order = np.argsort(cond, kind="stable")
    edges = (np.arange(B + 1) * cond.size / B).astype(np.int64)
    num, den, used, cells = 0.0, 0, 0, 0
    for b in range(B):
        idx = order[edges[b]:edges[b + 1]]
        assert idx.size > 0, (B, b, cond.size)
        r = spearman(x[idx], z[idx])
        if np.isfinite(r):
            num += r * idx.size
            den += idx.size
            used += 1
            cells += idx.size
    return (num / den if den else float("nan")), used, cells


# ---------------------------------------------------------------- G-10, before any data
rng = np.random.default_rng(10)
lvl = np.repeat(np.arange(20, dtype=np.float64) + 1.0, 400)
u1, u2 = rng.random(lvl.size), rng.random(lvl.size)
neg_s, neg_e = lvl * (1.0 + 0.4 * u1), lvl * (1.0 + 0.4 * u2)
pos_s, pos_e = lvl * (1.0 + 0.9 * u1), lvl * (1.0 + 0.9 * u1)
neg_pool, pos_pool = spearman(neg_s, neg_e), spearman(pos_s, pos_e)
neg_cond = partial_spearman(lvl, neg_s, neg_e, 20)[0]
pos_cond = partial_spearman(lvl, pos_s, pos_e, 20)[0]
ok10 = (abs(neg_cond) < 0.05 and neg_pool > 0.80 and pos_cond > 0.95)
print(f"{'PASS' if ok10 else '⛔ FAIL'}  G-10  controls: "
      f"negative pooled {neg_pool:+.4f} -> conditional {neg_cond:+.4f} (|.|<0.05) | "
      f"positive conditional {pos_cond:+.4f} (>0.95)")
if not ok10:
    print("\n⛔⛔ G-10 FAILED — the conditional estimator cannot detect the confound it")
    print("    exists to detect. Nothing below this line would be readable.")
    sys.exit(1)

# ------------------------------------------------------------------------------ the draw
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
chunk = 20_000_000 // m_tot          # identical to spread_skill.py, required by G-9

print("=" * 110)
print("Q2 AMENDMENT 2 — IS THE SPREAD–ERROR CORRELATION ANYTHING MORE THAN THE MEAN?")
print("=" * 110)
print(f"  population      {len(sco_m)} months {OS_LO}-{OS_HI} x {len(land_cells):,} = {N:,}")
print(f"  truth           ged_sb_best counts, no transform. mean {y.mean():.6f} max {y.max():.0f}")
print(f"  draws           K={K} per MC pass x {D} passes = {m_tot:,} members")
print("  inference mode  ⛔ TEACHER-FORCED (vehicle.py:301-307)")

store = {c: {k: np.empty(N) for k in ("sd", "mean")} for c in COMPS}
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
        store[c]["sd"][lo:hi] = E.std(axis=1, ddof=1)
        store[c]["mean"][lo:hi] = E.mean(axis=1)
        del E
    del masks
print(f"\n  drawn in {time.time() - T0:.1f} s")

print("\n" + "=" * 110)
print("1. THE POOLED STATISTIC, REPRODUCED — G-9")
print("=" * 110)
sub = {}
for c in COMPS:
    sd, mean = store[c]["sd"], store[c]["mean"]
    nz = sd > 0.0
    s, m, e = sd[nz], mean[nz], np.abs(mean[nz] - y[nz])
    ev = y[nz] > 0
    sub[c] = (s, m, e, ev)
    got = spearman(s, e)
    ok = abs(got - G9[c]) < 1e-6
    print(f"  {'PASS' if ok else '⛔ FAIL'}  G-9  COPY {c}: "
          f"Spearman(sd, |error|) = {got:+.6f} (committed {G9[c]:+.6f})")
    if not ok:
        print("\n⛔⛔ G-9 FAILED — this is not the ensemble spread_skill.out measured.")
        sys.exit(1)
    print(f"        {int(nz.sum()):,} cells with spread, of which "
          f"{int(ev.sum()):,} are events ({100.0 * ev.mean():.1f}%) and "
          f"{int((~ev).sum()):,} are true zeros ({100.0 * (~ev).mean():.1f}%)")
    print("        ⛔ on every true zero, |error| = mean EXACTLY — the statistic degenerates")

print("\n" + "=" * 110)
print("2. A · HOW MUCH OF THE SPREAD *IS* THE MEAN")
print("=" * 110)
for c in COMPS:
    s, m, e, _ = sub[c]
    print(f"  {c:<16s} Spearman(sd, ensemble mean)  = {spearman(s, m):+.6f}")
    print(f"  {'':<16s} Spearman(mean, |error|)      = {spearman(m, e):+.6f}"
          "   ⇐ the mean's own rank skill at error magnitude")

print("\n" + "=" * 110)
print("3. B · CONDITIONED ON THE MEAN, all cells with spread  (D-4: bin count swept)")
print("=" * 110)
print(f"  {'comp':<16s}{'B':>5s}{'rho(sd,|err| | mean)':>24s}{'bins used':>12s}{'cells':>10s}")
for c in COMPS:
    s, m, e, _ = sub[c]
    for B in B_SWEEP:
        r, used, cells = partial_spearman(m, s, e, B)
        assert cells <= s.size, (cells, s.size)
        print(f"  {c:<16s}{B:>5d}{r:>+24.6f}{f'{used}/{B}':>12s}{cells:>10,d}")

print("\n" + "=" * 110)
print("4. ⭐⭐ C · THE DECISIVE READ — conditioned on the mean, EVENT CELLS ONLY")
print("=" * 110)
print("  (removes both confounds: the mean is held fixed, and |error| is no longer")
print("   identically the mean because truth > 0)")
print(f"\n  {'comp':<16s}{'B':>5s}{'rho(sd,|err| | mean)':>24s}{'bins used':>12s}{'cells':>10s}")
for c in COMPS:
    s, m, e, ev = sub[c]
    se, me, ee = s[ev], m[ev], e[ev]
    print(f"  {c:<16s}{'pool':>5s}{spearman(se, ee):>+24.6f}{'—':>12s}{se.size:>10,d}")
    for B in B_SWEEP:
        r, used, cells = partial_spearman(me, se, ee, B)
        assert cells <= se.size, (cells, se.size)
        print(f"  {c:<16s}{B:>5d}{r:>+24.6f}{f'{used}/{B}':>12s}{cells:>10,d}")

print("\n" + "=" * 110)
print("5. ⚠️ LIMITS")
print("=" * 110)
print("  · TEACHER-FORCED. The emission's spread given the trunk's mu/theta (C-399).")
print("  · A diagnostic, not a FAO-02 column. Selection use needs Simon's amendment first.")
print("  · Conditioning on the ensemble MEAN is not conditioning on mu: the composition")
print("    and the gate both enter the mean. It removes the mean's confound, not the gate's.")
print("  · Equal-count bins of a heavily tied variable split ties across bins; a bin with")
print("    no rank variance in sd or |error| is dropped and counted above, not imputed.")
print("  · n=1 checkpoint, sb only, no sigma-hat on any number here.")
print(f"\n  wall time {time.time() - T0:.1f} s   peak RSS "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576:.2f} GiB")
