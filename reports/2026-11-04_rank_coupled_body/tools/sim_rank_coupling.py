"""Q-B / Q-C / Q-D — simulate the rank-coupled body on the REAL gate, pool and truth.

Programme: ``reports/2026-11-04_rank_coupled_body_programme.md``. ⛔ Committed before it ran (D-5).

⭐ WHAT THIS SETTLES, and it is three of Simon's questions:

  Q-B  Does v1's per-cell dispersion profile match the defect profile ``M90`` measured? ``M90``:
       OVER-dispersed (sd/RMSE > 1) in 9 of 10 spread bins and UNDER-dispersed (0.48 / 0.39) in the
       top bin only. ⇒ The correction the data needs is NON-MONOTONE in the spread.
  Q-C  How much per-cell spread does each of P6's three sources supply, and at which ranks?
       Answered by holding each source fixed in turn.
  Q-D  Can the stopping rule fire? i.e. is the v1-versus-permutation gap outside the noise floor?
       Answered by sweeping the ORDERING QUALITY rather than assuming this head's.

**C-351 slots.** Population: 64,818 global ``land`` cells, score month 463, fit months 457-462.
Truth: ``lr_sb_best``, raw counts. Gate: the delivered h1 posterior-mean gate, 16 draws, from
``gate_endpoints``. Pool: the fit window's strictly positive counts. Inference mode: ⭐ DELIVERED
AUTOREGRESSIVE h1. Arm: ``ctlfullzero_fortytwo``. Estimator: plug-in CRPS per cell (⚠️ plug-in, not
fair, because ``crps_concentration`` refuses a negative vector and the fair estimator can go
negative — ``rollout_ruler_core.py:465-474``).

⭐ THE ORDERING SWEEP, and why it is better than using this head's own ordering. The ordering is
blended between a truth-based oracle and pure noise:

    score_i = lam * z(rank(truth_i)) + (1 - lam) * z(noise_i),   lam in [0, 1]

⇒ ``lam = 0`` IS the permutation control; ``lam = 1`` is the best achievable ordering. Sweeping
``lam`` gives the gap as a FUNCTION of ordering quality, so the answer is *what quality the
mechanism
requires*, not merely whether one head happens to have it.

⛔⛔ THE ORACLE IS A CALIBRATION AXIS, NOT AN ARM. Device 1 declares an oracle-only
improvement a FAIL. Here the truth-based ordering exists only to parameterise the sweep; ⛔ **no
number from ``lam > 0`` built on truth may be reported as a result about the model.** The only arm
that is a statement about the model is ``REAL``, below.

⭐ ``REAL`` — the anchor, and it is partial by construction. ``threshold_gate`` leaves the body
draws
UNTOUCHED wherever ``gate >= tau``, so on those cells the composed draws ARE the body's
and their mean is the regression output. Measured: open cells are only 0.5% of the grid but carry
**55.9%** of the expected firing mass. ⇒ The real ordering is available for the TOP of the order
and
lost on the bottom, where the closed cells sit. The ``REAL`` arm therefore orders open cells by the
recovered regression output and tie-breaks the closed ones at random — i.e. it is the real ordering
on the half that matters and the control on the other half. ⚠️ It is a LOWER BOUND on what a full
forward pass would give.

⛔ THE GATES, declared before the run:

  G-20  The ablation is PAIRED on common random numbers: every arm sees the same gate draws and the
        same pool draws in sample r, differing only in the order. Asserted by checking that the
        per-sample firing sets and drawn multisets are identical across arms.
  G-21  ⭐ A1/A2 hold numerically: the permutation control's field total matches v1's EXACTLY in
        every sample, and its pooled per-cell magnitude distribution matches the pool.
  G-22 NON-VACUITY: the lam sweep must actually move the realised Spearman between the ordering and
        the truth on the firing set, monotonically. A sweep that does not move the ordering cannot
        measure what ordering quality buys.
  G-23  P2 numerically: E[sum] over samples equals n_t * pool mean to within Monte Carlo error.

Run: ``conda run --no-capture-output -n views-hydranet-env python <this file>``
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-04_delivered_gate_lookup/tools"))
import e1_data  # noqa: E402

sys.path.insert(0, str(e1_data.REPO / "reports/2026-10-29_delivered_forecast"))

TAU = 0.20
R = 400
FIT = (457, 458, 459, 460, 461, 462)
SCORE = 463
LAMS = (0.0, 0.15, 0.30, 0.50, 0.75, 1.0)
SEED = 0


# ── the lifted scorer. Same provenance rule as e1_baseline: LIFTED, NOT EDITED. ──────────────
def crps_ensemble(Y: np.ndarray, z: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """Per-row CRPS of an m-member ensemble ``Y`` [N, m] against truth ``z`` [N].

    Returns (plug-in `1/m^2`, fair `1/(m(m-1))`). Same pairwise identity, row-wise.
    """
    Ys = np.sort(Y, axis=1)
    m = Ys.shape[1]
    e1 = np.abs(Ys - z[:, None]).mean(1)
    w = 2.0 * np.arange(m) - (m - 1)
    A = 2.0 * (Ys * w).sum(1)
    return e1 - A / (2.0 * m * m), e1 - A / (2.0 * m * (m - 1))


def recovered_yhat(month: int) -> np.ndarray:
    """-> the regression output per cell where it survives composition, else nan.

    ``threshold_gate`` zeroes a cell entirely below tau and leaves it untouched above, so the mean
    of
    the composed draws IS the body's expectation on open cells.
    """
    arm_dir, arms = e1_data.arm_dir, e1_data.arm_directories()
    delivery = arm_dir(e1_data.ARM, arms=arms) / e1_data.DELIVERY
    src = sorted((delivery / "draws").glob("*_lr.tar.zst"))[0]
    origin = month - e1_data.H1_MONTHS[0]
    with tempfile.TemporaryDirectory(prefix="e2_draws_") as td:
        subprocess.run(
            [
                "tar",
                "-I",
                "zstd",
                "-xf",
                str(src),
                "-C",
                td,
                "--wildcards",
                f"*/origin_{origin}/lr_sb_best/*",
            ],
            check=True,
        )
        f = sorted(Path(td).glob(f"*/origin_{origin}/lr_sb_best/y_pred.npy"))[0]
        ids = np.load(f.parent / "identifiers.npz")
        sel = ids["time"].astype(np.int64) == month
        return np.load(f, mmap_mode="r")[sel].astype(np.float64).mean(1)


def simulate(pi, pool, truth, order_key, rng, *, fix_gate=False, fix_ranks=False, fix_draws=False):
    """-> (y[R, N], fired[R, N], totals[R]). One arm, R samples, common random numbers by seed."""
    n_cells = pi.size
    y = np.zeros((R, n_cells))
    fired = np.zeros((R, n_cells), bool)
    base_fire = rng.random(n_cells) < pi if fix_gate else None
    # ⭐ ``fix_ranks`` holds the cell-to-QUANTILE map: a cell's rank becomes a property OF THE CELL
    # rather than of whichever other cells happened to fire. Computed once, over all cells, outside
    # the sample loop. ⛔ The previous form did ``key = np.sort(key)``, which makes ``argsort`` the
    # identity and therefore assigns magnitudes in CELL-INDEX order — it decoupled the assignment
    # from the ordering entirely instead of holding it, and its column was the largest spread in
    # the output. See ``reports/2026-11-04_rank_coupled_body_programme.md:423-424``.
    q_fixed = (
        np.argsort(np.argsort(order_key, kind="stable")) / max(n_cells - 1, 1)
        if fix_ranks
        else None
    )
    for r in range(R):
        f = base_fire if fix_gate else (rng.random(n_cells) < pi)
        idx = np.flatnonzero(f)
        n = idx.size
        if n == 0:
            continue
        fired[r, idx] = True
        d = np.sort(pool[:n] if fix_draws else rng.choice(pool, n, replace=True))
        if fix_ranks:
            # ⚠️ A resampling of ``d`` with repeats, NOT a permutation of it: holding the rank map
            # under a VARYING firing set cannot also preserve the assigned multiset. ⇒ this arm's
            # field total is not comparable with the others'; its per-cell spread is.
            y[r, idx] = d[np.rint(q_fixed[idx] * (n - 1)).astype(np.int64)]
        else:
            order = idx[np.argsort(order_key[idx], kind="stable")]
            y[r, order] = d
    return y, fired, y.sum(1)


def main() -> None:
    t0 = time.time()
    print(__doc__, flush=True)
    print("=" * 100)
    print("Q-B / Q-C / Q-D — the rank-coupled body simulated on the real gate, pool and truth")
    print("=" * 100)

    grid, cells, m0 = e1_data.truth_grid()
    gate = e1_data.load_gate_h1()
    o = [k for k in gate if gate[k][2] == SCORE][0]
    pi = gate[o][0]
    truth = grid[SCORE - m0]
    pool = np.concatenate([grid[m - m0] for m in FIT])
    pool = pool[pool > 0]
    print(
        f"\n  score month {SCORE}, fit {FIT[0]}-{FIT[-1]}, pool {pool.size:,} positives "
        f"(median {np.median(pool):.0f}, max {pool.max():.0f})"
    )
    print(
        f"  gate mean {pi.mean():.6f}, E[fire] {pi.sum():.1f}; "
        f"truth {int((truth > 0).sum())} events, "
        f"max {truth.max():.0f}"
    )

    # ── build the ordering keys
    rng0 = np.random.default_rng(SEED)
    zt = (np.argsort(np.argsort(truth + rng0.random(truth.size) * 1e-9)) / truth.size - 0.5) * 2
    noise = rng0.standard_normal(truth.size)
    keys = {f"lam={lam:.2f}": lam * zt + (1 - lam) * noise / noise.std() for lam in LAMS}
    yh = recovered_yhat(SCORE)
    open_ = pi >= TAU
    real_key = np.where(open_, yh, -np.inf)
    real_key = real_key + np.where(open_, 0.0, rng0.random(truth.size))
    keys["REAL (open cells)"] = real_key
    print(
        f"  REAL arm: {int(open_.sum())} open cells ({open_.mean():.4%}) carrying "
        f"{pi[open_].sum() / pi.sum():.1%} of expected firing mass"
    )

    # ── G-22 · the sweep must move the ordering
    print("\n" + "-" * 100)
    print("G-22 · does the lam sweep move the ordering? (Spearman vs truth on firing cells)")
    print("-" * 100)
    from scipy.stats import spearmanr

    f0 = rng0.random(pi.size) < pi
    rhos = {}
    for name, k in keys.items():
        rhos[name] = float(spearmanr(k[f0], truth[f0]).statistic)
        print(f"  {name:<20} realised Spearman {rhos[name]:>7.4f}")
    lam_rhos = [rhos[f"lam={lam:.2f}"] for lam in LAMS]
    ok22 = all(b >= a - 1e-9 for a, b in zip(lam_rhos, lam_rhos[1:]))
    print(f"  {'PASS' if ok22 else '⛔ FAIL'}  G-22 monotone in lam")

    # ── the arms
    print("\n" + "-" * 100)
    print("Q-D · the v1-versus-control gap as a function of ordering quality")
    print("-" * 100)
    print(
        f"  {'arm':<20}{'Spearman':>10}{'CRPS pooled':>14}"
        f"{'CRPS firing':>14}{'gap vs lam=0':>15}{'95% CI of the gap':>26}"
    )
    res, base = {}, None
    for name, k in keys.items():
        y, fired, tot = simulate(pi, pool, truth, k, np.random.default_rng(SEED))
        plug, _fair = crps_ensemble(y.T.copy(), truth)
        union = fired.any(0)
        res[name] = dict(plug=plug, union=union, tot=tot, y=y, fired=fired)
        if base is None:
            base = plug
        d = base - plug  # positive ⇒ this arm is better
        du = d[union]
        bs = np.array(
            [
                du[np.random.default_rng(1000 + i).integers(0, du.size, du.size)].mean()
                for i in range(300)
            ]
        )
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(
            f"  {name:<20}{rhos[name]:>10.4f}{plug.mean():>14.6f}{plug[union].mean():>14.4f}"
            f"{du.mean():>15.4f}  [{lo:>8.4f},{hi:>8.4f}] "
            f"{'EXCLUDES 0' if lo > 0 or hi < 0 else 'spans 0'}"
        )

    # ── G-20/G-21/G-23
    print("\n" + "-" * 100)
    print("THE GATES")
    print("-" * 100)
    a, b = res["lam=0.00"], res["lam=1.00"]
    ok20 = np.array_equal(a["fired"], b["fired"])
    print(
        f"  {'PASS' if ok20 else '⛔ FAIL'}  G-20  paired: identical firing sets across arms "
        f"({int(a['fired'].sum())} firings over {R} samples)"
    )
    ok21 = np.allclose(a["tot"], b["tot"])
    print(
        f"  {'PASS' if ok21 else '⛔ FAIL'}  G-21  A1: field totals identical across arms, "
        f"max |diff| {np.abs(a['tot'] - b['tot']).max():.3e}"
    )
    exp = a["fired"].sum(1).mean() * pool.mean()
    print(
        f"  {'PASS' if abs(a['tot'].mean() / exp - 1) < 0.05 else '⛔ FAIL'}  G-23  P2: "
        f"E[sum] {a['tot'].mean():.0f} vs n*mu {exp:.0f}  ratio {a['tot'].mean() / exp:.4f}"
    )

    # ── Q-B · dispersion profile
    print("\n" + "-" * 100)
    print("Q-B · dispersion profile. M90: OVER-dispersed in 9 of 10 bins, UNDER in the top only")
    print("-" * 100)
    for name in ("lam=0.00", "lam=1.00", "REAL (open cells)"):
        y = res[name]["y"]
        sd = y.std(0, ddof=1)
        err = np.abs(y.mean(0) - truth)
        has = sd > 0
        print(
            f"\n  {name}: {int(has.sum()):,} cells with spread ({has.mean():.4%}); "
            f"{int((~has).sum()):,} are a point mass"
        )
        q = np.quantile(sd[has], np.linspace(0, 1, 11))
        print(f"    {'bin':>4}{'n':>8}{'mean sd':>11}{'RMSE':>11}{'sd/RMSE':>10}")
        for i in range(10):
            m = has & (sd >= q[i]) & (sd <= q[i + 1] if i == 9 else sd < q[i + 1])
            if m.sum() < 5:
                continue
            rmse = float(np.sqrt((err[m] ** 2).mean()))
            print(
                f"    {i:>4}{int(m.sum()):>8}{sd[m].mean():>11.3f}{rmse:>11.3f}"
                f"{(sd[m].mean() / rmse if rmse > 0 else np.nan):>10.3f}"
            )

    # ── Q-C · variance decomposition
    print("\n" + "-" * 100)
    print("Q-C · which of P6's sources supplies the per-cell spread, by rank decile")
    print("-" * 100)
    print(
        "  ⚠️ 'ranks FIXED' holds the cell-to-quantile map (fixed 2026-11-10, epic #579 / S3 #582).\n"
        "     Its earlier form re-sorted the key, which assigned magnitudes in cell-index order and\n"
        "     printed the largest spread in this table as an artefact. ⚠️ Under the corrected hold the\n"
        "     assigned values are a resampling of the draw multiset WITH REPEATS, so this column's\n"
        "     field total is not comparable with the others' — only its per-cell spread is."
    )
    k = keys["lam=1.00"]
    full = res["lam=1.00"]
    arms = {"all three on": full}
    for tag, kw in (
        ("gate FIXED", dict(fix_gate=True)),
        ("ranks FIXED", dict(fix_ranks=True)),
        ("draws FIXED", dict(fix_draws=True)),
    ):
        y, fired, tot = simulate(pi, pool, truth, k, np.random.default_rng(SEED), **kw)
        arms[tag] = dict(y=y, fired=fired)
    union = full["union"]
    rank_of = np.full(pi.size, np.nan)
    idx = np.flatnonzero(union)
    rank_of[idx] = np.argsort(np.argsort(k[idx])) / max(idx.size - 1, 1)
    print(f"  {'rank decile':<14}" + "".join(f"{t:>16}" for t in arms))
    for d in range(10):
        m = union & (rank_of >= d / 10) & (rank_of < (d + 1) / 10 + (1e-9 if d == 9 else 0))
        if m.sum() < 5:
            continue
        row = "".join(f"{arms[t]['y'][:, m].std(0, ddof=1).mean():>16.3f}" for t in arms)
        print(f"  {f'{d * 10}-{(d + 1) * 10}%':<14}{row}")

    print(f"\n  {time.time() - t0:.1f} s, CPU, R={R} samples per arm, zero GPU.")


if __name__ == "__main__":
    main()
