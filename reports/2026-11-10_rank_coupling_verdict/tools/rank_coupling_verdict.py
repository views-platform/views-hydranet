"""S5 — the rank-coupled draw on A's OWN gate, over E-1's seven folds, fair CRPS beside plug-in.

Epic #579, story S5 #584. Pre-registration: ``../PREREG.md``, committed before any tool existed.
⛔ Nothing in this file may relax a threshold that ``PREREG.md`` fixed.

⭐ WHAT THIS CHANGES RELATIVE TO 2026-11-04. Three things, each one a defect that story closed:

  S3 #582  the ``ranks FIXED`` ablation held nothing — it re-sorted the key and assigned magnitudes
           in cell-index order. Fixed in ``sim_rank_coupling.py``; its Q-C column was the largest
           spread in that output and was an artefact.
  S4 #583  the gate. ⛔ The 2026-11-04 simulation drew ``rng.random(n) < pi``, a **Bernoulli** gate,
           while A ships ``threshold_gate``. ⇒ "same gate, one varying thing" was NOT true. Here B
           fires on ``gate >= tau`` deterministically, tau **read from the arm config** by
           ``arm_config.resolve_tau()``, so A's open set and B's are the SAME SET by construction
           and ``PREREG.md`` V2 is structurally unsatisfiable.
  S5 #584  one score month became **seven folds**, and fair CRPS is printed beside plug-in with the
           estimator named on every line.

⚠️ ⇒ The committed 2026-11-04 REAL gap of **-0.0137, CI [-0.1138, 0.0826]** is a Bernoulli-gate,
one-month number. It is **not** comparable with anything below and is not quoted as though it were.

**C-351 slots.** Population: the gate-open set (``gate >= tau``) over score months 463-469,
64,818 global ``land`` cells, 7 folds with a fixed 6-month fit window. ⭐ Open-cell counts are
PRINTED, never assumed. Truth: ``lr_sb_best``, raw counts, no transform — asserted by printing the
column maximum. Pool: each fold's OWN fit window's strictly positive counts. Estimator: fair and
plug-in CRPS, both, named on every line; dispersion at ``ddof=1``, declared. Inference mode:
⭐ DELIVERED AUTOREGRESSIVE, h = 1. Arm: ``ctlfullzero_fortytwo``.

⛔ THE ARMS, and which of them is a statement about the model:

  A            the delivered emission's own 16 draws. The incumbent. Not up for promotion.
  B/REAL       rank-coupled, ordered by **the recovered regression output** — the only arm that is
               a statement about the model.
  B/lam=0.00   ⭐ the NO-OP CONTROL. The ordering is pure noise, so it MUST land at zero against a
               permutation of itself. ``PREREG.md`` V3: if it does not, the criterion is broken,
               not the arm.
  B/lam=1.00   ⭐ the ORACLE / POTENCY CONTROL. The ordering is truth-derived, so the test MUST see
               it. ``PREREG.md`` V4: if it does not, there is no power and we report that.

⛔⛔ THE ORACLE IS A CALIBRATION AXIS, NOT AN ARM (``PREREG.md`` §8 device 1). No number built on
truth may be reported as a statement about the model. It exists to show the instrument can see.

Run: ``conda run --no-capture-output -n views-hydranet-env python <this file>``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-04_delivered_gate_lookup/tools"))
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-04_rank_coupled_body/tools"))

import arm_config  # noqa: E402
import e1_data  # noqa: E402

# ⭐ LIFTED, NOT RE-IMPLEMENTED. ``e1_baseline``'s ``G-P1`` gate asserts by AST that this scorer is
# structurally identical to ``2026-10-29_delivered_forecast/ar_score.py``'s. Importing keeps that
# provenance; a third copy would silently break it.
from sim_rank_coupling import crps_ensemble  # noqa: E402

#: Replicate B-cubes per arm per fold. B's own draw is random, so a single cube would report that
#: randomness as a result. ⚠️ Averaging over replicates removes B's sampling noise from the point
#: estimate; it does NOT remove it from the CI, which is blocked over folds.
R = 400
#: ⭐ Bootstrap over ORIGINS (folds), not over cells. Cells within a month are spatially dependent,
#: so an iid-over-cells interval is anticonservative. 7 blocks is coarse and that is declared.
N_BOOT = 10_000
SEED = 0


def midrank(x: np.ndarray) -> np.ndarray:
    """-> ranks of ``x`` with TIES AVERAGED (midranks). ⛔ Not ``argsort(argsort(x))``.

    ⛔⛔ WHY THIS EXISTS, measured 2026-11-10. ``argsort(argsort(x))`` assigns tied values distinct
    ranks in whatever order they happen to sit in the array, so it is a function of CELL ORDER as
    well as of ``x``. On this data that is not a rounding detail: A's prediction is the mean of
    ``S = 16`` draws, so it is quantised at 1/16 and heavily tied, and the truth is tied at every
    repeated count. Two consequences, both caught by gates rather than by inspection:

      · ``G-L3`` failed at 1.571e-03 -- the nu dial APPEARED to move AUC while holding the key's
        value multiset fixed, which is impossible. The dial was correct; the estimator was not.
      · the oracle's Spearman read 0.9963 instead of 1.0, for the same reason.

    ⇒ Midranks are the correct rank transform for tied data, and they are what Spearman's rho and
    the Mann-Whitney AUC are defined on in the first place.
    """
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """-> Spearman's rho, on MIDRANKS so that ties do not make it a function of cell order."""
    ra, rb = midrank(a), midrank(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(score: np.ndarray, positive: np.ndarray) -> float:
    """-> AUC for separating events from zeros, by the Mann-Whitney identity on MIDRANKS.

    ⭐ Added by ``PREREG.md`` A-2: the lambda dial moves SEPARATION, not ordering among events, so
    the lambda grid has to be read in this column. 0.5 = no separation.
    """
    n_pos = int(positive.sum())
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = midrank(score)
    return float((r[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def lambda_key(truth: np.ndarray, lam: float, rng: np.random.Generator) -> np.ndarray:
    """-> the ordering key at ordering-quality ``lam``: ``lam*z(rank(truth)) + (1-lam)*z(noise)``.

    ⭐ Same blend as ``sim_rank_coupling.py:25``, so the dial means the same thing across dossiers.
    ``lam = 0`` is the no-op control; ``lam = 1`` is the best achievable ordering.
    """
    jitter = rng.random(truth.size) * 1e-9  # break truth ties without changing the order
    zt = (np.argsort(np.argsort(truth + jitter)) / truth.size - 0.5) * 2.0
    noise = rng.standard_normal(truth.size)
    return lam * zt + (1.0 - lam) * noise / noise.std()


def couple(pool: np.ndarray, open_idx: np.ndarray, key: np.ndarray, n_cells: int, n_draws: int,
           rng: np.random.Generator) -> np.ndarray:
    """-> ``yB[n_cells, n_draws]``: the rank-coupled empirical draw on a FIXED open set.

    ⭐ The whole mechanism, in three lines: sort the open cells by the key, draw ``n_open`` values
    from the empirical pool, sort them, and pair them off by rank. This is the Schaake shuffle /
    ensemble copula coupling construction.

    ⛔ The open set is an ARGUMENT, not drawn here. Under ``threshold_gate`` it is deterministic and
    identical to A's, which is what makes the A-vs-B comparison paired on the same cells.
    """
    n = open_idx.size
    order = open_idx[np.argsort(key[open_idx], kind="stable")]
    y = np.zeros((n_cells, n_draws))
    for s in range(n_draws):
        y[order, s] = np.sort(rng.choice(pool, n, replace=True))
    return y


def _fold_pool(grid: np.ndarray, m0: int, fit_months: tuple[int, ...]) -> np.ndarray:
    """-> the fold's own fit-window pool: strictly positive raw counts, nothing else.

    ⛔ Each fold's pool comes from ITS OWN fit window. A pool shared across folds would leak the
    score month's regime into the folds that precede it.
    """
    rows = [grid[m - m0] for m in fit_months]
    stacked = np.concatenate(rows)
    return stacked[stacked > 0].astype(np.float64)


def main() -> None:
    t0 = time.time()
    print(__doc__)
    print("=" * 100)
    print("S5 · THE RANK-COUPLED DRAW ON A's OWN GATE, SEVEN FOLDS")
    print("=" * 100)

    tau = arm_config.resolve_tau()
    print(f"  tau = {tau} — READ from the arm config (S4 #583), not transcribed")
    print(f"  composition = {arm_config.EXPECTED_COMPOSITION!r} ⇒ the open set is deterministic")

    grid, cells, m0 = e1_data.truth_grid()
    print(f"\n  truth: {grid.shape[0]} months x {grid.shape[1]} cells, max {grid.max():.0f}")
    assert grid.max() > 1000, "⛔ raw counts expected; a transform would compress the maximum"

    gates = e1_data.load_gate_h1()
    draws = e1_data.load_draws_h1()
    by_month_gate = {mo: g for (g, _u, mo) in gates.values()}
    by_month_draws = {mo: d for (d, _u, mo) in draws.values()}
    n_cells = e1_data.N_CELLS
    n_draws = e1_data.N_DRAWS
    print(f"  loaded {len(gates)} origins of gate and {len(draws)} of draws, S = {n_draws}")

    # ── G-4 LEAKAGE, by counting, before any score
    bad = [(f, s) for f, s in e1_data.FOLDS if s in f]
    print(f"\n  {'PASS' if not bad else '⛔ FAIL'}  G-4  LEAKAGE: fit ∩ score = ∅ on "
          f"{len(e1_data.FOLDS)} folds")
    assert not bad

    arms = ("A", "B/REAL", "B/lam=0.00", "B/lam=0.00'", "B/lam=1.00")
    per_fold: dict[str, list[float]] = {a: [] for a in arms}
    per_fold_plug: dict[str, list[float]] = {a: [] for a in arms}
    rows = []

    for i, (fit_months, score_month) in enumerate(e1_data.FOLDS, start=1):
        g = by_month_gate[score_month]
        a_draws = by_month_draws[score_month].astype(np.float64)
        truth = grid[score_month - m0].astype(np.float64)
        pool = _fold_pool(grid, m0, fit_months)

        open_m = arm_config.open_set(g, tau)
        open_idx = np.flatnonzero(open_m)
        n_open = open_idx.size

        # ⭐ The REAL ordering key: on open cells ``threshold_gate`` leaves the body untouched, so
        # the mean of A's composed draws IS the regression output there. Off the open set it is 0
        # and unused, because ordering happens only within the open set.
        real_key = a_draws.mean(axis=1)

        rng = np.random.default_rng(SEED + i)
        keys = {
            "B/REAL": real_key,
            "B/lam=0.00": lambda_key(truth, 0.0, np.random.default_rng(SEED + i)),
            # ⭐ THE PERMUTATION CONTROL for PREREG V3. A SECOND, independent noise ordering. The
            # no-op must read zero against a permutation OF ITSELF -- which is the only reference
            # under which "no-op" has meaning. ⛔ It is NOT zero against A: A is a different
            # forecaster, so a no-op-vs-A gap of zero would be a claim about A, not about B.
            "B/lam=0.00'": lambda_key(truth, 0.0, np.random.default_rng(9_000 + i)),
            "B/lam=1.00": lambda_key(truth, 1.0, np.random.default_rng(SEED + i)),
        }

        # A is scored once: it is not random.
        a_plug, a_fair = crps_ensemble(a_draws[open_m], truth[open_m])
        per_fold["A"].append(float(a_fair.mean()))
        per_fold_plug["A"].append(float(a_plug.mean()))

        for name, key in keys.items():
            acc_fair = np.zeros(n_open)
            acc_plug = np.zeros(n_open)
            for _ in range(R):
                yb = couple(pool, open_idx, key, n_cells, n_draws, rng)
                p, f = crps_ensemble(yb[open_m], truth[open_m])
                acc_plug += p
                acc_fair += f
            per_fold[name].append(float((acc_fair / R).mean()))
            per_fold_plug[name].append(float((acc_plug / R).mean()))

        # ── realised ordering quality. ⭐ TWO populations, and the difference is not cosmetic.
        #
        # ``rho_event`` is ``PREREG.md``'s quantity: Spearman vs truth on **event** cells. It is the
        # one that can be compared against lambda*, because it is the only one a perfect ordering
        # scores 1.0 on.
        #
        # ⚠️ ``rho_open`` includes the true ZEROS on the open set -- roughly half of it -- which are
        # all tied at 0. Both sides of the correlation then break those ties independently at
        # random, so even a PERFECT ordering scores well below 1. ⇒ ``rho_open`` understates every
        # arm by a tie-breaking artefact and is reported for context only. ⛔ Quoting an ordering
        # quality without naming which of these two it is makes the number meaningless.
        event_m = open_m & (truth > 0)
        rho, rho_open = {}, {}
        for name, key in keys.items():
            for m, dest in ((event_m, rho), (open_m, rho_open)):
                dest[name] = spearman(key[m], truth[m])
        rows.append(
            dict(fold=i, month=score_month, n_open=n_open, n_pool=pool.size,
                 n_event=int(event_m.sum()), rho=rho, rho_open=rho_open)
        )
        print(f"  fold {i} · month {score_month} · open {n_open:,} · pool {pool.size:,} · "
              f"events-on-open {rows[-1]['n_event']:,} · "
              f"rho_event(REAL) {rho['B/REAL']:+.4f}", flush=True)

    # ── per-fold table, reported WHOLE
    print("\n" + "-" * 100)
    print("1 · PER-FOLD CRPS ON THE GATE-OPEN SET  ⭐ fair-crps is the headline, plug-in beside it")
    print("-" * 100)
    print(f"  {'fold':<6}{'month':<8}{'n_open':>8}" + "".join(f"{a:>14}" for a in arms))
    for j, r in enumerate(rows):
        line = "".join(f"{per_fold[a][j]:>14.4f}" for a in arms)
        print(f"  {r['fold']:<6}{r['month']:<8}{r['n_open']:>8,}{line}   fair-crps")
        line = "".join(f"{per_fold_plug[a][j]:>14.4f}" for a in arms)
        print(f"  {'':<6}{'':<8}{'':>8}{line}   plug-in-crps")

    print(f"\n  {'pooled (mean over folds)':<22}" + "".join(f"{np.mean(per_fold[a]):>14.4f}" for a in arms)
          + "   fair-crps")
    print(f"  {'':<22}" + "".join(f"{np.mean(per_fold_plug[a]):>14.4f}" for a in arms)
          + "   plug-in-crps")
    print(f"  {'sd over folds, ddof=1':<22}"
          + "".join(f"{np.std(per_fold[a], ddof=1):>14.4f}" for a in arms) + "   fair-crps")

    # ── the paired gaps, origin-blocked
    boot_rng = np.random.default_rng(SEED)
    idx = boot_rng.integers(0, len(e1_data.FOLDS), (N_BOOT, len(e1_data.FOLDS)))

    def gap_table(ref: str, subjects: tuple[str, ...]) -> dict[str, tuple]:
        out = {}
        print(f"  {'arm':<14}{'mean gap':>12}{'95% CI of the gap':>28}{'sign':>12}  per-fold signs")
        for a in subjects:
            d = np.array(per_fold[a]) - np.array(per_fold[ref])
            bs = d[idx].mean(axis=1)
            lo, hi = np.quantile(bs, [0.025, 0.975])
            excl = "excludes 0" if lo > 0 or hi < 0 else "spans 0"
            signs = "".join("-" if x < 0 else "+" for x in d)
            print(f"  {a:<14}{d.mean():>+12.4f}   [{lo:>+9.4f}, {hi:>+9.4f}]{excl:>12}  {signs}")
            out[a] = (float(d.mean()), float(lo), float(hi), lo > 0 or hi < 0)
        return out

    print("\n" + "-" * 100)
    print("2a · ⭐ THE WITHIN-B CONTRAST — each ordering against the NO-OP (lam=0.00).")
    print("     This is the contrast the controls are defined on: it isolates *does the ordering")
    print("     carry information*, holding the gate, the open set, the pool and S fixed.")
    print("     ⛔ lower CRPS is better, so a NEGATIVE gap means the ordering HELPED.")
    print("-" * 100)
    within = gap_table("B/lam=0.00", ("B/lam=0.00'", "B/REAL", "B/lam=1.00"))

    print("\n  THE TWO CONTROLS, evaluated (PREREG.md §5):")
    noop_prime = within["B/lam=0.00'"]
    oracle = within["B/lam=1.00"]
    v3_pass = not noop_prime[3]          # must NOT exclude zero
    v4_pass = oracle[3] and oracle[0] < 0  # must exclude zero, in B's favour
    print(f"    {'PASS' if v3_pass else '⛔ V3 FIRES'}  V3  the NO-OP reads zero against a "
          f"permutation of itself: gap {noop_prime[0]:+.4f}, "
          f"CI [{noop_prime[1]:+.4f}, {noop_prime[2]:+.4f}]")
    print(f"    {'PASS' if v4_pass else '⛔ V4 FIRES'}  V4  the ORACLE separates, in B's favour: "
          f"gap {oracle[0]:+.4f}, CI [{oracle[1]:+.4f}, {oracle[2]:+.4f}]")

    print("\n" + "-" * 100)
    print("2b · THE PROMOTION-RELEVANT CONTRAST — each B against A, the incumbent.")
    print("     ⚠️ CRPS is a GUARDRAIL here, NEVER the decider (PREREG.md §4): it ranks an empty")
    print("     map top, and C-52 blocks a proper score against a miscalibrated A. The decision")
    print("     number is the conditional magnitude readout, S6 #585.")
    print("-" * 100)
    gap_table("A", arms[1:])

    # ── the realised ordering quality, which is what the epic is about
    print("\n" + "-" * 100)
    print("3 · REALISED ORDERING QUALITY (Spearman vs truth)")
    print("-" * 100)
    print("  ⭐ rho_event — on EVENT cells. PREREG.md's quantity, and the one lambda* is stated in.")
    print(f"  {'arm':<14}" + "".join(f"{'f' + str(r['fold']):>9}" for r in rows) + f"{'mean':>10}")
    for a in arms[1:]:
        vals = [r["rho"][a] for r in rows]
        print(f"  {a:<14}" + "".join(f"{v:>+9.4f}" for v in vals) + f"{np.mean(vals):>+10.4f}")
    print("\n  ⚠️ rho_open — on ALL open cells. Roughly half are true zeros, all tied, and both")
    print("     sides break those ties independently ⇒ even a PERFECT ordering scores well below")
    print("     1.0 here. Context only. ⛔ Never compare a figure from this block with one above.")
    print(f"  {'arm':<14}" + "".join(f"{'f' + str(r['fold']):>9}" for r in rows) + f"{'mean':>10}")
    for a in arms[1:]:
        vals = [r["rho_open"][a] for r in rows]
        print(f"  {a:<14}" + "".join(f"{v:>+9.4f}" for v in vals) + f"{np.mean(vals):>+10.4f}")

    print(f"\n  {time.time() - t0:.1f} s, CPU, R={R} replicate cubes per arm per fold, zero GPU, "
          "nothing retrained.")


if __name__ == "__main__":
    main()
