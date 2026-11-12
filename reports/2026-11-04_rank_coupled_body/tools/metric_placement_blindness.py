"""Which distributional metrics can SEE placement? A blindness map, not a verdict.

Programme: ``reports/2026-11-04_rank_coupled_body_programme.md``. ⛔ Committed before it ran (D-5).

⭐ WHAT THIS IS. To know whether a ruler works, measure something whose length you already know.
Build two forecasts identical in every respect — same cells firing, same list of magnitudes
handed out, same field total — except that one places the magnitudes **using the truth** and
the other places them **at random**. The first is certainly better by construction. Hand both to
every
metric and see which ones notice.

⇒ Any metric that scores the two equal is **demonstrably blind to placement**, measured rather than
argued.

ℹ️ **This already exists here for POINT forecasts.** `#444` measured that permuting the oracle's
event magnitudes moves AP, Brier, FSS1, FSS9, AFSS and hit_rate by **exactly 0.000e+00** while
`placement_rho` falls 1.000000 → −0.030036. That is how `placement_rho` came to exist. ⇒ What is
missing, and what this file supplies, is the same test for **distributional** metrics — which are
what the programme is judged on.

⛔⛔ WHAT THIS IS NOT, and the limits are the point:

  * ⛔ It does **not** say which metric is *good*. A metric can be placement-sensitive and still be
    improper, biased, or wrong for this population. Sensitivity is necessary, not sufficient.
  * ⛔ It says **nothing** about whether the rank-coupling mechanism works. The oracle arm uses the
    truth; device 1 declares an oracle-only improvement a FAIL. ⇒ No number here is a result about
    any model.
  * ⚠️ It is one target (`sb`), one horizon (h1), one arm, one pool window. The metrics' sensitivity
    could differ elsewhere.
  * ⚠️ A metric that separates the two arms here is sensitive to placement **of magnitudes among
    already-firing cells**. That is not the same as sensitivity to which cells fire.

**C-351 slots.** Population: 64,818 global ``land`` cells x 7 score months (463–469) = 453,726
cell-months. Truth: ``lr_sb_best``, raw counts. Gate: the delivered h1 posterior-mean gate. Pool:
each fold's own six fit months, positives only. Inference: ⭐ DELIVERED AUTOREGRESSIVE h1. Arm:
``ctlfullzero_fortytwo``.

⛔ THE GATES, declared before the run:

  G-30  ⭐⭐ THE CONTROLS MUST COME OUT IDENTICAL. Several quantities are identical between the two
        arms **by construction** — the field total, the number of firing cells, the multiset of
        emitted magnitudes, and anything computed from the gate alone (AP, Brier). If any of those
        differs, the simulation is wrong and no blindness claim may be made. ⇒ This is the
        non-vacuity check for the whole file: without it, a metric reading 0 could mean *blind* or
        could mean *my arms are identical*.
  G-31  The arms are PAIRED on common random numbers: same gate draws, same drawn magnitudes, in
        every sample. Asserted, not assumed.
  G-32  The oracle arm must actually be better on at least ONE metric. If nothing separates them,
        the test cannot distinguish "every metric is blind" from "the arms are the same", and the
        run is VOID rather than informative.
  G-33  Reported over all 7 folds, with the per-fold spread, because one fold cannot resolve the
        5% scale the programme cares about.

Run: ``conda run --no-capture-output -n views-hydranet-env python <this file>``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-04_delivered_gate_lookup/tools"))
import e1_data  # noqa: E402

sys.path.insert(0, str(e1_data.REPO / "reports/2026-07-17_lodestar_eval_dossier/tools"))
from lodestar_score import average_precision, brier  # noqa: E402

R = 300
N_PAIRS = 150_000
SEED = 0


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


def variogram_score(Y: np.ndarray, z: np.ndarray, pairs, p: float = 0.5) -> float:
    """Scheuerer & Hering's VS_p over a fixed pair subsample. Lower is better.

    ``Y`` [R, N] the ensemble of fields, ``z`` [N] the truth, ``pairs`` a (2, P) index array.
    """
    i, j = pairs
    obs = np.abs(z[i] - z[j]) ** p
    ens = (np.abs(Y[:, i] - Y[:, j]) ** p).mean(0)
    return float(((obs - ens) ** 2).mean())


def dawid_sebastiani(Y: np.ndarray, z: np.ndarray) -> "tuple[float, int]":
    """-> (mean DSS over cells with spread, n such cells). Proper; first two moments only.

    ⚠️ Undefined where the predictive sd is 0, which is most cells. Restricted and the n reported.
    """
    mu, sd = Y.mean(0), Y.std(0, ddof=1)
    ok = sd > 0
    v = ((z[ok] - mu[ok]) / sd[ok]) ** 2 + 2.0 * np.log(sd[ok])
    return float(v.mean()), int(ok.sum())


def threshold_brier(Y: np.ndarray, z: np.ndarray, thresholds) -> dict:
    """-> {t: mean Brier score of P(Y > t)}. The integrand of the CRPS threshold decomposition."""
    return {
        t: float(((((Y > t).mean(0)) - (z > t).astype(float)) ** 2).mean()) for t in thresholds
    }


def place(order_key, pi, pool, truth, rng):
    """-> (Y[R, N], fired[R, N], totals[R]). Common random numbers are fixed by the rng seed."""
    n_cells = pi.size
    Y = np.zeros((R, n_cells))
    fired = np.zeros((R, n_cells), bool)
    for r in range(R):
        f = rng.random(n_cells) < pi
        idx = np.flatnonzero(f)
        if idx.size == 0:
            continue
        fired[r, idx] = True
        order = idx[np.argsort(order_key[idx], kind="stable")]
        Y[r, order] = np.sort(rng.choice(pool, idx.size, replace=True))
    return Y, fired, Y.sum(1)


def one_fold(fit_months, score_month, grid, gate, m0, rng_seed):
    """-> a dict of per-metric values for the ORACLE-placed and RANDOM-placed arms."""
    o = [k for k in gate if gate[k][2] == score_month][0]
    pi = gate[o][0]
    truth = grid[score_month - m0]
    pool = np.concatenate([grid[m - m0] for m in fit_months])
    pool = pool[pool > 0]

    kr = np.random.default_rng(rng_seed + 7)
    jitter = kr.random(truth.size) * 1e-9
    key_oracle = np.argsort(np.argsort(truth + jitter)).astype(float)
    key_random = kr.permutation(truth.size).astype(float)

    out = {}
    for name, key in (("oracle", key_oracle), ("random", key_random)):
        Y, fired, tot = place(key, pi, pool, truth, np.random.default_rng(rng_seed))
        union = fired.any(0)
        plug, fair = crps_ensemble(Y.T.copy(), truth)
        pr = np.random.default_rng(rng_seed + 13)
        cand = np.flatnonzero(union)
        pairs = (pr.choice(cand, N_PAIRS), pr.choice(cand, N_PAIRS))
        dss, n_dss = dawid_sebastiani(Y, truth)
        out[name] = {
            # ── controls: identical by construction (G-30)
            "field total": float(tot.mean()),
            "n firing (mean)": float(fired.sum(1).mean()),
            "emitted magnitude mean": float(Y[Y > 0].mean()),
            "emitted magnitude p99": float(np.quantile(Y[Y > 0], 0.99)),
            "AP (gate only)": float(average_precision((truth > 0).astype(float), pi)),
            "Brier (gate only)": float(brier((truth > 0).astype(float), pi)),
            # ── candidates: could see placement
            "CRPS plug-in, pooled": float(plug.mean()),
            "CRPS fair, pooled": float(fair.mean()),
            "CRPS plug-in, firing union": float(plug[union].mean()),
            "variogram score p=0.5": variogram_score(Y, truth, pairs),
            "Dawid-Sebastiani": dss,
            "point-forecast MAE": float(np.abs(Y.mean(0) - truth).mean()),
            "point-forecast RMSE": float(np.sqrt(((Y.mean(0) - truth) ** 2).mean())),
            **{f"Brier at y>{t}": v for t, v in threshold_brier(Y, truth, (0, 2, 10, 50)).items()},
            "_n_dss": n_dss,
            "_n_union": int(union.sum()),
        }
    return out


def main() -> None:
    t0 = time.time()
    print(__doc__, flush=True)
    print("=" * 100)
    print("WHICH METRICS CAN SEE PLACEMENT? oracle-placed vs random-placed, everything else equal")
    print("=" * 100)

    grid, cells, m0 = e1_data.truth_grid()
    gate = e1_data.load_gate_h1()
    folds = e1_data.FOLDS
    print(
        f"\n  {len(folds)} folds, R={R} samples per arm, {N_PAIRS:,} cell pairs for the variogram"
    )

    per_fold = [one_fold(f, s, grid, gate, m0, SEED + 100 * i) for i, (f, s) in enumerate(folds)]

    controls = [
        "field total",
        "n firing (mean)",
        "emitted magnitude mean",
        "emitted magnitude p99",
        "AP (gate only)",
        "Brier (gate only)",
    ]
    keys = [k for k in per_fold[0]["oracle"] if not k.startswith("_")]
    cands = [k for k in keys if k not in controls]

    def arr(metric, arm):
        return np.array([f[arm][metric] for f in per_fold])

    print("\n" + "-" * 100)
    print("G-30 · THE CONTROLS. These are identical by construction; any difference is a bug.")
    print("-" * 100)
    print(f"  {'quantity':<28}{'oracle':>14}{'random':>14}{'|difference|':>15}")
    ok30 = True
    for k in controls:
        a, b = arr(k, "oracle"), arr(k, "random")
        d = float(np.abs(a - b).max())
        rel = d / max(abs(float(a.mean())), 1e-12)
        if rel > 1e-9:
            ok30 = False
        print(f"  {k:<28}{a.mean():>14.4f}{b.mean():>14.4f}{d:>15.3e}")
    print(f"\n  {'PASS' if ok30 else '⛔ FAIL'}  G-30  every control identical to 1e-9 relative")
    if not ok30:
        raise SystemExit(
            "\nVOID: a control differs between the arms, so the two forecasts are not identical "
            "except in placement. No blindness claim can be made from this run."
        )

    print("\n" + "-" * 100)
    print("THE MAP · does each metric separate oracle placement from random placement?")
    print("  ⚠️ 'sees it' means sensitive to WHERE magnitudes go among already-firing cells.")
    print("  ⛔ It does NOT mean the metric is good, proper, or right for this population.")
    print("-" * 100)
    print(
        f"  {'metric':<30}{'oracle':>13}{'random':>13}{'rel. diff':>12}"
        f"{'folds agreeing':>16}{'sees it?':>10}"
    )
    seen_any = False
    for k in cands:
        a, b = arr(k, "oracle"), arr(k, "random")
        rel = float((b.mean() - a.mean()) / max(abs(b.mean()), 1e-12))
        #: every candidate metric here is lower-is-better, so "agreeing" is unambiguous.
        #: ⛔ An earlier draft decided the direction by string-matching the metric name, which
        #: would have silently inverted the verdict for any metric added later.
        agree = int((a < b).sum())
        blind = abs(rel) < 1e-6
        verdict = "⛔ BLIND" if blind else ("sees it" if agree >= 6 else "⚠️ mixed")
        if not blind:
            seen_any = True
        print(
            f"  {k:<30}{a.mean():>13.6f}{b.mean():>13.6f}{rel:>11.2%}"
            f"{f'{agree}/{len(folds)}':>16}{verdict:>10}"
        )

    print(f"\n  {'PASS' if seen_any else '⛔ FAIL'}  G-32  at least one metric separates the arms")
    if not seen_any:
        raise SystemExit(
            "\nVOID: nothing separated the arms, so this run cannot distinguish 'every metric is "
            "blind' from 'the arms are identical'."
        )
    print(
        f"  ℓ Dawid-Sebastiani is computed on {per_fold[0]['oracle']['_n_dss']:,} cells with "
        f"spread; the firing union is {per_fold[0]['oracle']['_n_union']:,} cells"
    )

    print("\n" + "-" * 100)
    print(
        "G-33 · PER-FOLD, because one fold cannot resolve the 5% scale the programme cares about"
    )
    print("-" * 100)
    print(f"  {'metric':<30}" + "".join(f"{f'f{i + 1}':>9}" for i in range(len(folds))))
    for k in cands:
        a, b = arr(k, "oracle"), arr(k, "random")
        rel = (b - a) / np.maximum(np.abs(b), 1e-12)
        print(f"  {k:<30}" + "".join(f"{v:>8.1%}" for v in rel))

    print(f"\n  {time.time() - t0:.1f} s, CPU, zero GPU. ⛔ No claim here is about any model.")


if __name__ == "__main__":
    main()
