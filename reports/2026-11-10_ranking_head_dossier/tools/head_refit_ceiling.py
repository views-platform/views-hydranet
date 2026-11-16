# ⛔⛔⛔ DEPRECATED — DO NOT USE. ⛔⛔⛔
# Deprecated 2026-11-16 by Simon Polichinel von der Maase. Not to be used, for now and until
# further notice from him.
#
# Reason: this tool produces or consumes TEACHER-FORCED measurements (the model is handed the
# observed truth at every month instead of its own output). Teacher forcing has confounded and
# ruined multiple experiments in this repository and produced false conclusions. Simon's
# decision: "There will be no more teacher forcing."
#
# The teacher-forced comparison is removed from the ranking plan (04_roadmap.md). The ranking
# experiment will be compared against a NEW control run, made after the body refactor and
# WITHOUT teacher forcing. No number this tool has produced may be cited or compared.
# The file is kept only so that nothing is deleted before Simon says so.

"""head_refit_ceiling.py — is the body's ordering limited by the EMBEDDING or by the 289
coefficients that read it?

**The question, in one line.** `HydraBNrecurrentUnet_06_LSTM4.py:642-643` ends the magnitude
pathway with a `Conv2d(32 -> 2, 3x3)` followed only by a monotone activation, so the body's
ordering of cells is the rank of **one linear functional of a 32-channel embedding**. Those
coefficients were fitted by the negative binomial's log-likelihood. Nobody ever asked them to get
the ORDER right. ⇒ Refit them against rank agreement, with the trunk frozen, and see whether the
ordering moves.

⭐ **What this can decide, and what it cannot.** It is the one-layer version of the GPU experiment:
same embedding, different objective. If a refit beats the body, the embedding already carries
ranking information the NB fit discards. If no refit beats it, the embedding is the ceiling **for a
linear read of it** — which does NOT settle the real question, because a trained ranking head also
changes the trunk that MAKES the embedding, and that is only reachable on a GPU. ⛔ Neither outcome
is a verdict on the ranking body; `04_roadmap` phase 3's pre-commitment and `00_README` §6's kill
rule both apply.

⛔ **NO NEW ESTIMATOR.** `midrank` and `spearman` are imported from
`../../2026-11-10_rank_coupling_verdict/tools/rank_coupling_verdict.py` — the module that published
`+0.3127` — and `rho_event`, `_summarise`, `distinct_share` and `_fold_inputs` from
`rho_event_readout.py`, so both sides of every comparison are literally the same function objects.
The offset resolver is `rho_event_teacher_forced.resolve_offset`, not a reimplementation.

⭐⭐ **THREE GATES, and the arm rows are refused if any fails.**

* **G-1 · the control pin.** The body's own score, read through this script's own population,
  offset and event mask, must reproduce [ERASED 2026-11-16 by Simon: teacher-forced number, not to be used] to `_TOL`. That pins the hook, the offset, the
  month map and the firing set in one arithmetic check. ⚠️ The field here is the conv's
  **pre-activation** output, so passing G-1 additionally PROVES the activation is rank-preserving
  on this population — if it were not, the number would differ.
* **G-BODY · the nesting gate.** A refit given the body's score as its ONLY predictor must return
  the control's ρ exactly. A linear map of one predictor cannot reorder anything unless its
  coefficient is negative, so any deviation means the fitting path is broken, not informative.
* **G-2 · distinct values.** §0.2 of `2026-09-30_evaluation_learning_report.md`: a delivery served
  0.0 for all 2,330,712 cells and every structural check passed. Every row's distinct share is
  printed beside it.

⛔⛔ **OUT-OF-FOLD BY DEFAULT, and the in-sample row is LABELLED A CEILING.** Coefficients are
fitted on the fold's own six-month fit window and read on its score month — the pre-registered
design, unchanged. The in-sample row (fit and read the same month) is reported **only** as an upper
bound on what any linear read could reach, never as a result. Fitting 33 coefficients on ~148
events and reading them on the same 148 would be guaranteed to "win".

⛔ **The ridge sweep is reported WHOLE** (`C-393`): quoting the best lambda without the grid beside
it is not a result.

**Population, per `C-351`:** 7 folds (fixed six-month fit, one score month, months 463..469), `sb`,
h1, truth `lr_sb_best` in raw counts, no transform. `ρ_event` is Spearman on **midranks** between
the score and the truth over the **event cells** of the firing set — `open_m & (truth > 0)` at `τ`
from the arm config — on the **global** 360x720 grid. Mean ± SD over the 7 folds at `ddof=1`.
⛔ `input_mode="teacher_forced"` throughout (`C-399`): comparable to [ERASED 2026-11-16 by Simon: teacher-forced number, not to be used], never to `+0.3127`.

The regression target is the truth's midrank **within each fit month**, mapped to `(r - 0.5) / n_m`
so that months with different event counts contribute on one scale. Features are z-scored on the
fit window's own statistics, never the score month's.

Usage::

    python head_refit_ceiling.py --emb results/emb_grid_CTL_EMB_calm_lander.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-04_delivered_gate_lookup/tools"))
sys.path.insert(0, str(_HERE.parents[1] / "2026-11-10_rank_coupling_verdict/tools"))

import arm_config  # noqa: E402
import e1_data  # noqa: E402
from rank_coupling_verdict import midrank  # noqa: E402
from rho_event_readout import (  # noqa: E402
    _DISTINCT_FLOOR,
    DDOF,
    _fold_inputs,
    _summarise,
    distinct_share,
    rho_event,
)
from rho_event_teacher_forced import (  # noqa: E402
    _ANCHOR_TF_MU,
    _ANCHOR_TF_MU_SD,
    _TOL,
    forecast_index,
    resolve_offset,
)

#: The ridge grid, swept and reported whole. 0.0 is plain least squares.
_LAMBDAS = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
_TF_MODE = "teacher_forced"


def _per_month_masks(months: list[int]) -> dict[int, dict]:
    """-> {month: {truth, open_m, event_m}} for the DELIVERED gate's firing set the folds use."""
    grid, _cells, m0 = e1_data.truth_grid()
    gates = {mo: g for (g, _u, mo) in e1_data.load_gate_t0().values()}
    tau = arm_config.resolve_tau()
    out = {}
    for m in months:
        truth = grid[m - m0].astype(np.float64)
        open_m = arm_config.open_set(gates[m], tau)
        out[m] = {"truth": truth, "open_m": open_m, "event_m": open_m & (truth > 0)}
    return out


def _design(feats: dict[int, np.ndarray], months: list[int], masks: dict[int, dict]):
    """-> (X [n_events, n_feat], y [n_events]) pooled over ``months``, y = within-month midrank."""
    xs, ys = [], []
    for m in months:
        ev = masks[m]["event_m"]
        n = int(ev.sum())
        if n < 2:
            continue
        xs.append(feats[m][:, ev].T)
        ys.append((midrank(masks[m]["truth"][ev]) - 0.5) / n)
    return np.vstack(xs), np.concatenate(ys)


def _fit_ridge(
    X: np.ndarray, y: np.ndarray, lam: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (w, mu, sd). Ridge on z-scored features; the intercept is never penalised.

    ⛔ `lstsq`, not `solve`. The normal equations ARE singular here at lambda 0: a ReLU'd
    embedding channel that is zero on every event cell of a fit window gives a constant column,
    and `solve` raised `LinAlgError` on the first run. `lstsq` returns the minimum-norm solution
    instead, which is the right answer for a rank-deficient design and keeps the lambda 0 row in
    the sweep rather than dropping it.
    """
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((X.shape[0], 1)), (X - mu) / sd])
    A = Z.T @ Z
    if lam > 0:
        pen = np.eye(A.shape[0]) * lam
        pen[0, 0] = 0.0
        A = A + pen
    w = np.linalg.lstsq(A, Z.T @ y, rcond=None)[0]
    return w, mu, sd


def _apply(w: np.ndarray, mu: np.ndarray, sd: np.ndarray, F: np.ndarray) -> np.ndarray:
    """-> the fitted linear score for every cell of a [n_feat, n_cells] field."""
    return w[0] + ((F.T - mu) / sd) @ w[1:]


def _rows_for(
    feats: dict[int, np.ndarray],
    folds: list[dict],
    masks: dict[int, dict],
    lam: float,
    in_sample: bool,
) -> tuple[list[float], float, int]:
    """-> (per-fold rho, mean distinct share on firing cells, n_fit pooled over folds)."""
    rhos, shares, n_fit = [], [], 0
    for fit_months, score_month in e1_data.FOLDS:
        train = [score_month] if in_sample else [m for m in fit_months if m in masks]
        X, y = _design(feats, train, masks)
        n_fit += X.shape[0]
        w, mu, sd = _fit_ridge(X, y, lam)
        s = _apply(w, mu, sd, feats[score_month])
        f = next(f for f in folds if f["month"] == score_month)
        rhos.append(rho_event(s, f["truth"], f["event_m"]))
        shares.append(distinct_share(s[f["open_m"]]))
    return rhos, float(np.mean(shares)), n_fit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--emb", type=Path, required=True, help="embedding_on_the_grid.py output")
    args = ap.parse_args()

    t0 = time.time()
    print(__doc__.split("Usage::")[0])
    print("=" * 100)
    folds, tau, cells = _fold_inputs()
    months = [f["month"] for f in folds]
    fit_months = sorted({m for fm, _ in e1_data.FOLDS for m in fm})
    print(
        f"  population: {len(folds)} folds, score months {months[0]}..{months[-1]}, fit months "
        f"{fit_months[0]}..{fit_months[-1]}, sb h1,\n  firing set = DELIVERED gate >= tau {tau} · "
        f"truth in raw counts · ddof={DDOF} · input_mode={_TF_MODE}"
    )

    z = np.load(args.emb, allow_pickle=False)
    if str(z["input_mode"]) != _TF_MODE:
        print(f"  ⛔ REFUSED: input_mode={z['input_mode']!s}, not {_TF_MODE} (C-399).")
        return 2
    if not np.array_equal(z["land_cells"], cells):
        print("  ⛔ REFUSED: the embedding's land_cells are not this design's cells, in order.")
        return 2
    fed = [int(m) for m in z["months"]]
    emb, mu_pre, gate = z["emb"], z["mu_pre"], z["gate"]
    print(
        f"  embedding {emb.shape[1]} channels · fed months {fed[0]}..{fed[-1]} · "
        f"sha {str(z['sha'])[:16]}… · head {str(z['head_conv'])}"
    )

    off = resolve_offset(gate, fed)
    idx = forecast_index(fed, off)
    need = set(months) | set(fit_months)
    missing = sorted(need - set(idx))
    if missing:
        print(f"  ⛔ REFUSED: the dump does not cover forecast months {missing}.")
        return 2
    masks = _per_month_masks(sorted(need))
    body = {m: mu_pre[i][None, :].astype(np.float64) for m, i in idx.items() if m in need}
    embf = {m: emb[i].astype(np.float64) for m, i in idx.items() if m in need}
    nested = {m: np.vstack([body[m], embf[m]]) for m in need}
    n_ev = sum(int(masks[m]["event_m"].sum()) for m in months)
    print(
        f"  {n_ev:,} event cell-months on the score months · "
        f"{sum(int(masks[m]['event_m'].sum()) for m in fit_months):,} on the fit months"
    )

    # ---- G-1 -------------------------------------------------------------------------------
    ctl = [rho_event(body[f["month"]][0], f["truth"], f["event_m"]) for f in folds]
    ctl_share = float(np.mean([distinct_share(body[f["month"]][0][f["open_m"]]) for f in folds]))
    c_mean, c_sd = _summarise(ctl)
    d_mean, d_sd = abs(c_mean - _ANCHOR_TF_MU), abs(c_sd - _ANCHOR_TF_MU_SD)
    ok = d_mean <= _TOL and d_sd <= _TOL
    print(
        f"\n  {'⭐ PASS' if ok else '⛔ FAIL'} G-1 · the body's PRE-ACTIVATION score reads "
        f"{c_mean:+.4f} ± {c_sd:.4f} against the pin {_ANCHOR_TF_MU:+.4f} ± "
        f"{_ANCHOR_TF_MU_SD:.4f} (|Δ| {d_mean:.2e} / {d_sd:.2e}, tol {_TOL:.0e})"
    )
    if not ok:
        print(
            "     REFUSED: the control did not reproduce, so the hook, the offset, the month map "
            "or the\n     firing set is not the one that was pinned. No refit row may be printed."
        )
        return 1
    print(
        "     ⇒ and therefore the monotone activation is rank-preserving on this population: the "
        "pin was\n       measured post-activation and this field is pre-activation."
    )

    # ---- G-BODY ----------------------------------------------------------------------------
    b_rhos, _b_share, _n = _rows_for(body, folds, masks, 0.0, in_sample=False)
    b_mean, b_sd = _summarise(b_rhos)
    g_ok = abs(b_mean - c_mean) <= _TOL and abs(b_sd - c_sd) <= _TOL
    print(
        f"\n  {'⭐ PASS' if g_ok else '⛔ FAIL'} G-BODY · a refit on the body's score ALONE reads "
        f"{b_mean:+.4f} ± {b_sd:.4f}, the control to "
        f"{max(abs(b_mean - c_mean), abs(b_sd - c_sd)):.1e}"
    )
    if not g_ok:
        print("     REFUSED: the fitting path reorders a single predictor. It is broken.")
        return 1

    # ---- the readout -----------------------------------------------------------------------
    sets = {"NESTED  (mu+emb32)": nested, "EMB ONLY (emb32)": embf}
    print("\n  OUT-OF-FOLD · fit on the fold's six-month window, read on its score month")
    print(
        f"  {'index':<22}{'lambda':>9}{'rho_event':>12}{'SD':>9}{'vs BODY':>10}"
        f"{'distinct':>11}{'n_fit':>8}"
    )
    print(
        f"  {'BODY mu (TF) · control':<22}{'—':>9}{c_mean:>+12.4f}{c_sd:>9.4f}"
        f"{0.0:>+10.4f}{ctl_share:>11.6f}{'—':>8}"
    )
    gate_rhos = [
        rho_event(gate[idx[f["month"]]].astype(np.float64), f["truth"], f["event_m"])
        for f in folds
    ]
    g_mean, g_sd = _summarise(gate_rhos)
    g_share = float(np.mean([distinct_share(gate[idx[f["month"]]][f["open_m"]]) for f in folds]))
    print(
        f"  {'GATE (TF) · context':<22}{'—':>9}{g_mean:>+12.4f}{g_sd:>9.4f}"
        f"{g_mean - c_mean:>+10.4f}{g_share:>11.6f}{'—':>8}"
    )

    best = {}
    for label, feats in sets.items():
        for lam in _LAMBDAS:
            rhos, share, n_fit = _rows_for(feats, folds, masks, lam, in_sample=False)
            mean, sd = _summarise(rhos)
            flag = "" if share >= _DISTINCT_FLOOR else "  ⛔ G-2"
            print(
                f"  {label:<22}{lam:>9.3g}{mean:>+12.4f}{sd:>9.4f}{mean - c_mean:>+10.4f}"
                f"{share:>11.6f}{n_fit:>8,}{flag}"
            )
            if label not in best or mean > best[label][1]:
                best[label] = (lam, mean, sd, rhos)
        print()

    print("  IN-SAMPLE CEILING · ⛔ fit and read the SAME month — an upper bound, never a result")
    for label, feats in sets.items():
        rhos, share, n_fit = _rows_for(feats, folds, masks, 0.0, in_sample=True)
        mean, sd = _summarise(rhos)
        print(
            f"  {label:<22}{0.0:>9.3g}{mean:>+12.4f}{sd:>9.4f}{mean - c_mean:>+10.4f}"
            f"{share:>11.6f}{n_fit:>8,}"
        )

    print("\n  ⇒ best out-of-fold per feature set, with the paired per-fold gap against the body")
    for label, (lam, mean, sd, rhos) in best.items():
        gaps = np.asarray(rhos) - np.asarray(ctl)
        gm, gs = _summarise(list(gaps))
        wins = int((gaps > 0).sum())
        print(
            f"  {label:<22} lambda {lam:<8.3g} rho {mean:+.4f} ± {sd:.4f} · "
            f"paired gap {gm:+.4f} ± {gs:.4f} (ddof={DDOF}) · {wins}/{len(gaps)} folds better"
        )

    print(
        "\n  ⛔ INSTRUMENT READING, not a verdict. A refit that loses rules out only the CHEAP "
        "half of the\n     question: the trunk that MAKES the embedding is unchanged here, and "
        "changing it needs a GPU."
    )
    print(f"\n  {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
