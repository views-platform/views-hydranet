"""Error correlation + pooled-ensemble scoring over the saved cubes (Q3 of `05_analysis_plan`).

Member SCORES rank models. They do not say which models fail in the SAME PLACES, and an ensemble
gains only from members that fail differently. This computes the quantity the roster choice turns
on and which nothing in this repo had measured.

Per-cell error is |mean(draws) - truth| in count space, on the composed forecast, at one horizon.
Correlation is Pearson over cells, pooled across origins.
"""

from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np

RES = pathlib.Path(__file__).resolve().parent.parent / "results"
TRUTH = pathlib.Path(
    "/home/simon/Documents/scripts/views_platform/views-hydranet/reports/"
    "2026-07-29_v2_scoreboard_dossier/tools"
)
sys.path.insert(0, str(TRUTH))


def load_arm(tag: str, target: str, horizon: int):
    """Return (key, pred_mean) pooled over origins for one arm at one horizon.

    `key` is (unit, time) so arms are aligned on identity, never on row order — two cubes with the
    same shape are not thereby the same cells.
    """
    keys, preds = [], []
    for od in sorted((RES / f"cube_{tag}").glob("origin_*")):
        d = od / target
        if not (d / "y_pred.npy").exists():
            continue
        y = np.load(d / "y_pred.npy")
        ids = np.load(d / "identifiers.npz")
        t, u = ids["time"], ids["unit"]
        # horizon h = the h-th distinct month after the origin's first
        months = np.unique(t)
        if horizon > len(months):
            continue
        m = months[horizon - 1]
        sel = t == m
        keys.append(np.stack([u[sel], t[sel]], 1))
        preds.append(y[sel].mean(1))
    if not keys:
        return None, None
    return np.concatenate(keys), np.concatenate(preds)


def main() -> int:
    target, horizon = "lr_sb_best", 18
    tags = sorted(p.name[5:] for p in RES.glob("cube_*") if p.name.endswith("_cell"))
    print(f"arms: {len(tags)}  target={target}  h={horizon}\n")
    data = {}
    for t in tags:
        k, p = load_arm(t, target, horizon)
        if k is None:
            print(f"  SKIP {t}")
            continue
        data[t] = (k, p)
    # align every arm on the intersection of keys
    base = None
    for t, (k, _) in data.items():
        s = {tuple(r) for r in k}
        base = s if base is None else (base & s)
    order = sorted(base)
    idx = {t: {tuple(r): i for i, r in enumerate(k)} for t, (k, _) in data.items()}
    aligned = {t: np.array([data[t][1][idx[t][o]] for o in order]) for t in data}
    print(f"aligned cells: {len(order)}\n")
    # --- truth, so the correlation is of ERRORS (what 05 pre-registered), not predictions ---
    import pandas as pd

    tp = pathlib.Path(
        "/home/simon/Documents/scripts/views_platform/views-hydranet/reports/"
        "2026-07-28_datafactory_migration_dossier/tools/v2_truth/"
        "calibration_datafactory_df.parquet"
    )
    df = pd.read_parquet(tp)
    tcol = target  # the truth parquet's columns ARE the target names
    tmap = {}
    ix = df.index
    if isinstance(ix, pd.MultiIndex):
        for (mm, uu), v in zip(ix, df[tcol].to_numpy()):
            tmap[(int(uu), int(mm))] = float(v)
    truth = np.array([tmap.get(o, np.nan) for o in order])
    ok = ~np.isnan(truth)
    print(f"truth matched: {ok.sum()}/{len(order)} cells  (events: {(truth[ok] > 0).sum()})")
    names = sorted(aligned)
    err = {n: np.abs(aligned[n][ok] - truth[ok]) for n in names}
    M = np.corrcoef(np.stack([err[n] for n in names]))
    print("\n=== ERROR correlation between arms (lower = more diverse) ===")
    short = {n: n.replace("_cell", "").replace("_gate", "").replace("threshold", "hard")
             .replace("soft", "soft") for n in names}
    print(f"{'':26s}" + "".join(f"{i:>6d}" for i in range(len(names))))
    for i, n in enumerate(names):
        row = "".join(f"{M[i, j]:6.2f}" for j in range(len(names)))
        print(f"{i:2d} {short[n][:23]:23s}" + row)
    print("\n=== least-correlated PAIRS ===")
    pairs = sorted(
        ((M[i, j], names[i], names[j]) for i, j in itertools.combinations(range(len(names)), 2))
    )
    for c, a, b in pairs[:8]:
        print(f"  {c:5.2f}  {short[a][:26]:26s} x  {short[b][:26]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
