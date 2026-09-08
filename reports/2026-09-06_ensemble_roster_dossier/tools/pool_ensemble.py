"""Pool candidate member sets by concat and score the POOLED forecast (Q3 / F2 of `05`).

Member scores rank models; they do not say what a pool does. `rusty_bucket` concatenates draws on
the sample axis, so pooling is exact here: stack the members' draw arrays. This scores the pool the
same way a member is scored, on the same cells, so the comparison is like-for-like.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

RES = pathlib.Path(__file__).resolve().parent.parent / "results"
TRUTH = pathlib.Path(
    "/home/simon/Documents/scripts/views_platform/views-hydranet/reports/"
    "2026-07-28_datafactory_migration_dossier/tools/v2_truth/calibration_datafactory_df.parquet"
)


def ap(y_true_bin: np.ndarray, score: np.ndarray) -> float:
    """Average precision, ranking-based — the same definition the frozen ruler uses."""
    o = np.argsort(-score, kind="mergesort")
    y = y_true_bin[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    n_pos = y.sum()
    return float((prec * y).sum() / n_pos) if n_pos else float("nan")


def load(tag: str, target: str, horizon: int):
    keys, draws = [], []
    for od in sorted((RES / f"cube_{tag}").glob("origin_*")):
        d = od / target
        if not (d / "y_pred.npy").exists():
            continue
        y = np.load(d / "y_pred.npy")
        ids = np.load(d / "identifiers.npz")
        t, u = ids["time"], ids["unit"]
        months = np.unique(t)
        if horizon > len(months):
            continue
        sel = t == months[horizon - 1]
        keys.append(np.stack([u[sel], t[sel]], 1))
        draws.append(y[sel])
    return np.concatenate(keys), np.concatenate(draws)


def main() -> int:
    MODELS = ["violet_visitor", "bright_starship", "bold_comet", "blazing_meteor",
              "heavy_freighter", "pink_pirate", "blue_stranger", "purple_alien"]
    tags = sorted(p.name[5:] for p in RES.glob("cube_*") if p.name.endswith("_cell"))
    df = pd.read_parquet(TRUTH)
    grand = {}
    for target in ("lr_sb_best", "lr_ns_best", "lr_os_best"):
        tmap = {(int(u), int(m)): float(v) for (m, u), v in zip(df.index, df[target].to_numpy())}
        for horizon in (1, 6, 12, 18, 24, 36):
            data = {tg: load(tg, target, horizon) for tg in tags}
            if any(k is None or len(k) == 0 for k, _ in data.values()):
                continue
            base = None
            for k, _ in data.values():
                s = {tuple(r) for r in k}
                base = s if base is None else base & s
            order = sorted(base)
            pos = {tg: {tuple(r): i for i, r in enumerate(k)} for tg, (k, _) in data.items()}
            al = {tg: data[tg][1][[pos[tg][o] for o in order]] for tg in tags}
            truth = np.array([tmap.get(o, np.nan) for o in order])
            ok = ~np.isnan(truth)
            ybin = (truth[ok] > 0).astype(float)
            if ybin.sum() < 20:
                continue

            def sc(members, _al=al, _ok=ok, _yb=ybin):
                pool = np.concatenate([_al[m][_ok] for m in members], axis=1)
                return ap(_yb, pool.mean(1))

            solo = sorted(((sc([tg]), tg) for tg in tags), reverse=True)
            smap = {tg: a for a, tg in solo}
            cands = {
                "incumbent8_soft": [f"{m}_soft_gate_cell" for m in MODELS],
                "all8_threshold": [f"{m}_threshold_gate_cell" for m in MODELS],
                "per_model_best": [
                    max([f"{m}_soft_gate_cell", f"{m}_threshold_gate_cell"], key=lambda x: smap[x])
                    for m in MODELS
                ],
                "top8_arms": [tg for _, tg in solo[:8]],
                "best4_x_both": [
                    f"{m}_{c}_cell"
                    for m in ["violet_visitor", "purple_alien", "bright_starship", "blue_stranger"]
                    for c in ("soft_gate", "threshold_gate")
                ],
            }
            row = {k: sc(v) for k, v in cands.items()}
            row["best_solo"] = solo[0][0]
            grand[(target, horizon)] = row
            print(f"{target} h={horizon:<2d} events={int(ybin.sum()):5d} | " +
                  " ".join(f"{k}={v:.4f}" for k, v in row.items()), flush=True)

    print("\n=== SUMMARY: mean rank of each roster across all target x horizon cells ===")
    keys = ["incumbent8_soft", "all8_threshold", "per_model_best", "top8_arms", "best4_x_both"]
    import statistics as st
    for k in keys:
        vals = [r[k] for r in grand.values()]
        wins = sum(1 for r in grand.values() if r[k] == max(r[x] for x in keys))
        print(f"  {k:18s} mean AP {st.mean(vals):.4f}  best-in-cell {wins}/{len(grand)}")
    beats = sum(1 for r in grand.values() if max(r[x] for x in keys) > r["best_solo"])
    print(f"\n  pool beats best single member in {beats}/{len(grand)} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
