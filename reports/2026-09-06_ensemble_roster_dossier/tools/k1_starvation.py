"""K1: is a candidate STARVED at t=36? (06_acceptance_criteria.md)

    recall_mass(h=36, sb) = sum of predicted mean over cells WITH events
                            / sum of truth over those cells
    K1 FIRES if recall_mass < 0.01

`size_ratio` cannot answer this: it reads 0.0000 for every model at every horizon on this vehicle —
saturated, no resolution. `recall_mass` asks the same question with discrimination left in it:
of the deaths that actually occurred, what share did the model put on the right cells?

Reported alongside, not as criteria: waste_frac (share of predicted mass landing on zero-truth
cells) and the same numbers at h18, for shape.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

D = pathlib.Path(__file__).resolve().parent.parent
TRUTH = D / "results" / "v2_truth_validation.parquet"
MODELS = pathlib.Path("/home/simon/Documents/scripts/views_platform/views-models/models")
ROSTER = ["purple_alien", "pink_pirate", "blue_stranger", "bold_comet",
          "blazing_meteor", "heavy_freighter", "bright_starship", "violet_visitor"]
TARGET = "lr_sb_best"


def load(model: str, horizon: int):
    pds = sorted(MODELS.glob(f"{model}/data/generated/predictions_validation_*"))
    if not pds:
        return None, None
    keys, preds = [], []
    for od in sorted(pds[0].glob("origin_*")):
        d = od / TARGET
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
        preds.append(y[sel].mean(1))
    if not keys:
        return None, None
    return np.concatenate(keys), np.concatenate(preds)


def main() -> int:
    df = pd.read_parquet(TRUTH)
    tmap = {(int(u), int(m)): float(v) for (m, u), v in zip(df.index, df[TARGET].to_numpy())}
    print(f"K1 — starvation, target={TARGET}, validation partition\n")
    hdr = f"{'h18 recall':>10s} {'waste%':>7s} | {'h36 recall':>10s} {'waste%':>7s}"
    print(f"{'model':17s} | " + hdr + " | K1")
    print("-" * 74)
    for m in ROSTER:
        out = []
        for h in (18, 36):
            k, p = load(m, h)
            if k is None:
                out.append((float("nan"), float("nan")))
                continue
            tv = np.array([tmap.get(tuple(r), np.nan) for r in k])
            ok = ~np.isnan(tv)
            p, tv = p[ok], tv[ok]
            ev = tv > 0
            out.append((p[ev].sum() / tv.sum() if tv.sum() else float("nan"),
                        100 * p[~ev].sum() / p.sum() if p.sum() else float("nan")))
        r36 = out[1][0]
        verdict = "no data" if np.isnan(r36) else ("*** FIRES" if r36 < 0.01 else "pass")
        print(f"{m:17s} | {out[0][0]:10.4f} {out[0][1]:7.1f} | {out[1][0]:10.4f} "
              f"{out[1][1]:7.1f} | {verdict}")
    print("\nK1 threshold: recall_mass(h36) < 0.01 fires.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
