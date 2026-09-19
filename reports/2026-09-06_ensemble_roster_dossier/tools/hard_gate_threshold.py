"""M75 — where the hard-gate threshold τ would have to sit to fire on as many cells, or as many
fatalities, as were observed.

Uses each model's own gate and body-mean dump (last origin → 36 forecast months) against the
validation truth for those same months. Nothing is retrained or re-emitted.

Two criteria, one answer each per model:
  cells:  τ such that  #{cell-months with gate >= τ}          == #{cell-months with truth > 0}
  fatal:  τ such that  Σ_{gate >= τ} E[Y|body]                == Σ truth

⚠️ Matches against the FORECAST WINDOW's own truth. Operationally τ must be derived from the
trailing 36 months before the origin — the base rate is what drives it, so expect a similar number,
but this script is the answer to "roughly where", not the production rule.

Run from the repo root:
    python reports/2026-09-06_ensemble_roster_dossier/tools/hard_gate_threshold.py [--target sb]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_forecast_maps as pm  # noqa: E402  (dump reading, orientation, truth placement)

TAUS = np.array([0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5])


def crossing(curve: np.ndarray, target: float) -> float:
    """τ at which a curve DECREASING in τ crosses `target`; nan if it never does."""
    for i in range(len(TAUS) - 1):
        a, b = curve[i], curve[i + 1]
        if a >= target >= b:
            w = 0.0 if a == b else (a - target) / (a - b)
            return float(TAUS[i] + w * (TAUS[i + 1] - TAUS[i]))
    return float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="sb", choices=pm.TARGETS)
    args = ap.parse_args()
    tgt = args.target
    umap = pm.build_unit_grid(str(pm.GEOMETRY))

    rows = []
    for model in pm.ROSTER:
        mu, gate, _origin = pm.read_dump(pm.dump_path(model, -1), tgt)
        months = pm.cube_months(model, -1, list(range(1, 37)), tgt)
        flip = pm.resolve_orientation(model, umap, gate[0], 1, tgt)
        truth = np.stack([pm.truth_grid(umap, m, flip, tgt) for m in months])  # raw counts
        mask = ~np.isnan(truth)  # the study cells; the model grid is larger
        obs_cells = int((truth[mask] > 0).sum())
        obs_fat = float(truth[mask].sum())
        g, b = gate[mask], mu[mask]
        n_cells = np.array([(g >= t).sum() for t in TAUS], dtype=float)
        fat = np.array([(b * (g >= t)).sum() for t in TAUS])
        rows.append(
            (
                model,
                obs_cells,
                obs_fat,
                crossing(n_cells, obs_cells),
                crossing(fat, obs_fat),
                int(n_cells[-1]),
                float(g.sum()),
                float((g * b).sum()),
                n_cells,
                fat,
            )
        )

    print(f"\ntarget={tgt}  36 forecast months, {int(mask.sum()):,} study cell-months")
    print(
        f"{'model':<16} {'obs cells':>9} {'τ(cells)':>9} {'obs fatal':>10} {'τ(fatal)':>9} "
        f"{'cells@0.5':>9} {'Σgate':>8} {'Σgate·μ':>9}"
    )
    for m, oc, of, tc, tf, c5, sg, sgm, *_ in rows:
        print(
            f"{m:<16} {oc:>9,} {tc:>9.3f} {of:>10,.0f} {tf:>9.3f} {c5:>9,} {sg:>8,.0f} "
            f"{sgm:>9,.0f}"
        )
    print("\nτ grid:", TAUS.tolist())
    for m, *_, nc, ft in rows:
        print(f"  {m:<16} cells:", " ".join(f"{int(x):>6}" for x in nc))
        print(f"  {'':<16} fatal:", " ".join(f"{int(x):>6}" for x in ft))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
