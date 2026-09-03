"""Shared loader: dumped fields -> per-origin arrays on the study cells, plus truth.

Exists so the **grid flip lives in exactly one place**. C-322: the model field's H axis runs
opposite to priogrid row order, and placing study cells at the naive ``(row-87, col-310)``
correlates 0.026 against the model's own gate while ``(179-row, col)`` correlates 1.0000 with a
max difference of exactly 0. The failure is silent -- every downstream number is well-formed,
plausible, and computed on the wrong cells -- so it must not be re-derived per tool.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

_HYD = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_HYD / "scripts"))
sys.path.insert(0, str(_HYD / "reports" / "2026-07-17_lodestar_eval_dossier" / "tools"))
sys.path.insert(0, str(_HYD / "reports" / "2026-07-25_t0_rollout_skill_dossier" / "tools"))

from lodestar_score import average_precision  # noqa: E402,F401  (re-exported)
from rollout_skill_score import _truth_map  # noqa: E402
from sharpness_scorecard import build_unit_grid  # noqa: E402

GRID = 180
RESULTS = Path(__file__).resolve().parents[1] / "results"
IDENTIFIERS = (
    _HYD / "reports" / "2026-08-21_persistence_reference_dossier" / "results" / "identifiers"
)
TRUTH = (
    _HYD
    / "reports"
    / "2026-07-28_datafactory_migration_dossier"
    / "tools"
    / "v2_truth"
    / "calibration_datafactory_df.parquet"
)
RAW = (
    _HYD.parent
    / "views-models"
    / "models"
    / "fullzero_fortytwo"
    / "data"
    / "raw"
    / "calibration_datafactory_df.parquet"
)
N_ORIGINS = 13


def unit_rowcol(units, umap):
    """Study units -> (rows, cols) into a model field, WITH the C-322 vertical flip."""
    rows = np.array([GRID - 1 - umap[int(u)][0] for u in units])
    cols = np.array([umap[int(u)][1] for u in units])
    if rows.min() < 0 or rows.max() >= GRID or cols.min() < 0 or cols.max() >= GRID:
        raise ValueError("a study unit falls outside the model grid after the flip")
    return rows, cols


def load_origins():
    """[(m0, units)] in origin order, from the preserved per-origin identifier files."""
    files = sorted(
        glob.glob(str(IDENTIFIERS / "*.npz")),
        key=lambda p: int(p.split("origin_")[1].split(".")[0]),
    )
    if not files:
        raise FileNotFoundError(f"no identifier files under {IDENTIFIERS}")
    out = []
    for f in files:
        z = np.load(f)
        t, u = z["time"], z["unit"]
        n = len(np.unique(u))
        out.append((int(t.min()), u.reshape(n, -1)[:, 0]))
    out.sort(key=lambda x: x[0])
    return out


TARGETS = ("sb", "ns", "os")  # regression_targets order; index into mu/gate


def load_truth(origins, horizons, target="sb"):
    """Truth map covering every scored month AND every origin month (m0-1).

    ``m0-1`` is defect #282's month: the arm's own scorer never loaded it, so the first origin's
    persistence was silently all-zeros. Onset/continuation needs it, so it is requested explicitly
    and its absence is an error rather than a zero.
    """
    months = {m0 + h - 1 for m0, _ in origins for h in horizons} | {m0 - 1 for m0, _ in origins}
    tm = _truth_map(str(TRUTH), f"lr_{target}_best", months)
    missing = sorted(
        m for m in months if not any((m, int(u)) in tm for _, us in origins for u in us[:1])
    )
    if missing:
        raise ValueError(
            f"truth is missing months {missing[:5]} — refusing to score them as zeros (#282)"
        )
    return tm


def arm_fields(arm_dir, origins, umap, *, target=0):
    """Yield (m0, units, gate[T, n_cells], mu[T, n_cells]) per origin, on the study cells."""
    dumps = sorted(
        glob.glob(str(Path(arm_dir) / "bodymean_origin*.npz")),
        key=lambda p: int(p.split("origin")[-1].split(".")[0]),
    )
    if len(dumps) != len(origins):
        raise ValueError(
            f"{arm_dir}: {len(dumps)} dumps but {len(origins)} origins — incomplete arm"
        )
    for (m0, units), dp in zip(origins, dumps):
        z = np.load(dp)
        r, c = unit_rowcol(units, umap)
        yield m0, units, z["gate"][:, r, c, target], z["mu"][:, target][:, r, c]


def truth_vec(tm, m0, units, month_offset):
    """Truth values at ``m0 + month_offset`` for these units (0.0 where a cell is absent)."""
    m = m0 + month_offset
    return np.array([tm.get((m, int(u)), 0.0) for u in units])
