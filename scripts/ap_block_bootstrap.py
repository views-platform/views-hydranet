#!/usr/bin/env python3
"""ap_block_bootstrap.py — an origin-block CI and MDE for gate AP, so FG-C has a real number.

``floor_gate``'s FG-C clause asks whether the pre-registered effect exceeds the measurement's
resolution. That needs a minimum detectable effect on **AP**, and the repo had one only for CRPS:
``rollout_skill_score.block_bootstrap_crps`` implements the origin-resampling loop but hard-codes
``crps_ensemble``. This swaps the statistic and reuses that loop's structure verbatim.

Why origins and not cells
-------------------------
There are 13 origins and ~170k cells. Cells within an origin share a forecast state and
are nowhere near independent, so an iid-over-cells interval is far too narrow.
``gw_stratified._bootstrap_mean_ci`` draws the same distinction and labels its iid mode
"contrast only". At P=13 the origin-block construction is the only one with a valid error
bar, so it is the one the gate and the test use.

AP is not decomposable per origin
---------------------------------
``average_precision`` is a global rank statistic over the pooled score vector. It is not a
mean of per-origin values and cannot be recombined from origin-level pieces — the same
property that makes AP, ``size_ratio`` and the top-k columns un-decomposable in
``score_v2_horizons``. This module therefore recomputes AP from scratch on every resampled
cell set, which is why it is slower than a mean-based bootstrap and why ``n_boot`` defaults
to 400 rather than 2000.

The MDE
-------
``mde = half-width of the (1-alpha) origin-block interval`` — the smallest AP difference
this setup could resolve at that confidence, which is exactly what FG-C compares the
pre-registered effect against.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HN = Path(__file__).resolve().parent.parent
for _p in (
    _HN / "reports/2026-07-17_lodestar_eval_dossier/tools",
    _HN / "reports/2026-07-25_t0_rollout_skill_dossier/tools",
    _HN / "reports/2026-07-28_datafactory_migration_dossier/tools",
):
    sys.path.insert(0, str(_p))

from lodestar_score import average_precision  # noqa: E402
from rollout_skill_score import (  # noqa: E402
    _fixed_support,
    _truth_map,
    gather_all_horizons,
)

__all__ = ["ap_origin_block_ci", "mde_ap"]


def ap_origin_block_ci(
    *,
    pred_dir: str,
    truth_parquet: str,
    target: str,
    horizons: tuple[int, ...],
    lr_tmpl: str = "lr_{t}_best",
    by_tmpl: str | None = "by_{t}_best",
    n_boot: int = 400,
    seed: int = 0,
    ci: float = 0.90,
) -> dict[int, dict[str, float]]:
    """Origin-block bootstrap CI on gate AP, per horizon.

    Returns ``{h: {"ap", "lo", "hi", "mde", "n_origins"}}``. Deterministic given ``seed``.

    Raises:
        ValueError: fewer than 3 origins (a block bootstrap over 2 units is meaningless), or a
            horizon whose truth is degenerate (all-zero labels make AP undefined).
    """
    g = gather_all_horizons(
        pred_dir, lr_tmpl.format(t=target), by_tmpl.format(t=target) if by_tmpl else None
    )
    support = _fixed_support({"arm": g}, horizons)
    origins = sorted({m0 for (m0, _u) in support})
    if len(origins) < 3:
        raise ValueError(
            f"ap_origin_block_ci: {len(origins)} origin(s) — a block bootstrap over fewer than 3 "
            "units cannot produce a meaningful interval"
        )
    by_origin: dict[int, list] = {o: [] for o in origins}
    for m0, u in support:
        by_origin[m0].append((m0, u))

    months = {m0 + h - 1 for (m0, _u) in support for h in horizons}
    tmap = _truth_map(truth_parquet, f"lr_{target}_best", months)
    has_gate = g[(support[0][0], horizons[0], support[0][1])][1] is not None
    rng = np.random.default_rng(seed)
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    out: dict[int, dict[str, float]] = {}

    for h in horizons:

        def _ap(cells) -> float:
            cs = np.stack([g[(m0, h, u)][0] for (m0, u) in cells])
            p = np.array([g[(m0, h, u)][1] for (m0, u) in cells]) if has_gate else (cs > 0).mean(1)
            truth = np.array([tmap[(m0 + h - 1, u)] for (m0, u) in cells], dtype=float)
            return float(average_precision((truth > 0).astype(float), p))

        point = _ap(support)
        if not np.isfinite(point):
            raise ValueError(
                f"ap_origin_block_ci: AP is not finite at h={h} — the label vector is degenerate"
            )
        vals = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.choice(origins, size=len(origins), replace=True)
            vals[b] = _ap([c for o in pick for c in by_origin[o]])
        lo, hi = float(np.percentile(vals, lo_q)), float(np.percentile(vals, hi_q))
        out[h] = {
            "ap": point,
            "lo": lo,
            "hi": hi,
            "mde": (hi - lo) / 2.0,
            "n_origins": float(len(origins)),
        }
    return out


def mde_ap(ci_row: dict[str, float]) -> float:
    """The half-width — the smallest AP difference this setup could resolve."""
    return ci_row["mde"]


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--target", default="sb")
    ap.add_argument("--horizons", default="1,18")
    ap.add_argument("--truth", default=None)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    truth = a.truth
    if truth is None:
        sys.path.insert(0, str(_HN / "reports/2026-07-28_datafactory_migration_dossier/tools"))
        from v2_ruler import V2_TRUTH

        truth = str(V2_TRUTH)

    res = ap_origin_block_ci(
        pred_dir=a.pred_dir,
        truth_parquet=truth,
        target=a.target,
        horizons=tuple(int(x) for x in a.horizons.split(",")),
        n_boot=a.n_boot,
        seed=a.seed,
    )
    text = json.dumps({str(k): v for k, v in res.items()}, indent=2)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
