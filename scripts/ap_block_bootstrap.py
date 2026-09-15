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

Retention, and why it needs its own bootstrap
---------------------------------------------
``retention = AP(h18) / AP(h1)`` is a co-primary endpoint of the lesson-curve programme
(``reports/2026-08-18_lesson_curve_dossier``). Its interval cannot be obtained by propagating the
two per-horizon intervals above: both horizons are scored on the *same* 13 origins, so their
errors are correlated and independent propagation is not a valid construction. Note the bias does
not have a predictable sign — measured on the 40-lesson cube the paired interval came out ~13%
**wider** than propagation suggests (mde 0.0104 vs 0.0092), not narrower.
``ap_ratio_origin_block_ci`` therefore draws origins **once per replicate** and computes both APs
on that one resampled cell set, which is the paired construction the correlation demands.
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

__all__ = ["ap_origin_block_ci", "ap_ratio_origin_block_ci", "mde_ap"]


def _load_indexed(
    *,
    pred_dir: str,
    truth_parquet: str,
    target: str,
    horizons: tuple[int, ...],
    lr_tmpl: str,
    by_tmpl: str | None,
    who: str,
):
    """Load one cube, fix the support, index it by origin, and pull the truth it needs.

    Shared verbatim by the two bootstraps below so they resample the *same* units — the ratio
    bootstrap is only paired if its two horizons sit on one support.
    """
    g = gather_all_horizons(
        pred_dir, lr_tmpl.format(t=target), by_tmpl.format(t=target) if by_tmpl else None
    )
    support = _fixed_support({"arm": g}, horizons)
    origins = sorted({m0 for (m0, _u) in support})
    if len(origins) < 3:
        raise ValueError(
            f"{who}: {len(origins)} origin(s) — a block bootstrap over fewer than 3 "
            "units cannot produce a meaningful interval"
        )
    by_origin: dict[int, list] = {o: [] for o in origins}
    for m0, u in support:
        by_origin[m0].append((m0, u))

    months = {m0 + h - 1 for (m0, _u) in support for h in horizons}
    tmap = _truth_map(truth_parquet, f"lr_{target}_best", months)
    has_gate = g[(support[0][0], horizons[0], support[0][1])][1] is not None
    return g, support, origins, by_origin, tmap, has_gate


def _ap_fn(g, tmap, has_gate, h):
    """AP over an explicit cell list at one horizon — recomputed from scratch, never recombined."""

    def _ap(cells) -> float:
        cs = np.stack([g[(m0, h, u)][0] for (m0, u) in cells])
        p = np.array([g[(m0, h, u)][1] for (m0, u) in cells]) if has_gate else (cs > 0).mean(1)
        truth = np.array([tmap[(m0 + h - 1, u)] for (m0, u) in cells], dtype=float)
        return float(average_precision((truth > 0).astype(float), p))

    return _ap


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
    g, support, origins, by_origin, tmap, has_gate = _load_indexed(
        pred_dir=pred_dir,
        truth_parquet=truth_parquet,
        target=target,
        horizons=horizons,
        lr_tmpl=lr_tmpl,
        by_tmpl=by_tmpl,
        who="ap_origin_block_ci",
    )
    rng = np.random.default_rng(seed)
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    out: dict[int, dict[str, float]] = {}

    for h in horizons:
        _ap = _ap_fn(g, tmap, has_gate, h)
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


def ap_ratio_origin_block_ci(
    *,
    pred_dir: str,
    truth_parquet: str,
    target: str,
    numerator_h: int = 18,
    denominator_h: int = 1,
    lr_tmpl: str = "lr_{t}_best",
    by_tmpl: str | None = "by_{t}_best",
    n_boot: int = 400,
    seed: int = 0,
    ci: float = 0.90,
) -> dict[str, float]:
    """Paired origin-block bootstrap CI on the AP ratio ``AP(numerator_h)/AP(denominator_h)``.

    One origin draw per replicate feeds both horizons, so the correlation between them is carried
    rather than assumed away. Returns ``{"ratio", "lo", "hi", "mde", "ap_num", "ap_den",
    "numerator_h", "denominator_h", "n_origins"}``. Deterministic given ``seed``.

    Raises:
        ValueError: fewer than 3 origins; a non-finite AP at either horizon; or a denominator AP of
            zero, where the ratio is undefined rather than large.
    """
    horizons = (denominator_h, numerator_h)
    g, support, origins, by_origin, tmap, has_gate = _load_indexed(
        pred_dir=pred_dir,
        truth_parquet=truth_parquet,
        target=target,
        horizons=horizons,
        lr_tmpl=lr_tmpl,
        by_tmpl=by_tmpl,
        who="ap_ratio_origin_block_ci",
    )
    ap_num_fn = _ap_fn(g, tmap, has_gate, numerator_h)
    ap_den_fn = _ap_fn(g, tmap, has_gate, denominator_h)

    ap_num, ap_den = ap_num_fn(support), ap_den_fn(support)
    if not np.isfinite(ap_num):
        raise ValueError(
            f"ap_ratio_origin_block_ci: numerator AP is not finite at h={numerator_h} — the label "
            "vector is degenerate"
        )
    # One check, not two: a degenerate label vector gives nan and a skill-less control cannot give
    # exactly zero (AP with any positive label is positive), so both failures mean the same thing —
    # the ratio is undefined, not large, and returning an infinity would be worse than refusing.
    if not (np.isfinite(ap_den) and ap_den > 0.0):
        raise ValueError(
            f"ap_ratio_origin_block_ci: denominator AP at h={denominator_h} is {ap_den} — the "
            "ratio is undefined, not large. Refusing rather than returning an infinity."
        )

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(origins, size=len(origins), replace=True)
        cells = [c for o in pick for c in by_origin[o]]
        den = ap_den_fn(cells)
        vals[b] = ap_num_fn(cells) / den if den > 0.0 else np.nan

    finite = vals[np.isfinite(vals)]
    if finite.size < n_boot // 2:
        raise ValueError(
            f"ap_ratio_origin_block_ci: only {finite.size}/{n_boot} replicates had a usable "
            "denominator — the interval would be built on a biased subsample"
        )
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    lo, hi = float(np.percentile(finite, lo_q)), float(np.percentile(finite, hi_q))
    return {
        "ratio": ap_num / ap_den,
        "lo": lo,
        "hi": hi,
        "mde": (hi - lo) / 2.0,
        "ap_num": ap_num,
        "ap_den": ap_den,
        "numerator_h": float(numerator_h),
        "denominator_h": float(denominator_h),
        "n_origins": float(len(origins)),
    }


def ap_diff_origin_block_ci(
    *,
    pred_dir_a: str,
    pred_dir_b: str,
    truth_parquet: str,
    target: str,
    h: int,
    lr_tmpl: str = "lr_{t}_best",
    by_tmpl: str | None = "by_{t}_best",
    n_boot: int = 400,
    seed: int = 0,
    ci: float = 0.90,
) -> dict[str, float]:
    """Paired origin-block bootstrap CI on ``AP_a(h) - AP_b(h)`` for two arms on ONE support.

    The companion to :func:`ap_ratio_origin_block_ci`, pairing across **arms** instead of across
    **horizons**. One origin draw per replicate scores *both* arms on the same resampled cell set,
    so the correlation between them is carried rather than assumed away. That correlation is the
    whole point here: two arms of the same model on the same origins move together, and an
    unpaired interval built from two independent bootstraps throws that away, reporting a width
    dominated by variance the comparison does not actually face.

    This is the construction #281 argues for. On the state-freeze arms the unpaired MDE (0.0541,
    inherited from the SS sweep's between-seed design) exceeds the measured effect (~0.036), so the
    unpaired reading is "unresolvable" for an effect that is identical in sign and close in size on
    every seed.

    Both arms MUST share a support, and it is **compared and refused** rather than silently
    intersected: ``_load_indexed``
    derives the support per directory, and a mismatch raises instead of scoring different cell sets
    against each other.

    Returns ``{"diff", "lo", "hi", "mde", "ap_a", "ap_b", "h", "n_origins", "n_support"}``.
    Deterministic given ``seed``.

    Raises:
        ValueError: fewer than 3 origins; a non-finite AP in either arm; or supports that differ
            between the two prediction directories.
    """
    ga, sup_a, org_a, byo_a, tmap, gate_a = _load_indexed(
        pred_dir=pred_dir_a,
        truth_parquet=truth_parquet,
        target=target,
        horizons=(h,),
        lr_tmpl=lr_tmpl,
        by_tmpl=by_tmpl,
        who="ap_diff_origin_block_ci",
    )
    gb, sup_b, org_b, _byo_b, _tmap_b, gate_b = _load_indexed(
        pred_dir=pred_dir_b,
        truth_parquet=truth_parquet,
        target=target,
        horizons=(h,),
        lr_tmpl=lr_tmpl,
        by_tmpl=by_tmpl,
        who="ap_diff_origin_block_ci",
    )
    if set(sup_a) != set(sup_b):
        raise ValueError(
            f"ap_diff_origin_block_ci: the two arms do not share a support "
            f"({len(sup_a)} vs {len(sup_b)} cells) — a paired draw would score different cell sets"
        )
    if sorted(org_a) != sorted(org_b):
        raise ValueError("ap_diff_origin_block_ci: the two arms do not share an origin set")
    if gate_a != gate_b:
        # `_ap_fn` ranks on the gate probability when a gate cube exists and on `(cs > 0).mean(1)`
        # when it does not. Pairing a gated arm against an ungated one therefore differences TWO
        # DIFFERENT STATISTICS and reports it as an arm effect, with no error — the same class as
        # the S=1 binary-vs-continuous mismatch that understated persistence (#282, C-293).
        raise ValueError(
            f"ap_diff_origin_block_ci: arm A has_gate={gate_a} but arm B has_gate={gate_b}. "
            "The two arms would be ranked on different statistics and the difference would not be "
            "an arm effect."
        )

    ap_a_fn = _ap_fn(ga, tmap, gate_a, h)
    ap_b_fn = _ap_fn(gb, tmap, gate_b, h)
    point_a, point_b = ap_a_fn(sup_a), ap_b_fn(sup_a)
    for name, v in (("a", point_a), ("b", point_b)):
        if not np.isfinite(v):
            raise ValueError(f"ap_diff_origin_block_ci: AP is not finite for arm {name} at h={h}")

    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.choice(org_a, size=len(org_a), replace=True)
        cells = [c for o in pick for c in byo_a[o]]  # ONE draw, scored by both arms
        vals[i] = ap_a_fn(cells) - ap_b_fn(cells)
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    lo, hi = float(np.percentile(vals, lo_q)), float(np.percentile(vals, hi_q))
    return {
        "diff": point_a - point_b,
        "lo": lo,
        "hi": hi,
        "mde": (hi - lo) / 2.0,
        "ap_a": point_a,
        "ap_b": point_b,
        "h": float(h),
        "n_origins": float(len(org_a)),
        "n_support": float(len(sup_a)),
    }


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
    ap.add_argument(
        "--ratio",
        action="store_true",
        help="bootstrap retention AP(h_num)/AP(h_den) with one paired origin draw per replicate, "
        "instead of a per-horizon CI",
    )
    ap.add_argument("--num-h", type=int, default=18, help="--ratio numerator horizon")
    ap.add_argument("--den-h", type=int, default=1, help="--ratio denominator horizon")
    a = ap.parse_args()

    truth = a.truth
    if truth is None:
        sys.path.insert(0, str(_HN / "reports/2026-07-28_datafactory_migration_dossier/tools"))
        from v2_ruler import V2_TRUTH

        truth = str(V2_TRUTH)

    if a.ratio:
        res = ap_ratio_origin_block_ci(
            pred_dir=a.pred_dir,
            truth_parquet=truth,
            target=a.target,
            numerator_h=a.num_h,
            denominator_h=a.den_h,
            n_boot=a.n_boot,
            seed=a.seed,
        )
    else:
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
