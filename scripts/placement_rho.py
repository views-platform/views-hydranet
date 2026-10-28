#!/usr/bin/env python
"""placement_rho.py — the magnitude-ranking instrument the metric battery does not have.

Built because M2/M3 (#479/#480, epic #477) measured the hole #444 names. Of the twenty columns in
the battery, FOURTEEN are functions of the mean field alone, TWO read only the gate, and the four
that read the ensemble read it pooled. The consequence, measured: a forecast that fires in exactly
the truth-positive cells with a CONSTANT magnitude is given the oracle's exact score by twelve of
eighteen scoring columns (AP 1.0, Brier 0.0, crps_none 0.0, FSS 1.0 at all seven scales, AFSS 1.0,
hit_rate 1.0) while being the worst of eleven on crps_events.

    ==> nothing in the battery sees a magnitude error that PRESERVES PLACEMENT.

This module is that one statistic, and nothing else:

    placement_rho = Spearman rank correlation between the predicted magnitude and the observed
                    magnitude, over a DECLARED population of cells.

Why a rank correlation and not another ratio. The battery already has four ways to compare
magnitude LEVELS (size_ratio, pos_mcr, MCR_field, QS99) and they are all invariant to the ordering
among event cells. Ranking is the axis that is missing, it is scale-free so it cannot be confused
with the known mass deficit, and it is the statistic M67 already used to catch a composition defect
that AP scored 0.736228 on (+0.4925 -> -0.1605).

CONVENTIONS, declared because every one of them is a free parameter and D-4 forbids reporting a
scalar without sweeping it:
  * population  'events'  truth > 0 only (the conditional-magnitude question)
                'union'   truth > 0 OR pred > 0 (adds the false alarms' magnitudes)
                'all'     every cell (dominated by the 99.66% tied at zero)
  * pooling     'pooled'  one correlation over the whole population
                'monthly' per month, then the mean over months
  * ties        average ranks (scipy's default). A CONSTANT input vector carries no rank
                information at all, so this returns 0.0 rather than nan -- documented, not
                incidental: nan would propagate into a verdict, and 0.0 is the honest value
                ("no ranking skill"), which is also what a random forecast tends to.

NOT a replacement for anything. FAO-02 is locked and lodestar_score.py is frozen; this measures
what they are blind to and is quoted beside them, never instead of them.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

POPULATIONS = ("events", "union", "all")
POOLINGS = ("pooled", "monthly")


class PlacementRhoError(ValueError):
    """Raised rather than returning a number that cannot mean what the caller thinks."""


def _rho(pred: np.ndarray, truth: np.ndarray) -> float:
    """Spearman rho on one population, with constant inputs mapped to 0.0 (no rank information)."""
    if pred.size < 2:
        raise PlacementRhoError(f"need at least 2 cells to rank, got {pred.size}")
    if np.ptp(pred) == 0.0 or np.ptp(truth) == 0.0:
        return 0.0
    r = spearmanr(pred, truth).statistic
    if not np.isfinite(r):
        raise PlacementRhoError(
            "spearmanr returned non-finite on non-constant inputs -- refusing to report it"
        )
    return float(r)


def _select(pred: np.ndarray, truth: np.ndarray, population: str) -> np.ndarray:
    if population == "events":
        return truth > 0
    if population == "union":
        return (truth > 0) | (pred > 0)
    if population == "all":
        return np.ones(truth.shape, dtype=bool)
    raise PlacementRhoError(f"population must be one of {POPULATIONS}, got {population!r}")


def placement_rho(
    pred: np.ndarray,
    truth: np.ndarray,
    population: str = "events",
    pooling: str = "pooled",
    month: np.ndarray | None = None,
) -> float:
    """Spearman rho between predicted and observed magnitude over a declared population.

    ``pred`` and ``truth`` are flat per-cell arrays of the same length (a MEAN field, not an
    ensemble -- the ranking question is about the point prediction's ordering). ``month`` is
    required for ``pooling='monthly'`` and is the month id per cell.

    Returns a float in [-1, 1]. Raises rather than returning nan.
    """
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    if pred.shape != truth.shape or pred.ndim != 1:
        raise PlacementRhoError(f"need matching 1-D arrays, got {pred.shape} and {truth.shape}")
    if not (np.isfinite(pred).all() and np.isfinite(truth).all()):
        raise PlacementRhoError("non-finite input -- fail loud rather than rank a nan")
    if pooling not in POOLINGS:
        raise PlacementRhoError(f"pooling must be one of {POOLINGS}, got {pooling!r}")

    if pooling == "pooled":
        sel = _select(pred, truth, population)
        if sel.sum() < 2:
            raise PlacementRhoError(f"population {population!r} selected {int(sel.sum())} cells")
        return _rho(pred[sel], truth[sel])

    if month is None:
        raise PlacementRhoError("pooling='monthly' needs a month id per cell")
    month = np.asarray(month)
    if month.shape != truth.shape:
        raise PlacementRhoError(f"month shape {month.shape} != truth shape {truth.shape}")
    per = []
    for m in np.unique(month):
        inm = month == m
        sel = inm & _select(pred, truth, population)
        if sel.sum() < 2:
            raise PlacementRhoError(f"month {m}: population {population!r} selected "
                                    f"{int(sel.sum())} cells")
        per.append(_rho(pred[sel], truth[sel]))
    return float(np.mean(per))


def sweep(
    pred: np.ndarray,
    truth: np.ndarray,
    month: np.ndarray | None = None,
) -> dict[tuple[str, str], float]:
    """Every (population, pooling) cell of the free-parameter grid. D-4: the report is a table."""
    out: dict[tuple[str, str], float] = {}
    for pop in POPULATIONS:
        for pool in POOLINGS:
            if pool == "monthly" and month is None:
                continue
            out[(pop, pool)] = placement_rho(pred, truth, pop, pool, month)
    return out
