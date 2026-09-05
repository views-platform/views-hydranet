"""error_profile.py — what does the model's free-running error actually LOOK like?

S1 of Epic #311 (#313). The noise design cannot be copied from
`SanchezGonzalez2020_GraphNetworkSimulators`: their sigma=3e-4 is a fraction of a standard
deviation
of a dense, standardised field, and ours is `log1p(counts)`, unstandardised, overwhelmingly zero.
What transfers is their stated aim — "so the training distribution is closer to the distribution
generated during rollouts" — so the error has to be MEASURED and the noise matched to it.

TRUTH-REFERENCED BY CONSTRUCTION. Every quantity here compares the forecast to truth. **C-319** is
Tier 2 because occurrence, magnitude and alignment are all *internal* statistics: they survived a
roll
that displaced the whole forecast and collapsed AP 48x. An internal statistic cannot characterise
an
error against truth, so none is used.

NUMERIC SPACE, verified rather than assumed (the one thing that would silently invalidate all of
it):
the truth parquet is **count space** (max 113,395 for sb) and `gather_all_horizons` loads
`y_pred.npy`
with **no transform**, returning counts. Both sides are counts. The log1p round trip happens
upstream,
inside inference.

WHAT "FIRED" MEANS. Per cell, `q = P(draw > 0)` over the sample dimension — the model's probability
of
firing there. This is the per-cell analogue of the `act_pred` already in the score CSVs, which is
the
same quantity pooled over cells and draws. A per-cell *hard* version (`any draw fired`) is reported
alongside it, because it is a much weaker statement and the difference between them should be
visible
rather than chosen silently.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

_HN = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, f"{_HN}/scripts")
from input_noise_gate import cv, select_design  # noqa: E402,F401  re-exported for the CLI


@dataclass(frozen=True)
class OriginRates:
    """Truth-referenced error rates for one origin at one horizon."""

    origin: int
    n_cells: int
    n_event: int
    act_true: float
    fn_rate: float  # E[fraction of TRUE events the model silences]
    fp_rate: float  # E[fraction of TRUE zeros the model fires on]
    fn_rate_hard: float  # same, but "silenced" = no draw fired at all
    mag_err_median: float  # median log1p(pred) - log1p(truth) on cells active in both
    n_mag: int


def per_cell(g, support, tmap, h):
    """(origin, truth, q, ey, any_fired) per cell at horizon h. Counts throughout."""
    out = []
    for m0, u in support:
        cs = g[(m0, h, u)][0]
        truth = float(tmap[(m0 + h - 1, u)])
        n = len(cs)
        if n == 0:
            raise ValueError(f"empty sample vector at origin {m0}, unit {u}, h={h}")
        fired = sum(1 for x in cs if x > 0)
        out.append((int(m0), truth, fired / n, float(sum(cs) / n), fired > 0))
    return out


def origin_rates(cells, h) -> list[OriginRates]:
    """Group per-cell records by origin and reduce to truth-referenced rates."""
    by_origin: dict[int, list] = {}
    for rec in cells:
        by_origin.setdefault(rec[0], []).append(rec)

    out = []
    for origin in sorted(by_origin):
        rows = by_origin[origin]
        ev = [r for r in rows if r[1] > 0]
        ze = [r for r in rows if r[1] == 0]
        mags = [
            math.log1p(r[3]) - math.log1p(r[1]) for r in ev if r[3] > 0
        ]  # active in BOTH — a cell the model silenced has no magnitude error, it has an FN
        out.append(
            OriginRates(
                origin=origin,
                n_cells=len(rows),
                n_event=len(ev),
                act_true=len(ev) / len(rows) if rows else float("nan"),
                # A cell the model fires on with probability q contributes (1-q) of a silencing.
                fn_rate=sum(1.0 - r[2] for r in ev) / len(ev) if ev else float("nan"),
                fp_rate=sum(r[2] for r in ze) / len(ze) if ze else float("nan"),
                fn_rate_hard=sum(0.0 if r[4] else 1.0 for r in ev) / len(ev)
                if ev
                else float("nan"),
                mag_err_median=_median(mags) if mags else float("nan"),
                n_mag=len(mags),
            )
        )
    return out


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")
