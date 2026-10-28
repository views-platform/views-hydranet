"""The binned empirical-CDF scorer, as a module with NO side effects, so three scripts can share
one implementation instead of three copies.

⛔ PROVENANCE, and it is the point. The body of `binned_cdf_crps` is lifted from the sweep loop of
the committed `reports/2026-10-27_family_free_head/family_free_cdf.py`. It is the implementation
that produced the published `0.039186` and the whole bin ladder. It is NOT re-derived, NOT
tidied and NOT improved — a lifted scorer that has been "cleaned up" is a different scorer, and
the whole value of lifting it is that it is the same one.

⚠️ `l3a_split_sensitivity.py` carries its own INLINE copy of this function. That duplication is
deliberate: L3a is committed with its `.out`, and editing it to import this module would require
re-running it to prove the output unchanged. ⇒ `l3c_three_target_scores.py` gates the two against
each other by comparing their parsed ASTs (**G-S1**), so a divergence cannot go unnoticed even
though the duplication is allowed to stand.

⛔ This module computes a CRPS and nothing else. It does not know about targets, splits, cubes or
comparators, and it must not learn: every caller declares its own population.
"""

from __future__ import annotations

import numpy as np


def pair_sum_sorted(v: np.ndarray) -> float:
    """sum_ij |v_i - v_j| for an ALREADY-SORTED 1-D array, in O(n).

    The identity is 2 * sum_k v_k * (2k - (n-1)). `self_test` checks it against brute force on
    foreign input, because an identity asserted is not an identity measured.
    """
    n = v.size
    w = 2.0 * np.arange(n) - (n - 1)
    return float(2.0 * (v * w).sum())


def binned_cdf_crps(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_sco: np.ndarray,
    y_sco: np.ndarray,
    B: int,
) -> tuple[float, float, float, float]:
    """Fit an equal-count binned empirical CDF on (x_fit, y_fit); score (x_sco, y_sco).

    Returns (plug-in mean CRPS, fair mean CRPS, mean fit members per occupied bin,
             fraction of score cell-months that landed in an EMPTY fit bin and scored nan).
    """
    order_fit = np.argsort(x_fit, kind="stable")
    xf_s, yf_s = x_fit[order_fit], y_fit[order_fit]

    n_fit = xf_s.size
    edges_at = (np.arange(1, B) * n_fit) // B
    edges = xf_s[edges_at]                                   # B-1 interior edges
    bin_fit = np.searchsorted(edges, xf_s, side="right")      # already sorted by x
    bin_sco = np.searchsorted(edges, x_sco, side="right")     # clipped by construction

    o2 = np.lexsort((yf_s, bin_fit))
    yb, bb = yf_s[o2], bin_fit[o2]
    starts = np.searchsorted(bb, np.arange(B), side="left")
    stops = np.searchsorted(bb, np.arange(B), side="right")
    csum = np.concatenate(([0.0], np.cumsum(yb)))

    so = np.argsort(bin_sco, kind="stable")
    sb_, sy = bin_sco[so], y_sco[so]
    s_start = np.searchsorted(sb_, np.arange(B), side="left")
    s_stop = np.searchsorted(sb_, np.arange(B), side="right")

    crps_p = np.empty(y_sco.size)
    crps_f = np.empty(y_sco.size)
    occ: list[int] = []
    n_void = 0
    for b in range(B):
        lo, hi = starts[b], stops[b]
        qlo, qhi = s_start[b], s_stop[b]
        if qhi <= qlo:
            continue
        n = hi - lo
        if n == 0:                      # a bin with no fit members cannot forecast
            crps_p[qlo:qhi] = np.nan
            crps_f[qlo:qhi] = np.nan
            n_void += qhi - qlo
            continue
        occ.append(n)
        v = yb[lo:hi]
        tot = csum[hi] - csum[lo]
        A = pair_sum_sorted(v)
        yq = sy[qlo:qhi]
        k = np.searchsorted(v, yq, side="right")
        below = csum[lo + k] - csum[lo]
        e1 = (yq * k - below + (tot - below) - yq * (n - k)) / n
        crps_p[qlo:qhi] = e1 - A / (2.0 * n * n)
        crps_f[qlo:qhi] = e1 - (A / (2.0 * n * (n - 1)) if n > 1 else 0.0)
    inv = np.empty_like(so)
    inv[so] = np.arange(so.size)
    cp, cf = crps_p[inv], crps_f[inv]
    return (
        float(np.nanmean(cp)),
        float(np.nanmean(cf)),
        float(np.mean(occ)) if occ else float("nan"),
        n_void / y_sco.size,
    )


def self_test() -> None:
    """The sorted-pairwise identity against brute force, on foreign input. Raises on failure."""
    rg = np.random.default_rng(11)
    v = np.sort(rg.gamma(0.4, 15.0, size=257))
    brute = float(np.abs(v[:, None] - v[None, :]).sum())
    if abs(pair_sum_sorted(v) - brute) >= 1e-6:
        raise AssertionError(f"identity failed: {pair_sum_sorted(v)} vs {brute}")
    if abs(pair_sum_sorted(np.array([0.0, 2.0])) - 4.0) >= 1e-12:
        raise AssertionError("identity failed on the m=2 hand case {0,2} -> 4")
