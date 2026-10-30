"""The per-CELL binned empirical-CDF scorer that RP-3 L2 needs, with the per-bin facts.

⛔ WHY THIS EXISTS SEPARATELY FROM `binned_cdf.py`. That module is the lifted, frozen
implementation that produced the published `0.039186`, and its two functions' ASTs are gated
against `l3a_split_sensitivity.py`'s inline copy by **G-S1**. It returns four SCALARS. L2's whole
question is WHICH CELLS carry the cost, which needs the per-cell CRPS vector, each cell's bin id,
and each bin's occupancy / positive count / mean. ⇒ A second implementation, kept here so
`l2_background_split.py` and `l2b_pin_the_background.py` share ONE copy rather than two.

⛔ IT IS GATED, NOT TRUSTED. Both callers assert this scorer's mean against
`binned_cdf.binned_cdf_crps` on identical input to **1e-12**, and `l2_background_split.py`'s
G-L2a additionally pins all five of the committed `family_free_cdf.out` numbers at B=8,000. A
divergence cannot survive either script starting up.

⚠️ `score_cells` also carries the `sub_m` subsample path, which `binned_cdf_crps` has no
equivalent of. That path is therefore gated only by its own derivation (G-L2e), not by the frozen
module.
"""

from __future__ import annotations

import numpy as np


def equal_count_edges(xf_sorted: np.ndarray, B: int) -> np.ndarray:
    """The committed rule: the k-th of B bins holds fit ranks [k*N/B, (k+1)*N/B)."""
    return xf_sorted[(np.arange(1, B) * xf_sorted.size) // B]


def equal_width_edges(xf_sorted: np.ndarray, B: int, log: bool) -> np.ndarray:
    """Candidate 2. `log` takes equal width in log10(index) instead of in the index itself."""
    lo, hi = float(xf_sorted[0]), float(xf_sorted[-1])
    if log:
        return 10.0 ** np.linspace(np.log10(max(lo, 1e-300)), np.log10(hi), B + 1)[1:-1]
    return np.linspace(lo, hi, B + 1)[1:-1]


def score_cells(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_sco: np.ndarray,
    y_sco: np.ndarray,
    edges: np.ndarray,
    sub_m: int | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Per-CELL version of the committed scorer, plus the per-bin facts H-L2 needs.

    ⛔ This is a deliberate second implementation of `binned_cdf.binned_cdf_crps`, not an import,
    because that function returns four scalars and H-L2 is a statement about WHICH CELLS carry the
    cost. G-L2a gates the two against each other on identical input, so the duplication cannot
    drift silently — the same arrangement L3a/L3c use, and for the same reason.

    `sub_m` is candidate 3: score each bin with a random size-`sub_m` subsample of its fit members
    (all of them when the bin holds fewer), instead of with the whole bin.
    """
    B = edges.size + 1
    order_fit = np.argsort(x_fit, kind="stable")
    xf_s, yf_s = x_fit[order_fit], y_fit[order_fit]

    bin_fit = np.searchsorted(edges, xf_s, side="right")
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
    bin_n = stops - starts
    bin_npos = np.zeros(B, np.int64)
    bin_mean = np.full(B, np.nan)
    rg = np.random.default_rng(seed)
    occ: list[int] = []
    for b in range(B):
        lo, hi = starts[b], stops[b]
        n_all = hi - lo
        if n_all > 0:
            v_all = yb[lo:hi]
            bin_npos[b] = int(np.count_nonzero(v_all))
            bin_mean[b] = float(csum[hi] - csum[lo]) / n_all
        qlo, qhi = s_start[b], s_stop[b]
        if qhi <= qlo:
            continue
        if n_all == 0:                  # a bin with no fit member cannot forecast
            crps_p[qlo:qhi] = np.nan
            crps_f[qlo:qhi] = np.nan
            continue
        if sub_m is not None and n_all > sub_m:
            v = np.sort(rg.choice(v_all, size=sub_m, replace=False))
            n = sub_m
            cs = np.concatenate(([0.0], np.cumsum(v)))
            tot = float(cs[-1])
        else:
            v = v_all
            n = n_all
            cs = None
            tot = csum[hi] - csum[lo]
        occ.append(n)
        w = 2.0 * np.arange(n) - (n - 1)
        A = float(2.0 * (v * w).sum())
        yq = sy[qlo:qhi]
        k = np.searchsorted(v, yq, side="right")
        below = (cs[k] if cs is not None else csum[lo + k] - csum[lo])
        e1 = (yq * k - below + (tot - below) - yq * (n - k)) / n
        crps_p[qlo:qhi] = e1 - A / (2.0 * n * n)
        crps_f[qlo:qhi] = e1 - (A / (2.0 * n * (n - 1)) if n > 1 else 0.0)

    inv = np.empty_like(so)
    inv[so] = np.arange(so.size)
    return {
        "plug": crps_p[inv],
        "fair": crps_f[inv],
        "bin": bin_sco,
        "bin_n": bin_n,
        "bin_npos": bin_npos,
        "bin_mean": bin_mean,
        "occ": float(np.mean(occ)) if occ else float("nan"),
    }
