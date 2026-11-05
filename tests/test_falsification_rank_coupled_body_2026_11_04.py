"""RECORD of the 2026-11-04 falsification audit of the rank-coupled body programme.

Audited claim (Simon): *"the math here is unambiguous and clear"*. Eight probes ran; five
properties survived; four findings were raised. ⭐ **All four have since been repaired in
``reports/2026-11-04_rank_coupled_body_programme.md``** -- the section each one attacked now says
the corrected thing:

  F-1  §3.7 v3's bulk was a kernel estimate on the integers with no boundary correction, so it
       placed mass at and below 0 while the declared body support is {1,2,3,...} -- which is what
       P8 is proved from. REPAIRED: §3.7 now carries the truncate-and-renormalise row and states
       that the GPD route needs no kernel and no mass below 1.
  F-2  §3.4 P2 stated ``E[sum] = n * mu_z`` and ``Var[sum] = n * sigma2_z`` unconditionally, but
       under a generalized Pareto tail the variance is undefined for xi >= 1/2 and the mean for
       xi >= 1. REPAIRED: §3.4's P2 now carries that condition -- mean iff xi < 1, variance iff
       xi < 1/2, with a trimmed second moment required where it fails -- and §3.7 says the
       variance is undefined rather than merely larger. ⚠️ Its heading also reads "Total is
       unbiased, not exact", which was NOT by itself the repair F-2 asked for; an earlier
       version of this docstring cited the heading alone and /review-diff caught it.
  F-3  §3.6 asserted ``d = 46`` where §3.1's ``d_y = 1`` and ``d_h = 16`` give 33, and priced
       storage at 6 GB over all 64,818 x 500 cell-months rather than over ``P_z``, the positives
       only. REPAIRED: §3.6 states d = 33 and **6.02 MB**, ~1000x smaller, which strengthens v2.
  F-4  §3.4 P6 and §2.5 called high-ranked units under-dispersed. Wrong in sign at the top: the
       top-ranked cell receives the maximum of n_t draws, whose sd EXCEEDS the pool's. REPAIRED:
       §3.4 now reports the measured profile (up to **3.92x** the pool's sd at the top, 0.01x in
       the bulk, non-monotone) and §2.5 records that the widening covers only 5 of 263 ranks.

⛔⛔ WHAT THIS FILE IS, AND WHAT IT IS NOT

It is the **arithmetic evidence** for those four repairs, kept so a reader can re-run the numbers
rather than take the document's word. ⛔ **It is NOT a guard on the document.** Every probe
exercises the ORIGINAL specification's arithmetic with numpy only -- no repo code is imported and
nothing reads the markdown -- so repairing the prose cannot make a probe pass, and the four
``xfail(strict=True)`` cases stay xfail permanently by construction.

⚠️ An earlier version of this docstring claimed each test *"flips to XPASS when the corresponding
one-line repair lands in §3"*. That was false: the repairs landed and nothing flipped. A test that
cannot go green is a record, not a guard, and saying otherwise is how a stale finding survives --
so it is said plainly here instead.

⭐ What the ``strict=True`` does still buy: if a numpy or RNG change ever made one of these pass,
the suite goes red and someone must come back and decide whether the recorded arithmetic still
holds. That is a guard on the EVIDENCE, not on the programme.

Surviving properties, recorded because the denominator is the point: A2's order-statistic identity
(exact to Monte Carlo error at n = 3, 5 and 263, and kept below as a live regression test), P1, P3,
P4, the §1.3 and §2.4 arithmetic, and §3.4 P1's citation of ``Gneiting2023_ConditionalCalibration``
-- Theorem 2.3 and the two one-way counterexamples are verbatim in the PDF, theorem number
included.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

POOL_SEED = 7
N_T = 263  # §2.4, measured: 0.004063 * 64_818 expected firings per month


def _gaussian_kde_cdf(x: float, data: np.ndarray, beta: float) -> float:
    """CDF of a Gaussian KDE -- the estimator §3.1 names for ``B_beta``, unmodified."""
    z = (x - data) / (beta * math.sqrt(2.0))
    return float(np.mean(0.5 * (1.0 + np.vectorize(math.erf)(z))))


def _heavy_tailed_integer_pool() -> np.ndarray:
    """A positives-only pool shaped like the record: mass at 1, max near L4's 1,528."""
    rng = np.random.default_rng(POOL_SEED)
    return np.r_[
        np.ones(300, dtype=int),
        rng.integers(1, 10, 500),
        rng.integers(10, 80, 150),
        rng.integers(80, 600, 45),
        np.array([1100, 1300, 1528, 900, 1450]),
    ]


@pytest.mark.xfail(strict=True, reason="F-1, repaired in §3.7: the v3 bulk was unbounded below")
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, 5.0])
def test_f1_v3_bulk_respects_declared_positive_support(beta: float) -> None:
    """§3.7 / §3.1 / P8.

    ``F_z^v3(x) = (1 - p*) * B_beta(x) / B_beta(q*)`` for ``x <= q*``. For the declared support
    ``{1,2,...}`` to hold -- and therefore for P8's proof ("F_z is supported on {1,2,...} by
    construction, so no zero observation enters it") to carry over to v3 -- the bulk must place
    no mass at or below 0. A Gaussian kernel on integer data does, at every bandwidth.

    Repair that flips this test: truncate ``B_beta`` to ``[1, inf)`` and renormalise, or state a
    discretisation. One line in §3.7, and P8 then belongs in its surviving-properties list.
    """
    pool = _heavy_tailed_integer_pool()
    mass_at_or_below_zero = _gaussian_kde_cdf(0.0, pool, beta)
    assert mass_at_or_below_zero == 0.0, (
        f"beta={beta}: bulk places {mass_at_or_below_zero:.4f} of its mass at or below 0, "
        "so v3's body can emit a non-positive magnitude and P8 does not hold for v3"
    )


@pytest.mark.xfail(strict=True, reason="F-2, repaired in §3.4: moments undefined for xi >= 1/2")
@pytest.mark.parametrize("xi", [0.5, 0.6, 0.9, 1.0, 1.2])
def test_f2_p2_moments_exist_under_the_v3_tail(xi: float) -> None:
    """§3.4 P2 against §3.7.

    P2 asserts ``E[sum] = n * mu_z`` and ``Var[sum] = n * sigma2_z`` unconditionally. §3.7
    replaces ``F_z`` with a generalized Pareto tail and says only that "P2's variance grows and
    must be reported". A GPD has finite mean iff ``xi < 1`` and finite variance iff ``xi < 1/2``.
    Conflict-severity tails are routinely reported in the range that violates the second.

    Repair that flips this test: state ``xi < 1/2`` as a condition on P2 under v3, or report a
    trimmed/winsorised second moment and say so. §1.2 celebrates ``xi`` as the quantity Epic #230
    named; the moment condition is the price.
    """
    assert xi < 0.5, (
        f"xi={xi}: GPD variance is infinite, so P2's Var[sum | n_z_t] = n * sigma2_z is "
        "undefined under v3 and the document states no bound on xi"
    )


@pytest.mark.xfail(
    strict=True, reason="F-3, repaired in §3.6: d was not derivable and 6 GB the wrong population"
)
def test_f3_v2_embedding_dimension_and_storage_are_derivable() -> None:
    """§3.6 against §3.1 and §3.2.

    §3.6 states ``d = d_y + 2 d_h`` and then asserts ``d = 46``. §3.1 gives ``d_y = 1`` (point
    head) and ``d_h = 16``, which give 33. Separately, the 6 GB figure is ``64,818 x 500 x 46 x 4``
    bytes -- every cell-month -- while §3.2 defines ``P_z`` over pairs with ``o_jtau >= 1`` only.

    Repair that flips this test: say how many blocks ``h`` and ``c`` have (so ``d`` follows), and
    recompute storage over the positives. The corrected figure is ~21 MB, which strengthens v2
    rather than weakening it.
    """
    d_y, d_h = 1, 16
    assert d_y + 2 * d_h == 46, f"§3.1 gives d = {d_y + 2 * d_h}, §3.6 asserts 46"

    n_units, n_months, positive_rate = 64_818, 500, 0.003448
    pool_pairs = round(n_units * n_months * positive_rate)
    stated_gb = 6.0
    actual_gb = pool_pairs * 46 * 4 / 1e9
    assert actual_gb == pytest.approx(stated_gb, rel=0.25), (
        f"P_z holds {pool_pairs:,} pairs = {actual_gb * 1000:.1f} MB, not {stated_gb} GB"
    )


@pytest.mark.xfail(
    strict=True, reason="F-4, repaired in §3.4: the top-ranked cell is over-dispersed"
)
def test_f4_p6_high_rank_under_dispersion_claim() -> None:
    """§3.4 P6 and §2.5.

    Both say high-ranked units are under-dispersed, softened by source (3). At the measured
    ``n_t = 263`` the top-ranked cell always receives ``u_(n)``, so source (3) supplies it no
    variation at all -- and the sd of the maximum of 263 heavy-tailed draws *exceeds* the pool's.

    This finding runs in the programme's favour, which is why it should be stated rather than
    left as a surprise: v1's per-cell sd falls far below the pool's in the bulk and rises above
    it at the top, which is the non-monotone correction §1.1 says the data requires. P6 currently
    claims a monotone one, in the wrong direction at the end that matters -- M90's single
    under-dispersed bin.

    Repair that flips this test: restate P6 as "sharper than the pool in the bulk, wider at the
    top ranks", and cross-reference §2.4, which already noticed the level half of the effect.
    """
    pool = _heavy_tailed_integer_pool()
    rng = np.random.default_rng(POOL_SEED + 1)
    draws = rng.choice(pool, size=(40_000, N_T), replace=True)
    draws.sort(axis=1)

    sd_pool = pool.std()
    sd_top = draws[:, -1].std()
    sd_median_rank = draws[:, N_T // 2].std()

    assert sd_median_rank < sd_pool, "sanity: mid ranks must be sharper than the pool"
    assert sd_top <= sd_pool, (
        f"top-ranked cell sd {sd_top:.1f} is {sd_top / sd_pool:.2f}x the pool's {sd_pool:.1f}: "
        "P6's 'somewhat under-dispersed' is wrong in sign at the top ranks"
    )


def test_a2_order_statistic_identity_holds() -> None:
    """§3.5 A2 -- SURVIVED, kept as a regression guard on the one load-bearing derivation.

    A2 claims ``(1/n) sum_l P(u_(l) = k) = F_z(k) - F_z(k-)``: averaging over a uniformly random
    rank map returns each unit exactly the pool's marginal. It is the half of the ablation that
    makes the contrast one-variable, and it is non-obvious on a discrete pool with ties.
    """
    pool = np.array([1, 1, 1, 2, 2, 3, 5, 9, 40])
    values, counts = np.unique(pool, return_counts=True)
    pmf = counts / counts.sum()

    rng = np.random.default_rng(POOL_SEED + 2)
    for n in (3, 5, 263):
        draws = rng.choice(pool, size=(200_000, n), replace=True)
        draws.sort(axis=1)
        averaged = np.array(
            [[(draws[:, rank] == value).mean() for value in values] for rank in range(n)]
        ).mean(axis=0)
        assert np.abs(averaged - pmf).max() < 5e-3, f"A2 fails at n={n}"
