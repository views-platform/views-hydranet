"""Guards for S11's two hand-written primitives: the ensemble CDF, and the composition transcribed.

Epic #579, story S11. ⛔ Both halves of each guard are present: the correct form must PASS **and**
the defect must FAIL. A guard that only accepts is vacuous -- the discipline of
``test_rank_estimators.py`` and ``test_arm_config.py``.

⛔⛔ WHY ``ecdf_cube`` IS GUARDED AT ALL. Its first version returned the **survival** function
``1 - F`` instead of ``F`` -- a reversed ``cumsum``. ⚠️ It passed EVERY structural gate in the
instrument: ``G-S1`` (the ``dCRPS == MCB_B - MCB_A`` identity), ``G-S4`` and ``G-S5`` all read
``0.000e+00`` on it, because a wrong ``F`` fed both arms equally and the identity cancels it. The
only thing that caught it was that ``CRPS_A`` came back **673** against a climatological ``UNC``
of **3.561**. ⇒ An identity gate cannot catch a shared transform. This file can.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_TOOLS = _REPO / "reports/2026-11-10_rank_coupling_verdict/tools"

pytestmark = pytest.mark.skipif(
    not (_TOOLS / "tau_sweep_decomposition.py").exists(),
    reason="the rank-coupling-verdict dossier tools are not present in this tree",
)

if (_TOOLS / "tau_sweep_decomposition.py").exists():
    for rel in (
        "reports/2026-11-10_rank_coupling_verdict/tools",
        "reports/2026-11-04_delivered_gate_lookup/tools",
        "reports/2026-11-04_rank_coupled_body/tools",
        "reports/2026-11-08_ordering_discrimination/tools",
    ):
        sys.path.insert(0, str(_REPO / rel))


@pytest.fixture
def tsd():
    import tau_sweep_decomposition

    return tau_sweep_decomposition


def _case():
    """-> (cube, yt). ⭐ Integer counts with heavy ties and a value ABOVE the grid's top."""
    rng = np.random.default_rng(11)
    cube = rng.integers(0, 40, (37, 16)).astype(np.float64)
    cube[0, :] = 0.0  # a cell that predicts a point mass at zero
    cube[1, :] = 999.0  # ⭐ entirely above yt[-1]: F must be 0 on every gap
    yt = np.unique(np.concatenate([[0.0], rng.integers(1, 40, 25).astype(np.float64)]))
    return cube, yt


def _brute_force_cdf(cube: np.ndarray, yt: np.ndarray) -> np.ndarray:
    """F[i, j] = mean_s 1{cube[i, s] <= yt[j]}, straight from the definition, for j = 0..m-2."""
    return (cube[:, None, :] <= yt[None, :-1, None]).mean(axis=2)


def _reversed_cumsum_cdf(cube: np.ndarray, yt: np.ndarray) -> np.ndarray:
    """⛔ THE DEFECT, verbatim: ``cumsum`` over the reversed axis, which gives ``1 - F``."""
    n, s = cube.shape
    m = yt.size
    k = np.searchsorted(yt, cube, side="right")
    flat = (np.arange(n)[:, None] * (m + 1) + k).ravel()
    cnt = np.bincount(flat, minlength=n * (m + 1)).reshape(n, m + 1)
    cum = np.cumsum(cnt[:, ::-1], axis=1)[:, ::-1]
    return cum[:, 1:m] / float(s)


def test_ecdf_cube_matches_the_definition(tsd):
    """⭐ The correct form, asserted to PASS, against the brute-force definition itself."""
    cube, yt = _case()
    assert np.allclose(tsd.ecdf_cube(cube, yt), _brute_force_cdf(cube, yt), atol=0.0)


def test_ecdf_cube_is_a_cdf(tsd):
    """⭐ Properties a CDF must have and a survival function must not: non-decreasing, and 1 where
    the whole ensemble is at zero."""
    cube, yt = _case()
    f = tsd.ecdf_cube(cube, yt)
    assert np.all(np.diff(f, axis=1) >= -1e-15), "a CDF cannot decrease in the threshold"
    assert np.all(f[0] == 1.0), "a point mass at 0 must give F == 1 on every gap at or above 0"
    assert np.all(f[1] == 0.0), "an ensemble entirely above the grid must give F == 0"


def test_reversed_cumsum_is_refused(tsd):
    """⛔ The discriminating half: the defect must FAIL the check above, else it is vacuous."""
    cube, yt = _case()
    bad = _reversed_cumsum_cdf(cube, yt)
    assert not np.allclose(bad, _brute_force_cdf(cube, yt), atol=1e-9), (
        "the reversed-cumsum form agreed with the definition on this input, so this test does not "
        "reproduce the defect and the guard above proves nothing"
    )
    # ⭐ And it fails on the clearest cell there is: a point mass at zero, whose F is 1 everywhere.
    assert not np.allclose(bad[0], 1.0), "the defect must get the point-mass-at-zero cell wrong"


def test_compose_mean_matches_the_production_module(tsd):
    """⭐⭐ The transcription, checked against ``ADR-069``'s own torch implementation.

    ⛔ ``compose_mean_np`` exists only because this epic touches no production code. That is a
    reason to re-implement two lines, not a licence for them to differ.
    """
    torch = pytest.importorskip("torch")
    from views_hydranet.distributions import composition

    rng = np.random.default_rng(7)
    gate = rng.random(500)
    mean = rng.gamma(2.0, 5.0, 500)
    for comp, tau in (("soft_gate", None), ("threshold_gate", 0.2), ("threshold_gate", 0.26)):
        got = tsd.compose_mean_np(mean, gate, comp, tau)
        want = composition.compose_mean(
            torch.from_numpy(mean), torch.from_numpy(gate), comp, tau
        ).numpy()
        assert np.allclose(got, want, atol=0.0), f"{comp} tau={tau} disagrees with ADR-069"


def test_compose_cube_threshold_is_exact_and_soft_is_bernoulli(tsd):
    """⭐ ``threshold_gate`` zeroes whole cells; ``soft_gate`` masks PER DRAW, not per cell."""
    rng = np.random.default_rng(3)
    cube = np.full((400, 16), 5.0)
    gate = np.linspace(0.0, 1.0, 400)

    hard = tsd.compose_cube(cube, gate, "threshold_gate", 0.5, rng)
    kept = (hard > 0).all(axis=1)
    assert np.array_equal(kept, gate >= 0.5), "threshold_gate must keep exactly gate >= tau"
    assert np.all((hard > 0).any(axis=1) == kept), "threshold_gate must not mask within a cell"

    soft = tsd.compose_cube(cube, gate, "soft_gate", None, rng)
    mid = (gate > 0.3) & (gate < 0.7)
    survive = (soft[mid] > 0).mean(axis=1)
    assert np.any((survive > 0.0) & (survive < 1.0)), (
        "soft_gate must mask PER DRAW — a per-cell mask would make every row all-or-nothing"
    )
    assert soft[gate == 0.0].sum() == 0.0 and (soft[-1] == 5.0).all()


def test_firing_set_is_the_open_set_under_threshold_and_everything_under_soft(tsd):
    """⭐ The asymmetry is deliberate: a property of the composition, not a free parameter."""
    gate = np.array([0.01, 0.1, 0.2, 0.5, 0.9])
    assert tsd.firing_set(gate, "threshold_gate", 0.2).tolist() == [0, 0, 1, 1, 1]
    assert tsd.firing_set(gate, "soft_gate", None).all()


def test_couple_random_ties_breaks_ties_without_using_cell_order(tsd):
    """⛔⛔ The tie block is the point: A's body draws zero ~57% of the time, so a large share of
    open cells share ``x == 0`` exactly. Cell index on this grid is SPATIAL, so a stable sort would
    lay the bottom of the empirical pool out along a spatial sweep and call it a forecast.
    """
    pool = np.arange(1.0, 101.0)
    key = np.zeros(60)  # ⭐ EVERY cell tied: the key carries no information at all
    open_idx = np.arange(60)

    # ⚠️ The allocation is read as MONOTONICITY IN CELL INDEX, not as ``argsort`` of the assigned
    # values: the pool is drawn with replacement, so those values are themselves tied and
    # ``argsort`` on tied data is the very defect C-289 names.
    a = tsd.couple_random_ties(pool, open_idx, key, 60, 4, np.random.default_rng(1))
    assert not np.all(np.diff(a[:, 0]) >= 0), (
        "the magnitudes still rise with CELL INDEX on a fully tied key, so the tie-break is not "
        "random and the allocation is spatial"
    )

    # ⛔ The discriminating half: the committed stable-sort form must lay them out in cell order.
    from rank_coupling_verdict import couple

    for seed in (1, 2, 3):
        c = couple(pool, open_idx, key, 60, 4, np.random.default_rng(seed))
        assert np.all(np.diff(c[:, 0]) >= 0), (
            f"seed {seed}: the stable-sort form did not allocate in cell order, so this test does "
            "not reproduce the cell-order dependence it exists to contrast with"
        )

    # ⭐ And an INFORMATIVE key must still be respected exactly — the jitter is a tie-break only.
    key2 = np.arange(60.0)
    e = tsd.couple_random_ties(pool, open_idx, key2, 60, 1, np.random.default_rng(5))
    assert np.all(np.diff(e[:, 0]) >= 0), "a strictly rising key must give rising magnitudes"
