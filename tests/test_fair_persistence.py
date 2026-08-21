"""The fair-persistence repair: rank persistence by VALUE, not by the zero/non-zero indicator.

`score_v2_horizons` forms the AP score as `p = gate if has_gate else (cs > 0).mean(1)`.
`_persistence_gathered` supplies no gate at S=1, so persistence is scored from a two-level
signal while gated arms are scored from a continuous one — the Epic #263 matched-reference
defect. These tests pin the repair and, more importantly, pin the DIRECTION of the bias, since
that is what licenses the claim "our margin is an upper bound".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_D = Path(__file__).resolve().parents[1] / "reports" / "2026-08-21_persistence_reference_dossier"
sys.path.insert(0, str(_D / "tools"))

fp = pytest.importorskip("fair_persistence")


def _write_ids(tmp_path, origins, units, horizons):
    d = tmp_path / "ids"
    d.mkdir()
    for m0 in origins:
        t = np.array([m0 + h - 1 for h in horizons for _ in units])
        u = np.array([u for _ in horizons for u in units])
        np.savez(d / f"origin_{m0}.npz", time=t, unit=u)
    return str(d)


def test_support_requires_every_horizon(tmp_path):
    """A unit present at only some horizons is not in the support — the `_support_keys` rule."""
    d = tmp_path / "ids"
    d.mkdir()
    # unit 1 at h=1,2; unit 2 at h=1 only
    np.savez(d / "origin_100.npz", time=np.array([100, 101, 100]), unit=np.array([1, 1, 2]))
    got = fp.support_from_identifiers(str(d), (1, 2))
    assert got == {(100, 1)}


def test_missing_identifier_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        fp.support_from_identifiers(str(tmp_path / "nope"), (1,))


def test_origin_count_mismatch_raises(tmp_path):
    """Two files that report the SAME origin means a collapsed copy — the exact bug that ate
    the first run's support. It must raise, not silently score a third of the data."""
    d = tmp_path / "ids"
    d.mkdir()
    for name in ("a.npz", "b.npz"):
        np.savez(d / name, time=np.array([100, 101]), unit=np.array([1, 1]))
    with pytest.raises(ValueError, match="distinct origins"):
        fp.support_from_identifiers(str(d), (1, 2))


def test_value_ranking_beats_binary_when_magnitude_is_informative():
    """THE point of the repair, and the direction the whole claim rests on.

    Truth at h: units with a big history are events, units with a small history are not. The
    binary score cannot tell those apart — both are "non-zero" — so it ties them. The value
    score ranks them correctly.
    """
    support = [(100, u) for u in range(1, 5)]
    # history: unit1=50, unit2=40 (big) ; unit3=1, unit4=1 (small)
    # outcome: the big-history units are the events
    tmap = {(99, 1): 50.0, (99, 2): 40.0, (99, 3): 1.0, (99, 4): 1.0,
            (100, 1): 7.0, (100, 2): 3.0, (100, 3): 0.0, (100, 4): 0.0}
    y, val, binr = fp.persistence_scores(tmap, support, h=1)
    assert list(y) == [1.0, 1.0, 0.0, 0.0]
    ap_val = fp.average_precision(y, val)
    ap_bin = fp.average_precision(y, binr)
    assert ap_val == pytest.approx(1.0), "perfect ordering is available from the values"
    assert ap_bin < ap_val, "the indicator ties all four and cannot reach it"


def test_binary_never_beats_value_on_a_random_sweep():
    """The bias has a DIRECTION: collapsing a score to its indicator cannot help.

    If this ever failed, "our 3x margin is an upper bound" would be unsupported — so it is
    asserted over many random draws rather than one hand-made case.
    """
    rng = np.random.default_rng(0)
    losses = 0
    for _ in range(200):
        n = 60
        hist = rng.gamma(0.4, 8.0, n) * (rng.random(n) > 0.55)
        # events driven by history magnitude plus noise -> value is informative but not perfect
        y = ((hist + rng.normal(0, 4, n)) > 6).astype(float)
        if y.sum() in (0, n):
            continue
        av = fp.average_precision(y, hist)
        ab = fp.average_precision(y, (hist > 0).astype(float))
        if ab > av + 1e-12:
            losses += 1
    assert losses == 0, f"the indicator beat the value score {losses}/200 times"


def test_persistence_forecast_is_h_invariant_but_truth_is_not():
    """Persistence feeds truth[m0-1] at EVERY horizon, so only the outcome moves with h."""
    support = [(100, 1), (100, 2)]
    tmap = {(99, 1): 5.0, (99, 2): 0.0, (100, 1): 1.0, (100, 2): 0.0,
            (105, 1): 0.0, (105, 2): 9.0}
    _, v1, _ = fp.persistence_scores(tmap, support, h=1)
    y1, _, _ = fp.persistence_scores(tmap, support, h=1)
    y6, v6, _ = fp.persistence_scores(tmap, support, h=6)
    assert list(v1) == list(v6) == [5.0, 0.0]
    assert list(y1) == [1.0, 0.0]
    assert list(y6) == [0.0, 1.0]


def test_absent_history_is_zero_not_a_crash():
    """Missing months are 0.0 — the convention `_persistence_gathered` uses."""
    support = [(100, 1)]
    y, val, _ = fp.persistence_scores({(100, 1): 3.0}, support, h=1)
    assert val[0] == 0.0 and y[0] == 1.0
