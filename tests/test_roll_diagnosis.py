"""The roll-diagnosis instrument, validated on synthetic fields before it touches real data.

It has one job: given two fields, say whether one is the other MOVED, and by how much. If it
cannot recover a shift it was told to find, nothing it says about EXP-3 means anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_D = Path(__file__).resolve().parents[1] / "reports" / "2026-09-02_silence_vs_fade_dossier"
sys.path.insert(0, str(_D / "tools"))

rd = pytest.importorskip("roll_diagnosis")


def _smooth_field(seed=0, n=64):
    """A sparse, spatially-clustered field — the shape a conflict forecast actually has."""
    rng = np.random.default_rng(seed)
    f = np.zeros((n, n))
    for _ in range(12):
        y, x = rng.integers(0, n, 2)
        yy, xx = np.ogrid[:n, :n]
        d2 = ((yy - y) % n) ** 2 + ((xx - x) % n) ** 2
        f += rng.gamma(2, 3) * np.exp(-d2 / 18.0)
    return f


@pytest.mark.parametrize("shift", [(7, 13), (90 % 64, 90 % 64), (1, 0), (0, 31)])
def test_recovers_a_known_shift_exactly(shift):
    """The load-bearing property. If this fails, every EXP-3 conclusion drawn from it is void."""
    a = _smooth_field()
    b = np.roll(a, shift, axis=(0, 1))
    offset, peak, _ = rd.circular_xcorr_peak(a, b)
    assert offset == shift, f"expected {shift}, got {offset}"
    assert peak == pytest.approx(1.0, abs=1e-9), "a pure translation must correlate perfectly"


def test_identical_fields_peak_at_zero():
    a = _smooth_field(1)
    offset, peak, r0 = rd.circular_xcorr_peak(a, a)
    assert offset == (0, 0)
    assert peak == pytest.approx(1.0, abs=1e-9)
    assert r0 == pytest.approx(1.0, abs=1e-9)


def test_unrelated_fields_have_no_strong_peak():
    """'Broken' must be distinguishable from 'moved' — an unrelated field must not fake a match."""
    _, peak, _ = rd.circular_xcorr_peak(_smooth_field(2), _smooth_field(99))
    assert peak < 0.5, (
        f"unrelated fields correlated at {peak:.3f}; the test cannot tell them apart"
    )


def test_a_shifted_field_still_matches_under_noise():
    """Real fields are not pure translations; the peak must survive perturbation."""
    a = _smooth_field(3)
    rng = np.random.default_rng(5)
    b = np.roll(a, (11, 5), axis=(0, 1)) + rng.normal(0, a.std() * 0.25, a.shape)
    offset, peak, _ = rd.circular_xcorr_peak(a, b)
    assert offset == (11, 5)
    assert peak > 0.7


def test_constant_field_is_refused():
    """A constant field has no structure to align, so an argmax would be an arbitrary noise pick."""
    with pytest.raises(ValueError, match="constant"):
        rd.circular_xcorr_peak(np.ones((8, 8)), _smooth_field(4, 8))


def test_mismatched_shapes_are_refused():
    with pytest.raises(ValueError, match="equal shape"):
        rd.circular_xcorr_peak(np.zeros((8, 8)), np.zeros((8, 9)))
