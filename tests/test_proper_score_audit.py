"""Unit tests for scripts/proper_score_audit.py pure scoring functions.

The audit exists because the FSS/area-ratio sharpness scorecard may be an IMPROPER
diagnostic (a near-zero-everywhere forecast scores "sharp" on a 99.7%-zero map). These
tests pin the *honest* scores on synthetic cases where the answer is known:

  - SCRPS (Bolin & Wallin 2023): scale-invariant up to the known additive -0.5*ln(c) shift,
    and rewards a calibrated ensemble over a biased one.
  - randomized PIT (count-aware): piles at 1 for a too-low biased forecast; ~uniform when
    the truth is exchangeable with the ensemble (calibration).
  - CRPS sanity: a perfect ensemble scores 0.
"""

import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(__file__)
_SCRIPTS = os.path.abspath(os.path.join(_HERE, "..", "scripts"))
if _SCRIPTS not in sys.path:  # the module does `from mcr_readout import ...`
    sys.path.insert(0, _SCRIPTS)
_MOD = os.path.join(_SCRIPTS, "proper_score_audit.py")
_spec = importlib.util.spec_from_file_location("proper_score_audit", _MOD)
psa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(psa)


def test_scrps_scale_shift():
    """SCRPS(c*x, c*y) == SCRPS(x, y) - 0.5*ln(c)  (local scale invariance, reward form)."""
    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 1.0, size=500)
    x = rng.gamma(2.0, 1.0, size=(500, 32))
    c = 7.3
    base = psa.scrps_ensemble(y, x)
    scaled = psa.scrps_ensemble(c * y, c * x)
    # per-cell relationship must hold (up to numerical noise)
    assert np.allclose(scaled, base - 0.5 * np.log(c), atol=1e-9)


def test_scrps_rewards_calibration():
    """A calibrated ensemble must score higher (reward form) than a badly biased one."""
    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, 2.0, size=4000)
    good = rng.gamma(2.0, 2.0, size=(4000, 64))  # same law as truth
    bad = np.full((4000, 64), 0.01)  # near-zero-everywhere (the "sharp" cheat)
    assert np.mean(psa.scrps_ensemble(y, good)) > np.mean(psa.scrps_ensemble(y, bad))


def test_randomized_pit_biased_piles_at_one():
    """Forecast far too low (all samples ~0) vs positive truth → PIT == 1 for every cell."""
    rng = np.random.default_rng(2)
    y = np.full(200, 5.0)
    x = np.zeros((200, 16))
    pit = psa.randomized_pit(y, x, rng)
    assert np.allclose(pit, 1.0)


def test_randomized_pit_calibrated_is_uniform():
    """Truth exchangeable with the ensemble → PIT approximately uniform on [0, 1]."""
    rng = np.random.default_rng(3)
    y = rng.normal(0.0, 1.0, size=8000)
    x = rng.normal(0.0, 1.0, size=(8000, 64))
    pit = psa.randomized_pit(y, x, rng)
    assert 0.45 < pit.mean() < 0.55
    freq, noncal = psa.pit_noncalibration(pit, nbins=10)
    assert noncal < 0.02  # near-flat histogram


def test_pit_noncalibration_flags_piled():
    """A piled-up PIT scores far from flat; a flat one scores near 0."""
    flat = np.linspace(0, 1, 10000, endpoint=False)
    piled = np.ones(10000)
    _, ncf = psa.pit_noncalibration(flat, nbins=10)
    _, ncp = psa.pit_noncalibration(piled, nbins=10)
    assert ncf < 0.01
    assert ncp > ncf * 5


def test_crps_perfect_is_zero():
    """Sanity: a perfect (degenerate-at-truth) ensemble has CRPS 0 (native wiring)."""
    y = np.array([0.0, 3.0, 12.0])
    x = np.repeat(y[:, None], 8, axis=1)
    assert psa.crps_mean(y, x) == 0.0
