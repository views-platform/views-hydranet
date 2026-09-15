"""Q C.4's instrument: can it separate CONTINUATION skill from ONSET skill?

The question it exists to answer is whether the cell clamp's AP gain is entirely conflict that was
already there. That is only worth asking if the tool can actually tell the two apart, so the
load-bearing test plants a model that is perfect at one and useless at the other and requires the
tool to say so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_D = Path(__file__).resolve().parents[1] / "reports" / "2026-09-03_drivers_and_dynamics_dossier"
sys.path.insert(0, str(_D / "tools"))

sa = pytest.importorskip("subset_ap")


def _world(n_cont=200, n_onset=5000, seed=0):
    """Origin-active cells and origin-quiet cells, with realistic prevalences."""
    rng = np.random.default_rng(seed)
    truth_origin = np.concatenate([np.ones(n_cont), np.zeros(n_onset)])
    persists = rng.random(n_cont) < 0.6  # 60% of active conflicts continue
    starts = rng.random(n_onset) < 0.01  # 1% of quiet cells see onset
    truth_h = np.concatenate([persists, starts]).astype(float)
    return truth_origin, truth_h, n_cont, n_onset


def test_a_continuation_only_model_is_exposed_as_such():
    """THE test. A model perfect at persistence and blind to onset must score high / at-chance.

    This is the exact signature the clamp is suspected of having: pin a map of where conflict
    already is, ace the continuing cells, learn nothing about where it starts.
    """
    truth_origin, truth_h, n_cont, n_onset = _world()
    rng = np.random.default_rng(1)
    score = np.concatenate(
        [
            np.where(truth_h[:n_cont] > 0, 0.9, 0.1),  # perfect on continuation
            rng.random(n_onset) * 0.5,  # pure noise on onset
        ]
    )
    r = sa.partition_ap(score, truth_h, truth_origin)

    assert r["ap_cont"] > 0.95, "perfect continuation ranking was not detected"
    assert r["ap_onset"] == pytest.approx(r["base_onset"], abs=0.02), (
        "a chance-level onset ranker did not score at the onset base rate"
    )
    assert r["ap_onset"] < 0.1 * r["ap_cont"], "the asymmetry the tool exists to find is invisible"


def test_an_onset_only_model_is_exposed_as_such():
    """The mirror, so the tool is not merely biased toward reporting continuation skill."""
    truth_origin, truth_h, n_cont, n_onset = _world(seed=3)
    rng = np.random.default_rng(4)
    score = np.concatenate(
        [
            rng.random(n_cont) * 0.5,
            np.where(truth_h[n_cont:] > 0, 0.9, 0.1),
        ]
    )
    r = sa.partition_ap(score, truth_h, truth_origin)
    assert r["ap_onset"] > 0.95
    assert r["ap_cont"] == pytest.approx(r["base_cont"], abs=0.08)


def test_the_two_universes_partition_every_cell():
    """No cell may be counted twice or dropped — otherwise the two APs are not comparable."""
    truth_origin, truth_h, n_cont, n_onset = _world()
    r = sa.partition_ap(np.zeros_like(truth_h) + 0.5, truth_h, truth_origin)
    assert r["n_cont"] + r["n_onset"] == len(truth_h)
    assert r["n_cont"] == n_cont and r["n_onset"] == n_onset
    assert r["pos_cont"] + r["pos_onset"] == int((truth_h > 0).sum())


def test_base_rates_are_reported_and_differ_wildly():
    """They are reported precisely because the two APs must NOT be compared to each other."""
    truth_origin, truth_h, _, _ = _world()
    r = sa.partition_ap(np.random.default_rng(2).random(len(truth_h)), truth_h, truth_origin)
    assert r["base_cont"] > 20 * r["base_onset"], "fixture does not reproduce the real asymmetry"


def test_a_degenerate_universe_is_nan_not_a_number():
    """C-318: an unsupported statistic must be NaN, never a plausible-looking value."""
    truth_origin = np.ones(50)  # every cell active at origin => onset universe empty
    truth_h = (np.arange(50) < 25).astype(float)
    r = sa.partition_ap(np.random.default_rng(0).random(50), truth_h, truth_origin)
    assert r["n_onset"] == 0
    assert np.isnan(r["ap_onset"])
    assert not np.isnan(r["ap_cont"])


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="same length"):
        sa.partition_ap(np.zeros(5), np.zeros(5), np.zeros(4))
