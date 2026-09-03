"""The potency gate itself. A gate that cannot fail is worse than no gate — it launders confidence.

This one exists because of a specific failure (#308: a change that was unit tested, mutation tested
5/5, lint clean and suite green, and was a no-op on the production path). So the tests here are
about the gate's ABILITY TO REFUSE, not its ability to pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pc = pytest.importorskip("potency_check")


def test_it_refuses_an_inert_knob():
    """THE test. This is the #308 signature: flipping the knob changes nothing."""
    with pytest.raises(pc.PotencyError, match="INERT"):
        pc.assert_potent(lambda _: 3.0, off=False, on=True, name="dead knob")


def test_it_passes_a_knob_that_moves_something():
    r = pc.assert_potent(lambda on: 2.0 if on else 1.0, off=False, on=True, name="live knob")
    assert r["off"] == 1.0 and r["on"] == 2.0 and r["relative_change"] == pytest.approx(0.5)


def test_it_refuses_non_finite_readings():
    """A NaN measure would otherwise compare unequal to itself and pass as 'potent'."""
    with pytest.raises(pc.PotencyError, match="non-finite"):
        pc.assert_potent(lambda _: float("nan"), off=0, on=1, name="broken measure")


def test_two_zeros_are_inert_not_a_division_error():
    """Both readings zero is the commonest inert case; it must refuse, not raise ZeroDivisionError."""
    with pytest.raises(pc.PotencyError, match="INERT"):
        pc.assert_potent(lambda _: 0.0, off=False, on=True, name="zero knob")


def test_a_change_below_the_threshold_is_still_inert():
    """Guards against a knob that jiggles a float without doing anything."""
    with pytest.raises(pc.PotencyError, match="INERT"):
        pc.assert_potent(
            lambda on: 1.0 + (1e-12 if on else 0.0),
            off=False,
            on=True,
            name="jitter",
            min_relative_change=1e-6,
        )


def test_the_positive_control_helper_refuses_a_blind_readout():
    """The mirror question: could the READOUT see an effect if there were one?

    A null from a harness whose positive control does not fire is not a null — and after the fact
    this repo has no way to tell those apart.
    """
    with pytest.raises(pc.PotencyError, match="positive control"):
        pc.assert_control_fires(lambda _: 7.0, baseline=0, known_effect=999, name="blind readout")


def test_the_positive_control_passes_when_the_readout_works():
    r = pc.assert_control_fires(
        lambda x: float(x), baseline=1.0, known_effect=100.0, name="working readout"
    )
    assert r["relative_change"] > 0.9
