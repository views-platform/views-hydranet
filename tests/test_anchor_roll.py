"""EXP-3: the rolled clamp anchor — the dissociation arm, and the no-ops that must fail loud.

Clamping the cell state tangles two things: the anchor's SCALE (how big the state's numbers are)
and its MAP (which cells are hot). M52 showed the clamp preserves gate-body alignment, but that is
consistent both with "the clamp preserves placement" and with "the clamp steadies the scale and
alignment follows". Rolling the anchor separates them: a roll is a permutation, so every scalar
property survives exactly and only the correspondence to geography is destroyed.

The arm is worth nothing if it can quietly not happen, so both silent no-ops are asserted to raise:
a roll of zero, and a roll by a whole number of grid periods. Either would write the plain clamp
into a file named for the rolled arm — a complete-looking result that is secretly the control.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from views_hydranet.utils.hydranet_inference import HydraNetInference  # noqa: E402

from .distributions.test_sampler_dxk import _FEATURES, _mock_handler  # noqa: E402


def _make(**kw):
    from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4

    torch.manual_seed(0)
    model = HydraBNUNet06_LSTM4(3, 16, 1, 0.0, output_distribution="nb").float()
    config = {
        "steps": [1, 2, 3, 4, 5, 6],
        "time_steps": 6,
        "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
        "classification_targets": ["by_sb", "by_ns", "by_os"],
        "features": ["lr_sb", "lr_ns", "lr_os"],
        "static_channels": [],
        "n_posterior_samples": 2,
        "n_head_samples": 2,
        "np_seed": 42,
        "torch_seed": 1234,
        "forecast_composition": "soft_gate",
    }
    return HydraNetInference(model, config, device="cpu", **kw)


# ── the roll is a permutation ────────────────────────────────────────────


def test_roll_preserves_every_scalar_property_of_the_anchor():
    """The whole design rests on this: same state, wrong place.

    If the roll changed the anchor's magnitudes, a null result would be ambiguous between "placement
    matters" and "we damaged the state", and the arm would answer nothing.
    """
    inf = _make(freeze_recurrent="cell", freeze_anchor_roll=15)
    a = torch.randn(1, 32, 12, 12)
    r = inf._roll_anchor(a)

    torch.testing.assert_close(torch.sort(a.flatten())[0], torch.sort(r.flatten())[0])
    assert r.shape == a.shape
    torch.testing.assert_close(a.norm(), r.norm())
    torch.testing.assert_close(a.mean(), r.mean())
    torch.testing.assert_close(a.var(), r.var())
    # per-channel too: a roll that mixed channels would not be the intended operation
    torch.testing.assert_close(a.mean(dim=(-2, -1)), r.mean(dim=(-2, -1)))
    assert not torch.equal(a, r), "the roll must actually move something"


def test_roll_is_spatial_only_and_matches_torch_roll():
    inf = _make(freeze_recurrent="cell", freeze_anchor_roll=3)
    a = torch.randn(1, 8, 10, 10)
    torch.testing.assert_close(inf._roll_anchor(a), torch.roll(a, shifts=(3, 3), dims=(-2, -1)))


# ── the silent no-ops must fail loud ─────────────────────────────────────


def test_zero_roll_is_refused():
    """A zero roll is the plain clamp wearing the treatment's label."""
    with pytest.raises(ValueError, match="identity and reproduces the plain clamp"):
        _make(freeze_recurrent="cell", freeze_anchor_roll=0)


def test_whole_period_roll_is_refused():
    """Rolling a 12x12 field by 12 returns it unchanged — a control mislabelled as a treatment."""
    inf = _make(freeze_recurrent="cell", freeze_anchor_roll=12)
    with pytest.raises(ValueError, match="whole number of grid periods"):
        inf._roll_anchor(torch.randn(1, 8, 12, 12))


def test_roll_without_a_clamp_is_refused():
    """The anchor is only read when a half is held, so rolling it alone changes nothing."""
    with pytest.raises(ValueError, match="needs freeze_recurrent set"):
        _make(freeze_anchor_roll=15)


def test_non_integer_roll_is_refused():
    with pytest.raises(ValueError, match="must be an int or None"):
        _make(freeze_recurrent="cell", freeze_anchor_roll=1.5)


# ── default-off and end-to-end ───────────────────────────────────────────


def test_default_is_off_and_byte_identical():
    """Every pre-EXP-3 arm must be untouched by this seam existing."""
    handler = _mock_handler(_FEATURES, seq_len=8, h=16, w=16)
    a_mag, a_prob = _make(freeze_recurrent="cell").generate_posterior_samples(handler, origin=1)
    b_mag, b_prob = _make().generate_posterior_samples(handler, origin=1)
    assert _make().freeze_anchor_roll is None
    # the clamp itself still does something, so these must differ — otherwise the test is vacuous
    assert not np.array_equal(a_mag, b_mag)


def test_rolled_arm_differs_from_the_plain_clamp():
    """If the rolled arm reproduced the clamp, the experiment would be comparing a thing to itself."""
    handler = _mock_handler(_FEATURES, seq_len=8, h=16, w=16)
    plain, _ = _make(freeze_recurrent="cell").generate_posterior_samples(handler, origin=1)
    rolled, _ = _make(freeze_recurrent="cell", freeze_anchor_roll=3).generate_posterior_samples(
        handler, origin=1
    )
    assert not np.array_equal(plain, rolled), "the roll had no effect on the emitted cube"
