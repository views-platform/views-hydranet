"""Wave 2 attribution: roll ONE live driver per step and see which one the forecast follows.

Wave 1 answered the driver question only coarsely, through interventions. This arm asks it
directly: if the emitted field follows the rolled cell state, the cell drives it; if it follows the
rolled input, the input does. The measurement is the cross-correlation instrument EXP-3b already
validated, so what has to be established here is that the roll does exactly what it claims — moves
ONE driver, leaves the other untouched, and preserves everything except position.
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
        "steps": list(range(1, 7)),
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


# ── the roll moves exactly one driver ────────────────────────────────────


@pytest.mark.parametrize("which", ["input", "hidden", "cell"])
def test_exactly_one_driver_moves(which):
    """The whole comparison rests on this: arms must differ only in WHICH driver was displaced."""
    inf = _make(per_step_roll=f"{which}:3")
    inp = torch.randn(1, 3, 12, 12)
    state = torch.randn(1, 32, 12, 12)
    out_i, out_s = inf._roll_driver(inp, state)

    half = state.shape[1] // 2
    moved = {
        "input": not torch.equal(out_i, inp),
        "hidden": not torch.equal(out_s[:, :half], state[:, :half]),
        "cell": not torch.equal(out_s[:, half:], state[:, half:]),
    }
    assert moved[which], f"{which} did not move"
    assert sum(moved.values()) == 1, f"expected exactly one driver to move, got {moved}"


def test_the_roll_is_a_permutation_of_the_driver_it_moves():
    """If the roll changed magnitudes, a null result would be ambiguous between 'this driver
    matters' and 'we damaged it' — the same property EXP-3b's anchor roll had to have."""
    inf = _make(per_step_roll="cell:5")
    state = torch.randn(1, 32, 10, 10)
    _, out = inf._roll_driver(torch.randn(1, 3, 10, 10), state)
    half = 16
    torch.testing.assert_close(
        torch.sort(state[:, half:].flatten())[0], torch.sort(out[:, half:].flatten())[0]
    )
    torch.testing.assert_close(state.norm(), out.norm())


def test_the_state_split_matches_blend_recurrent_state():
    """hidden is the FIRST half (hs_1..hs_4). Getting this backwards would silently swap the two
    arms' labels and invert the attribution."""
    inf = _make(per_step_roll="hidden:4")
    state = torch.zeros(1, 8, 8, 8)
    state[:, :4] = 1.0  # hidden half marked
    _, out = inf._roll_driver(torch.randn(1, 3, 8, 8), state)
    assert out[:, :4].sum() == state[:, :4].sum(), "hidden half lost mass"
    assert torch.equal(out[:, 4:], state[:, 4:]), "cell half was touched by a hidden roll"


# ── the silent no-ops fail loud ──────────────────────────────────────────


def test_zero_shift_is_refused():
    with pytest.raises(ValueError, match="identity"):
        _make(per_step_roll="cell:0")


def test_whole_period_shift_is_refused():
    inf = _make(per_step_roll="cell:12")
    with pytest.raises(ValueError, match="whole number of grid periods"):
        inf._roll_driver(torch.randn(1, 3, 12, 12), torch.randn(1, 32, 12, 12))


def test_unknown_driver_is_refused():
    with pytest.raises(ValueError, match="input\\|hidden\\|cell"):
        _make(per_step_roll="state:90")


def test_malformed_spec_is_refused():
    with pytest.raises(ValueError, match="must look like"):
        _make(per_step_roll="cell90")


# ── default-off and end-to-end ───────────────────────────────────────────


def test_default_is_off():
    assert _make().per_step_roll is None


def test_each_arm_changes_the_emitted_cube_and_they_differ_from_each_other():
    """If two arms produced the same cube, the attribution would be comparing a thing to itself."""
    handler = _mock_handler(_FEATURES, seq_len=8, h=16, w=16)
    base, _ = _make().generate_posterior_samples(handler, origin=1)
    cubes = {}
    for which in ("input", "hidden", "cell"):
        cubes[which], _ = _make(per_step_roll=f"{which}:5").generate_posterior_samples(
            handler, origin=1
        )
        assert not np.array_equal(cubes[which], base), f"{which} roll had no effect on the cube"
    assert not np.array_equal(cubes["hidden"], cubes["cell"]), "hidden and cell arms are identical"
    assert not np.array_equal(cubes["input"], cubes["cell"]), "input and cell arms are identical"


@pytest.mark.parametrize("which", ["input", "hidden", "cell"])
def test_every_roll_is_spatial_and_matches_torch_roll(which):
    """Found by mutation (P3): rolling the INPUT on the channel axis survived every other test.

    A channel roll would permute the three targets into each other rather than displace the field,
    so the arm would measure something entirely different while still producing a plausible number
    — and the cross-correlation readout, which looks for a spatial peak, would find nothing and be
    read as "this driver does not drive the forecast".
    """
    inf = _make(per_step_roll=f"{which}:3")
    inp = torch.randn(1, 3, 11, 11)
    state = torch.randn(1, 32, 11, 11)
    out_i, out_s = inf._roll_driver(inp, state)

    if which == "input":
        torch.testing.assert_close(out_i, torch.roll(inp, shifts=(3, 3), dims=(-2, -1)))
    else:
        half = state.shape[1] // 2
        sl = slice(None, half) if which == "hidden" else slice(half, None)
        torch.testing.assert_close(
            out_s[:, sl], torch.roll(state[:, sl], shifts=(3, 3), dims=(-2, -1))
        )
    # a channel roll would change per-channel means; a spatial one cannot
    torch.testing.assert_close(inp.mean(dim=(-2, -1)), out_i.mean(dim=(-2, -1)))
    torch.testing.assert_close(state.mean(dim=(-2, -1)), out_s.mean(dim=(-2, -1)))
