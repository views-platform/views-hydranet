"""The decay dial on `blend_recurrent_state`.

M38/M39: hard-freezing the CELL state recovers ~23% of the oracle gap at h18 and leaves 77% open.
A hard freeze is the most extreme setting of a dial nobody had turned, so the hook gained a convex
blend weight. These tests pin the two properties the whole sweep rests on:

1. **`weight=1.0` is byte-identical to the pre-dial behaviour** — otherwise the already-published
   hard-freeze arms would silently drift under new float arithmetic and M38 would stop reproducing.
2. **The blend is a convex pull toward the anchor**, monotone in `weight`, and confined to the
   selected memory half.
"""

from __future__ import annotations

import pytest
import torch

from views_hydranet.utils.hydranet_inference import blend_recurrent_state

C = 16  # 8 short-term + 8 long-term; divisible by the 8-group split
HALF = C // 2


def _pair(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    new = torch.randn(2, C, 5, 5, generator=g)
    anchor = torch.randn(2, C, 5, 5, generator=g)
    return new, anchor


@pytest.mark.parametrize("mode", ["hidden", "cell", "all"])
def test_weight_one_is_byte_identical_to_the_hard_freeze(mode):
    """THE regression that protects M38/M39. The published arms must keep reproducing exactly."""
    new, anchor = _pair()
    got = blend_recurrent_state(new, anchor, mode, weight=1.0)
    if mode == "all":
        expected = anchor.clone()
    elif mode == "hidden":
        expected = torch.cat([anchor[:, :HALF], new[:, HALF:]], dim=1)
    else:
        expected = torch.cat([new[:, :HALF], anchor[:, HALF:]], dim=1)
    assert torch.equal(got, expected), "weight=1.0 drifted from the original branch"


@pytest.mark.parametrize("mode", ["hidden", "cell", "all"])
def test_weight_zero_is_a_no_op(mode):
    new, anchor = _pair()
    assert torch.allclose(blend_recurrent_state(new, anchor, mode, weight=0.0), new)


def test_the_blend_is_convex_and_only_touches_the_selected_half():
    """`cell` must leave the short-term half exactly as the model produced it, at ANY weight."""
    new, anchor = _pair()
    got = blend_recurrent_state(new, anchor, "cell", weight=0.4)
    assert torch.equal(got[:, :HALF], new[:, :HALF]), "the hidden half was disturbed"
    assert torch.allclose(got[:, HALF:], 0.4 * anchor[:, HALF:] + 0.6 * new[:, HALF:])


def test_hidden_mode_leaves_the_long_term_half_untouched():
    new, anchor = _pair()
    got = blend_recurrent_state(new, anchor, "hidden", weight=0.4)
    assert torch.equal(got[:, HALF:], new[:, HALF:])
    assert torch.allclose(got[:, :HALF], 0.4 * anchor[:, :HALF] + 0.6 * new[:, :HALF])


def test_larger_weight_moves_strictly_closer_to_the_anchor():
    """Monotonicity is what makes a dial a dial — the sweep reads a dose-response off it."""
    new, anchor = _pair()
    prev = None
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        d = (
            (blend_recurrent_state(new, anchor, "cell", weight=w)[:, HALF:] - anchor[:, HALF:])
            .abs()
            .sum()
        )
        if prev is not None:
            assert d < prev, f"distance to anchor did not shrink at weight={w}"
        prev = d


@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0, -1.0])
def test_weight_outside_the_unit_interval_raises(bad):
    """An extrapolating blend is not a decay: the state leaves the segment between what the model
    produced and what it learned from real observations."""
    new, anchor = _pair()
    with pytest.raises(ValueError, match=r"weight must be in \[0, 1\]"):
        blend_recurrent_state(new, anchor, "cell", weight=bad)


def test_neither_input_is_mutated_at_a_partial_weight():
    new, anchor = _pair()
    n0, a0 = new.clone(), anchor.clone()
    blend_recurrent_state(new, anchor, "all", weight=0.3)
    assert torch.equal(new, n0) and torch.equal(anchor, a0)


def test_the_inference_object_rejects_a_weight_outside_the_unit_interval():
    """Validated at construction, so a mistyped arm raises before any GPU time — the same contract
    `freeze_recurrent` has (`test_recurrent_state_freeze.py::test_invalid_mode_raises_...`)."""
    from tests.test_recurrent_state_freeze import _CFG, _StateSensitiveModel
    from views_hydranet.utils.hydranet_inference import HydraNetInference

    with pytest.raises(ValueError, match=r"freeze_recurrent_weight must be in \[0, 1\]"):
        HydraNetInference(
            _StateSensitiveModel(),
            dict(_CFG),
            device="cpu",
            freeze_recurrent="cell",
            freeze_recurrent_weight=1.5,
        )


def test_the_default_weight_is_a_hard_freeze_so_existing_callers_are_unchanged():
    """Every caller written before the dial passes no weight; they must still hard-freeze."""
    from tests.test_recurrent_state_freeze import _CFG, _StateSensitiveModel
    from views_hydranet.utils.hydranet_inference import HydraNetInference

    inf = HydraNetInference(
        _StateSensitiveModel(), dict(_CFG), device="cpu", freeze_recurrent="cell"
    )
    assert inf.freeze_recurrent_weight == 1.0
