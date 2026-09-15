"""Diagnostic recurrent-state freeze (#258/#262, C-222).

`freeze_h` was retired in 2026-06 on an ablation that measured **regression CRPS** and found the
freeze inert against the C-113 runaway. It never measured **classification**, and activation-aware
metrics did not exist until 2026-08 — so whether holding the ConvLSTM state preserves *gate* skill
across the rollout is untested rather than refuted. `freeze_recurrent` is the explicit diagnostic
that answers it. It is not a config key, so no model config can enable it, and
`test_inference_logic.py::test_freeze_h_option_retired` stays green.

Every test here was checked against a deliberate sabotage, because a guard that cannot fail is not
a guard. Three distinct leaks, three catches: an anchor captured one step late and a blend
mis-placed at the seed step are both caught by `test_freezing_all_makes_the_state_stop_contributing
_new_information`; a blend leaking into the history-digestion branch is caught by
`test_h1_is_byte_identical_across_every_mode`. See that test's docstring for why h=1 is immune to
the first two — the property is narrower than "the freeze starts after the seed step" suggests.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput
from views_hydranet.utils.hydranet_inference import (
    FREEZE_RECURRENT_MODES,
    HydraNetInference,
    blend_recurrent_state,
)

# ------------------------------------------------------------------ the pure blend


def _state(channels=8, fill=0.0):
    """[1, channels, 1, 1] with one distinct value per channel group when fill is a sequence."""
    t = torch.zeros(1, channels, 1, 1)
    t[:] = fill
    return t


def test_blend_hidden_holds_the_short_term_half_only():
    new, anchor = _state(fill=2.0), _state(fill=9.0)
    out = blend_recurrent_state(new, anchor, "hidden")
    assert torch.equal(out[:, :4], anchor[:, :4]), "hs_1..hs_4 must come from the anchor"
    assert torch.equal(out[:, 4:], new[:, 4:]), "hl_1..hl_4 must keep evolving"


def test_blend_cell_holds_the_long_term_half_only():
    new, anchor = _state(fill=2.0), _state(fill=9.0)
    out = blend_recurrent_state(new, anchor, "cell")
    assert torch.equal(out[:, :4], new[:, :4]), "hs_1..hs_4 must keep evolving"
    assert torch.equal(out[:, 4:], anchor[:, 4:]), "hl_1..hl_4 must come from the anchor"


def test_blend_all_returns_the_anchor():
    new, anchor = _state(fill=2.0), _state(fill=9.0)
    out = blend_recurrent_state(new, anchor, "all")
    assert torch.equal(out, anchor)


def test_blend_does_not_mutate_its_inputs():
    """The rollout reuses both tensors; an in-place blend would corrupt the next step."""
    new, anchor = _state(fill=2.0), _state(fill=9.0)
    blend_recurrent_state(new, anchor, "all")
    assert torch.equal(new, _state(fill=2.0))
    assert torch.equal(anchor, _state(fill=9.0))


def test_blend_rejects_an_unknown_mode():
    s = _state()
    with pytest.raises(ValueError, match="mode must be one of"):
        blend_recurrent_state(s, s, "hs")  # the retired freeze_h vocabulary


def test_blend_rejects_a_channel_count_that_cannot_split_into_8_groups():
    """An uneven split would silently hold the wrong memory type — worse than failing."""
    new, anchor = _state(channels=12), _state(channels=12)
    with pytest.raises(ValueError, match="not divisible by 8"):
        blend_recurrent_state(new, anchor, "hidden")


def test_blend_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        blend_recurrent_state(_state(channels=8), _state(channels=16), "all")


# ------------------------------------------------------- the argument on the inference object


class _StateSensitiveModel(nn.Module):
    """A mock whose prediction depends on BOTH memory halves, so a freeze is observable.

    The two halves evolve at different rates, so `hidden` and `cell` produce distinguishable
    trajectories rather than accidentally agreeing.
    """

    def __init__(self, base=32):
        super().__init__()
        self.base = base

    def init_hTtime(self, hidden_channels, H, W):
        return torch.zeros(1, hidden_channels, H, W)

    def forward(self, x, h):
        half = h.shape[1] // 2
        short = h[:, :half].mean(dim=1, keepdim=True)
        long_ = h[:, half:].mean(dim=1, keepdim=True)
        # The two halves must contribute at DIFFERENT rates (1/step vs 5/step). With equal rates
        # `hidden` and `cell` produce byte-identical rollouts and the arms silently stop being a
        # comparison — which is what the first draft of this mock did.
        reg = x + short + 10.0 * long_
        h_new = h.clone()
        h_new[:, :half] += 1.0
        h_new[:, half:] += 0.5
        return ModelOutput(reg=reg, cls=torch.zeros_like(reg), h_next=h_new)


_CFG = {
    "steps": list(range(1, 7)),
    "time_steps": 6,
    "features": ["feat"],
    "regression_targets": ["feat"],
    "classification_targets": ["class"],
    "sampling_strategy": "threshold",
    "n_posterior_samples": 1,
    "diagnostic_visualizations": False,
}


def _rollout(mode):
    """One 6-step rollout under `mode`; returns magnitudes [T, n_reg, H, W]."""
    inf = HydraNetInference(
        _StateSensitiveModel(), dict(_CFG), device="cpu", freeze_recurrent=mode
    )
    full_tensor = torch.ones(1, 4, 1, 2, 2)
    magnitudes, _ = inf.predict(full_tensor, 3, 0, ["feat"])
    return magnitudes


def test_default_is_none_and_the_state_evolves():
    inf = HydraNetInference(_StateSensitiveModel(), dict(_CFG), device="cpu")
    assert inf.freeze_recurrent is None
    mags = _rollout(None)
    steps = [float(mags[t].mean()) for t in range(mags.shape[0])]
    assert steps == sorted(steps) and steps[-1] > steps[0], (
        f"the default path must evolve the full state, got a flat trajectory: {steps}"
    )


def test_invalid_mode_raises_rather_than_silently_running_the_control():
    """A typo must not quietly produce the `none` arm and be reported as 'no effect'."""
    with pytest.raises(ValueError, match="freeze_recurrent must be None or one of"):
        HydraNetInference(
            _StateSensitiveModel(), dict(_CFG), device="cpu", freeze_recurrent="hidden_state"
        )


def test_h1_is_byte_identical_across_every_mode():
    """F1, the experiment's self-test: month 1 must be identical in every arm.

    **What this actually guards, verified by sabotage.** h=1 is produced by the seed step's forward
    pass, which reads the state built during history digestion. So it is immune to a freeze applied
    at or after that forward — a mis-placed blend there shows up at h>=2 instead (the increments
    test catches it). What moves h=1 is a freeze leaking into the **digestion** phase
    (`t < origin`), and that is what this catches: patching the blend into the digestion branch
    turns this test red while the other eleven stay green.

    If it fails, every arm's numbers are void rather than merely different — the arms would no
    longer share a common history.
    """
    baseline = _rollout(None)[0]
    for mode in FREEZE_RECURRENT_MODES:
        assert np.array_equal(_rollout(mode)[0], baseline), (
            f"h=1 moved under freeze_recurrent={mode!r}; the freeze is leaking into the "
            "digestion/seed phase and the probe is invalid"
        )


def test_each_mode_produces_a_distinct_rollout_beyond_h1():
    """Otherwise an arm silently duplicates another and the experiment compares nothing."""
    tails = {m: _rollout(m)[1:] for m in (None, *FREEZE_RECURRENT_MODES)}
    modes = list(tails)
    for i, a in enumerate(modes):
        for b in modes[i + 1 :]:
            assert not np.array_equal(tails[a], tails[b]), (
                f"freeze_recurrent={a!r} and {b!r} produced identical rollouts past h=1"
            )


def _increments(mode):
    mags = _rollout(mode)
    means = [float(mags[t].mean()) for t in range(mags.shape[0])]
    return [b - a for a, b in zip(means, means[1:])]


def test_freezing_all_makes_the_state_stop_contributing_new_information():
    """The hard prior in full: past the seed step the state adds a CONSTANT each step.

    Note what this does *not* claim. The trajectory still drifts under `all`, because the
    prediction→input feedback path compounds independently of the state — which is precisely the
    2026-06 finding that the C-113 runaway 'rides the prediction→input feedback path, not the
    recurrent state'. What the freeze removes is the state's *growing* contribution, so the
    per-step increment goes flat while the level keeps moving.
    """
    frozen = _increments("all")
    assert max(frozen) - min(frozen) < 1e-5, (
        f"'all' must hold the state fixed, so increments should be constant; got {frozen}"
    )

    evolving = _increments(None)
    assert max(evolving) - min(evolving) > 1e-3, (
        f"the control arm's increments should GROW as the state grows; got {evolving}"
    )
