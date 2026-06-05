"""Tests for the in-domain feedback-input clamp (C-113).

Pre-analysis: reports/preanalysis_feedback_clamp.md

The clamp bounds ONLY the autoregressive feedback copy of the prediction to the
per-target log1p training range, to break the runaway ratchet (violet settles at
log ~40 -> expm1 ~1e17). It must:
  - be a no-op when unset (default None) -> zero regression for existing models;
  - cap per-channel above the ceiling, pass through below it (legit predictions
    untouched -> magnitude-neutral on in-range values);
  - validate its config (positive, length == regression_targets).
"""

import pytest
import torch

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput
from views_hydranet.utils.hydranet_inference import HydraNetInference


class _MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = 32

    def forward(self, x, h):
        reg = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
        cls = torch.zeros(x.shape[0], 3, x.shape[2], x.shape[3])
        return ModelOutput(reg=reg, cls=cls, h_next=h)


def _cfg(clamp):
    return {
        "freeze_h": "none",
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "classification_targets": ["c1", "c2", "c3"],
        "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "feedback_clamp_log1p": clamp,
    }


def _inf(clamp):
    return HydraNetInference(_MockModel(), _cfg(clamp), device="cpu")


class TestGreenFeedbackClampMethod:
    """Contract of _clamp_feedback."""

    def test_identity_when_unset(self):
        inf = _inf(None)
        x = torch.tensor([1.0, 50.0, 100.0]).view(1, 3, 1, 1)
        out = inf._clamp_feedback(x)
        assert torch.equal(out, x), "None clamp must be a pure pass-through"

    def test_caps_above_ceiling_per_channel(self):
        inf = _inf([10.7828, 7.6014, 12.0921])
        # all channels driven way out of range (the runaway regime ~40)
        x = torch.full((1, 3, 2, 2), 40.0)
        out = inf._clamp_feedback(x)
        assert torch.allclose(out[:, 0], torch.tensor(10.7828))
        assert torch.allclose(out[:, 1], torch.tensor(7.6014))
        assert torch.allclose(out[:, 2], torch.tensor(12.0921))

    def test_passes_through_below_ceiling(self):
        inf = _inf([10.7828, 7.6014, 12.0921])
        # in-range predictions must be untouched (magnitude-neutral on legit values)
        x = torch.tensor([5.0, 3.0, 8.0]).view(1, 3, 1, 1)
        out = inf._clamp_feedback(x)
        assert torch.equal(out, x)

    def test_mixed_only_clamps_offenders(self):
        inf = _inf([10.0, 7.0, 12.0])
        x = torch.tensor([20.0, 2.0, 12.0]).view(1, 3, 1, 1)
        out = inf._clamp_feedback(x).view(3)
        assert torch.allclose(out, torch.tensor([10.0, 2.0, 12.0]))

    def test_does_not_mutate_input(self):
        inf = _inf([10.0, 7.0, 12.0])
        x = torch.full((1, 3, 1, 1), 40.0)
        x_before = x.clone()
        _ = inf._clamp_feedback(x)
        assert torch.equal(x, x_before), "must return a new tensor, not mutate in place"

    def test_floor_untouched_relu_already_nonneg(self):
        # only max is bounded; ReLU gives the >=0 floor, clamp must not raise small values
        inf = _inf([10.0, 7.0, 12.0])
        x = torch.zeros(1, 3, 1, 1)
        assert torch.equal(inf._clamp_feedback(x), x)


class TestRedFeedbackClampValidation:
    """Fail-loud on bad config (no silent correction)."""

    def test_negative_value_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _inf([10.0, -1.0, 12.0])

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="regression_targets"):
            _inf([10.0, 7.0])  # 2 entries, 3 targets

    def test_non_list_raises(self):
        with pytest.raises(ValueError):
            _inf(12.0)  # scalar, not a per-target list
