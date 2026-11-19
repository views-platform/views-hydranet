"""
Tests for Cluster E: Loss/training science (C-45, C-47, C-48).

C-47: ParetoLoss in registry (deleted 2026-11-18, issue #631)
C-45: Hurdle masking in _process_sequence
C-48: QS99 tail regularizer (distribution-free pinball)
"""

import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput


# ---------------------------------------------------------------------------
# C-45: Hurdle masking
# ---------------------------------------------------------------------------
def test_hurdle_masking_reduces_loss_contributors():
    """
    With body_supervision="active", regression loss should only come from
    positive target values. A tensor with all zeros should produce
    zero regression loss.
    """
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

    B, T, C, H, W = 1, 3, 1, 4, 4
    # All zeros — with hurdle masking, regression loss should be 0
    train_tensor = torch.zeros(B, T, C, H, W)
    h = torch.zeros(B, 8, H, W)

    model = make_tiny_model()
    idx = _SequenceIndices(
        ["feat"],
        {"regression_targets": ["feat"], "classification_targets": [], "features": ["feat"]},
    )

    result = _process_sequence(
        train_tensor,
        model,
        h,
        torch.nn.MSELoss(),
        torch.nn.BCELoss(),
        SumReducer(),
        idx,
        torch.device("cpu"),
        body_supervision="active",
    )

    # With all-zero targets and hurdle masking, no positive pixels exist
    # so regression loss should be 0
    reg_loss = result["reg"].item()
    assert reg_loss == 0.0, (
        f"Regression loss should be 0 with all-zero targets and hurdle, got {reg_loss}"
    )


def test_hurdle_none_processes_all_pixels():
    """
    With body_mask='none' (default), all pixels contribute to loss.
    Even all-zero targets should produce non-zero regression loss
    (because predictions are non-zero).
    """
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

    B, T, C, H, W = 1, 3, 1, 4, 4
    train_tensor = torch.zeros(B, T, C, H, W)
    h = torch.zeros(B, 8, H, W)

    model = make_tiny_model()
    idx = _SequenceIndices(
        ["feat"],
        {"regression_targets": ["feat"], "classification_targets": [], "features": ["feat"]},
    )

    result = _process_sequence(
        train_tensor,
        model,
        h,
        torch.nn.MSELoss(),
        torch.nn.BCELoss(),
        SumReducer(),
        idx,
        torch.device("cpu"),
    )

    # Without hurdle, predictions on zero targets still produce non-zero MSE
    total = result["total"].item()
    assert total > 0, f"Without hurdle, loss should be > 0 even for zero targets, got {total}"


# ---------------------------------------------------------------------------
# C-48: QS99 regularizer
# ---------------------------------------------------------------------------
def test_qs99_regularizer_adds_to_loss():
    """
    With qs99_weight > 0 and hurdle enabled, the total loss should differ
    from the case without the regularizer.
    """
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

    B, T, C, H, W = 1, 3, 1, 4, 4
    train_tensor = torch.randn(B, T, C, H, W).abs()  # positive values
    h = torch.zeros(B, 8, H, W)

    model = make_tiny_model()
    idx = _SequenceIndices(
        ["feat"],
        {"regression_targets": ["feat"], "classification_targets": [], "features": ["feat"]},
    )

    # Without regularizer
    result_no_qs = _process_sequence(
        train_tensor,
        model,
        h.clone(),
        torch.nn.MSELoss(),
        torch.nn.BCELoss(),
        SumReducer(),
        idx,
        torch.device("cpu"),
        body_supervision="active",
        qs99_weight=0.0,
    )

    # With regularizer
    result_with_qs = _process_sequence(
        train_tensor,
        model,
        h.clone(),
        torch.nn.MSELoss(),
        torch.nn.BCELoss(),
        SumReducer(),
        idx,
        torch.device("cpu"),
        body_supervision="active",
        qs99_weight=0.1,
        qs99_tau=0.99,
    )

    loss_no = result_no_qs["total"].item()
    loss_with = result_with_qs["total"].item()
    assert loss_no != loss_with, (
        f"QS99 regularizer should change the total loss: without={loss_no}, with={loss_with}"
    )


def test_qs99_disabled_when_no_hurdle():
    """QS99 weight > 0 but body_mask='none' should NOT add regularizer."""
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

    B, T, C, H, W = 1, 3, 1, 4, 4
    train_tensor = torch.randn(B, T, C, H, W).abs()
    h = torch.zeros(B, 8, H, W)

    model = make_tiny_model()
    idx = _SequenceIndices(
        ["feat"],
        {"regression_targets": ["feat"], "classification_targets": [], "features": ["feat"]},
    )

    result_a = _process_sequence(
        train_tensor,
        model,
        h.clone(),
        torch.nn.MSELoss(),
        torch.nn.BCELoss(),
        SumReducer(),
        idx,
        torch.device("cpu"),
        qs99_weight=0.0,
    )

    result_b = _process_sequence(
        train_tensor,
        model,
        h.clone(),
        torch.nn.MSELoss(),
        torch.nn.BCELoss(),
        SumReducer(),
        idx,
        torch.device("cpu"),
        qs99_weight=0.1,
    )

    assert abs(result_a["total"].item() - result_b["total"].item()) < 1e-6, (
        "QS99 should have no effect when hurdle is disabled"
    )


# ---------------------------------------------------------------------------
# C-45: Hurdle masking with mixed data
# ---------------------------------------------------------------------------
def test_hurdle_masking_mixed_data():
    """
    With mixed data (some positive, some zero), hurdle masking should
    produce a different loss than no masking — because the zero-pixel
    gradients are excluded.
    """
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

    B, T, C, H, W = 1, 3, 1, 4, 4
    train_tensor = torch.zeros(B, T, C, H, W)
    # Plant a few positive values
    train_tensor[0, 1, 0, 0, 0] = 5.0
    train_tensor[0, 2, 0, 0, 0] = 3.0
    train_tensor[0, 1, 0, 2, 2] = 10.0
    train_tensor[0, 2, 0, 2, 2] = 7.0
    h = torch.zeros(B, 8, H, W)

    model = make_tiny_model()
    idx = _SequenceIndices(
        ["feat"],
        {"regression_targets": ["feat"], "classification_targets": [], "features": ["feat"]},
    )

    result_no_hurdle = _process_sequence(
        train_tensor,
        model,
        h.clone(),
        torch.nn.MSELoss(),
        torch.nn.BCELoss(),
        SumReducer(),
        idx,
        torch.device("cpu"),
    )

    result_hurdle = _process_sequence(
        train_tensor,
        model,
        h.clone(),
        torch.nn.MSELoss(),
        torch.nn.BCELoss(),
        SumReducer(),
        idx,
        torch.device("cpu"),
        body_supervision="active",
    )

    loss_no = result_no_hurdle["total"].item()
    loss_hurdle = result_hurdle["total"].item()
    assert loss_no != loss_hurdle, (
        f"Hurdle masking should change loss with mixed data: "
        f"no_hurdle={loss_no:.6f}, hurdle={loss_hurdle:.6f}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class SumReducer(nn.Module):
    """Mock MultiTaskLoss that sums all losses to a scalar."""

    def forward(self, losses):
        return losses.sum()


def make_tiny_model():
    """Minimal model that matches HydraNet's forward signature."""

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 1)
            self.base = 8

        def forward(self, x, h):
            out = self.conv(x)
            return ModelOutput(reg=out, cls=out, h_next=h)

        def init_hTtime(self, hidden_channels, H, W):
            return torch.zeros(1, hidden_channels, H, W)

    return TinyModel()
