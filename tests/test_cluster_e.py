"""
Tests for Cluster E: Loss/training science (C-45, C-47, C-48).

C-47: ParetoLoss in registry
C-45: Hurdle masking in _process_sequence
C-48: QS99 tail regularizer (distribution-free pinball)
"""

import torch


# ---------------------------------------------------------------------------
# C-47: ParetoLoss
# ---------------------------------------------------------------------------
def test_pareto_loss_importable():
    from views_hydranet.utils.pareto_loss import ParetoLoss  # noqa: F401


def test_pareto_loss_is_nn_module():
    from views_hydranet.utils.pareto_loss import ParetoLoss

    assert isinstance(ParetoLoss(alpha=1.0), torch.nn.Module)


def test_pareto_loss_forward_returns_scalar():
    from views_hydranet.utils.pareto_loss import ParetoLoss

    loss_fn = ParetoLoss(alpha=1.0)
    pred = torch.randn(1, 4, 4, requires_grad=True)
    target = torch.randn(1, 4, 4)
    result = loss_fn(pred, target)
    assert result.ndim == 0
    assert result.requires_grad
    assert result >= 0, f"Pareto loss should be non-negative, got {result.item()}"


def test_pareto_loss_registered_in_choose_loss():
    from views_hydranet.utils.pareto_loss import ParetoLoss
    from views_hydranet.utils.utils import choose_loss

    config = {
        "loss_reg": "pareto",
        "loss_reg_pareto_alpha": 1.0,
        "loss_class": "bce",
        "regression_targets": ["lr_sb_best"],
        "classification_targets": [],
    }
    criterion_reg, _, _ = choose_loss(config, torch.device("cpu"))
    assert isinstance(criterion_reg, ParetoLoss)


def test_pareto_loss_gradient_between_l1_and_mse():
    """
    Pareto gradient should be smaller than MSE but larger than zero
    for large errors — it's sublinear, not quadratic.
    """
    from views_hydranet.utils.pareto_loss import ParetoLoss

    pred = torch.zeros(1, 4, 4, requires_grad=True)
    target = torch.zeros(1, 4, 4)
    target[0, 0, 0] = 100.0

    ParetoLoss(alpha=1.0)(pred, target).backward()
    grad_pareto = pred.grad[0, 0, 0].abs().item()
    pred.grad = None

    torch.nn.MSELoss()(pred, target).backward()
    grad_mse = pred.grad[0, 0, 0].abs().item()

    assert grad_pareto < grad_mse, (
        f"Pareto gradient ({grad_pareto}) should be < MSE ({grad_mse}) for outliers"
    )
    assert grad_pareto > 0, "Pareto gradient should be > 0"


# ---------------------------------------------------------------------------
# C-45: Hurdle masking
# ---------------------------------------------------------------------------
def test_hurdle_masking_reduces_loss_contributors():
    """
    With hurdle_threshold=0.0, regression loss should only come from
    positive target values. A tensor with all zeros should produce
    zero regression loss.
    """
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

    B, T, C, H, W = 1, 3, 1, 4, 4
    # All zeros — with hurdle masking, regression loss should be 0
    train_tensor = torch.zeros(B, T, C, H, W)
    h = torch.zeros(B, 8, H, W)

    model = _make_tiny_model()
    idx = _SequenceIndices(["feat"], {"regression_targets": ["feat"],
                                       "classification_targets": [],
                                       "features": ["feat"]})

    result = _process_sequence(
        train_tensor, model, h,
        torch.nn.MSELoss(), torch.nn.BCELoss(), _SumReducer(),
        idx, torch.device("cpu"),
        hurdle_threshold=0.0,
    )

    # With all-zero targets and hurdle masking, no positive pixels exist
    # so regression loss should be 0
    reg_loss = result["reg"].item()
    assert reg_loss == 0.0, (
        f"Regression loss should be 0 with all-zero targets and hurdle, got {reg_loss}"
    )


def test_hurdle_none_processes_all_pixels():
    """
    With hurdle_threshold=None (default), all pixels contribute to loss.
    Even all-zero targets should produce non-zero regression loss
    (because predictions are non-zero).
    """
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

    B, T, C, H, W = 1, 3, 1, 4, 4
    train_tensor = torch.zeros(B, T, C, H, W)
    h = torch.zeros(B, 8, H, W)

    model = _make_tiny_model()
    idx = _SequenceIndices(["feat"], {"regression_targets": ["feat"],
                                       "classification_targets": [],
                                       "features": ["feat"]})

    result = _process_sequence(
        train_tensor, model, h,
        torch.nn.MSELoss(), torch.nn.BCELoss(), _SumReducer(),
        idx, torch.device("cpu"),
        hurdle_threshold=None,
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

    model = _make_tiny_model()
    idx = _SequenceIndices(["feat"], {"regression_targets": ["feat"],
                                       "classification_targets": [],
                                       "features": ["feat"]})

    # Without regularizer
    result_no_qs = _process_sequence(
        train_tensor, model, h.clone(),
        torch.nn.MSELoss(), torch.nn.BCELoss(), _SumReducer(),
        idx, torch.device("cpu"),
        hurdle_threshold=0.0, qs99_weight=0.0,
    )

    # With regularizer
    result_with_qs = _process_sequence(
        train_tensor, model, h.clone(),
        torch.nn.MSELoss(), torch.nn.BCELoss(), _SumReducer(),
        idx, torch.device("cpu"),
        hurdle_threshold=0.0, qs99_weight=0.1, qs99_tau=0.99,
    )

    loss_no = result_no_qs["total"].item()
    loss_with = result_with_qs["total"].item()
    assert loss_no != loss_with, (
        f"QS99 regularizer should change the total loss: "
        f"without={loss_no}, with={loss_with}"
    )


def test_qs99_disabled_when_no_hurdle():
    """QS99 weight > 0 but hurdle_threshold=None should NOT add regularizer."""
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices

    B, T, C, H, W = 1, 3, 1, 4, 4
    train_tensor = torch.randn(B, T, C, H, W).abs()
    h = torch.zeros(B, 8, H, W)

    model = _make_tiny_model()
    idx = _SequenceIndices(["feat"], {"regression_targets": ["feat"],
                                       "classification_targets": [],
                                       "features": ["feat"]})

    result_a = _process_sequence(
        train_tensor, model, h.clone(),
        torch.nn.MSELoss(), torch.nn.BCELoss(), _SumReducer(),
        idx, torch.device("cpu"),
        hurdle_threshold=None, qs99_weight=0.0,
    )

    result_b = _process_sequence(
        train_tensor, model, h.clone(),
        torch.nn.MSELoss(), torch.nn.BCELoss(), _SumReducer(),
        idx, torch.device("cpu"),
        hurdle_threshold=None, qs99_weight=0.1,
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

    model = _make_tiny_model()
    idx = _SequenceIndices(
        ["feat"],
        {"regression_targets": ["feat"], "classification_targets": [], "features": ["feat"]},
    )

    result_no_hurdle = _process_sequence(
        train_tensor, model, h.clone(),
        torch.nn.MSELoss(), torch.nn.BCELoss(), _SumReducer(),
        idx, torch.device("cpu"), hurdle_threshold=None,
    )

    result_hurdle = _process_sequence(
        train_tensor, model, h.clone(),
        torch.nn.MSELoss(), torch.nn.BCELoss(), _SumReducer(),
        idx, torch.device("cpu"), hurdle_threshold=0.0,
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
class _SumReducer(torch.nn.Module):
    """Mock MultiTaskLoss that sums all losses to a scalar."""

    def forward(self, losses):
        return losses.sum()


def _make_tiny_model():
    """Minimal model that matches HydraNet's forward signature."""
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(1, 1, 1)
            self.base = 8

        def forward(self, x, h):
            out = self.conv(x)
            return out, out, h

        def init_hTtime(self, hidden_channels, H, W):
            return torch.zeros(1, hidden_channels, H, W)

    return TinyModel()
