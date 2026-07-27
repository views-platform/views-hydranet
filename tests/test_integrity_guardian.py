"""
Tests for IntegrityGuardian: numerical stability monitor.

Green/Beige/Red taxonomy (ADR-005).
"""

import pytest
import torch
import torch.nn as nn

from views_hydranet.utils.integrity_guardian import IntegrityGuardian


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_model():
    """A minimal model with known finite weights."""
    m = nn.Linear(4, 4)
    nn.init.ones_(m.weight)
    nn.init.zeros_(m.bias)
    return m


@pytest.fixture
def healthy_prediction():
    return torch.randn(1, 1, 4, 4)


@pytest.fixture
def healthy_loss():
    return torch.tensor(1.5)


# ---------------------------------------------------------------------------
# GREEN TEAM — happy path
# ---------------------------------------------------------------------------
class TestGreen:
    def test_green_monitor_passes_healthy(self, tiny_model, healthy_prediction, healthy_loss):
        """Finite loss + small preds + finite grads -> no raise."""
        # Generate gradients
        out = tiny_model(torch.randn(1, 4))
        loss = out.sum()
        loss.backward()

        IntegrityGuardian.monitor(tiny_model, healthy_prediction, healthy_loss)


# ---------------------------------------------------------------------------
# BEIGE TEAM — boundary & robustness
# ---------------------------------------------------------------------------
class TestBeige:
    def test_beige_prediction_at_boundary(self, tiny_model, healthy_loss):
        """Prediction magnitude just under ceiling -> no raise."""
        pred = torch.full((1, 1, 4, 4), 999.0)
        IntegrityGuardian.monitor(tiny_model, pred, healthy_loss)

    def test_beige_prediction_just_over(self, tiny_model, healthy_loss):
        """Prediction magnitude just over ceiling -> RuntimeError."""
        pred = torch.full((1, 1, 4, 4), 1001.0)
        with pytest.raises(RuntimeError, match="EXPLOSION"):
            IntegrityGuardian.monitor(tiny_model, pred, healthy_loss)

    def test_beige_prediction_at_1500_raises(self, tiny_model, healthy_loss):
        """C-51: predictions at 1500 must trigger halt (was silent before)."""
        pred = torch.full((1, 1, 4, 4), 1500.0)
        with pytest.raises(RuntimeError, match="EXPLOSION"):
            IntegrityGuardian.monitor(tiny_model, pred, healthy_loss)

    def test_beige_no_gradients(self, tiny_model, healthy_prediction, healthy_loss):
        """Params with grad=None -> no raise (no backward called)."""
        assert all(p.grad is None for p in tiny_model.parameters())
        IntegrityGuardian.monitor(tiny_model, healthy_prediction, healthy_loss)

    def test_beige_empty_model(self, healthy_prediction, healthy_loss):
        """nn.Module() with no parameters -> no raise."""
        empty = nn.Module()
        IntegrityGuardian.monitor(empty, healthy_prediction, healthy_loss)


# ---------------------------------------------------------------------------
# RED TEAM — failure detection
# ---------------------------------------------------------------------------
class TestRed:
    def test_red_nan_loss(self, tiny_model, healthy_prediction):
        loss = torch.tensor(float("nan"))
        with pytest.raises(RuntimeError, match="Loss"):
            IntegrityGuardian.monitor(tiny_model, healthy_prediction, loss)

    def test_red_inf_loss(self, tiny_model, healthy_prediction):
        loss = torch.tensor(float("inf"))
        with pytest.raises(RuntimeError, match="Loss"):
            IntegrityGuardian.monitor(tiny_model, healthy_prediction, loss)

    def test_red_nan_prediction(self, tiny_model, healthy_loss):
        pred = torch.tensor([[[float("nan")]]])
        with pytest.raises(RuntimeError, match="EXPLOSION"):
            IntegrityGuardian.monitor(tiny_model, pred, healthy_loss)

    def test_red_inf_gradient(self, tiny_model, healthy_prediction, healthy_loss):
        # Manually inject Inf into gradients
        for p in tiny_model.parameters():
            p.grad = torch.full_like(p, float("inf"))
            break  # corrupt just one
        with pytest.raises(RuntimeError, match="GRADIENT"):
            IntegrityGuardian.monitor(tiny_model, healthy_prediction, healthy_loss)

    def test_red_nan_weight(self, tiny_model, healthy_prediction, healthy_loss):
        """A NaN in the WEIGHTS themselves (not the gradient) is caught at the source — before it
        corrupts the next forward. Backs the CIC 'Weight NaN/Inf -> RuntimeError' guarantee."""
        with torch.no_grad():
            tiny_model.weight[0, 0] = float("nan")
        with pytest.raises(RuntimeError, match="WEIGHT"):
            IntegrityGuardian.monitor(tiny_model, healthy_prediction, healthy_loss)

    def test_red_inf_weight(self, tiny_model, healthy_prediction, healthy_loss):
        """An Inf weight is caught with the FATAL WEIGHT CORRUPTION message."""
        with torch.no_grad():
            tiny_model.bias[2] = float("inf")
        with pytest.raises(RuntimeError, match="WEIGHT"):
            IntegrityGuardian.monitor(tiny_model, healthy_prediction, healthy_loss)
