"""Tests for WeightedBCEWithLogitsLoss — the hurdle-NB onset gate (#99)."""

import pytest
import torch
import torch.nn as nn

from views_hydranet.utils.weighted_bce_loss import WeightedBCEWithLogitsLoss


def test_reduces_to_plain_bce_when_pos_weight_none():
    logits = torch.tensor([-2.0, 0.0, 1.5, 3.0])
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    ours = WeightedBCEWithLogitsLoss(pos_weight=None)(logits, targets)
    ref = nn.BCEWithLogitsLoss()(logits, targets)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_consumes_logits_not_probs():
    # If it treated inputs as probabilities it would differ from BCEWithLogits on logits.
    logits = torch.tensor([4.0, -4.0])
    targets = torch.tensor([1.0, 0.0])
    ours = WeightedBCEWithLogitsLoss()(logits, targets)
    ref = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(ours, ref, atol=1e-6)


def test_pos_weight_upweights_positive_class():
    logits = torch.tensor([0.0])
    target_pos = torch.tensor([1.0])
    base = WeightedBCEWithLogitsLoss(pos_weight=1.0)(logits, target_pos)
    heavy = WeightedBCEWithLogitsLoss(pos_weight=10.0)(logits, target_pos)
    assert (heavy > base).item()


def test_invalid_pos_weight_raises():
    with pytest.raises(ValueError, match="pos_weight must be > 0"):
        WeightedBCEWithLogitsLoss(pos_weight=0.0)


def test_gradients_finite():
    logits = torch.randn(8, requires_grad=True)
    targets = (torch.rand(8) > 0.5).float()
    out = WeightedBCEWithLogitsLoss(pos_weight=5.0)(logits, targets)
    out.backward()
    assert torch.isfinite(out)
    assert torch.isfinite(logits.grad).all()
