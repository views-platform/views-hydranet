"""Gate loss regression tests — the two focal hygiene defects (gate audit 2026-06-23), now FIXED.

Two SOFT falsifications found while auditing "weighted_bce and focal are both correctly
implemented." The loss *math* was always correct (focal == torchvision.sigmoid_focal_loss;
weighted_bce == F.binary_cross_entropy_with_logits); the defects were hygiene: (P2) the docstring
claimed BCE-equivalence at the wrong params, and (P5) an internal `unsqueeze(0)` leaked a leading
dim under reduction='none'. Both fixed in `focal_loss.py` (tech-debt-cleanup); these tests now pin
the CORRECT contract.
"""

import torch
import torch.nn.functional as F

from views_hydranet.utils.focal_loss import FocalLoss


def test_focal_bce_equivalence_contract():
    """P2 (fixed): the true BCE relationship the corrected docstring states.

    At alpha=0.5 the alpha_t factor is the constant 0.5, so focal(g=0, a=0.5) = 0.5*BCE (NOT BCE).
    True BCE-equivalence requires alpha DISABLED (alpha<0) with gamma=0. Both pinned here.
    """
    torch.manual_seed(0)
    logits = torch.randn(4, 8, 8)
    targets = (torch.rand(4, 8, 8) > 0.7).float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    focal_g0_a05 = FocalLoss(alpha=0.5, gamma=0.0)(logits, targets)
    focal_g0_alpha_off = FocalLoss(alpha=-1.0, gamma=0.0)(logits, targets)
    assert torch.allclose(focal_g0_a05, 0.5 * bce, atol=1e-6), (
        f"focal(g=0,a=0.5) should be 0.5*BCE, got {focal_g0_a05.item():.6f} vs "
        f"{(0.5 * bce).item():.6f}"
    )
    assert torch.allclose(focal_g0_alpha_off, bce, atol=1e-6), (
        f"focal(g=0, alpha disabled) should equal BCE, got {focal_g0_alpha_off.item():.6f} "
        f"vs {bce.item():.6f}"
    )


def test_focal_reduction_none_preserves_input_shape():
    """P5 (fixed): reduction='none' preserves the input shape (the internal unsqueeze(0) that
    leaked a leading dim is gone), matching every other loss (weighted_bce)."""
    logits = torch.randn(4, 8, 8)
    targets = (torch.rand(4, 8, 8) > 0.7).float()
    out = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")(logits, targets)
    assert out.shape == logits.shape, (
        f"reduction='none' should preserve input shape {tuple(logits.shape)}, "
        f"got {tuple(out.shape)}"
    )
