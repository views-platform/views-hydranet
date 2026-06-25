"""Falsification stubs — gate loss audit (2026-06-23).

Two SOFT falsifications found while auditing the claim "weighted_bce and focal are both
correctly implemented." The loss *math* is correct (focal == torchvision.sigmoid_focal_loss;
weighted_bce == F.binary_cross_entropy_with_logits) and integration is correct (head emits
logits; y_cls is binary). These stubs encode the two hygiene defects in `focal_loss.py`. They
FAIL against current code by design.

Neither affects the 2026-06-22 gate sweep (which used reduction='mean' and α∈{0.25,0.75},
γ∈{1.5,3.0}).
"""

import torch
import torch.nn.functional as F

from views_hydranet.utils.focal_loss import FocalLoss


def test_focal_docstring_bce_equivalence_is_accurate():
    """SOFT (P2): FocalLoss docstring claims it 'reduces to BCE when gamma=0 and alpha=0.5'.

    It does NOT: at alpha=0.5 the alpha_t factor is a constant 0.5, so focal(g=0,a=0.5) = 0.5*BCE.
    True BCE-equivalence requires alpha disabled (alpha<0) and gamma=0. Fix = correct the docstring
    (the computed value is correct vs torchvision; only the comment lies). This test asserts the
    docstring's claim and therefore FAILS until the docstring is fixed (or behavior changed).
    """
    torch.manual_seed(0)
    logits = torch.randn(4, 8, 8)
    targets = (torch.rand(4, 8, 8) > 0.7).float()
    focal_g0_a05 = FocalLoss(alpha=0.5, gamma=0.0)(logits, targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    assert torch.allclose(focal_g0_a05, bce, atol=1e-6), (
        f"docstring claims focal(g=0,a=0.5)==BCE, but got {focal_g0_a05.item():.6f} "
        f"vs BCE {bce.item():.6f} (ratio {focal_g0_a05.item()/bce.item():.3f} — it is 0.5*BCE)"
    )


def test_focal_reduction_none_preserves_input_shape():
    """SOFT (P5): FocalLoss.forward does `logits.unsqueeze(0)` internally, so reduction='none'
    returns shape [1, *input] instead of [*input] — a leaked dim no other loss has (weighted_bce
    preserves shape). Harmless under 'mean'/'sum' (production), but a latent contract inconsistency
    for any per-cell ('none') use. Fix = drop the internal unsqueeze. FAILS until then.
    """
    logits = torch.randn(4, 8, 8)
    targets = (torch.rand(4, 8, 8) > 0.7).float()
    out = FocalLoss(alpha=0.25, gamma=2.0, reduction="none")(logits, targets)
    assert out.shape == logits.shape, (
        f"reduction='none' should preserve input shape {tuple(logits.shape)}, "
        f"got {tuple(out.shape)} (internal unsqueeze(0) leaks a leading dim)"
    )
