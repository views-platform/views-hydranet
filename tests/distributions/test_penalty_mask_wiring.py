"""A-S9 (#176): training wiring of the π-ridge penalty (C-200) + family `weight=` mask (C-199).

- C-200: `_process_sequence` adds `pi_penalty_weight * family.parameter_penalty(...)` to the loss
  (qs99/decay additive-penalty precedent). weight None/0 ⇒ no-op; weight>0 ⇒ the loss
  changes and grads still reach the reg head.
- C-199: the family body-mask path passes `weight=mask` into `family.nll` instead of boolean-index
  `pred_j[mask]` — numerically identical for a 0/1 mask, but a graph-connected 0 on an empty mask.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.distributions import resolve_family  # noqa: E402
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402


def _run(output_distribution, **kwargs):
    from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
    from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices
    from views_hydranet.utils.focal_loss import FocalLoss
    from views_hydranet.utils.mtloss import MultiTaskLoss

    torch.manual_seed(0)
    model = HydraBNUNet06_LSTM4(3, 16, 1, 0.0, output_distribution=output_distribution).float()
    feature_names = ["lr_sb", "lr_ns", "lr_os", "by_sb", "by_ns", "by_os"]
    config = {
        "regression_targets": ["lr_sb", "lr_ns", "lr_os"],
        "classification_targets": ["by_sb", "by_ns", "by_os"],
        "features": ["lr_sb", "lr_ns", "lr_os"],
    }
    idx = _SequenceIndices(feature_names, config)
    b, t, c, hh, ww = 1, 4, 6, 8, 8
    train_tensor = torch.rand(b, t, c, hh, ww).abs()
    h = model.init_hTtime(hidden_channels=model.base, H=hh, W=ww)
    criterion_reg = FamilyLoss(resolve_family(output_distribution))
    criterion_class = FocalLoss(alpha=0.75, gamma=1.5)
    mt = MultiTaskLoss(torch.Tensor([True, True, True, False, False, False]), reduction="sum")
    result = _process_sequence(
        train_tensor,
        model,
        h,
        criterion_reg,
        criterion_class,
        mt,
        idx,
        torch.device("cpu"),
        **kwargs,
    )
    return model, result


# ── C-200: π-ridge additive penalty ──


def test_pi_penalty_weight_none_is_noop():
    _, base = _run("zinb")
    _, off = _run("zinb", pi_penalty_weight=None)
    assert torch.equal(base["total"], off["total"])


def test_pi_penalty_changes_loss_and_backprops():
    _, off = _run("zinb", pi_penalty_weight=0.0)
    model, on = _run("zinb", pi_penalty_weight=1.0, pi_penalty_prior_logit=2.0)
    assert on["total"].isfinite()
    assert not torch.equal(on["total"], off["total"])  # the ridge shifted the loss
    on["total"].backward()
    g = model.dec_conv4_head1_reg.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_pi_penalty_noop_for_nb_family():
    # nb's parameter_penalty is 0, so a weight>0 must not change an nb loss
    _, off = _run("nb", pi_penalty_weight=0.0)
    _, on = _run("nb", pi_penalty_weight=5.0, pi_penalty_prior_logit=1.0)
    assert torch.equal(on["total"], off["total"])


# ── C-199: family masking via weight= equals boolean-index ──


def test_family_weight_mask_equals_boolean_index():
    fam = resolve_family("nb")
    pred = fam.activate(torch.randn(6, 6, 2))
    target = torch.log1p(torch.rand(6, 6) * 10)
    mask = torch.rand(6, 6) > 0.5
    via_weight = fam.nll(pred, target, weight=mask.to(pred.dtype))
    via_index = fam.nll(pred[mask], target[mask])
    assert torch.allclose(via_weight, via_index, atol=1e-6)


def test_family_empty_mask_is_graph_connected_zero():
    fam = resolve_family("nb")
    raw = torch.randn(4, 4, 2, requires_grad=True)
    pred = fam.activate(raw)
    target = torch.log1p(torch.rand(4, 4) * 10)
    zero_mask = torch.zeros(4, 4)
    loss = fam.nll(pred, target, weight=zero_mask)
    assert float(loss) == 0.0
    loss.backward()  # graph-connected 0 -> no error (unlike a detached tensor(0.0))
    assert raw.grad is not None  # leaf gradient populated (graph stayed connected)


def test_family_body_mask_step_runs_finite():
    # exercise the training-engine apply_body_mask branch with a family (weight= path)
    model, result = _run("nb", body_mask="pos_cells", event_threshold=0.0)
    assert result["total"].isfinite()
    result["total"].backward()
    g = model.dec_conv4_head1_reg.weight.grad
    assert g is not None and torch.isfinite(g).all()
