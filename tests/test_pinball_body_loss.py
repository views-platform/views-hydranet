"""Unit tests for PinballBodyLoss — the bulk-magnitude dial (dossier 2026-07-16, P1).

Verify the numerically-delicate contract: the minimiser is the τ-quantile of the WINSORIZED target,
τ=0.5 ⇒ median, τ>0.5 lifts, the cap winsorizes, and the gradient is finite. This gates the first
run.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _q(x, tau):
    return torch.quantile(x, tau)


def test_minimiser_is_tau_quantile():
    """Pinball at τ is minimised at the τ-quantile of the target (lower than at any shifted
    pred)."""
    from views_hydranet.utils.pinball_body_loss import PinballBodyLoss

    torch.manual_seed(0)
    t = torch.distributions.LogNormal(1.0, 1.0).sample((5000,))
    for tau in (0.5, 0.7, 0.9):
        loss = PinballBodyLoss(tau=tau)
        qt = _q(t, tau)
        at = loss(qt.expand_as(t), t)
        assert at <= loss((qt + 0.5).expand_as(t), t) + 1e-6
        assert at <= loss((qt - 0.5).expand_as(t), t) + 1e-6


def test_tau_lifts_the_optimum():
    """A higher τ has a higher optimal prediction (the dial lifts magnitude)."""
    from views_hydranet.utils.pinball_body_loss import PinballBodyLoss

    torch.manual_seed(1)
    t = torch.distributions.LogNormal(1.0, 1.0).sample((5000,)).requires_grad_(False)

    def fitted(tau):
        p = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([p], lr=0.05)
        for _ in range(400):
            opt.zero_grad()
            PinballBodyLoss(tau=tau)(p.expand_as(t), t).backward()
            opt.step()
        return p.item()

    assert fitted(0.5) < fitted(0.7) < fitted(0.9), "higher τ must fit a higher magnitude"


def test_winsorize_caps_the_target():
    """A target above the cap is treated as the cap (winsorize) — extreme values can't drag the
    fit."""
    from views_hydranet.utils.pinball_body_loss import PinballBodyLoss

    pred = torch.zeros(4)
    spiked = torch.tensor([1.0, 2.0, 3.0, 400.0])  # one extreme above the cap
    capped = torch.tensor([1.0, 2.0, 3.0, 5.0])  # the extreme replaced by the cap
    capped_loss = PinballBodyLoss(tau=0.7, cap=5.0)(pred, spiked)
    uncapped_on_capped = PinballBodyLoss(tau=0.7)(pred, capped)
    assert torch.isclose(capped_loss, uncapped_on_capped), "cap neutralises the extreme"
    # and the cap must actually change the loss vs no cap
    assert capped_loss < PinballBodyLoss(tau=0.7)(pred, spiked), "cap lowers the loss"


def test_gradient_finite_and_tau_half_is_mae_like():
    from views_hydranet.utils.pinball_body_loss import PinballBodyLoss

    torch.manual_seed(2)
    t = torch.rand(100) * 5
    p = torch.zeros(100, requires_grad=True)
    loss = PinballBodyLoss(tau=0.5)(p, t)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all()
    # τ=0.5 ⇒ 0.5·|err| (MAE/2)
    assert torch.isclose(loss, 0.5 * (t - 0).abs().mean(), atol=1e-6)


def test_invalid_tau_rejected():
    from views_hydranet.utils.pinball_body_loss import PinballBodyLoss

    for bad in (0.0, 1.0, 1.5, -0.1):
        with pytest.raises(ValueError):
            PinballBodyLoss(tau=bad)


def test_config_accepts_pinball(valid_config_dict):
    """HydraNetConfig accepts loss_reg='pinball' with τ + cap."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    cfg = dict(valid_config_dict)
    cfg.update({"loss_reg": "pinball", "loss_reg_tau": 0.7, "loss_reg_cap": 5.7})
    c = HydraNetConfig(**cfg)
    assert c.loss_reg == "pinball" and c.loss_reg_tau == 0.7 and c.loss_reg_cap == 5.7
