"""Tests for the #100 output_distribution head flag.

Default "standard" => softplus (ReLU until #632, 2026-11-20).
Parity is verified against the captured pre-activation latent (reg_latent, training-only).
"""

import torch
import torch.nn.functional as F

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
from views_hydranet.utils.config_initializer import HydraNetConfig

H = W = 16
IN_CH, HID_CH = 8, 32


def _build(output_distribution=None, seed=7):
    torch.manual_seed(seed)
    if output_distribution is None:
        m = HydraBNUNet06_LSTM4(IN_CH, HID_CH, 1, 0.0)  # default arg
    else:
        m = HydraBNUNet06_LSTM4(IN_CH, HID_CH, 1, 0.0, output_distribution=output_distribution)
    return m.float().train()  # train() => reg_latent populated


def _fwd(m, seed=11):
    torch.manual_seed(seed)
    x = torch.randn(1, IN_CH, H, W).float()
    h = m.init_hTtime(HID_CH, H, W).float()
    return m(x, h)


def test_default_is_standard_softplus():
    m = _build(None)
    assert m.output_distribution == "standard"
    assert m._reg_activation is F.softplus
    out = _fwd(m)
    assert torch.allclose(out.reg, F.softplus(out.reg_latent))
    assert (out.reg > 0).all()


def test_explicit_standard_equals_default():
    assert _build("standard")._reg_activation is _build(None)._reg_activation is F.softplus


def test_the_config_has_no_default_body():
    """#633 (ADR-081): a config must state its body; there is no default to fall back to."""
    assert "output_distribution" not in HydraNetConfig.model_fields
    assert HydraNetConfig.model_fields["body"].is_required()
