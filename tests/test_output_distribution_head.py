"""Tests for the #100 output_distribution head flag.

Default "standard" => ReLU (byte-identical to pre-#100); "hurdle_nb" => softplus (count-space mu).
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


def test_default_is_standard_relu():
    m = _build(None)
    assert m.output_distribution == "standard"
    assert m._reg_activation is F.relu
    out = _fwd(m)
    assert torch.allclose(out.reg, F.relu(out.reg_latent))
    assert (out.reg >= 0).all()


def test_explicit_standard_equals_default():
    assert _build("standard")._reg_activation is _build(None)._reg_activation is F.relu


def test_hurdle_nb_is_softplus():
    m = _build("hurdle_nb")
    assert m._reg_activation is F.softplus
    out = _fwd(m)
    assert torch.allclose(out.reg, F.softplus(out.reg_latent))
    assert (out.reg > 0).all()  # softplus is strictly positive


def test_standard_vs_hurdle_differ_on_negative_latent():
    # Same seed => identical weights => identical latent for the same input.
    o_std = _fwd(_build("standard", seed=3))
    o_hnb = _fwd(_build("hurdle_nb", seed=3))
    assert torch.allclose(o_std.reg_latent, o_hnb.reg_latent)
    neg = o_std.reg_latent < 0
    assert neg.any(), "expected some negative latents to distinguish relu vs softplus"
    assert (o_std.reg[neg] == 0).all()  # relu zeros them
    assert (o_hnb.reg[neg] > 0).all()  # softplus keeps them positive


def test_config_field_default_is_standard():
    assert HydraNetConfig.model_fields["output_distribution"].default == "standard"
