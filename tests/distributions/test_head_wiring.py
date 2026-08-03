"""A-S6 (#173): architecture head wiring via resolve_family (ADR-067 strangler-fig).

The regression head sizes reg channels to `family.n_params` and installs `family.activate` when
`output_distribution` is a registered family (nb/zinb); legacy values fall through to the untouched
`.startswith("hurdle")` / `_is_quantile` path (byte-identical). `output_channels`/`cls` (the AR
feedback width) is unchanged.
"""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from views_hydranet.architectures import HydraBNrecurrentUnet_06_LSTM4 as arch_mod  # noqa: E402
from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (  # noqa: E402
    HydraBNUNet06_LSTM4,
)

H = W = 16
IN_CH, HID_CH = 8, 32


def _build(output_distribution, seed=7, **kw):
    torch.manual_seed(seed)
    m = HydraBNUNet06_LSTM4(IN_CH, HID_CH, 1, 0.0, output_distribution=output_distribution, **kw)
    return m.float().train()  # train() => reg_latent populated


def _fwd(m, seed=11):
    torch.manual_seed(seed)
    x = torch.randn(1, IN_CH, H, W).float()
    h = m.init_hTtime(HID_CH, H, W).float()
    return m(x, h)


# ── family head width: 3 targets × n_params reg channels; cls (feedback) stays 3 ──


@pytest.mark.parametrize("fam,n_params", [("nb", 2), ("zinb", 3)])
def test_family_head_reg_width_is_targets_times_n_params(fam, n_params):
    out = _fwd(_build(fam))
    assert out.reg.shape[1] == 3 * n_params  # 3 targets × n_params
    assert out.cls.shape[1] == 3  # classification / AR-feedback width unchanged


# ── family activation installed (softplus mu/theta > 0; zinb pi in (0,1)) ──


def test_nb_activation_is_family_activate():
    from views_hydranet.distributions import resolve_family

    m = _build("nb")
    out = _fwd(m)
    fam = resolve_family("nb")
    # per target head: [B, n_params, H, W]; family.activate expects params in the last dim
    head0 = out.reg[:, 0:2]  # first target's (mu, theta)
    latent0 = out.reg_latent[:, 0:2]
    expected = fam.activate(latent0.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
    assert torch.allclose(head0, expected, atol=1e-6)
    assert (out.reg > 0).all()  # softplus everywhere for nb (mu, theta)


def test_zinb_pi_channel_in_unit_interval():
    m = _build("zinb")
    out = _fwd(m)
    pi = out.reg[:, 2::3]  # every 3rd channel (per target) is pi = sigmoid(...)
    assert ((pi > 0) & (pi < 1)).all()


# ── legacy heads byte-identical (family branch is skipped for None) ──


def test_standard_head_unchanged_relu_width_3():
    m = _build("standard")
    assert m._reg_activation is F.relu
    out = _fwd(m)
    assert out.reg.shape[1] == 3  # reg_out_ch == output_channels == 1, ×3 targets
    assert torch.allclose(out.reg, F.relu(out.reg_latent))


def test_hurdle_nb_head_unchanged_softplus():
    m = _build("hurdle_nb")
    assert m._reg_activation is F.softplus
    out = _fwd(m)
    assert out.reg.shape[1] == 3


def test_dense_nb_head_unchanged_relu():
    m = _build("dense_nb")
    assert m._reg_activation is F.relu  # dense_nb does NOT start with "hurdle"


def test_quantile_head_unchanged_width_and_monotone():
    k = 5
    out = _fwd(_build("quantile", n_quantiles=k))
    assert out.reg.shape[1] == 3 * k
    # monotone non-decreasing along each target's quantile channels
    q = out.reg[:, 0:k]
    assert (q[:, 1:] - q[:, :-1] >= -1e-6).all()


# ── informed init: the family's initial_raw_bias seeds the reg conv biases ──


def test_family_reg_bias_is_initial_raw_bias():
    from views_hydranet.distributions import resolve_family

    for fam_name in ("nb", "zinb"):
        m = _build(fam_name)
        fam = resolve_family(fam_name)
        expected = fam.initial_raw_bias()
        for conv in (m.dec_conv4_head1_reg, m.dec_conv4_head2_reg, m.dec_conv4_head3_reg):
            assert torch.allclose(conv.bias.detach(), expected, atol=1e-6)
        # activates to the intended priors: theta ~ 1.0 (channel 1)
        theta = fam.activate(expected.view(1, -1))[0, 1]
        assert theta.item() == pytest.approx(1.0, rel=1e-4)


def test_family_head_is_differentiable():
    """Gradient must flow through family.activate to the reg conv weights, not just the bias."""
    for fam_name in ("nb", "zinb"):
        m = _build(fam_name)
        out = _fwd(m)
        out.reg.sum().backward()
        for conv in (m.dec_conv4_head1_reg, m.dec_conv4_head2_reg, m.dec_conv4_head3_reg):
            assert conv.weight.grad is not None and torch.isfinite(conv.weight.grad).all()
            assert conv.weight.grad.abs().sum() > 0, f"{fam_name}: dead reg-conv gradient"


# ── dispatch is via resolve_family, not new hardcoded family names ──


def test_head_source_has_no_hardcoded_family_names():
    src = inspect.getsource(arch_mod)
    assert '"nb"' not in src and "'nb'" not in src
    assert '"zinb"' not in src and "'zinb'" not in src


def test_legacy_values_resolve_to_none():
    from views_hydranet.distributions import resolve_family

    legacy_values = (
        "standard",
        "hurdle_nb",
        "hurdle_lognormal",
        "hurdle_shrinkage",
        "dense_nb",
        "quantile",
    )
    for legacy in legacy_values:
        assert resolve_family(legacy) is None
