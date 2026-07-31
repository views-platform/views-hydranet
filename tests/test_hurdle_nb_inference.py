"""Tests for #101 — hurdle-NB inference compose + artifact persistence (reload threading).

The exact zero-truncated hurdle mean: E[y] = P(y>0) * mu / (1 - NB0(mu, theta)); emitted as
log1p(E[y]) so the downstream inverse_transform (expm1) recovers E[y] in count space (C-140).
"""

import math

import pytest
import torch
import torch.nn as nn

from views_hydranet.utils.hydranet_inference import HydraNetInference


class _MockModel(nn.Module):
    """Minimal stand-in carrying the attributes inference reads (output_distribution + theta)."""

    def __init__(self, output_distribution="standard", hurdle_nb_theta=None):
        super().__init__()
        self.output_distribution = output_distribution
        self.hurdle_nb_theta = hurdle_nb_theta

    def forward(self, x, h):  # pragma: no cover - not exercised by these unit tests
        raise NotImplementedError


def _inf(output_distribution="standard", theta=None, targets=("lr_ged_sb",)):
    model = _MockModel(output_distribution, theta)
    return HydraNetInference(model, {"regression_targets": list(targets)}, device="cpu")


def test_emit_standard_is_identity():
    inf = _inf("standard")
    reg = torch.tensor([[[[2.0]]]])
    assert torch.equal(inf._emit_magnitude(reg, torch.tensor([[[[0.5]]]])), reg)


def test_emit_hurdle_nb_known_value():
    # mu=2, p=0.5, theta=1 -> NB0=1/3 -> E[y]=0.5*2/(2/3)=1.5 -> emit=log1p(1.5)
    inf = _inf("hurdle_nb", {"lr_ged_sb": 1.0})
    out = inf._emit_magnitude(torch.tensor([[[[2.0]]]]), torch.tensor([[[[0.5]]]]))
    assert torch.allclose(out, torch.tensor([[[[math.log1p(1.5)]]]]), atol=1e-5)


def test_emit_roundtrip_recovers_e_y_no_double_expm1():
    inf = _inf("hurdle_nb", {"lr_ged_sb": 1.0})
    out = inf._emit_magnitude(torch.tensor([[[[2.0]]]]), torch.tensor([[[[0.5]]]]))
    assert torch.allclose(torch.expm1(out), torch.tensor([[[[1.5]]]]), atol=1e-5)


def test_emit_small_mu_is_finite_and_tends_to_p():
    # E[y] -> p as mu -> 0 (truncated body mean -> 1); must be finite (no 0/0 blow-up).
    inf = _inf("hurdle_nb", {"lr_ged_sb": 1.0})
    out = inf._emit_magnitude(torch.tensor([[[[1e-6]]]]), torch.tensor([[[[0.4]]]]))
    assert torch.isfinite(out).all()
    assert torch.allclose(torch.expm1(out), torch.tensor([[[[0.4]]]]), atol=1e-2)


def test_hurdle_nb_requires_theta():
    with pytest.raises(ValueError, match="requires per-target theta"):
        _inf("hurdle_nb", theta=None)


def test_reload_threads_output_distribution_via_choose_model():
    # The confirmed reload bug: output_distribution must survive the sidecar -> choose_model path.
    from views_hydranet.utils.utils import choose_model

    cfg = {
        "model": "HydraBNUNet06_LSTM4",
        "input_channels": 8,
        "total_hidden_channels": 32,
        "output_channels": 1,
        "dropout_rate": 0.0,
        "output_distribution": "hurdle_nb",
    }
    m = choose_model(cfg, torch.device("cpu"))
    assert m.output_distribution == "hurdle_nb"
