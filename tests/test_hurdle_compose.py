"""Generic hurdle compose (lognormal + point bodies) — emit = log1p(count-space E[y]).

The contract (same as hurdle-NB): _emit_magnitude returns log1p(E[y]) so the downstream
inverse_transform (expm1) recovers E[y] in count space — never expm1 a free prediction (C-140).

Bodies are trained in log1p space on positive cells (body_mask='pos_cells'):
  - lognormal: log1p(y) ~ N(reg, sigma^2)  ⇒  E[y|y>0] = expm1(reg + sigma^2/2)
  - point/shrinkage: reg is the log1p-space point  ⇒  E[y|y>0] = expm1(reg)
Hurdle: E[y] = P(y>0) · E[y|y>0].
"""

import math

import pytest
import torch
import torch.nn as nn

from views_hydranet.utils.hurdle_nb import (
    hurdle_lognormal_expected_log1p,
    hurdle_point_expected_log1p,
)
from views_hydranet.utils.hydranet_inference import HydraNetInference


# ---- pure compose functions ----
def test_lognormal_known_value_and_roundtrip():
    reg, prob, sigma = (torch.tensor([[[[0.5]]]]) for _ in range(3))
    sigma = torch.tensor([[[[0.9]]]])
    out = hurdle_lognormal_expected_log1p(reg, prob, sigma)
    ey = 0.5 * math.expm1(0.5 + 0.5 * 0.9**2)  # prob=0.5 here (reg/prob both 0.5 above)
    # rebuild with explicit prob to avoid confusion
    out = hurdle_lognormal_expected_log1p(
        torch.tensor([[[[0.5]]]]), torch.tensor([[[[0.8]]]]), torch.tensor([[[[0.9]]]])
    )
    ey = 0.8 * math.expm1(0.5 + 0.5 * 0.9**2)
    assert torch.allclose(torch.expm1(out), torch.tensor([[[[ey]]]]), atol=1e-5)


def test_point_known_value_and_roundtrip():
    out = hurdle_point_expected_log1p(torch.tensor([[[[2.0]]]]), torch.tensor([[[[0.5]]]]))
    ey = 0.5 * math.expm1(2.0)
    assert torch.allclose(torch.expm1(out), torch.tensor([[[[ey]]]]), atol=1e-5)


def test_composes_finite_at_small_reg():
    for fn, args in [
        (
            hurdle_lognormal_expected_log1p,
            (torch.tensor([[[[1e-6]]]]), torch.tensor([[[[0.4]]]]), torch.tensor([[[[0.9]]]])),
        ),
        (hurdle_point_expected_log1p, (torch.tensor([[[[1e-6]]]]), torch.tensor([[[[0.4]]]]))),
    ]:
        out = fn(*args)
        assert torch.isfinite(out).all() and (out >= 0).all()


# ---- dispatch through HydraNetInference._emit_magnitude ----
class _Mock(nn.Module):
    def __init__(self, output_distribution, sigma=None):
        super().__init__()
        self.output_distribution = output_distribution
        self.hurdle_nb_theta = None
        self.hurdle_lognormal_sigma = sigma

    def forward(self, x, h):  # pragma: no cover
        raise NotImplementedError


def _inf(output_distribution, sigma=None, targets=("lr_ged_sb",)):
    return HydraNetInference(
        _Mock(output_distribution, sigma), {"regression_targets": list(targets)}, device="cpu"
    )


def test_dispatch_hurdle_lognormal():
    inf = _inf("hurdle_lognormal", sigma=0.9)
    out = inf._emit_magnitude(torch.tensor([[[[0.5]]]]), torch.tensor([[[[0.8]]]]))
    ey = 0.8 * math.expm1(0.5 + 0.5 * 0.9**2)
    assert torch.allclose(torch.expm1(out), torch.tensor([[[[ey]]]]), atol=1e-5)


def test_dispatch_hurdle_shrinkage():
    inf = _inf("hurdle_shrinkage")
    out = inf._emit_magnitude(torch.tensor([[[[2.0]]]]), torch.tensor([[[[0.5]]]]))
    ey = 0.5 * math.expm1(2.0)
    assert torch.allclose(torch.expm1(out), torch.tensor([[[[ey]]]]), atol=1e-5)


def test_dispatch_dense_nb_emits_mu():
    # Dense (non-truncated, NO-gate) NB body: E[y] = mu, the raw count-space softplus output (reg).
    # The gate prob is IGNORED (dense has no hurdle). emit = log1p(mu) so downstream expm1 -> mu.
    inf = _inf("dense_nb")
    mu = torch.tensor([[[[7.0]]]])
    out = inf._emit_magnitude(mu, torch.tensor([[[[0.3]]]]))
    assert torch.allclose(torch.expm1(out), mu, atol=1e-5)
    # prob-invariance: dense mean does not depend on the gate
    out2 = inf._emit_magnitude(mu, torch.tensor([[[[0.95]]]]))
    assert torch.allclose(out, out2)


def test_hurdle_lognormal_requires_sigma():
    with pytest.raises(ValueError, match="sigma"):
        _inf("hurdle_lognormal", sigma=None)


def test_overflow_guard_keeps_emit_finite_at_huge_reg():
    # Pathological rollout reg would overflow expm1->inf (EvaluationFrame then rejects the frame).
    # The guard caps the expm1 argument so the emit stays FINITE; T=0-range values are untouched.
    huge = torch.tensor([[[[800.0]]]])
    p = sig = torch.tensor([[[[0.9]]]])
    assert torch.isfinite(hurdle_lognormal_expected_log1p(huge, p, sig)).all()
    assert torch.isfinite(hurdle_point_expected_log1p(huge, p)).all()
    # a realistic value is unchanged by the guard (arg << ceiling)
    small = hurdle_point_expected_log1p(torch.tensor([[[[2.0]]]]), torch.tensor([[[[0.5]]]]))
    exp = torch.tensor([[[[0.5 * math.expm1(2.0)]]]])
    assert torch.allclose(torch.expm1(small), exp, atol=1e-5)
