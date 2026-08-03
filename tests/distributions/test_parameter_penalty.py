"""A-S9 (#176): the `parameter_penalty` ABC hook (C-205) — a default-0 family-agnostic penalty.

`DistributionFamily.parameter_penalty` returns a graph-safe 0 by default so a family-agnostic loop
can add a per-cell parameter regularizer without `isinstance`-branching to a concrete family.
`NegativeBinomialFamily` inherits the 0; `ZINBFamily` overrides it to its π-ridge (`pi_penalty`,
the C-200 deep-zero regularizer).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.distributions import resolve_family  # noqa: E402


def test_nb_parameter_penalty_is_zero():
    fam = resolve_family("nb")
    params = fam.activate(torch.randn(4, 5, 2))
    pen = fam.parameter_penalty(params, prior_logit=0.3, scale=1.0)
    assert pen.shape == ()
    assert float(pen) == 0.0


def test_zinb_parameter_penalty_equals_pi_penalty():
    fam = resolve_family("zinb")
    params = fam.activate(torch.randn(4, 5, 3))
    a = fam.parameter_penalty(params, prior_logit=0.5, scale=2.0)
    b = fam.pi_penalty(params, prior_logit=0.5, scale=2.0)
    assert torch.equal(a, b)


def test_zinb_parameter_penalty_nonneg_and_scales():
    fam = resolve_family("zinb")
    params = fam.activate(torch.randn(4, 5, 3))
    p1 = fam.parameter_penalty(params, prior_logit=0.0, scale=1.0)
    p2 = fam.parameter_penalty(params, prior_logit=0.0, scale=2.0)
    assert float(p1) >= 0.0
    assert torch.allclose(p2, 2.0 * p1)


def test_parameter_penalty_default_scale_zero_is_zero():
    # the ABC default signature defaults scale=0 -> 0 even for zinb
    fam = resolve_family("zinb")
    params = fam.activate(torch.randn(2, 3))
    assert float(fam.parameter_penalty(params)) == 0.0
