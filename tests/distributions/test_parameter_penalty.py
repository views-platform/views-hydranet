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
