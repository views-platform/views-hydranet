"""A-S2 (#169): the DistributionFamily ABC contract (ADR-067 §2)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def test_distribution_family_is_abstract():
    from views_hydranet.distributions.base import DistributionFamily

    with pytest.raises(TypeError):
        DistributionFamily()  # abstract — all seven members must be overridden


def test_minimal_concrete_subclass_satisfies_interface():
    from views_hydranet.distributions.base import DistributionFamily

    class _Toy(DistributionFamily):
        n_params = 1  # a class attr satisfies the abstract property

        def activate(self, raw):
            return torch.nn.functional.softplus(raw)

        def nll(self, params, target, *, weight=None):
            return params.sum() * 0.0

        def sample(self, params, k, generator=None):
            return params[..., :1].expand(*params.shape[:-1], k)

        def mean(self, params):
            return params[..., 0]

        def prob_positive(self, params):
            return torch.zeros(params.shape[:-1])

        def initial_raw_bias(self, *, priors=None):
            return torch.zeros(self.n_params)

    fam = _Toy()
    assert fam.n_params == 1
    assert fam.needs_latent is False
    x = torch.randn(2, 3, 1)
    assert fam.activate(x).shape == x.shape
    assert fam.sample(x, 5).shape == (2, 3, 5)
    assert fam.mean(x).shape == (2, 3)
    assert fam.prob_positive(x).shape == (2, 3)
