"""Restore the emit_family_core wiring (ADR-068/069) so the ZINB π-stripped CORE can be emitted and
externally gated (gated_ZINBcore / th_gated_ZINBcore). The retirement commit (62d19ae) kept
`sample_core` but removed the `to_cube_samples(core=)` param + the `emit_family_core` config flag
(and `mean_core` was never present). These tests pin the restored contract.
"""

from __future__ import annotations

import torch

from views_hydranet.distributions import resolve_family
from views_hydranet.distributions.sampling import to_cube_samples


def _zinb_params(mu, theta, pi):
    return torch.tensor([mu, theta, pi], dtype=torch.float32)


def test_zinb_mean_core_strips_pi():
    """mean_core = mu (bare NB core mean); mean = (1-pi)*mu (self-zeroed). So mean_core > mean."""
    z = resolve_family("zinb")
    p = _zinb_params(5.0, 1.0, 0.9)
    assert torch.allclose(z.mean(p), torch.tensor(0.5), atol=1e-5)  # (1-0.9)*5
    assert torch.allclose(z.mean_core(p), torch.tensor(5.0), atol=1e-5)  # mu, no pi
    assert float(z.mean_core(p)) > float(z.mean(p))


def test_nb_mean_core_equals_mean():
    """nb has no structural zero → core mean == mean (ABC default)."""
    nb = resolve_family("nb")
    p = torch.tensor([3.0, 1.0], dtype=torch.float32)
    assert torch.allclose(nb.mean_core(p), nb.mean(p), atol=1e-6)


def test_zinb_sample_core_drops_structural_zeros():
    """sample_core (NB core) has FAR more nonzeros than sample (which applies the pi zero-mask)."""
    z = resolve_family("zinb")
    p = _zinb_params(5.0, 1.0, 0.9).expand(400, 3).contiguous()
    g = torch.Generator().manual_seed(0)
    core = z.sample_core(p, k=1, generator=g)
    g2 = torch.Generator().manual_seed(0)
    full = z.sample(p, k=1, generator=g2)
    frac_core = float((core > 0).float().mean())
    frac_full = float((full > 0).float().mean())
    assert frac_core > 0.8, frac_core  # NB(5) is almost always > 0
    assert frac_full < 0.3, frac_full  # pi=0.9 zeroes ~90%
    assert frac_core > frac_full + 0.4


def test_to_cube_samples_core_uses_bulk_body():
    """to_cube_samples accepts core=; core=True (bulk) has more mass than core=False (self-zeroed)
    for a high-pi zinb under the same gate."""
    z = resolve_family("zinb")
    T, H, W, n_reg = 1, 6, 6, 1
    # params [T, n_reg*3, H, W] = activated (mu=5, theta=1, pi=0.9) everywhere
    params = torch.zeros(T, n_reg * 3, H, W)
    params[:, 0] = 5.0
    params[:, 1] = 1.0
    params[:, 2] = 0.9
    gate = torch.full((T, H, W, n_reg), 0.9)  # gate fires everywhere
    g = torch.Generator().manual_seed(1)
    core = to_cube_samples(
        params, z, k=8, generator=g, n_reg=n_reg, gate=gate, composition="soft_gate", core=True
    )
    g2 = torch.Generator().manual_seed(1)
    self_z = to_cube_samples(
        params, z, k=8, generator=g2, n_reg=n_reg, gate=gate, composition="soft_gate", core=False
    )
    assert core.shape == (T, H, W, n_reg, 8)
    assert float((core > 0).mean()) > float((self_z > 0).mean())


def test_sample_feedback_is_core_aware_under_emit_family_core():
    """C-234 (S1): _sample_feedback must mirror _emit_magnitude — draw the π-stripped CORE when
    emit_family_core=True, not the self-zeroed body. Otherwise a th_gated_ZINBcore rollout emits
    the large core but feeds back the small self-zeroed history (incoherent AR). Called on a
    minimal stand-in (the method reads only self._family + self.config)."""
    from types import SimpleNamespace

    from views_hydranet.utils.hydranet_inference import HydraNetInference

    z = resolve_family("zinb")
    B, n_reg, H, W = 1, 1, 12, 12
    reg = torch.zeros(B, n_reg * 3, H, W)  # activated zinb params
    reg[:, 0] = 5.0  # mu
    reg[:, 1] = 1.0  # theta
    reg[:, 2] = 0.9  # pi -> self-zeroed sample ~10% nonzero; core sample ~>80%
    prob = torch.ones(B, n_reg, H, W)  # gate fires everywhere (soft_gate ~ passthrough)

    def _feedback(emit_core):
        obj = SimpleNamespace(
            _family=z,
            config={"emit_family_core": emit_core, "forecast_composition": "soft_gate"},
            # production default: independent Bernoulli. The correlated-feedback diagnostic
            # (#258/#262) is opt-in, and this mock must carry the real object's contract.
            _feedback_length_scale=None,
        )
        g = torch.Generator().manual_seed(7)
        out = HydraNetInference._sample_feedback(obj, reg, prob, g)
        return float((out > 0).float().mean())

    frac_core = _feedback(True)  # should feed back the LARGE NB core
    frac_self = _feedback(False)  # self-zeroed body
    # Under emit_family_core the fed-back history must be substantially denser (the core), not the
    # ~90%-zeroed self-zeroed draw.
    assert frac_core > frac_self + 0.4, (frac_core, frac_self)
    assert frac_core > 0.7, frac_core


def test_to_cube_samples_core_self_zeroed_raises():
    """C-240 (S1): core=True + composition='self_zeroed' is invalid (ungated core → zeros nowhere).
    to_cube_samples must fail loud, not silently over-forecast."""
    import pytest

    z = resolve_family("zinb")
    params = torch.zeros(1, 3, 4, 4)
    params[:, 0] = 5.0
    params[:, 1] = 1.0
    params[:, 2] = 0.9
    g = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="external gate|self_zeroed|zero mechanism"):
        to_cube_samples(params, z, k=4, generator=g, n_reg=1, composition="self_zeroed", core=True)
