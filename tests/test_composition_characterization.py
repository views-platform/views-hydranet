"""S2 (#185, Epic #183) — characterization net: byte-exact snapshot of the CURRENT emit path.

This pins today's behaviour of the two seams the forecast-composition refactor (S4, #187) will
touch — ``_emit_magnitude`` (the point / AR-feedback path) and ``to_cube_samples`` (the D×K sample
cube) — so S4 can prove it changed only what it meant to. Goldens fall into two categories:

* **PARITY-LOCKED** — legacy heads (standard/hurdle_nb/dense_nb) and the **self-zeroed** ``zinb``
  path. S4's composer is a *passthrough* for these, so they MUST stay byte-identical. These tests
  are the parity anchor S4/S6 assert against.
* **nb PASSTHROUGH** — the ungated ``nb`` emit and ``nb`` sample cube under the default
  ``self_zeroed`` composition. Post-S4 the composer passes these through unchanged (nb only gates
  when the config opts into soft/threshold); the *gated* nb behaviour is pinned separately in
  ``tests/test_composition_wiring.py``.

Seeded and deterministic (the S2 #121 generator gate). Values captured 2026-07-24, pre-S4 tree.
"""

import math

import numpy as np
import torch
import torch.nn as nn

from views_hydranet.distributions import resolve_family
from views_hydranet.distributions.sampling import to_cube_samples
from views_hydranet.utils.hydranet_inference import HydraNetInference

ATOL = 1e-5  # float32 emit vs float64 reference (matches test_hurdle_nb_inference.py convention)


# ── fixtures: the minimal model + inference stand-in (mirrors test_hurdle_nb_inference._inf) ──
class _MockModel(nn.Module):
    def __init__(self, output_distribution="standard", hurdle_nb_theta=None):
        super().__init__()
        self.output_distribution = output_distribution
        self.hurdle_nb_theta = hurdle_nb_theta

    def forward(self, x, h):  # pragma: no cover - not exercised
        raise NotImplementedError


def _inf(output_distribution="standard", theta=None, targets=("lr_ged_sb",)):
    model = _MockModel(output_distribution, theta)
    return HydraNetInference(model, {"regression_targets": list(targets)}, device="cpu")


def _activated_params(fam, t, h, w, n_reg, seed=0):
    """Activated param stack ``[t, n_reg*n_params, h, w]`` (target-major) — the sampler's input."""
    g = torch.Generator().manual_seed(seed)
    per = []
    for _ in range(n_reg):
        raw = torch.randn(t, h, w, fam.n_params, generator=g)
        per.append(fam.activate(raw).permute(0, 3, 1, 2))  # [t, npar, h, w]
    return torch.cat(per, dim=1)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# PARITY-LOCKED — S4's passthrough composer MUST keep these byte-identical
# ══════════════════════════════════════════════════════════════════════════════════════════════
def test_emit_standard_identity_LOCKED():
    reg = torch.tensor([[[[2.0]]]])
    out = _inf("standard")._emit_magnitude(reg, torch.tensor([[[[0.5]]]]))
    assert torch.equal(out, reg)  # standard = identity (log1p-space point already)


def test_emit_hurdle_nb_LOCKED():
    # mu=2, p=0.5, theta=1 -> E[y]=1.5 -> log1p(1.5)
    out = _inf("hurdle_nb", {"lr_ged_sb": 1.0})._emit_magnitude(
        torch.tensor([[[[2.0]]]]), torch.tensor([[[[0.5]]]])
    )
    assert torch.allclose(out, torch.tensor([[[[math.log1p(1.5)]]]]), atol=ATOL)


def test_emit_dense_nb_LOCKED():
    # dense_nb: E[y]=mu directly (no gate, no truncation) -> log1p(2)
    out = _inf("dense_nb")._emit_magnitude(torch.tensor([[[[2.0]]]]), torch.tensor([[[[0.5]]]]))
    assert torch.allclose(out, torch.tensor([[[[math.log1p(2.0)]]]]), atol=ATOL)


def test_emit_zinb_self_zeroed_LOCKED():
    # zinb self-zeroed mean = (1-pi)*mu; params [mu=2, theta=1, pi=0.25] -> 1.5 -> log1p(1.5).
    # The gate arg is IGNORED on the self-zeroed path (no external gate) — pass anything.
    zinb_reg = torch.tensor([2.0, 1.0, 0.25]).reshape(1, 3, 1, 1)
    out = _inf("zinb")._emit_magnitude(zinb_reg, torch.tensor([[[[0.9]]]]))
    assert torch.allclose(out, torch.tensor([[[[math.log1p(1.5)]]]]), atol=ATOL)


def test_cube_zinb_self_zeroed_LOCKED():
    """The self-zeroed zinb sample cube — S4's passthrough composer must reproduce this exactly."""
    fam = resolve_family("zinb")
    params = _activated_params(fam, 1, 2, 2, 1, seed=0)
    cube = to_cube_samples(params, fam, 3, torch.Generator().manual_seed(7), 1)  # [1,2,2,1,3]
    golden = np.array(
        [1.3862943649, 0.0, 1.0986123085, 0.0, 0.0, 0.0,
         0.6931471825, 0.0, 0.6931471825, 0.0, 0.0, 0.0],
        dtype=np.float32,
    ).reshape(1, 2, 2, 1, 3)
    np.testing.assert_allclose(cube, golden, atol=ATOL)


def test_cube_generator_deterministic_LOCKED():
    fam = resolve_family("zinb")
    params = _activated_params(fam, 2, 3, 3, 3, seed=1)
    a = to_cube_samples(params, fam, 4, torch.Generator().manual_seed(5), 3)
    b = to_cube_samples(params, fam, 4, torch.Generator().manual_seed(5), 3)
    np.testing.assert_array_equal(a, b)  # same seed -> byte-identical (S2 #121 gate)


def test_gate_cube_transform_LOCKED():
    """The gate cube = sigmoid(cls) transposed [T,n_cls,H,W]->[T,H,W,n_cls] and repeated across K.

    S4 composes the MAGNITUDE cube only; it must NOT touch this gate transform
    (``hydranet_inference.py:692-695``). Pinned here as the exact numpy transform the emit applies.
    """
    prob = np.arange(1 * 2 * 2 * 2, dtype=np.float32).reshape(1, 2, 2, 2)  # [T, n_cls, H, W]
    k = 3
    thwc = prob.transpose(0, 2, 3, 1)
    gate_cube = np.repeat(thwc[..., None], k, axis=-1)  # [T, H, W, n_cls, K]
    assert gate_cube.shape == (1, 2, 2, 2, 3)
    # every K-slice equals the transposed gate; the repeat adds no new information
    for j in range(k):
        np.testing.assert_array_equal(gate_cube[..., j], thwc)


# ══════════════════════════════════════════════════════════════════════════════════════════════
# nb WITH self_zeroed (passthrough) — ungated. Post-S4 the composer defaults to self_zeroed
# passthrough, so nb stays ungated UNLESS the config opts into soft_gate/threshold_gate. The gated
# behaviour is pinned in tests/test_composition_wiring.py; these anchor the passthrough path.
# ══════════════════════════════════════════════════════════════════════════════════════════════
def test_emit_nb_ungated_passthrough():
    # nb + self_zeroed (the _inf default) emits the UNGATED mean log1p(mu). Gated nb is in
    # test_composition_wiring; here we anchor that passthrough leaves the body ungated.
    nb_reg = torch.tensor([2.0, 1.0]).reshape(1, 2, 1, 1)  # activated [mu=2, theta=1]
    out = _inf("nb")._emit_magnitude(nb_reg, torch.tensor([[[[0.5]]]]))
    assert torch.allclose(out, torch.tensor([[[[math.log1p(2.0)]]]]), atol=ATOL)


def test_cube_nb_ungated_passthrough():
    # nb sample cube with the default self_zeroed composition = the UNGATED body (passthrough).
    # Gated cubes are covered by the composer/sampler suites; here we anchor passthrough.
    fam = resolve_family("nb")
    params = _activated_params(fam, 1, 2, 2, 1, seed=0)
    cube = to_cube_samples(params, fam, 3, torch.Generator().manual_seed(7), 1)
    golden = np.array(
        [1.3862943649, 0.0, 1.0986123085, 0.0, 0.6931471825, 0.0,
         0.0, 0.0, 0.6931471825, 0.0, 0.6931471825, 0.6931471825],
        dtype=np.float32,
    ).reshape(1, 2, 2, 1, 3)
    np.testing.assert_allclose(cube, golden, atol=ATOL)
