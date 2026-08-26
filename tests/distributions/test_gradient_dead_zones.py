"""The NB parameter clamp is a gradient dead zone — where its edge is, and what it costs.

``nb_core._clamp`` (``nb_core.py:23-24``) does::

    return mu.clamp_min(_EPS), theta.clamp_min(_EPS)      # _EPS = 1e-6

``clamp_min`` passes zero gradient below its threshold. The head activates with ``softplus``, so
the boundary in raw (pre-activation) space is exactly ``softplus(raw) == 1e-6``, i.e.
``raw == log(expm1(1e-6)) ≈ -13.8155``.

This matters because it reinstates the failure mode C-178 was opened for and
``HydraBNrecurrentUnet_06_LSTM4.py:104-106`` claims is gone:

    "drifts 100% negative => ReLU==0 with zero gradient => unrecoverable). softplus is always
    positive with non-zero gradient, so it cannot die."

Softplus alone cannot die. Softplus *followed by* ``clamp_min(1e-6)`` can, and the transition is
abrupt, not gradual — measured at ``raw_theta = 0`` on an active cell:

===================  ===========================
``raw_mu``           ``dNLL/d(raw_mu)``
===================  ===========================
-13.81               -6.389
-13.82               **exactly 0.0**
===================  ===========================

(at ``raw_theta = 0``, count 6.39. The magnitude is theta-dependent — see C-313 — but the drop to
exactly zero below the edge is not.)

Below the edge there is no gradient in either direction, so the unit cannot climb back out.

**Severity, measured rather than assumed.** Running the trained L=300 incumbent
(``fullzero_fortytwo``) forward on a sparse field, the head's minimum raw values are ``mu``
-3.81 and ``theta`` -9.99 — 3.8 and 3.8 nats clear of the edge. So this is a **latent** hazard on
the current vehicle, not an active defect, and no past result is in question. It becomes live if
``_EPS`` rises, if the head's activation or bias init changes, or if a longer/harder run pushes
the head further down. These tests pin the edge so any of those is a visible break.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from views_hydranet.distributions import get_family  # noqa: E402
from views_hydranet.distributions.nb_core import _EPS  # noqa: E402

#: raw value at which ``softplus(raw)`` equals the clamp floor — derived, never hardcoded.
EDGE = math.log(math.expm1(_EPS))


def _grad_wrt_raw(raw_mu: float, raw_theta: float, count: float = 6.0):
    """``dNLL/d(raw)`` through the REAL path, with the target in the space the loss expects.

    Two contracts are easy to get wrong here, and this audit got both wrong once before fixing them:

    * ``nll`` receives **activated** params, not raw ones. Passing raw values measures a different
      function entirely and yields a flat, meaningless gradient surface.
    * ``target`` is in **log1p space** — ``nll`` recovers counts via ``to_raw_counts``. Passing a
      raw count of 50 actually asks about a count of ``expm1(50) ≈ 5e21``, which manufactures a
      spurious explosion.

    ``count`` is therefore a real count and is log1p-transformed here.
    """
    family = get_family("nb")
    raw = torch.tensor([[raw_mu, raw_theta]], requires_grad=True)
    nll = family.nll(family.activate(raw), torch.log1p(torch.tensor([count])))
    (grad,) = torch.autograd.grad(nll, raw)
    return grad[0, 0].item(), grad[0, 1].item()


def test_the_dead_zone_edge_is_where_softplus_meets_the_clamp_floor():
    """Pins the boundary itself. A change to ``_EPS`` or the activation moves it and fails here."""
    assert torch.nn.functional.softplus(torch.tensor(EDGE)).item() == pytest.approx(_EPS, rel=1e-3)
    just_above, _ = _grad_wrt_raw(EDGE + 0.01, 0.0)
    just_below, _ = _grad_wrt_raw(EDGE - 0.01, 0.0)
    assert just_above != 0.0, f"gradient already dead at raw_mu={EDGE + 0.01:.4f}"
    assert just_below == 0.0, f"gradient still alive at raw_mu={EDGE - 0.01:.4f}: {just_below}"


def test_mu_gradient_dies_abruptly_not_gradually_characterisation():
    """CHARACTERISATION (C-178 reinstated): the gradient does not fade, it stops.

    Measured at ``raw_theta = 0`` (theta ~ 0.693) the magnitude just above the edge is ~6.4 — if
    the clamp merely damped an already-negligible gradient this would be harmless bookkeeping, and
    it does not. Note the magnitude is strongly theta-dependent: at the *incumbent's* operating
    point (``raw_theta = -9.99``) it is ~0.2 at count 100 and points away from the cliff, which is
    why C-313 is registered as remote rather than acute.
    """
    alive, _ = _grad_wrt_raw(EDGE + 0.01, 0.0, count=6.0)
    assert abs(alive) > 1.0, (
        f"gradient just above the clamp edge is only {alive:.3e}; the abrupt-cutoff framing of "
        "this finding no longer holds — re-measure before trusting the docstring above."
    )
    for depth in (0.01, 1.0, 5.0, 10.0):
        dead, _ = _grad_wrt_raw(EDGE - depth, 0.0, count=6.0)
        assert dead == 0.0, f"raw_mu={EDGE - depth:.4f}: expected exactly 0.0, got {dead}"


def test_the_dead_zone_is_irrecoverable_in_both_directions():
    """No gradient means no way back. Confirms it is a trap, not a soft floor."""
    for count in (0.0, 1.0, 10.0):
        dead, _ = _grad_wrt_raw(EDGE - 2.0, 0.0, count=count)
        assert dead == 0.0, f"count={count}: no target size should revive the unit, got {dead}"


def test_theta_dies_at_the_same_edge():
    """The dispersion channel has the same clamp, and it is the one C-199/C-203 call fragile."""
    _, alive = _grad_wrt_raw(0.0, EDGE + 0.01)
    _, dead = _grad_wrt_raw(0.0, EDGE - 0.01)
    assert alive != 0.0, "theta gradient already dead above the edge"
    assert dead == 0.0, f"theta gradient still alive below the edge: {dead}"


def test_the_existing_theta_bound_test_is_vacuous_at_its_deepest_point():
    """``test_theta_gradient_bound.py`` asserts ``|d/d raw_theta| <= 1.5`` at ``raw_theta=-16.0``.

    -16.0 is **inside** the dead zone (edge -13.8155), so at that point the bound is satisfied by
    a gradient of exactly 0.0 — by the channel being dead, not by softplus cancelling an
    explosion as its docstring claims. Its other sample points (-10, -13) are above the edge and
    do demonstrate the cancellation, so the test is sound apart from this one vacuous row.

    Asserted here rather than by editing that test, because deleting the row would erase the
    evidence that the trap exists.
    """
    assert -16.0 < EDGE, "the -16.0 sample point is no longer inside the dead zone"
    _, at_minus_16 = _grad_wrt_raw(0.0, -16.0)
    assert at_minus_16 == 0.0, (
        f"raw_theta=-16.0 now yields gradient {at_minus_16:.3e}; the vacuity this documents is "
        "gone, so the note in test_theta_gradient_bound.py can go too."
    )
    _, at_minus_13 = _grad_wrt_raw(0.0, -13.0)
    assert at_minus_13 != 0.0 and abs(at_minus_13) <= 1.5, (
        "the -13.0 sample point should be alive AND bounded — that is the row that actually "
        f"demonstrates the cancellation, got {at_minus_13:.4f}"
    )


def test_the_trained_incumbent_sits_clear_of_the_edge():
    """Severity check: a randomly-initialised head must not already be in the trap.

    The measured trained incumbent sits ~3.8 nats above the edge on both channels; this asserts
    the weaker, checkpoint-free property that initialisation alone does not put the head there,
    so the finding stays classified as latent.
    """
    from views_hydranet.architectures.registry import get_architecture

    torch.manual_seed(0)
    model = get_architecture("HydraBNUNet06_LSTM4")(
        3, 32, 1, 0.0, output_distribution="nb"
    ).float()
    raws: dict[str, torch.Tensor] = {}
    for head in (1, 2, 3):
        name = f"dec_conv4_head{head}_reg"
        dict(model.named_modules())[name].register_forward_hook(
            lambda _m, _i, out, n=name: raws.__setitem__(n, out.detach())
        )
    model.eval()
    torch.manual_seed(1)
    x = (torch.rand(1, 3, 32, 32) < 0.03).float() * torch.rand(1, 3, 32, 32) * 4
    with torch.no_grad():
        model(x, model.init_hTtime(model.base, 32, 32).float())

    assert raws, "no reg head fired — the hook names are stale"
    for name, out in raws.items():
        assert out.min().item() > EDGE, (
            f"{name}: raw head output reaches {out.min().item():.3f}, below the dead-zone edge "
            f"{EDGE:.3f}, at INITIALISATION. This finding is no longer latent."
        )
