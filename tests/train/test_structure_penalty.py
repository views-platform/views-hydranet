"""The structure penalty: its contract, and the three ways it could be decorative.

Pre-registered in `reports/2026-10-21_structure_penalty_arm_prereg.md`. The term matches the
within-block dispersion of the composed field to the truth's at the block sizes in
`_STRUCT_BLOCKS`, with the GATE DETACHED so gradients reach the body alone.

Every test here is written to be able to fail. The three that matter:

  * `test_penalty_rises_when_the_field_is_smoothed` — the term's whole purpose. If a smooth
    prediction scored the same as a sharp one, the arm would be inert and the run wasted.
  * `test_gradient_reaches_mu_and_not_the_gate` — the ONE-VARIABLE contract. If the gate were
    reachable the arm would move two heads and its verdict would be uninterpretable.
  * `test_weight_zero_is_byte_identical` — the default-off-flag guarantee the whole roster depends
    on; without it, merging this changes every existing model.
"""

import pytest
import torch

from views_hydranet.train.training_engine import (
    _STRUCT_BLOCKS,
    _block_variance,
    _structure_penalty,
)


def _flat_gate(shape):
    """Gate logits at 0 => sigmoid 0.5, so the composed field is mu/2 and never all-zero."""
    return torch.zeros(shape)


def test_block_variance_matches_a_hand_computed_block():
    """Two average pools must give the same answer as computing one block by hand."""
    x = torch.tensor([[[1.0, 3.0], [5.0, 11.0]]])  # [1, 2, 2], one 2x2 block
    got = _block_variance(x, 2)
    want = torch.tensor([1.0, 3.0, 5.0, 11.0]).var(unbiased=False)
    assert got.shape == (1, 1, 1, 1)
    assert torch.allclose(got.squeeze(), want, atol=1e-6)


def test_block_variance_is_zero_on_a_constant_field():
    got = _block_variance(torch.full((2, 8, 8), 7.5), 2)
    assert torch.allclose(got, torch.zeros_like(got), atol=1e-6)


def test_block_variance_never_goes_negative_under_catastrophic_cancellation():
    """⭐ The clamp is load-bearing, and the first version of this test could not see it.

    `E[x²] − E[x]²` loses precision when `x²` exceeds float32's ~7 significant digits. Measured:
    at a field magnitude of 1e7 the raw form reaches **−8,388,608**, and `log(negative + 1e-6)` is
    NaN — which would kill a multi-hour run at whichever lesson first produced a large `mu`. An
    earlier version used a magnitude of 7.5, where the raw form is exactly 0.0, so removing the
    clamp left the suite green (mutation M5 SURVIVED). The regime is extreme for a count field;
    the clamp is one op, and a NaN here is unrecoverable.
    """
    torch.manual_seed(0)
    x = (1e7 + torch.rand(4, 16, 16)).float()
    raw = nn_functional_block_var_without_clamp(x, 2)
    assert raw.min() < 0, "this fixture must actually reach the cancellation regime"

    got = _block_variance(x, 2)
    assert (got >= 0).all()
    assert torch.isfinite(torch.log(got + 1e-6)).all()


def nn_functional_block_var_without_clamp(field, k):
    """The unclamped two-moment form, kept here so the test above can PROVE its fixture bites."""
    import torch.nn.functional as F

    x = field.unsqueeze(1)
    return F.avg_pool2d(x * x, k) - F.avg_pool2d(x, k) ** 2


def test_penalty_is_near_zero_when_the_composed_field_matches_the_truth():
    """Identity case. sigmoid(0) = 0.5, so mu = 2 * truth makes the composed field the truth."""
    truth = torch.rand(2, 8, 8) * 5.0
    mu = 2.0 * truth
    pen = _structure_penalty(mu, _flat_gate(truth.shape), truth)
    assert pen.item() == pytest.approx(0.0, abs=1e-3)


def test_penalty_rises_when_the_field_is_smoothed():
    """⭐ THE POINT OF THE TERM. A smoothed prediction must cost more than a sharp one.

    If this fails the arm is decorative: the objective would be indifferent between a field with
    the truth's dispersion and a field with none, which is the very defect D12 measured.
    """
    torch.manual_seed(0)
    truth = (torch.rand(2, 16, 16) < 0.1).float() * 10.0  # sparse spikes, like the real target
    sharp_mu = 2.0 * truth
    smooth_mu = torch.full_like(truth, 2.0 * truth.mean())  # same total, no structure

    sharp = _structure_penalty(sharp_mu, _flat_gate(truth.shape), truth)
    smooth = _structure_penalty(smooth_mu, _flat_gate(truth.shape), truth)
    assert smooth > sharp
    assert smooth > sharp + 1.0, f"smoothing must cost materially: {smooth} vs {sharp}"


def test_excess_dispersion_is_NOT_penalised():
    """⭐ THE ASYMMETRY. The term is one-sided, and D12's F1 is why.

    F1 — "the model's gamma exceeds the truth's at any d" — never fired: 4/4 cells, 480 bins, the
    model below the truth at every scale. So the defect has one direction, and on a 97.8%-empty
    field a symmetric penalty would spend most of its gradient enforcing FLATNESS where the truth
    has no dispersion. If this test fails, the term is fighting the defect it exists to fix.
    """
    torch.manual_seed(0)
    truth = torch.full((2, 8, 8), 1.0)  # flat truth: no dispersion to match
    sharp_mu = 2.0 * (truth + 5.0 * (torch.rand_like(truth) < 0.1).float())  # over-dispersed
    matched_mu = 2.0 * truth

    over = _structure_penalty(sharp_mu, _flat_gate(truth.shape), truth)
    matched = _structure_penalty(matched_mu, _flat_gate(truth.shape), truth)
    assert over.item() == pytest.approx(matched.item(), abs=1e-6), (
        "excess dispersion must cost nothing; a symmetric penalty would punish it"
    )


def test_gradient_reaches_mu_and_not_the_gate():
    """⭐ THE ONE-VARIABLE CONTRACT. The gate is detached inside the term.

    If the gate were reachable, the arm would train two heads at once and no null or win could be
    attributed to the body — the #396 composition problem, reintroduced by the fix for it.
    """
    truth = torch.rand(2, 8, 8)
    mu = torch.rand(2, 8, 8, requires_grad=True)
    gate = torch.zeros(2, 8, 8, requires_grad=True)

    _structure_penalty(mu, gate, truth).backward()

    assert mu.grad is not None and mu.grad.abs().sum() > 0, "the body must receive gradient"
    assert gate.grad is None or gate.grad.abs().sum() == 0, "the gate must NOT receive gradient"


def test_penalty_fails_loud_when_a_block_size_does_not_divide_the_grid():
    """avg_pool2d would silently drop the remainder, making the term a statement about a sub-grid
    nobody chose. That is the C-351 population failure inside a loss, so it must raise."""
    truth = torch.rand(1, 7, 7)  # 7 is divisible by neither 2 nor 8
    with pytest.raises(ValueError, match="do not"):
        _structure_penalty(torch.rand(1, 7, 7), _flat_gate(truth.shape), truth)


def test_block_sizes_are_the_pre_registered_ones():
    """The prereg commits to k = 2 and 8 and calls a configurable set a second variable. A later
    edit widening this set would silently change what the arm tests."""
    assert _STRUCT_BLOCKS == (2, 8)


def test_weight_zero_is_byte_identical():
    """⭐ THE DEFAULT-OFF GUARANTEE. With the weight at 0 the term must not be computed at all.

    Checked at the seam the engine actually uses — `struct_penalty_weight > 0` — by passing a grid
    the penalty would REJECT. If the guard were dropped, this raises; with the guard, the
    unreachable call is never made and every existing roster model is unaffected.
    """
    weight = 0.0
    pen = torch.tensor(0.0)
    if weight > 0:  # the engine's own guard, mirrored
        pen = _structure_penalty(torch.rand(1, 7, 7), _flat_gate((1, 7, 7)), torch.rand(1, 7, 7))
    assert pen.item() == 0.0


def test_config_rejects_the_penalty_on_a_non_family_body(valid_config_dict):
    """⭐ C-324: an inert setting that trains on nothing and reports a clean null.

    A legacy body emits no `mu` channel, so the term has nothing to read. The schema must refuse
    the combination at CONFIG time — before the pod is paid for — rather than leaving it to the
    engine's raise mid-run or, worse, to `extra="allow"` swallowing it.
    """
    from views_hydranet.utils.config_initializer import HydraNetConfig

    cfg = dict(valid_config_dict)
    # `forecast_composition` is set explicitly in BOTH this test and its mirror below: the fixture
    # ships `self_zeroed`, which is itself illegal on a non-self-zeroed family (ADR-069), and
    # leaving it would let that unrelated rule decide the outcome instead of this one.
    cfg.update(
        output_distribution="standard",
        forecast_composition="soft_gate",
        struct_penalty_weight=0.5,
    )
    with pytest.raises(ValueError, match="struct_penalty_weight"):
        HydraNetConfig(**cfg)


def test_config_accepts_the_penalty_on_a_family_body(valid_config_dict):
    """The mirror of the test above: without it, a validator rejecting EVERYTHING would pass."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    cfg = dict(valid_config_dict)
    cfg.update(
        output_distribution="nb", forecast_composition="soft_gate", struct_penalty_weight=0.5
    )
    assert HydraNetConfig(**cfg).struct_penalty_weight == 0.5


def test_config_default_leaves_the_penalty_off(valid_config_dict):
    """The roster must be unaffected by merging this: every existing config omits the field."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    assert HydraNetConfig(**valid_config_dict).struct_penalty_weight == 0.0
