"""Invariants for the feedback-field diagnostic transforms (#258/#262).

Each transform exists to move **exactly one** axis of field realism. If it moves a second axis, the
dose-response arm built on it is confounded and the experiment silently measures the wrong thing —
this programme's stated main failure mode. So every test here pins the axis that must move *and*
the axes that must not.

The two that matter most:

* ``spatial_scramble`` must preserve temporal persistence (it uses one rollout-constant
  permutation), because the naive per-step reshuffle would destroy persistence too and make the
  spatial and temporal arms the same arm.

The temporal axis is a **step remapping** (``shuffle_months``), not a field transform, so it is
tested with the seam rather than here. A torus roll was tried and rejected — see the module
docstring: it preserves the statistics but decouples the field from the geography the static
channels encode.
"""

import pytest
import torch

from views_hydranet.utils.feedback_field_transforms import (
    FEEDBACK_TRANSFORMS,
    inject,
    magnitude_perturb,
    parse_feedback_transform,
    spatial_scramble,
    splice_occurrence_magnitude,
    thin,
)


def _gen(seed=0):
    return torch.Generator().manual_seed(seed)


def _field(active_cells, *, h=16, w=16, value=5.0):
    """[1, 1, h, w] count-space field with the given (row, col) cells active."""
    f = torch.zeros(1, 1, h, w)
    for i, (r, c) in enumerate(active_cells):
        f[0, 0, r, c] = value + i
    return f


def _clustered(h=16, w=16):
    """A field with two compact blobs — spatial structure a shuffle should destroy."""
    f = torch.zeros(1, 1, h, w)
    f[0, 0, 2:5, 2:5] = 3.0
    f[0, 0, 10:13, 10:13] = 7.0
    return f


def _neighbour_agreement(field):
    """Fraction of active cells whose right/down neighbour is also active — a clustering proxy."""
    a = (field[0, 0] > 0).float()
    pairs = (a[:, :-1] * a[:, 1:]).sum() + (a[:-1, :] * a[1:, :]).sum()
    return float(pairs / a.sum())


# --------------------------------------------------------------------- spec parsing


def test_parse_accepts_parameterised_and_bare_specs():
    assert parse_feedback_transform("thin:0.25") == ("thin", 0.25)
    assert parse_feedback_transform("spatial_scramble") == ("spatial_scramble", None)


def test_parse_rejects_an_unknown_name():
    """A typo'd arm must not fall through to the control and be reported as 'no effect'."""
    with pytest.raises(ValueError, match="unknown feedback transform"):
        parse_feedback_transform("thinn:0.25")


def test_parse_rejects_a_missing_or_surplus_parameter():
    with pytest.raises(ValueError, match="requires a parameter"):
        parse_feedback_transform("thin")
    with pytest.raises(ValueError, match="takes no parameter"):
        parse_feedback_transform("spatial_scramble:0.5")


def test_every_registered_transform_parses():
    # wrong_month takes a non-zero INTEGER offset; the others take a fraction.
    valid = {"wrong_month": "-60"}
    for name, takes_param in FEEDBACK_TRANSFORMS.items():
        spec = f"{name}:{valid.get(name, '0.5')}" if takes_param else name
        assert parse_feedback_transform(spec)[0] == name


@pytest.mark.parametrize("bad", ["wrong_month:0", "wrong_month:0.5", "wrong_month:-1.5"])
def test_parse_rejects_a_wrong_month_offset_that_would_run_the_control(bad):
    """A fractional offset truncates and 0 IS use_real — both would score the control
    as if it were the treatment."""
    with pytest.raises(ValueError, match="NON-ZERO INTEGER"):
        parse_feedback_transform(bad)


def test_magnitude_perturb_preserves_the_mean_of_the_active_values():
    """Otherwise the magnitude axis confounds realism with INFLATION — exp(sZ) has mean exp(s^2/2),
    a x2.5 blow-up at sigma=1.5, on a model with documented runaway-feedback sensitivity."""
    f = torch.full((1, 1, 200, 200), 4.0)
    for sigma in (0.5, 1.5):
        out = magnitude_perturb(f, sigma=sigma, generator=_gen(5))
        assert float(out.mean()) == pytest.approx(4.0, rel=0.05), f"sigma={sigma} moved the mean"


def test_splice_can_degrade_instead_of_raising_when_the_donor_is_empty():
    """The rollout arm's donor is the MODEL's field, and the model going quiet is the phenomenon
    under study — raising would abort an unattended batch exactly when the effect appears."""
    out = splice_occurrence_magnitude(
        _clustered(), torch.zeros(1, 1, 16, 16), generator=_gen(), on_empty_donor="zeros"
    )
    assert float(out.abs().sum()) == 0.0


# --------------------------------------------------------------------- thin (recall)


def test_thin_removes_only_active_cells_and_leaves_survivors_exact():
    f = _clustered()
    out = thin(f, p=0.5, generator=_gen())
    removed = (f > 0) & (out == 0)
    assert removed.any(), "p=0.5 removed nothing — the transform is inert"
    # never invents an active cell
    assert not ((f == 0) & (out > 0)).any()
    # survivors keep their exact values: only occurrence moves, never magnitude
    survived = out > 0
    assert torch.equal(out[survived], f[survived])


@pytest.mark.parametrize("p,expected", [(0.0, 18), (1.0, 0)])
def test_thin_endpoints_are_exact(p, expected):
    out = thin(_clustered(), p=p, generator=_gen())
    assert int((out > 0).sum()) == expected


def test_thin_rejects_a_probability_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"p must be in \[0, 1\]"):
        thin(_clustered(), p=1.5, generator=_gen())


# --------------------------------------------------------------------- inject (precision)


def test_inject_adds_only_to_empty_cells_and_leaves_originals_exact():
    f = _clustered()
    out = inject(f, q=0.2, generator=_gen())
    added = (f == 0) & (out > 0)
    assert added.any(), "q=0.2 added nothing — the transform is inert"
    # originals untouched: this axis adds false positives, it does not move true ones
    orig = f > 0
    assert torch.equal(out[orig], f[orig])


def test_inject_draws_values_from_the_fields_own_active_distribution():
    """Otherwise the arm would confound a precision change with a magnitude change."""
    f = _clustered()
    out = inject(f, q=1.0, generator=_gen())
    pool = set(f[f > 0].tolist())
    added = out[(f == 0) & (out > 0)]
    assert set(added.tolist()) <= pool


def test_inject_raises_on_a_field_with_no_actives():
    """No empirical distribution to draw from — inventing one would be uninterpretable."""
    with pytest.raises(ValueError, match="no active cells"):
        inject(torch.zeros(1, 1, 8, 8), q=0.1, generator=_gen())


# ------------------------------------------------- spatial_scramble (structure, NOT persistence)


def test_spatial_scramble_preserves_the_value_multiset_exactly():
    f = _clustered()
    perm = torch.randperm(16 * 16, generator=_gen(1))
    out = spatial_scramble(f, permutation=perm)
    assert torch.equal(f.flatten().sort().values, out.flatten().sort().values)


def test_spatial_scramble_destroys_clustering():
    f = _clustered()
    perm = torch.randperm(16 * 16, generator=_gen(1))
    out = spatial_scramble(f, permutation=perm)
    assert _neighbour_agreement(out) < _neighbour_agreement(f) / 2


def test_spatial_scramble_preserves_persistence_across_steps():
    """THE orthogonality guarantee: one rollout-constant permutation keeps temporal overlap exact.

    A per-step random permutation would destroy persistence too, making this arm
    indistinguishable from `shuffle_months` — two of five axes silently measuring one thing.
    """
    perm = torch.randperm(16 * 16, generator=_gen(1))
    f_t = _field([(2, 2), (2, 3), (7, 7)])
    f_t1 = _field([(2, 2), (2, 3), (9, 9)])  # two cells persist
    overlap_before = int(((f_t > 0) & (f_t1 > 0)).sum())
    a, b = spatial_scramble(f_t, permutation=perm), spatial_scramble(f_t1, permutation=perm)
    assert int(((a > 0) & (b > 0)).sum()) == overlap_before == 2


def test_spatial_scramble_rejects_a_permutation_of_the_wrong_length():
    with pytest.raises(ValueError, match="entries for a"):
        spatial_scramble(_clustered(), permutation=torch.arange(10))


# --------------------------------------------------------------------- magnitude_perturb


def test_magnitude_perturb_leaves_the_occurrence_mask_byte_identical():
    """This axis moves *how much*, never *where*."""
    f = _clustered()
    out = magnitude_perturb(f, sigma=1.5, generator=_gen())
    assert torch.equal(f > 0, out > 0)


def test_magnitude_perturb_actually_moves_the_values():
    f = _clustered()
    out = magnitude_perturb(f, sigma=1.0, generator=_gen())
    assert not torch.equal(f, out)


def test_magnitude_perturb_at_sigma_zero_is_the_identity():
    f = _clustered()
    assert torch.allclose(magnitude_perturb(f, sigma=0.0, generator=_gen()), f)


def test_magnitude_perturb_rejects_a_negative_sigma():
    with pytest.raises(ValueError, match="sigma must be >= 0"):
        magnitude_perturb(_clustered(), sigma=-1.0, generator=_gen())


# --------------------------------------------------------------------- splice (E4)


def test_splice_takes_occurrence_from_one_field_and_magnitudes_from_the_other():
    occ = _field([(1, 1), (2, 2), (3, 3)], value=1.0)
    mag = _field([(9, 9), (10, 10)], value=100.0)
    out = splice_occurrence_magnitude(occ, mag, generator=_gen())
    assert torch.equal(out > 0, occ > 0), "the WHERE must come from the occurrence field"
    assert set(out[out > 0].tolist()) <= set(mag[mag > 0].tolist()), "the HOW MUCH from the donor"


def test_splice_does_not_leak_the_donors_occurrence_pattern():
    """Reading the donor cell-wise would zero every cell the donor happens to be inactive at,
    re-introducing the donor's occurrence into an arm meant to isolate magnitude."""
    occ = _clustered()
    mag = _field([(0, 0)], value=42.0)  # donor active in exactly one cell
    out = splice_occurrence_magnitude(occ, mag, generator=_gen())
    assert int((out > 0).sum()) == int((occ > 0).sum())


def test_splice_raises_when_the_donor_has_no_actives():
    with pytest.raises(ValueError, match="no active cells"):
        splice_occurrence_magnitude(_clustered(), torch.zeros(1, 1, 16, 16), generator=_gen())


# --------------------------------------------------------------------- determinism


@pytest.mark.parametrize(
    "call",
    [
        lambda g: thin(_clustered(), p=0.4, generator=g),
        lambda g: inject(_clustered(), q=0.1, generator=g),
        lambda g: magnitude_perturb(_clustered(), sigma=0.7, generator=g),
    ],
)
def test_transforms_are_reproducible_from_a_seeded_generator(call):
    """Arms must be re-derivable; a run that cannot be reproduced cannot be audited."""
    assert torch.equal(call(_gen(11)), call(_gen(11)))
    assert not torch.equal(call(_gen(11)), call(_gen(12)))
