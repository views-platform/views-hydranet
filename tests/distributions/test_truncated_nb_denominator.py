"""The NLL denominator axis on the zero-truncated NB body (glossary §3b).

`body_nll_denominator` selects what the truncated body's per-cell losses are divided by:
`"positives"` (historical; the count of y>0 cells) or `"all_cells"` (every supervised cell, which
is the denominator the JOINT HURDLE NLL has). Pre-registered in
`reports/2026-10-15_loss_balance_prereg.md`; the measured consequence of getting it wrong is in
`reports/2026-10-13_composition_arm_outcome/` — a 65x rescale of one task-loss term that no
config-identity check could see.

These tests pin BOTH settings, because the whole defect class here is a reduction that changes
silently. `test_truncated_negative_binomial.py:118` pins the default; this file pins that the
default is still reachable, that the new value does what it claims, and that it is rejected where
it would be inert.
"""

import pytest
import torch

from views_hydranet.distributions import resolve_family
from views_hydranet.distributions.truncated_negative_binomial import TruncatedNBFamily


def _pos():
    return TruncatedNBFamily(body_nll_denominator="positives")


def _all():
    return TruncatedNBFamily(body_nll_denominator="all_cells")


# one positive cell, plus a growing pad of zero cells
_P = torch.tensor([[2.0, 1.5], [9.0, 9.0], [0.3, 4.0], [7.0, 1.0], [1.0, 1.0]])
_Y = torch.log1p(torch.tensor([5.0, 0.0, 0.0, 0.0, 0.0]))


def test_default_is_positives_and_unchanged():
    """The constructor default must preserve the historical guarantee #3 exactly."""
    assert TruncatedNBFamily().body_nll_denominator == "positives"
    assert torch.equal(TruncatedNBFamily().nll(_P, _Y), _pos().nll(_P, _Y))


def test_positives_is_undiluted_by_zero_padding():
    """Guarantee #3, restated here so this file fails if the default silently flips."""
    two = _pos().nll(_P[:2], _Y[:2])
    five = _pos().nll(_P, _Y)
    assert torch.allclose(two, five, atol=1e-6)


def test_all_cells_IS_diluted_by_zero_padding():
    """The whole point: under `all_cells` the same positive over more cells is a SMALLER mean."""
    two = _all().nll(_P[:2], _Y[:2])
    five = _all().nll(_P, _Y)
    assert five < two
    # exactly 2/5 of it — one positive over N=2 against the same positive over N=5
    assert torch.allclose(five, two * (2.0 / 5.0), atol=1e-6)


def test_all_cells_equals_positives_times_positive_fraction():
    """The identity the pre-registered magnitude table is built on: all_cells = positives * P/N."""
    n = _P.shape[0]
    p = int((torch.expm1(_Y) > 0).sum())
    assert torch.allclose(_all().nll(_P, _Y), _pos().nll(_P, _Y) * (p / n), atol=1e-6)


def test_numerator_is_identical_only_the_denominator_moves():
    """Scaling by the cell count must recover the positives reduction bit for bit."""
    n = _P.shape[0]
    p = int((torch.expm1(_Y) > 0).sum())
    assert torch.allclose(_all().nll(_P, _Y) * (n / p), _pos().nll(_P, _Y), atol=1e-6)


def test_no_positive_cells_is_zero_under_both_and_stays_graph_connected():
    params = torch.tensor([[2.0, 1.5], [9.0, 9.0]], requires_grad=True)
    y0 = torch.log1p(torch.tensor([0.0, 0.0]))
    for fam in (_pos(), _all()):
        loss = fam.nll(params, y0)
        assert float(loss) == 0.0
        assert loss.requires_grad, "an all-zero window must stay in the graph (ADR-014)"


def test_caller_weight_defines_supervised_in_the_denominator():
    """With an active-cell mask, `all_cells` divides by the SUPERVISED count, not the grid."""
    # supervise the positive + one zero => denominator 2, not 5
    w = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    masked = _all().nll(_P, _Y, weight=w)
    two = _all().nll(_P[:2], _Y[:2])
    assert torch.allclose(masked, two, atol=1e-6)


@pytest.mark.parametrize("bad", ["all cells", "dense", "ALL_CELLS", "", "N", None])
def test_family_rejects_an_unknown_denominator(bad):
    with pytest.raises(ValueError, match="body_nll_denominator"):
        TruncatedNBFamily(body_nll_denominator=bad)


def test_resolve_family_forwards_the_kwarg():
    fam = resolve_family("truncated_nb", body_nll_denominator="all_cells")
    assert fam.body_nll_denominator == "all_cells"


def test_resolve_family_without_the_kwarg_is_the_emit_path_and_unchanged():
    """The emit-time caller passes nothing; a reduction has no role in drawing a sample."""
    assert resolve_family("truncated_nb").body_nll_denominator == "positives"


# ── the config axis: valid set, and REJECTED where it would be inert (C-324) ──
#
# `HydraNetConfig` has many required fields, so these build on the suite's `valid_config_dict`
# fixture rather than constructing it bare.


def _cfg(valid_config_dict, **over):
    from views_hydranet.utils.config_initializer import HydraNetConfig

    cfg = dict(valid_config_dict)
    cfg.update(over)
    return HydraNetConfig(**cfg)


def test_config_default_is_positives(valid_config_dict):
    assert _cfg(valid_config_dict).body_nll_denominator == "positives"


def test_config_accepts_all_cells_on_truncated_nb(valid_config_dict):
    cfg = _cfg(
        valid_config_dict,
        output_distribution="truncated_nb",
        forecast_composition="soft_gate",
        body_nll_denominator="all_cells",
    )
    assert cfg.body_nll_denominator == "all_cells"


@pytest.mark.parametrize("bad", ["dense", "all cells", "ALL_CELLS", "positive"])
def test_config_rejects_an_unknown_denominator(valid_config_dict, bad):
    with pytest.raises(ValueError, match="body_nll_denominator"):
        _cfg(valid_config_dict, body_nll_denominator=bad)


@pytest.mark.parametrize("dist", ["nb", "zinb"])
def test_config_rejects_all_cells_where_it_would_be_inert(valid_config_dict, dist):
    """C-324: every other family already reduces over all cells, so the flag changes nothing."""
    with pytest.raises(ValueError, match="requires output_distribution='truncated_nb'"):
        _cfg(
            valid_config_dict,
            output_distribution=dist,
            forecast_composition="self_zeroed" if dist == "zinb" else "soft_gate",
            body_nll_denominator="all_cells",
        )


def test_positives_is_accepted_on_every_family_because_it_is_the_default(valid_config_dict):
    """The default must never be the thing that fails validation."""
    for dist, comp in (
        ("nb", "soft_gate"),
        ("zinb", "self_zeroed"),
        ("truncated_nb", "soft_gate"),
    ):
        cfg = _cfg(
            valid_config_dict,
            output_distribution=dist,
            forecast_composition=comp,
            body_nll_denominator="positives",
        )
        assert cfg.body_nll_denominator == "positives"
