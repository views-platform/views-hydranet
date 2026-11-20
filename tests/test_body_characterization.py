"""The body dispatch, pinned before ADR-081 moved anything and rewritten in its spelling (#633).

ADR-081 collapses `loss_reg` + `output_distribution` + `reg_activation` into one `body` field and
claims that no model changes what it computes. This file was written against the old spelling at
commit e57ef2b (story #604) and every assertion on the built objects was kept when the configs
were rewritten as `body` values (story #671): the criterion type and its parameters per row are
the ones the old spelling built.

🟥 Red by the ADR-005 taxonomy: it describes behaviour that must be preserved, and it is the file
every later story in #603 is measured against. It asserts the criterion object **and a constructor
parameter** of it, never bare `isinstance` — ADR-081 §5 invariant 1 requires "an object of the
identical type, with identical parameters".

Until #633 a ranking loss written beside a distribution was silently discarded (the defect
ADR-081 exists for). A `body` value cannot express that pair; the test that pinned the discard is
replaced by the acceptance test of epic #633: two configs that differ only in `body` build two
different losses.

Every number here was read off the tree at commit e57ef2b. `choose_loss` is given plain dicts, not
`HydraNetConfig` objects, deliberately: the dispatch is what is under test, and a raw dict is also
how the pod drivers and the measurement scripts reach it.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import (  # noqa: E402
    HydraBNUNet06_LSTM4,
)
from views_hydranet.body.bodies import RankingBody, body_class_for  # noqa: E402
from views_hydranet.body.vocabulary import (  # noqa: E402
    BODY_NAMES,
    body_value_for_old_spelling,
)
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.distributions.mixture_negative_binomial import (  # noqa: E402
    MixtureNBFamily,
)
from views_hydranet.distributions.negative_binomial import NegativeBinomialFamily  # noqa: E402
from views_hydranet.distributions.truncated_negative_binomial import (  # noqa: E402
    TruncatedNBFamily,
)
from views_hydranet.utils.listwise_level_loss import ListwiseLevelLoss  # noqa: E402
from views_hydranet.utils.utils import choose_loss  # noqa: E402

CPU = torch.device("cpu")

# The five combinations live in views-models before #633 — ADR-081 §2's rename table, left-hand
# column — and the `body` value each became, so this file and the ADR cannot drift.
LIVE = [
    ("mse", "standard", "softplus", "mse_softplus_point"),
    ("listwise_level", "standard", "softplus", "listwise_softplus_point"),
    ("mse", "nb", None, "nll_softplus_nb"),
    ("mse", "truncated_nb", None, "nll_softplus_truncated_nb"),
    ("mse", "mixture_nb", None, "nll_softplus_mixture_nb"),
]


def _cfg(valid_config_dict, **overrides):
    """A raw config dict for `choose_loss`, with the listwise body's required param available."""
    cfg = {**valid_config_dict, "loss_reg_y_max": 10000.0}
    cfg.update(overrides)
    return cfg


def _reg(cfg):
    criterion_reg, _, _ = choose_loss(cfg, CPU)
    return criterion_reg


# ───────────────────────────────── the five live combinations ─────────────────────────────────


def test_mse_standard_builds_mseloss(valid_config_dict):
    reg = _reg(_cfg(valid_config_dict, body="mse_softplus_point"))
    assert type(reg) is nn.MSELoss
    assert reg.reduction == "mean"


def test_listwise_standard_builds_the_listwise_loss_with_its_declared_level_set(valid_config_dict):
    reg = _reg(_cfg(valid_config_dict, body="listwise_softplus_point"))
    assert type(reg) is ListwiseLevelLoss
    # The constructor-visible parameters: the transform T and the level set derived from y_max.
    assert reg.transform == "exp"
    assert reg.level_max == 14  # ceil(10000).bit_length()
    # The ranking body trains on every cell (#630: stated by the body, no longer a loss flag).
    assert body_class_for("standard", "listwise_level") is RankingBody
    assert RankingBody.requires_all_cells is True


@pytest.mark.parametrize(
    "distribution,family_cls,n_params",
    [
        ("nb", NegativeBinomialFamily, 2),
        ("truncated_nb", TruncatedNBFamily, 2),
        ("mixture_nb", MixtureNBFamily, 5),
    ],
)
def test_the_family_rows_build_a_familyloss_over_that_family(
    valid_config_dict, distribution, family_cls, n_params
):
    reg = _reg(_cfg(valid_config_dict, body=f"nll_softplus_{distribution}"))
    assert type(reg) is FamilyLoss
    # The parameters, not just the type: which family, and how many channels it consumes.
    assert type(reg.family) is family_cls
    assert reg.n_params == n_params


def test_the_rename_table_covers_every_live_combination():
    """The five rows are the five rows, and the code's table agrees with ADR-081's."""
    assert len(LIVE) == 5
    assert {becomes for *_, becomes in LIVE} == set(BODY_NAMES)
    for loss_reg, distribution, _, becomes in LIVE:
        assert body_value_for_old_spelling(loss_reg, distribution) == becomes


# ──────────────── acceptance (#633): two configs differing only in `body` ────────────────


def test_two_configs_that_differ_only_in_body_build_two_different_losses(valid_config_dict):
    """⛔ The definition of done of epic #633, and the pair that could not be written before it.

    The ranking experiment's two arms differ in the body and nothing else. Before #633 a ranking
    loss written beside `nb` was discarded and the arm trained the NB likelihood (ADR-081 §1).
    """
    control = _cfg(valid_config_dict, body="nll_softplus_nb")
    ranking = _cfg(valid_config_dict, body="listwise_softplus_point")
    assert {k for k in control if control[k] != ranking.get(k)} == {"body"}
    a, b = _reg(control), _reg(ranking)
    assert type(a) is FamilyLoss and type(a.family) is NegativeBinomialFamily
    assert type(b) is ListwiseLevelLoss
    assert not any(isinstance(m, ListwiseLevelLoss) for m in a.modules())


# ──────────────────────────────── the head: width and activation ────────────────────────────────


def _head(output_distribution, reg_activation=None):
    torch.manual_seed(0)
    return HydraBNUNet06_LSTM4(
        3,
        16,
        1,
        0.0,
        output_distribution=output_distribution,
        reg_activation=reg_activation,
    )


@pytest.mark.parametrize(
    "distribution,width",
    [("standard", 1), ("nb", 2), ("truncated_nb", 2), ("mixture_nb", 5)],
)
def test_head_width_is_the_emit_distributions_parameter_count(distribution, width):
    """ADR-081 §2: the width is read off the distribution, never declared. Pinned per target."""
    model = _head(distribution)
    assert model.dec_conv4_head1_reg.out_channels == width
    assert model.dec_conv4_head2_reg.out_channels == width
    assert model.dec_conv4_head3_reg.out_channels == width


@pytest.mark.parametrize("distribution", ["standard"])
def test_an_explicit_softplus_wins_for_every_legacy_distribution(distribution):
    assert _head(distribution, reg_activation="softplus")._reg_activation is F.softplus


@pytest.mark.parametrize("distribution", ["nb", "truncated_nb", "mixture_nb"])
def test_a_family_owns_its_activation_although_the_field_says_softplus(distribution):
    """The family supplies the map; the field's softplus does not replace it."""
    model = _head(distribution, reg_activation="softplus")
    assert model._reg_activation is not F.relu
    assert model._reg_activation is not F.softplus
    assert model._family is not None


@pytest.mark.parametrize("distribution", ["nb", "truncated_nb", "mixture_nb"])
def test_a_family_with_the_field_set_to_relu_is_refused(distribution):
    """#632: the field accepts only implemented activations, beside a family too."""
    with pytest.raises(ValueError, match="is not implemented"):
        _head(distribution, reg_activation="relu")


def test_standard_without_an_activation_trains_softplus():
    """ADR-081 §2's "case to watch", closed by #632.

    `(mse, standard, unset)` trained ReLU until 2026-11-20. It now trains softplus, the same as
    `mse_softplus_point`, so a rename keyed on two fields no longer renames it into something
    untrue.
    """
    assert _head("standard")._reg_activation is F.softplus
