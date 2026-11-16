"""S1 (#604, epic #603): what the body dispatch does TODAY, pinned before ADR-079 moves anything.

ADR-079 collapses `loss_reg` + `output_distribution` + `reg_activation` into one `body` field and
claims that no model changes what it computes. That claim is only checkable against a pinned
baseline, and before this file the baseline was not pinned.

🟥 Red by the ADR-005 taxonomy: it describes behaviour that must be preserved, and it is the file
every later story in #603 is measured against. It asserts the criterion object **and a constructor
parameter** of it, never bare `isinstance` — ADR-079 §5 invariant 1 requires "an object of the
identical type, with identical parameters".

⛔ The gap this file exists to close. `tests/distributions/test_loss_wiring.py` already pins two
cases, but in both of them the stated `loss_reg` AGREES with the outcome:

    cfg = {**valid_config_dict, "output_distribution": fam_name}   # loss_reg is 'mse' either way

So inverting `if _family is not None:` (`views_hydranet/utils/utils.py:220`) fails nothing there.
`test_loss_reg_is_discarded_when_a_distribution_is_named` below uses pairs that DISAGREE, which is
what makes the precedence detectable.

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
from views_hydranet.distributions.family_loss import FamilyLoss  # noqa: E402
from views_hydranet.distributions.mixture_negative_binomial import (  # noqa: E402
    MixtureNBFamily,
)
from views_hydranet.distributions.negative_binomial import NegativeBinomialFamily  # noqa: E402
from views_hydranet.distributions.truncated_negative_binomial import (  # noqa: E402
    TruncatedNBFamily,
)
from views_hydranet.distributions.zero_inflated_negative_binomial import (  # noqa: E402
    ZINBFamily,
)
from views_hydranet.utils.listwise_level_loss import ListwiseLevelLoss  # noqa: E402
from views_hydranet.utils.tobit_loss import TobitLoss  # noqa: E402
from views_hydranet.utils.utils import choose_loss  # noqa: E402

CPU = torch.device("cpu")

# The five combinations live in views-models today — ADR-079 §2's rename table, left-hand column.
# `becomes` is the value each is renamed to, carried here so this file and the ADR cannot drift.
LIVE = [
    ("mse", "standard", "softplus", "mse_softplus_point"),
    ("listwise_level", "standard", "softplus", "listwise_softplus_point"),
    ("mse", "nb", None, "nll_softplus_nb"),
    ("mse", "truncated_nb", None, "nll_softplus_truncated_nb"),
    ("mse", "mixture_nb", None, "nll_softplus_mixture_nb"),
]

# The five `output_distribution` values that keep their code but get no vocabulary row
# (owner decision 2026-11-15). A sixth, `quantile`, was purged on 2026-11-16 (ADR-079 §4).
ORPHANS = [
    "zinb",
    "hurdle_nb",
    "hurdle_lognormal",
    "hurdle_shrinkage",
    "dense_nb",
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
    reg = _reg(_cfg(valid_config_dict, loss_reg="mse", output_distribution="standard"))
    assert type(reg) is nn.MSELoss
    assert reg.reduction == "mean"


def test_listwise_standard_builds_the_listwise_loss_with_its_declared_level_set(valid_config_dict):
    reg = _reg(_cfg(valid_config_dict, loss_reg="listwise_level", output_distribution="standard"))
    assert type(reg) is ListwiseLevelLoss
    # The constructor-visible parameters: the transform T and the level set derived from y_max.
    assert reg.transform == "exp"
    assert reg.level_max == 14  # ceil(10000).bit_length()
    assert reg.needs_latent is True


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
    reg = _reg(_cfg(valid_config_dict, loss_reg="mse", output_distribution=distribution))
    assert type(reg) is FamilyLoss
    # The parameters, not just the type: which family, and how many channels it consumes.
    assert type(reg.family) is family_cls
    assert reg.n_params == n_params
    assert reg.needs_latent is False


def test_the_rename_table_covers_every_live_combination():
    """The five rows are the five rows. A sixth live combination must fail here first."""
    assert len(LIVE) == 5
    assert len({becomes for *_, becomes in LIVE}) == 5


# ──────────────────── the discard: loss_reg is skipped when a family is named ────────────────────


def test_loss_reg_is_discarded_when_a_distribution_is_named(valid_config_dict):
    """⛔ The defect, with a pair that DISAGREES — the gap ADR-079 §5 names as mattering most.

    A config asking for the listwise ranking loss beside `nb` trains the NB likelihood instead,
    and nothing warns. This is how the defect was found, and it is what would have made the
    ranking arm a second copy of the control.
    """
    reg = _reg(_cfg(valid_config_dict, loss_reg="listwise_level", output_distribution="nb"))
    assert type(reg) is FamilyLoss
    assert type(reg.family) is NegativeBinomialFamily
    # The requested loss is nowhere in the result — not wrapped, not stored, not warned about.
    assert not isinstance(reg, ListwiseLevelLoss)
    assert not any(isinstance(m, ListwiseLevelLoss) for m in reg.modules())


def test_the_discard_is_not_a_tobit_special_case(valid_config_dict):
    """The same silence covers the dict-valued tobit branch at `utils.py:244`.

    Were the discard one special case, a loss reached by a different branch might survive. It does
    not: the family branch precedes all four of them.
    """
    reg = _reg(
        _cfg(
            valid_config_dict,
            loss_reg="tobit",
            loss_reg_sigma={t: 1.0 for t in valid_config_dict["regression_targets"]},
            output_distribution="mixture_nb",
        )
    )
    assert type(reg) is FamilyLoss
    assert reg.n_params == 5
    assert not isinstance(reg, dict)


# ───────────────────────────── the orphan losses keep their branches ─────────────────────────────


@pytest.mark.parametrize("loss_reg", ["hurdle_nb", "dense_nb"])
def test_the_per_target_nb_losses_build_one_instance_per_target(valid_config_dict, loss_reg):
    """One loss per regression target, so each learnable theta reaches the optimizer."""
    cfg = _cfg(
        valid_config_dict,
        loss_reg=loss_reg,
        loss_reg_theta_init=1.0,
        output_distribution="standard",
    )
    reg = _reg(cfg)
    assert isinstance(reg, dict)
    assert set(reg) == set(valid_config_dict["regression_targets"])
    for criterion in reg.values():
        # the learnable per-target theta, the reason these are per-target at all
        assert criterion.raw_theta.requires_grad is True


def test_the_dict_valued_tobit_builds_one_instance_per_target(valid_config_dict):
    sigmas = {t: 1.5 for t in valid_config_dict["regression_targets"]}
    reg = _reg(
        _cfg(
            valid_config_dict,
            loss_reg="tobit",
            loss_reg_sigma=sigmas,
            output_distribution="standard",
        )
    )
    assert isinstance(reg, dict)
    assert set(reg) == set(sigmas)
    assert all(type(c) is TobitLoss for c in reg.values())


@pytest.mark.parametrize("distribution", ORPHANS)
def test_an_orphan_distribution_still_resolves_its_criterion(valid_config_dict, distribution):
    """`zinb` is the only orphan that names a family; the other four fall to the `loss_reg` path.

    That asymmetry is the thing to remember when reading ADR-079: `output_distribution` is a
    distribution name for four values and a head-shape name for five.
    """
    cfg = _cfg(valid_config_dict, loss_reg="mse", output_distribution=distribution)
    reg = _reg(cfg)
    if distribution == "zinb":
        assert type(reg) is FamilyLoss
        assert type(reg.family) is ZINBFamily
        assert reg.n_params == 3
    else:
        assert type(reg) is nn.MSELoss


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
    [("standard", 1), ("nb", 2), ("truncated_nb", 2), ("mixture_nb", 5), ("zinb", 3)],
)
def test_head_width_is_the_emit_distributions_parameter_count(distribution, width):
    """ADR-079 §2: the width is read off the distribution, never declared. Pinned per target."""
    model = _head(distribution)
    assert model.dec_conv4_head1_reg.out_channels == width
    assert model.dec_conv4_head2_reg.out_channels == width
    assert model.dec_conv4_head3_reg.out_channels == width


@pytest.mark.parametrize("distribution", ["standard", "hurdle_nb", "dense_nb"])
def test_an_explicit_softplus_wins_for_every_legacy_distribution(distribution):
    assert _head(distribution, reg_activation="softplus")._reg_activation is F.softplus


@pytest.mark.parametrize("distribution", ["nb", "truncated_nb", "mixture_nb", "zinb"])
def test_a_family_owns_its_activation_and_ignores_the_field(distribution):
    """The field is not consulted at all when a family is named — the family supplies the map."""
    model = _head(distribution, reg_activation="relu")
    assert model._reg_activation is not F.relu
    assert model._reg_activation is not F.softplus
    assert model._family is not None


def test_standard_without_an_activation_trains_relu():
    """⛔ ADR-079 §2's "case to watch", and the reason S5 must REFUSE this combination.

    `(mse, standard, unset)` is not `mse_softplus_point`: it trains relu. All ten live `standard`
    configs set `reg_activation='softplus'` (checked 2026-11-14), so no live model is affected —
    but a rename keyed on two fields would have renamed this one into something untrue.
    """
    assert _head("standard")._reg_activation is F.relu


@pytest.mark.parametrize(
    "distribution,expected_name",
    [
        ("hurdle_nb", "softplus"),
        ("hurdle_lognormal", "softplus"),
        ("hurdle_shrinkage", "softplus"),
        ("dense_nb", "relu"),
    ],
)
def test_the_legacy_default_activation_keys_off_the_hurdle_prefix(distribution, expected_name):
    """`startswith("hurdle")` is the rule, so `dense_nb` gets relu while the three hurdles get
    softplus. A string prefix deciding an activation is exactly what ADR-079 removes."""
    assert _head(distribution)._reg_activation.__name__ == expected_name
