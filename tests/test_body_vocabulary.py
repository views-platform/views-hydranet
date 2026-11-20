"""The `body` setting and its five values (ADR-081, issue #633, story #668).

🟩 Green and 🟫 beige by the ADR-005 taxonomy: what each value fixes, and the configs refused.
"""

from __future__ import annotations

import pytest

from views_hydranet.body.vocabulary import (
    BODY_NAMES,
    OLD_BODY_SETTINGS,
    body_name,
    body_value_for_old_spelling,
    check_body,
)

# ADR-081 §2, the declared vocabulary and the rename table, written out here so that a change to
# the table in the code is a change this test sees.
EXPECTED = {
    "mse_softplus_point": ("standard", "mse"),
    "listwise_softplus_point": ("standard", "listwise_level"),
    "nll_softplus_nb": ("nb", None),
    "nll_softplus_truncated_nb": ("truncated_nb", None),
    "nll_softplus_mixture_nb": ("mixture_nb", None),
}

# The five combinations the 37 views-models configs wrote before #633 (ADR-081 §2 rename table).
OLD_SPELLING = [
    ("mse", "standard", "mse_softplus_point"),
    ("listwise_level", "standard", "listwise_softplus_point"),
    ("mse", "nb", "nll_softplus_nb"),
    ("mse", "truncated_nb", "nll_softplus_truncated_nb"),
    ("mse", "mixture_nb", "nll_softplus_mixture_nb"),
]


def test_the_vocabulary_is_exactly_the_five_values_of_adr_081():
    assert {v: (r.distribution, r.loss_name) for v, r in BODY_NAMES.items()} == EXPECTED


@pytest.mark.parametrize("value", list(EXPECTED))
def test_a_declared_body_is_read(value):
    row = body_name({"body": value})
    assert (row.value, row.distribution, row.loss_name) == (value, *EXPECTED[value])
    assert row.has_distribution == (EXPECTED[value][1] is None)


@pytest.mark.parametrize("value", ["nb", "mse", "nll_softplus_zinb", "MSE_SOFTPLUS_POINT", 3])
def test_an_undeclared_body_is_refused_and_every_declared_value_is_listed(value):
    with pytest.raises(ValueError) as excinfo:
        body_name({"body": value})
    for declared in EXPECTED:
        assert repr(declared) in str(excinfo.value)


@pytest.mark.parametrize("config", [{}, {"body": None}, {"body": "__MISSING_body__"}])
def test_a_missing_body_is_refused_as_required(config):
    with pytest.raises(ValueError, match="'body' is required"):
        body_name(config)


@pytest.mark.parametrize(("loss_reg", "distribution", "replacement"), OLD_SPELLING)
def test_the_old_spelling_is_refused_and_the_error_names_the_replacement(
    loss_reg, distribution, replacement
):
    config = {"loss_reg": loss_reg, "output_distribution": distribution}
    with pytest.raises(ValueError, match=f"body='{replacement}'"):
        body_name(config)


@pytest.mark.parametrize("key", OLD_BODY_SETTINGS)
def test_each_old_setting_alone_is_refused_even_beside_a_body(key):
    """Any one of the three, with any value, beside a valid `body`: refused, not ignored."""
    with pytest.raises(ValueError, match=key):
        body_name({"body": "nll_softplus_nb", key: None})


def test_a_standard_config_that_named_no_loss_gets_no_suggested_value():
    """The old spelling did not determine a body, so the message must not invent one."""
    with pytest.raises(ValueError) as excinfo:
        body_name({"output_distribution": "standard", "reg_activation": "softplus"})
    assert "trained body=" not in str(excinfo.value)


@pytest.mark.parametrize(("loss_reg", "distribution", "replacement"), OLD_SPELLING)
def test_the_rename_table_maps_each_old_combination(loss_reg, distribution, replacement):
    assert body_value_for_old_spelling(loss_reg, distribution) == replacement


def test_on_a_distribution_the_old_loss_setting_did_not_decide():
    """A distribution trained its likelihood whatever `loss_reg` said (ADR-081 §1)."""
    assert body_value_for_old_spelling(None, "nb") == "nll_softplus_nb"
    assert body_value_for_old_spelling("listwise_level", "nb") == "nll_softplus_nb"


@pytest.mark.parametrize(
    ("loss_reg", "distribution"), [(None, "standard"), ("mae", "standard"), ("mse", "zinb")]
)
def test_an_old_combination_with_no_body_maps_to_none(loss_reg, distribution):
    assert body_value_for_old_spelling(loss_reg, distribution) is None


def test_check_body_accepts_every_declared_value():
    for value in EXPECTED:
        check_body(value)


@pytest.mark.parametrize("distribution", ["nb", "truncated_nb", "mixture_nb"])
def test_a_ranking_loss_beside_a_distribution_gets_no_suggested_value(distribution):
    """That pair trained the likelihood and threw the ranking loss away; suggesting the
    distribution's body would repeat the swap (code review of #633)."""
    with pytest.raises(ValueError) as excinfo:
        body_name({"loss_reg": "listwise_level", "output_distribution": distribution})
    assert "trained body=" not in str(excinfo.value)
