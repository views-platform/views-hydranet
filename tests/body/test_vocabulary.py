"""S2 (#605, epic #603): the declared body vocabulary (ADR-081), and that it cannot lie.

🟩 Green. Three of these tests exist because of a specific way the vocabulary could go wrong and
still look right:

- `test_the_declared_set_is_exactly_five_values` — a sixth row that builds, prints and resolves
  correctly passes every other guard in this file. Only an exact-set assertion catches it.
- `test_the_printed_set_equals_the_executing_set` — #602's actual requirement. A membership
  assertion ("`nll_softplus_nb` appears in the message") survives a widened list, which is the
  defect being prevented: a value that prints but does not train.
- `test_every_declared_body_builds_a_working_criterion` — the other half of the same thing. A row
  can be perfectly well-formed data and still name a loss that cannot be constructed.
"""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from views_hydranet.body import BODIES, BodySpec, declared_bodies, legacy_body_spec, resolve_body
from views_hydranet.distributions.registry import DISTRIBUTION_REGISTRY, FAMILY_N_PARAMS

DECLARED = {
    "mse_softplus_point",
    "listwise_softplus_point",
    "nll_softplus_nb",
    "nll_softplus_truncated_nb",
    "nll_softplus_mixture_nb",
}


# ────────────────────────────────── the set, and its closure ──────────────────────────────────


def test_the_declared_set_is_exactly_five_values():
    """⛔ Exact equality. A sixth row passes every other guard in this file."""
    assert set(BODIES) == DECLARED


def test_every_declared_body_states_one_objective():
    for name, spec in BODIES.items():
        assert (spec.loss is None) != (spec.family is None), name


def test_a_spec_with_both_or_neither_objective_is_refused():
    with pytest.raises(ValueError, match="exactly one objective"):
        BodySpec(activation="softplus", loss="mse", family="nb")
    with pytest.raises(ValueError, match="exactly one objective"):
        BodySpec(activation="softplus")


# ──────────────────────────────── width is derived, not declared ────────────────────────────────


@pytest.mark.parametrize(
    "name,width",
    [
        ("mse_softplus_point", 1),
        ("listwise_softplus_point", 1),
        ("nll_softplus_nb", 2),
        ("nll_softplus_truncated_nb", 2),
        ("nll_softplus_mixture_nb", 5),
    ],
)
def test_the_declared_width_matches_adr_079s_table(name, width):
    assert BODIES[name].width == width


def test_width_is_read_off_the_distribution_registry_not_stored():
    """ADR-081 §2: the width is the emit distribution's parameter count, with no second home.

    Asserted against the registry rather than against a literal, so the two cannot drift.
    """
    for name, spec in BODIES.items():
        if spec.family is not None:
            assert spec.width == FAMILY_N_PARAMS[spec.family], name
        else:
            assert spec.width == 1, name
    assert not any(hasattr(spec, "__dict__") and "width" in vars(spec) for spec in BODIES.values())


# ─────────────────────────── every declared value names something real ───────────────────────────


def test_every_declared_loss_is_a_registry_key():
    from views_hydranet.utils.utils import LOSS_REG_REGISTRY

    for name, spec in BODIES.items():
        if spec.loss is not None:
            assert spec.loss in LOSS_REG_REGISTRY, name


def test_every_declared_family_is_a_registry_key():
    for name, spec in BODIES.items():
        if spec.family is not None:
            assert spec.family in DISTRIBUTION_REGISTRY, name


def test_every_declared_body_builds_a_working_criterion():
    """⛔ #602: a value that prints but does not train is the defect being prevented.

    Goes through `choose_loss` in the OLD spelling — the vocabulary is not wired into the config
    yet (that is #606), so this proves the rows name reachable objectives, not that a config works.
    """
    torch = pytest.importorskip("torch")
    from views_hydranet.utils.utils import choose_loss

    base = {
        "loss_class": "bce",
        "regression_targets": ["a", "b", "c"],
        "classification_targets": ["x", "y", "z"],
        "loss_reg_y_max": 10000.0,
    }
    for name, spec in BODIES.items():
        cfg = {
            **base,
            "loss_reg": spec.loss or "mse",
            "output_distribution": spec.family or "standard",
        }
        criterion_reg, _, _ = choose_loss(cfg, torch.device("cpu"))
        assert criterion_reg is not None, name
        params = torch.rand(4, 3, spec.width) if spec.family else torch.rand(4, 3)
        target = torch.rand(4, 3)
        out = criterion_reg(params, target) if spec.family else criterion_reg(params, target)
        assert torch.isfinite(out).all(), name


# ──────────────────────────────────── the refusal (#602) ────────────────────────────────────


def test_the_printed_set_equals_the_executing_set():
    """⛔ #602's real requirement: SET EQUALITY, not membership.

    A membership assertion survives a message that advertises a value the vocabulary does not
    contain — which is exactly the failure #602 was filed about.
    """
    with pytest.raises(ValueError) as excinfo:
        resolve_body("listwise_softplus_nb")
    printed = set(re.findall(r"^\s{2}(\S+)", str(excinfo.value), flags=re.MULTILINE))
    assert printed == set(BODIES)


def test_the_rejection_message_names_the_offending_value():
    with pytest.raises(ValueError, match="listwise_softplus_nb"):
        resolve_body("listwise_softplus_nb")


@pytest.mark.parametrize(
    "orphan", ["zinb", "hurdle_nb", "hurdle_lognormal", "hurdle_shrinkage", "dense_nb"]
)
def test_an_orphan_distribution_is_not_a_declared_body(orphan):
    """The six values that keep their code and get no row (owner decision 2026-11-15)."""
    assert orphan not in BODIES
    with pytest.raises(ValueError, match="not a declared body"):
        resolve_body(orphan)


def test_every_declared_value_appears_in_the_gloss_listing():
    listing = declared_bodies()
    for name in BODIES:
        assert name in listing


def test_resolve_body_returns_the_spec_for_each_declared_value():
    for name, spec in BODIES.items():
        assert resolve_body(name) is spec


# ─────────────────────────────── the quarantined legacy rule ───────────────────────────────


@pytest.mark.parametrize(
    "loss_reg,output_distribution,reg_activation,becomes",
    [
        ("mse", "standard", "softplus", "mse_softplus_point"),
        ("listwise_level", "standard", "softplus", "listwise_softplus_point"),
        ("mse", "nb", None, "nll_softplus_nb"),
        ("mse", "truncated_nb", None, "nll_softplus_truncated_nb"),
        ("mse", "mixture_nb", None, "nll_softplus_mixture_nb"),
    ],
)
def test_legacy_body_spec_reproduces_the_rename_table(
    loss_reg, output_distribution, reg_activation, becomes
):
    """ADR-081 §2's rename table, as an equality between the old spelling and the new."""
    assert legacy_body_spec(loss_reg, output_distribution, reg_activation) == BODIES[becomes]


def test_legacy_body_spec_discards_loss_reg_when_a_family_is_named():
    """The precedence, in the one place it still lives. The characterization file pins it too."""
    assert legacy_body_spec("listwise_level", "nb") == BODIES["nll_softplus_nb"]
    assert legacy_body_spec("tobit", "mixture_nb") == BODIES["nll_softplus_mixture_nb"]


@pytest.mark.parametrize(
    "output_distribution,expected_activation",
    [
        ("standard", "relu"),
        ("hurdle_nb", "softplus"),
        ("hurdle_lognormal", "softplus"),
        ("hurdle_shrinkage", "softplus"),
        ("dense_nb", "relu"),
    ],
)
def test_legacy_body_spec_mirrors_the_architectures_activation_default(
    output_distribution, expected_activation
):
    """The `startswith("hurdle")` rule, copied from HydraBNrecurrentUnet_06_LSTM4.py:118-127."""
    spec = legacy_body_spec("mse", output_distribution, None)
    assert spec.activation == expected_activation


def test_a_family_head_ignores_whatever_the_activation_field_said():
    """The field is not consulted when a family is named, so `relu` beside `nb` changes nothing.

    The declared body for such a config is `nll_softplus_<family>`, and its activation reads
    `softplus` — the name's own middle segment — because that is what the vocabulary declares. The
    family still applies the map itself; see `HydraBNrecurrentUnet_06_LSTM4.py:118-119`.
    """
    for family in ("nb", "truncated_nb", "mixture_nb"):
        assert legacy_body_spec("mse", family, "relu") == legacy_body_spec("mse", family, None)
        assert legacy_body_spec("mse", family, "relu").activation == "softplus"


def test_every_declared_body_names_the_activation_its_own_value_spells():
    """⛔ The code and ADR-081 §2's table must agree, value by value.

    Every declared value's middle segment is `softplus`, so every spec's activation is `softplus`.
    A row whose stored activation disagreed with its own name would be the same class of defect
    this ADR removes: one thing said in two places, free to drift.
    """
    for name, spec in BODIES.items():
        assert f"_{spec.activation}_" in name, name


def test_legacy_body_spec_is_total_over_the_orphans():
    for orphan in ("zinb", "hurdle_nb", "hurdle_lognormal", "hurdle_shrinkage", "dense_nb"):
        assert legacy_body_spec("mse", orphan) is not None


# ──────────────────────────────────── the CRP property ────────────────────────────────────


def test_the_vocabulary_module_imports_no_torch():
    """`config_initializer` imports this, and the schema layer must stay torch-free (CRP)."""
    code = "import views_hydranet.body, sys; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout
