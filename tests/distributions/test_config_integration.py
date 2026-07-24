"""A-S5 (#172): config boundary for the distribution families (ADR-067).

Validates that HydraNetConfig accepts `nb`/`zinb` as `output_distribution` (registry-driven,
unioned with the legacy values), the new `n_head_samples` (K) field + its cross-field rule, the
C-197 family/legacy disjointness guard, and that `config_initializer` stays torch-free (CRP).
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from views_hydranet.utils.config_initializer import (
    LEGACY_OUTPUT_DISTRIBUTIONS,
    HydraNetConfig,
)


def _cfg(valid_config_dict, **overrides):
    cfg = dict(valid_config_dict)
    cfg.update(overrides)
    return cfg


# ── output_distribution: families accepted, legacy preserved, unknown rejected ──


@pytest.mark.parametrize("fam", ["nb", "zinb"])
def test_family_output_distributions_accepted(valid_config_dict, fam):
    # ADR-069/#183: nb must declare a gate composition; zinb keeps the default self_zeroed.
    comp = "soft_gate" if fam == "nb" else "self_zeroed"
    cfg = _cfg(valid_config_dict, output_distribution=fam, forecast_composition=comp)
    assert HydraNetConfig(**cfg).output_distribution == fam


@pytest.mark.parametrize(
    "legacy",
    ["standard", "hurdle_nb", "hurdle_lognormal", "hurdle_shrinkage", "dense_nb", "quantile"],
)
def test_legacy_output_distributions_still_accepted(valid_config_dict, legacy):
    cfg = _cfg(valid_config_dict, output_distribution=legacy)
    assert HydraNetConfig(**cfg).output_distribution == legacy


def test_unknown_output_distribution_rejected_with_union_list(valid_config_dict):
    cfg = _cfg(valid_config_dict, output_distribution="not_a_family")
    with pytest.raises(ValidationError, match="output_distribution.*not valid") as excinfo:
        HydraNetConfig(**cfg)
    # the rejection message actually lists the family names (the union), not just the legacy set
    msg = str(excinfo.value)
    assert "'nb'" in msg and "'zinb'" in msg


def test_legacy_config_constructs_unchanged_with_default_k(valid_config_dict):
    """AC #3 (byte-identical): a legacy config (no output_distribution, no K) validates unchanged;
    the new n_head_samples defaults to 1 without disturbing any existing field."""
    assert "output_distribution" not in valid_config_dict
    assert "n_head_samples" not in valid_config_dict
    cfg = HydraNetConfig(**valid_config_dict)
    assert cfg.output_distribution == "standard"  # legacy default preserved
    assert cfg.n_head_samples == 1  # new field defaults, no behaviour change
    assert cfg.loss_reg == valid_config_dict["loss_reg"]  # unrelated fields untouched
    assert cfg.n_posterior_samples == valid_config_dict["n_posterior_samples"]


def test_default_legacy_head_with_k_gt_1_raises(valid_config_dict):
    """K>1 must raise even for the default (legacy) output_distribution."""
    cfg = _cfg(valid_config_dict, n_head_samples=3)  # output_distribution defaults to "standard"
    with pytest.raises(ValidationError, match="n_head_samples"):
        HydraNetConfig(**cfg)


# ── n_head_samples (K) field + cross-field rule ──


def test_n_head_samples_defaults_to_one(valid_config_dict):
    assert "n_head_samples" not in valid_config_dict
    assert HydraNetConfig(**valid_config_dict).n_head_samples == 1


@pytest.mark.parametrize("bad_k", [0, -1])
def test_n_head_samples_below_one_rejected(valid_config_dict, bad_k):
    cfg = _cfg(valid_config_dict, n_head_samples=bad_k)
    with pytest.raises(ValidationError):
        HydraNetConfig(**cfg)


@pytest.mark.parametrize("legacy", ["standard", "hurdle_nb", "quantile"])
def test_k_gt_1_on_legacy_head_raises(valid_config_dict, legacy):
    cfg = _cfg(valid_config_dict, output_distribution=legacy, n_head_samples=3)
    with pytest.raises(ValidationError, match="n_head_samples"):
        HydraNetConfig(**cfg)


@pytest.mark.parametrize("fam", ["nb", "zinb"])
def test_k_gt_1_on_family_accepted(valid_config_dict, fam):
    comp = "soft_gate" if fam == "nb" else "self_zeroed"  # ADR-069/#183: nb must declare a gate
    cfg = _cfg(valid_config_dict, output_distribution=fam, n_head_samples=3,
               forecast_composition=comp)
    assert HydraNetConfig(**cfg).n_head_samples == 3


def test_family_k1_is_unaffected_by_the_cross_field_rule(valid_config_dict):
    # K=1 on a family (or a legacy head) never trips the K>1 rule
    cfg = _cfg(valid_config_dict, output_distribution="nb", n_head_samples=1,
               forecast_composition="soft_gate")  # ADR-069/#183: nb must declare a gate
    assert HydraNetConfig(**cfg).n_head_samples == 1


# ── C-197: family names must be disjoint from legacy output_distribution values ──


def test_family_names_are_disjoint_from_legacy():
    from views_hydranet.distributions import family_names

    assert family_names().isdisjoint(LEGACY_OUTPUT_DISTRIBUTIONS)


def test_registry_name_colliding_with_legacy_fails_config_load(valid_config_dict, monkeypatch):
    """C-197: if a registered family name equals a legacy value, config load must fail loud."""
    from views_hydranet.distributions import registry

    monkeypatch.setitem(registry.DISTRIBUTION_REGISTRY, "standard", lambda: None)  # collision
    with pytest.raises(ValidationError, match="collides|disjoint|C-197"):
        HydraNetConfig(**valid_config_dict)


# ── CRP: config stays torch-free even though it now reads family_names() ──


def test_importing_config_initializer_does_not_import_torch():
    code = (
        "import sys; import views_hydranet.utils.config_initializer; "
        "assert 'torch' not in sys.modules, "
        "sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"config_initializer import pulled torch:\n{result.stdout}\n{result.stderr}"
    )
