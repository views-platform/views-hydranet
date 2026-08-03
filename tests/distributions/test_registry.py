"""A-S2 (#169): the explicit distribution registry + resolve_family seam (ADR-067).

Validation command (#169): `pytest tests/distributions/test_registry.py -q`.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.fixture
def registry():
    """Yield the registry module with the DISTRIBUTION_REGISTRY dict restored after each test."""
    from views_hydranet.distributions import registry as reg

    snapshot = dict(reg.DISTRIBUTION_REGISTRY)
    try:
        yield reg
    finally:
        reg.DISTRIBUTION_REGISTRY.clear()
        reg.DISTRIBUTION_REGISTRY.update(snapshot)


def test_get_family_unknown_raises_fail_loud_with_available_list(registry):
    with pytest.raises(ValueError, match="not a registered distribution family"):
        registry.get_family("does_not_exist")


def test_resolve_family_returns_none_for_legacy_output_distribution_values(registry):
    for legacy in ("standard", "hurdle_nb", "hurdle_shrinkage", "dense_nb", "quantile"):
        assert registry.resolve_family(legacy) is None


def test_registering_one_entry_flows_end_to_end_no_other_edits(registry):
    """OCP proof: a family is reachable purely by adding one registry entry."""

    class _Dummy:
        n_params = 1

    registry.DISTRIBUTION_REGISTRY["dummy"] = lambda: _Dummy()
    assert isinstance(registry.get_family("dummy"), _Dummy)
    assert isinstance(registry.resolve_family("dummy"), _Dummy)
    assert "dummy" in registry.family_names()


def test_family_names_is_derived_from_registry_keys(registry):
    assert registry.family_names() == frozenset(registry.DISTRIBUTION_REGISTRY)
    registry.DISTRIBUTION_REGISTRY["dummy"] = lambda: object()
    assert registry.family_names() == frozenset(registry.DISTRIBUTION_REGISTRY)


def test_importing_the_registry_does_not_import_torch():
    """Config reads family_names() torch-free (CRP): registry import must not pull torch."""
    code = (
        "import sys; import views_hydranet.distributions.registry as r; "
        "assert 'torch' not in sys.modules, "
        "sorted(m for m in sys.modules if m == 'torch' or m.startswith('torch.'))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, (
        f"registry import pulled torch:\n{result.stdout}\n{result.stderr}"
    )


def test_self_zeroed_mirror_matches_class_attribute():
    """ADR-069: SELF_ZEROED_FAMILIES (torch-free mirror, read by config) must equal the ground
    truth from each family's `self_zeroed` class attribute — so the two cannot silently drift."""
    from views_hydranet.distributions import family_names, get_family
    from views_hydranet.distributions.registry import self_zeroed_family_names

    truth = frozenset(n for n in family_names() if get_family(n).self_zeroed)
    assert self_zeroed_family_names() == truth
