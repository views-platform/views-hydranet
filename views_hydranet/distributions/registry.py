"""Explicit registry of output-distribution families (ADR-067). **Torch-free at import.**

A new family is added in exactly ONE place: one entry in ``DISTRIBUTION_REGISTRY`` mapping its
config name to a *lazy factory* that ``importlib``-imports the torch-heavy family module only when
called. ``family_names()`` derives the valid-name set from the keys — one source of truth.
``resolve_family()`` is the single dispatch seam the strangler-fig wiring (A-S6..A-S8) calls.

Why explicit map (not a decorator): fail-loud discovery, order-independence, and one readable
source of truth (v1 /expert-code-review; mirrors ``utils.LOSS_REG_REGISTRY``).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from views_hydranet.distributions.base import DistributionFamily


def _lazy(module: str, cls: str) -> Callable[[], "DistributionFamily"]:
    """Return a factory that imports ``views_hydranet.distributions.<module>.<cls>`` on first call.

    Deferring the import keeps this module (and therefore ``config_initializer``) torch-free until
    a family is actually instantiated.
    """

    def factory() -> "DistributionFamily":
        mod = importlib.import_module(f"views_hydranet.distributions.{module}")
        return getattr(mod, cls)()

    return factory


#: name -> lazy factory.
DISTRIBUTION_REGISTRY: dict[str, Callable[[], "DistributionFamily"]] = {
    "nb": _lazy("negative_binomial", "NegativeBinomialFamily"),  # A-S3 (#170)
    "zinb": _lazy("zero_inflated_negative_binomial", "ZINBFamily"),  # A-S4 (#171)
}


#: Torch-free MIRROR of ``DistributionFamily.self_zeroed`` — names of families that produce their
#: own zeros structurally (no external gate). The class attribute is the AUTHORITY; this mirror
#: exists only so ``config_initializer`` (which must stay torch-free — CRP) can enforce the ADR-069
#: rules without importing a family. A parity test (``test_registry``) asserts this set equals
#: ``{n for n in family_names() if get_family(n).self_zeroed}`` so they cannot drift.
SELF_ZEROED_FAMILIES: frozenset[str] = frozenset({"zinb"})


def family_names() -> frozenset[str]:
    """The set of registered family names — the single source of truth (torch-free)."""
    return frozenset(DISTRIBUTION_REGISTRY)


def self_zeroed_family_names() -> frozenset[str]:
    """Registered self-zeroed families (torch-free mirror of the ``self_zeroed`` attribute)."""
    return SELF_ZEROED_FAMILIES


def get_family(name: str) -> "DistributionFamily":
    """Instantiate the family registered under ``name``; fail-loud on an unknown name."""
    factory = DISTRIBUTION_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(sorted(DISTRIBUTION_REGISTRY)) or "(none registered)"
        raise ValueError(
            f"'{name}' is not a registered distribution family. Available: {available}."
        )
    return factory()


def resolve_family(name: str) -> "DistributionFamily | None":
    """The strangler-fig seam: return the family for a registered ``name``, else ``None``.

    A ``None`` return means ``name`` is not a registered family (e.g. an existing
    ``output_distribution`` value) — the caller falls through to the existing code path, unchanged.
    """
    factory = DISTRIBUTION_REGISTRY.get(name)
    return factory() if factory is not None else None
