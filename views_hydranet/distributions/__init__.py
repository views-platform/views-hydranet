"""Output-distribution families (ADR-067).

The stable, torch-free core: the ``DistributionFamily`` abstraction (``base``) and the explicit
registry (``registry``). The concrete families (``nb``, ``zinb``, ``mixture_nb``, ``truncated_nb``)
and the shared ``NBCore`` are imported lazily, so importing this package for ``family_names`` stays
torch-free.
"""

from views_hydranet.distributions.registry import (
    DISTRIBUTION_REGISTRY,
    family_names,
    get_family,
    resolve_family,
)

__all__ = ["DISTRIBUTION_REGISTRY", "family_names", "get_family", "resolve_family"]
