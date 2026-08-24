"""Architecture registry — the extension seam for new architectures (#258 spatial precision).

`choose_model` was a hardcoded ``if config["model"] == ...`` chain with a single branch. Adding an
architecture meant editing the dispatcher, which is the OCP violation that a bake-off of six
candidates would multiply by six. This mirrors `views_hydranet.distributions.registry` exactly:
a name -> lazy-factory dict, so importing the registry does not import torch or every architecture.

**Contract every registered architecture must satisfy** (pinned by
`tests/architectures/test_architecture_registry.py`):

* constructor ``(input_channels, total_hidden_channels, output_channels, dropout_rate, *,
  output_distribution, n_static_channels, static_top_skip, reg_activation, n_quantiles)`` —
  the incumbent's signature, so `choose_model` stays uniform;
* ``forward(x, h) -> ModelOutput`` with ``reg`` of width ``n_targets * n_params``, ``cls`` of width
  ``n_targets``, and ``h_next`` the same shape as ``h``;
* ``total_hidden_channels`` divisible by **8** — `blend_recurrent_state` splits the state into
  4 short-term + 4 long-term groups, and the state-freeze diagnostics (M38/M39) silently
  mis-assign memory types otherwise;
* a ``base`` attribute, because `init_hTtime` and every caller size the state from it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch.nn as nn


def _lazy(module: str, cls: str) -> Callable[[], "type[nn.Module]"]:
    """Return a factory that imports `module` and yields the class `cls` on first call."""

    def factory() -> "type[nn.Module]":
        mod = importlib.import_module(f"views_hydranet.architectures.{module}")
        return getattr(mod, cls)

    return factory


#: name -> lazy factory returning the CLASS (not an instance; architectures take ctor args).
ARCHITECTURE_REGISTRY: dict[str, Callable[[], "type[nn.Module]"]] = {
    # the incumbent
    "HydraBNUNet06_LSTM4": _lazy("HydraBNrecurrentUnet_06_LSTM4", "HydraBNUNet06_LSTM4"),
    # 2026-08-24 spatial-precision bake-off candidates (dossier 02_design)
    "AntiAliasedPool": _lazy("anti_aliased_pool", "AntiAliasedPool"),
    "DynamicTopSkip": _lazy("dynamic_skip", "DynamicTopSkip"),
    "FiLMSkip": _lazy("dynamic_skip", "FiLMSkip"),
    "ShallowPool": _lazy("shallow_pool", "ShallowPool"),
    "DualStream": _lazy("dual_stream", "DualStream"),
    "WideMemory": _lazy("wide_memory", "WideMemory"),
}


def architecture_names() -> frozenset[str]:
    """Every registered architecture name."""
    return frozenset(ARCHITECTURE_REGISTRY)


def get_architecture(name: str) -> "type[nn.Module]":
    """Resolve an architecture class by name, or raise naming what IS available.

    Fails loud rather than returning None: a typo in `config['model']` must stop the run at
    construction, not produce a silently different network hours into training.
    """
    factory = ARCHITECTURE_REGISTRY.get(name)
    if factory is None:
        available = ", ".join(sorted(ARCHITECTURE_REGISTRY)) or "(none registered)"
        raise ValueError(f"Unknown model type: {name!r}. Registered architectures: {available}.")
    return factory()
