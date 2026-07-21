"""Single source of truth for the VIEWS grid-entity column name (GH #144).

The platform is consolidating the PGM grid index name (views-frames ADR-015): the legacy
``priogrid_gid`` (UCDP/PRIO-GRID source name) and the canonical ``priogrid_id`` are both in
circulation. views-hydranet reads cached parquet DIRECTLY off disk (data_fetcher reads
``data/raw/*.parquet`` without going through pipeline-core's normalization), so it must resolve the
grid entity from the DATA — not a hardcoded literal or a (possibly stale) model config.

This module is that one rule. Its load-bearing consumers — ``data_fetcher`` (load path) and
``mcr_readout`` (prediction↔truth join key) — depend on it so the name lives in exactly one place.
"""

from __future__ import annotations

from typing import Iterable

# Canonical first (views-frames ADR-015), legacy second. Membership is what matters, not order.
GRID_ID_ALIASES = ("priogrid_id", "priogrid_gid")


def grid_id_col(names: Iterable[str]) -> str:
    """Resolve the grid-entity column among ``names`` (index levels OR DataFrame columns).

    Resolves by NAME-SET membership, not position — a reordered or extra index level cannot
    silently mis-resolve (a positional ``names[1]`` could). Fail-loud: raises if zero or more than
    one alias is present, preserving the load path's existing loud-failure contract.
    """
    names = list(names)
    found = [n for n in names if n in GRID_ID_ALIASES]
    if len(found) != 1:
        raise ValueError(
            f"grid id column: expected exactly one of {GRID_ID_ALIASES}, found {found} in {names}"
        )
    return found[0]


def canonicalize_config_grid_name(config: dict, grid: str) -> None:
    """Rewrite ``config``'s grid-entity keys to ``grid`` — the alias the DATA uses (GH #144).

    ``data_fetcher`` already resolves the grid name from the data for its own processing, but the
    ``DataSniffer`` and ``VolumeHandler`` read ``config['id_col']`` / ``identity_cols`` /
    ``index_names`` **literally** — so a stale ``priogrid_gid`` config rejects ``priogrid_id`` data
    downstream. This closes that gap: only ``GRID_ID_ALIASES`` members are replaced (other columns
    are untouched), it mutates ``config`` in place, and it is a **no-op** when the config already
    matches (so ``priogrid_gid`` data + ``priogrid_gid`` config stays byte-identical). Missing keys
    are tolerated.
    """
    if config.get("id_col") in GRID_ID_ALIASES:
        config["id_col"] = grid
    for key in ("identity_cols", "index_names"):
        seq = config.get(key)
        if seq is not None:
            config[key] = [grid if x in GRID_ID_ALIASES else x for x in seq]
