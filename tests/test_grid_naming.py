"""TDD: grid-name-agnostic loading (GH #144) — the platform priogrid_gid -> priogrid_id flip.

views-hydranet reads the cached parquet DIRECTLY off disk (data_fetcher.py:59), bypassing
pipeline-core's normalization, so it must resolve the grid entity from the DATA, not a hardcoded
literal or a (possibly stale) config. `grid_id_col` is the single rule — name-set membership,
fail-loud. `standardize_raw_df` and `mcr_readout.load_truth_index` must then tolerate EITHER grid
name even when the config still says the legacy `priogrid_gid`.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("views_pipeline_core")

from views_hydranet.utils.data_fetcher import DataFetcher  # noqa: E402
from views_hydranet.utils.grid_naming import GRID_ID_ALIASES, grid_id_col  # noqa: E402


# ---- grid_id_col: the single rule (semantic, fail-loud) ----
def test_resolves_legacy_and_canonical_names():
    assert grid_id_col(["month_id", "priogrid_gid"]) == "priogrid_gid"
    assert grid_id_col(["month_id", "priogrid_id"]) == "priogrid_id"


def test_works_on_columns_not_just_index_levels():
    assert grid_id_col(pd.Index(["c_id", "priogrid_id", "row"])) == "priogrid_id"


def test_fail_loud_when_no_grid_col():
    with pytest.raises(ValueError):
        grid_id_col(["month_id", "country_id"])


def test_fail_loud_when_ambiguous():
    with pytest.raises(ValueError):
        grid_id_col(["priogrid_gid", "priogrid_id"])


def test_canonical_listed_first():
    assert GRID_ID_ALIASES[0] == "priogrid_id"


# ---- characterization: standardize tolerates EITHER grid name with a STALE config ----
def _df(grid_name, n_t=4, n_c=4, ocean=False):
    times = np.repeat(np.arange(1, n_t + 1), n_c)
    cells = np.tile(np.arange(1, n_c + 1), n_t).astype(float)
    if ocean:
        cells[0] = 0.0  # an ocean cell (grid id 0)
    idx = pd.MultiIndex.from_arrays([times, cells], names=["month_id", grid_name])
    return pd.DataFrame(
        {"lr_ged_sb": np.random.rand(len(times)), "feat_a": np.random.rand(len(times))},
        index=idx,
    )


_STALE_CFG = {
    "index_names": ["month_id", "priogrid_gid"],  # config still says LEGACY (pre-tidy)
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "derivations": {},
}


@pytest.mark.parametrize("grid_name", ["priogrid_gid", "priogrid_id"])
def test_standardize_tolerates_either_grid_name_with_stale_config(grid_name):
    out = DataFetcher.standardize_raw_df(_df(grid_name), dict(_STALE_CFG))
    assert grid_name in out.columns  # the DERIVED grid col survives reset_index


@pytest.mark.parametrize("grid_name", ["priogrid_gid", "priogrid_id"])
def test_ocean_filter_uses_the_derived_grid_col(grid_name):
    out = DataFetcher.standardize_raw_df(_df(grid_name, ocean=True), dict(_STALE_CFG))
    assert (out[grid_name] > 0).all()  # grid==0 ocean dropped via the derived col, not the config
