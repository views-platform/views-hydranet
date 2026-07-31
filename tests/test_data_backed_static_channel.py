"""Data-backed static channels (Epic #203 S7) — extend the ADR-060 seam so a static channel can be
sourced from a df COLUMN (e.g. per-cell ln_pop) instead of derived from grid geometry.

A data-backed static channel keeps every ADR-060 invariant of a geometry static channel: input-
only, never a target, re-injected unchanged each AR step (constant per cell), window-sliced +
flip-synced, never inverse-transformed. The ONLY difference is where its per-cell values come from
(the df, not `derive(geom)`). ADR-005 Green/Beige/Red.
"""

import numpy as np
import pandas as pd
import pytest

from views_hydranet.utils import static_channels as sc
from views_hydranet.utils.volume_handler import VolumeHandler

_FEATURES = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
_POP = "ln_pop"  # data-backed static channel: a df column, constant per cell across time


def _grid_df(height: int = 4, width: int = 4, t: int = 3, with_pop: bool = True) -> pd.DataFrame:
    rows = []
    for m in range(t):
        for r in range(height):
            for c in range(width):
                row = {
                    "month_id": 500 + m,
                    "priogrid_gid": 1000 + r * width + c,
                    "c_id": r * width + c,
                    "row": r,  # row_offset=0 => r_idx == r
                    "col": c,
                    "lr_ged_sb": float(r),
                    "lr_ged_ns": float(c),
                    "lr_ged_os": 0.0,
                }
                if with_pop:
                    # constant per cell across months (a static per-cell covariate)
                    row[_POP] = float(r * width + c)
                rows.append(row)
    return pd.DataFrame(rows)


def _vol_config(height: int = 4, width: int = 4, static: list[str] | None = None) -> dict:
    return {
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"],
        "identity_cols": ["c_id", "row", "col"],
        "features": list(_FEATURES),
        "static_channels": static or [],
        "row_offset": 0,
        "col_offset": 0,
        "height": height,
        "width": width,
    }


def test_data_backed_static_filled_from_df_and_flip_synced():
    """A static channel that is a df column fills from the df per cell, and flips in sync with the
    dynamic channels (same fill path)."""
    height = 4
    vh = VolumeHandler.from_df(_grid_df(height=height), _vol_config(height=height, static=[_POP]))
    ci = vh.channel_map.index(_POP)
    static = np.asarray(vh.data)[:, :, :, ci]  # [T, H, W], post-flip
    # pre-flip value at (r,c) == r*width+c; North-Up flip on H => post-flip row i holds row (H-1-i)
    width = 4
    pre = (np.arange(height)[:, None] * width + np.arange(width)[None, :]).astype(float)
    expected_postflip = pre[::-1, :]
    assert np.allclose(static[0], expected_postflip), "data-backed static: from df, flip-synced"


def test_data_backed_static_constant_across_time():
    """I3: a data-backed static is constant across time (the df column is constant per cell) and
    reaches the model (in feature_cols)."""
    vh = VolumeHandler.from_df(_grid_df(t=3), _vol_config(static=[_POP]))
    ci = vh.channel_map.index(_POP)
    static = np.asarray(vh.data)[:, :, :, ci]
    assert np.allclose(static[0], static[1]) and np.allclose(static[1], static[2])
    assert _POP in vh.feature_cols  # kept => reaches the model


def test_data_backed_and_geometry_static_coexist(seam_row_derivation):
    """A data-backed static (df column) and a geometry static (registry) can both be declared."""
    vh = VolumeHandler.from_df(_grid_df(), _vol_config(static=[_POP, seam_row_derivation]))
    assert _POP in vh.channel_map and seam_row_derivation in vh.channel_map
    ci_pop = vh.channel_map.index(_POP)
    ci_geo = vh.channel_map.index(seam_row_derivation)
    assert not np.allclose(
        np.asarray(vh.data)[0, :, :, ci_pop], np.asarray(vh.data)[0, :, :, ci_geo]
    )  # distinct sources → distinct values


def test_unknown_static_channel_fails_loud():
    """A static channel that is neither a df column nor a geometry derivation fails loud."""
    with pytest.raises((ValueError, KeyError)):
        VolumeHandler.from_df(
            _grid_df(with_pop=False), _vol_config(static=["not_a_column_or_derivation"])
        )


def test_data_backed_static_with_gap_fails_loud():
    """C-235/C-236 (S4): a data-backed static with a coverage gap (NaN/inf in its df column — e.g.
    population not joined for some cell) must FAIL LOUD, not silently enter the model as a 0-hole
    or an unguarded NaN in the encoder."""
    df = _grid_df()
    # punch a coverage hole: one cell's population is missing (NaN)
    df.loc[df.index[0], _POP] = np.nan
    with pytest.raises(ValueError, match="non-finite|NaN|coverage|hole|complete"):
        VolumeHandler.from_df(df, _vol_config(static=[_POP]))


def test_data_backed_static_with_inf_fails_loud():
    """C-236 (S4): a non-finite (inf) data-backed static value also fails loud."""
    df = _grid_df()
    df.loc[df.index[0], _POP] = np.inf
    with pytest.raises(ValueError, match="non-finite|inf|complete"):
        VolumeHandler.from_df(df, _vol_config(static=[_POP]))


def test_data_backed_static_raw_magnitude_fails_loud():
    """C-236 (S5): a data-backed static arriving RAW/unscaled (e.g. population in the millions)
    would dominate the log1p-scaled encoder inputs. A sanity guard fails loud, steering the
    operator to a pre-scaled covariate (e.g. ln_pop). Deeper model-side scaling tracked apart."""
    df = _grid_df()
    df[_POP] = 5.0e6  # raw population magnitude, constant per cell (well-formed but unscaled)
    with pytest.raises(ValueError, match="raw|unscaled|dominate|pre-scale|magnitude"):
        VolumeHandler.from_df(df, _vol_config(static=[_POP]))


def test_data_backed_static_prescaled_ok():
    """C-236 (S5): a pre-scaled data-backed static (log/standardized, model-range magnitude) passes
    the sanity guard."""
    df = _grid_df()  # _POP values are 0..15 (model-range) -> fine
    vh = VolumeHandler.from_df(df, _vol_config(static=[_POP]))
    assert _POP in vh.channel_map


def test_data_backed_static_time_varying_fails_loud():
    """C-238 (S6): a static must be CONSTANT per cell across time. A time-varying df column set
    as a static is digested varying during history but pinned to the origin in the rollout (ADR-060
    I3) — treated two ways in one inference. Fail loud rather than silently mis-handle it.
    Decision: reject (the full dynamic-covariate path is out of scope — C-229)."""
    df = _grid_df()
    # make one cell's value differ across its months (non-constant per cell)
    df.loc[df.index[0], _POP] = float(df[_POP].iloc[0]) + 99.0
    with pytest.raises(ValueError, match="constant|time-varying|per cell|per-cell|vary"):
        VolumeHandler.from_df(df, _vol_config(static=[_POP]))


def test_data_backed_static_constant_per_cell_ok():
    """C-238 (S6): a genuinely constant-per-cell static (population at a fixed vintage) passes."""
    vh = VolumeHandler.from_df(_grid_df(t=3), _vol_config(static=[_POP]))  # _POP constant per cell
    assert _POP in vh.channel_map


def test_registry_and_df_collision_fails_loud(seam_row_derivation):
    """C-237 (S3): a static-channel name that is BOTH a registered geometry derivation AND a df
    column is ambiguous — role must be resolved authoritatively by the registry and the collision
    must FAIL LOUD, not silently take the raw df column (which would feed a different, unnormalized
    channel to the model with zero diagnostic)."""
    df = _grid_df()
    df[seam_row_derivation] = 7.0  # the registered geometry name now ALSO exists as a df column
    with pytest.raises(ValueError, match="both|ambiguous|registered.*df|collision"):
        VolumeHandler.from_df(df, _vol_config(static=[seam_row_derivation]))


def test_registered_geometry_static_is_derived_not_df_sourced(seam_row_derivation):
    """C-237 (S3): classification is by the REGISTRY, not df-column presence. A registered name
    absent from the df is geometry-derived (uses derive(), not the df)."""
    # seam_row_derivation returns row index r broadcast over cols; distinct from any df column.
    vh = VolumeHandler.from_df(_grid_df(), _vol_config(static=[seam_row_derivation]))
    ci = vh.channel_map.index(seam_row_derivation)
    static = np.asarray(vh.data)[0, :, :, ci]  # post-flip [H, W]
    height, width = 4, 4
    pre = np.broadcast_to(np.arange(height, dtype=float)[:, None], (height, width))
    assert np.allclose(static, pre[::-1, :]), "registered static must be geometry-derived"


@pytest.fixture
def seam_row_derivation():
    def _row(geom: sc.GridGeometry) -> np.ndarray:
        return np.broadcast_to(
            np.arange(geom.height, dtype=np.float32)[:, None], (geom.height, geom.width)
        ).copy()

    name = "_seam_test_row_db"
    sc.STATIC_CHANNEL_DERIVATIONS[name] = _row
    yield name
    sc.STATIC_CHANNEL_DERIVATIONS.pop(name, None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
