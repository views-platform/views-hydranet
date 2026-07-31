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
    vh = VolumeHandler.from_df(
        _grid_df(), _vol_config(static=[_POP, seam_row_derivation])
    )
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
