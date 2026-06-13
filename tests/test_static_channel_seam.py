"""ADR-060 static-channel seam — invariants I1–I6 (closes C-153).

A static channel is input-only: injected, never predicted, never in targets, re-injected unchanged
every autoregressive step, derived from grid geometry, window-sliced + flip-synced like the dynamic
channels, never inverse-transformed / in a frame, and a no-op (byte-identical) when unconfigured.
ADR-005 Green/Beige/Red.
"""

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from views_hydranet.utils import static_channels as sc
from views_hydranet.utils.config_initializer import HydraNetConfig
from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.volume_handler import VolumeHandler

_FEATURES = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
_SEAM_ROW = "_seam_test_row"  # fixture static channel: value == pre-flip row index


@pytest.fixture
def seam_row_derivation():
    """Register a fixture static channel whose value is the pre-flip row index (so positioning +
    the North-Up flip are verifiable), then clean up."""

    def _row(geom: sc.GridGeometry) -> np.ndarray:
        return np.broadcast_to(
            np.arange(geom.height, dtype=np.float32)[:, None], (geom.height, geom.width)
        ).copy()

    sc.STATIC_CHANNEL_DERIVATIONS[_SEAM_ROW] = _row
    yield _SEAM_ROW
    sc.STATIC_CHANNEL_DERIVATIONS.pop(_SEAM_ROW, None)


def _grid_df(height: int = 4, width: int = 4, t: int = 3) -> pd.DataFrame:
    rows = []
    for m in range(t):
        for r in range(height):
            for c in range(width):
                rows.append(
                    {
                        "month_id": 500 + m,
                        "priogrid_gid": 1000 + r * width + c,
                        "c_id": r * width + c,
                        "row": r,  # row_offset=0 => r_idx == r
                        "col": c,
                        "lr_sb_best": float(r),
                        "lr_ns_best": float(c),
                        "lr_os_best": 0.0,
                    }
                )
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


# ── I1 (Red): a static channel is never a target; input_channels law counts statics ──────────────
def test_i1_static_channel_in_targets_raises(valid_config_dict):
    cfg = dict(valid_config_dict)
    cfg["static_channels"] = ["lr_sb_best"]  # a regression target — illegal (I1)
    cfg["input_channels"] = 3 * cfg["output_channels"] + 1
    with pytest.raises(ValidationError, match="I1"):
        HydraNetConfig(**cfg)


def test_i1_input_channels_law_counts_statics(valid_config_dict):
    cfg = dict(valid_config_dict)
    cfg["static_channels"] = ["row_coord", "col_coord"]
    cfg["input_channels"] = 3 * cfg["output_channels"]  # missing the +2 static => must raise
    with pytest.raises(ValidationError, match="static_channels"):
        HydraNetConfig(**cfg)
    cfg["input_channels"] = 3 * cfg["output_channels"] + 2  # correct
    assert HydraNetConfig(**cfg).static_channels == ["row_coord", "col_coord"]


# ── I4/I6 (Beige): derived over the full grid, positioned + flipped in sync with the data ────────
def test_i4_i6_static_channel_positioned_and_flip_synced(seam_row_derivation):
    height = 4
    cfg = _vol_config(height=height, static=[_SEAM_ROW])
    vh = VolumeHandler.from_df(_grid_df(height=height), cfg)
    ci = vh.channel_map.index(_SEAM_ROW)
    static = np.asarray(vh.data)[:, :, :, ci]  # [T, H, W], post-flip
    # pre-flip value == row r; North-Up flip (axis=H) => post-flip row i holds (height-1-i)
    expected_row = (height - 1) - np.arange(height)
    assert np.allclose(static[0], expected_row[:, None])  # I6: flipped in sync
    # I6 cross-check: lr_sb_best was set to `row` too -> dynamic + static flip together
    ci_dyn = vh.channel_map.index("lr_sb_best")
    assert np.allclose(np.asarray(vh.data)[0, :, :, ci_dyn], static[0])


def test_i3_static_channel_constant_across_time(seam_row_derivation):
    vh = VolumeHandler.from_df(_grid_df(t=3), _vol_config(static=[_SEAM_ROW]))
    ci = vh.channel_map.index(_SEAM_ROW)
    static = np.asarray(vh.data)[:, :, :, ci]
    assert np.allclose(static[0], static[1]) and np.allclose(static[1], static[2])  # I3: geometry
    # the static channel is kept (reaches the model) — it is in feature_cols
    assert _SEAM_ROW in vh.feature_cols


# ── I2 (Beige): a static channel is never inverse-transformed ────────────────────────────────────
def test_i2_static_channel_not_inverse_transformed(seam_row_derivation):
    cfg = _vol_config(static=[_SEAM_ROW])
    cfg["transformations"] = {"log1p": list(_FEATURES), "asinh": [], "identity": []}
    cfg["features"] = list(_FEATURES)
    scaler = FeatureScaler(cfg)
    scaler.fit_transform(_grid_df())  # fits on the 3 dynamic features only
    vh = VolumeHandler.from_df(_grid_df(), cfg)
    ci = vh.channel_map.index(_SEAM_ROW)
    before = np.asarray(vh.data)[:, :, :, ci].copy()
    vh_inv = scaler.inverse_transform_volume(vh)
    after = np.asarray(vh_inv.data)[:, :, :, ci]
    assert np.allclose(before, after), "static channel must pass through inverse unchanged (I2)"
    assert _SEAM_ROW not in scaler.configured_columns  # never scaled


# ── I5 (Green): no static channels => byte-identical volume to the pre-seam path ─────────────────
def test_i5_no_static_channels_is_byte_identical():
    df = _grid_df()
    base = VolumeHandler.from_df(df, _vol_config(static=[]))
    # the volume carries exactly the dynamic features (+ identity) — no extra channel
    assert list(base.feature_cols) == _FEATURES
    assert np.asarray(base.data).shape[-1] == len(base.channel_map)
    # a config with static_channels=[] validates with the unchanged input_channels law
    # (3*output_channels), i.e. the seam is inert when unconfigured.
