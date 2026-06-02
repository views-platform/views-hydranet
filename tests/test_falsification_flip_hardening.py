"""
Falsification stubs for flip symmetry hardening gaps.

These tests cover gaps identified by the falsification audit of the claim
"North-Up flip symmetry hardening is now robust and fully test covered."

Each stub corresponds to a soft falsification finding (F-01, F-02, F-07).
"""

import numpy as np
import pandas as pd
import pytest


def _make_small_vh():
    """Minimal NORTH_UP VolumeHandler for propagation tests."""
    from views_hydranet.utils.volume_handler import VolumeHandler

    rows = []
    for t in range(2):
        for r in range(4):
            for c in range(4):
                rows.append(
                    {
                        "month_id": 500 + t,
                        "priogrid_gid": (10 + r) * 1000 + (20 + c),
                        "row": 10 + r,
                        "col": 20 + c,
                        "value": float(r * 100 + c),
                    }
                )
    df = pd.DataFrame(rows)
    cfg = {
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"],
        "identity_cols": ["month_id", "priogrid_gid"],
        "features": ["value"],
        "row_offset": 10,
        "col_offset": 20,
        "height": 4,
        "width": 4,
        "transformations": {"identity": ["value"]},
        "derivations": {},
    }
    return VolumeHandler.from_df(df, cfg), cfg


# ---------------------------------------------------------------------------
# F-01: Convention propagation through pipeline methods
# ---------------------------------------------------------------------------


class TestF01ConventionPropagation:
    """Each pipeline method that creates a derived VolumeHandler must preserve NORTH_UP."""

    def test_slice_time_preserves_convention(self):
        from views_hydranet.utils.volume_handler import SpatialConvention

        vh, _ = _make_small_vh()
        sliced = vh.slice_time(0, 1)
        assert sliced.spatial_convention == SpatialConvention.NORTH_UP

    def test_extrapolate_time_preserves_convention(self):
        from views_hydranet.utils.volume_handler import SpatialConvention

        vh, _ = _make_small_vh()
        ext = vh.extrapolate_time(2)
        assert ext.spatial_convention == SpatialConvention.NORTH_UP

    def test_wrap_predictions_preserves_convention(self):
        from views_hydranet.utils.volume_handler import SpatialConvention

        vh, _ = _make_small_vh()
        posterior = np.random.rand(2, 4, 4, 1)
        wrapped = vh.wrap_predictions(posterior, ["value"])
        assert wrapped.spatial_convention == SpatialConvention.NORTH_UP

    def test_collapse_to_point_preserves_convention(self):
        from views_hydranet.utils.volume_handler import SpatialConvention

        vh, _ = _make_small_vh()
        posterior = np.random.rand(2, 4, 4, 1, 3)
        wrapped = vh.wrap_predictions(posterior, ["value"])
        collapsed = wrapped.collapse_to_point("arithmetic_mean")
        assert collapsed.spatial_convention == SpatialConvention.NORTH_UP

    def test_inverse_transform_preserves_convention(self):
        from views_hydranet.utils.feature_scaler import FeatureScaler
        from views_hydranet.utils.volume_handler import SpatialConvention

        vh, cfg = _make_small_vh()
        scaler = FeatureScaler(cfg)
        scaler._is_fitted = True
        inverted = scaler.inverse_transform_volume(vh)
        assert inverted.spatial_convention == SpatialConvention.NORTH_UP

    def test_permute_preserves_convention(self):
        from views_hydranet.utils.volume_handler import SpatialConvention

        vh, _ = _make_small_vh()
        permuted = vh._permute((0, 2, 1, 3))  # swap H and W
        assert permuted.spatial_convention == SpatialConvention.NORTH_UP


# ---------------------------------------------------------------------------
# F-02: Asymmetric convention mismatch
# ---------------------------------------------------------------------------


class TestF02AsymmetricMismatch:
    """Both directions of convention mismatch must raise."""

    def test_provider_geographic_signal_north_up_raises(self):
        from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler
        from views_hydranet.utils.volume_handler import SpatialConvention, VolumeHandler

        vh, _ = _make_small_vh()
        assert vh.spatial_convention == SpatialConvention.NORTH_UP

        provider = VolumeHandler(
            data=vh.data.copy(),
            axes=vh.axes,
            channel_map=vh.channel_map,
            time_col=vh.time_col,
            id_col=vh.id_col,
            spatial_cols=vh.spatial_cols,
            identity_cols=vh.identity_cols,
            feature_cols=vh.feature_cols,
            spatial_offset=vh.spatial_offset,
        )
        assert provider.spatial_convention == SpatialConvention.GEOGRAPHIC

        asm = PredictionFrameAssembler()
        with pytest.raises(ValueError, match="North-Up"):
            asm._valid_cell_indices(vh, provider)


# ---------------------------------------------------------------------------
# F-07: Guards use raise ValueError (upgraded from assert)
# ---------------------------------------------------------------------------


class TestF07GuardsUseRaise:
    """Convention guards use raise ValueError, not assert.

    Upgraded from assert to ensure guards survive python -O.
    """

    def test_guards_use_raise_not_assert(self):
        import inspect

        from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler

        src = inspect.getsource(PredictionFrameAssembler._valid_cell_indices)
        assert "raise ValueError" in src, (
            "Convention guards must use raise ValueError, not assert. "
            "assert is stripped by python -O."
        )
        assert "signal.spatial_convention" in src
        assert "provider.spatial_convention" in src
