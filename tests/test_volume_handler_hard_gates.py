"""
VolumeHandler hard-gate tests (ADR-005 taxonomy).

Green: identity striping, head dressing, geographic anchoring, spatial gradient,
       flip symmetry, extrapolate_time shape/continuity/cloning.
Beige: extrapolate_time single-step edge case.
Red: negative offset rejection, slice_time bounds checks.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from views_hydranet.utils.volume_handler import VolumeHandler

# SHARED PHYSICS CONFIG
PHYSICS_CFG = {
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "identity_cols": ["month_id", "priogrid_gid"],
    "features": ["lr_feature_a", "lr_feature_b"],
    "row_offset": 10,
    "col_offset": 20,
    "height": 4,
    "width": 4,
    # ADR 046 Compliance
    "transformations": {"identity": ["lr_feature_a", "lr_feature_b"]},
    "derivations": {
        "binary": [
            {"from": "lr_feature_a", "to": "by_feature_a", "threshold": 0},
            {"from": "lr_feature_b", "to": "by_feature_b", "threshold": 0},
        ]
    },
}


class TestGreen:
    """Green: VolumeHandler expected-condition gates."""

    def test_gate_11_identity_striping(self):
        """Assert to_pytorch strips identities by name."""
        df = pd.DataFrame(
            {
                "month_id": [1, 1],
                "priogrid_gid": [1, 2],
                "row": [10, 10],
                "col": [20, 21],
                "lr_feature_a": [1.0, 2.0],
                "lr_feature_b": [0.0, 0.0],
            }
        )
        handler = VolumeHandler.from_df(df, PHYSICS_CFG)
        tensor = handler.to_pytorch(torch.device("cpu"), include_identities=False)

        assert tensor.shape == (1, 1, 4, 4, 4), (
            f"Expected tensor shape (1, 1, 4, 4, 4) [B, T, C, H, W] after stripping identities, "
            f"got {tensor.shape}. Check include_identities=False path in to_pytorch()."
        )

    def test_gate_12_6_head_dressing(self):
        """Assert wrap_predictions correctly dresses 6 semantic heads."""
        posterior = torch.ones((1, 1, 4, 4, 4))  # T=1, C=4, H=4, W=4 (2 reg, 2 class)
        target_names = ["lr_a", "lr_b", "by_a", "by_b"]

        handler = VolumeHandler(
            data=np.zeros((1, 4, 4, 2)),
            axes=("T", "H", "W", "C"),
            channel_map=["month_id", "priogrid_gid"],
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=["row", "col"],
        )

        pred_handler = handler.wrap_predictions(posterior, target_names=target_names)

        assert "pred_lr_a" in pred_handler.channel_map
        assert "pred_by_a" in pred_handler.channel_map
        assert "pred_lr_b" in pred_handler.channel_map
        assert "pred_by_b" in pred_handler.channel_map

    def test_gate_15_geographic_anchoring(self):
        """Assert row_offset correctly anchors the grid."""
        df = pd.DataFrame(
            {
                "month_id": [1],
                "priogrid_gid": [1],
                "row": [10],
                "col": [20],
                "lr_feature_a": [1.0],
                "lr_feature_b": [1.0],
            }
        )
        handler = VolumeHandler.from_df(df, PHYSICS_CFG)
        data = handler.data

        feat_idx = handler.channel_map.index("lr_feature_a")
        actual = data[0, 3, 0, feat_idx]
        assert actual == 1.0, (
            f"\nGeographic anchor test failed.\n"
            f"Expected data[T=0, H=3, W=0, C={feat_idx}] == 1.0 (North-Up flip: "
            f"row=10 with offset=10 -> r_idx=0, flipped to H-1=3).\n"
            f"Got {actual}. Check row_offset and North-Up flip logic in from_df()."
        )

    def test_gate_16_spatial_gradient_preservation(self):
        """Gradient Oracle: each cell's value = local row index. After from_df(),
        the volume must contain a monotonically ordered gradient along every column."""
        H, W = 4, 4
        cfg = {
            "time_col": "month_id",
            "id_col": "priogrid_gid",
            "spatial_cols": ["row", "col"],
            "identity_cols": ["month_id", "priogrid_gid"],
            "features": ["value"],
            "row_offset": 10,
            "col_offset": 20,
            "height": H,
            "width": W,
            "transformations": {"identity": ["value"]},
            "derivations": {},
        }

        rows, cols, values = [], [], []
        for r_local in range(H):
            for c_local in range(W):
                rows.append(10 + r_local)
                cols.append(20 + c_local)
                values.append(float(r_local))

        df = pd.DataFrame(
            {
                "month_id": [1] * (H * W),
                "priogrid_gid": list(range(H * W)),
                "row": rows,
                "col": cols,
                "value": values,
            }
        )

        handler = VolumeHandler.from_df(df, cfg)
        data = handler.data
        feat_idx = handler.channel_map.index("value")

        for c in range(W):
            col_slice = data[0, :, c, feat_idx]
            diffs = np.diff(col_slice)
            assert np.all(diffs < 0), (
                f"\nSpatial gradient not preserved in column {c}."
                f"\nExpected strictly decreasing values (south->north in array)."
                f"\nActual slice: {col_slice}"
                f"\nDiffs (all should be -1): {diffs}"
            )

        assert data[0, 0, 0, feat_idx] == float(H - 1)
        assert data[0, H - 1, 0, feat_idx] == 0.0

    def test_gate_flip_symmetry_from_df_to_output(self):
        """C-08: North-Up flip in from_df() and _valid_cell_indices() must be symmetric."""
        H, W = 4, 4
        cfg = {
            "time_col": "month_id",
            "id_col": "priogrid_gid",
            "spatial_cols": ["row", "col"],
            "identity_cols": ["month_id", "priogrid_gid"],
            "features": ["value"],
            "row_offset": 10,
            "col_offset": 20,
            "height": H,
            "width": W,
            "transformations": {"identity": ["value"]},
            "derivations": {},
        }

        rows, cols, values, gids = [], [], [], []
        for r in range(H):
            for c in range(W):
                rows.append(10 + r)
                cols.append(20 + c)
                values.append(float(10 + r))
                gids.append(1 + r * W + c)

        df = pd.DataFrame(
            {
                "month_id": [1] * (H * W),
                "priogrid_gid": gids,
                "row": rows,
                "col": cols,
                "value": values,
            }
        )

        handler = VolumeHandler.from_df(df, cfg)
        val_idx = handler.channel_map.index("value")
        gid_idx = handler.channel_map.index("priogrid_gid")

        data = handler.data
        for r_array in range(H):
            for c_array in range(W):
                gid = data[0, r_array, c_array, gid_idx]
                if gid > 0:
                    planted_value = data[0, r_array, c_array, val_idx]
                    geo_row = cfg["row_offset"] + (H - 1 - r_array)
                    assert planted_value == geo_row, (
                        f"Flip symmetry broken at array[{r_array},{c_array}]: "
                        f"value={planted_value} but geo_row={geo_row}."
                    )

    def test_extrapolate_time_shape_preservation(self):
        """C-23: extrapolate_time(steps) must return [steps, H, W, C] with H, W, C unchanged."""
        T, H, W, C = 3, 2, 2, 3
        data = np.ones((T, H, W, C), dtype=np.float32)
        handler = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=["month_id", "priogrid_gid", "value"],
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=["row", "col"],
        )

        result = handler.extrapolate_time(5)
        assert result.shape == (5, H, W, C), (
            f"Expected shape (5, {H}, {W}, {C}), got {result.shape}"
        )

    def test_extrapolate_time_temporal_continuity(self):
        """C-23: Time channel must increment by 1 per step from the last observed value."""
        T, H, W = 3, 2, 2
        data = np.zeros((T, H, W, 3), dtype=np.float32)
        data[0, :, :, 0] = 100.0
        data[1, :, :, 0] = 101.0
        data[2, :, :, 0] = 102.0

        handler = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=["month_id", "priogrid_gid", "value"],
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=["row", "col"],
        )

        result = handler.extrapolate_time(3)
        time_idx = result.channel_map.index("month_id")

        for step in range(3):
            expected = 103.0 + step
            actual = result.data[step, 0, 0, time_idx]
            assert actual == expected, f"Step {step}: expected month_id={expected}, got {actual}"

    def test_extrapolate_time_non_time_channels_cloned(self):
        """C-23: Non-time channels must replicate the last frame exactly."""
        T, H, W = 3, 2, 2
        data = np.zeros((T, H, W, 3), dtype=np.float32)
        data[0, :, :, 0] = 100.0
        data[1, :, :, 0] = 101.0
        data[2, :, :, 0] = 102.0
        data[2, :, :, 1] = 42.0
        data[2, 0, 0, 2] = 7.0
        data[2, 1, 1, 2] = 13.0

        handler = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=["month_id", "priogrid_gid", "value"],
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=["row", "col"],
        )

        result = handler.extrapolate_time(4)

        for step in range(4):
            assert result.data[step, 0, 0, 1] == 42.0, f"Step {step}: priogrid_gid not cloned"
            assert result.data[step, 0, 0, 2] == 7.0, f"Step {step}: value[0,0] not cloned"
            assert result.data[step, 1, 1, 2] == 13.0, f"Step {step}: value[1,1] not cloned"


class TestBeige:
    """Beige: Edge cases in VolumeHandler."""

    def test_extrapolate_time_single_step(self):
        """C-23 edge case: steps=1 produces a single future frame."""
        data = np.zeros((2, 2, 2, 2), dtype=np.float32)
        data[1, :, :, 0] = 50.0

        handler = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=["month_id", "priogrid_gid"],
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=["row", "col"],
        )

        result = handler.extrapolate_time(1)
        assert result.shape[0] == 1
        assert result.data[0, 0, 0, 0] == 51.0


class TestRed:
    """Red: Failure modes in VolumeHandler."""

    def test_gate_17_negative_offset_rejection(self):
        """VolumeHandler must raise BEFORE writing when offsets produce negative indices."""
        # Case A: Loud failure (small grid, index out of bounds)
        bad_row_cfg = dict(PHYSICS_CFG, row_offset=50, col_offset=20)
        df_bad_row = pd.DataFrame(
            {
                "month_id": [1],
                "priogrid_gid": [1],
                "row": [20],
                "col": [20],
                "lr_feature_a": [1.0],
                "lr_feature_b": [1.0],
            }
        )
        with pytest.raises(ValueError, match="row"):
            VolumeHandler.from_df(df_bad_row, bad_row_cfg)

        # Case B: Silent failure (large grid, index wraps without error)
        silent_cfg = {
            "time_col": "month_id",
            "id_col": "priogrid_gid",
            "spatial_cols": ["row", "col"],
            "identity_cols": ["month_id", "priogrid_gid"],
            "features": ["lr_feature_a"],
            "row_offset": 87,
            "col_offset": 310,
            "height": 180,
            "width": 180,
            "transformations": {"identity": ["lr_feature_a"]},
            "derivations": {},
        }
        df_silent = pd.DataFrame(
            {
                "month_id": [1],
                "priogrid_gid": [1],
                "row": [0],
                "col": [310],
                "lr_feature_a": [1.0],
            }
        )
        with pytest.raises(ValueError, match="row"):
            VolumeHandler.from_df(df_silent, silent_cfg)

        # Case C: Col offset produces negative c_idx
        bad_col_cfg = dict(PHYSICS_CFG, row_offset=10, col_offset=50)
        df_bad_col = pd.DataFrame(
            {
                "month_id": [1],
                "priogrid_gid": [1],
                "row": [10],
                "col": [10],
                "lr_feature_a": [1.0],
                "lr_feature_b": [1.0],
            }
        )
        with pytest.raises(ValueError, match="col"):
            VolumeHandler.from_df(df_bad_col, bad_col_cfg)

        # Case D: Span violation (positive index out of bounds)
        span_cfg = dict(PHYSICS_CFG, height=4)
        df_span = pd.DataFrame(
            {
                "month_id": [1],
                "priogrid_gid": [1],
                "row": [15],
                "col": [20],
                "lr_feature_a": [1.0],
                "lr_feature_b": [1.0],
            }
        )
        with pytest.raises(ValueError, match="Span Violation"):
            VolumeHandler.from_df(df_span, span_cfg)

        # Boundary: exact match must NOT raise
        exact_cfg = dict(PHYSICS_CFG, row_offset=10, col_offset=20)
        df_exact = pd.DataFrame(
            {
                "month_id": [1],
                "priogrid_gid": [1],
                "row": [10],
                "col": [20],
                "lr_feature_a": [1.0],
                "lr_feature_b": [1.0],
            }
        )
        VolumeHandler.from_df(df_exact, exact_cfg)  # must not raise

    def test_gate_wrap_predictions_duration_mismatch_raises(self):
        """C-15/ADR-015: wrap_predictions must raise when the posterior's time
        dimension (T) differs from the carrier handler's duration."""
        # Carrier handler duration T=1 (data shape [T=1, H=4, W=4, C=2]).
        handler = VolumeHandler(
            data=np.zeros((1, 4, 4, 2)),
            axes=("T", "H", "W", "C"),
            channel_map=["month_id", "priogrid_gid"],
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=["row", "col"],
        )

        # Posterior with T=2 [B=1, T=2, C=4, H=4, W=4] → work_data.shape[0]=2 != 1.
        posterior = torch.ones((1, 2, 4, 4, 4))
        target_names = ["lr_a", "lr_b", "by_a", "by_b"]

        with pytest.raises(
            ValueError, match=r"Signal duration \(2\) does not match Handler duration \(1\)"
        ):
            handler.wrap_predictions(posterior, target_names=target_names)

    def test_gate_slice_time_beyond_bounds_raises(self):
        """C-24: Time slice beyond handler's temporal extent must raise ValueError."""
        handler = VolumeHandler(
            data=np.zeros((5, 4, 4, 2)),
            axes=("T", "H", "W", "C"),
            channel_map=["month_id", "priogrid_gid"],
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=["row", "col"],
        )

        with pytest.raises(ValueError, match="Invalid time slice"):
            handler.slice_time(3, 7)

        with pytest.raises(ValueError, match="Invalid time slice"):
            handler.slice_time(-1, 3)

        with pytest.raises(ValueError, match="Invalid time slice"):
            handler.slice_time(3, 3)

        result = handler.slice_time(0, 5)
        assert result.shape[0] == 5

    def test_gate_slice_time_origin_plus_duration_oob(self):
        """C-24: origin + duration exceeding time extent must raise."""
        handler = VolumeHandler(
            data=np.zeros((10, 4, 4, 2)),
            axes=("T", "H", "W", "C"),
            channel_map=["month_id", "priogrid_gid"],
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=["row", "col"],
        )

        origin = 8
        duration = 5
        start = origin + 1
        end = origin + 1 + duration

        with pytest.raises(ValueError, match="Invalid time slice"):
            handler.slice_time(start, end)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
