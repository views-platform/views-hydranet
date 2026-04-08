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


def test_gate_11_identity_striping():
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

    # 4 features (2 base + 2 derived), T=1, H=4, W=4
    assert tensor.shape == (1, 1, 4, 4, 4), (
        f"Expected tensor shape (1, 1, 4, 4, 4) [B, T, C, H, W] after stripping identities, "
        f"got {tensor.shape}. Check include_identities=False path in to_pytorch()."
    )


def test_gate_12_6_head_dressing():
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

    # Check internal signal names
    assert "pred_lr_a" in pred_handler.channel_map, (
        f"Expected 'pred_lr_a' in channel_map, got: {pred_handler.channel_map}"
    )
    assert "pred_by_a" in pred_handler.channel_map, (
        f"Expected 'pred_by_a' in channel_map, got: {pred_handler.channel_map}"
    )
    assert "pred_lr_b" in pred_handler.channel_map, (
        f"Expected 'pred_lr_b' in channel_map, got: {pred_handler.channel_map}"
    )
    assert "pred_by_b" in pred_handler.channel_map, (
        f"Expected 'pred_by_b' in channel_map, got: {pred_handler.channel_map}"
    )


def test_gate_15_geographic_anchoring():
    """Assert row_offset correctly anchors the grid."""
    df = pd.DataFrame(
        {
            "month_id": [1],
            "priogrid_gid": [1],
            "row": [10],
            "col": [20],  # Matches offsets exactly
            "lr_feature_a": [1.0],
            "lr_feature_b": [1.0],
        }
    )
    handler = VolumeHandler.from_df(df, PHYSICS_CFG)
    data = handler.data  # [T, H, W, C]

    # Coordinates (10, 20) with offsets (10, 20) should map to grid index (0, 0)
    # But wait, it is flipped (North-Up)
    # Row 0 in input (bottom) becomes Row 3 in North-Up (top)
    # So we check index [0, 3, 0, :]
    feat_idx = handler.channel_map.index("lr_feature_a")
    actual = data[0, 3, 0, feat_idx]
    assert actual == 1.0, (
        f"\nGeographic anchor test failed.\n"
        f"Expected data[T=0, H=3, W=0, C={feat_idx}] == 1.0 (North-Up flip: "
        f"row=10 with offset=10 → r_idx=0, flipped to H-1=3).\n"
        f"Got {actual}. Check row_offset and North-Up flip logic in from_df()."
    )


def test_gate_16_spatial_gradient_preservation():
    """
    GREEN+RED GATE: Verifies that VolumeHandler.from_df() correctly maps
    spatial coordinates — values at specific (row, col) locations in the
    DataFrame must land at the corresponding locations in the volume.

    Method: the 'Gradient Oracle'. Each cell's value is set to its local
    row index (a perfect north-south gradient). After from_df(), the volume
    must contain a monotonically ordered gradient along every column.
    Any scrambling (wrong offset, wrong indexing, wrong flip) breaks monotonicity.

    This is the automated equivalent of the Visual Diagnostics Gradient Test
    (2026-02-19 investigation plan). Once this test exists, spatial scrambling
    cannot be introduced without a visible test failure.

    North-Up note: VolumeHandler flips axis=0 so array row 0 holds the
    SOUTHERNMOST data (highest r_local). The gradient runs high→low as
    array row index increases (0 = south = H-1, H-1 = north = 0).
    """
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

    # Build full H×W grid for one time step.
    # value = r_local → perfect north-south gradient (0 at north, H-1 at south).
    rows, cols, values = [], [], []
    for r_local in range(H):
        for c_local in range(W):
            rows.append(10 + r_local)  # global = offset + local
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
    data = handler.data  # [T=1, H=4, W=4, C]
    feat_idx = handler.channel_map.index("value")

    # After North-Up flip: array row 0 = south (r_local H-1), row H-1 = north (r_local 0).
    # Along every column, values must be strictly DECREASING (south→north).
    for c in range(W):
        col_slice = data[0, :, c, feat_idx]
        diffs = np.diff(col_slice)
        assert np.all(diffs < 0), (
            f"\nSpatial gradient not preserved in column {c}."
            f"\nExpected strictly decreasing values (south→north in array)."
            f"\nActual slice: {col_slice}"
            f"\nDiffs (all should be -1): {diffs}"
            f"\nThis indicates spatial scrambling — check row_offset config."
        )

    # Exact boundary values
    assert data[0, 0, 0, feat_idx] == float(H - 1), (
        f"Southernmost array row should hold value {H - 1}, got {data[0, 0, 0, feat_idx]}"
    )
    assert data[0, H - 1, 0, feat_idx] == 0.0, (
        f"Northernmost array row should hold value 0.0, got {data[0, H - 1, 0, feat_idx]}"
    )


def test_gate_17_negative_offset_rejection():
    """
    RED GATE: VolumeHandler must raise BEFORE writing when row/col offsets
    produce negative indices. Two cases must both be caught:

    Case A — "Loud": negative index exceeds numpy array bounds → currently
              raises IndexError deep in numpy (wrong layer, wrong type).
              After the guard: raises ValueError early with a clear message.

    Case B — "Silent" (the real danger): negative index is within numpy's
              valid wrap-around range (e.g., -87 in a height=180 array).
              numpy writes to the WRONG location and raises nothing.
              After the guard: raises ValueError early.

    This is Phase 1, Step 1 of the Test Remediation Plan (2026-02-19).
    """
    # --- Case A: Loud failure (small grid, index well out of bounds) ---
    # row=20, row_offset=50 → r_idx = 20-50 = -30. height=4, so -30 < -4 → IndexError
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

    # --- Case B: Silent failure (large grid, index wraps without error) ---
    # Mimics the production scenario: height=180, row_offset=87, but data
    # starts at row=0 (local coords). r_idx = 0-87 = -87. In a 180-tall array,
    # -87 is a VALID numpy index (wraps to 93). No IndexError. Data is silently
    # written 87 rows from the bottom instead of the top. Map is inverted.
    silent_cfg = {
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"],
        "identity_cols": ["month_id", "priogrid_gid"],
        "features": ["lr_feature_a"],
        "row_offset": 87,  # <-- typical Africa offset
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
            "row": [0],  # local coord — r_idx = 0-87 = -87, wraps to 93 silently
            "col": [310],
            "lr_feature_a": [1.0],
        }
    )
    with pytest.raises(ValueError, match="row"):
        VolumeHandler.from_df(df_silent, silent_cfg)

    # --- Case C: Col offset produces negative c_idx ---
    bad_col_cfg = dict(PHYSICS_CFG, row_offset=10, col_offset=50)
    df_bad_col = pd.DataFrame(
        {
            "month_id": [1],
            "priogrid_gid": [1],
            "row": [10],
            "col": [10],  # col=10, offset=50 → c_idx=-40
            "lr_feature_a": [1.0],
            "lr_feature_b": [1.0],
        }
    )
    with pytest.raises(ValueError, match="col"):
        VolumeHandler.from_df(df_bad_col, bad_col_cfg)

    # --- Case D: Span violation (Positive index out of bounds) ---
    # row=15, row_offset=10 → r_idx=5. height=4, so 5 >= 4 → ValueError
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

    # --- Boundary: exact match must NOT raise ---
    # row_offset == df.row.min() → r_idx=0, valid.
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


# ─── C-24: Temporal discontinuity — slice_time bounds check ──────────────────


def test_gate_slice_time_beyond_bounds_raises():
    """
    C-24: Requesting a time slice beyond the handler's temporal extent must
    raise ValueError with ADR-008 compliant log-before-raise.

    This is the temporal discontinuity failure mode declared in the
    InferenceOrchestrator CIC. The orchestrator delegates bounds checking
    to VolumeHandler.slice_time(); this test verifies that delegation works.
    """
    handler = VolumeHandler(
        data=np.zeros((5, 4, 4, 2)),  # T=5, H=4, W=4, C=2
        axes=("T", "H", "W", "C"),
        channel_map=["month_id", "priogrid_gid"],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
    )

    # Beyond end
    with pytest.raises(ValueError, match="Invalid time slice"):
        handler.slice_time(3, 7)  # end=7 > T=5

    # Negative start
    with pytest.raises(ValueError, match="Invalid time slice"):
        handler.slice_time(-1, 3)

    # Start >= end (empty slice)
    with pytest.raises(ValueError, match="Invalid time slice"):
        handler.slice_time(3, 3)

    # Valid boundary: must NOT raise
    result = handler.slice_time(0, 5)  # full extent
    assert result.shape[0] == 5


def test_gate_slice_time_origin_plus_duration_oob():
    """
    C-24: Simulates the orchestrator's temporal alignment calculation.
    When origin + duration exceeds the handler's time extent, slice_time
    must raise — not silently return truncated data.
    """
    handler = VolumeHandler(
        data=np.zeros((10, 4, 4, 2)),  # T=10
        axes=("T", "H", "W", "C"),
        channel_map=["month_id", "priogrid_gid"],
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=["row", "col"],
    )

    origin = 8
    duration = 5
    start = origin + 1  # = 9
    end = origin + 1 + duration  # = 14, but T=10

    with pytest.raises(ValueError, match="Invalid time slice"):
        handler.slice_time(start, end)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
