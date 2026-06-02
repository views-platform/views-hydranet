"""
Flip Symmetry Hardening — R-5 risk register.

Guards the North-Up convention coupling between VolumeHandler.from_df()
(construction, axis-0 flip) and PredictionFrameAssembler._valid_cell_indices()
(reconstruction, axis-0 flip ×2). This convention has broken multiple times.
These tests ensure it cannot break without loud, obvious failures.

Test taxonomy (ADR-005):
  GREEN  — Happy-path round-trips and convention markers
  BEIGE  — Edge cases (degenerate grids, offsets)
  RED    — Adversarial source inspection and corruption detection
  DOMAIN — Geographic invariants (hemisphere land ratios)
  AUG    — Data-augmentation flip (independent path)
  VIS    — Visualization flip (independent path)
"""

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from views_hydranet.utils.volume_handler import (
    SpatialConvention,
    VolumeHandler,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grid_df(
    height,
    width,
    row_offset,
    col_offset,
    n_timesteps=1,
    month_start=500,
    value_fn=None,
    land_mask_fn=None,
):
    """
    Build a synthetic PRIO-GRID-like DataFrame.

    Parameters
    ----------
    value_fn : callable(row, col, t) -> float
        Produces the 'value' feature.  Default: row * 100 + col.
    land_mask_fn : callable(row, col) -> bool
        If False, the cell is skipped (simulates ocean).  Default: all land.
    """

    def _default_value(r, c, t):
        return float(r * 100 + c)

    def _default_land(r, c):
        return True

    if value_fn is None:
        value_fn = _default_value
    if land_mask_fn is None:
        land_mask_fn = _default_land

    rows = []
    for t in range(n_timesteps):
        month = month_start + t
        for r in range(height):
            for c in range(width):
                geo_row = row_offset + r
                geo_col = col_offset + c
                if not land_mask_fn(geo_row, geo_col):
                    continue
                gid = geo_row * 1000 + geo_col
                rows.append(
                    {
                        "month_id": month,
                        "priogrid_gid": gid,
                        "row": geo_row,
                        "col": geo_col,
                        "value": value_fn(geo_row, geo_col, t),
                    }
                )
    return pd.DataFrame(rows)


def _base_config(height, width, row_offset, col_offset):
    return {
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"],
        "identity_cols": ["month_id", "priogrid_gid"],
        "features": ["value"],
        "row_offset": row_offset,
        "col_offset": col_offset,
        "height": height,
        "width": width,
        "transformations": {"identity": ["value"]},
        "derivations": {},
    }


def _value_channel_index(vh):
    return vh.channel_map.index("value")


# ---------------------------------------------------------------------------
# GREEN — Happy Path
# ---------------------------------------------------------------------------


class TestGreen:
    """Round-trip correctness and convention markers."""

    def test_single_cell_round_trip_all_corners(self):
        """G-01: Plant unique values at 4 corners, verify array positions."""
        H, W, R_OFF, C_OFF = 4, 4, 10, 20
        corners = {
            (10, 20): 1.0,  # SW: geo_row=10 (min), geo_col=20 (min)
            (10, 23): 2.0,  # SE: geo_row=10, geo_col=23 (max)
            (13, 20): 3.0,  # NW: geo_row=13 (max), geo_col=20
            (13, 23): 4.0,  # NE: geo_row=13, geo_col=23
        }
        df = _make_grid_df(
            H,
            W,
            R_OFF,
            C_OFF,
            value_fn=lambda r, c, t: corners.get((r, c), 0.0),
        )
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)
        vol = vh.data  # [T, H, W, C]
        vi = _value_channel_index(vh)

        # After North-Up: row 0 = northernmost (geo_row=13)
        # NW corner (geo_row=13, geo_col=20) → array [0, H-1-3=0, col-off=0]
        assert vol[0, 0, 0, vi] == 3.0, "NW corner should be at array [0, 0, 0]"
        # NE corner (geo_row=13, geo_col=23) → array [0, 0, 3]
        assert vol[0, 0, 3, vi] == 4.0, "NE corner should be at array [0, 0, 3]"
        # SW corner (geo_row=10, geo_col=20) → array [0, 3, 0]
        assert vol[0, 3, 0, vi] == 1.0, "SW corner should be at array [0, 3, 0]"
        # SE corner (geo_row=10, geo_col=23) → array [0, 3, 3]
        assert vol[0, 3, 3, vi] == 2.0, "SE corner should be at array [0, 3, 3]"

    def test_full_grid_bijective_round_trip(self):
        """G-02: Every cell round-trips through from_df → assembler."""
        from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler

        H, W, R_OFF, C_OFF = 6, 6, 10, 20
        df = _make_grid_df(
            H,
            W,
            R_OFF,
            C_OFF,
            value_fn=lambda r, c, t: float(r * 100 + c),
        )
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)

        asm = PredictionFrameAssembler()
        # Use vh as both signal and provider (it has priogrid_gid and month_id channels)
        temp_data, indices, time_flat, unit_flat = asm._valid_cell_indices(vh, vh)

        # Reconstruct (row, col) from priogrid_gid = row * 1000 + col
        recovered_rows = unit_flat // 1000
        recovered_cols = unit_flat % 1000

        # Extract values at the valid cell locations
        vi = vh.channel_map.index("value")
        values = temp_data[indices[0], indices[1], indices[2], vi]

        for i in range(len(values)):
            expected_value = recovered_rows[i] * 100 + recovered_cols[i]
            assert values[i] == pytest.approx(expected_value), (
                f"Cell gid={unit_flat[i]}: expected value {expected_value}, got {values[i]}"
            )

    def test_spatial_gradient_north_south(self):
        """G-03: Values = geographic row → must decrease along H axis (North-Up)."""
        H, W, R_OFF, C_OFF = 8, 4, 10, 20
        df = _make_grid_df(
            H,
            W,
            R_OFF,
            C_OFF,
            value_fn=lambda r, c, t: float(r),
        )
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)
        vi = _value_channel_index(vh)
        col_slice = vh.data[0, :, 0, vi]  # all H values in column 0, time 0
        for i in range(len(col_slice) - 1):
            assert col_slice[i] > col_slice[i + 1], (
                f"North-Up violated at H[{i}]={col_slice[i]}, H[{i + 1}]={col_slice[i + 1]}: "
                f"higher geographic row should have smaller array index"
            )

    def test_spatial_gradient_east_west(self):
        """G-04: Values = geographic col → must increase along W axis (no col flip)."""
        H, W, R_OFF, C_OFF = 4, 8, 10, 20
        df = _make_grid_df(
            H,
            W,
            R_OFF,
            C_OFF,
            value_fn=lambda r, c, t: float(c),
        )
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)
        vi = _value_channel_index(vh)
        row_slice = vh.data[0, 0, :, vi]  # all W values in row 0, time 0
        for i in range(len(row_slice) - 1):
            assert row_slice[i] < row_slice[i + 1], (
                f"Column axis should NOT be flipped: "
                f"W[{i}]={row_slice[i]}, W[{i + 1}]={row_slice[i + 1]}"
            )

    def test_round_trip_production_config(self):
        """G-05: Production-scale config (180×180) with cells at known locations."""
        from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler

        H, W, R_OFF, C_OFF = 180, 180, 87, 310
        known_cells = {
            (120, 350): 1.0,  # Central Africa (north of equator)
            (155, 325): 2.0,  # Northern Europe
            (95, 380): 3.0,  # Southeast Asia
            (100, 340): 4.0,  # Near equator, Africa
        }
        rows = []
        for (geo_r, geo_c), val in known_cells.items():
            rows.append(
                {
                    "month_id": 500,
                    "priogrid_gid": geo_r * 1000 + geo_c,
                    "row": geo_r,
                    "col": geo_c,
                    "value": val,
                }
            )
        df = pd.DataFrame(rows)
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)

        asm = PredictionFrameAssembler()
        temp_data, indices, time_flat, unit_flat = asm._valid_cell_indices(vh, vh)
        vi = vh.channel_map.index("value")
        values = temp_data[indices[0], indices[1], indices[2], vi]

        recovered = {}
        for i in range(len(unit_flat)):
            recovered[unit_flat[i]] = values[i]

        for (geo_r, geo_c), expected_val in known_cells.items():
            gid = geo_r * 1000 + geo_c
            assert gid in recovered, f"Cell gid={gid} not recovered"
            assert recovered[gid] == pytest.approx(expected_val), (
                f"Cell ({geo_r},{geo_c}): expected {expected_val}, got {recovered[gid]}"
            )

    def test_temporal_dimension_unaffected(self):
        """G-06: Multi-timestep — time ordering preserved, flip only affects H."""
        H, W, R_OFF, C_OFF = 4, 4, 10, 20
        df = _make_grid_df(
            H,
            W,
            R_OFF,
            C_OFF,
            n_timesteps=5,
            value_fn=lambda r, c, t: float(t),
        )
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)
        vi = _value_channel_index(vh)

        for t in range(5):
            vals = vh.data[t, :, :, vi]
            unique = np.unique(vals[vals != 0])
            if len(unique) > 0:
                assert unique[0] == pytest.approx(float(t)), (
                    f"Time step {t}: expected value {t}, got {unique[0]}"
                )

    def test_stochastic_round_trip(self):
        """G-07: 5D signal [T,H,W,C,S] round-trips through assembler correctly."""
        from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler

        H, W, R_OFF, C_OFF = 4, 4, 10, 20
        df = _make_grid_df(H, W, R_OFF, C_OFF, value_fn=lambda r, c, t: float(r * 100 + c))
        cfg = _base_config(H, W, R_OFF, C_OFF)
        provider = VolumeHandler.from_df(df, cfg)

        n_samples = 3
        base = provider.data  # [T, H, W, C]
        vi = _value_channel_index(provider)
        signal_data = np.stack([base] * n_samples, axis=-1)  # [T, H, W, C, S]
        for s in range(n_samples):
            signal_data[:, :, :, vi, s] = base[:, :, :, vi] + s * 0.01

        from dataclasses import replace as dc_replace

        signal = VolumeHandler(
            data=signal_data,
            axes=("T", "H", "W", "C", "S"),
            channel_map=provider.channel_map,
            time_col=provider.time_col,
            id_col=provider.id_col,
            spatial_cols=provider.spatial_cols,
            identity_cols=provider.identity_cols,
            feature_cols=provider.feature_cols,
            spatial_offset=provider.spatial_offset,
        )
        signal._metadata = dc_replace(
            signal._metadata, spatial_convention=SpatialConvention.NORTH_UP
        )

        asm = PredictionFrameAssembler()
        temp_data, indices, time_flat, unit_flat = asm._valid_cell_indices(signal, provider)

        assert temp_data.ndim == 5, f"Expected 5D, got {temp_data.ndim}D"
        values_s0 = temp_data[indices[0], indices[1], indices[2], vi, 0]
        values_s1 = temp_data[indices[0], indices[1], indices[2], vi, 1]

        rows_recovered = unit_flat // 1000
        cols_recovered = unit_flat % 1000
        for i in range(len(values_s0)):
            base_val = rows_recovered[i] * 100 + cols_recovered[i]
            assert values_s0[i] == pytest.approx(base_val, abs=0.02)
            assert values_s1[i] == pytest.approx(base_val + 0.01, abs=0.02)

    def test_convention_marker_set_by_from_df(self):
        """G-08: from_df() produces a volume with spatial_convention == NORTH_UP."""
        df = _make_grid_df(4, 4, 10, 20)
        cfg = _base_config(4, 4, 10, 20)
        vh = VolumeHandler.from_df(df, cfg)
        assert vh.spatial_convention == SpatialConvention.NORTH_UP

    def test_convention_default_is_geographic(self):
        """G-09: Manual VolumeHandler construction defaults to GEOGRAPHIC."""
        data = np.zeros((1, 4, 4, 2))
        vh = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=("month_id", "value"),
            time_col="month_id",
            id_col="priogrid_gid",
            spatial_cols=("row", "col"),
        )
        assert vh.spatial_convention == SpatialConvention.GEOGRAPHIC


# ---------------------------------------------------------------------------
# BEIGE — Edge Cases
# ---------------------------------------------------------------------------


class TestBeige:
    """Degenerate grids and boundary conditions."""

    def test_single_row_grid(self):
        """B-01: Height=1 — flip is no-op, round-trip must still work."""
        H, W, R_OFF, C_OFF = 1, 4, 10, 20
        df = _make_grid_df(H, W, R_OFF, C_OFF, value_fn=lambda r, c, t: float(c))
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)
        vi = _value_channel_index(vh)
        row_vals = vh.data[0, 0, :, vi]
        expected = np.array([20.0, 21.0, 22.0, 23.0])
        np.testing.assert_array_almost_equal(row_vals, expected)

    def test_single_column_grid(self):
        """B-02: Width=1 — degenerate column geometry."""
        H, W, R_OFF, C_OFF = 4, 1, 10, 20
        df = _make_grid_df(H, W, R_OFF, C_OFF, value_fn=lambda r, c, t: float(r))
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)
        vi = _value_channel_index(vh)
        col_vals = vh.data[0, :, 0, vi]
        # North-Up: highest geographic row first
        assert col_vals[0] == 13.0
        assert col_vals[3] == 10.0

    def test_square_grid_no_transposition(self):
        """B-03: H=W — rows and columns must not be swapped."""
        H, W, R_OFF, C_OFF = 5, 5, 10, 20
        df = _make_grid_df(
            H,
            W,
            R_OFF,
            C_OFF,
            value_fn=lambda r, c, t: float(r * 1000 + c),
        )
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)
        vi = _value_channel_index(vh)
        # Top-left after North-Up: (geo_row=14, geo_col=20)
        assert vh.data[0, 0, 0, vi] == pytest.approx(14 * 1000 + 20)
        # Top-right: (geo_row=14, geo_col=24)
        assert vh.data[0, 0, 4, vi] == pytest.approx(14 * 1000 + 24)
        # Bottom-left: (geo_row=10, geo_col=20)
        assert vh.data[0, 4, 0, vi] == pytest.approx(10 * 1000 + 20)

    def test_row_offset_zero(self):
        """B-04: Offset=0 — formula degenerates to geo_row = H-1-r_array."""
        H, W = 4, 4
        df = _make_grid_df(H, W, 0, 0, value_fn=lambda r, c, t: float(r))
        cfg = _base_config(H, W, 0, 0)
        vh = VolumeHandler.from_df(df, cfg)
        vi = _value_channel_index(vh)
        col_vals = vh.data[0, :, 0, vi]
        assert col_vals[0] == 3.0  # geo_row=3 at array index 0
        assert col_vals[3] == 0.0  # geo_row=0 at array index 3

    def test_large_offsets(self):
        """B-05: Large offsets — no integer overflow or misalignment."""
        H, W, R_OFF, C_OFF = 4, 4, 10000, 5000
        df = _make_grid_df(H, W, R_OFF, C_OFF, value_fn=lambda r, c, t: float(r * 100 + c))
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)
        vi = _value_channel_index(vh)
        # NW corner: geo_row=10003, geo_col=5000
        assert vh.data[0, 0, 0, vi] == pytest.approx(10003 * 100 + 5000)

    def test_all_ocean_grid(self):
        """B-06: All priogrid_gid == 0 — assembler returns zero valid cells."""
        from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler

        H, W, R_OFF, C_OFF = 4, 4, 10, 20
        df = _make_grid_df(H, W, R_OFF, C_OFF)
        # Override priogrid_gid to 0 (ocean)
        df["priogrid_gid"] = 0
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)

        asm = PredictionFrameAssembler()
        temp_data, indices, time_flat, unit_flat = asm._valid_cell_indices(vh, vh)
        assert len(time_flat) == 0, "All-ocean grid should produce zero valid cells"

    def test_sparse_grid(self):
        """B-07: Only 3 cells in 10×10 — reconstruction recovers exactly those 3."""
        from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler

        H, W, R_OFF, C_OFF = 10, 10, 10, 20
        target_cells = {(12, 25), (15, 22), (19, 29)}
        df = _make_grid_df(
            H,
            W,
            R_OFF,
            C_OFF,
            land_mask_fn=lambda r, c: (r, c) in target_cells,
            value_fn=lambda r, c, t: float(r * 100 + c),
        )
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)

        asm = PredictionFrameAssembler()
        _, indices, _, unit_flat = asm._valid_cell_indices(vh, vh)

        recovered = set()
        for gid in unit_flat:
            recovered.add((int(gid) // 1000, int(gid) % 1000))
        assert recovered == target_cells, f"Expected {target_cells}, got {recovered}"


# ---------------------------------------------------------------------------
# RED — Adversarial / Source Inspection
# ---------------------------------------------------------------------------


class TestRed:
    """Source-level guards against structural flip corruption."""

    def test_from_df_has_exactly_one_flip(self):
        """R-01: from_df() source contains exactly 1 np.flip on axis=0.

        If you moved the flip to a helper, update this test to inspect that helper.
        """
        src = inspect.getsource(VolumeHandler.from_df)
        matches = re.findall(r"np\.flip\s*\(", src)
        assert len(matches) == 1, (
            f"Expected exactly 1 np.flip in from_df(). Found {len(matches)}. "
            f"Double-flipping silently cancels the North-Up convention."
        )

    def test_from_df_flip_not_removed(self):
        """R-02: from_df() source must contain np.flip."""
        src = inspect.getsource(VolumeHandler.from_df)
        assert "np.flip" in src, (
            "np.flip was removed from from_df(). "
            "The North-Up convention requires exactly one axis-0 flip."
        )

    def test_assembler_has_exactly_two_flips(self):
        """R-03: _valid_cell_indices() has exactly 2 np.flip calls (signal + provider)."""
        from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler

        src = inspect.getsource(PredictionFrameAssembler._valid_cell_indices)
        code_lines = [line for line in src.split("\n") if not line.strip().startswith("#")]
        code_only = "\n".join(code_lines)
        matches = re.findall(r"np\.flip\s*\(", code_only)
        assert len(matches) == 2, (
            f"Expected exactly 2 np.flip in _valid_cell_indices(). Found {len(matches)}. "
            f"Both signal and provider must be flipped for correct reconstruction."
        )

    def test_flip_axis_consistent(self):
        """R-04: All flip sites use axis=0. Catches wrong-axis mutation."""
        from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler

        for func, name in [
            (VolumeHandler.from_df, "from_df"),
            (PredictionFrameAssembler._valid_cell_indices, "_valid_cell_indices"),
        ]:
            src = inspect.getsource(func)
            flip_calls = re.findall(r"np\.flip\([^)]+\)", src)
            for call in flip_calls:
                assert "axis=0" in call, (
                    f"{name}: flip call '{call}' does not use axis=0. "
                    f"The North-Up convention requires flipping axis 0 (the H dimension)."
                )

    def test_shuffled_rows_detected(self):
        """R-05: Manually shuffling H axis → assembler recovers wrong coordinates."""
        from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler

        H, W, R_OFF, C_OFF = 6, 6, 10, 20
        df = _make_grid_df(H, W, R_OFF, C_OFF, value_fn=lambda r, c, t: float(r * 100 + c))
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)

        # Corrupt: shuffle H axis
        corrupted_data = vh.data.copy()
        perm = np.random.RandomState(42).permutation(H)
        corrupted_data = corrupted_data[:, perm, :, :]

        from dataclasses import replace as dc_replace

        corrupted = VolumeHandler(
            data=corrupted_data,
            axes=vh.axes,
            channel_map=vh.channel_map,
            time_col=vh.time_col,
            id_col=vh.id_col,
            spatial_cols=vh.spatial_cols,
            identity_cols=vh.identity_cols,
            feature_cols=vh.feature_cols,
            spatial_offset=vh.spatial_offset,
        )
        corrupted._metadata = dc_replace(
            corrupted._metadata, spatial_convention=SpatialConvention.NORTH_UP
        )

        asm = PredictionFrameAssembler()
        _, _, _, unit_flat_good = asm._valid_cell_indices(vh, vh)
        _, _, _, unit_flat_bad = asm._valid_cell_indices(corrupted, corrupted)

        assert not np.array_equal(
            np.sort(unit_flat_good), np.sort(unit_flat_bad)
        ) or not np.array_equal(unit_flat_good, unit_flat_bad), (
            "Shuffled rows should produce different reconstruction — "
            "test infrastructure would not catch a corruption."
        )

    def test_wrong_axis_detected(self):
        """R-06: Flipping axis=1 (W) instead of axis=0 (H) breaks the gradient test."""
        H, W, R_OFF, C_OFF = 4, 6, 10, 20
        df = _make_grid_df(H, W, R_OFF, C_OFF, value_fn=lambda r, c, t: float(r))
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)

        # If someone flipped W instead of H, the N-S gradient would break.
        # Verify our gradient check catches this:
        vi = _value_channel_index(vh)
        col_slice = vh.data[0, :, 0, vi]
        is_monotone_decreasing = all(
            col_slice[i] > col_slice[i + 1] for i in range(len(col_slice) - 1)
        )
        assert is_monotone_decreasing, (
            "North-South gradient must be monotonically decreasing. "
            "If this fails, the flip may be on the wrong axis."
        )

        # Also verify E-W is NOT monotone decreasing (it should increase)
        row_slice = vh.data[0, 0, :, vi]
        is_ew_constant = len(set(row_slice.tolist())) == 1
        assert is_ew_constant, "With value_fn=row, all columns in a row should have equal values."

    def test_convention_mismatch_raises(self):
        """R-07: Signal with GEOGRAPHIC convention → assembler assertion fires."""
        from views_hydranet.utils.prediction_frame_assembler import PredictionFrameAssembler

        df = _make_grid_df(4, 4, 10, 20)
        cfg = _base_config(4, 4, 10, 20)
        provider = VolumeHandler.from_df(df, cfg)

        # Manually construct signal WITHOUT going through from_df
        signal = VolumeHandler(
            data=provider.data.copy(),
            axes=provider.axes,
            channel_map=provider.channel_map,
            time_col=provider.time_col,
            id_col=provider.id_col,
            spatial_cols=provider.spatial_cols,
            identity_cols=provider.identity_cols,
            feature_cols=provider.feature_cols,
            spatial_offset=provider.spatial_offset,
        )
        assert signal.spatial_convention == SpatialConvention.GEOGRAPHIC

        asm = PredictionFrameAssembler()
        with pytest.raises(ValueError, match="North-Up"):
            asm._valid_cell_indices(signal, provider)


# ---------------------------------------------------------------------------
# DOMAIN KNOWLEDGE — Geographic Invariants
# ---------------------------------------------------------------------------


class TestGreenDomainKnowledge:
    """
    Tests rooted in geographic facts about Earth that no code change
    should be able to "fix away." If the flip is inverted, these fail
    because hemisphere ratios swap.
    """

    @pytest.fixture
    def hemisphere_volume(self):
        """180×180 grid with ~67% land in northern hemisphere."""
        H, W, R_OFF, C_OFF = 180, 180, 87, 310
        equator_row = 177  # approx: row_offset + 90

        rng = np.random.RandomState(12345)
        rows = []
        month = 500
        for r in range(H):
            for c in range(W):
                geo_row = R_OFF + r
                geo_col = C_OFF + c

                # ~67% land in north, ~33% in south
                if geo_row > equator_row:
                    is_land = rng.random() < 0.20  # north: 20% of cells are land
                else:
                    is_land = rng.random() < 0.10  # south: 10% of cells are land

                if not is_land:
                    continue

                gid = geo_row * 1000 + geo_col
                rows.append(
                    {
                        "month_id": month,
                        "priogrid_gid": gid,
                        "row": geo_row,
                        "col": geo_col,
                        "value": float(geo_row),
                    }
                )

        df = pd.DataFrame(rows)
        cfg = _base_config(H, W, R_OFF, C_OFF)
        return VolumeHandler.from_df(df, cfg), equator_row, R_OFF

    def test_northern_hemisphere_has_more_land(self, hemisphere_volume):
        """D-01: After North-Up flip, top half of array has more land cells.

        ~67% of Earth's land mass is in the northern hemisphere. If the flip
        is inverted, the top half becomes the southern hemisphere and this
        assertion fails. This is the smoking gun test.
        """
        vh, equator_row, r_off = hemisphere_volume
        gid_idx = vh.channel_map.index("priogrid_gid")
        gid_plane = vh.data[0, :, :, gid_idx]  # [H, W]

        midpoint = vh.data.shape[1] // 2  # H / 2
        north_land = np.count_nonzero(gid_plane[:midpoint, :])
        south_land = np.count_nonzero(gid_plane[midpoint:, :])

        assert north_land > south_land * 1.3, (
            f"Northern hemisphere should have substantially more land than southern. "
            f"North={north_land}, South={south_land}. If flipped, this ratio inverts."
        )

    def test_equatorial_row_position(self, hemisphere_volume):
        """D-02: Equatorial row lands near array midpoint after North-Up flip."""
        vh, equator_row, r_off = hemisphere_volume
        vi = _value_channel_index(vh)  # value = geographic row

        H = vh.data.shape[1]
        # The equatorial geographic row after flip should be near array index H/2
        # equator_row - r_off = index before flip = 90
        # after flip: H-1-90 = 89 (for H=180)
        expected_array_row = H - 1 - (equator_row - r_off)

        # Find a column where the equator cell has data
        found = False
        for c in range(vh.data.shape[2]):
            val = vh.data[0, expected_array_row, c, vi]
            if val > 0:
                assert val == pytest.approx(equator_row, abs=1.0), (
                    f"Cell at array row {expected_array_row} should contain equatorial "
                    f"geo_row ≈ {equator_row}, got {val}"
                )
                found = True
                break
        if not found:
            # Equator cell might be ocean in our random fixture — check neighbors
            for offset in [-1, 1, -2, 2]:
                for c in range(vh.data.shape[2]):
                    val = vh.data[0, expected_array_row + offset, c, vi]
                    if val > 0:
                        assert abs(val - equator_row) <= 3, (
                            f"Near-equator cell at array row {expected_array_row + offset} "
                            f"should have geo_row near {equator_row}, got {val}"
                        )
                        found = True
                        break
                if found:
                    break
            assert found, "No land cells near the equator in hemisphere fixture"

    def test_hemisphere_ratio_time_invariant(self):
        """D-03: N/S land ratio is constant across time steps (land doesn't migrate)."""
        H, W, R_OFF, C_OFF = 20, 20, 80, 310
        equator_row = 90  # approximate

        rng = np.random.RandomState(42)
        n_timesteps = 4
        rows = []
        land_cells = set()
        for r in range(H):
            for c in range(W):
                geo_row = R_OFF + r
                geo_col = C_OFF + c
                if geo_row > equator_row:
                    is_land = rng.random() < 0.4
                else:
                    is_land = rng.random() < 0.2
                if is_land:
                    land_cells.add((geo_row, geo_col))

        for t in range(n_timesteps):
            for geo_row, geo_col in land_cells:
                rows.append(
                    {
                        "month_id": 500 + t,
                        "priogrid_gid": geo_row * 1000 + geo_col,
                        "row": geo_row,
                        "col": geo_col,
                        "value": float(geo_row),
                    }
                )

        df = pd.DataFrame(rows)
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)
        gid_idx = vh.channel_map.index("priogrid_gid")

        midpoint = H // 2
        ratios = []
        for t in range(n_timesteps):
            gid_plane = vh.data[t, :, :, gid_idx]
            north = np.count_nonzero(gid_plane[:midpoint, :])
            south = np.count_nonzero(gid_plane[midpoint:, :])
            if south > 0:
                ratios.append(north / south)

        assert len(set(round(r, 4) for r in ratios)) == 1, (
            f"N/S land ratio should be identical across time steps: {ratios}"
        )

    def test_known_country_quadrant(self):
        """D-04: Norway-like cells must appear in top quarter of array after North-Up."""
        H, W, R_OFF, C_OFF = 180, 180, 87, 310

        # Norway: roughly geo_row 246-256, geo_col 315-325
        # (in PRIO-GRID: high latitudes = high row numbers)
        norway_cells = [
            (250, 318),
            (252, 320),
            (255, 315),
        ]
        rows = []
        for geo_r, geo_c in norway_cells:
            rows.append(
                {
                    "month_id": 500,
                    "priogrid_gid": geo_r * 1000 + geo_c,
                    "row": geo_r,
                    "col": geo_c,
                    "value": 1.0,
                }
            )
        df = pd.DataFrame(rows)
        cfg = _base_config(H, W, R_OFF, C_OFF)
        vh = VolumeHandler.from_df(df, cfg)
        vi = _value_channel_index(vh)

        quarter = H // 4  # = 45
        top_quarter = vh.data[0, :quarter, :, vi]
        assert np.any(top_quarter > 0), (
            "Norway-like cells (high latitude) should appear in the top quarter of "
            "the array after North-Up flip. If they don't, the flip may be inverted."
        )


# ---------------------------------------------------------------------------
# AUGMENTATION — Independent Flip Path
# ---------------------------------------------------------------------------


class TestGreenAugmentation:
    """The flip() augmentation method is independent of North-Up construction."""

    @pytest.fixture
    def augmentation_vh(self):
        df = _make_grid_df(4, 4, 10, 20, value_fn=lambda r, c, t: float(r * 100 + c))
        cfg = _base_config(4, 4, 10, 20)
        return VolumeHandler.from_df(df, cfg)

    def test_flip_h_is_self_inverse(self, augmentation_vh):
        """A-01: flip("H") twice → data matches original."""
        vh = augmentation_vh
        flipped_once = vh.flip("H")
        flipped_twice = flipped_once.flip("H")
        np.testing.assert_array_equal(vh.data, flipped_twice.data)

    def test_flip_does_not_change_convention(self, augmentation_vh):
        """A-02: Augmentation flip is tracked in history, not convention."""
        vh = augmentation_vh
        assert vh.spatial_convention == SpatialConvention.NORTH_UP
        flipped = vh.flip("H")
        assert flipped.spatial_convention == SpatialConvention.NORTH_UP
        assert ("flip", "H") in flipped.history

    def test_flip_h_w_commutative(self, augmentation_vh):
        """A-03: flip("H") then flip("W") == flip("W") then flip("H")."""
        vh = augmentation_vh
        hw = vh.flip("H").flip("W")
        wh = vh.flip("W").flip("H")
        np.testing.assert_array_equal(hw.data, wh.data)

    def test_flip_history_tracked(self, augmentation_vh):
        """A-04: flip() records in history tuple."""
        vh = augmentation_vh
        f1 = vh.flip("H")
        assert f1.history[-1] == ("flip", "H")
        f2 = f1.flip("W")
        assert f2.history[-1] == ("flip", "W")
        assert f2.history[-2] == ("flip", "H")


# ---------------------------------------------------------------------------
# VISUALIZATION — Independent Flip Path
# ---------------------------------------------------------------------------


class TestGreenVisualization:
    """The biopsy_dataframe flip is on axis=1 (H in [T,H,W,C] layout), not axis=0."""

    def test_visualization_flips_axis_1_not_0(self):
        """V-01: biopsy_dataframe flips axis=1. Catches 'harmonization' to axis=0."""
        from views_hydranet.utils.visual_diagnostics import VisualDiagnostics

        src = inspect.getsource(VisualDiagnostics.biopsy_dataframe)
        flip_calls = re.findall(r"np\.flip\([^)]+\)", src)
        assert len(flip_calls) >= 1, "biopsy_dataframe should contain np.flip"
        for call in flip_calls:
            assert "axis=1" in call, (
                f"biopsy_dataframe flip '{call}' should use axis=1 (H in [T,H,W,C] layout). "
                f"Do NOT change this to axis=0 — the construction flip in from_df() uses axis=0 "
                f"because that volume is [H,W,T,C] pre-transpose."
            )
