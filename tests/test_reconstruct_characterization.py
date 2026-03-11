"""
Item 5: Characterization tests for VolumeHandler._reconstruct_from_provider().

These tests pin the exact behavior of the DF reconstruction path using a
small (T=2, H=3, W=3) volume with known values. They serve as a safety net
for the eventual deprecation of the DF path — any behavioral change during
refactoring will be caught immediately.

Tests cover:
1. Output DataFrame shape and columns (point path)
2. Output DataFrame shape and columns (stochastic path)
3. Index structure (month_id, priogrid_gid multi-index)
4. Value correctness for known prediction channels
5. Identity column passthrough from provider
6. Mask filtering (priogrid_gid > 0 only)
"""

import numpy as np
import pandas as pd
import pytest

from views_hydranet.utils.volume_handler import VolumeHandler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T, H, W = 2, 3, 3
TIME_COL = "month_id"
ID_COL = "priogrid_gid"
SPATIAL_COLS = ("row", "col")
IDENTITY_COLS = ("c_id",)
FEATURE_COLS = ("lr_sb_best",)
TARGET_NAMES = ["lr_sb_best"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_history_handler():
    """
    Create a history VolumeHandler with known identity scaffold.
    Shape: [T=2, H=3, W=3, C=5] where C = month_id, priogrid_gid, c_id, lr_sb_best, by_sb_best
    """
    channel_map = [TIME_COL, ID_COL, "c_id", "lr_sb_best", "by_sb_best"]
    data = np.zeros((T, H, W, len(channel_map)), dtype=np.float64)

    # Fill month_id: 500, 501 across time
    for t in range(T):
        data[t, :, :, 0] = 500 + t

    # Fill priogrid_gid: unique per cell, 0 for one cell (ocean mask)
    gid = 1
    for r in range(H):
        for c in range(W):
            data[:, r, c, 1] = gid
            gid += 1
    # Set one cell to priogrid_gid=0 (ocean — should be masked out)
    data[:, 0, 0, 1] = 0

    # Fill c_id
    cid = 10
    for r in range(H):
        for c in range(W):
            data[:, r, c, 2] = cid
            cid += 1

    # Fill features with known values
    data[:, :, :, 3] = 1.5  # lr_sb_best
    data[:, :, :, 4] = 0.0  # by_sb_best

    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=channel_map,
        time_col=TIME_COL,
        id_col=ID_COL,
        spatial_cols=SPATIAL_COLS,
        identity_cols=IDENTITY_COLS,
        feature_cols=("lr_sb_best", "by_sb_best"),
    )


def _build_point_pred_handler(history):
    """
    Create a prediction VolumeHandler (point, 4D) via wrap_predictions.
    Posterior: [T=2, H=3, W=3, C=1] with known prediction value = 3.14
    """
    posterior = np.full((T, H, W, len(TARGET_NAMES)), 3.14, dtype=np.float64)
    return history.wrap_predictions(posterior, target_names=TARGET_NAMES)


def _build_stochastic_pred_handler(history, n_samples=2):
    """
    Create a prediction VolumeHandler (stochastic, 5D) via wrap_predictions.
    Posterior: [T=2, H=3, W=3, C=1, S=n_samples] with known values.
    """
    posterior = np.full(
        (T, H, W, len(TARGET_NAMES), n_samples), 2.71, dtype=np.float32
    )
    # Distinguish samples: sample 0 = 2.71, sample 1 = 3.14
    if n_samples >= 2:
        posterior[:, :, :, :, 1] = 3.14
    return history.wrap_predictions(posterior, target_names=TARGET_NAMES)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def history():
    return _build_history_handler()


@pytest.fixture
def point_pred(history):
    return _build_point_pred_handler(history)


@pytest.fixture
def stochastic_pred(history):
    return _build_stochastic_pred_handler(history, n_samples=2)


# ---------------------------------------------------------------------------
# Tests: Point Reconstruction (4D)
# ---------------------------------------------------------------------------

class TestReconstructPointPath:
    """Characterize _reconstruct_from_provider for point (4D) volumes."""

    def test_output_is_dataframe(self, point_pred, history):
        df = point_pred._reconstruct_from_provider(history)
        assert isinstance(df, pd.DataFrame)

    def test_index_is_multi_index(self, point_pred, history):
        df = point_pred._reconstruct_from_provider(history)
        assert isinstance(df.index, pd.MultiIndex)
        assert list(df.index.names) == [TIME_COL, ID_COL]

    def test_masked_cells_excluded(self, point_pred, history):
        """Cells with priogrid_gid=0 must be excluded (ocean mask)."""
        df = point_pred._reconstruct_from_provider(history)
        # We masked cell (0,0) to priogrid_gid=0. Grid is 3x3 = 9 cells.
        # 8 valid cells × 2 time steps = 16 rows.
        n_valid_cells = H * W - 1  # minus the masked cell
        expected_rows = T * n_valid_cells
        assert len(df) == expected_rows, (
            f"Expected {expected_rows} rows (T={T} × {n_valid_cells} valid cells), "
            f"got {len(df)}"
        )

    def test_prediction_column_present(self, point_pred, history):
        df = point_pred._reconstruct_from_provider(history)
        assert "pred_lr_sb_best" in df.columns, (
            f"Missing prediction column. Columns: {list(df.columns)}"
        )

    def test_prediction_values_correct(self, point_pred, history):
        """Point predictions should be scalar 3.14 for all valid cells."""
        df = point_pred._reconstruct_from_provider(history)
        pred_col = df["pred_lr_sb_best"]
        np.testing.assert_allclose(
            pred_col.values, 3.14, rtol=1e-5,
            err_msg="Point prediction values do not match expected 3.14"
        )

    def test_identity_columns_passthrough(self, point_pred, history):
        """Provider's identity columns (c_id) must appear in output."""
        df = point_pred._reconstruct_from_provider(history)
        assert "c_id" in df.columns, (
            f"Identity column 'c_id' missing from output. Columns: {list(df.columns)}"
        )

    def test_provider_features_in_output(self, point_pred, history):
        """Provider's feature columns should be in the scaffold."""
        df = point_pred._reconstruct_from_provider(history)
        assert "lr_sb_best" in df.columns, (
            "Provider feature 'lr_sb_best' missing from scaffold"
        )


# ---------------------------------------------------------------------------
# Tests: Stochastic Reconstruction (5D)
# ---------------------------------------------------------------------------

class TestReconstructStochasticPath:
    """Characterize _reconstruct_from_provider for stochastic (5D) volumes."""

    def test_output_is_dataframe(self, stochastic_pred, history):
        df = stochastic_pred._reconstruct_from_provider(history)
        assert isinstance(df, pd.DataFrame)

    def test_prediction_is_list_in_cell(self, stochastic_pred, history):
        """Stochastic predictions are list-in-cell: each cell has S values."""
        df = stochastic_pred._reconstruct_from_provider(history)
        pred_col = df["pred_lr_sb_best"]
        first_value = pred_col.iloc[0]
        assert isinstance(first_value, (list, np.ndarray)), (
            f"Stochastic predictions should be list-in-cell, got {type(first_value)}"
        )

    def test_stochastic_sample_count(self, stochastic_pred, history):
        """Each cell should have exactly n_samples values."""
        df = stochastic_pred._reconstruct_from_provider(history)
        pred_col = df["pred_lr_sb_best"]
        first_list = pred_col.iloc[0]
        assert len(first_list) == 2, f"Expected 2 samples, got {len(first_list)}"

    def test_stochastic_values_correct(self, stochastic_pred, history):
        """Sample 0 = 2.71, sample 1 = 3.14."""
        df = stochastic_pred._reconstruct_from_provider(history)
        pred_col = df["pred_lr_sb_best"]
        first_list = pred_col.iloc[0]
        np.testing.assert_allclose(first_list[0], 2.71, rtol=1e-4)
        np.testing.assert_allclose(first_list[1], 3.14, rtol=1e-4)

    def test_masked_cells_excluded_stochastic(self, stochastic_pred, history):
        """Ocean cells must be excluded in stochastic path too."""
        df = stochastic_pred._reconstruct_from_provider(history)
        n_valid_cells = H * W - 1
        expected_rows = T * n_valid_cells
        assert len(df) == expected_rows
