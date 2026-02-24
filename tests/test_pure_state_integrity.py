"""
Team Audit Suite for ADR 032: Authoritative Spatiotemporal Output Schema.
Verifies Accuracy (Green), Robustness (Beige), and Invincibility (Red).
"""

import numpy as np
import pandas as pd
import pytest

from views_hydranet.utils.data_sniffer import DataSniffer
from views_hydranet.utils.volume_handler import VolumeHandler


@pytest.fixture
def mock_config():
    return {
        "height": 10,
        "width": 10,
        "time_col": "month_id",
        "id_col": "priogrid_gid",
        "spatial_cols": ["row", "col"],
        "row_offset": 0,
        "col_offset": 0,
        "identity_cols": ["month_id", "priogrid_gid", "row", "col", "c_id"],
        "features": ["lr_sb_best"],
        # ADR 046 Compliance
        "regression_targets": ["lr_sb_best"],
        "classification_targets": ["by_sb_best"],
        "transformations": {"identity": ["lr_sb_best"]},
        "derivations": {"binary": [{"from": "lr_sb_best", "to": "by_sb_best", "threshold": 0}]},
    }


@pytest.fixture
def input_df():
    """Bit-perfect input data scaffold."""
    data = {
        "month_id": sorted(list(range(500, 505)) * 2),
        "priogrid_gid": [100, 101] * 5,
        "row": [1, 1] * 5,
        "col": [1, 2] * 5,
        "c_id": [42] * 10,
        "lr_sb_best": np.random.rand(10) * 10,
    }
    return pd.DataFrame(data)


# --- GREEN TEAM: ACCURACY ---


def test_green_roundtrip_parity(input_df, mock_config):
    """Prove bit-perfect round-trip: DF -> Volume -> DF."""
    # 1. Ingest
    handler = VolumeHandler.from_df(input_df, mock_config)

    # 2. Reconstruct (Historical)
    output_df = handler.to_historical_df()

    # 3. Verify via Sniffer
    sniffer = DataSniffer(mock_config)
    # This should pass without error
    sniffer.sniff_pure_state_parity(input_df, output_df)

    # Explicit check for bookkeeping identity
    # Use .loc for precise index-based lookup (month_id, priogrid_gid)
    first_month = input_df["month_id"].min()
    assert output_df.loc[(first_month, 100), "c_id"] == 42


def test_green_schema_conformance(input_df, mock_config):
    """Verify prefix-based naming and suffix retirement."""
    handler = VolumeHandler.from_df(input_df, mock_config)

    # Create mock predictions: [T, H, W, C]
    # For a 10x10xGrid grid, 1 task (sb) -> 2 heads (lr, by)
    n_months = input_df["month_id"].nunique()
    mock_preds = np.random.rand(n_months, 10, 10, 2)

    # Wrap: Must use lr_ prefixed name
    pred_handler = handler.wrap_predictions(mock_preds, ["lr_sb_best", "by_sb_best"])

    # Reconstruct
    output_df = pred_handler.to_historical_df()

    # Verify Schema
    sniffer = DataSniffer(mock_config)
    sniffer.sniff_pure_state_schema(output_df, mock_config)

    assert "pred_lr_sb_best" in output_df.columns
    assert "pred_by_sb_best" in output_df.columns
    assert "pred_lr_sb_best_raw" not in output_df.columns  # retired
    assert "pred_by_sb_best_prob" not in output_df.columns  # retired


def test_green_prefix_smart_naming(input_df, mock_config):
    """
    PROVE that providing 'lr_' prefixed targets results in literal
    pred_lr_ and pred_by_ mapping.
    """
    # 1. Config with already-prefixed feature
    prefixed_config = mock_config.copy()
    prefixed_config["features"] = ["lr_sb_best"]

    handler = VolumeHandler.from_df(input_df, prefixed_config)
    n_months = input_df["month_id"].nunique()
    mock_preds = np.random.rand(n_months, 10, 10, 2)

    # 2. Wrap: Passing names should yield pred_ mapping
    pred_handler = handler.wrap_predictions(mock_preds, ["lr_sb_best", "by_sb_best"])

    # 3. Assert Naming Engine Literalism
    assert "pred_lr_sb_best" in pred_handler.channel_map
    assert "pred_by_sb_best" in pred_handler.channel_map

    # 4. Check Final Reconstruction
    output_df = pred_handler.to_historical_df()
    assert "pred_lr_sb_best" in output_df.columns
    assert "pred_by_sb_best" in output_df.columns


# --- BEIGE TEAM: ROBUSTNESS ---


def test_beige_missing_id_fails(input_df, mock_config):
    """Verify that providing a DF with missing role columns triggers an immediate crash."""
    bad_df = input_df.drop(columns=["month_id"])
    with pytest.raises(ValueError, match="Missing columns"):
        VolumeHandler.from_df(bad_df, mock_config)


def test_beige_misnamed_index_fails(input_df, mock_config):
    """Verify that misnamed index in Sniffer triggers error."""
    handler = VolumeHandler.from_df(input_df, mock_config)
    output_df = handler.to_historical_df()

    # Corrupt config for Sniffer
    bad_config = mock_config.copy()
    bad_config["time_col"] = "wrong_month"

    sniffer = DataSniffer(bad_config)
    with pytest.raises(ValueError, match="Index Mismatch"):
        sniffer.sniff_pure_state_schema(output_df, bad_config)


# --- RED TEAM: INVINCIBILITY ---


def test_red_shuffle_immunity(input_df, mock_config):
    """Prove that row-shuffling has zero impact on identity restoration."""
    # 1. Shuffle Input
    shuffled_df = input_df.sample(frac=1).reset_index(drop=True)
    assert not shuffled_df.equals(input_df)  # verify shuffle

    # 2. Convert to Volume
    handler = VolumeHandler.from_df(shuffled_df, mock_config)

    # 3. Reconstruct
    output_df = handler.to_historical_df()

    # 4. Verify parity against ORIGINAL (pre-shuffle) input
    sniffer = DataSniffer(mock_config)
    sniffer.sniff_pure_state_parity(input_df, output_df)


def test_red_ocean_breach_prevention(input_df, mock_config):
    """Verify that non-land pixels are strictly excluded from output DF."""
    handler = VolumeHandler.from_df(input_df, mock_config)

    # Injected data at [0,0] which is NOT in input_df (indices were 1,1 and 1,2)
    # Even if predictions exist there, to_historical_df must mask it via the scaffold
    n_months = input_df["month_id"].nunique()
    mock_preds = np.ones((n_months, 10, 10, 2))
    pred_handler = handler.wrap_predictions(mock_preds, ["lr_sb_best", "by_sb_best"])

    output_df = pred_handler.to_historical_df()

    # The output row count must match input row count (Land only)
    assert len(output_df) == len(input_df)

    # Verify that no priogrid_gid=0 (default np.zeros value) exists in output
    assert (output_df.index.get_level_values("priogrid_gid") > 0).all()
