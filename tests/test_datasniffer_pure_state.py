"""
Unit tests for DataSniffer pure state methods.

Verifies CIC DataSniffer §3-C (Pure State Parity) and §3-D (Pure State Schema):
- sniff_pure_state_parity(df_input, df_output)
- sniff_pure_state_schema(df, config)
"""

import pandas as pd
import pytest

from views_hydranet.utils.data_sniffer import DataSniffer

# ─── Helpers ────────────────────────────────────────────────────────────────

CONFIG = {
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "identity_cols": ["c_id", "row", "col"],
    "height": 4,
    "width": 4,
    "regression_targets": ["lr_ged_sb"],
    "classification_targets": ["by_sb_best"],
}


def _make_sniffer():
    return DataSniffer(CONFIG)


def _make_input_df():
    """Flat DataFrame with 4 rows — the 'input' before predictions."""
    return pd.DataFrame(
        {
            "month_id": [500, 500, 501, 501],
            "priogrid_gid": [1, 2, 1, 2],
            "c_id": [10, 20, 10, 20],
            "row": [0.0, 1.0, 0.0, 1.0],
            "col": [0.0, 0.0, 0.0, 0.0],
            "lr_ged_sb": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _make_output_df(df_input):
    """Output = input + prediction columns (as if model ran)."""
    df_out = df_input.copy()
    df_out["pred_lr_ged_sb"] = [0.5, 0.6, 0.7, 0.8]
    df_out["pred_by_sb_best"] = [0.1, 0.2, 0.3, 0.4]
    df_out["by_sb_best"] = [0, 1, 1, 1]  # derived binary column
    return df_out


def _make_schema_valid_df():
    """DataFrame with correct MultiIndex and mandatory columns for schema check."""
    df = pd.DataFrame(
        {
            "month_id": [500, 500, 501, 501],
            "priogrid_gid": [1, 2, 1, 2],
            "c_id": [10, 20, 10, 20],
            "row": [0.0, 1.0, 0.0, 1.0],
            "col": [0.0, 0.0, 0.0, 0.0],
            "pred_lr_ged_sb": [0.5, 0.6, 0.7, 0.8],
            "pred_by_sb_best": [0.1, 0.2, 0.3, 0.4],
        }
    )
    return df.set_index(["month_id", "priogrid_gid"])


# ─── Green Team: §3 Guarantees ─────────────────────────────────────────────


class TestGreen:
    def test_green_parity_passes_on_clean_round_trip(self):
        """Output with predictions stripped matches input — no raise."""
        sniffer = _make_sniffer()
        df_input = _make_input_df()
        df_output = _make_output_df(df_input)
        # Should not raise
        sniffer.sniff_pure_state_parity(df_input, df_output)

    def test_green_schema_passes_on_valid_output(self):
        """Valid MultiIndex + mandatory identities + pred_* columns → no raise."""
        sniffer = _make_sniffer()
        df = _make_schema_valid_df()
        sniffer.sniff_pure_state_schema(df, CONFIG)

    def test_green_schema_validates_prediction_columns_present(self):
        """All pred_{target} columns must exist for configured targets."""
        sniffer = _make_sniffer()
        df = _make_schema_valid_df()
        # This should pass since both pred_lr_ged_sb and pred_by_sb_best are present
        sniffer.sniff_pure_state_schema(df, CONFIG)


# ─── Beige Team: Realistic Scenarios ───────────────────────────────────────


class TestBeige:
    def test_beige_parity_tolerates_float_precision(self):
        """Float values within atol=1e-5 pass parity check."""
        sniffer = _make_sniffer()
        df_input = _make_input_df()
        df_output = _make_output_df(df_input)
        # Add tiny float noise within tolerance
        df_output["lr_ged_sb"] = df_input["lr_ged_sb"] + 1e-7
        sniffer.sniff_pure_state_parity(df_input, df_output)

    def test_beige_parity_order_agnostic(self):
        """Shuffled row order in output still passes."""
        sniffer = _make_sniffer()
        df_input = _make_input_df()
        df_output = _make_output_df(df_input)
        # Reverse row order
        df_output = df_output.iloc[::-1].reset_index(drop=True)
        sniffer.sniff_pure_state_parity(df_input, df_output)

    def test_beige_schema_handles_both_reg_and_cls_targets(self):
        """Both pred_lr_* and pred_by_* columns validated together."""
        sniffer = _make_sniffer()
        df = _make_schema_valid_df()
        config = {
            **CONFIG,
            "regression_targets": ["lr_ged_sb"],
            "classification_targets": ["by_sb_best"],
        }
        sniffer.sniff_pure_state_schema(df, config)


# ─── Red Team: §6 Failure Modes ───────────────────────────────────────────


class TestRed:
    def test_red_parity_detects_drifted_float_column(self):
        """Output float column values differ from input → ValueError."""
        sniffer = _make_sniffer()
        df_input = _make_input_df()
        df_output = _make_output_df(df_input)
        # Introduce significant drift
        df_output["lr_ged_sb"] = df_input["lr_ged_sb"] + 1.0
        with pytest.raises(ValueError, match="Parity Violation"):
            sniffer.sniff_pure_state_parity(df_input, df_output)

    def test_red_parity_detects_drifted_integer_column(self):
        """Integer identity column mutated → ValueError."""
        sniffer = _make_sniffer()
        df_input = _make_input_df()
        df_output = _make_output_df(df_input)
        # Mutate an integer identity column
        df_output["c_id"] = [99, 99, 99, 99]
        with pytest.raises(ValueError, match="Parity Violation"):
            sniffer.sniff_pure_state_parity(df_input, df_output)

    def test_red_schema_rejects_wrong_multiindex(self):
        """DataFrame indexed by wrong columns → ValueError."""
        sniffer = _make_sniffer()
        df = _make_schema_valid_df().reset_index()
        df = df.set_index(["c_id", "row"])  # Wrong index
        with pytest.raises(ValueError, match="Index Mismatch"):
            sniffer.sniff_pure_state_schema(df, CONFIG)

    def test_red_schema_rejects_missing_mandatory_identities(self):
        """c_id or spatial_cols missing → ValueError."""
        sniffer = _make_sniffer()
        df = _make_schema_valid_df()
        df = df.drop(columns=["c_id"])
        with pytest.raises(ValueError, match="Missing Mandatory"):
            sniffer.sniff_pure_state_schema(df, CONFIG)
