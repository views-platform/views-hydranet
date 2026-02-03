import pytest
import pandas as pd
import numpy as np
import torch
from views_hydranet.utils.volume_handler import VolumeHandler

class TestVolumeHandlerLedger:
    """
    Rigorously verifies compliance with ADR 007: VolumeHandler (The Custodian).
    Tests focus on Zero-Magic, Ledger-First, and Bit-Perfect integrity.
    """

    @pytest.fixture
    def standard_config(self):
        return {
            "time_col": "month_id",
            "id_col": "priogrid_gid",
            "spatial_cols": ["row", "col"],
            "identity_cols": ["month_id", "priogrid_gid", "row", "col"], # Added row/col here
            "features": ["feat_1", "feat_2"],
            "row_offset": 10,
            "col_offset": 20,
            "height": 5,
            "width": 5,
            "steps": [1],
            "n_posterior_samples": 10
        }

    @pytest.fixture
    def custom_alias_config(self):
        return {
            "time_col": "temporal",
            "id_col": "unit",
            "spatial_cols": ["y", "x"],
            "identity_cols": ["temporal", "unit", "y", "x"],
            "features": ["signal"],
            "row_offset": 0,
            "col_offset": 0,
            "height": 2,
            "width": 2,
            "steps": [1]
        }

    def test_core_initialization_and_validation(self, standard_config):
        """Verify Ledger role storage and channel dimension enforcement."""
        data = np.zeros((1, 5, 5, 4)) # T=1, H=5, W=5, C=4
        map_4 = ["month_id", "priogrid_gid", "feat_1", "feat_2"]
        
        # Positive Case
        vh = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=map_4,
            time_col=standard_config["time_col"],
            id_col=standard_config["id_col"],
            spatial_cols=standard_config["spatial_cols"],
            identity_cols=standard_config["identity_cols"],
            feature_cols=standard_config["features"]
        )
        assert vh.time_col == "month_id"
        assert vh.id_col == "priogrid_gid"
        assert vh.data.shape[-1] == 4

        # Negative Case: Channel Mismatch
        with pytest.raises(ValueError, match="Channel mismatch"):
            VolumeHandler(
                data=np.zeros((1, 5, 5, 10)), # Wrong size
                axes=("T", "H", "W", "C"),
                channel_map=map_4,
                time_col="month_id",
                id_col="priogrid_gid",
                spatial_cols=["row", "col"]
            )

    def test_from_df_handshake_rigor(self, custom_alias_config):
        """Verify that from_df rejects DFs with missing roles and uses aliases correctly."""
        # Missing 'temporal' column
        df_bad = pd.DataFrame([{"unit": 1, "y": 0, "x": 0, "signal": 0.5}])
        with pytest.raises(ValueError, match="Handshake Failed"):
            VolumeHandler.from_df(df_bad, custom_alias_config)

        # Positive Case with Aliases
        df_good = pd.DataFrame([{
            "temporal": 100, "unit": 55, "y": 1, "x": 1, "signal": 0.75
        }])
        vh = VolumeHandler.from_df(df_good, custom_alias_config, height=2, width=2)
        assert vh.data.shape == (1, 2, 2, 5) # T=1, H=2, W=2, C=5 (4 IDs + 1 Signal)

    def test_to_historical_df_parity_and_masking(self, standard_config):
        """Verify bit-perfect reconstruction and Ledger-based masking."""
        data_in = [
            {"month_id": 400, "priogrid_gid": 123, "row": 12, "col": 22, "feat_1": 1.5, "feat_2": 2.5},
            {"month_id": 401, "priogrid_gid": 456, "row": 14, "col": 24, "feat_1": 0.0, "feat_2": -1.0}
        ]
        df_in = pd.DataFrame(data_in)
        vh = VolumeHandler.from_df(df_in, standard_config, height=5, width=5)
        
        df_out = vh.to_historical_df()
        
        # 1. Bit-perfect Identity Equality
        # Sort both by month_id to ensure order
        df_in_sorted = df_in.sort_values("month_id").reset_index(drop=True)
        df_out_sorted = df_out.sort_values("month_id").reset_index(drop=True)

        for col in ["month_id", "priogrid_gid", "row", "col"]:
            assert (df_out_sorted[col].values == df_in_sorted[col].values).all()
            assert df_out_sorted[col].dtype == np.int64 or df_out_sorted[col].dtype == np.int32

        # 2. Feature Precision
        assert np.allclose(df_out_sorted["feat_1"].values, df_in_sorted["feat_1"].values, atol=1e-7)

        # 3. Masking Rigor: Corrupt ocean cell should not leak
        c_idx = vh.channel_map.index("feat_1")
        # Ensure we are setting a value in a cell that was originally 0.0 (Ocean)
        vh.data[0, 0, 0, c_idx] = 999.0
        
        df_masked = vh.to_historical_df()
        assert 999.0 not in df_masked["feat_1"].values

    def test_to_pytorch_and_recovery(self, standard_config):
        """Verify PyTorch permutation, identity stripping, and 5D wrapping."""
        df = pd.DataFrame([{"month_id": 100, "priogrid_gid": 1, "row": 10, "col": 20, "feat_1": 1.0, "feat_2": 2.0}])
        vh = VolumeHandler.from_df(df, standard_config, height=5, width=5)
        
        # 1. Strip Identities
        tensor = vh.to_pytorch(torch.device("cpu"), include_identities=False)
        # Expected: [B=1, T=1, C=2, H=5, W=5] (Identity count is 4 in standard_config fixture)
        assert tensor.shape == (1, 1, 2, 5, 5)

        # 2. Recovery: 5D Batch Squeeze
        pred_vh = vh.wrap_predictions(tensor, ["pred_1", "pred_2"])
        assert pred_vh.data.shape == (1, 5, 5, 2) # [T, H, W, C]
        assert pred_vh.id_col == "priogrid_gid" # Inherited

        # 3. Recovery: 5D Samples Preservation (ADR 007 Section 3.4)
        samples_5d = np.ones((1, 5, 5, 2, 10)) * 5.0
        pred_vh_samples = vh.wrap_predictions(samples_5d, ["p1", "p2"])
        # Expect 5D shape [T, H, W, C, S]
        assert pred_vh_samples.data.shape == (1, 5, 5, 2, 10)
        assert np.all(pred_vh_samples.data == 5.0)
        assert "S" in pred_vh_samples.axes

    def test_to_evaluation_df_contract(self, standard_config):
        """Verify strict temporal contract enforcement for evaluation."""
        df_hist = pd.DataFrame([{"month_id": 100, "priogrid_gid": 1, "row": 10, "col": 20, "feat_1": 1.0, "feat_2": 2.0}])
        history = VolumeHandler.from_df(df_hist, standard_config, height=5, width=5)

        # 2-month prediction
        df_pred = pd.DataFrame([
            {"month_id": 100, "priogrid_gid": 1, "row": 10, "col": 20, "p1": 9.9},
            {"month_id": 101, "priogrid_gid": 1, "row": 10, "col": 20, "p1": 8.8}
        ])
        pred_config = standard_config.copy()
        pred_config["features"] = ["p1"]
        pred_vh = VolumeHandler.from_df(df_pred, pred_config, height=5, width=5)

        # Negative Case: Overflow (Pred=2, Hist=1)
        with pytest.raises(ValueError, match="Contract Violation"):
            pred_vh.to_evaluation_df(history, start_idx=0)

        # Positive Case: Aligned (Pred=2, Hist=3)
        data_3 = [
            {"month_id": 100, "priogrid_gid": 1, "row": 10, "col": 20, "feat_1": 1.0, "feat_2": 2.0},
            {"month_id": 101, "priogrid_gid": 1, "row": 10, "col": 20, "feat_1": 1.0, "feat_2": 2.0},
            {"month_id": 102, "priogrid_gid": 1, "row": 10, "col": 20, "feat_1": 1.0, "feat_2": 2.0},
        ]
        history_3 = VolumeHandler.from_df(pd.DataFrame(data_3), standard_config, height=5, width=5)
        res_df = pred_vh.to_evaluation_df(history_3, start_idx=0)
        assert len(res_df) == 2
        assert sorted(res_df["month_id"].tolist()) == [100, 101]

    def test_array_like_metadata_access(self, standard_config):
        """Verify that VolumeHandler exposes shape and len properties correctly."""
        data = np.zeros((10, 5, 5, 4)) # T=10
        vh = VolumeHandler(
            data=data,
            axes=("T", "H", "W", "C"),
            channel_map=["t", "i", "f1", "f2"],
            time_col="t", id_col="i", spatial_cols=["y", "x"]
        )
        assert vh.shape == (10, 5, 5, 4)
        assert len(vh) == 10

    def test_to_forecast_df_continuity(self, custom_alias_config):
        """Verify future calendar projection and Ledger adherence."""
        df_hist = pd.DataFrame([{"temporal": 500, "unit": 1, "y": 0, "x": 0, "signal": 1.0}])
        history = VolumeHandler.from_df(df_hist, custom_alias_config, height=2, width=2)

        # 3-month future prediction
        # Must have distinct months to have duration 3
        df_pred = pd.DataFrame([
            {"temporal": 501, "unit": 1, "y": 0, "x": 0, "p1": 0.1},
            {"temporal": 502, "unit": 1, "y": 0, "x": 0, "p1": 0.1},
            {"temporal": 503, "unit": 1, "y": 0, "x": 0, "p1": 0.1},
        ])
        pred_config = custom_alias_config.copy()
        pred_config["features"] = ["p1"]
        pred_vh = VolumeHandler.from_df(df_pred, pred_config, height=2, width=2)

        res_df = pred_vh.to_forecast_df(history)
        
        assert len(res_df) == 3
        # Check calendar: 500 -> 501, 502, 503
        assert sorted(res_df["temporal"].unique().tolist()) == [501, 502, 503]
        # Check Identity authority
        assert (res_df["unit"] == 1).all()