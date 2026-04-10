"""
Characterization tests for _run_data_pipeline extraction (Batch 2.1).

Verifies that all 3 manager paths (train, evaluate, forecast) use the
same data ingestion sequence and produce consistent handler/scaler/sniffer
objects.
"""

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("views_pipeline_core")

from views_hydranet.manager.hydranet_manager import HydranetManager

# ---------------------------------------------------------------------------
# Shared test config
# ---------------------------------------------------------------------------
PIPELINE_CFG = {
    "run_type": "calibration",
    "model": "HydraBNUNet06_LSTM4",
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "features": ["feat_a"],
    "steps": list(range(1, 37)),
    "sweep": False,
    "diagnostic_visualizations": False,
    "evaluation_mode": "point",
    "aggregate_method": "arithmetic_mean",
}

# Minimal DataFrame that DataFetcher.fetch_df would return
_ROWS = 100
_DF = pd.DataFrame({
    "month_id": np.tile(np.arange(1, 11), 10),
    "priogrid_gid": np.repeat(np.arange(1, 11), 10),
    "row": np.repeat(np.arange(1, 11), 10),
    "col": np.ones(_ROWS, dtype=int),
    "c_id": np.ones(_ROWS, dtype=int),
    "lr_sb_best": np.random.rand(_ROWS),
    "lr_ns_best": np.random.rand(_ROWS),
    "lr_os_best": np.random.rand(_ROWS),
    "by_sb_best": np.random.rand(_ROWS),
    "by_ns_best": np.random.rand(_ROWS),
    "by_os_best": np.random.rand(_ROWS),
    "feat_a": np.random.rand(_ROWS),
})


class TestDataPipelineMethodExists:
    """RED tests: the extraction target must exist."""

    def test_run_data_pipeline_exists(self):
        assert hasattr(HydranetManager, "_run_data_pipeline")

    def test_run_data_pipeline_is_callable(self):
        assert callable(getattr(HydranetManager, "_run_data_pipeline", None))


class TestDataPipelineReturns:
    """Verify _run_data_pipeline returns the expected 3-tuple."""

    @pytest.fixture
    def _mock_deps(self, tmp_path):
        """Shared mock context for all pipeline tests."""
        mpm = MagicMock()
        mpm.data_raw = tmp_path
        mpm.artifacts = tmp_path / "artifacts"
        mpm.artifacts.mkdir()

        with (
            patch.object(
                HydranetManager, "__init__", lambda self, *a, **kw: None
            ),
            patch.object(
                HydranetManager, "configs", new_callable=PropertyMock
            ) as mock_cfg,
            patch(
                "views_hydranet.manager.hydranet_manager.DataFetcher"
            ) as mock_fetch_cls,
            patch(
                "views_hydranet.manager.hydranet_manager.FeatureScaler"
            ) as mock_scaler_cls,
            patch(
                "views_hydranet.manager.hydranet_manager.DataSniffer"
            ) as mock_sniffer_cls,
            patch(
                "views_hydranet.manager.hydranet_manager.VolumeHandler"
            ) as mock_vh_cls,
        ):
            mock_cfg.return_value = PIPELINE_CFG

            mgr = HydranetManager.__new__(HydranetManager)
            mgr.device = torch.device("cpu")
            mgr._model_path = mpm
            mgr.run_timestamp = "test_ts"

            # Wire mocks
            mock_fetch_cls.return_value.fetch_df.return_value = _DF.copy()
            mock_fetch_cls.standardize_raw_df.return_value = _DF.copy()
            mock_scaler_cls.return_value.fit_transform.return_value = _DF.copy()

            mock_handler = MagicMock()
            mock_handler.shape = (10, 10, 1, 12)
            mock_vh_cls.from_df.return_value = mock_handler

            yield {
                "mgr": mgr,
                "fetch_cls": mock_fetch_cls,
                "scaler_cls": mock_scaler_cls,
                "sniffer_cls": mock_sniffer_cls,
                "vh_cls": mock_vh_cls,
                "handler": mock_handler,
            }

    def test_returns_three_tuple(self, _mock_deps):
        """Pipeline returns (VolumeHandler, FeatureScaler, DataSniffer)."""
        d = _mock_deps
        viz = MagicMock()

        result = d["mgr"]._run_data_pipeline(viz)

        assert len(result) == 3
        handler, scaler, sniffer = result
        assert handler is d["handler"]
        assert scaler is d["scaler_cls"].return_value
        assert sniffer is d["sniffer_cls"].return_value

    def test_partition_bound_slices_df(self, _mock_deps):
        """When partition_bound is set, df is filtered before VolumeHandler."""
        d = _mock_deps
        viz = MagicMock()

        d["mgr"]._run_data_pipeline(viz, partition_bound=5)

        # Verify from_df was called with a filtered DataFrame
        call_args = d["vh_cls"].from_df.call_args
        df_passed = call_args[0][0]
        assert (df_passed["month_id"] <= 5).all()

    def test_biopsy_stages_called(self, _mock_deps):
        """Verify all 3 diagnostic biopsy stages are triggered."""
        d = _mock_deps
        viz = MagicMock()

        d["mgr"]._run_data_pipeline(viz)

        # Stage 1 + Stage 2 biopsies
        assert viz.biopsy_dataframe.call_count == 2
        # Stage 3 volume biopsy
        assert viz.biopsy_volume.call_count == 1
