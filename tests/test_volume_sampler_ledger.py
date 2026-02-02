import pytest
import numpy as np
import pandas as pd
import torch
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

class TestVolumeSamplerLedger:
    """
    Rigorously verifies compliance with ADR 013: VolumeSampler (The Pure Lens).
    """

    @pytest.fixture
    def global_config(self):
        return {
            "time_col": "t", "id_col": "i", "spatial_cols": ["y", "x"],
            "identity_cols": ["t", "i", "y", "x"],
            "features": ["f1"],
            "row_offset": 100, "col_offset": 200, "height": 10, "width": 10,
            "window_dim": 2, "batch_size": 2, "np_seed": 42, "steps": [1, 2],
            "min_events": 1
        }

    @pytest.fixture
    def global_handler(self, global_config):
        data = np.zeros((5, 10, 10, 5))
        for y in range(10):
            for x in range(10):
                data[:, y, x, 2] = 100 + y 
                data[:, y, x, 3] = 200 + x 
                data[:, y, x, 1] = (100+y)*1000 + (200+x) 
        data[0, 5, 5, 4] = 1.0 # Busy pixel at (5,5)
        return VolumeHandler(
            data=data, axes=("T", "H", "W", "C"), channel_map=["t", "i", "y", "x", "f1"],
            time_col="t", id_col="i", spatial_cols=["y", "x"],
            identity_cols=["t", "i", "y", "x"], feature_cols=["f1"],
            spatial_offset=(100, 200)
        )

    def test_handshake_validation(self, global_handler, global_config):
        bad_config = global_config.copy()
        bad_config["window_dim"] = 32
        with pytest.raises(ValueError, match="exceeds handler spatial bounds"):
            VolumeSampler(global_handler, bad_config)

    def test_reproducibility(self, global_handler, global_config):
        """Verify that two samplers with same seed produce identical windows."""
        s1 = VolumeSampler(global_handler, global_config)
        s2 = VolumeSampler(global_handler, global_config)
        
        # New API: get_batch(target, threshold) -> (batch, qualified_count)
        b1, _ = s1.get_batch("f1", 0, batch_size=2)
        b2, _ = s2.get_batch("f1", 0, batch_size=2)
        
        for v1, v2 in zip(b1, b2):
            assert np.array_equal(v1.data, v2.data)

    def test_topological_integrity(self, global_handler, global_config):
        sampler = VolumeSampler(global_handler, global_config)
        batch, _ = sampler.get_batch("f1", 0, batch_size=1)
        sample = batch[0]
        
        df_sample = sample.to_historical_df()
        test_row = df_sample.iloc[0]
        df_global = global_handler.to_historical_df()
        
        match = df_global[(df_global["y"] == test_row["y"]) & (df_global["x"] == test_row["x"])]
        assert len(match) > 0
        assert match.iloc[0]["i"] == test_row["i"]

    def test_ledger_inheritance(self, global_handler, global_config):
        """Verify that mini-handlers inherit all ledger roles."""
        sampler = VolumeSampler(global_handler, global_config)
        batch, _ = sampler.get_batch("f1", 0, batch_size=1)
        sample = batch[0]
        
        assert sample.time_col == global_handler.time_col
        assert sample.id_col == global_handler.id_col
        assert sample.spatial_cols == global_handler.spatial_cols

    def test_busy_first_strategy(self, global_handler, global_config):
        sampler = VolumeSampler(global_handler, global_config)
        
        # Force threshold=1 to find the busy pixel at geographic (105, 205)
        for i in range(10):
            batch, count = sampler.get_batch("f1", threshold=1, batch_size=1)
            assert count == 1 # Precisely one busy cell
            sample = batch[0]
            y0, x0 = sample.spatial_offset
            dim = global_config["window_dim"]
            assert y0 <= 105 < y0 + dim
            assert x0 <= 205 < x0 + dim