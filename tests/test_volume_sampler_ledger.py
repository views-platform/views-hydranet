
import pytest
import numpy as np
import pandas as pd
import torch
from views_hydranet.utils.volume_handler import VolumeHandler
from views_hydranet.utils.volume_sampler import VolumeSampler

class TestVolumeSamplerLedger:
    """
    Rigorously verifies compliance with ADR 009: VolumeSampler (The Lens).
    Tests focus on Zero-Magic, Reproducibility, and Topological Integrity.
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
        # 5 months total. Train will be 5 - 2 = 3 months.
        # Populate coordinates geographically
        data = np.zeros((5, 10, 10, 5))
        for y in range(10):
            for x in range(10):
                data[:, y, x, 2] = 100 + y # y channel
                data[:, y, x, 3] = 200 + x # x channel
                data[:, y, x, 1] = (100+y)*1000 + (200+x) # i channel (id)
        
        # Busy pixel at T=0, (5,5)
        data[0, 5, 5, 4] = 1.0

        return VolumeHandler(
            data=data, axes=("T", "H", "W", "C"), channel_map=["t", "i", "y", "x", "f1"],
            time_col="t", id_col="i", spatial_cols=["y", "x"],
            identity_cols=["t", "i", "y", "x"], feature_cols=["f1"],
            spatial_offset=(100, 200)
        )

    def test_handshake_validation(self, global_handler, global_config):
        """Verify that sampler rejects invalid configurations."""
        bad_config = global_config.copy()
        bad_config["window_dim"] = 32
        with pytest.raises(ValueError, match="exceeds handler spatial bounds"):
            VolumeSampler(global_handler, bad_config)

    def test_reproducibility(self, global_handler, global_config):
        """Verify that two samplers with same seed produce identical windows."""
        s1 = VolumeSampler(global_handler, global_config)
        s2 = VolumeSampler(global_handler, global_config)
        
        b1 = s1.get_next_batch(0)
        b2 = s2.get_next_batch(0)
        
        for v1, v2 in zip(b1, b2):
            assert np.array_equal(v1.data, v2.data)

    def test_topological_integrity(self, global_handler, global_config):
        """Verify that sampled windows correctly map to global geographic coordinates."""
        sampler = VolumeSampler(global_handler, global_config)
        batch = sampler.get_next_batch(0)
        sample = batch[0]
        
        # Reconstruct sample to DF
        df_sample = sample.to_historical_df()
        test_row = df_sample.iloc[0]
        
        # Reconstruct global to DF
        df_global = global_handler.to_historical_df()
        
        # Find the same pixel in global using the ledger values from the sample
        match = df_global[(df_global["y"] == test_row["y"]) & (df_global["x"] == test_row["x"])]
        assert len(match) > 0
        assert match.iloc[0]["i"] == test_row["i"]

    def test_ledger_inheritance(self, global_handler, global_config):
        """Verify that mini-handlers inherit all ledger roles."""
        sampler = VolumeSampler(global_handler, global_config)
        batch = sampler.get_next_batch(0)
        sample = batch[0]
        
        assert sample.time_col == global_handler.time_col
        assert sample.id_col == global_handler.id_col
        assert sample.spatial_cols == global_handler.spatial_cols

    def test_busy_first_strategy(self, global_handler, global_config):
        """Verify that windows prioritize cells with activity."""
        sampler = VolumeSampler(global_handler, global_config)
        
        # Pull multiple batches, all should contain the busy pixel (5,5)
        # Busy pixel is at (5,5) relative to (0,0) in the global 10x10.
        # Since global offset is (100, 200), its geographic coord is (105, 205).
        for i in range(10):
            batch = sampler.get_next_batch(i)
            sample = batch[0]
            y0, x0 = sample.spatial_offset
            dim = global_config["window_dim"]
            # Check if (105, 205) is in range [y0, y0+dim], [x0, x0+dim]
            assert y0 <= 105 < y0 + dim
            assert x0 <= 205 < x0 + dim
