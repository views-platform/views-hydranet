import logging

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from views_hydranet.forecast.execution import forecast_with_model_artifact

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

class ToyHydranet(nn.Module):
    """
    A minimal model that mimics the HydraNet interface required by predict() and sample_posterior().
    """
    def __init__(self, base=16):
        super().__init__()
        self.base = base
        # Mock layers to satisfy any potential internal checks
        self.conv = nn.Conv2d(3, 6, kernel_size=3, padding=1) # 3 in, 6 out (3 magnitude + 3 class)

    def forward(self, x, h):
        # input x: [batch, features, H, W]
        # Hydranet returns (magnitude_logits, class_logits, next_h)
        # We simulate 3 targets (sb, ns, os)
        batch, _, h_dim, w_dim = x.shape
        mag = torch.zeros((batch, 3, h_dim, w_dim))
        cls = torch.zeros((batch, 3, h_dim, w_dim))
        return mag, cls, h

    def init_hTtime(self, hidden_channels, H, W):
        # Hydranet's hidden state initialization
        return torch.zeros((1, hidden_channels, H, W))

@pytest.fixture
def mock_setup(tmp_path):
    """
    Sets up a temporary environment with a model artifact and mock input data.
    """
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    model = ToyHydranet(base=8)
    model_path = artifacts_dir / "toy_model.pt"
    torch.save(model, model_path)

    # Mock views_vol: [months, H, W, features]
    # Channels: 0:pg_id, 1:col, 2:row, 3:month_id, 4:c_id, 5:sb, 6:ns, 7:os
    H, W = 10, 10
    months = 5
    vol = np.zeros((months, H, W, 8))
    for t in range(months):
        vol[t, :, :, 0] = np.arange(1, H*W + 1).reshape(H, W) # pg_id
        vol[t, :, :, 3] = 500 + t # month_id
        vol[t, :, :, 4] = 10 # c_id

    config = {
        "time_steps": 2,
        "test_samples": 3,
        "run_type": "test_run",
        "model_time_stamp": "20260128_120000",
        "input_channels": 3,
        "freeze_h": "none"
    }

    return model_path, torch.from_numpy(vol).float(), config, artifacts_dir

def test_forecast_integration_end_to_end(mock_setup):
    """
    Test the full flow: .pt file -> forecast_with_model_artifact -> Contract DataFrames.
    """
    model_path, views_vol, config, artifacts_dir = mock_setup
    device = torch.device("cpu")

    # Execute the forecasting flow
    results = forecast_with_model_artifact(
        config=config,
        device=device,
        views_vol=views_vol,
        PATH_ARTIFACTS=artifacts_dir,
        artifact_name="toy_model.pt"
    )

    # Assertions
    assert isinstance(results, dict)
    assert set(results.keys()) == {"sb", "ns", "os"}

    for target, df_list in results.items():
        assert isinstance(df_list, list)
        assert len(df_list) == 1
        df = df_list[0]

        # Check Contract Compliance
        assert isinstance(df.index, pd.MultiIndex)
        assert df.index.names == ["month_id", "priogrid_gid"]

        expected_col = f"pred_lr_{target}"
        assert expected_col in df.columns

        # Check sample size
        first_val = df.iloc[0][expected_col]
        assert isinstance(first_val, list)
        assert len(first_val) == config["test_samples"]

        # Verify month_id logic (predicting beyond input)
        unique_months = df.index.get_level_values("month_id").unique()
        assert list(unique_months) == [503, 504]

if __name__ == "__main__":
    # Allow manual run
    pytest.main([__file__])
