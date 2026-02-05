
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch
from views_hydranet.utils.model_artifact_fetcher import ModelArtifactFetcher

# Minimal model for testing
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.ones(1))

@pytest.fixture
def mock_setup(tmp_path):
    # Setup directories
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    
    # Create a dummy artifact
    # Name format: runtype_model_YYYYMMDD_HHMMSS.pt
    timestamp = "20260204_120000"
    latest_name = f"test_model_{timestamp}.pt"
    latest_path = artifacts_dir / latest_name
    
    model = DummyModel()
    torch.save(model, latest_path)
    
    config = {"run_type": "test"}
    add_config_mock = MagicMock()
    
    return artifacts_dir, latest_path, config, add_config_mock

# --- GREEN TEAM: SUCCESSFUL RETRIEVAL ---

def test_fetcher_green_path_latest(mock_setup):
    """Prove that the fetcher correctly loads the latest artifact and registers metadata."""
    artifacts_dir, latest_path, config, add_config_mock = mock_setup
    
    fetcher = ModelArtifactFetcher(
        path_model_artifacts=artifacts_dir,
        path_latest_model_artifacts=latest_path,
        config=config,
        add_config_function=add_config_mock,
        device=torch.device("cpu")
    )
    
    # Execute
    model, timestamp = fetcher.fetch_model_artifact()
    
    # Audit
    assert isinstance(model, DummyModel)
    assert timestamp == "20260204_120000"
    # Verify the config handshake
    add_config_mock.assert_called_once_with({"timestamp": timestamp})
    print("✅ Green Team: Latest retrieval verified.")

def test_fetcher_green_path_specific(mock_setup):
    """Prove that the fetcher correctly loads a specific named artifact."""
    artifacts_dir, _, config, add_config_mock = mock_setup
    
    # Create another specific artifact
    specific_ts = "20250101_000000"
    specific_name = f"manual_model_{specific_ts}.pt"
    specific_path = artifacts_dir / specific_name
    torch.save(DummyModel(), specific_path)
    
    fetcher = ModelArtifactFetcher(
        path_model_artifacts=artifacts_dir,
        path_latest_model_artifacts=MagicMock(), # Should not be used
        config=config,
        add_config_function=add_config_mock,
        device=torch.device("cpu")
    )
    
    # Execute with specific name (without extension)
    model, timestamp = fetcher.fetch_model_artifact(model_artifact_name=f"manual_model_{specific_ts}")
    
    # Audit
    assert timestamp == specific_ts
    add_config_mock.assert_called_once_with({"timestamp": specific_ts})
    print("✅ Green Team: Specific retrieval verified.")

# --- BEIGE TEAM: ROBUSTNESS ---

def test_fetcher_beige_missing_file(mock_setup):
    """Prove that missing files raise a contract violation error."""
    artifacts_dir, _, config, add_config_mock = mock_setup
    
    fetcher = ModelArtifactFetcher(
        path_model_artifacts=artifacts_dir,
        path_latest_model_artifacts=artifacts_dir / "non_existent.pt",
        config=config,
        add_config_function=add_config_mock,
        device=torch.device("cpu")
    )
    
    with pytest.raises(FileNotFoundError, match="Retriever Contract Violation"):
        fetcher.fetch_model_artifact()
    print("✅ Beige Team: Missing file handled correctly.")

if __name__ == "__main__":
    pytest.main([__file__])
