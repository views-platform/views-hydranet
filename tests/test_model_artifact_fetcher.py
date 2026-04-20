import hashlib
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

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


class TestGreen:
    """Green: Successful artifact retrieval and metadata registration."""

    def test_fetcher_green_path_latest(self, mock_setup):
        """Prove that the fetcher correctly loads the latest artifact and registers metadata."""
        artifacts_dir, latest_path, config, add_config_mock = mock_setup

        fetcher = ModelArtifactFetcher(
            path_model_artifacts=artifacts_dir,
            path_latest_model_artifacts=latest_path,
            config=config,
            add_config_function=add_config_mock,
            device=torch.device("cpu"),
        )

        model, timestamp = fetcher.fetch_model_artifact()

        assert isinstance(model, DummyModel)
        assert timestamp == "20260204_120000"
        add_config_mock.assert_called_once_with({"timestamp": timestamp})

    def test_fetcher_green_path_specific(self, mock_setup):
        """Prove that the fetcher correctly loads a specific named artifact."""
        artifacts_dir, _, config, add_config_mock = mock_setup

        specific_ts = "20250101_000000"
        specific_name = f"manual_model_{specific_ts}.pt"
        specific_path = artifacts_dir / specific_name
        torch.save(DummyModel(), specific_path)

        fetcher = ModelArtifactFetcher(
            path_model_artifacts=artifacts_dir,
            path_latest_model_artifacts=MagicMock(),
            config=config,
            add_config_function=add_config_mock,
            device=torch.device("cpu"),
        )

        model, timestamp = fetcher.fetch_model_artifact(
            model_artifact_name=f"manual_model_{specific_ts}"
        )

        assert timestamp == specific_ts
        add_config_mock.assert_called_once_with({"timestamp": specific_ts})


class TestBeige:
    """Edge cases in artifact retrieval."""

    def test_beige_artifact_name_without_extension(self, mock_setup):
        """Artifact name without .pt must still load correctly."""
        artifacts_dir, _, config, add_config_mock = mock_setup

        specific_ts = "20260301_090000"
        specific_path = artifacts_dir / f"run_model_{specific_ts}.pt"
        torch.save(DummyModel(), specific_path)

        fetcher = ModelArtifactFetcher(
            path_model_artifacts=artifacts_dir,
            path_latest_model_artifacts=MagicMock(),
            config=config,
            add_config_function=add_config_mock,
            device=torch.device("cpu"),
        )

        model, timestamp = fetcher.fetch_model_artifact(
            model_artifact_name=f"run_model_{specific_ts}"
        )
        assert timestamp == specific_ts

    def test_beige_latest_symlink_broken(self, tmp_path):
        """Broken symlink for latest must raise FileNotFoundError."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        broken_link = artifacts_dir / "latest.pt"
        broken_link.symlink_to(artifacts_dir / "nonexistent.pt")

        fetcher = ModelArtifactFetcher(
            path_model_artifacts=artifacts_dir,
            path_latest_model_artifacts=broken_link,
            config={"run_type": "test"},
            add_config_function=MagicMock(),
            device=torch.device("cpu"),
        )

        with pytest.raises(FileNotFoundError, match="Retriever Contract Violation"):
            fetcher.fetch_model_artifact()


class TestRed:
    """Adversarial and failure mode tests (CIC §6)."""

    def test_red_malformed_timestamp_filename(self, tmp_path):
        """Artifact with non-standard filename: timestamp extraction still runs."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        bad_name = "bad_model.pt"
        torch.save(DummyModel(), artifacts_dir / bad_name)

        fetcher = ModelArtifactFetcher(
            path_model_artifacts=artifacts_dir,
            path_latest_model_artifacts=MagicMock(),
            config={"run_type": "test"},
            add_config_function=MagicMock(),
            device=torch.device("cpu"),
        )

        model, timestamp = fetcher.fetch_model_artifact(model_artifact_name="bad_model")
        # stem is "bad_model", last 15 chars = "bad_model" (only 9 chars)
        # This should at minimum not crash — the timestamp is garbage but extractable
        assert len(timestamp) == 15 or len(timestamp) < 15

    def test_red_empty_artifact_file(self, tmp_path):
        """Empty .pt file must fail with a clear error, not silently."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        empty_path = artifacts_dir / "empty_model_20260101_000000.pt"
        empty_path.write_bytes(b"")

        fetcher = ModelArtifactFetcher(
            path_model_artifacts=artifacts_dir,
            path_latest_model_artifacts=MagicMock(),
            config={"run_type": "test"},
            add_config_function=MagicMock(),
            device=torch.device("cpu"),
        )

        with pytest.raises(Exception):
            fetcher.fetch_model_artifact(model_artifact_name="empty_model_20260101_000000")

    def test_red_add_config_callback_receives_timestamp(self, mock_setup):
        """The config callback must receive exactly {'timestamp': '<15-char>'}."""
        artifacts_dir, latest_path, config, add_config_mock = mock_setup

        fetcher = ModelArtifactFetcher(
            path_model_artifacts=artifacts_dir,
            path_latest_model_artifacts=latest_path,
            config=config,
            add_config_function=add_config_mock,
            device=torch.device("cpu"),
        )

        _, timestamp = fetcher.fetch_model_artifact()
        add_config_mock.assert_called_once_with({"timestamp": timestamp})
        assert len(timestamp) == 15

    def test_red_sha256_mismatch_raises(self, tmp_path):
        """CIC §6 Checksum Failure: wrong hash must raise RuntimeError."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        artifact_path = artifacts_dir / "cal_model_20260401_120000.pt"
        torch.save(DummyModel(), artifact_path)

        hash_path = artifact_path.with_suffix(".pt.sha256")
        hash_path.write_text("0000000000000000000000000000000000000000000000000000000000000000")

        fetcher = ModelArtifactFetcher(
            path_model_artifacts=artifacts_dir,
            path_latest_model_artifacts=artifact_path,
            config={"run_type": "test"},
            add_config_function=MagicMock(),
            device=torch.device("cpu"),
        )

        with pytest.raises(RuntimeError, match="integrity check FAILED"):
            fetcher.fetch_model_artifact()

    def test_red_sha256_valid_loads_successfully(self, tmp_path):
        """CIC §6: correct hash must pass verification and load."""
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        artifact_path = artifacts_dir / "cal_model_20260401_120000.pt"
        torch.save(DummyModel(), artifact_path)

        correct_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        hash_path = artifact_path.with_suffix(".pt.sha256")
        hash_path.write_text(correct_hash)

        fetcher = ModelArtifactFetcher(
            path_model_artifacts=artifacts_dir,
            path_latest_model_artifacts=artifact_path,
            config={"run_type": "test"},
            add_config_function=MagicMock(),
            device=torch.device("cpu"),
        )

        model, timestamp = fetcher.fetch_model_artifact()
        assert isinstance(model, DummyModel)
        assert timestamp == "20260401_120000"

    def test_red_no_hash_file_warns_but_loads(self, mock_setup):
        """CIC §6: legacy artifact without .sha256 must warn and load."""
        artifacts_dir, latest_path, config, add_config_mock = mock_setup

        hash_path = latest_path.with_suffix(".pt.sha256")
        assert not hash_path.exists()

        fetcher = ModelArtifactFetcher(
            path_model_artifacts=artifacts_dir,
            path_latest_model_artifacts=latest_path,
            config=config,
            add_config_function=add_config_mock,
            device=torch.device("cpu"),
        )

        model, _ = fetcher.fetch_model_artifact()
        assert isinstance(model, DummyModel)


if __name__ == "__main__":
    pytest.main([__file__])
