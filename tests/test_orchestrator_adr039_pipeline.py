"""
TDD — RED + characterization tests for Item 2: Extract shared ADR-039 Steps 1-5
from InferenceOrchestrator.

The three public methods (generate_forecasts, generate_prediction_frames,
generate_prediction_frames_streaming) currently duplicate Steps 1-5 verbatim.
A single _run_adr039_pipeline() should enforce parity.

RED tests: assert the extracted method exists.
Characterization: assert parity between all three paths on a toy volume.
"""

from unittest.mock import MagicMock, patch

import pytest

from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config(mode="stochastic"):
    return {
        "regression_targets": ["lr_sb_best"],
        "classification_targets": ["by_sb_best"],
        "features": ["lr_sb_best"],
        "evaluation_mode": mode,
        "aggregate_method": "arithmetic_mean",
        "n_posterior_samples": 2,
        "freeze_h": "none",
        "time_steps": 2,
        "steps": [1, 2],
    }


def _make_orchestrator(config=None):
    cfg = config or _make_config()
    model = MagicMock()
    device = MagicMock()
    device.type = "cpu"
    device.__str__ = lambda self: "cpu"
    return InferenceOrchestrator(cfg, model, device)


# ---------------------------------------------------------------------------
# RED TESTS — _run_adr039_pipeline must exist
# ---------------------------------------------------------------------------

class TestAdr039PipelineMethodExists:
    """RED GATE: The shared pipeline method must exist after refactor."""

    def test_method_exists(self):
        """InferenceOrchestrator must have _run_adr039_pipeline as a method."""
        orch = _make_orchestrator()
        assert hasattr(orch, "_run_adr039_pipeline"), (
            "InferenceOrchestrator is missing _run_adr039_pipeline(). "
            "Steps 1-5 of generate_forecasts / generate_prediction_frames / "
            "generate_prediction_frames_streaming must be extracted to this "
            "shared private method to enforce parity."
        )

    def test_method_is_callable(self):
        """_run_adr039_pipeline must be callable."""
        orch = _make_orchestrator()
        assert callable(getattr(orch, "_run_adr039_pipeline", None)), (
            "_run_adr039_pipeline must be a callable method."
        )


# ---------------------------------------------------------------------------
# STRUCTURAL PARITY TESTS — all three paths must call the shared method
# ---------------------------------------------------------------------------

class TestAdr039PipelineParity:
    """
    After the refactor, all three public methods must delegate Steps 1-5
    to _run_adr039_pipeline. This test verifies the shared method is called.
    """

    def test_generate_prediction_frames_calls_shared_pipeline(self):
        """generate_prediction_frames must delegate to _run_adr039_pipeline."""
        orch = _make_orchestrator()
        if not hasattr(orch, "_run_adr039_pipeline"):
            pytest.skip("_run_adr039_pipeline not yet implemented")

        mock_pred = MagicMock()
        mock_pred.to_evaluation_pf.return_value = {"lr_sb_best": MagicMock()}
        mock_window = MagicMock()

        with (
            patch.object(orch, "_run_adr039_pipeline", return_value=(mock_pred, mock_window)),
            patch("views_hydranet.utils.inference_orchestrator.HydraNetInference"),
        ):
            handler = MagicMock()
            handler.shape = (10, 4, 4, 5)
            scaler = MagicMock()
            orch.generate_prediction_frames(
                handler, scaler, origins=[5], all_targets=["lr_sb_best", "by_sb_best"],
            )
            orch._run_adr039_pipeline.assert_called_once()

    def test_generate_forecasts_calls_shared_pipeline(self):
        """generate_forecasts must delegate to _run_adr039_pipeline."""
        orch = _make_orchestrator()
        if not hasattr(orch, "_run_adr039_pipeline"):
            pytest.skip("_run_adr039_pipeline not yet implemented")

        mock_pred = MagicMock()
        mock_pred.to_evaluation_df.return_value = MagicMock()
        mock_window = MagicMock()

        with (
            patch.object(orch, "_run_adr039_pipeline", return_value=(mock_pred, mock_window)),
            patch("views_hydranet.utils.inference_orchestrator.HydraNetInference"),
        ):
            handler = MagicMock()
            handler.shape = (10, 4, 4, 5)
            scaler = MagicMock()
            orch.generate_forecasts(handler, scaler, origins=[5])
            orch._run_adr039_pipeline.assert_called_once()

    def test_streaming_calls_shared_pipeline(self):
        """generate_prediction_frames_streaming must delegate to _run_adr039_pipeline."""
        orch = _make_orchestrator()
        if not hasattr(orch, "_run_adr039_pipeline"):
            pytest.skip("_run_adr039_pipeline not yet implemented")

        mock_pred = MagicMock()
        mock_pred.to_evaluation_pf.return_value = {"lr_sb_best": MagicMock()}
        mock_window = MagicMock()

        with (
            patch.object(orch, "_run_adr039_pipeline", return_value=(mock_pred, mock_window)),
            patch("views_hydranet.utils.inference_orchestrator.HydraNetInference"),
        ):
            handler = MagicMock()
            handler.shape = (10, 4, 4, 5)
            scaler = MagicMock()
            sink = MagicMock()
            orch.generate_prediction_frames_streaming(
                handler, scaler, origins=[5],
                all_targets=["lr_sb_best", "by_sb_best"],
                origin_sink=sink,
            )
            orch._run_adr039_pipeline.assert_called_once()
