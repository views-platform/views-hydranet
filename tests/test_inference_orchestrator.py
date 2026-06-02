import gc
import weakref
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from tests.test_inference_memory_hygiene import (
    MEMORY_CFG,
    _make_handler,
    _make_mock_inference,
)
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator

# ─── Streaming tests: InferenceOrchestrator.generate_prediction_frames_streaming ─


class TestGreen:
    """Green: InferenceOrchestrator streaming path calls sink, frees memory."""

    def test_streaming_calls_sink_once_per_origin(self):
        """origin_sink must be called exactly len(origins) times."""
        inference = _make_mock_inference()
        handler = _make_handler()
        scaler = MagicMock()
        scaler.inverse_transform_volume.side_effect = lambda h: h

        origins = [3]
        orchestrator = InferenceOrchestrator(MEMORY_CFG, inference.model, torch.device("cpu"))

        with patch(
            "views_hydranet.utils.inference_orchestrator.HydraNetInference",
            return_value=inference,
        ):
            sink_calls = []
            orchestrator.generate_prediction_frames_streaming(
                handler,
                scaler,
                origins=origins,
                all_targets=(
                    MEMORY_CFG["regression_targets"] + MEMORY_CFG["classification_targets"]
                ),
                origin_sink=lambda i, d: sink_calls.append(i),
            )

        assert len(sink_calls) == len(origins)

    def test_streaming_pf_dict_contains_correct_target_keys(self):
        """Every pf_dict passed to sink has the correct target keys."""

        inference = _make_mock_inference()
        handler = _make_handler()
        scaler = MagicMock()
        scaler.inverse_transform_volume.side_effect = lambda h: h

        all_targets = MEMORY_CFG["regression_targets"] + MEMORY_CFG["classification_targets"]
        origins = [3]
        orchestrator = InferenceOrchestrator(MEMORY_CFG, inference.model, torch.device("cpu"))
        received_key_sets = []

        with patch(
            "views_hydranet.utils.inference_orchestrator.HydraNetInference",
            return_value=inference,
        ):
            orchestrator.generate_prediction_frames_streaming(
                handler,
                scaler,
                origins=origins,
                all_targets=all_targets,
                origin_sink=lambda i, d: received_key_sets.append(set(d.keys())),
            )

        assert len(received_key_sets) == 1
        assert received_key_sets[0] == set(all_targets)

    def test_streaming_frees_pf_dict_after_sink(self):
        """pf_dict must not be alive after origin_sink returns."""
        inference = _make_mock_inference()
        handler = _make_handler()
        scaler = MagicMock()
        scaler.inverse_transform_volume.side_effect = lambda h: h

        all_targets = MEMORY_CFG["regression_targets"] + MEMORY_CFG["classification_targets"]
        origins = [3, 2]

        weak_refs = []

        def capturing_sink(i, pf_dict):
            for pf in pf_dict.values():
                weak_refs.append(weakref.ref(pf))
            # sink does NOT hold pf_dict

        orchestrator = InferenceOrchestrator(MEMORY_CFG, inference.model, torch.device("cpu"))

        with patch(
            "views_hydranet.utils.inference_orchestrator.HydraNetInference",
            return_value=inference,
        ):
            orchestrator.generate_prediction_frames_streaming(
                handler,
                scaler,
                origins=origins,
                all_targets=all_targets,
                origin_sink=capturing_sink,
            )

        gc.collect()
        alive = [r for r in weak_refs if r() is not None]
        assert len(alive) == 0, (
            f"{len(alive)} PredictionFrame(s) still alive after streaming "
            "completed. generate_prediction_frames_streaming() must del "
            "pf_dict after origin_sink."
        )

    def test_streaming_pf_matches_batch_pf(self):
        """
        Parity test: streaming and batch paths must produce identical
        y_pred arrays for the same input.
        """

        inference = _make_mock_inference()
        handler = _make_handler()
        scaler = MagicMock()
        scaler.inverse_transform_volume.side_effect = lambda h: h

        all_targets = MEMORY_CFG["regression_targets"] + MEMORY_CFG["classification_targets"]
        origins = [3]
        orchestrator = InferenceOrchestrator(MEMORY_CFG, inference.model, torch.device("cpu"))

        with patch(
            "views_hydranet.utils.inference_orchestrator.HydraNetInference",
            return_value=inference,
        ):
            # Batch path
            batch_pf_dicts = orchestrator.generate_prediction_frames(
                handler,
                scaler,
                origins=origins,
                all_targets=all_targets,
            )

            # Streaming path
            streaming_pf_dicts = []
            orchestrator.generate_prediction_frames_streaming(
                handler,
                scaler,
                origins=origins,
                all_targets=all_targets,
                origin_sink=lambda i, d: streaming_pf_dicts.append(d),
            )

        assert len(batch_pf_dicts) == len(streaming_pf_dicts)
        for batch_dict, stream_dict in zip(batch_pf_dicts, streaming_pf_dicts):
            assert set(batch_dict.keys()) == set(stream_dict.keys())
            for target in batch_dict:
                np.testing.assert_array_equal(
                    batch_dict[target].y_pred,
                    stream_dict[target].y_pred,
                    err_msg=(
                        f"y_pred mismatch for target '{target}': "
                        "batch and streaming paths diverged."
                    ),
                )
