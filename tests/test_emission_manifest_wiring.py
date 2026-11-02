"""ADR-074: the manifest's values must come from the objects the run used.

The module that holds the record is tested in `tests/infrastructure/test_emission_manifest.py`.
This file tests the half that can go wrong invisibly: whether the values recorded are the ones the
run actually used, or a plausible reconstruction of them.

⛔ The distinction is the whole point of ADR-074 §2.3. A manifest assembled by re-reading the
configuration, or by re-deriving a month from a convention, would agree with the configuration **by
construction** — it would pass every check that compares the two, while describing a run that never
happened. So the tests here are about provenance, not about shape.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("views_pipeline_core")

from views_hydranet.utils.feature_scaler import FeatureScaler
from views_hydranet.utils.inference_orchestrator import InferenceOrchestrator
from views_hydranet.utils.volume_handler import VolumeHandler

TARGETS = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os", "by_sb_best", "by_ns_best", "by_os_best"]

FIRST_MONTH = 100
N_MONTHS = 21
S = 4

CFG = {
    "run_type": "calibration",
    "steps": [1, 2],
    "time_steps": 2,
    "input_channels": 3,
    "output_channels": 1,
    "regression_targets": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "identity_cols": ["row", "col"],
    "features": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
    "transformations": {"identity": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]},
    "derivations": {
        "binary": [
            {"from": "lr_ged_sb", "to": "by_sb_best", "threshold": 0},
            {"from": "lr_ged_ns", "to": "by_ns_best", "threshold": 0},
            {"from": "lr_ged_os", "to": "by_os_best", "threshold": 0},
        ]
    },
    "height": 2,
    "width": 2,
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "index_names": ["month_id", "priogrid_gid"],
    "spatial_cols": ["row", "col"],
    "row_offset": 0,
    "col_offset": 0,
    "model": "HydraBNUNet06_LSTM4",
    "window_dim": 2,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "norm",
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "windows_per_lesson": 1,
    "scheduler": "none",
    "warmup_steps": 1,
    "clip_grad_norm": True,
    "loss_reg": "mse",
    "loss_class": "bce",
    "loss_reg_a": 1,
    "loss_reg_c": 1,
    "loss_class_gamma": 1,
    "loss_class_alpha": 1,
    "total_lessons": 1,
    "n_posterior_samples": 4,
    "np_seed": 42,
    "torch_seed": 42,
    "min_events": 0,
    "slope_ratio": 0.1,
    "roof_ratio": 0.1,
    "max_ratio": 0.9,
    "min_ratio": 0.1,
    "freeze_h": "none",
    "sampling_strategy": "threshold",
    "evaluation_mode": "stochastic",
    "aggregate_method": "arithmetic_mean",
}


def _df() -> pd.DataFrame:
    rows = []
    for t in range(FIRST_MONTH, FIRST_MONTH + N_MONTHS):
        for r in range(2):
            for c in range(2):
                rows.append(
                    {
                        "month_id": t,
                        "priogrid_gid": r * 2 + c + 1,
                        "row": float(r),
                        "col": float(c),
                        "lr_ged_sb": float(r + c + t * 0.01),
                        "lr_ged_ns": 0.5,
                        "lr_ged_os": 0.0,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def env():
    handler = VolumeHandler.from_df(_df(), CFG)
    scaler = FeatureScaler(CFG)
    scaler._is_fitted = True
    return handler, scaler, MagicMock(spec=torch.nn.Module)


def _run(env, origins):
    """Run the real orchestrator over *origins* with a mocked model, and return it."""
    handler, scaler, model = env
    posterior = np.random.rand(CFG["time_steps"], 2, 2, 6, S)
    orchestrator = InferenceOrchestrator(CFG, model, torch.device("cpu"))
    with patch("views_hydranet.utils.inference_orchestrator.HydraNetInference") as mock_cls:
        mock_cls.return_value.generate_posterior_samples.return_value = (posterior, None)
        orchestrator.generate_prediction_frames(
            handler, scaler, origins=origins, all_targets=TARGETS
        )
    return orchestrator


def test_both_origin_numberings_are_recorded_and_they_differ(env):
    """The sequence number and the origin itself are different facts, and both are recorded.

    Today only the sequence number reaches disk, in a directory name, which is why two artifacts
    of one run cannot be joined without re-deriving how origins were chosen (ADR-074 §1).
    """
    handler, _, _ = env
    origins = [handler.shape[0] - 4, handler.shape[0] - 3]
    orch = _run(env, origins)

    assert [o.sequence_number for o in orch.emitted_origins] == [0, 1]
    assert [o.origin_index for o in orch.emitted_origins] == origins
    assert origins != [0, 1], "the fixture must make the two numberings distinguishable"


def test_the_months_match_the_window_each_origin_forecast_over(env):
    """The recorded months are this origin's window, and a different origin gives different months.

    ⚠️ **This is deliberately not claimed as a provenance test, and cannot be one here.** A volume's
    time axis is always a dense range from the observed minimum (see issue #512), so a map produced
    by arithmetic on the origin index would be indistinguishable from one read off the window. The
    test therefore establishes that the map is per-origin and correct for that origin — not that it
    was read rather than computed.

    What makes the implementation read rather than compute is a different guard: the module cannot
    import a configuration loader or subprocess, asserted in
    `tests/infrastructure/test_emission_manifest.py`.
    """
    handler, _, _ = env
    first, second = handler.shape[0] - 4, handler.shape[0] - 3
    orch = _run(env, [first, second])

    def window_months(origin):
        window = handler.slice_time(origin + 1, origin + 1 + CFG["time_steps"])
        t_axis = window.get_axis_idx("T")
        m_idx = window.channel_map.index(window.time_col)
        times = np.asarray(window.data)[..., m_idx]
        return tuple(
            int(np.take(times, k, axis=t_axis).flat[0]) for k in range(times.shape[t_axis])
        )

    for recorded, origin in zip(orch.emitted_origins, (first, second)):
        expected = window_months(origin)
        assert recorded.horizon_months == expected
        assert recorded.first_month == expected[0]
        assert len(recorded.horizon_months) == CFG["time_steps"]

    assert orch.emitted_origins[0].horizon_months != orch.emitted_origins[1].horizon_months, (
        "both origins recorded the same months; the map is not per-origin"
    )


def test_rows_per_horizon_matches_the_emitted_frame(env):
    """The row count is measured from what was emitted, not assumed from the grid."""
    handler, _, _ = env
    orch = _run(env, [handler.shape[0] - 4])
    assert orch.emitted_origins[0].rows_per_horizon == 4  # the 2x2 fixture grid


def test_a_second_run_does_not_inherit_the_first_run_s_origins(env):
    """Without the reset, a reused orchestrator would report origins from a previous run."""
    handler, _, _ = env
    orch = _run(env, [handler.shape[0] - 4, handler.shape[0] - 3])
    assert len(orch.emitted_origins) == 2

    handler, scaler, model = env
    posterior = np.random.rand(CFG["time_steps"], 2, 2, 6, S)
    with patch("views_hydranet.utils.inference_orchestrator.HydraNetInference") as mock_cls:
        mock_cls.return_value.generate_posterior_samples.return_value = (posterior, None)
        orch.generate_prediction_frames(
            handler, scaler, origins=[handler.shape[0] - 4], all_targets=TARGETS
        )
    assert len(orch.emitted_origins) == 1, "origins accumulated across runs"


def test_a_month_that_varies_across_cells_is_refused(env):
    """⭐ A month is a property of a time step, not of a cell.

    If the time channel ever varied spatially, a horizon-to-month map built from it would be
    confidently wrong — one cell's month presented as the step's. Refusing is the only safe
    reading, and this repository's worst defects were of exactly this shape: a value taken from a
    position that happened to work.
    """
    handler, _, _ = env
    window = handler.slice_time(1, 1 + CFG["time_steps"])
    m_idx = window.channel_map.index(window.time_col)
    corrupted = np.array(np.asarray(window.data), copy=True)
    corrupted[0, 0, 0, m_idx] += 1  # one cell disagrees with its own time step

    broken = VolumeHandler(
        data=corrupted,
        axes=window.axes,
        channel_map=window.channel_map,
        time_col=window.time_col,
        id_col=window.id_col,
        spatial_cols=window.spatial_cols,
        identity_cols=window.identity_cols,
        feature_cols=window.feature_cols,
        spatial_offset=window.spatial_offset,
    )

    with pytest.raises(ValueError, match="not constant across the spatial axes"):
        InferenceOrchestrator._months_of(broken)
