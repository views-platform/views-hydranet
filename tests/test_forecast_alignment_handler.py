"""The manager hands DataSniffer a history handler and calls it a forecast one.

`HydranetManager._run_data_pipeline` builds `handler = VolumeHandler.from_df(df, ...)` — from
history, on every run type — and then calls

    sniffer.sniff_forecast_alignment(df, handler, is_forecast=forecast)

`DataSniffer.sniff_forecast_alignment`'s forecast branch requires
`min(volume months) == max(df months) + 1`: the volume must begin one month *after* the frame it
is checked against. A handler built from that same frame begins where the frame begins, so the
condition is unsatisfiable and the call raises "Forecast Continuity Broken".

The CIC states the contract explicitly (`docs/CICs/DataSniffer.md` §8):

    sniffer.sniff_forecast_alignment(history_df, forecast_handler, is_forecast=True)

— the **forecast** handler. The manager passes the history one.

Calibration is unaffected and always has been: `forecast=True` is passed at exactly one call site
(`hydranet_manager.py`, the forecasting branch), so every other path already selects the history
branch, which is the correct check for a history-built handler.

## Why this was not caught

Six test files patch `views_hydranet.manager.hydranet_manager.DataSniffer` wholesale, and the one
that reaches this path (`test_falsification_end_to_end_claim.py`) asserts
`sniff_forecast_alignment.assert_called_once()` and inspects the kwargs. **It checks that the call
happened, never that it passes.** No test anywhere runs the manager's forecast path against the
real sniffer — so the tests below deliberately do not mock `DataSniffer` or `VolumeHandler`.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

pytest.importorskip("views_pipeline_core")

from views_hydranet.manager.hydranet_manager import HydranetManager

MONTHS = list(range(101, 109))  # eight months of history
SIDE = 3  # a 3x3 grid, so the volume is small and the arithmetic is checkable by hand

CONFIG = {
    "run_type": "forecasting",
    "model": "HydraBNUNet06_LSTM4",
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "identity_cols": ["month_id", "priogrid_gid", "row", "col"],
    "features": ["feat_a"],
    "regression_targets": ["lr_ged_sb"],
    "classification_targets": ["by_sb_best"],
    "height": SIDE,
    "width": SIDE,
    # VolumeHandler requires these explicitly (volume_handler.py: "Offsets are also strictly
    # required now"); the grid starts at the origin, so both are zero.
    "row_offset": 0,
    "col_offset": 0,
    "steps": [1, 2, 3],
    "sweep": False,
    "diagnostic_visualizations": False,
    "evaluation_mode": "point",
    "aggregate_method": "arithmetic_mean",
}


def _history_frame() -> pd.DataFrame:
    """A complete, well-formed history: every cell observed in every month."""
    rows, cols = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    cell_row = np.tile(rows.ravel(), len(MONTHS))
    cell_col = np.tile(cols.ravel(), len(MONTHS))
    month = np.repeat(MONTHS, SIDE * SIDE)
    n = len(month)
    return pd.DataFrame(
        {
            "month_id": month.astype("int64"),
            "priogrid_gid": (cell_row * SIDE + cell_col + 1).astype("int64"),
            "row": cell_row.astype("int64"),
            "col": cell_col.astype("int64"),
            "feat_a": np.linspace(0.0, 1.0, n),
            "lr_ged_sb": np.linspace(0.0, 2.0, n),
            "by_sb_best": np.zeros(n),
        }
    )


@pytest.fixture
def manager(tmp_path):
    """A manager whose data pipeline runs with the REAL DataSniffer and VolumeHandler.

    Only the two stages that would need real data or a fitted scaler are patched. Patching the
    sniffer here would erase the only thing these tests exist to observe.
    """
    frame = _history_frame()
    model_path = MagicMock()
    model_path.data_raw = tmp_path
    model_path.artifacts = tmp_path / "artifacts"
    model_path.artifacts.mkdir()

    with (
        patch.object(HydranetManager, "__init__", lambda self, *a, **kw: None),
        patch.object(HydranetManager, "configs", new_callable=PropertyMock) as configs,
        patch("views_hydranet.manager.hydranet_manager.DataFetcher") as fetcher,
        patch("views_hydranet.manager.hydranet_manager.FeatureScaler") as scaler,
    ):
        configs.return_value = dict(CONFIG)

        fetcher.return_value.fetch_df.return_value = frame.copy()
        fetcher.standardize_raw_df.return_value = frame.copy()
        scaler.return_value.fit_transform.return_value = frame.copy()

        mgr = HydranetManager.__new__(HydranetManager)
        mgr.device = torch.device("cpu")
        mgr._model_path = model_path
        mgr.run_timestamp = "test_ts"
        yield mgr


def test_the_forecast_path_survives_the_real_sniffer(manager):
    """RED GATE.

    Before the fix, `_run_data_pipeline(viz, forecast=True)` raised
    `ValueError: ... Forecast Continuity Broken! History ends at 108. Forecast starts at 101.0
    (Expected 109).` — the handler is the history volume, so of course it starts where history
    starts.

    The call below carries no `forecast=` argument because **the parameter no longer exists**:
    the data pipeline does the same work on every run type, so the forecasting entry point
    (`hydranet_manager.py`, the `if forecast:` branch of the evaluation context) now makes this
    exact call. Running it against the real `DataSniffer` is therefore running the forecasting
    path, and the defect cannot be reintroduced by passing the wrong flag — there is no flag.

    A forecasting run trains before it forecasts, so this used to cost a full training run before
    failing. That is why it went unobserved while eight models were run to completion.
    """
    handler, _, _ = manager._run_data_pipeline(MagicMock())

    assert handler.shape[0] == len(MONTHS), (
        "the handler the pipeline returns is the HISTORY volume — the forecast volume is "
        "built later, per origin, by InferenceOrchestrator via extrapolate_time()"
    )


def test_the_calibration_path_is_unchanged(manager):
    """The working path must stay working.

    Calibration already selected the history branch before the fix (`forecast` defaulted to
    False), so this test cannot distinguish the two versions of the code, and is not meant to.
    It is here so that a later change which *does* alter the calibration path cannot land
    unnoticed — the calibration frames already delivered to researchers came through here.
    """
    handler, _, _ = manager._run_data_pipeline(MagicMock())

    assert handler.shape[0] == len(MONTHS)


def test_the_history_branch_is_what_the_forecast_path_now_asks_for(manager):
    """Pins the fix itself, not merely its effect.

    The distinction matters: the pipeline could be made to pass by weakening the sniffer's
    threshold, by dropping the call, or by swallowing the error. Each would turn the red gate
    above green while leaving the data unvalidated. This asserts the specific correct thing —
    that the manager describes the handler it actually holds, on every run type.
    """
    with patch("views_hydranet.manager.hydranet_manager.DataSniffer") as sniffer_cls:
        manager._run_data_pipeline(MagicMock())

    sniffer_cls.return_value.sniff_forecast_alignment.assert_called_once()
    _, kwargs = sniffer_cls.return_value.sniff_forecast_alignment.call_args
    assert kwargs["is_forecast"] is False, (
        "the handler passed is VolumeHandler.from_df(df) — a history volume — so it must be "
        "checked against the history contract, whatever the run type is"
    )
