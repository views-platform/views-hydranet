"""
Unit tests for VolumeHandler.to_evaluation_pf() and to_forecast_pf().

TDD — tests are written BEFORE the implementation.

Verifies:
1. Correct return type: dict[str, PredictionFrame]
2. All targets present as dict keys
3. Correct y_pred shapes (stochastic vs point mode)
4. Identifiers populated (time, unit arrays)
5. Parity with to_evaluation_df() / to_forecast_df():
   - Same (time, unit) pairs per observation
   - Same prediction values (point mode: exact; stochastic: per-sample)

The _df methods are not changed; parity tests guard against any divergence.
"""

import numpy as np
import pandas as pd
import pytest
from views_pipeline_core.data.prediction_frame import PredictionFrame

from views_hydranet.utils.volume_handler import VolumeHandler

# ─── Constants ───────────────────────────────────────────────────────────────

TARGETS = [
    "lr_sb_best", "lr_ns_best", "lr_os_best",
    "by_sb_best", "by_ns_best", "by_os_best",
]

# Minimal VolumeHandler config — 2×2 grid, no c_id to keep the fixture simple
CONFIG = {
    "height": 2,
    "width": 2,
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "row_offset": 0,
    "col_offset": 0,
    "identity_cols": ["row", "col"],
    "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
}

N_CELLS = 4   # 2×2, all cells valid (priogrid_gid > 0)
N_MONTHS = 5  # history length; prediction uses last month
S = 4         # posterior samples (stochastic tests)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_df(n_months: int = N_MONTHS) -> pd.DataFrame:
    """2×2 grid spanning month_id 100 .. 100+n_months-1."""
    rows = []
    for t in range(100, 100 + n_months):
        for r in range(2):
            for c in range(2):
                rows.append({
                    "month_id": t,
                    "priogrid_gid": r * 2 + c + 1,
                    "row": float(r),
                    "col": float(c),
                    "lr_sb_best": float(r + c + t * 0.01),
                    "lr_ns_best": 0.5,
                    "lr_os_best": 0.0,
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def history_handler():
    return VolumeHandler.from_df(_make_df(), CONFIG)


@pytest.fixture(scope="module")
def window_handler(history_handler):
    """Single-month slice used as prediction window."""
    return history_handler.slice_time(N_MONTHS - 1, N_MONTHS)


@pytest.fixture(scope="module")
def stochastic_pred_handler(window_handler):
    """5D pred_handler [T=1, H=2, W=2, C=6, S=4]."""
    np.random.seed(42)
    posterior = np.random.rand(1, 2, 2, 6, S)
    return window_handler.wrap_predictions(posterior, target_names=TARGETS)


@pytest.fixture(scope="module")
def point_pred_handler(window_handler):
    """4D pred_handler after collapse_to_point (arithmetic_mean)."""
    np.random.seed(42)
    posterior = np.random.rand(1, 2, 2, 6, S)
    stoch = window_handler.wrap_predictions(posterior, target_names=TARGETS)
    return stoch.collapse_to_point("arithmetic_mean")


# ─── Tests: to_evaluation_pf ─────────────────────────────────────────────────

class TestToEvaluationPF:
    """
    Tests for VolumeHandler.to_evaluation_pf().
    Mirrors the guarantees of to_evaluation_df() but returns PredictionFrames.
    """

    def test_returns_dict(self, stochastic_pred_handler, window_handler):
        result = stochastic_pred_handler.to_evaluation_pf(
            history=window_handler, start_idx=0, all_targets=TARGETS
        )
        assert isinstance(result, dict)

    def test_all_targets_present(self, stochastic_pred_handler, window_handler):
        result = stochastic_pred_handler.to_evaluation_pf(
            history=window_handler, start_idx=0, all_targets=TARGETS
        )
        for target in TARGETS:
            assert target in result, f"Missing target key: {target}"
            assert isinstance(result[target], PredictionFrame)

    def test_stochastic_shape(self, stochastic_pred_handler, window_handler):
        """Stochastic mode: y_pred.shape == (N_CELLS, S)."""
        result = stochastic_pred_handler.to_evaluation_pf(
            history=window_handler, start_idx=0, all_targets=TARGETS
        )
        for target in TARGETS:
            pf = result[target]
            assert pf.y_pred.ndim == 2, f"{target}: expected 2D, got ndim={pf.y_pred.ndim}"
            assert pf.y_pred.shape == (N_CELLS, S), (
                f"{target}: expected ({N_CELLS}, {S}), got {pf.y_pred.shape}"
            )

    def test_point_shape(self, point_pred_handler, window_handler):
        """Point mode: y_pred.shape == (N_CELLS, 1)."""
        result = point_pred_handler.to_evaluation_pf(
            history=window_handler, start_idx=0, all_targets=TARGETS
        )
        for target in TARGETS:
            pf = result[target]
            assert pf.y_pred.ndim == 2, f"{target}: expected 2D, got ndim={pf.y_pred.ndim}"
            assert pf.y_pred.shape == (N_CELLS, 1), (
                f"{target}: expected ({N_CELLS}, 1), got {pf.y_pred.shape}"
            )

    def test_identifiers_populated(self, stochastic_pred_handler, window_handler):
        """identifiers['time'] and identifiers['unit'] are non-empty and correct length."""
        result = stochastic_pred_handler.to_evaluation_pf(
            history=window_handler, start_idx=0, all_targets=TARGETS
        )
        pf = result["lr_sb_best"]
        assert "time" in pf.identifiers
        assert "unit" in pf.identifiers
        assert len(pf.identifiers["time"]) == N_CELLS
        assert len(pf.identifiers["unit"]) == N_CELLS
        # No NaN values
        assert not np.any(np.isnan(pf.identifiers["time"].astype(float)))
        assert not np.any(np.isnan(pf.identifiers["unit"].astype(float)))

    def test_parity_identifiers_with_df(self, stochastic_pred_handler, window_handler):
        """
        Parity: (time, unit) pairs must be identical to those in to_evaluation_df().
        The same valid-cell mask is applied in both paths.
        """
        df = stochastic_pred_handler.to_evaluation_df(
            history=window_handler, start_idx=0
        )
        pf_dict = stochastic_pred_handler.to_evaluation_pf(
            history=window_handler, start_idx=0, all_targets=TARGETS
        )
        pf = pf_dict["lr_sb_best"]

        df_time = df.index.get_level_values("month_id").values
        df_unit = df.index.get_level_values("priogrid_gid").values
        df_pairs = set(zip(df_time.tolist(), df_unit.tolist()))
        pf_pairs = set(zip(pf.identifiers["time"].tolist(), pf.identifiers["unit"].tolist()))

        assert df_pairs == pf_pairs, (
            f"Identifier set mismatch:\n"
            f"  DF:  {sorted(df_pairs)}\n"
            f"  PF:  {sorted(pf_pairs)}"
        )

    def test_parity_values_point_mode(self, point_pred_handler, window_handler):
        """
        Parity: in point mode, prediction values must match to_evaluation_df() exactly.
        Both paths extract from the same numpy volume; the only difference is the
        Polars join (df path) vs direct indexing (pf path).
        """
        df = point_pred_handler.to_evaluation_df(history=window_handler, start_idx=0)
        pf_dict = point_pred_handler.to_evaluation_pf(
            history=window_handler, start_idx=0, all_targets=TARGETS
        )

        # Sort both by (time, unit) for reproducible comparison
        df_time = df.index.get_level_values("month_id").values
        df_unit = df.index.get_level_values("priogrid_gid").values
        df_order = np.lexsort((df_unit, df_time))

        pf = pf_dict["lr_sb_best"]
        pf_order = np.lexsort((pf.identifiers["unit"], pf.identifiers["time"]))

        df_vals = df["pred_lr_sb_best"].values.astype(float)[df_order]
        pf_vals = pf.y_pred[:, 0][pf_order]

        np.testing.assert_allclose(
            pf_vals, df_vals, rtol=1e-5,
            err_msg="Point-mode prediction values diverge from to_evaluation_df()"
        )

    def test_parity_values_stochastic_mode(self, stochastic_pred_handler, window_handler):
        """
        Parity: in stochastic mode, each row's sample vector must match the list
        stored in the corresponding to_evaluation_df() cell.
        """
        df = stochastic_pred_handler.to_evaluation_df(history=window_handler, start_idx=0)
        pf_dict = stochastic_pred_handler.to_evaluation_pf(
            history=window_handler, start_idx=0, all_targets=TARGETS
        )

        df_time = df.index.get_level_values("month_id").values
        df_unit = df.index.get_level_values("priogrid_gid").values
        df_order = np.lexsort((df_unit, df_time))

        pf = pf_dict["lr_sb_best"]
        pf_order = np.lexsort((pf.identifiers["unit"], pf.identifiers["time"]))

        df_vals = np.array(
            [np.array(v) for v in df["pred_lr_sb_best"].values[df_order]]
        )  # (N, S)
        pf_vals = pf.y_pred[pf_order]  # (N, S)

        np.testing.assert_allclose(
            pf_vals, df_vals, rtol=1e-5,
            err_msg="Stochastic prediction values diverge from to_evaluation_df()"
        )

    def test_bounds_check_raises_on_bad_start_idx(self, stochastic_pred_handler, window_handler):
        """Inherits the same bounds validation as to_evaluation_df()."""
        with pytest.raises(ValueError, match="Contract Violation"):
            stochastic_pred_handler.to_evaluation_pf(
                history=window_handler, start_idx=999, all_targets=TARGETS
            )


# ─── Tests: to_forecast_pf ───────────────────────────────────────────────────

class TestToForecastPF:
    """
    Tests for VolumeHandler.to_forecast_pf().
    Mirrors the guarantees of to_forecast_df() but returns PredictionFrames.
    """

    def test_returns_dict(self, stochastic_pred_handler, history_handler):
        result = stochastic_pred_handler.to_forecast_pf(
            history=history_handler, all_targets=TARGETS
        )
        assert isinstance(result, dict)

    def test_all_targets_present(self, stochastic_pred_handler, history_handler):
        result = stochastic_pred_handler.to_forecast_pf(
            history=history_handler, all_targets=TARGETS
        )
        for target in TARGETS:
            assert target in result
            assert isinstance(result[target], PredictionFrame)

    def test_stochastic_shape(self, stochastic_pred_handler, history_handler):
        result = stochastic_pred_handler.to_forecast_pf(
            history=history_handler, all_targets=TARGETS
        )
        for target in TARGETS:
            pf = result[target]
            assert pf.y_pred.ndim == 2
            assert pf.y_pred.shape == (N_CELLS, S)

    def test_point_shape(self, point_pred_handler, history_handler):
        result = point_pred_handler.to_forecast_pf(
            history=history_handler, all_targets=TARGETS
        )
        for target in TARGETS:
            pf = result[target]
            assert pf.y_pred.ndim == 2
            assert pf.y_pred.shape == (N_CELLS, 1)

    def test_parity_identifiers_with_df(self, stochastic_pred_handler, history_handler):
        """
        Parity: (time, unit) pairs must be identical to those in to_forecast_df().
        """
        df = stochastic_pred_handler.to_forecast_df(history=history_handler)
        pf_dict = stochastic_pred_handler.to_forecast_pf(
            history=history_handler, all_targets=TARGETS
        )
        pf = pf_dict["lr_sb_best"]

        df_time = df.index.get_level_values("month_id").values
        df_unit = df.index.get_level_values("priogrid_gid").values
        df_pairs = set(zip(df_time.tolist(), df_unit.tolist()))
        pf_pairs = set(zip(pf.identifiers["time"].tolist(), pf.identifiers["unit"].tolist()))

        assert df_pairs == pf_pairs, (
            f"Identifier set mismatch:\n"
            f"  DF: {sorted(df_pairs)}\n"
            f"  PF: {sorted(pf_pairs)}"
        )


# ─── Tests: wrap_predictions dtype contract ───────────────────────────────────

class TestWrapPredictionsDtype:
    """
    Guard against the silent float64 upcast in wrap_predictions().

    The history VolumeHandler is float64 (from_df uses np.zeros dtype=float64).
    Posterior data from PyTorch is float32.  np.concatenate promotes everything
    to float64 if not cast first — doubling the pred_handler size for no benefit.

    Identity values (priogrid_gid, month_id, row, col) are small integers well
    within float32's exact range (integers ≤ 2^24 ≈ 16.7 million).
    """

    def test_stochastic_pred_handler_is_float32(self, stochastic_pred_handler):
        """
        wrap_predictions() with 5D float32 posterior must produce float32 pred_handler.
        Identity watermarks sliced from float64 history must be cast to float32
        before np.concatenate to prevent the silent upcast.
        """
        assert stochastic_pred_handler._data.dtype == np.float32, (
            f"wrap_predictions() stochastic branch produced "
            f"{stochastic_pred_handler._data.dtype}, expected float32. "
            "The float64 identity watermarks must be cast to float32 before "
            "np.concatenate to prevent 2× peak RAM from NumPy's silent upcast."
        )

    def test_point_pred_handler_preserves_source_dtype(self, window_handler):
        """
        wrap_predictions() with 4D float64 posterior preserves float64.
        The point (DF) path is unchanged — float64 is appropriate there.
        """
        posterior_4d = np.ones((1, 2, 2, 6), dtype=np.float64)
        pred = window_handler.wrap_predictions(posterior_4d, target_names=TARGETS)
        assert pred._data.dtype == np.float64, (
            "wrap_predictions() point branch must preserve the source dtype (float64). "
            "Only the stochastic branch casts to float32."
        )
