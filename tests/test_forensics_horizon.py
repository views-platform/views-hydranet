"""Horizon-split forensics: TrainingForensics tracks per-timestep (t0/early/late) stats so the
dossier can separate the calibrated first forecast step from the inflated later rollout steps
(hidden in the full-window average). 2026-06-28.
"""

import pytest

torch = pytest.importorskip("torch")
import numpy as np  # noqa: E402

from views_hydranet.utils.training_forensics import TrainingForensics  # noqa: E402

CFG = {
    "regression_targets": ["lr_sb_best"],
    "classification_targets": [],
    "regression_metrics": ["mse"],
    "classification_metrics": [],
}


def test_record_step_idx_populates_horizon_slices_lesson_aligned():
    tf = TrainingForensics(CFG)
    n_lessons = 3
    for _ in range(n_lessons):
        for win in range(2):
            for t in range(4):  # later steps predict bigger -> the split must capture inflation
                y = torch.full((1, 1, 4, 4), 0.1)
                yh = torch.full((1, 1, 4, 4), 0.5 + 0.5 * t)
                tf.record("REG:lr_sb_best", y, yh, step_idx=t)
        tf.finalize_lesson()
    d = tf.get_dossier("REG:lr_sb_best")

    # Horizon series exist and stay aligned with the lesson axis.
    for k in [
        "t0_y_hat_bar",
        "early_y_hat_bar",
        "late_y_hat_bar",
        "t0_bias",
        "late_bias",
        "t0_mse",
    ]:
        assert k in d, f"missing horizon series {k}"
        assert len(d[k]) == n_lessons, f"{k} not lesson-aligned"

    # The split must order by horizon: late step prediction > t0 prediction (the inflation).
    assert d["late_y_hat_bar"][-1] > d["t0_y_hat_bar"][-1] > 0
    # The full-window y_hat_bar sits between t0 and late (it averages all steps).
    assert d["t0_y_hat_bar"][-1] < d["y_hat_bar"][-1] < d["late_y_hat_bar"][-1]


def test_record_without_step_idx_leaves_horizon_series_aligned():
    """Back-compat: omitting step_idx still finalizes; horizon series stay aligned."""
    tf = TrainingForensics(CFG)
    for _ in range(2):
        tf.record("REG:lr_sb_best", torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))  # no step_idx
        tf.finalize_lesson()
    d = tf.get_dossier("REG:lr_sb_best")
    assert len(d["t0_y_hat_bar"]) == len(d["y_hat_bar"]) == 2
    assert not np.isnan(d["t0_y_hat_bar"]).any()
