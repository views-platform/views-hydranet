"""Tests for the sharpness scorecard — the metric must rank a SHARP field above a SMOOTH one."""

import importlib.util
import os

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "sharpness_scorecard",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "sharpness_scorecard.py"),
)
ssc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ssc)


def test_to_grid_places_values():
    umap = {10: (0, 0), 20: (1, 2)}
    g = ssc.to_grid(np.array([5.0, 7.0]), np.array([10, 20]), umap)
    assert g[0, 0] == 5.0 and g[1, 2] == 7.0 and g[3, 3] == 0.0


def test_matched_pred_thresh_gives_truth_event_count():
    truth = np.zeros((10, 10))
    truth[0, 0] = truth[1, 1] = truth[2, 2] = 1.0  # 3 events
    pred = np.random.RandomState(0).rand(10, 10)
    th = ssc.matched_pred_thresh(pred, truth)
    assert int((pred > th).sum()) == 3


def test_fss_perfect_overlap_is_one():
    o = np.zeros((20, 20))
    o[5, 5] = o[10, 10] = 1.0
    f = ssc.fss(o.copy(), o.copy(), [1, 3], pred_thresh=0.5)
    assert abs(f[1] - 1.0) < 1e-9 and abs(f[3] - 1.0) < 1e-9


def test_fss_disjoint_is_low_at_fine_scale():
    o = np.zeros((20, 20))
    o[2, 2] = 1.0
    p = np.zeros((20, 20))
    p[18, 18] = 1.0
    f = ssc.fss(p, o, [1], pred_thresh=0.5)
    assert f[1] < 0.1  # far apart -> ~no skill at n=1


def test_sharp_field_more_concentrated_than_smooth():
    from scipy.ndimage import gaussian_filter

    sharp = np.zeros((20, 20))
    sharp[10, 10] = 10.0
    sharp[5, 5] = 10.0
    truth = sharp.copy()
    smooth = gaussian_filter(sharp, sigma=2.0)
    _, _, conc_sharp = ssc._score_field(sharp, truth, [1])
    _, _, conc_smooth = ssc._score_field(smooth, truth, [1])
    assert conc_sharp > conc_smooth  # mass-in-top-1% is higher for the sharp field


def test_smooth_field_has_higher_area_ratio():
    from scipy.ndimage import gaussian_filter

    truth = np.zeros((20, 20))
    truth[10, 10] = truth[5, 5] = 3.0
    sharp = truth.copy()
    smooth = gaussian_filter(truth, sigma=2.0) * 20.0  # spread + lifted above 0.5
    _, area_sharp, _ = ssc._score_field(sharp, truth, [1])
    _, area_smooth, _ = ssc._score_field(smooth, truth, [1])
    assert area_smooth > area_sharp  # smooth smears predicted area beyond the sparse truth


def test_fss_localization_high_when_colocated():
    truth = np.zeros((30, 30))
    truth[10, 10] = truth[20, 20] = 1.0
    f_sharp, _, _ = ssc._score_field(truth.copy(), truth, [1, 5])
    assert f_sharp[1] > 0.5  # matched threshold + co-located peaks -> good localization


def test_build_unit_grid_requires_columns(tmp_path):
    import pandas as pd

    p = tmp_path / "bad.parquet"
    pd.DataFrame({"month_id": [1], "priogrid_gid": [5]}).to_parquet(p)
    with pytest.raises(ValueError, match="missing"):
        ssc.build_unit_grid(str(p))


def test_build_unit_grid_accepts_priogrid_id_alias(tmp_path):
    """GH#144 grid rename: the viewser-native `priogrid_id` column must work as well as the legacy
    `priogrid_gid` (C-167 — the instrument was stale on the current family parquet)."""
    import pandas as pd

    p = tmp_path / "new.parquet"
    pd.DataFrame(
        {"month_id": [1, 1], "priogrid_id": [5, 6], "row": [87, 88], "col": [310, 311]}
    ).to_parquet(p)
    umap = ssc.build_unit_grid(str(p))
    assert umap == {5: (0, 0), 6: (1, 1)}  # row-87, col-310


def test_build_unit_grid_rejects_missing_grid_col(tmp_path):
    """Neither priogrid_gid nor priogrid_id -> fail loud."""
    import pandas as pd

    p = tmp_path / "nogrid.parquet"
    pd.DataFrame({"month_id": [1], "row": [87], "col": [310]}).to_parquet(p)
    with pytest.raises(ValueError, match="priogrid"):
        ssc.build_unit_grid(str(p))
