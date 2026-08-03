"""Unit tests for the Tier-A parity harness pure functions (Epic #203 S2). Network-free."""
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "tier_a_parity",
    str(pathlib.Path(__file__).with_name("tier_a_parity.py")),
)
tap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tap)


def _frame(months, cells, gid_name, cols, fill):
    """Build a dense (month, gid) frame. `fill` maps colname -> array over the (m,c) grid."""
    idx = pd.MultiIndex.from_product([months, cells], names=["month_id", gid_name])
    return pd.DataFrame({c: fill[c] for c in cols}, index=idx)


def _pair(vv_vals, df_vals, months=range(121, 505), cells=(1, 2, 3)):
    n = len(list(months)) * len(cells)
    vcols = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
    dcols = ["ged_sb_best", "ged_ns_best", "ged_os_best"]
    vv = _frame(list(months), list(cells), "priogrid_id", vcols,
                {c: np.full(n, vv_vals[i], float) for i, c in enumerate(vcols)})
    df = _frame(list(months), list(cells), "priogrid_gid", dcols,
                {c: np.full(n, df_vals[i], float) for i, c in enumerate(dcols)})
    return vv, df


def test_identical_frames_pass():
    vv, df = _pair((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    vv.iloc[0, 0] = 5.0  # one matching positive so maxima > 0
    df.iloc[0, 0] = 5.0
    sc = tap.parity_scorecard(vv, df)
    assert sc["cell_set_identical"] and sc["index_identical"]
    assert sc["datafactory_last_month"] == 504
    for m in sc["targets"].values():
        assert m["exact_pct"] == 100.0 and m["maxima_identical"]
    v = tap.evaluate_falsifiers(sc)
    assert v["passed"] and not any(v["fired"].values())


def test_priogrid_gid_is_normalized():
    """datafactory's priogrid_gid level must align to viewser's priogrid_id."""
    vv, df = _pair((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    sc = tap.parity_scorecard(vv, df)
    assert sc["index_identical"] and sc["cell_set_identical"]


def test_cell_set_difference_fires_fa1():
    vv, df = _pair((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), cells=(1, 2, 3))
    _, df2 = _pair((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), cells=(1, 2, 9))  # cell 9 not in viewser
    sc = tap.parity_scorecard(vv, df2)
    v = tap.evaluate_falsifiers(sc)
    assert not sc["cell_set_identical"]
    assert v["fired"]["F-A1_cell_set"] and not v["passed"]


def test_short_coverage_fires_fa3():
    vv, df = _pair((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), months=range(121, 493))  # ends 492
    sc = tap.parity_scorecard(vv, df)
    v = tap.evaluate_falsifiers(sc)
    assert sc["datafactory_last_month"] == 492
    assert v["fired"]["F-A3_coverage"] and not v["passed"]


def test_maxima_mismatch_fires_fa4():
    vv, df = _pair((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    vv.iloc[0, 0] = 100.0  # a big event present only in viewser
    sc = tap.parity_scorecard(vv, df)
    v = tap.evaluate_falsifiers(sc)
    assert not sc["targets"]["lr_sb_best"]["maxima_identical"]
    assert v["fired"]["F-A4_maxima"]


def test_large_drift_fires_fa2():
    # every cell differs a lot -> exact% collapses and drift explodes
    vv, df = _pair((10.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    sc = tap.parity_scorecard(vv, df)
    v = tap.evaluate_falsifiers(sc)
    assert sc["targets"]["lr_sb_best"]["exact_pct"] < tap.EXACT_MIN
    assert v["fired"]["F-A2_unexplained_residual"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
