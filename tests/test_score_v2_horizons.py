"""TDD for the v2 horizon scorer (score_v2_horizons.py in the 2026-07-29 v2_scoreboard dossier).

The v2 scoreboard adds two Magnitude-Calibration-Ratio columns to the frozen rollout ruler's metric
set — `mcr_all` (mean(pred)/mean(true) over ALL cells) and `mcr_none` (leaked mass = mean(pred) on
true-zero cells) beside the existing `pos_mcr` (= mcr_events). These tests pin the two new columns
to hand-computed values on a tiny fixture, and pin h=1 to the FROZEN lodestar T=0 scorer (the
faithfulness anchor: the horizon ruler's h=1 must byte-reproduce the frozen T=0 number).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# C-247/F-Z2: repo-relative (tests/ is one level below the repo root) — NEVER a hardcoded machine
# path. `reports/` is gitignored, but both tools below are **force-tracked** (`git add -f`), so a
# normal clone has them and this module runs in CI. `score_v2_horizons.py` was NOT tracked until
# 2026-08-15, which silently skipped this module everywhere and left `rescore_v2.py` — a tracked
# driver that loads it at runtime — unable to run in a clean clone at all.
_HN = Path(__file__).resolve().parents[1]
_LODE_DIR = _HN / "reports/2026-07-17_lodestar_eval_dossier/tools"
_V2_TOOL = _HN / "reports/2026-07-29_v2_scoreboard_dossier/tools/score_v2_horizons.py"

if not _V2_TOOL.exists() or not (_LODE_DIR / "lodestar_score.py").exists():
    pytest.skip(
        "v2 scoreboard / lodestar tools are absent — both are force-tracked, so this means a "
        "sparse or partial checkout, not a normal clone (C-247).",
        allow_module_level=True,
    )


def _load_v2_tool():
    """Import the versioned v2 scorer by path (dossier tool, not an installed package)."""
    spec = importlib.util.spec_from_file_location("score_v2_horizons", _V2_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_origin(pred_dir: Path, origin: int, time, unit, lr_samples, by_samples):
    """Write one origin dir with lr_sb_best + by_sb_best cubes (count samples + gate draws)."""
    od = pred_dir / f"origin_{origin}"
    for name, samp in (("lr_sb_best", lr_samples), ("by_sb_best", by_samples)):
        d = od / name
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "y_pred.npy", np.asarray(samp, dtype=float))
        np.savez(
            d / "identifiers.npz",
            time=np.asarray(time, dtype=int),
            unit=np.asarray(unit, dtype=int),
        )


@pytest.fixture
def fixture(tmp_path):
    """One origin (m0=457), 2 cells (7=event, 8=zero), 2 horizons (h=1 month457, h=2 month458).

    Rows order per identifiers: (457,7),(457,8),(458,7),(458,8).
    h=1 (month 457): truth [3.0, 0.0]; preds cell7 [2,4] (ey=3), cell8 [0,1] (ey=0.5).
      => mcr_all  = mean([3,0.5]) / mean([3,0]) = 1.75/1.5 = 7/6
         mcr_none = ey[zero].mean() = 0.5
         pos_mcr  = mean(ratio_events) = (3/3) = 1.0
         size_ratio = median(ratio_events) = 1.0
         crps_all = mean(0.5, 0.25) = 0.375   (hand-verified energy CRPS)
         AP = 1.0, Brier = 0.0  (gate 1.0 on event, 0.0 on zero)
    """
    pred_dir = tmp_path / "pred"
    time = [457, 457, 458, 458]
    unit = [7, 8, 7, 8]
    lr = [[2.0, 4.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    by = [[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]
    _write_origin(pred_dir, 457, time, unit, lr, by)

    truth = pd.DataFrame(
        {
            "month_id": [457, 457, 458, 458],
            "priogrid_id": [7, 8, 7, 8],
            "lr_sb_best": [3.0, 0.0, 1.0, 0.0],
        }
    )
    truth_path = tmp_path / "truth.parquet"
    truth.to_parquet(truth_path)
    return pred_dir, truth_path


def test_mcr_columns_hand_computed(fixture):
    pred_dir, truth_path = fixture
    mod = _load_v2_tool()
    registry = [("m", str(pred_dir), "lr_{t}_best", "by_{t}_best")]
    rows = mod.score_horizons_v2(
        registry, targets=["sb"], horizons=(1, 2), truth_parquet=str(truth_path)
    )
    h1 = next(r for r in rows if r["model"] == "m" and r["h"] == 1 and r["target"] == "sb")
    assert h1["mcr_all"] == pytest.approx(7.0 / 6.0), h1
    assert h1["mcr_none"] == pytest.approx(0.5), h1
    assert h1["pos_mcr"] == pytest.approx(1.0), h1  # = mcr_events
    assert h1["size_ratio"] == pytest.approx(1.0), h1
    assert h1["crps_all"] == pytest.approx(0.375), h1
    assert h1["AP"] == pytest.approx(1.0), h1
    assert h1["Brier"] == pytest.approx(0.0), h1


def test_h1_matches_frozen_lodestar(fixture):
    """The horizon ruler's h=1 must reproduce the FROZEN lodestar T=0 crps_all exactly (anchor)."""
    pred_dir, truth_path = fixture
    sys.path.insert(0, str(_LODE_DIR))
    import lodestar_score  # noqa: E402

    lode_rows = lodestar_score.score_models(
        str(truth_path),
        [("m", str(pred_dir), "lr_{t}_best", "by_{t}_best")],
        ["sb"],
    )
    lode = lode_rows[0]

    mod = _load_v2_tool()
    rows = mod.score_horizons_v2(
        [("m", str(pred_dir), "lr_{t}_best", "by_{t}_best")],
        targets=["sb"],
        horizons=(1, 2),
        truth_parquet=str(truth_path),
    )
    h1 = next(r for r in rows if r["h"] == 1 and r["target"] == "sb")
    assert h1["crps_all"] == pytest.approx(lode["crps_all"])
    assert h1["crps_events"] == pytest.approx(lode["crps_events"])
    assert h1["crps_none"] == pytest.approx(lode["crps_none"])
    assert h1["size_ratio"] == pytest.approx(lode["size_ratio"])
    assert h1["pos_mcr"] == pytest.approx(lode["pos_mcr"])


# --- GH #282: the persistence baseline's history month was never loaded -------------------------


def _rollout_skill_mod():
    """Import the sibling that owns `_persistence_gathered` (dossier tool, force-tracked)."""
    p = _HN / "reports/2026-07-25_t0_rollout_skill_dossier/tools/rollout_skill_score.py"
    if not p.exists():
        pytest.skip("rollout_skill_score.py absent — sparse checkout (C-247)")
    spec = importlib.util.spec_from_file_location("rollout_skill_score", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rollout_skill_score"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_persistence_refuses_when_the_history_month_was_never_loaded():
    """GH #282, the guard.

    `score_horizons_v2` built `tmap` from `{m0+h-1}` only. `_persistence_gathered` reads
    `truth_map[(m0-1, u)]`, which is present ONLY because origins are usually consecutive — origin
    k's history is origin k-1's h=1 forecast month. The FIRST origin has no predecessor, so its
    history fell through `.get(..., 0.0)` and its whole persistence forecast became zeros, with no
    error. Understated the baseline by 4-10% at 13 origins, and by everything at one.
    """
    rs = _rollout_skill_mod()
    support = [(100, 1), (101, 1)]
    tmap = {(100, 1): 5.0, (101, 1): 6.0}
    with pytest.raises(ValueError, match="never loaded"):
        rs._persistence_gathered(tmap, support, (1,), months_loaded={100, 101})


def test_persistence_accepts_a_truth_map_that_includes_the_history_months():
    rs = _rollout_skill_mod()
    support = [(100, 1), (101, 1)]
    tmap = {(99, 1): 5.0, (100, 1): 7.0, (101, 1): 6.0}
    got = rs._persistence_gathered(tmap, support, (1,), months_loaded={99, 100, 101})
    assert float(got[(100, 1, 1)][0][0]) == 5.0  # truth[m0-1], not truth[m0]
    assert float(got[(101, 1, 1)][0][0]) == 7.0


def test_both_scorers_load_the_pre_origin_month():
    """The fix must live in BOTH scorers, or one of them silently keeps the defect.

    Asserted on source text because the alternative is building two full cube fixtures for a
    one-line set-union; the runtime guard above is what actually protects the number.
    """
    for tool in (
        _V2_TOOL,
        _HN / "reports/2026-07-25_t0_rollout_skill_dossier/tools/rollout_skill_score.py",
    ):
        if not tool.exists():
            pytest.skip(f"{tool.name} absent — sparse checkout (C-247)")
        src = tool.read_text()
        assert "months |= {m0 - 1 for (m0, _u) in support}" in src, (
            f"{tool.name} does not load the persistence history month — GH #282 regressed"
        )
