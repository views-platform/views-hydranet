"""`aggregate_seeds.py` produced the headline n=4 claim and shipped with no tests at all.

The guards here are what stand between "four seeds agree" and "four files were read and one of
them silently wasn't". Every test below pins a path that would otherwise fail QUIETLY — a wrong
number, not an error — which is the only failure mode that matters for a results tool.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_D = Path(__file__).resolve().parents[1] / "reports" / "2026-08-21_persistence_reference_dossier"
sys.path.insert(0, str(_D / "tools"))

ag = pytest.importorskip("aggregate_seeds")

HS = (1, 18)


def _write(path: Path, label: str, arm: dict, pers: dict, n: int = 100, target: str = "sb"):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "model", "h", "N", "AP"])
        for h in HS:
            w.writerow([target, label, h, n, arm[h]])
        for h in HS:
            w.writerow([target, "persistence", h, n, pers[h]])


def _fair(path: Path, pers: dict):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["h", "AP_persistence_value_ranked"])
        for h in HS:
            w.writerow([h, pers[h]])


def _argv(res, fair, out):
    return ["aggregate_seeds", "--results", str(res), "--fair", str(fair), "--out", str(out)]


def _seed_dir(tmp_path, seeds, pers, n_by_seed=None, pers_by_seed=None):
    res = tmp_path / "results"
    res.mkdir(exist_ok=True)
    for label, arm in seeds.items():
        _write(
            res / f"score_persistence_ref_{label}.csv",
            label,
            arm,
            (pers_by_seed or {}).get(label, pers),
            n=(n_by_seed or {}).get(label, 100),
        )
    fair = tmp_path / "fair.csv"
    _fair(fair, pers)
    return res, fair


def test_happy_path_reports_worst_not_mean(tmp_path, monkeypatch, capsys):
    """The worst seed is the reported statistic — a mean can be carried by one lucky draw."""
    pers = {1: 0.10, 18: 0.05}
    seeds = {"a": {1: 0.40, 18: 0.20}, "b": {1: 0.20, 18: 0.10}}
    res, fair = _seed_dir(tmp_path, seeds, pers)
    out = tmp_path / "s.csv"
    monkeypatch.setattr(sys, "argv", _argv(res, fair, out))
    assert ag.main() == 0
    rows = {int(r["h"]): r for r in csv.DictReader(open(out))}
    assert float(rows[1]["arm_min"]) == pytest.approx(0.20)
    assert float(rows[1]["arm_mean"]) == pytest.approx(0.30)
    assert float(rows[1]["ratio_worst"]) == pytest.approx(2.0)
    assert rows[1]["worst_beats_persistence"] == "True"


def test_refuses_when_persistence_differs_between_seeds(tmp_path, monkeypatch):
    """THE guard M37 claims. Persistence is truth-only: one support -> one number. Different
    persistence means different supports, and the seeds are not comparable."""
    pers = {1: 0.10, 18: 0.05}
    seeds = {"a": {1: 0.40, 18: 0.20}, "b": {1: 0.30, 18: 0.15}}
    res, fair = _seed_dir(tmp_path, seeds, pers, pers_by_seed={"b": {1: 0.11, 18: 0.05}})
    monkeypatch.setattr(sys, "argv", _argv(res, fair, tmp_path / "s.csv"))
    with pytest.raises(SystemExit, match="do NOT share a support"):
        ag.main()


def test_equal_N_does_not_excuse_different_persistence(tmp_path, monkeypatch):
    """N is the weak check: two different origin windows can share a row count. This is why the
    persistence comparison exists — an earlier version compared only N."""
    pers = {1: 0.10, 18: 0.05}
    seeds = {"a": {1: 0.40, 18: 0.20}, "b": {1: 0.30, 18: 0.15}}
    res, fair = _seed_dir(
        tmp_path,
        seeds,
        pers,
        n_by_seed={"a": 100, "b": 100},
        pers_by_seed={"b": {1: 0.20, 18: 0.09}},
    )
    monkeypatch.setattr(sys, "argv", _argv(res, fair, tmp_path / "s.csv"))
    with pytest.raises(SystemExit, match="do NOT share a support"):
        ag.main()


def test_refuses_on_N_mismatch(tmp_path, monkeypatch):
    pers = {1: 0.10, 18: 0.05}
    seeds = {"a": {1: 0.40, 18: 0.20}, "b": {1: 0.30, 18: 0.15}}
    res, fair = _seed_dir(tmp_path, seeds, pers, n_by_seed={"b": 99})
    monkeypatch.setattr(sys, "argv", _argv(res, fair, tmp_path / "s.csv"))
    with pytest.raises(SystemExit, match="support mismatch"):
        ag.main()


def test_duplicate_label_across_files_is_refused_not_silently_merged(tmp_path, monkeypatch):
    """Two files sharing an internal label used to collapse into ONE seed, under-reporting
    n_seeds with no error. Live risk: the archived seed-43 file carries `l300_seed43` while the
    runner now writes `fullzero_*`, so a re-run creates exactly this collision."""
    pers = {1: 0.10, 18: 0.05}
    res = tmp_path / "results"
    res.mkdir()
    _write(res / "score_persistence_ref_x.csv", "same", {1: 0.4, 18: 0.2}, pers)
    _write(res / "score_persistence_ref_y.csv", "same", {1: 0.3, 18: 0.1}, pers)
    fair = tmp_path / "fair.csv"
    _fair(fair, pers)
    monkeypatch.setattr(sys, "argv", _argv(res, fair, tmp_path / "s.csv"))
    with pytest.raises(SystemExit, match="duplicate arm label"):
        ag.main()


def test_multiple_arms_in_one_file_is_refused(tmp_path, monkeypatch):
    """`score_v2_horizons` accepts several arms per call; folding them into one 'seed' would mix
    per-horizon values from different models under a single label."""
    pers = {1: 0.10, 18: 0.05}
    res = tmp_path / "results"
    res.mkdir()
    p = res / "score_persistence_ref_x.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "model", "h", "N", "AP"])
        for lab in ("armA", "armB"):
            for h in HS:
                w.writerow(["sb", lab, h, 100, 0.4])
        for h in HS:
            w.writerow(["sb", "persistence", h, 100, pers[h]])
    _write(res / "score_persistence_ref_y.csv", "armC", {1: 0.3, 18: 0.1}, pers)
    fair = tmp_path / "fair.csv"
    _fair(fair, pers)
    monkeypatch.setattr(sys, "argv", _argv(res, fair, tmp_path / "s.csv"))
    with pytest.raises(SystemExit, match="arms in one file"):
        ag.main()


def test_refuses_fewer_than_two_seeds(tmp_path, monkeypatch):
    pers = {1: 0.10, 18: 0.05}
    res, fair = _seed_dir(tmp_path, {"a": {1: 0.4, 18: 0.2}}, pers)
    monkeypatch.setattr(sys, "argv", _argv(res, fair, tmp_path / "s.csv"))
    with pytest.raises(SystemExit, match="need >= 2 seeds"):
        ag.main()


def test_file_with_no_rows_for_target_is_refused_not_skipped(tmp_path, monkeypatch):
    """`if not ap: continue` silently dropped a malformed file and under-counted the seeds."""
    pers = {1: 0.10, 18: 0.05}
    res, fair = _seed_dir(tmp_path, {"a": {1: 0.4, 18: 0.2}, "b": {1: 0.3, 18: 0.1}}, pers)
    _write(res / "score_persistence_ref_z.csv", "z", {1: 0.3, 18: 0.1}, pers, target="ns")
    monkeypatch.setattr(sys, "argv", _argv(res, fair, tmp_path / "s.csv"))
    with pytest.raises(SystemExit, match="no rows for target"):
        ag.main()


def test_zero_persistence_gives_no_ratio(tmp_path, monkeypatch):
    """A ratio against a zero reference is not a number; refuse rather than emit inf."""
    pers = {1: 0.0, 18: 0.05}
    seeds = {"a": {1: 0.4, 18: 0.2}, "b": {1: 0.3, 18: 0.1}}
    res, fair = _seed_dir(tmp_path, seeds, pers)
    monkeypatch.setattr(sys, "argv", _argv(res, fair, tmp_path / "s.csv"))
    with pytest.raises(SystemExit, match="no ratio is defined"):
        ag.main()
