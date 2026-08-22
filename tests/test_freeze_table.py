"""`freeze_table.py` replaces a finisher that shipped WRONG, so it gets tests the first one lacked.

The original built its table from the CSV's `model` column — `none`/`hidden`/`cell`/`all`, which
does not identify the seed. Both seeds collapsed onto four rows (last-write-wins) and the baseline
lookup matched nothing, so the comparison section rendered empty. Same class as the `aggregate_seeds`
label collision: **keying on a label that does not identify the run when the filename does.**
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

_D = Path(__file__).resolve().parents[1] / "reports" / "2026-08-22_state_freeze_l300_dossier"
sys.path.insert(0, str(_D / "tools"))

ft = pytest.importorskip("freeze_table")


def _write(d: Path, seed: str, arm: str, ap: dict, target: str = "sb"):
    with open(d / f"score_{seed}_{arm}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["target", "model", "h", "AP"])
        for h, v in ap.items():
            w.writerow([target, arm, h, v])  # note: `model` is the ARM, with no seed in it


def _two_seeds(tmp_path, s43_h18=0.3318, s42_h18=0.3298):
    d = tmp_path / "res"
    d.mkdir()
    for seed, base in (("fullzero_fortythree", s43_h18), ("fullzero_fortytwo", s42_h18)):
        for arm, bump in (("none", 0.0), ("hidden", -0.005), ("cell", 0.035), ("all", 0.037)):
            _write(d, seed, arm, {1: 0.4774, 6: 0.40, 18: base + bump, 36: 0.22})
    return d


def test_seeds_are_not_collapsed(tmp_path):
    """THE regression. Four arms x two seeds must yield EIGHT distinct readings, not four."""
    data = ft.read_results(str(_two_seeds(tmp_path)))
    assert set(data) == {"fullzero_fortythree", "fullzero_fortytwo"}
    assert all(set(v) == {"none", "hidden", "cell", "all"} for v in data.values())
    assert data["fullzero_fortythree"]["none"][18] != data["fullzero_fortytwo"]["none"][18]


def test_baseline_comparison_actually_renders(tmp_path):
    """The original's lookup keys never matched, so this section silently came out empty."""
    out = ft.render(ft.read_results(str(_two_seeds(tmp_path))))
    assert "Falsifiers pass" in out
    assert "+0.0350" in out and "-0.0050" in out
    # scoped to the per-seed sections: the "Mean over seeds" table also carries a `none` row, so a
    # bare count over the whole document is 3, not 2. The original assertion said 2 and was wrong.
    per_seed = out.split("## Mean over seeds")[0]
    assert per_seed.count("| `none` |") == 2, "one control row per seed"
    assert per_seed.count("## `fullzero_") == 2


def test_control_that_does_not_reproduce_the_published_value_fires(tmp_path):
    """If `none` drifts from M34's published free-running number, the vehicle is not what we think
    and the whole table is unreadable — say so ABOVE the numbers, not in a footnote."""
    out = ft.render(ft.read_results(str(_two_seeds(tmp_path, s43_h18=0.2000))))
    assert "FALSIFIER FAILED" in out
    assert "vehicle mismatch" in out


def test_h1_differing_across_arms_fires(tmp_path):
    """There is no feedback at step 1, so freezing cannot move h1. If it does, the arm is not what
    it claims to be."""
    d = tmp_path / "res"
    d.mkdir()
    for seed in ("fullzero_fortythree", "fullzero_fortytwo"):
        _write(d, seed, "none", {1: 0.4774, 18: 0.3318})
        _write(d, seed, "cell", {1: 0.5000, 18: 0.3700})  # h1 moved — impossible
    out = ft.render(ft.read_results(str(d)))
    assert "FALSIFIER FAILED" in out
    assert "cannot affect h1" in out


def test_unparseable_filenames_are_ignored_not_guessed(tmp_path):
    d = tmp_path / "res"
    d.mkdir()
    _write(d, "fullzero_fortythree", "none", {1: 0.4774, 18: 0.3318})
    _write(d, "fullzero_fortytwo", "none", {1: 0.4779, 18: 0.3298})
    (d / "score_something_else.csv").write_text("target,model,h,AP\nsb,x,1,0.5\n")
    data = ft.read_results(str(d))
    assert set(data) == {"fullzero_fortythree", "fullzero_fortytwo"}


def test_empty_results_dir_raises(tmp_path):
    d = tmp_path / "res"
    d.mkdir()
    with pytest.raises(SystemExit, match="no score_"):
        sys.argv = ["freeze_table", "--results", str(d), "--out", str(tmp_path / "o.md")]
        ft.main()
