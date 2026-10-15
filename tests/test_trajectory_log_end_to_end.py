"""The trajectory CSV, driven END TO END through `training_loop`.

⛔ WHY THIS FILE EXISTS. An independent guard audit of PR #398 found the per-task CSV columns
**DECORATIVE**: eight one-line production changes left the whole suite green, including one that
reinstated the exact defect the feature was written to prevent (`train()` returning
`"per_task": None`). The guards in `test_per_task_loss_instrumentation.py` bind to
`_process_sequence` and to two extracted helpers — all strong — but **nothing in the repository
executed the WRITE path**. `tests/test_training_engine.py` and `tests/test_degenerate_configs.py`
do drive `training_loop`, but never with `trajectory_log_path` set, so the header and the rows were
never produced by a real run.

Two AST guards stood in for that execution, and they could not do the job: they compare key NAMES
to key NAMES and never look at a value, so the guard against "a value silently dropped at a
function boundary" was itself blind to a value silently dropped at a function boundary. They have
been removed in favour of this file.

The rig is a deliberate copy of `test_training_engine.py`'s tiny synthetic world rather than an
import across test modules — the repo's WET-for-test-rigs precedent, and it keeps this file's
provenance fixed if that one changes.
"""

import csv
import math

import numpy as np
import pytest
import torch

from views_hydranet.utils.volume_handler import VolumeHandler

T, H, W = 6, 8, 8
FEATURES = ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"]
IDENTITY = ["month_id", "priogrid_gid"]
CHANNEL_MAP = IDENTITY + FEATURES

TINY_CONFIG = {
    "model": "HydraBNUNet06_LSTM4",
    "input_channels": len(FEATURES),
    "output_channels": 1,
    "regression_targets": FEATURES,
    "classification_targets": [],
    "features": FEATURES,
    "steps": [1, 2],
    "total_lessons": 2,
    "windows_per_lesson": 1,
    "window_dim": 8,
    "np_seed": 42,
    "torch_seed": 42,
    "learning_rate": 1e-3,
    "weight_decay": 0.0,
    "scheduler": "none",
    "loss_reg": "mse",
    "loss_class": "bce",
    "clip_grad_norm": True,
    "random_flips": False,
    "diagnostic_visualizations": False,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "xavier_uni",
    "min_events": 1,
    "sampling_strategy": "threshold",
    "min_ratio": 0.0,
    "max_ratio": 1.0,
    "slope_ratio": 0.5,
    "roof_ratio": 1.0,
    "identity_cols": IDENTITY,
    "spatial_cols": ["row", "col"],
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "transformations": {"identity": FEATURES},
    "derivations": {},
}


@pytest.fixture
def tiny_handler():
    rng = np.random.RandomState(42)
    data = rng.rand(T, H, W, len(CHANNEL_MAP)).astype(np.float32)
    for t in range(T):
        data[t, :, :, 0] = 500 + t
    for r in range(H):
        for c in range(W):
            data[:, r, c, 1] = 1 + r * W + c
    return VolumeHandler(
        data=data,
        axes=("T", "H", "W", "C"),
        channel_map=CHANNEL_MAP,
        time_col="month_id",
        id_col="priogrid_gid",
        spatial_cols=("row", "col"),
        identity_cols=tuple(IDENTITY),
        feature_cols=tuple(FEATURES),
    )


@pytest.fixture
def written_csv(tiny_handler, tmp_path):
    """Run a real (tiny) `training_loop` with the trajectory log ON, return the parsed CSV."""
    from views_hydranet.train.training_engine import make, training_loop

    path = tmp_path / "nested" / "logs" / "trajectory.csv"  # parent does NOT exist
    config = dict(TINY_CONFIG)
    config["trajectory_log_path"] = str(path)
    device = torch.device("cpu")
    model, criterion, optimizer, scheduler = make(config, device)
    training_loop(config, model, criterion, optimizer, scheduler, tiny_handler, device)

    assert path.exists(), "training_loop with trajectory_log_path set must write the file"
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    return {"config": config, "header": rows[0], "rows": rows[1:]}


def test_the_header_is_the_base_columns_plus_one_column_per_task_BY_VALUE(written_csv):
    """⛔ By VALUE, not by count. Renaming or reordering the base columns, dropping the task
    columns, or building them from the wrong config key all mislabel real data."""
    from views_hydranet.train.training_engine import (
        _TRAJECTORY_BASE_COLUMNS,
        _trajectory_task_columns,
    )

    expected = _TRAJECTORY_BASE_COLUMNS + _trajectory_task_columns(written_csv["config"])
    assert written_csv["header"] == expected


def test_rows_were_actually_written(written_csv):
    assert len(written_csv["rows"]) == TINY_CONFIG["total_lessons"]


def test_every_row_has_the_header_width(written_csv):
    """A ragged CSV is what `read_csv` either rejects or silently column-shifts."""
    n = len(written_csv["header"])
    for i, row in enumerate(written_csv["rows"]):
        assert len(row) == n, f"row {i} has {len(row)} cells, header has {n}"


def test_the_per_task_cells_are_populated_and_finite(written_csv):
    """⛔ THE GUARD FOR THE DEFECT THIS FEATURE KEEPS REGROWING.

    `train()` returning `per_task: None`, inverting the accumulator's `is not None` test, or
    swallowing the accumulation in a bare except all leave these cells EMPTY while every other
    test in the repo passes. That is how the feature shipped inert the first time.
    """
    header = written_csv["header"]
    task_idx = [i for i, c in enumerate(header) if c.startswith("loss/")]
    assert task_idx, "the header has no per-task columns at all"
    for row in written_csv["rows"]:
        for i in task_idx:
            assert row[i] != "", f"{header[i]} is EMPTY — the per-task split never arrived"
            assert math.isfinite(float(row[i])), f"{header[i]} is not finite"


def test_the_per_task_cells_sum_to_the_pooled_columns_in_the_SAME_row(written_csv):
    """The cross-check that catches a wrong scale factor.

    `loss_reg` + `loss_cls` and the six `loss/*` cells are two views of the same lesson, so they
    must agree. Dividing by `windows_per_lesson` where the pooled columns multiply (or vice versa)
    puts them out by a factor — exactly the class of loss-scale error this instrumentation exists
    to diagnose, so it must not be possible to introduce it silently.
    """
    header = written_csv["header"]
    i_reg, i_cls = header.index("loss_reg"), header.index("loss_cls")
    task_idx = [i for i, c in enumerate(header) if c.startswith("loss/")]
    for row in written_csv["rows"]:
        pooled = float(row[i_reg]) + float(row[i_cls])
        per_task = sum(float(row[i]) for i in task_idx)
        assert per_task == pytest.approx(pooled, rel=1e-5, abs=1e-8), (
            f"per-task cells sum to {per_task} but loss_reg+loss_cls is {pooled} in the same row"
        )


def test_the_file_is_written_under_a_parent_that_did_not_exist(written_csv, tmp_path):
    """The pod case: an absolute path under a directory the runner never created."""
    assert (tmp_path / "nested" / "logs" / "trajectory.csv").exists()
