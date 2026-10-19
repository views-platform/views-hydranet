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
    # ⛔ 2, not 1. With one window per lesson the per-task divisor is `/1`, so multiplying
    # instead of dividing by it is undetectable — a mutation doing exactly that survived this
    # file's first version (independent guard audit, PR #398).
    "windows_per_lesson": 2,
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


def test_the_header_is_exactly_these_literal_column_names(written_csv):
    """⛔ LITERAL strings, not the production constants.

    The first version of this test asserted
    `header == _TRAJECTORY_BASE_COLUMNS + _trajectory_task_columns(config)` — which compares the
    written header against the very constants that produced it. Renaming `"lesson"` to
    `"LESSON_IDX"`, or swapping `"loss_reg"` and `"loss_cls"` while leaving the row order alone,
    changed BOTH sides and the test stayed green while two columns of real data were mislabelled.
    A tautology, and the third one I wrote in this PR.

    So the names are spelled out here. Changing a column name is now a deliberate edit to this
    literal — which is the point: downstream readers (and six-months-later me) parse by NAME.
    """
    assert written_csv["header"] == [
        "lesson",
        "raw_grad_norm",
        "loss_reg",
        "loss_cls",
        "gate_logit_mean",
        "fed_grad_max",
        "loss/body/lr_ged_sb",
        "loss/body/lr_ged_ns",
        "loss/body/lr_ged_os",
    ]


def test_the_base_columns_are_in_the_order_the_row_is_written(written_csv):
    """`loss_reg` must label the regression value and `loss_cls` the classification one.

    Swapping the two header entries while the row-writing order is unchanged mislabels both
    columns with no other symptom. TINY_CONFIG has no classification targets, so `loss_cls` is
    identically 0 and `loss_reg` is not — which is what makes the swap detectable here.
    """
    header = written_csv["header"]
    i_reg, i_cls = header.index("loss_reg"), header.index("loss_cls")
    for row in written_csv["rows"]:
        assert float(row[i_cls]) == 0.0, "no classification targets, so loss_cls must be 0"
        assert float(row[i_reg]) != 0.0, "loss_reg must carry the regression loss"


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


# ── the two properties that are NOT reachable end-to-end, tested as a unit ──


def test_per_task_cells_are_the_lesson_MEAN():
    """Divides by windows_per_lesson. Multiplying puts the per-task numbers out by its square
    against the pooled columns in the same row."""
    from views_hydranet.train.training_engine import _trajectory_per_task_cells

    cells = _trajectory_per_task_cells(np.array([6.0, 9.0, 3.0]), 3, 3)
    assert cells == pytest.approx([2.0, 3.0, 1.0])


def test_per_task_cells_pad_to_the_header_width_when_there_is_no_vector():
    """A lesson that produced nothing must still yield the header's width, or the CSV goes
    ragged. Unreachable in a healthy run, which is why it is tested here and not end to end."""
    from views_hydranet.train.training_engine import _trajectory_per_task_cells

    assert _trajectory_per_task_cells(None, 6, 3) == ["", "", "", "", "", ""]
    assert len(_trajectory_per_task_cells(None, 6, 3)) == 6


def test_per_task_cells_width_always_matches_the_column_count():
    from views_hydranet.train.training_engine import _trajectory_per_task_cells

    for n in (1, 3, 6, 12):
        assert len(_trajectory_per_task_cells(None, n, 1)) == n


def test_fed_grad_max_is_BLANK_not_zero_when_the_feedback_wire_is_off(written_csv):
    """⛔ The config in this rig leaves `ss_backprop_through_feedback` at its False default, so
    `_attach_feedback_grad_clip` is never called and nothing is ever measured.

    The column used to be initialised to 0.0 and written as a real number, making "the wire is
    off" and "the wire is on and the gradient died" indistinguishable in a delivered
    trajectory.csv — a misreading this repo's own validator docstrings name four times
    (`config_initializer.py:951,958,977,989`) while only blocking the two configurations where it
    is dangerous, never the ordinary one that ships.

    A blank cell is what `read_csv` turns into NaN, i.e. "not measured", which is the truth.
    """
    assert "ss_backprop_through_feedback" not in written_csv["config"], (
        "this guard is only meaningful while the rig leaves the feedback wire at its default"
    )
    col = written_csv["header"].index("fed_grad_max")
    cells = [row[col] for row in written_csv["rows"]]
    assert cells, "no rows were written, so this guard proved nothing"
    assert all(c == "" for c in cells), (
        f"fed_grad_max must be blank when nothing was measured, got {sorted(set(cells))}"
    )
    assert not any(c in ("0", "0.0") for c in cells), (
        "a literal zero here reads as a healthy measured gradient — the exact defect"
    )
