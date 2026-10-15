"""The per-task loss split must survive out of `_process_sequence` and into the trajectory CSV.

WHY THIS EXISTS. `training_engine` used to collapse the balancer's 6-vector to two sums
(`reg`, `cls`) and nothing else. `trajectory_log_path` — the only sink for even those two — was
unset on BOTH delivered runs of the composition arm, so when one task-loss term turned out to be
**65x** what the design assumed, the per-target split had to be reconstructed from the saved
WEIGHTS weeks later (`reports/2026-10-13_composition_arm_outcome/r2_loss_scale.py`, `r3_*`). r3
then showed the finding was **per target**, which the two pooled sums structurally could not show.

The load-bearing invariant is self-consistency: the six per-task values must sum to the two pooled
values. That is what catches a wrong task order, a dropped task, or an accumulator that misses a
step — none of which a column count would catch.
"""

import numpy as np
import pytest
import torch

from views_hydranet.train.training_engine import _process_sequence, _SequenceIndices


class _TinyNet(torch.nn.Module):
    """Two 1x1 convs over a 2-channel feature map -> 3 body params-pairs + 3 gate logits."""

    def __init__(self, n_reg: int, n_cls: int, n_params: int):
        super().__init__()
        self.base = 2
        self.reg_head = torch.nn.Conv2d(2, n_reg * n_params, 1)
        self.cls_head = torch.nn.Conv2d(2, n_cls, 1)

    def init_hTtime(self, hidden_channels, H, W):
        return torch.zeros(1, hidden_channels, H, W)

    def forward(self, x, h):
        from types import SimpleNamespace

        feat = h + x[:, :2]
        return SimpleNamespace(reg=self.reg_head(feat), cls=self.cls_head(feat), h_next=feat)


@pytest.fixture
def rig():
    torch.manual_seed(0)
    from views_hydranet.distributions import resolve_family
    from views_hydranet.distributions.family_loss import FamilyLoss
    from views_hydranet.utils.mtloss import MultiTaskLoss
    from views_hydranet.utils.weighted_bce_loss import WeightedBCEWithLogitsLoss

    reg_names = ["lr_sb_best", "lr_ns_best", "lr_os_best"]
    cls_names = ["by_sb_best", "by_ns_best", "by_os_best"]
    feature_names = reg_names + cls_names + ["f_a", "f_b"]
    config = {
        "regression_targets": reg_names,
        "classification_targets": cls_names,
        "features": ["f_a", "f_b"],
    }
    idx = _SequenceIndices(feature_names, config)
    fam = resolve_family("nb")
    model = _TinyNet(len(reg_names), len(cls_names), fam.n_params)
    mtl = MultiTaskLoss(
        torch.tensor([True] * len(reg_names) + [False] * len(cls_names)), reduction="sum"
    )
    T, H, W = 4, 5, 5
    tt = torch.zeros(1, T, len(feature_names), H, W)
    tt[:, :, idx.reg] = torch.log1p(torch.randint(0, 4, (1, T, len(reg_names), H, W)).float())
    tt[:, :, idx.cls] = (tt[:, :, idx.reg] > 0).float()
    tt[:, :, idx.feat] = torch.randn(1, T, len(config["features"]), H, W) * 0.1
    return dict(
        tt=tt,
        model=model,
        idx=idx,
        crit_reg=FamilyLoss(fam),
        crit_cls=WeightedBCEWithLogitsLoss(pos_weight=2.0),
        mtl=mtl,
        names=[f"body/{n}" for n in reg_names] + [f"gate/{n}" for n in cls_names],
    )


def _run(rig):
    h = rig["model"].init_hTtime(hidden_channels=2, H=rig["tt"].shape[-2], W=rig["tt"].shape[-1])
    return _process_sequence(
        rig["tt"],
        rig["model"],
        h,
        rig["crit_reg"],
        rig["crit_cls"],
        rig["mtl"],
        rig["idx"],
        torch.device("cpu"),
        family=resolve_nb(),
    )


def resolve_nb():
    from views_hydranet.distributions import resolve_family

    return resolve_family("nb")


def test_per_task_is_returned_with_one_entry_per_task(rig):
    out = _run(rig)
    assert out["per_task"] is not None, "the per-task split must survive out of the step function"
    assert out["per_task"].shape == (6,)
    assert out["task_names"] == rig["names"]


def test_per_task_sums_to_the_two_pooled_values(rig):
    """The invariant: body terms sum to `reg`, gate terms sum to `cls`.

    This is what catches a wrong task ORDER, a dropped task, or an accumulator that misses a step.
    A column count would catch none of them.
    """
    out = _run(rig)
    pt = out["per_task"].detach().numpy()
    n_reg = rig["idx"].n_reg
    assert np.isclose(pt[:n_reg].sum(), float(out["reg"]), rtol=1e-5)
    assert np.isclose(pt[n_reg:].sum(), float(out["cls"]), rtol=1e-5)


def test_task_names_order_matches_the_balancer_is_regression_flags(rig):
    """`task_names` must put the regression (body) tasks first — the order the balancer's
    `is_regression` vector assumes. A silent re-order would mislabel every column."""
    out = _run(rig)
    body = [n for n in out["task_names"] if n.startswith("body/")]
    gate = [n for n in out["task_names"] if n.startswith("gate/")]
    assert out["task_names"] == body + gate
    assert len(body) == rig["idx"].n_reg and len(gate) == rig["idx"].n_cls


def test_per_task_is_detached_and_cannot_carry_a_graph(rig):
    """The accumulator must not hold the graph — it runs on every step of every window."""
    out = _run(rig)
    assert not out["per_task"].requires_grad


def test_per_task_values_are_finite_and_nonzero(rig):
    out = _run(rig)
    pt = out["per_task"].detach().numpy()
    assert np.isfinite(pt).all()
    assert (pt != 0).any(), "an all-zero split would pass the sum invariant vacuously"


# ── the trajectory log must be able to come HOME from a pod ──────────────────────────────────
#
# The previous version of this block had two tests and /code-review (PR #398) was right to reject
# both: one RE-IMPLEMENTED the production `os.makedirs` line inside the test and then asserted the
# file existed, so deleting the production line left it green; the other asserted source-text
# ordering via `str.index`, which proves nothing about behaviour and breaks on reformatting. The
# production line is now a function, so these test behaviour.


def test_opening_the_trajectory_file_creates_a_missing_parent(tmp_path):
    """`pod_run_model.sh` creates only artifact/, parquet/ and draws/ under the deliver dir, so an
    absolute path anywhere else has an unmade parent and `open()` alone would raise mid-run —
    after the data pull and the model build."""
    from views_hydranet.train.training_engine import _open_trajectory_file

    target = tmp_path / "deliver" / "amber_kestrel" / "logs" / "trajectory.csv"
    assert not target.parent.exists()
    fh = _open_trajectory_file(str(target))
    fh.close()
    assert target.exists()


def test_opening_the_trajectory_file_still_fails_loud_on_a_file_as_parent(tmp_path):
    """`exist_ok=True` must not mask a real misconfiguration."""
    from views_hydranet.train.training_engine import _open_trajectory_file

    clash = tmp_path / "logs"
    clash.write_text("i am a file, not a directory")
    with pytest.raises((FileExistsError, NotADirectoryError)):
        _open_trajectory_file(str(clash / "trajectory.csv"))


def test_opening_a_bare_filename_works(tmp_path, monkeypatch):
    """A path with no directory component must not break on `dirname` being empty."""
    from views_hydranet.train.training_engine import _open_trajectory_file

    monkeypatch.chdir(tmp_path)
    _open_trajectory_file("trajectory.csv").close()
    assert (tmp_path / "trajectory.csv").exists()


# ── the contract that actually broke: a hand-built dict silently dropped keys ────────────────


def test_train_returns_every_key_the_training_loop_reads_from_it():
    """⛔ THE REGRESSION GUARD FOR THE DEFECT THIS FILE EXISTS BECAUSE OF.

    `_process_sequence` returned `per_task`/`task_names`, but `training_loop` does not call it —
    it calls `train()`, whose return was a hand-built dict literal that dropped both. The
    per-task CSV columns were therefore inert in every real run while every unit test passed,
    because the tests called `_process_sequence` directly. That is C-324, inside the PR whose
    purpose was to prevent a C-324.

    So this asserts the CONTRACT structurally: every key `training_loop` reads out of the dict
    `train` returns must be a key `train` actually puts in it. It is AST-based rather than
    text-based, so it survives reformatting, and it generalises — it will catch the next key
    somebody adds to `_process_sequence` and forgets to forward.
    """
    import ast
    import inspect

    from views_hydranet.train import training_engine

    tree = ast.parse(inspect.getsource(training_engine))
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert {"train", "training_loop"} <= set(fns), "renamed functions — update this guard"

    produced = set()
    for node in ast.walk(fns["train"]):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            produced |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    assert produced, "train() no longer returns a dict literal — update this guard"

    consumed = set()
    for node in ast.walk(fns["training_loop"]):
        # losses["x"]
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "losses"
            and isinstance(node.slice, ast.Constant)
        ):
            consumed.add(node.slice.value)
        # losses.get("x")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "losses"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            consumed.add(node.args[0].value)

    assert consumed, "training_loop no longer reads from `losses` — update this guard"
    missing = consumed - produced
    assert not missing, (
        f"training_loop reads {sorted(missing)} from the dict train() returns, but train() does "
        f"not produce them. train() produces {sorted(produced)}. A key dropped at this boundary "
        f"makes the feature that depends on it silently inert — this is exactly how the per-task "
        f"trajectory columns shipped dead (C-324, PR #398)."
    )


def test_per_task_keys_specifically_survive_the_train_boundary():
    """The named instance of the contract above, so a failure says WHICH feature died."""
    import ast
    import inspect

    from views_hydranet.train import training_engine

    tree = ast.parse(inspect.getsource(training_engine))
    train_fn = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train"
    )
    keys = set()
    for node in ast.walk(train_fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
    assert {"per_task", "task_names"} <= keys, (
        "train() must forward per_task and task_names, or the per-task trajectory columns are "
        "inert in every real run while the unit tests on _process_sequence still pass."
    )


# ── the premise the CSV header's correctness rests on ──
#
# An earlier version of this file had four tests for a `_assert_trajectory_columns_match` guard
# that compared the header (built from config) against the balancer's task names. /review-diff
# (PR #398) found the guard could not fire: `_SequenceIndices.__init__` assigns
# `self.reg_names = config["regression_targets"]` — the SAME list object — so it compared a list
# to itself, and my justification for writing it ("config is a second source of truth that might
# disagree") was false. The guard and its tests are gone. What replaces them is a test of the
# premise that makes the header correct without any guard: if that premise is ever broken, THIS
# fails, and the header's correctness has to be re-argued.


def test_sequence_indices_names_are_the_config_lists(rig):
    """`idx.reg_names`/`cls_names` must BE the config lists, not copies or a re-derivation.

    The trajectory CSV header is built from `config["regression_targets"]` +
    `config["classification_targets"]`, while the balancer sums `losses_list` in
    `range(idx.n_reg)` then `range(idx.n_cls)` order. Those agree only because `idx` takes its
    names straight from config. Break that and the header silently mislabels its columns.
    """
    idx = rig["idx"]
    cfg = {
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
        "features": ["f_a", "f_b"],
    }
    assert list(idx.reg_names) == cfg["regression_targets"]
    assert list(idx.cls_names) == cfg["classification_targets"]
    # and the task names the engine reports must be exactly those, body-first
    assert rig["names"] == (
        [f"body/{n}" for n in cfg["regression_targets"]]
        + [f"gate/{n}" for n in cfg["classification_targets"]]
    )


def test_sequence_indices_reg_names_is_the_very_config_object():
    """Stronger: the SAME object, which is why no comparison guard is needed."""
    from views_hydranet.train.training_engine import _SequenceIndices

    cfg = {
        "regression_targets": ["a", "b"],
        "classification_targets": ["c"],
        "features": ["f"],
    }
    idx = _SequenceIndices(["a", "b", "c", "f"], cfg)
    assert idx.reg_names is cfg["regression_targets"]
    assert idx.cls_names is cfg["classification_targets"]


# ── the trajectory column NAMES, which nothing tested until a mutation survived ──
#
# Deleting the tautological comparison guard (/review-diff) left a real gap: building the header
# from the wrong config key mislabels every column, and that survived the whole suite. The concern
# was genuine even though the guard was not — it just cannot be tested by comparing two things
# that are the same object. So test the construction instead.


def test_trajectory_task_columns_names_and_order():
    from views_hydranet.train.training_engine import _trajectory_task_columns

    cfg = {
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    }
    assert _trajectory_task_columns(cfg) == [
        "loss/body/lr_sb_best",
        "loss/body/lr_ns_best",
        "loss/body/lr_os_best",
        "loss/gate/by_sb_best",
        "loss/gate/by_ns_best",
        "loss/gate/by_os_best",
    ]


def test_trajectory_task_columns_are_body_before_gate():
    """The order the balancer's `is_regression` vector assumes. A swap mislabels every column."""
    from views_hydranet.train.training_engine import _trajectory_task_columns

    cols = _trajectory_task_columns(
        {"regression_targets": ["r1", "r2"], "classification_targets": ["c1"]}
    )
    assert [c.split("/")[1] for c in cols] == ["body", "body", "gate"]


def test_trajectory_task_columns_do_not_read_the_wrong_config_key():
    """Asymmetric target lists, so reading the wrong key cannot coincidentally pass."""
    from views_hydranet.train.training_engine import _trajectory_task_columns

    cols = _trajectory_task_columns(
        {"regression_targets": ["only_reg"], "classification_targets": ["c1", "c2", "c3"]}
    )
    assert cols == [
        "loss/body/only_reg",
        "loss/gate/c1",
        "loss/gate/c2",
        "loss/gate/c3",
    ]


def test_trajectory_header_width_matches_the_per_task_vector(rig):
    """The header's task-column count must equal the length of `per_task`, or rows go ragged."""
    from views_hydranet.train.training_engine import (
        _TRAJECTORY_BASE_COLUMNS,
        _trajectory_task_columns,
    )

    cfg = {
        "regression_targets": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
        "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    }
    out = _run(rig)
    assert len(_trajectory_task_columns(cfg)) == out["per_task"].shape[0]
    assert len(_TRAJECTORY_BASE_COLUMNS) == 6
