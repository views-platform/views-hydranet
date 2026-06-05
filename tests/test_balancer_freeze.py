"""Tests for the C-111 balancer-freeze flag (C-113 bisect).

Pre-analysis: reports/preanalysis_balancer_bisect.md

`freeze_multitask_balancer` (default False) gates the C-111 change in
training_engine.make(): when True, the MultiTaskLoss `log_vars` are NOT added to
the optimizer, so they stay at their zero init (equal weighting = the pre-C-111,
years-stable regime). It must:
  - default to the C-111 behavior (log_vars trainable) -> zero regression;
  - when True, exclude ONLY the balancer log_vars from the optimizer;
  - leave the ADR-055 Tobit sigma params trainable either way (isolate C-111).
"""

import copy

import torch

from tests.conftest import tobit_config_3target
from tests.test_training_engine import TINY_CONFIG
from views_hydranet.train.training_engine import make


def _ids(params):
    return {id(p) for p in params}


def _optimizer_param_ids(optimizer):
    return {id(p) for g in optimizer.param_groups for p in g["params"]}


class TestGreenBalancerActiveByDefault:
    """C-111 behavior is the default (no regression)."""

    def test_log_vars_trainable_by_default(self):
        cfg = copy.deepcopy(TINY_CONFIG)
        cfg.pop("freeze_multitask_balancer", None)  # unset => default
        _, criterion, optimizer, _ = make(cfg, torch.device("cpu"))
        log_vars = _ids(criterion[2].parameters())
        assert log_vars, "MultiTaskLoss should expose log_var parameters"
        assert log_vars <= _optimizer_param_ids(optimizer), (
            "C-111: balancer log_vars must be in the optimizer by default"
        )

    def test_explicit_false_matches_default(self):
        cfg = copy.deepcopy(TINY_CONFIG)
        cfg["freeze_multitask_balancer"] = False
        _, criterion, optimizer, _ = make(cfg, torch.device("cpu"))
        log_vars = _ids(criterion[2].parameters())
        assert log_vars <= _optimizer_param_ids(optimizer)


class TestGreenBalancerFrozen:
    """freeze_multitask_balancer=True restores the pre-C-111 frozen balancer."""

    def test_log_vars_excluded_when_frozen(self):
        cfg = copy.deepcopy(TINY_CONFIG)
        cfg["freeze_multitask_balancer"] = True
        _, criterion, optimizer, _ = make(cfg, torch.device("cpu"))
        log_vars = _ids(criterion[2].parameters())
        assert log_vars, "MultiTaskLoss should still expose log_var parameters"
        assert log_vars.isdisjoint(_optimizer_param_ids(optimizer)), (
            "frozen balancer: log_vars must be EXCLUDED from the optimizer"
        )

    def test_freeze_removes_exactly_the_balancer_group(self):
        # Freezing must remove ONLY the balancer's param group — every other group
        # (model weights, and any ADR-055 Tobit sigma group) stays intact. Uses the
        # tobit config so a sigma group is present in the optimizer if applicable.
        cfg_active = tobit_config_3target()
        cfg_active["freeze_multitask_balancer"] = False
        cfg_frozen = tobit_config_3target()
        cfg_frozen["freeze_multitask_balancer"] = True
        _, _, opt_active, _ = make(cfg_active, torch.device("cpu"))
        _, _, opt_frozen, _ = make(cfg_frozen, torch.device("cpu"))
        assert len(opt_frozen.param_groups) == len(opt_active.param_groups) - 1, (
            "freezing the balancer must drop exactly one optimizer param group "
            "(the log_vars), leaving model + sigma groups untouched"
        )
