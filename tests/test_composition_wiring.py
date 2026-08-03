"""S4 (#187, Epic #183) — emit-time composition wiring for the point / AR-feedback path.

Confirms `_emit_magnitude` composes the family mean with the cls gate per `forecast_composition`
(ADR-069): nb+soft_gate emits `log1p(gate×μ)`, nb+threshold_gate zeros below τ, while zinb
(self-zeroed) and legacy heads are unchanged. The sample-cube side is covered by the sampler +
composer suites; this pins the AR-feedback value the T>0 rollout will consume.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from views_hydranet.utils.hydranet_inference import HydraNetInference

ATOL = 1e-5


class _MockModel(nn.Module):
    def __init__(self, output_distribution="standard"):
        super().__init__()
        self.output_distribution = output_distribution
        self.hurdle_nb_theta = None

    def forward(self, x, h):  # pragma: no cover
        raise NotImplementedError


def _inf(output_distribution, composition="self_zeroed", threshold=None, targets=("lr_ged_sb",)):
    cfg = {
        "regression_targets": list(targets),
        "forecast_composition": composition,
        "gate_threshold": threshold,
    }
    return HydraNetInference(_MockModel(output_distribution), cfg, device="cpu")


def test_nb_soft_gate_emits_gate_times_mean():
    # nb mean = μ; activated params [μ=2, θ=1]; gate=0.5 -> E[y]=0.5*2=1 -> log1p(1)
    reg = torch.tensor([2.0, 1.0]).reshape(1, 2, 1, 1)
    out = _inf("nb", "soft_gate")._emit_magnitude(reg, torch.tensor([[[[0.5]]]]))
    assert torch.allclose(out, torch.tensor([[[[math.log1p(1.0)]]]]), atol=ATOL)


def test_nb_threshold_gate_keeps_above_tau_zeros_below():
    reg = torch.tensor([2.0, 1.0]).reshape(1, 2, 1, 1)  # μ=2
    inf = _inf("nb", "threshold_gate", threshold=0.5)
    # gate 0.9 >= 0.5 -> full mean μ=2 -> log1p(2)
    kept = inf._emit_magnitude(reg, torch.tensor([[[[0.9]]]]))
    assert torch.allclose(kept, torch.tensor([[[[math.log1p(2.0)]]]]), atol=ATOL)
    # gate 0.2 < 0.5 -> zeroed -> log1p(0) = 0
    zeroed = inf._emit_magnitude(reg, torch.tensor([[[[0.2]]]]))
    assert torch.allclose(zeroed, torch.zeros(1, 1, 1, 1), atol=ATOL)


def test_zinb_self_zeroed_unchanged_by_wiring():
    # zinb mean = (1-π)μ; [μ=2, θ=1, π=0.25] -> 1.5 -> log1p(1.5); gate ignored on the self path
    reg = torch.tensor([2.0, 1.0, 0.25]).reshape(1, 3, 1, 1)
    out = _inf("zinb", "self_zeroed")._emit_magnitude(reg, torch.tensor([[[[0.9]]]]))
    assert torch.allclose(out, torch.tensor([[[[math.log1p(1.5)]]]]), atol=ATOL)


def test_legacy_standard_unchanged_by_wiring():
    # a legacy head never enters the family branch: identity, no composition
    reg = torch.tensor([[[[2.0]]]])
    out = _inf("standard", "self_zeroed")._emit_magnitude(reg, torch.tensor([[[[0.5]]]]))
    assert torch.equal(out, reg)
