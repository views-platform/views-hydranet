"""The standard head emits its output unchanged, and a reload builds the head the sidecar names.

Until 2026-11-17 this file also held the hurdle-NB emit tests; that body is deleted (issue #628).
"""

import torch
import torch.nn as nn

from views_hydranet.utils.hydranet_inference import HydraNetInference


class _MockModel(nn.Module):
    """Minimal stand-in carrying the attribute inference reads."""

    def __init__(self, output_distribution="standard"):
        super().__init__()
        self.output_distribution = output_distribution

    def forward(self, x, h):  # pragma: no cover - not exercised by these unit tests
        raise NotImplementedError


def test_emit_standard_is_identity():
    inf = HydraNetInference(_MockModel(), {"regression_targets": ["lr_ged_sb"]}, device="cpu")
    reg = torch.tensor([[[[2.0]]]])
    assert torch.equal(inf._emit_magnitude(reg, torch.tensor([[[[0.5]]]])), reg)


def test_reload_threads_output_distribution_via_choose_model():
    # The confirmed reload bug: output_distribution must survive the sidecar -> choose_model path.
    from views_hydranet.utils.utils import choose_model

    cfg = {
        "model": "HydraBNUNet06_LSTM4",
        "input_channels": 8,
        "total_hidden_channels": 32,
        "output_channels": 1,
        "dropout_rate": 0.0,
        "body": "nll_softplus_nb",
    }
    m = choose_model(cfg, torch.device("cpu"))
    assert m.output_distribution == "nb"
    assert m.dec_conv4_head1_reg.out_channels == 2, "the head must be built for the named body"
