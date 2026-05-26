import pytest
import torch

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import ModelOutput
from views_hydranet.utils.hydranet_inference import HydraNetInference


class MockModel(torch.nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.base = 32
        self.channels = channels

    def forward(self, x, h):
        # Update h by adding x (simulated update)
        h_new = h + 0.1
        # Dummy predictions
        reg = torch.randn(x.shape[0], 3, x.shape[2], x.shape[3])
        cls = torch.randn(x.shape[0], 3, x.shape[2], x.shape[3])
        return ModelOutput(reg=reg, cls=cls, h_next=h_new)


@pytest.fixture
def inf_setup():
    model = MockModel(channels=64)
    cfg = {
        "freeze_h": "none",
        "sampling_strategy": "threshold",
        "time_steps": 1,
        "steps": [1],
        "regression_targets": ["r1", "r2", "r3"],
        "classification_targets": ["c1", "c2", "c3"],
        "features": ["f1", "f2"],
    }
    inf = HydraNetInference(model, cfg, device="cpu")
    return inf, model


def test_freeze_h_none(inf_setup):
    inf, model = inf_setup
    inf.config["freeze_h"] = "none"

    h = torch.zeros(1, 64, 4, 4)
    x = torch.randn(1, 2, 4, 4)

    _, _, h_out = inf.execute_freeze_h_option(x, h)

    # Should be fully updated
    assert torch.allclose(h_out, h + 0.1)


def test_freeze_h_all(inf_setup):
    inf, model = inf_setup
    inf.config["freeze_h"] = "all"

    h = torch.zeros(1, 64, 4, 4)
    x = torch.randn(1, 2, 4, 4)

    _, _, h_out = inf.execute_freeze_h_option(x, h)

    # Should NOT be updated
    assert torch.allclose(h_out, h)


def test_freeze_h_hl(inf_setup):
    inf, model = inf_setup
    inf.config["freeze_h"] = "hl"

    h = torch.zeros(1, 64, 4, 4)
    x = torch.randn(1, 2, 4, 4)

    _, _, h_out = inf.execute_freeze_h_option(x, h)

    # First half (hs) updated, second half (hl) frozen
    assert torch.allclose(h_out[:, :32], h[:, :32] + 0.1)
    assert torch.allclose(h_out[:, 32:], h[:, 32:])


def test_freeze_h_hs(inf_setup):
    inf, model = inf_setup
    inf.config["freeze_h"] = "hs"

    h = torch.zeros(1, 64, 4, 4)
    x = torch.randn(1, 2, 4, 4)

    _, _, h_out = inf.execute_freeze_h_option(x, h)

    # First half (hs) frozen, second half (hl) updated
    assert torch.allclose(h_out[:, :32], h[:, :32])
    assert torch.allclose(h_out[:, 32:], h[:, 32:] + 0.1)


def test_freeze_h_random(inf_setup):
    inf, model = inf_setup
    inf.config["freeze_h"] = "random"

    h = torch.zeros(1, 64, 4, 4)
    x = torch.randn(1, 2, 4, 4)

    # Run multiple times to ensure we see both states (probabilistic)
    results = []
    for _ in range(20):
        _, _, h_out = inf.execute_freeze_h_option(x, h)
        results.append(h_out)

    # Check that for each chunk of 8 channels, it's either 0 or 0.1
    for res in results:
        for i in range(8):
            chunk = res[:, i * 8 : (i + 1) * 8]
            is_old = torch.allclose(chunk, torch.zeros_like(chunk))
            is_new = torch.allclose(chunk, torch.zeros_like(chunk) + 0.1)
            assert is_old or is_new
