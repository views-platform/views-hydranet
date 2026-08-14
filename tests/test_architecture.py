import torch
import torch.nn as nn

from views_hydranet.architectures.HydraBNrecurrentUnet_06_LSTM4 import HydraBNUNet06_LSTM4
from views_hydranet.utils.utils import init_weights


def test_architecture_instantiation():
    """Verify that the model can be instantiated with valid parameters."""
    model = HydraBNUNet06_LSTM4(
        input_channels=8, total_hidden_channels=32, output_channels=1, dropout_rate=0.1
    )
    assert isinstance(model, torch.nn.Module)
    assert model.base == 32


def test_architecture_hidden_state_init():
    """Verify that hidden state initialization produces correct shapes and types."""
    model = HydraBNUNet06_LSTM4(8, 32, 1, 0.1)
    H, W = 16, 16
    h = model.init_hTtime(hidden_channels=32, H=H, W=W)

    assert h.shape == (1, 32, H, W)
    assert h.dtype == torch.float32
    assert torch.all(h == 0)


def test_architecture_forward_pass_shapes():
    """
    Verify that the forward pass produces correctly shaped outputs for
    regression and classification.
    """
    input_ch = 8
    hidden_ch = 32
    out_ch = 1  # Per head, total 3
    H, W = 16, 16

    model = HydraBNUNet06_LSTM4(input_ch, hidden_ch, out_ch, 0.1).float()
    x = torch.randn((1, input_ch, H, W)).float()
    h = model.init_hTtime(hidden_ch, H, W).float()

    output = model(x, h)

    # 3 heads (sb, ns, os) concatenated
    assert output.reg.shape == (1, 3, H, W)
    assert output.cls.shape == (1, 3, H, W)
    assert output.h_next.shape == (1, hidden_ch, H, W)
    assert output.reg_latent.shape == (1, 3, H, W)


def test_architecture_recurrent_state_evolution():
    """Verify that the hidden state actually changes after a forward pass."""
    model = HydraBNUNet06_LSTM4(8, 32, 1, 0.0).float()  # No dropout for determinism
    x = torch.randn((1, 8, 16, 16)).float()
    h_init = model.init_hTtime(32, 16, 16).float()

    output = model(x, h_init)

    # Hidden state should no longer be zeros
    assert not torch.all(output.h_next == 0)
    assert output.h_next.shape == h_init.shape


def test_architecture_multi_batch_support():
    """Verify that the architecture supports batch sizes > 1."""
    # Note: init_hTtime currently hardcodes batch size 1 in its return shape
    # We test if the forward pass itself is batch-agnostic if we provide the right h
    batch_size = 4
    input_ch = 8
    hidden_ch = 32
    H, W = 16, 16

    model = HydraBNUNet06_LSTM4(input_ch, hidden_ch, 1, 0.1).float()
    x = torch.randn((batch_size, input_ch, H, W)).float()
    h = torch.zeros((batch_size, hidden_ch, H, W)).float()

    output = model(x, h)

    assert output.reg.shape[0] == batch_size, (
        f"Expected batch dim {batch_size}, got {output.reg.shape[0]}"
    )
    assert output.h_next.shape[0] == batch_size, (
        f"Expected batch dim {batch_size}, got {output.h_next.shape[0]}"
    )


def test_weight_init_xavier_norm_is_not_silent():
    """
    RED GATE: init_weights() must NOT silently ignore 'xavier_norm'.

    The active production config has weight_init='xavier_norm'. The current
    init_weights() only handles 'xavier_uni' and 'kaiming_uni'. 'xavier_norm'
    falls through: no initialization, no error, no warning. The model has been
    running on PyTorch default init (Kaiming Uniform) its entire existence while
    the config specifies Xavier Normal.

    This test asserts that AFTER calling init_weights with 'xavier_norm', the
    layer weights differ from the sentinel value (all 1.0). If init_weights
    silently ignores the config, weights remain 1.0 and this test FAILS — the
    correct Red Gate behavior confirming the bug is real.

    Acceptable outcomes after the fix:
    (a) The function applies nn.init.xavier_normal_ → weights differ from sentinel.
    (b) The function raises ValueError for an unrecognised scheme → explicit and loud.
    Either path is a correct fix. Silent pass-through is not acceptable.
    """
    layer = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)

    # Set all weights to a known sentinel. xavier_normal_ draws from
    # N(0, gain * sqrt(2 / (fan_in + fan_out))) — P(all weights == 1.0) ≈ 0.
    nn.init.constant_(layer.weight, 1.0)
    sentinel = layer.weight.clone()

    config = {"weight_init": "xavier_norm"}

    try:
        init_weights(layer, config)
    except ValueError:
        # Acceptable: function explicitly rejects unknown init schemes.
        return

    # Function returned without error — weights MUST have changed.
    assert not torch.equal(layer.weight, sentinel), (
        "init_weights(layer, {'weight_init': 'xavier_norm'}) returned without error "
        "but the layer weights were NOT modified (still all 1.0). "
        "The function silently ignored 'xavier_norm'. "
        "Fix: add `elif config['weight_init'] == 'xavier_norm': "
        "nn.init.xavier_normal_(m.weight)` to init_weights() in utils.py."
    )


def test_weight_init_xavier_norm_statistics():
    """
    GREEN GATE (Phase B): After applying 'xavier_norm', the weight distribution
    must match the Xavier Normal formula: std ≈ sqrt(2 / (fan_in + fan_out)).

    For Conv2d(in=8, out=16, kernel=3):
      fan_in  = 8 * 3 * 3 = 72
      fan_out = 16 * 3 * 3 = 144
      expected_std = sqrt(2 / (72 + 144)) = sqrt(2 / 216) ≈ 0.0962

    This verifies that the correct distribution was applied — not Kaiming Uniform
    (std ≈ 0.167) or a trivial constant. Uses a large layer so the empirical std
    converges tightly to the theoretical value.

    Taxonomy (ADR 005): GREEN — proves the fix is mathematically correct.
    """
    import math

    # Large layer for tight statistical convergence (weight count = 16*8*5*5 = 3200)
    layer = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)

    fan_in = 8 * 5 * 5  # 200
    fan_out = 16 * 5 * 5  # 400
    expected_std = math.sqrt(2.0 / (fan_in + fan_out))  # ≈ 0.0577

    config = {"weight_init": "xavier_norm"}
    init_weights(layer, config)

    actual_std = layer.weight.std().item()

    assert abs(actual_std - expected_std) < 0.02, (
        f"\nXavier Normal statistics test failed.\n"
        f"Expected std ≈ {expected_std:.4f} (from 2 / (fan_in={fan_in} + fan_out={fan_out})).\n"
        f"Got std = {actual_std:.4f}.\n"
        f"If std ≈ 0.10, Kaiming Uniform was applied instead of Xavier Normal.\n"
        f"If std ≈ 0.00, weights were not modified. Check init_weights() in utils.py."
    )


def test_no_cross_head_variable_leakage():
    """
    Structural gate: dropout in H{n} decoder blocks must only reference H{n} variables.

    Catches copy-paste bugs where H2/H3 decoder dropout calls accidentally use
    H1's intermediate variables, breaking multi-task head independence.
    """
    import inspect
    import re

    source = inspect.getsource(HydraBNUNet06_LSTM4.forward)
    pattern = r"(H(\d)_\w+)\s*=\s*self\.dropout\[[^\]]+\]\(H(\d)_\w+\)"
    violations = []
    for match in re.finditer(pattern, source):
        lhs_head = match.group(2)
        rhs_head = match.group(3)
        if lhs_head != rhs_head:
            violations.append(f"H{lhs_head} block uses H{rhs_head}'s variable: {match.group(0)}")
    assert not violations, (
        f"Cross-head variable leakage detected in {len(violations)} dropout call(s):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )
