import numpy as np
import pytest

from views_hydranet.utils.utils import get_full_tensor, get_train_tensors


def test_scaling_parity_between_train_and_eval():
    """
    REGRESSION GUARD: Ensure both training and eval paths apply the same JIT scaling.
    If this fails, the model is being fed different data scales during training vs inference.
    """
    # Create a dummy volume with raw fatalities (e.g. 100)
    # 8 channels: metadata(0-4) + 3 features(5-7)
    steps, H, W, channels = 10, 32, 32, 8
    vol = np.zeros((steps, H, W, channels))
    # Fill channel 5 (target) with raw values
    vol[:, :, :, 5] = 100.0

    config = {
        "input_channels": 3,
        "first_feature_idx": 5,
        "transform": "log1p",
        "time_steps": 0, # for train hold-out
        "window_dim": 32
    }

    # Define columns to trigger JIT scaling logic
    columns = ["pg", "col", "row", "month", "c_id", "lr_sb_best", "lr_ns_best", "lr_os_best"]

    # 1. Get Eval Tensor (should be scaled)
    full_tensor, _ = get_full_tensor(vol, config=config, columns=columns)
    eval_max = full_tensor.max().item()

    # 2. Get Train Tensor (should be scaled)
    train_tensor = get_train_tensors(vol, sample=0, config=config, device="cpu", columns=columns)
    train_max = train_tensor.max().item()

    # EXPECTATION:
    # 100.0 is preserved exactly (Semantic Space)
    expected_val = 100.0

    print("\nScaling Check:")
    print(f"  Expected Max: {expected_val:.4f}")
    print(f"  Eval Path Max: {eval_max:.4f}")
    print(f"  Train Path Max: {train_max:.4f}")

    assert pytest.approx(eval_max) == expected_val, "Eval path failed to scale raw input!"
    assert pytest.approx(train_max) == expected_val, "Train path failed to scale raw input!"
    assert pytest.approx(train_max) == eval_max, "Symmetry Broken! Train and Eval paths see different scales."
