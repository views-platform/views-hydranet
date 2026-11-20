"""Regression guards — round-2 head/mask falsify findings (2026-06-26), now fixed.

C-179: the activation must be recoverable from the artifact sidecar (else reload silently uses the
       current default activation — wrong forward, no error; softplus/relu share weight shapes).
       Since #633 the sidecar's `body` names it.
C-180: a positives body mask must NOT be a silent no-op under a latent loss — fail loud. Originally
       a warn-once in training_engine; ADR-065 (Epic #158) promoted it to a fail-loud config
       validator (validate_body_mask_latent), so this now guards the stronger contract.

These started as failing falsification stubs; the fixes (train_model.py / config_initializer.py)
make them green. They guard against regression of either fix.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1] / "views_hydranet"


# ── C-179: the activation round-trips the artifact sidecar ──
# Since #633 (ADR-081) the sidecar records `body`, whose name states the activation; that the
# sidecar records it is tested in tests/train/test_sidecar_records_the_loss.py.
def test_reg_activation_respected_on_reload():
    """Behavioural: the reloaded head applies the activation the body names, and a record saved
    before #633 that says relu is refused rather than reloaded as softplus (#632)."""
    from views_hydranet.utils.model_artifact_fetcher import body_record_from_old_sidecar
    from views_hydranet.utils.utils import choose_model

    sidecar = {
        "model": "HydraBNUNet06_LSTM4",
        "input_channels": 3,
        "output_channels": 1,
        "total_hidden_channels": 32,
        "dropout_rate": 0.0,
        "weight_init": "xavier_norm",
        "h_init": "abs_rand_exp-100",
    }
    m = choose_model({**sidecar, "body": "mse_softplus_point"}, torch.device("cpu"))
    assert m._reg_activation.__name__ == "softplus"
    old = {**sidecar, "output_distribution": "standard", "reg_activation": "relu"}
    with pytest.raises(ValueError, match="Only softplus is implemented"):
        body_record_from_old_sidecar(old, "old.pt")


# ── C-180: a positives body mask under a latent loss fails loud (not a silent no-op) ──
def test_pos_mask_under_latent_loss_fails_loud(valid_config_dict):
    """A positives body_mask under a latent loss is a no-op, so the config must REJECT it (ADR-065
    validate_body_mask_latent) — the stronger successor to the old C-180 warn-once."""
    import pytest as _pytest

    from views_hydranet.utils.config_initializer import HydraNetConfig

    cfg = {
        **valid_config_dict,
        "body_supervision": "active",
        "body": "listwise_softplus_point",
        "loss_reg_y_max": 100.0,
    }
    with _pytest.raises(ValueError, match="trains on every cell"):
        HydraNetConfig(**cfg)
