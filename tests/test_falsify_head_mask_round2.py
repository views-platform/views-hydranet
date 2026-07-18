"""Regression guards — round-2 head/mask falsify findings (2026-06-26), now fixed.

C-179: reg_activation must be persisted in the artifact sidecar (else reload silently uses the
       current default activation — wrong forward, no error; softplus/relu share weight shapes).
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


# ── C-179: reg_activation round-trips the artifact sidecar ──
def test_reg_activation_persisted_in_sidecar():
    """train_model must persist the resolved reg-head activation in the .config.json snapshot,
    so a reload uses the trained activation even if the output_distribution-keyed default
    changes (ADR-063 / C-178)."""
    src = (ROOT / "train" / "train_model.py").read_text()
    assert 'config_snapshot["reg_activation"]' in src, (
        "train_model no longer persists reg_activation in the sidecar — a model trained with one "
        "activation will silently reload with the current default (C-179)."
    )


def test_reg_activation_respected_on_reload():
    """Behavioural: a sidecar that carries reg_activation='relu' reloads to a relu head
    (the reload side of the round-trip)."""
    from views_hydranet.utils.utils import choose_model

    sidecar = {
        "model": "HydraBNUNet06_LSTM4",
        "input_channels": 3,
        "output_channels": 1,
        "total_hidden_channels": 32,
        "dropout_rate": 0.0,
        "weight_init": "xavier_norm",
        "h_init": "abs_rand_exp-100",
        "output_distribution": "hurdle_shrinkage",
        "reg_activation": "relu",
    }
    m = choose_model(sidecar, torch.device("cpu"))
    assert m._reg_activation.__name__ == "relu"


# ── C-180: a positives body mask under a latent loss fails loud (not a silent no-op) ──
def test_pos_mask_under_latent_loss_fails_loud(valid_config_dict):
    """A positives body_mask under a latent loss is a no-op, so the config must REJECT it (ADR-065
    validate_body_mask_latent) — the stronger successor to the old C-180 warn-once."""
    import pytest as _pytest

    from views_hydranet.utils.config_initializer import HydraNetConfig

    cfg = {
        **valid_config_dict,
        "body_mask": "pos_timelines",
        "loss_reg": "hurdle_nb",
        "loss_reg_theta_init": 1.0,
    }
    with _pytest.raises(ValueError, match="latent likelihood"):
        HydraNetConfig(**cfg)
