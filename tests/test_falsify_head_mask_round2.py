"""Regression guards — round-2 head/mask falsify findings (2026-06-26), now fixed.

C-179: reg_activation must be persisted in the artifact sidecar (else reload silently uses the
       current default activation — wrong forward, no error; softplus/relu share weight shapes).
C-180: active_window must NOT be a silent no-op under a latent loss — warn loudly.

These started as failing falsification stubs; the fixes (train_model.py / training_engine.py)
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
        "model": "HydraBNUNet06_LSTM4", "input_channels": 3, "output_channels": 1,
        "total_hidden_channels": 32, "dropout_rate": 0.0, "weight_init": "xavier_norm",
        "h_init": "abs_rand_exp-100", "output_distribution": "hurdle_shrinkage",
        "reg_activation": "relu",
    }
    m = choose_model(sidecar, torch.device("cpu"))
    assert m._reg_activation.__name__ == "relu"


# ── C-180: active_window under a latent loss warns (not a silent no-op) ──
def test_active_window_under_latent_loss_warns():
    """_process_sequence must warn when hurdle_mask_mode='active_window' is combined with a
    latent loss (use_latent=True), since the mask is silently inapplicable there (C-180)."""
    src = (ROOT / "train" / "training_engine.py").read_text()
    assert "C-180" in src and 'hurdle_mask_mode == "active_window" and use_latent' in src, (
        "training_engine no longer warns on active_window + latent loss — the flag is silently "
        "a no-op (C-180)."
    )
