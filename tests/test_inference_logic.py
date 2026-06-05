"""Inference-path logic tests.

`freeze_h` was retired 2026-06-05 (see `reports/2026-06-05_rollout_training_dossier/`,
ADR-027 update, register C-113). The autoregressive rollout now always evolves the full
ConvLSTM state (the former `"none"` behaviour) — the only mode that did not create a
train/inference mismatch, and the freeze was empirically inert against the C-113 runaway
(which rides the prediction→input feedback path, not the recurrent state). The five
`execute_freeze_h_option` capability tests were removed with the method; the guard below
prevents silent reintroduction.
"""

from views_hydranet.utils.hydranet_inference import HydraNetInference


def test_freeze_h_option_retired():
    """Regression guard: the freeze_h hack must not be reintroduced.

    `execute_freeze_h_option` was deleted when freeze_h was retired; if it reappears,
    the durable fix (Axis-B rollout training) is being bypassed by a hard-prior hack.
    """
    assert not hasattr(HydraNetInference, "execute_freeze_h_option"), (
        "execute_freeze_h_option was retired (2026-06-05) — the rollout evolves the "
        "full state. Reintroducing it reinstates the inference-time freeze hack."
    )
