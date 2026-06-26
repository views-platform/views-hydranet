"""Falsification stub + opt-in guard — C-181 (gate loss has no valid-cell mask, 2026-06-26).

The classification (gate) loss is computed on the full zero-filled grid, so it trains on ~60%
structural-zero ocean cells — unlike the hurdle-masked reg loss and the priogrid-masked eval.
An opt-in `cls_valid_mask` was added to enable an A/B (gate masked to land vs current). The fix
(making masking the default) is a DECISION pending that A/B.
"""
import inspect

import pytest

torch = pytest.importorskip("torch")


# C-181 characterization: gate trains on the FULL grid by default (A/B-confirmed benign)
def test_gate_trains_full_grid_by_default():
    """The gate loss is computed on the full zero-filled grid (incl. ~60% ocean) by default.
    Asymmetric with the hurdle-masked reg loss + priogrid-masked eval, BUT the A/B (2026-06-26,
    dossier 18) showed land-masking makes no material difference (MCR_pos/CRPS_pos within seed
    noise) — the ocean dilution is benign. So full-grid stays the default; the cls_valid_mask
    opt-in is kept (sibling test) for a future smarter gate. This characterizes the *current*
    default — it does NOT forbid masking (don't lock out future work)."""
    from views_hydranet.train.training_engine import _process_sequence

    # Default: cls_valid_mask is off ⇒ gate supervised on the full grid (byte-unchanged behavior).
    assert inspect.signature(_process_sequence).parameters["cls_valid_mask"].default is None


# ── GREEN: the opt-in mechanism exists and restricts to land ──
def test_cls_valid_mask_opt_in_restricts_to_land():
    """The cls_valid_mask param is threaded into _process_sequence, and the [:, mask] application
    selects only land cells (so an A/B can compare land-masked vs full-grid gate training)."""
    from views_hydranet.train.training_engine import _process_sequence

    assert "cls_valid_mask" in inspect.signature(_process_sequence).parameters
    assert inspect.signature(_process_sequence).parameters["cls_valid_mask"].default is None
    pred = torch.randn(2, 4, 4)            # [B, H, W]
    land = torch.zeros(4, 4, dtype=torch.bool)
    land[0, 0] = True
    land[1, 1] = True
    assert pred[:, land].shape == (2, 2)   # B x n_land — the masked-select the loss applies
