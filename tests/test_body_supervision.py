"""S2 (ADR-065 amendment 2026-07-28) — RED tests for the graded `body_supervision` axis.

Retires the `body_mask ∈ {none,pos_cells,pos_timelines}` keyword for a supervision WINDOW:
`body_supervision ∈ {all,active}` + `onset_lead` (months before onset) + `cessation_lag` (months
after cessation). A timestep t in a cell is supervised iff an active month t' exists in
`[t - cessation_lag, t + onset_lead]` (asymmetric temporal dilation of the per-step-positive mask).

These tests define the API and the byte-identical endpoint parity that gates the clean-break
retirement. They import the renamed module `views_hydranet.utils.body_supervision`.
"""

import pytest
import torch

from views_hydranet.utils.body_supervision import (
    event_threshold_from_config,
    resolve_body_supervision,
)


# ---- windows (hand-computable) ----------------------------------------------
def _win_T2():
    # [B=1,T=2,n_reg=1,H=1,W=2] — cell A active at t=0 then decays; cell B never active.
    w = torch.zeros(1, 2, 1, 1, 2)
    w[0, 0, 0, 0, :] = torch.tensor([5.0, 0.0])
    return w


def _win_T5_single():
    # [B=1,T=5,n_reg=1,H=1,W=1] — one cell active ONLY at t=2 (a lone episode).
    w = torch.zeros(1, 5, 1, 1, 1)
    w[0, 2, 0, 0, 0] = 5.0
    return w


def _active_at(mask):  # collapse [1,T,1,1,1] bool -> python list over T
    return mask[0, :, 0, 0, 0].tolist()


# ---- endpoint byte-identical parity (the retirement gate) -------------------
def test_all_endpoint_is_all_true():
    w = _win_T2()
    m = resolve_body_supervision(
        0, 0, 0.0
    )  # resolver is for 'active'; 'all' handled at config/loop
    # 'active' with 0/0 is per-step-positive, NOT all-cell — assert that distinction holds:
    assert not m(w).all()


def test_zero_zero_is_byte_identical_to_per_step_positive():
    """active,onset_lead=0,cessation_lag=0  ==  old pos_cells (window > thr)."""
    w = _win_T2()
    got = resolve_body_supervision(0, 0, 0.0)(w)
    expected = w > 0.0
    assert torch.equal(got, expected)


def test_saturated_radii_are_byte_identical_to_pos_timelines():
    """active,onset_lead>=W,cessation_lag>=W  ==  old pos_timelines (active-anywhere broadcast)."""
    w = _win_T2()
    got = resolve_body_supervision(9, 9, 0.0)(w)
    active_any = (w > 0.0).any(dim=1, keepdim=True).expand_as(w)
    assert torch.equal(got, active_any)


# ---- asymmetric dilation truth-table ----------------------------------------
def test_onset_lead_supervises_the_run_up_not_the_decay():
    """onset_lead=1, cessation_lag=0: episode at t=2 -> supervise t=1 (run-up) and t=2 only."""
    m = resolve_body_supervision(1, 0, 0.0)(_win_T5_single())
    assert _active_at(m) == [False, True, True, False, False]


def test_cessation_lag_supervises_the_decay_not_the_run_up():
    """onset_lead=0, cessation_lag=1: episode at t=2 -> supervise t=2 and t=3 (decay) only."""
    m = resolve_body_supervision(0, 1, 0.0)(_win_T5_single())
    assert _active_at(m) == [False, False, True, True, False]


def test_zero_zero_supervises_only_the_active_step():
    m = resolve_body_supervision(0, 0, 0.0)(_win_T5_single())
    assert _active_at(m) == [False, False, True, False, False]


def test_asymmetric_reaches_both_sides_independently():
    """onset_lead=2, cessation_lag=1: t' in [t-1, t+2] hits t=2 -> supervise t in {0,1,2,3}."""
    m = resolve_body_supervision(2, 1, 0.0)(_win_T5_single())
    assert _active_at(m) == [True, True, True, True, False]


def test_threshold_is_a_live_input():
    w = _win_T5_single()  # value 5 at t=2
    assert _active_at(resolve_body_supervision(0, 0, 4.0)(w)) == [False, False, True, False, False]
    assert _active_at(resolve_body_supervision(0, 0, 5.0)(w)) == [False] * 5  # 5 not > 5


def test_never_active_cell_never_supervised_even_saturated():
    w = _win_T2()  # cell B (col 1) never active
    m = resolve_body_supervision(9, 9, 0.0)(w)
    assert m[0, :, 0, 0, 1].tolist() == [False, False]


# ---- event threshold: single authority = derivation config (C-195, unchanged) ----
def test_event_threshold_defaults_to_zero():
    assert event_threshold_from_config({}) == 0.0


def test_event_threshold_read_from_binary_derivation():
    cfg = {"derivations": {"binary": [{"from": "lr", "to": "by", "threshold": 3}]}}
    assert event_threshold_from_config(cfg) == 3.0


# ---- config boundary (ADR-009) ----------------------------------------------
def test_default_body_supervision_is_all(valid_config_dict):
    from views_hydranet.utils.config_initializer import HydraNetConfig

    assert "body_supervision" not in valid_config_dict
    c = HydraNetConfig(**valid_config_dict)
    assert c.body_supervision == "all"


@pytest.mark.parametrize("mode", ["all", "active"])
def test_accepts_all_and_active(valid_config_dict, mode):
    from views_hydranet.utils.config_initializer import HydraNetConfig

    c = HydraNetConfig(**{**valid_config_dict, "body_supervision": mode})
    assert c.body_supervision == mode


@pytest.mark.parametrize("bad", ["none", "pos_cells", "pos_timelines", "ALL", "windowed"])
def test_rejects_other_supervision_values(valid_config_dict, bad):
    from views_hydranet.utils.config_initializer import HydraNetConfig

    with pytest.raises(ValueError, match="body_supervision"):
        HydraNetConfig(**{**valid_config_dict, "body_supervision": bad})


@pytest.mark.parametrize("field", ["onset_lead", "cessation_lag"])
def test_radii_reject_negative(valid_config_dict, field):
    from views_hydranet.utils.config_initializer import HydraNetConfig

    with pytest.raises(ValueError, match=field):
        HydraNetConfig(**{**valid_config_dict, "body_supervision": "active", field: -1})


def test_retired_body_mask_key_fails_loud_with_migration_hint(valid_config_dict):
    """A config still setting the retired keyword must fail loud, never silently ignore."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    with pytest.raises(ValueError, match="body_mask.*retired|body_supervision"):
        HydraNetConfig(**{**valid_config_dict, "body_mask": "pos_cells"})


def test_active_with_latent_loss_raises(valid_config_dict):
    """C-193 retained: a positives window on a latent-likelihood body is a no-op -> raise."""
    from views_hydranet.utils.config_initializer import HydraNetConfig

    with pytest.raises(ValueError, match="latent likelihood"):
        HydraNetConfig(
            **{
                **valid_config_dict,
                "body_supervision": "active",
                "loss_reg": "tobit",
                "loss_reg_sigma": 1.0,
            }
        )
