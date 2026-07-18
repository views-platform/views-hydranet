"""S3 (#161) — config-layer tests for the `body_mask` field + fail-loud validators (ADR-065).

Scope is the CONFIG boundary only (ADR-009): the field exists, accepts the three values, rejects
anything else, defaults to 'none' (today's foundation), and raises when a positives mask is paired
with a latent-likelihood body where it would be a silent no-op (C-193). Wiring into the training
loop is S4; retiring the legacy knobs is S5 — neither is exercised here.

Note on the C-193 coupling: the single authority for "is this loss latent?" is the loss class's
`needs_latent` flag (LOSS_REG_REGISTRY), not a hardcoded name list. `lognormal_nll` is
needs_latent=False — a positive-only POINT body (lognormal NLL is undefined at y=0) that REQUIRES a
positives mask — so pos_* + lognormal_nll is ALLOWED, while tobit/hurdle_nb/dense_nb are rejected.
"""

import pytest

from views_hydranet.utils.config_initializer import BODY_MASK_VALUES, HydraNetConfig


def _cfg(valid_config_dict, **overrides):
    return HydraNetConfig(**{**valid_config_dict, **overrides})


# ---- field: valid values, default, backward-compat --------------------------


def test_default_is_none_when_field_absent(valid_config_dict):
    """Backward-compat: an existing config with no body_mask validates and defaults to all-cell."""
    assert "body_mask" not in valid_config_dict
    assert _cfg(valid_config_dict).body_mask == "none"


@pytest.mark.parametrize("value", ["none", "pos_cells", "pos_timelines"])
def test_accepts_the_three_values(valid_config_dict, value):
    assert _cfg(valid_config_dict, body_mask=value).body_mask == value


def test_the_three_values_are_exactly_the_authority_constant():
    assert BODY_MASK_VALUES == ("none", "pos_cells", "pos_timelines")


@pytest.mark.parametrize("bad", ["positives", "per_step", "active_window", "all", "POS_CELLS"])
def test_rejects_any_other_value(valid_config_dict, bad):
    """ADR-009: a typo must fail loud, never silently degrade to a different mask (C-194)."""
    with pytest.raises(ValueError, match="body_mask"):
        _cfg(valid_config_dict, body_mask=bad)


# ---- model validator: C-193 latent coupling (fail-loud) ---------------------


@pytest.mark.parametrize("mask", ["pos_cells", "pos_timelines"])
def test_pos_mask_with_latent_hurdle_nb_raises(valid_config_dict, mask):
    with pytest.raises(ValueError, match="latent likelihood"):
        _cfg(valid_config_dict, body_mask=mask, loss_reg="hurdle_nb", loss_reg_theta_init=1.0)


def test_pos_mask_with_latent_tobit_raises(valid_config_dict):
    with pytest.raises(ValueError, match="latent likelihood"):
        _cfg(valid_config_dict, body_mask="pos_cells", loss_reg="tobit", loss_reg_sigma=1.0)


def test_none_with_latent_loss_is_allowed(valid_config_dict):
    """none IS the all-cell case — legal with a latent body (the production floor shape)."""
    cfg = _cfg(valid_config_dict, body_mask="none", loss_reg="hurdle_nb", loss_reg_theta_init=1.0)
    assert cfg.body_mask == "none"


def test_pos_mask_with_point_mse_is_allowed(valid_config_dict):
    assert _cfg(valid_config_dict, body_mask="pos_cells", loss_reg="mse").body_mask == "pos_cells"


def test_pos_mask_with_lognormal_is_allowed_point_body(valid_config_dict):
    """lognormal_nll is needs_latent=False: a positive-only point body that NEEDS a pos mask."""
    cfg = _cfg(
        valid_config_dict, body_mask="pos_cells", loss_reg="lognormal_nll", loss_reg_sigma=1.0
    )
    assert cfg.body_mask == "pos_cells"
