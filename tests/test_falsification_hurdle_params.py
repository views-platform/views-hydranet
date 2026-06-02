"""
Falsification audit: ADR-050 hurdle parameter hardening.

Red tests from falsification probes 1, 2, and 4 — these guards were
missing and have now been implemented. Tests must PASS (not xfail).
"""

import pytest


class TestRedQS99Range:
    """Probes 1 & 2: qs99_weight and qs99_tau must reject nonsensical values."""

    def test_negative_qs99_weight_rejected(self, valid_config_dict):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = {
            **valid_config_dict,
            "hurdle_threshold": 0.0,
            "qs99_weight": -0.5,
            "qs99_tau": 0.99,
        }
        with pytest.raises((ValueError, Exception)):
            HydraNetConfig(**cfg)

    def test_qs99_tau_above_one_rejected(self, valid_config_dict):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = {
            **valid_config_dict,
            "hurdle_threshold": 0.0,
            "qs99_weight": 0.1,
            "qs99_tau": 5.0,
        }
        with pytest.raises((ValueError, Exception)):
            HydraNetConfig(**cfg)

    def test_qs99_tau_negative_rejected(self, valid_config_dict):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = {
            **valid_config_dict,
            "hurdle_threshold": 0.0,
            "qs99_weight": 0.1,
            "qs99_tau": -1.0,
        }
        with pytest.raises((ValueError, Exception)):
            HydraNetConfig(**cfg)


class TestRedBasuDegenerate:
    """Probe 4: ADR-050 §5 — Basu alpha=0/sigma=0 must raise at config time."""

    def test_basu_alpha_zero_rejected(self, valid_config_dict):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = {
            **valid_config_dict,
            "loss_reg": "basu_dpd",
            "loss_reg_alpha": 0.0,
            "loss_reg_sigma": 1.0,
        }
        with pytest.raises((ValueError, Exception)):
            HydraNetConfig(**cfg)

    def test_basu_sigma_zero_rejected(self, valid_config_dict):
        from views_hydranet.utils.config_initializer import HydraNetConfig

        cfg = {
            **valid_config_dict,
            "loss_reg": "basu_dpd",
            "loss_reg_alpha": 0.5,
            "loss_reg_sigma": 0.0,
        }
        with pytest.raises((ValueError, Exception)):
            HydraNetConfig(**cfg)
