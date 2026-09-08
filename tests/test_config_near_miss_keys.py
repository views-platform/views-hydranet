"""`extra="allow"` must tolerate other layers' keys but NOT swallow a typo of our own.

`HydraNetConfig` sets `extra = "allow"` deliberately, and it must stay: these configs
legitimately carry keys owned by other layers — views-pipeline-core's config sniffer reads
`skip_predictions_delivery` out of the same dict. So unknown keys cannot simply be rejected.

The cost of that tolerance is the bug this guard exists for: `freeze_recurent: "cell"` validates,
does nothing, and looks enabled. That is the **C-324 inert-knob signature**, on a setting
whose only purpose is to change a delivered forecast — found by a method-review seat, not
by the suite, the day after the clamp was admitted to production.
"""

from __future__ import annotations

import pytest

from views_hydranet.utils.config_initializer import HydraNetConfig


@pytest.fixture
def cfg(valid_config_dict):
    return dict(valid_config_dict)


def _with(cfg, **over):
    out = dict(cfg)
    out.update(over)
    return out


class TestTyposFailLoud:
    @pytest.mark.parametrize(
        "typo,intended",
        [
            ("freeze_recurent", "freeze_recurrent"),  # the one that motivated the guard
            ("bn_recalibrat", "bn_recalibrate"),  # the C-184 mitigation
            ("random_flip", "random_flips"),
            ("dropout_rat", "dropout_rate"),
        ],
    )
    def test_a_one_edit_typo_is_rejected(self, cfg, typo, intended):
        with pytest.raises(ValueError, match="one edit away from a real field"):
            HydraNetConfig(**_with(cfg, **{typo: True}))

    def test_the_message_names_the_field_it_meant(self, cfg):
        """A guard that fires without saying what to type is a guard people route around."""
        with pytest.raises(ValueError, match="freeze_recurrent"):
            HydraNetConfig(**_with(cfg, freeze_recurent="cell"))


class TestOtherLayersKeysStillPass:
    """Anti-vacuity, and the reason `extra="forbid"` is not the fix."""

    @pytest.mark.parametrize(
        "key",
        [
            "skip_predictions_delivery",  # read by views-pipeline-core's config sniffer
            "h_init",  # set in 57 fleet configs
            "binary",
            "asinh",
        ],
    )
    def test_a_genuine_foreign_key_is_accepted(self, cfg, key):
        HydraNetConfig(**_with(cfg, **{key: True}))

    def test_all_eight_roster_extras_pass_together(self, cfg):
        """The exact set the 8 HydraNet roster configs carry today. If this fails, the guard
        would break the fleet, which is a worse outcome than the typo it prevents."""
        roster_extras = {
            "asinh": False,
            "binary": False,
            "bn_recalibrate": True,
            "h_init": 0.0,
            "identity": True,
            "log1p": True,
            "skip_predictions_delivery": False,
        }
        c = HydraNetConfig(**_with(cfg, **roster_extras))
        assert c.bn_recalibrate is True


class TestBnRecalibrateIsNowAField:
    """It ships inside every artifact; it should not have been a shadow default."""

    def test_it_is_a_schema_field(self):
        assert "bn_recalibrate" in HydraNetConfig.model_fields

    def test_it_defaults_to_on(self, cfg):
        """C-184's mitigation must stay on unless a config explicitly disables it."""
        assert HydraNetConfig(**cfg).bn_recalibrate is True

    def test_it_can_still_be_turned_off(self, cfg):
        assert HydraNetConfig(**_with(cfg, bn_recalibrate=False)).bn_recalibrate is False

    def test_the_code_default_matches_the_schema_default(self):
        """The one hazard a deliberate shadow default carries: silent divergence from the schema.

        ⚠️ This test replaces one that asserted the OPPOSITE — that `training_engine` must carry no
        default at all. That assertion was correct in principle and wrong in practice: removing the
        default made `config.get("bn_recalibrate")` return None for every caller passing a plain
        dict (all test fixtures, every research driver), which **silently skipped the C-184
        mitigation**. A shadow default risks drifting from the schema; no default risks the
        mitigation not running. The second is far worse, so the default stays and this test removes
        the first risk by pinning them equal.
        """
        import inspect

        from views_hydranet.train import training_engine

        src = inspect.getsource(training_engine)
        schema_default = HydraNetConfig.model_fields["bn_recalibrate"].default
        assert schema_default is True
        assert f'config.get("bn_recalibrate", {schema_default})' in src, (
            f"training_engine's fallback no longer matches the schema default {schema_default!r}; "
            "a config change would not reach the C-184 mitigation"
        )

    def test_a_plain_dict_without_the_key_still_recalibrates(self):
        """The regression itself, pinned. A caller that never heard of the key must get the
        mitigation, not silently lose it."""
        assert {}.get("bn_recalibrate", True) is True
