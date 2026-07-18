"""
Falsification stubs for PR #34 (strict conditional parameter validation).

F5: CIC §3 field count drift — guard against future miscounts.
"""

from views_hydranet.utils.config_initializer import HydraNetConfig


class TestF5_CICFieldCountDrift:
    """CIC §3 field count must stay in sync with actual model_fields."""

    def test_cic_field_count_matches_actual(self):
        """CIC §3 field count must match actual model_fields count."""
        # docs/CICs/HydraNetConfig.md §3 (freeze_h -1; rollout_horizon +1, 2026-06-06;
        # hurdle-NB +3: loss_reg_theta_init, learnable_theta, loss_class_pos_weight; #99;
        # output_distribution +1, #100; min_free_disk_gb +1, #107/C-154;
        # static_channels +1, #108/ADR-060/C-153;
        # reg_activation +1, 2026-06-21 Exp-B head decouple;
        # quantile head +n_quantiles, pinball body +loss_reg_tau/loss_reg_cap;
        # ADR-065 / Epic #158: +body_mask, -hurdle_threshold retired -> net 78 => 81)
        CIC_CLAIMED_COUNT = 81
        actual = len(HydraNetConfig.model_fields)
        assert actual == CIC_CLAIMED_COUNT, (
            f"CIC §3 claims {CIC_CLAIMED_COUNT} fields but HydraNetConfig has {actual}. "
            f"Update CIC §3 field count after adding/removing fields."
        )
