"""
TDD — RED tests for Item 4: ConfigInitializer.get_config() should return
HydraNetConfig (typed Pydantic object), NOT a plain dict.

Attribute access (config.evaluation_mode) is safer than dict access
(config["evaluation_mode"]) — typos are caught by the IDE and at import time,
not silently at runtime hour 6.
"""


from views_hydranet.utils.config_initializer import ConfigInitializer, HydraNetConfig

# Minimal valid config that satisfies all HydraNetConfig validators
MINIMAL_CONFIG = {
    "run_type": "calibration",
    "steps": [1, 2],
    "time_steps": 2,
    "input_channels": 3,
    "output_channels": 1,
    "regression_targets": ["lr_sb_best"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "identity_cols": ["month_id", "priogrid_gid"],
    "features": ["lr_sb_best", "lr_ns_best", "lr_os_best"],
    "transformations": {
        "log1p": ["lr_sb_best"],
        "asinh": ["lr_ns_best"],
        "identity": ["lr_os_best"],
    },
    "derivations": {
        "binary": [
            {"from": "lr_sb_best", "to": "by_sb_best", "threshold": 0},
            {"from": "lr_ns_best", "to": "by_ns_best", "threshold": 0},
            {"from": "lr_os_best", "to": "by_os_best", "threshold": 0},
        ]
    },
    "height": 4,
    "width": 4,
    "index_names": ["month_id", "priogrid_gid"],
    "time_col": "month_id",
    "id_col": "priogrid_gid",
    "spatial_cols": ["row", "col"],
    "row_offset": 0,
    "col_offset": 0,
    "model": "Dummy",
    "window_dim": 1,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "norm",
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "windows_per_lesson": 1,
    "scheduler": "none",
    "warmup_steps": 1,
    "clip_grad_norm": True,
    "loss_reg": "lr_b",
    "loss_class": "lr_b",
    "loss_reg_a": 1,
    "loss_reg_c": 1,
    "loss_class_gamma": 1,
    "loss_class_alpha": 1,
    "total_lessons": 1,
    "n_posterior_samples": 1,
    "np_seed": 1,
    "torch_seed": 1,
    "min_events": 0,
    "slope_ratio": 0.1,
    "roof_ratio": 0.1,
    "max_ratio": 0.9,
    "min_ratio": 0.1,
    "freeze_h": "none",
    "evaluation_mode": "point",
    "aggregate_method": "mean",
}


class TestConfigInitializerReturnsTypedObject:
    """RED GATE: get_config() must return HydraNetConfig, not dict."""

    def test_get_config_returns_hydranet_config(self):
        """The return type must be HydraNetConfig so consumers use attribute access."""
        ci = ConfigInitializer(MINIMAL_CONFIG)
        result = ci.get_config()
        assert isinstance(result, HydraNetConfig), (
            f"ConfigInitializer.get_config() returned {type(result).__name__}, "
            f"expected HydraNetConfig. Pass the typed object, not model_dump()."
        )

    def test_attribute_access_works(self):
        """Consumers should access config.evaluation_mode, not config['evaluation_mode']."""
        ci = ConfigInitializer(MINIMAL_CONFIG)
        result = ci.get_config()
        assert result.evaluation_mode == "point"
        assert result.aggregate_method == "arithmetic_mean"  # alias resolved
        assert result.run_type == "calibration"

    def test_extra_fields_preserved(self):
        """Extra fields (Config.extra='allow') must survive the typed return."""
        extended = {**MINIMAL_CONFIG, "custom_field": "hello"}
        ci = ConfigInitializer(extended)
        result = ci.get_config()
        assert result.custom_field == "hello"

    def test_regression_targets_is_list(self):
        """Typed access should preserve list types without needing dict unpacking."""
        ci = ConfigInitializer(MINIMAL_CONFIG)
        result = ci.get_config()
        assert isinstance(result.regression_targets, list)
        assert result.regression_targets == ["lr_sb_best"]
