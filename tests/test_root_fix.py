import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from views_hydranet.utils.config_initializer import HydraNetConfig
from views_hydranet.utils.feature_scaler import FeatureScaler

# BIT-PERFECT ROOT AUDIT STANDARDS
BASE_CONFIG = {
    "run_type": "calibration",
    "steps": [1, 2],
    "time_steps": 2,
    "input_channels": 3,
    "output_channels": 1,
    "regression_targets": ["lr_ged_sb"],
    "classification_targets": ["by_sb_best", "by_ns_best", "by_os_best"],
    "identity_cols": ["month_id", "priogrid_gid"],
    "features": ["lr_ged_sb", "lr_ged_ns", "lr_ged_os"],
    "transformations": {
        "log1p": ["lr_ged_sb"],
        "asinh": ["lr_ged_ns"],
        "identity": ["lr_ged_os"],
    },
    "derivations": {
        "binary": [
            {"from": "lr_ged_sb", "to": "by_sb_best", "threshold": 0},
            {"from": "lr_ged_ns", "to": "by_ns_best", "threshold": 0},
            {"from": "lr_ged_os", "to": "by_os_best", "threshold": 0},
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
    "window_dim": 2,
    "total_hidden_channels": 8,
    "dropout_rate": 0.0,
    "weight_init": "norm",
    "h_init": "zero",
    "learning_rate": 0.01,
    "weight_decay": 0.0,
    "windows_per_lesson": 1,
    "scheduler": "none",
    "warmup_steps": 1,
    "clip_grad_norm": True,
    "loss_reg": "mse",
    "loss_class": "bce",
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
    "sampling_strategy": "threshold",
    "evaluation_mode": "point",
    "aggregate_method": "arithmetic_mean",
}


def test_root_scaling_validation():
    """Assert that the scaler fails if a feature is missing from the 'transform' dict."""
    bad = BASE_CONFIG.copy()
    bad["transformations"] = {"log1p": ["lr_ged_sb"]}  # Missing ns and os
    with pytest.raises(ValidationError, match="Feature Lifecycle Violation"):
        HydraNetConfig(**bad)


def test_root_checksum_input_channels():
    """Assert input_channels checksum."""
    bad = BASE_CONFIG.copy()
    bad["input_channels"] = 99
    with pytest.raises(ValidationError, match="Checksum Law Violation: input_channels"):
        HydraNetConfig(**bad)


def test_scaler_root_consumption():
    """Assert FeatureScaler correctly uses the 'transform' dict."""
    scaler = FeatureScaler(BASE_CONFIG)
    df = pd.DataFrame({"lr_ged_sb": [10.0], "lr_ged_ns": [10.0], "lr_ged_os": [10.0]})
    semantic = scaler.fit_transform(df)
    # log1p(10) is ~2.39
    assert semantic["lr_ged_sb"].iloc[0] < 3.0

    recovered = scaler.inverse_transform(semantic)
    np.testing.assert_allclose(df["lr_ged_sb"], recovered["lr_ged_sb"])


def test_scaler_rejects_nan_input():
    """
    RED GATE: FeatureScaler.fit_transform() must raise if the input DataFrame
    contains NaN or Inf values in any configured feature column.

    Context: log1p(NaN) = NaN. asinh(Inf) = Inf. The scaler currently applies
    transforms silently, propagating poison values into the volume and then into
    the training loss. A NaN loss causes optimizer.step() to produce NaN weights,
    collapsing the model to random noise. This gate must fire BEFORE the transform
    touches the data.

    Taxonomy (ADR 005): RED — adversarial: silent data poisoning via missing values.
    """
    scaler = FeatureScaler(BASE_CONFIG)

    # Case A: NaN in a feature column
    df_nan = pd.DataFrame(
        {"lr_ged_sb": [1.0, float("nan")], "lr_ged_ns": [2.0, 3.0], "lr_ged_os": [4.0, 5.0]}
    )
    with pytest.raises(ValueError, match=r"[Nn]aN|[Nn]ull|[Mm]issing|[Pp]oison"):
        scaler.fit_transform(df_nan)

    # Case B: Inf in a feature column
    scaler_b = FeatureScaler(BASE_CONFIG)
    df_inf = pd.DataFrame(
        {"lr_ged_sb": [1.0, float("inf")], "lr_ged_ns": [2.0, 3.0], "lr_ged_os": [4.0, 5.0]}
    )
    with pytest.raises(ValueError, match=r"[Ii]nf|[Pp]oison|[Ii]nvalid"):
        scaler_b.fit_transform(df_inf)


def test_scaler_rejects_unmapped_columns():
    """
    RED GATE: FeatureScaler.fit_transform() must raise if a configured feature
    column is present in the DataFrame but absent from the 'transform' dict.

    Context: HydraNetConfig validates this at config-level, but the FeatureScaler
    boundary itself has no guard. If config validation is bypassed (e.g. in tests,
    during inference, or via direct instantiation), feat_b would pass through raw.
    Raw counts (e.g. battle deaths in hundreds) sent to the model cause gradient
    explosion. This gate must exist independently of config-level validation.

    Taxonomy (ADR 005): RED — adversarial bypass of the validation layer.
    """
    # Config omits 'feat_b' from transform — but feat_b IS in features list
    config = {"transformations": {"log1p": ["feat_a"]}, "features": ["feat_a", "feat_b"]}
    scaler = FeatureScaler(config)

    df = pd.DataFrame({"feat_a": [1.0, 2.0], "feat_b": [3.0, 4.0]})

    with pytest.raises(ValueError, match=r"feat_b|unmapped|transform"):
        scaler.fit_transform(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
