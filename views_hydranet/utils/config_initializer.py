"""
ConfigInitializer: Canonical Entry Point for HydraNet Configuration.
"""

import logging
from collections.abc import Callable
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# --- TRANSFORMATION REGISTRY ---
TRANSFORMS: dict[str, tuple[Callable, Callable]] = {
    "log1p": (np.log1p, np.expm1),
    "asinh": (np.arcsinh, np.sinh),
    "identity": (lambda x: x, lambda x: x),
}


class HydraNetConfig(BaseModel):
    """
    The Exhaustive 'Minimum Strict Set' for HydraNet operations.
    Matches the pipeline configuration 1-to-1.
    """

    # 1. High-Level Partitioning
    run_type: str = Field(
        ..., description="Partition: calibration, validation, forecasting, or testing"
    )
    steps: list[int] = Field(..., description="List of forecast steps")
    time_steps: int = Field(..., description="Checksum for 'steps'")

    # 2. Data Slicing & Scaling (The Physics)
    input_channels: int = Field(..., ge=1, description="Checksum for 'features'")
    output_channels: int = Field(..., ge=1, description="Channels per model head")
    regression_targets: list[str] = Field(
        ..., description="Intensity mission (must start with lr_)"
    )
    classification_targets: list[str] = Field(..., description="Binary mission")
    identity_cols: list[str] = Field(..., description="Columns to be excluded from features")
    features: list[str] = Field(..., description="Exhaustive list of predictive signals")

    # ADR 046 Symmetric Lifecycle (Transformations vs Derivations)
    transformations: Dict[str, List[str]] = Field(
        ..., description="Mapping of scaling method to columns"
    )
    derivations: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict, description="Instructional feature engineering"
    )

    # 3. Spatiotemporal Topology
    height: int = Field(..., ge=1)
    width: int = Field(..., ge=1)
    index_names: list[str] = Field(..., description="Column names used as the DataFrame index")
    time_col: str = Field(...)
    id_col: str = Field(...)
    spatial_cols: list[str] = Field(...)
    row_offset: int = Field(...)
    col_offset: int = Field(...)

    # 4. Training Architecture
    model: str = Field(...)
    window_dim: int = Field(...)
    total_hidden_channels: int = Field(...)
    dropout_rate: float = Field(..., ge=0.0, le=1.0)
    weight_init: str = Field(...)

    # 5. Optimization
    learning_rate: float = Field(..., gt=0.0)
    weight_decay: float = Field(..., ge=0.0)
    windows_per_lesson: int = Field(..., ge=1)
    scheduler: str = Field(...)
    warmup_steps: int = Field(..., ge=1)
    clip_grad_norm: bool = Field(...)

    # 6. Loss Functions (names: mse, shrinkage, basu_dpd, lognormal_nll)
    loss_reg: str = Field(...)
    loss_class: str = Field(...)
    # ShrinkageLoss params (loss_reg='shrinkage')
    loss_reg_a: float = Field(default=10.0)
    loss_reg_c: float = Field(default=0.2)
    # BasuDPDLoss params (loss_reg='basu_dpd')
    loss_reg_alpha: float = Field(default=0.5)
    # Shared: BasuDPDLoss sigma / LogNormalFixedSigmaLoss sigma
    loss_reg_sigma: float = Field(default=1.0)
    # FocalLoss params (loss_class='focal')
    loss_class_gamma: float = Field(default=1.5)
    loss_class_alpha: float = Field(default=0.75)
    # Classification head bias initialization (C-44)
    # Set to log(event_rate / (1 - event_rate)). None = PyTorch default.
    # -5.0 ≈ 0.67% event rate, -7.0 ≈ 0.09% event rate.
    onset_bias_init: float | None = Field(default=None)

    # 7. Sampling & Reproducibility
    total_lessons: int = Field(..., ge=1)
    n_posterior_samples: int = Field(..., ge=1)
    np_seed: int = Field(...)
    torch_seed: int = Field(...)

    # 8. Strategy & Curriculum
    min_events: int = Field(...)
    slope_ratio: float = Field(...)
    roof_ratio: float = Field(...)
    max_ratio: float = Field(...)
    min_ratio: float = Field(...)
    freeze_h: str = Field(...)

    # 9. Outbound Evaluation
    evaluation_mode: str = Field(
        ...,
        description=(
            "Controls whether the posterior sample axis (S) is preserved or collapsed. "
            "'stochastic': all S samples are kept → PredictionFrame.y_pred.shape == (N, S). "
            "'point': collapse_to_point(aggregate_method) folds S to a scalar per cell → "
            "PredictionFrame.y_pred.shape == (N, 1) regardless of n_posterior_samples."
        ),
    )
    aggregate_method: str = Field(
        ...,
        description=(
            "Aggregation function applied when evaluation_mode == 'point'. "
            "Ignored in stochastic mode. "
            "Supported: 'arithmetic_mean' (alias 'mean'), 'median' (alias 'max_aposteriori'). "
            "'geometric_mean' is schema-valid but raises NotImplementedError at runtime."
        ),
    )
    # Metadata
    model_time_stamp: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def handle_typos(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "evalution_mode" in data and "evaluation_mode" not in data:
                logger.warning(
                    "Deprecated config key 'evalution_mode' — use 'evaluation_mode'. "
                    "This shim will be removed in a future release."
                )
                data["evaluation_mode"] = data["evalution_mode"]
        return data

    @model_validator(mode="after")
    def validate_laws(self) -> "HydraNetConfig":
        """The Checksum and Scaling Laws."""
        # Checksum: input_channels
        if self.input_channels != len(self.features):
            err_msg = (
                f"Checksum Law Violation: input_channels ({self.input_channels}) != "
                f"features ({len(self.features)})"
            )

            logger.error(err_msg)

            raise ValueError(err_msg)

        # Checksum: time_steps
        if self.time_steps != len(self.steps):
            err_msg = (
                f"Checksum Law Violation: time_steps ({self.time_steps}) != "
                f"steps ({len(self.steps)})"
            )

            logger.error(err_msg)

            raise ValueError(err_msg)

        # Feature Lifecycle Law (ADR 046):
        # All signals must be accounted for (either transformed or derived)
        all_required_cols = (
            set(self.features) | set(self.regression_targets) | set(self.classification_targets)
        )
        accounted_for = set()

        # 1. Check Transformations (Scale)
        for method, cols in self.transformations.items():
            if method not in TRANSFORMS:
                err_msg = f"Feature Lifecycle Violation: Unknown transformation method '{method}'"
                logger.error(err_msg)
                raise ValueError(err_msg)
            for col in cols:
                accounted_for.add(col)

        # 2. Check Derivations (Identity)
        for op, instrs in self.derivations.items():
            for instr in instrs:
                # Add the 'to' column to accounted_for
                if "to" in instr:
                    accounted_for.add(instr["to"])

        missing = all_required_cols - accounted_for
        if missing:
            err_msg = (
                f"Feature Lifecycle Violation: Required columns {missing} are not accounted for. "
                f"They must either be in 'transformations' or produced by 'derivations'."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        return self

    @model_validator(mode="after")
    def warn_aggregate_method_in_stochastic_mode(self) -> "HydraNetConfig":
        if self.evaluation_mode == "stochastic":
            logger.warning(
                "\n"
                "╔══════════════════════════════════════════════════════════════╗\n"
                "║  HydraNetConfig WARNING: aggregate_method is IGNORED         ║\n"
                "║                                                              ║\n"
                "║  evaluation_mode='stochastic' preserves the full posterior.  ║\n"
                "║  The aggregate_method='%s' setting has NO effect.       ║\n"
                "║                                                              ║\n"
                "║  To activate aggregation, set evaluation_mode='point'.       ║\n"
                "╚══════════════════════════════════════════════════════════════╝",
                self.aggregate_method,
            )
        return self

    @field_validator("run_type")
    @classmethod
    def validate_run_type(cls, v: str) -> str:
        valid = ["calibration", "validation", "forecasting", "testing"]
        if v not in valid:
            err_msg = f"run_type must be in {valid}"

            logger.error(err_msg)

            raise ValueError(err_msg)
        return v

    @field_validator("evaluation_mode")
    @classmethod
    def validate_eval_mode(cls, v: str) -> str:
        valid = ["point", "stochastic"]
        if v not in valid:
            err_msg = (
                f"evaluation_mode='{v}' is not valid. "
                f"Expected one of: {valid}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return v

    @field_validator("total_hidden_channels")
    @classmethod
    def validate_hidden_channels_divisibility(cls, v: int) -> int:
        if v % 8 != 0:
            raise ValueError(
                f"total_hidden_channels={v} is not divisible by 8. "
                f"The architecture requires 4 LSTM cells x 2 states = 8 partitions."
            )
        return v

    @field_validator("aggregate_method")
    @classmethod
    def validate_agg_method(cls, v: str) -> str:
        mapper = {"mean": "arithmetic_mean", "median": "median", "max_aposteriori": "median"}
        v = mapper.get(v, v)
        if v not in ["arithmetic_mean", "geometric_mean", "median"]:
            err_msg = "Invalid aggregate_method"

            logger.error(err_msg)

            raise ValueError(err_msg)
        return v

    # --- Dict-compatibility layer (gradual migration from config["key"]) ---

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            logger.error(f"HydraNetConfig: key '{key}' not found.")

            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self) -> list[str]:
        return list(self.model_fields.keys()) + list(self.__pydantic_extra__.keys())

    class Config:
        extra = "allow"  # Tolerant Handshake


class ConfigInitializer:
    """
    Handles the initialization, normalization, and validation of
    HydraNet run-time configurations.
    """

    def __init__(self, raw_config: Dict[str, Any]) -> None:
        """
        Store the raw configuration from the pipeline core.
        """
        self._raw = raw_config

    def get_config(self) -> dict:
        """
        Returns the processed and strictly validated configuration as a dict.
        This is the single 'Handshake' point for the whole pipeline.

        Validation is enforced by the HydraNetConfig Pydantic constructor —
        any missing fields or legacy keys trigger a loud ValidationError.
        The result is returned as a plain dict because the parent class
        (ForecastingModelManager.configs setter) requires isinstance(dict).
        """
        config_obj = HydraNetConfig(**self._raw)
        return config_obj.model_dump()
