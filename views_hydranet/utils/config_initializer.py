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
    loss_reg_a: float | None = Field(default=None)
    loss_reg_c: float | None = Field(default=None)
    # BasuDPDLoss params (loss_reg='basu_dpd')
    loss_reg_alpha: float | None = Field(default=None)
    # Shared: BasuDPDLoss sigma / LogNormalFixedSigmaLoss sigma
    loss_reg_sigma: float | None = Field(default=None)
    # ParetoLoss params (loss_reg='pareto')
    loss_reg_pareto_alpha: float | None = Field(default=None)
    # FocalLoss params (loss_class='focal')
    loss_class_gamma: float | None = Field(default=None)
    loss_class_alpha: float | None = Field(default=None)
    # Classification head bias initialization (C-44)
    onset_bias_init: float | None = Field(default=None)
    # Hurdle masking (C-45): None = disabled, 0.0 = standard hurdle (y > 0)
    hurdle_threshold: float | None = Field(default=None)
    # QS99 tail regularizer (C-48): strict when hurdle active + weight > 0.
    qs99_weight: float | None = Field(default=None, ge=0.0)
    qs99_tau: float | None = Field(default=None, gt=0.0, lt=1.0)
    # Per-target regression loss weights (C-87): None = uniform.
    target_weights: Dict[str, float] | None = Field(default=None)

    # 7. Sampling & Reproducibility
    total_lessons: int = Field(..., ge=1)
    n_posterior_samples: int = Field(..., ge=1)
    np_seed: int = Field(...)
    torch_seed: int = Field(...)
    # Sampling strategy (ADR-049): must be explicit — no hidden defaults.
    sampling_strategy: str = Field(...)
    sampling_alpha: float | None = Field(default=None)
    sampling_temperature: float | None = Field(default=None)
    sampling_steepness: float | None = Field(default=None)

    # 8. Strategy & Curriculum
    min_events: int = Field(...)
    slope_ratio: float = Field(...)
    roof_ratio: float = Field(...)
    max_ratio: float = Field(...)
    min_ratio: float = Field(...)
    freeze_h: str = Field(...)

    # 9. Runtime Flags
    sweep: bool = Field(default=False)
    random_flips: bool = Field(default=True)
    diagnostic_visualizations: bool = Field(default=False)

    # 10. Outbound Evaluation
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
    def handle_typos_and_missing_guidance(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "evalution_mode" in data and "evaluation_mode" not in data:
                logger.warning(
                    "Deprecated config key 'evalution_mode' — use 'evaluation_mode'. "
                    "This shim will be removed in a future release."
                )
                data["evaluation_mode"] = data["evalution_mode"]

            # For fields with registry/enum semantics, inject a sentinel so that
            # Pydantic continues validating ALL fields (collecting every error)
            # rather than stopping at the first missing one. The sentinel then
            # hits the field_validator which produces an informative error.
            _SENTINEL_FIELDS = {
                "sampling_strategy": "__MISSING_sampling_strategy__",
                "run_type": "__MISSING_run_type__",
                "loss_reg": "__MISSING_loss_reg__",
                "loss_class": "__MISSING_loss_class__",
                "evaluation_mode": "__MISSING_evaluation_mode__",
                "aggregate_method": "__MISSING_aggregate_method__",
            }
            for field, sentinel in _SENTINEL_FIELDS.items():
                if field not in data:
                    data[field] = sentinel
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
            prefix = "'run_type' is required." if v.startswith("__MISSING_") else f"run_type='{v}'"
            err_msg = f"{prefix} Valid options: {valid}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        return v

    @field_validator("evaluation_mode")
    @classmethod
    def validate_eval_mode(cls, v: str) -> str:
        valid = ["point", "stochastic"]
        if v not in valid:
            prefix = (
                "'evaluation_mode' is required."
                if v.startswith("__MISSING_")
                else f"evaluation_mode='{v}' is not valid."
            )
            err_msg = f"{prefix} Expected one of: {valid}."
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

    @field_validator("loss_reg")
    @classmethod
    def validate_loss_reg(cls, v: str) -> str:
        from views_hydranet.utils.utils import LOSS_REG_REGISTRY

        if v not in LOSS_REG_REGISTRY:
            prefix = "'loss_reg' is required." if v.startswith("__MISSING_") else f"loss_reg='{v}'"
            err_msg = f"{prefix} Available: {list(LOSS_REG_REGISTRY.keys())}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        return v

    @field_validator("loss_class")
    @classmethod
    def validate_loss_class(cls, v: str) -> str:
        from views_hydranet.utils.utils import LOSS_CLASS_REGISTRY

        if v not in LOSS_CLASS_REGISTRY:
            prefix = (
                "'loss_class' is required." if v.startswith("__MISSING_") else f"loss_class='{v}'"
            )
            err_msg = f"{prefix} Available: {list(LOSS_CLASS_REGISTRY.keys())}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        return v

    @field_validator("sampling_strategy")
    @classmethod
    def validate_sampling_strategy(cls, v: str) -> str:
        from views_hydranet.utils.sampling_strategies import SAMPLING_STRATEGY_REGISTRY

        if v.startswith("__MISSING_"):
            strategies = list(SAMPLING_STRATEGY_REGISTRY.keys())
            params = {
                k: entry["params"]
                for k, entry in SAMPLING_STRATEGY_REGISTRY.items()
                if entry["params"]
            }
            err_msg = (
                f"'sampling_strategy' is required (ADR-049). "
                f"Valid strategies: {strategies}. "
                f"Strategy-specific params (also required): {params}. "
                f"To preserve current behaviour, add: "
                f"sampling_strategy='threshold' (no extra params needed)."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        if v not in SAMPLING_STRATEGY_REGISTRY:
            err_msg = (
                f"sampling_strategy='{v}' is not registered. "
                f"Available: {list(SAMPLING_STRATEGY_REGISTRY.keys())}"
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return v

    @field_validator("aggregate_method")
    @classmethod
    def validate_agg_method(cls, v: str) -> str:
        valid = ["arithmetic_mean", "geometric_mean", "median"]
        mapper = {"mean": "arithmetic_mean", "median": "median", "max_aposteriori": "median"}
        v = mapper.get(v, v)
        if v not in valid:
            prefix = (
                "'aggregate_method' is required."
                if v.startswith("__MISSING_")
                else f"aggregate_method='{v}' is not valid."
            )
            aliases = "'mean' → 'arithmetic_mean', 'max_aposteriori' → 'median'"
            err_msg = f"{prefix} Expected one of: {valid}. Aliases: {aliases}."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return v

    @field_validator("slope_ratio")
    @classmethod
    def validate_slope_ratio(cls, v: float) -> float:
        if v <= 0.0:
            err_msg = (
                f"slope_ratio must be > 0.0 (got {v}). Zero causes division-by-zero in curriculum."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return v

    @field_validator("roof_ratio")
    @classmethod
    def validate_roof_ratio(cls, v: float) -> float:
        if v <= 0.0:
            err_msg = f"roof_ratio must be > 0.0 (got {v}). Zero eliminates curriculum variation."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return v

    @field_validator("window_dim")
    @classmethod
    def validate_window_dim(cls, v: int) -> int:
        if v < 2:
            err_msg = (
                f"window_dim must be >= 2 (got {v}). Single-pixel patches have no spatial context."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return v

    @model_validator(mode="after")
    def validate_ratio_ordering(self) -> "HydraNetConfig":
        if self.min_ratio >= self.max_ratio:
            err_msg = (
                f"min_ratio ({self.min_ratio}) must be < max_ratio ({self.max_ratio}). "
                f"Inverted range breaks curriculum sampling."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def validate_sampling_params(self) -> "HydraNetConfig":
        from views_hydranet.utils.sampling_strategies import SAMPLING_STRATEGY_REGISTRY

        entry = SAMPLING_STRATEGY_REGISTRY.get(self.sampling_strategy)
        if entry is None:
            return self
        for param in entry["params"]:
            if getattr(self, param) is None:
                err_msg = (
                    f"sampling_strategy='{self.sampling_strategy}' requires '{param}' "
                    f"but it was not provided. Add '{param}' to your config."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def validate_loss_reg_params(self) -> "HydraNetConfig":
        from views_hydranet.utils.utils import LOSS_REG_REGISTRY

        entry = LOSS_REG_REGISTRY.get(self.loss_reg)
        if entry is None:
            return self
        for param in entry["params"]:
            if getattr(self, param) is None:
                err_msg = (
                    f"loss_reg='{self.loss_reg}' requires '{param}' "
                    f"but it was not provided. Add '{param}' to your config."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def validate_basu_dpd_range(self) -> "HydraNetConfig":
        if self.loss_reg != "basu_dpd":
            return self
        if self.loss_reg_alpha is not None and self.loss_reg_alpha <= 0:
            err_msg = (
                f"loss_reg='basu_dpd' requires loss_reg_alpha > 0, "
                f"got {self.loss_reg_alpha}. alpha=0 degenerates to MLE "
                f"(no robustness), alpha < 0 is undefined."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        if self.loss_reg_sigma is not None and self.loss_reg_sigma <= 0:
            err_msg = (
                f"loss_reg='basu_dpd' requires loss_reg_sigma > 0, "
                f"got {self.loss_reg_sigma}. sigma=0 causes division by zero "
                f"in the density power divergence formula."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def validate_loss_class_params(self) -> "HydraNetConfig":
        from views_hydranet.utils.utils import LOSS_CLASS_REGISTRY

        entry = LOSS_CLASS_REGISTRY.get(self.loss_class)
        if entry is None:
            return self
        for param in entry["params"]:
            if getattr(self, param) is None:
                err_msg = (
                    f"loss_class='{self.loss_class}' requires '{param}' "
                    f"but it was not provided. Add '{param}' to your config."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def validate_hurdle_params(self) -> "HydraNetConfig":
        if (
            self.hurdle_threshold is not None
            and self.qs99_weight is not None
            and self.qs99_weight > 0
        ):
            if self.qs99_tau is None:
                err_msg = (
                    f"hurdle_threshold={self.hurdle_threshold} with "
                    f"qs99_weight={self.qs99_weight} requires 'qs99_tau' "
                    f"but it was not provided. Add 'qs99_tau' to your config."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def validate_target_weights(self) -> "HydraNetConfig":
        if self.target_weights is None:
            return self
        for w in self.target_weights.values():
            if w < 0:
                err_msg = f"target_weights values must be >= 0, got {self.target_weights}."
                logger.error(err_msg)
                raise ValueError(err_msg)
        missing = [t for t in self.regression_targets if t not in self.target_weights]
        if missing:
            err_msg = (
                f"target_weights is missing entries for regression targets: "
                f"{missing}. All regression_targets must have a weight."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return self

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
