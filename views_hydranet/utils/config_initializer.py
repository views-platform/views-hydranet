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

# --- BODY-MASK VALUES (ADR-065, Epic #158) ---
# The single validated front door for how the POINT body is masked during training. Single
# authority for the allowed keywords (reused by the S4 resolver). 'none' = the all-cell foundation.
BODY_MASK_VALUES: tuple[str, ...] = ("none", "pos_cells", "pos_timelines")

# authority for the LEGACY output_distribution values (ADR-067). The valid set is this tuple
# UNIONED with the registered distribution-family names (`family_names()`); the two must stay
# disjoint (C-197) so a legacy config never silently routes to a new family.
LEGACY_OUTPUT_DISTRIBUTIONS: tuple[str, ...] = (
    "standard",
    "hurdle_nb",
    "hurdle_lognormal",
    "hurdle_shrinkage",
    "dense_nb",
    "quantile",
)


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
    n_quantiles: int | None = Field(
        default=None, ge=2, description="Quantile head: K monotone quantiles per reg target (K>=2)"
    )
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

    # 6. Loss Functions (names: mse, shrinkage, lognormal_nll, tobit)
    loss_reg: str = Field(...)
    loss_class: str = Field(...)
    # ShrinkageLoss params (loss_reg='shrinkage')
    loss_reg_a: float | None = Field(default=None)
    loss_reg_c: float | None = Field(default=None)
    # Shared: LogNormalFixedSigmaLoss / TobitLoss sigma
    # Per-target Tobit (issue #44): dict[str, float] maps regression target → sigma.
    loss_reg_sigma: float | Dict[str, float] | None = Field(default=None)
    # ParetoLoss params (loss_reg='pareto')
    loss_reg_pareto_alpha: float | None = Field(default=None)
    # FocalLoss params (loss_class='focal')
    loss_class_gamma: float | None = Field(default=None)
    loss_class_alpha: float | None = Field(default=None)
    # Classification head bias initialization (C-44)
    onset_bias_init: float | None = Field(default=None)
    # body_mask (ADR-065, Epic #158): the single validated front door for the POINT-body training
    # mask. 'none' = all cells (default; foundation); 'pos_cells' = per-step positives (y>thr);
    # 'pos_timelines' = cells active anywhere in the curriculum window (active_window decay mask).
    # S4 resolves it to the mask mechanism; the legacy hurdle_threshold/hurdle_mask_mode knobs
    # retire in S5. Validated by validate_body_mask (values) + validate_body_mask_latent (C-193).
    body_mask: str = Field(default="none")
    # QS99 tail regularizer (C-48): strict when hurdle active + weight > 0.
    qs99_weight: float | None = Field(default=None, ge=0.0)
    qs99_tau: float | None = Field(default=None, gt=0.0, lt=1.0)
    # Per-target regression loss weights (C-87): None = uniform.
    target_weights: Dict[str, float] | None = Field(default=None)
    # Learnable Tobit sigma (ADR-055): optimizer adjusts sigma during training.
    learnable_sigma: bool = Field(default=False)
    # Hurdle-NB body (D2, #99): NB dispersion θ init + whether θ is learned (loss_reg='hurdle_nb').
    loss_reg_theta_init: float | None = Field(default=None, gt=0.0)
    learnable_theta: bool = Field(default=True)
    # Hurdle-NB gate (loss_class='weighted_bce'): positive-class weight; None = plain BCE.
    # scalar (shared across sb/ns/os) OR a list of one per classification target (per-target gate)
    loss_class_pos_weight: float | list[float] | None = Field(default=None)
    # Regression-head output activation (#100): "standard" (ReLU) or "hurdle_nb" (softplus mu).
    output_distribution: str = Field(default="standard")
    # Optional emit-activation override, decoupled from output_distribution (Exp B). None => keyed
    # off output_distribution ('softplus' if hurdle_nb else 'relu'); else 'softplus'/'relu'.
    reg_activation: str | None = Field(default=None)

    # 11. Scheduled Sampling (ADR-056): close train/inference gap.
    ss_schedule: str | None = Field(default=None)
    ss_epsilon_max: float = Field(default=1.0, ge=0.0, le=1.0)
    ss_warmup_lessons: int | None = Field(default=None, ge=1)
    ss_k: float | None = Field(default=None, gt=0.0)

    # 7. Sampling & Reproducibility
    total_lessons: int = Field(..., ge=1)
    n_posterior_samples: int = Field(..., ge=1)
    # K = per-cell head draws (aleatoric) for a sampleable distribution family (ADR-067 D×K).
    # Legacy point/quantile heads keep K=1; K>1 requires an nb/zinb family (see model validator).
    n_head_samples: int = Field(default=1, ge=1)
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
    feedback_clamp_log1p: list[float] | None = Field(
        default=None,
        description=(
            "C-113: optional per-target log1p ceiling for the autoregressive "
            "feedback INPUT (one value per regression target, in target order). "
            "Bounds only the fed-back copy of the prediction, never an emitted "
            "value; None disables. See reports/preanalysis_feedback_clamp.md."
        ),
    )
    min_free_disk_gb: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "C-154: optional pre-evaluation disk-headroom budget (GiB). When set, the run aborts "
            "(fail loud) before writing predictions if free space is below this — a run writes "
            "~2.5 GB/origin-set and otherwise truncates silently (as the 6-run sweep did to "
            "S3_seed4). None disables (default-off => unchanged behaviour)."
        ),
    )
    static_channels: list[str] = Field(
        default_factory=list,
        description=(
            "ADR-060: input-only channels (e.g. coordinates) injected into the model, never "
            "predicted, never in targets, re-injected every autoregressive step. Declared HERE "
            "(not in `features`); derived in static_channels.py over the full grid. "
            "input_channels must be 3*output_channels + len(static_channels). Empty => unchanged."
        ),
    )

    # 9. Runtime Flags
    freeze_multitask_balancer: bool = Field(
        default=False,
        description=(
            "C-113 bisect: when True, the MultiTaskLoss balancer log_vars are NOT "
            "added to the optimizer (frozen at zero init = equal weighting, the "
            "pre-C-111 regime). Default False = C-111 active balancer. See "
            "reports/preanalysis_balancer_bisect.md."
        ),
    )
    rollout_horizon: int = Field(
        default=1,
        ge=1,
        description=(
            "C-113 / Axis B: number of autoregressive steps to train through "
            "(the rollout-training 'look-ahead'). Default 1 = the current one-step "
            "path (byte-identical parity); >1 enables the B1 pushforward stability "
            "term. Candidate K=12. See reports/2026-06-05_rollout_training_dossier/."
        ),
    )
    sweep: bool = Field(default=False)
    random_flips: bool = Field(default=True)
    diagnostic_visualizations: bool = Field(default=False)

    # 10. Outbound Evaluation
    evaluation_mode: str = Field(
        ...,
        description=(
            "Controls whether the posterior sample axis (S) is preserved or collapsed. "
            "'stochastic': all S samples are kept → PredictionFrame.values.shape == (N, S). "
            "'point': collapse_to_point(aggregate_method) folds S to a scalar per cell → "
            "PredictionFrame.values.shape == (N, 1) regardless of n_posterior_samples."
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
        # Checksum: input_channels == dynamic features + static channels (ADR-060). Static channels
        # are extra input-only channels carried in the volume (not in `features`).
        if self.input_channels != len(self.features) + len(self.static_channels):
            err_msg = (
                f"Checksum Law Violation: input_channels ({self.input_channels}) != "
                f"features ({len(self.features)}) + static_channels ({len(self.static_channels)})"
            )

            logger.error(err_msg)

            raise ValueError(err_msg)

        # Architectural advisory (C-105): autoregressive feedback requires
        # features == regression_targets (same variables)
        if set(self.features) != set(self.regression_targets):
            logger.warning(
                "features %s != regression_targets %s. During autoregressive "
                "inference, the model's regression output feeds back as input — "
                "a mismatch will cause a shape error at inference time.",
                self.features,
                self.regression_targets,
            )

        # Architectural invariant (C-98, extended by ADR-060/C-153): the 3 regression heads
        # (3 × output_channels) feed back as the DYNAMIC input; static channels (ADR-060) are
        # extra input-only channels. So input_channels == 3*output_channels + len(static_channels).
        n_static = len(self.static_channels)
        if self.input_channels != 3 * self.output_channels + n_static:
            err_msg = (
                f"Architectural invariant violation: input_channels ({self.input_channels}) "
                f"must equal 3*output_channels ({self.output_channels}) + len(static_channels) "
                f"({n_static}) = {3 * self.output_channels + n_static} (3 reg heads feed back as "
                f"the dynamic input; static channels are extra input-only channels, ADR-060)."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        # ADR-060 I1: a static channel is input-only — it must NEVER be a prediction target.
        bad_static = set(self.static_channels) & (
            set(self.regression_targets) | set(self.classification_targets)
        )
        if bad_static:
            err_msg = (
                f"ADR-060 I1 violation: static_channels {sorted(bad_static)} also appear in "
                f"regression/classification targets — static channels are input-only and must "
                f"never be predicted (declare them only in `static_channels`)."
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

    @field_validator("output_distribution")
    @classmethod
    def validate_output_distribution(cls, v: str) -> str:
        # ADR-067: the valid set is the LEGACY values unioned with the registered family names.
        # `family_names()` is torch-free (A-S2), so config validation stays torch-free (CRP). The
        # family/legacy disjointness invariant (C-197) is enforced in a model_validator, which
        # always runs (a field_validator is skipped for the default value).
        from views_hydranet.distributions import family_names

        valid = list(LEGACY_OUTPUT_DISTRIBUTIONS) + sorted(family_names())
        if v not in valid:
            err_msg = f"output_distribution='{v}' is not valid. Expected one of: {valid}."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return v

    @field_validator("body_mask")
    @classmethod
    def validate_body_mask(cls, v: str) -> str:
        # ADR-009/065: the point-body mask is a validated boundary contract — fail loud on any
        # value outside the closed set (a typo must never silently degrade to another mask; C-194).
        if v not in BODY_MASK_VALUES:
            err_msg = f"body_mask='{v}' is not valid. Expected one of: {list(BODY_MASK_VALUES)}."
            logger.error(err_msg)
            raise ValueError(err_msg)
        return v

    @field_validator("loss_class_pos_weight")
    @classmethod
    def validate_pos_weight_positive(cls, v):
        if v is None:
            return v
        vals = v if isinstance(v, list) else [v]
        if any(x is None or x <= 0 for x in vals):
            raise ValueError(f"loss_class_pos_weight values must be > 0; got {v}")
        return v

    @field_validator("reg_activation")
    @classmethod
    def validate_reg_activation(cls, v: str | None) -> str | None:
        if v is not None and v not in ("softplus", "relu"):
            err_msg = f"reg_activation='{v}' is not valid. Expected 'softplus', 'relu', or None."
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
    def validate_pos_weight_length(self) -> "HydraNetConfig":
        """A per-target pos_weight list needs one entry per classification target (sb/ns/os)."""
        pw = self.loss_class_pos_weight
        if isinstance(pw, list):
            n = len(self.classification_targets or [])
            if len(pw) != n:
                raise ValueError(
                    f"loss_class_pos_weight list has {len(pw)} entries but there are "
                    f"{n} classification_targets — provide one pos_weight per target."
                )
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

    @model_validator(mode="before")
    @classmethod
    def reject_retired_hurdle_knobs(cls, data):
        # ADR-065 (Epic #158 S5): hurdle_threshold + hurdle_mask_mode are RETIRED — body_mask is
        # the sole point-body mask front door. A config still setting them would otherwise be
        # SILENTLY ignored (worse than before), so reject loud (ADR-008/009) and name the fix.
        if isinstance(data, dict):
            retired = [k for k in ("hurdle_threshold", "hurdle_mask_mode") if k in data]
            if retired:
                raise ValueError(
                    f"{retired} is retired (ADR-065). Use body_mask ∈ "
                    "{'none','pos_cells','pos_timelines'}: hurdle_threshold=None → 'none'; "
                    "hurdle_threshold=0 per_step → 'pos_cells'; hurdle_mask_mode='active_window' "
                    "→ 'pos_timelines'."
                )
        return data

    @model_validator(mode="after")
    def validate_qs99_requires_tau(self) -> "HydraNetConfig":
        # The QS99 tail regularizer applies on the masked positive cells, so it is gated by the
        # point-body mask (body_mask != 'none'), not by the retired hurdle_threshold. (The tobit
        # zero-inflation contradiction is now subsumed by validate_body_mask_latent: tobit is a
        # latent loss, so body_mask='pos_*' + tobit already fails loud.)
        if self.body_mask != "none" and self.qs99_weight is not None and self.qs99_weight > 0:
            if self.qs99_tau is None:
                err_msg = (
                    f"body_mask='{self.body_mask}' with qs99_weight={self.qs99_weight} requires "
                    "'qs99_tau' but it was not provided. Add 'qs99_tau' to your config."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def validate_family_legacy_disjoint(self) -> "HydraNetConfig":
        # C-197 (Tier 2): registered distribution-family names MUST be disjoint from the legacy
        # output_distribution values. If they intersect, a legacy config value would silently route
        # to a new family via resolve_family (ADR-067 strangler-fig) — changing a proven, byte-
        # identical model with no error. A model_validator always runs (even for the default),
        # so this fires on every config load regardless of output_distribution.
        from views_hydranet.distributions import family_names

        collisions = family_names() & set(LEGACY_OUTPUT_DISTRIBUTIONS)
        if collisions:
            err_msg = (
                f"distribution-family name(s) {sorted(collisions)} collide with legacy "
                f"output_distribution values — the two sets must be disjoint (C-197)."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def validate_head_samples_family(self) -> "HydraNetConfig":
        # ADR-067 (D×K): per-cell head draws (K = n_head_samples > 1) are only meaningful for a
        # SAMPLEABLE distribution family (nb/zinb). A legacy point/quantile head has no per-cell
        # sampler, so K>1 there would be silently ignored — RAISE. Single authority for "is this
        # sampleable?" is the registry's `family_names()` (torch-free), not a hardcoded list.
        if self.n_head_samples > 1:
            from views_hydranet.distributions import family_names

            if self.output_distribution not in family_names():
                err_msg = (
                    f"n_head_samples={self.n_head_samples} (>1) requires a sampleable "
                    f"family (output_distribution in {sorted(family_names())}), but "
                    f"output_distribution='{self.output_distribution}' is a legacy head with no "
                    f"per-cell sampler. Use n_head_samples=1 with this head, or select nb/zinb."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def validate_body_mask_latent(self) -> "HydraNetConfig":
        # C-193 (ADR-008/065): a positives mask on the POINT body is a silent no-op under a latent
        # likelihood loss — the training loop only masks `if not use_latent` (needs_latent=False),
        # and a latent body models zeros/censoring/truncation itself. Requesting pos_cells/
        # pos_timelines there is meaningless, so RAISE (replacing the old warn-once), not ignore.
        # Single authority for "is this loss latent?" is the loss class's `needs_latent` flag in
        # LOSS_REG_REGISTRY — NOT a hardcoded name list (which would wrongly forbid lognormal_nll,
        # a positive-only *point* body that REQUIRES a positives mask).
        if self.body_mask in ("pos_cells", "pos_timelines"):
            from views_hydranet.utils.utils import LOSS_REG_REGISTRY

            loss_cls = LOSS_REG_REGISTRY.get(self.loss_reg, {}).get("cls")
            if loss_cls is not None and getattr(loss_cls, "needs_latent", False):
                err_msg = (
                    f"body_mask='{self.body_mask}' masks the POINT body, but loss_reg="
                    f"'{self.loss_reg}' is a latent likelihood (needs_latent=True) that models "
                    f"zeros/censoring internally — the point mask would be silently ignored. Use "
                    f"body_mask='none' with this loss, or a non-latent point body (mse/mae/huber/"
                    f"shrinkage/lognormal_nll) with a positives mask."
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
        extra = [k for k in self.target_weights if k not in self.regression_targets]
        if extra:
            err_msg = (
                f"target_weights contains keys not in regression_targets: "
                f"{extra}. Remove them or check for typos."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def validate_per_target_sigma(self) -> "HydraNetConfig":
        if not isinstance(self.loss_reg_sigma, dict):
            return self
        if self.loss_reg != "tobit":
            err_msg = (
                f"Per-target loss_reg_sigma (dict) is only supported for "
                f"loss_reg='tobit', got loss_reg='{self.loss_reg}'."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        for v in self.loss_reg_sigma.values():
            if v <= 0:
                err_msg = (
                    f"All per-target loss_reg_sigma values must be positive, "
                    f"got {self.loss_reg_sigma}."
                )
                logger.error(err_msg)
                raise ValueError(err_msg)
        missing = [t for t in self.regression_targets if t not in self.loss_reg_sigma]
        if missing:
            err_msg = (
                f"Per-target loss_reg_sigma is missing entries for regression "
                f"targets: {missing}. All regression_targets must have a sigma."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        extra = [k for k in self.loss_reg_sigma if k not in self.regression_targets]
        if extra:
            err_msg = (
                f"Per-target loss_reg_sigma contains keys not in regression_targets: "
                f"{extra}. Remove them or check for typos."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)
        return self

    @model_validator(mode="after")
    def validate_scheduled_sampling_params(self) -> "HydraNetConfig":
        if self.ss_schedule is None:
            return self
        valid = ["linear", "inverse_sigmoid", "exponential"]
        if self.ss_schedule not in valid:
            err_msg = f"ss_schedule='{self.ss_schedule}' is not valid. Expected one of: {valid}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        if self.ss_schedule == "linear" and self.ss_warmup_lessons is None:
            err_msg = "ss_schedule='linear' requires 'ss_warmup_lessons'."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if self.ss_schedule in ("inverse_sigmoid", "exponential") and self.ss_k is None:
            err_msg = f"ss_schedule='{self.ss_schedule}' requires 'ss_k'."
            logger.error(err_msg)
            raise ValueError(err_msg)
        if self.ss_schedule == "exponential" and self.ss_k is not None and self.ss_k >= 1.0:
            err_msg = f"ss_schedule='exponential' requires ss_k < 1.0, got {self.ss_k}."
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
