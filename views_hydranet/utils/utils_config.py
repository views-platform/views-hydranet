import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# --- TRANSFORMATION REGISTRY ---
TRANSFORMS: dict[str, tuple[Callable, Callable]] = {
    "log1p": (np.log1p, np.expm1),
    "asinh": (np.arcsinh, np.sinh),
    "identity": (lambda x: x, lambda x: x)
}

class TargetVariable(str, Enum):
    SB = "sb"
    NS = "ns"
    OS = "os"
    SB_BEST = "sb_best"
    NS_BEST = "ns_best"
    OS_BEST = "os_best"
    LR_SB_BEST = "lr_sb_best"
    LR_NS_BEST = "lr_ns_best"
    LR_OS_BEST = "lr_os_best"

class HydraNetConfig(BaseModel):
    """
    The Exhaustive 'Minimum Strict Set' for HydraNet operations.
    Matches the pipeline configuration 1-to-1.
    """
    # 1. High-Level Partitioning
    run_type: str = Field(..., description="Partition: calibration, validation, forecasting, or testing")
    steps: list[int] = Field(..., description="List of forecast steps")
    time_steps: int = Field(..., description="Checksum for 'steps'")

    # 2. Data Slicing & Scaling (The Physics)
    input_channels: int = Field(..., ge=1, description="Checksum for 'features'")
    output_channels: int = Field(..., ge=1, description="Channels per model head")
    target_variable: TargetVariable = Field(..., description="The primary target")
    targets: list[str] = Field(default_factory=list)
    classification_outputs: list[str] = Field(..., description="Semantic names for model heads")
    identity_cols: list[str] = Field(..., description="Columns to be excluded from features")
    features: list[str] = Field(..., description="Exhaustive list of predictive signals")

    # The Root Scaling Field (Matches user config 1-to-1)
    transform: Dict[str, List[str]] = Field(..., description="Mapping of method to columns")

    # 3. Spatiotemporal Topology
    height: int = Field(..., ge=1)
    width: int = Field(..., ge=1)
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
    h_init: str = Field(...)

    # 5. Optimization
    learning_rate: float = Field(..., gt=0.0)
    weight_decay: float = Field(..., ge=0.0)
    windows_per_lesson: int = Field(..., ge=1)
    scheduler: str = Field(...)
    warmup_steps: int = Field(..., ge=1)
    clip_grad_norm: bool = Field(...)

    # 6. Loss Functions
    loss_reg: str = Field(...)
    loss_class: str = Field(...)
    loss_reg_a: float = Field(...)
    loss_reg_c: float = Field(...)
    loss_class_gamma: float = Field(...)
    loss_class_alpha: float = Field(...)

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
    evalution_mode: str = Field(...)
    aggregate_method: str = Field(...)

    # Metadata
    model_time_stamp: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def handle_typos(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "evaluation_mode" in data and "evalution_mode" not in data:
                data["evalution_mode"] = data["evaluation_mode"]
        return data

    @model_validator(mode="after")
    def validate_laws(self) -> "HydraNetConfig":
        """The Checksum and Scaling Laws."""
        # Checksum: input_channels
        if self.input_channels != len(self.features):
            raise ValueError(f"Checksum Law Violation: input_channels ({self.input_channels}) != features ({len(self.features)})")

        # Checksum: time_steps
        if self.time_steps != len(self.steps):
            raise ValueError(f"Checksum Law Violation: time_steps ({self.time_steps}) != steps ({len(self.steps)})")

        # Scaling Law: All features must be in the 'transform' dictionary
        features_set = set(self.features)
        mapped_set = set()
        for method, cols in self.transform.items():
            if method not in TRANSFORMS:
                raise ValueError(f"Scaling Law Violation: Unknown method '{method}'")
            for col in cols:
                mapped_set.add(col)

        missing = features_set - mapped_set
        if missing:
            raise ValueError(f"Scaling Law Violation: Features {missing} are not assigned a transform in the 'transform' dict.")

        return self

    @field_validator("run_type")
    @classmethod
    def validate_run_type(cls, v: str) -> str:
        valid = ["calibration", "validation", "forecasting", "testing"]
        if v not in valid: raise ValueError(f"run_type must be in {valid}")
        return v

    @field_validator("evalution_mode")
    @classmethod
    def validate_eval_mode(cls, v: str) -> str:
        if v == "stocastic": return "stochastic"
        if v not in ["point", "stochastic"]: raise ValueError("evaluation_mode must be 'point' or 'stochastic'")
        return v

    @field_validator("aggregate_method")
    @classmethod
    def validate_agg_method(cls, v: str) -> str:
        mapper = {"mean": "geometric_mean", "median": "median", "max_aposteriori": "median"}
        v = mapper.get(v, v)
        if v not in ["arithmetic_mean", "geometric_mean", "median"]: raise ValueError("Invalid aggregate_method")
        return v

    class Config:
        extra = "allow" # Tolerant Handshake
