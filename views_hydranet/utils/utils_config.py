import logging
from collections.abc import Callable
from enum import Enum
from typing import Any, List, Optional

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

# Centralized Registry for Multi-Task Heads
TARGET_REGISTRY = {
    "sb": 0,
    "ns": 1,
    "os": 2
}

def get_target_index(target_name: str) -> int:
    """Determines the tensor channel index for a given target name."""
    target_name = target_name.lower()
    for key, idx in TARGET_REGISTRY.items():
        if key in target_name:
            return idx
    raise ValueError(f"Target '{target_name}' not recognized in TARGET_REGISTRY.")

class HydraNetConfig(BaseModel):
    """
    The 'Minimum Strict Set' for HydraNet operations. 
    
    This schema defines the fields that HydraNet REQUIRES to function.
    Extra fields (pipeline baggage) are ALLOWED but ignored by the core logic.
    """
    # 1. High-Level Partitioning
    run_type: str = Field(..., description="Partition: calibration, validation, or forecasting")
    steps: list[int] = Field(..., description="List of forecast steps (e.g. range(1,37))")
    time_steps: int = Field(default=0, description="Calculated automatically from steps")

    # 2. Data Slicing & Scaling (The Physics)
    input_channels: int = Field(..., ge=1)
    output_channels: int = Field(default=1, ge=1, description="Channels per model head (usually 1)")
    target_variable: TargetVariable = Field(..., description="The primary target (sb, ns, os)")
    targets: list[str] = Field(default_factory=list, description="Requested targets for the outbound contract")
    classification_outputs: list[str] = Field(..., description="Semantic names for the classification heads")
    identity_cols: list[str] = Field(..., description="Non-predictive metadata columns to be stripped")
    transforms: dict[str, list[str]] = Field(..., description="Mapping of transform method to list of columns")
    
    # 3. Spatiotemporal Topology (Structural Invariants)
    height: int = Field(..., ge=1, description="Grid height")
    width: int = Field(..., ge=1, description="Grid width")
    time_col: str = Field(..., description="Temporal index name")
    id_col: str = Field(..., description="Unit index name")
    spatial_cols: list[str] = Field(..., description="[row, col] column names")
    row_offset: int = Field(..., description="Row anchor offset")
    col_offset: int = Field(..., description="Column anchor offset")
    features: list[str] = Field(..., description="Exhaustive list of input feature columns")

    # 4. Training Architecture
    model: str = Field(..., description="Architecture name")
    window_dim: int = Field(..., description="Temporal window size")
    total_hidden_channels: int = Field(..., description="Base hidden width")
    dropout_rate: float = Field(..., ge=0.0, le=1.0)
    weight_init: str = Field(default="xavier_norm", description="Weight initialization strategy")
    h_init: str = Field(default="abs_rand_exp-100", description="Hidden state initialization string")

    # 5. Optimization
    learning_rate: float = Field(..., gt=0.0)
    weight_decay: float = Field(..., ge=0.0)
    windows_per_lesson: int = Field(..., description="Accumulation steps (ADR 014)")
    scheduler: str = Field(..., description="Learning rate scheduler name")
    warmup_steps: int = Field(..., description="Scheduler warmup period")
    clip_grad_norm: bool = Field(default=True, description="Enable gradient clipping")

    # 6. Loss Functions
    loss_reg: str = Field(..., description="Regression loss type")
    loss_class: str = Field(..., description="Classification loss type")
    loss_reg_a: float = Field(...)
    loss_reg_c: float = Field(...)
    loss_class_gamma: float = Field(...)
    loss_class_alpha: float = Field(...)

    # 7. Sampling & Reproducibility
    total_lessons: int = Field(..., description="Curriculum length")
    n_posterior_samples: int = Field(..., ge=1, description="MC Dropout sample count")
    np_seed: int = Field(...)
    torch_seed: int = Field(...)

    # 8. Spatial Filtering & Curriculum Logic
    min_events: int = Field(..., description="Minimum events per window for training")
    slope_ratio: float = Field(...)
    roof_ratio: float = Field(...)
    max_ratio: float = Field(...)
    min_ratio: float = Field(...)
    freeze_h: str = Field(..., description="Hidden state reset strategy (ADR 013)")

    # 9. Evaluation & Aggregation (Downstream Compatibility)
    evalution_mode: str = Field(..., description="Mode: 'point' or 'stochastic'")
    aggregate_method: str = Field(..., description="Aggregation strategy")

    # Metadata
    model_time_stamp: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def handle_typos_and_dependencies(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if "steps" in data:
            data["time_steps"] = len(data["steps"])
        if "evaluation_mode" in data and "evalution_mode" not in data:
            data["evalution_mode"] = data["evaluation_mode"]
        return data

    @model_validator(mode="after")
    def validate_scaling_ledger(self) -> "HydraNetConfig":
        """
        The Scaling Law Checksum:
        Ensures every predictive feature is explicitly mapped to a transformation.
        """
        features_set = set(self.features)
        mapped_set = set()
        
        for method, cols in self.transforms.items():
            if method not in TRANSFORMS:
                raise ValueError(f"Scaling Law Violation: Transform method '{method}' not in registry.")
            for col in cols:
                if col in mapped_set:
                    raise ValueError(f"Scaling Law Violation: Feature '{col}' mapped multiple times.")
                mapped_set.add(col)
        
        missing = features_set - mapped_set
        if missing:
            raise ValueError(f"Scaling Law Violation: Features {missing} are missing from 'transforms' mapping.")
            
        unrecognized = mapped_set - features_set
        if unrecognized:
            raise ValueError(f"Scaling Law Violation: 'transforms' contains unknown features: {unrecognized}")
            
        return self

    @field_validator("run_type")
    @classmethod
    def validate_run_type(cls, v: str) -> str:
        valid = ["calibration", "validation", "forecasting", "testing"]
        if v not in valid:
            raise ValueError(f"run_type must be one of {valid}")
        return v

    @field_validator("evalution_mode")
    @classmethod
    def validate_eval_mode(cls, v: str) -> str:
        valid = ["point", "stochastic"]
        if v == "stocastic": return "stochastic"
        if v not in valid:
            raise ValueError(f"evalution_mode must be one of {valid}")
        return v

    @field_validator("aggregate_method")
    @classmethod
    def validate_agg_method(cls, v: str) -> str:
        mapper = {"mean": "geometric_mean", "median": "median", "max_aposteriori": "median"}
        v = mapper.get(v, v)
        valid_options = ["arithmetic_mean", "geometric_mean", "median"]
        if v not in valid_options:
            raise ValueError(f"aggregate_method must be one of {valid_options}")
        return v

    class Config:
        extra = "allow" # Tolerant Handshake: Accept pipeline baggage but keep our domain strict.