import logging
import numpy as np
from enum import Enum
from typing import List, Optional, Dict, Any, Callable, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# --- TRANSFORMATION REGISTRY ---
# Maps a transform name to a tuple of (Forward Function, Inverse Function)
# Forward: Used for training/inference input scaling.
# Inverse: Used for converting back to raw counts for the Producer Contract.
TRANSFORMS: Dict[str, Tuple[Callable, Callable]] = {
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

# Centralized Registry for Multi-Task Heads
# The order defined here corresponds to the channel index in the model output tensors.
TARGET_REGISTRY = {
    "sb": 0,
    "ns": 1,
    "os": 2
}

def get_target_index(target_name: str) -> int:
    """
    Determines the tensor channel index for a given target name.
    
    Example: 'ln_sb_best' -> 0, 'lr_ns_best' -> 1
    """
    target_name = target_name.lower()
    for key, idx in TARGET_REGISTRY.items():
        if key in target_name:
            return idx
    raise ValueError(f"Target '{target_name}' not recognized in TARGET_REGISTRY.")

class HydraNetConfig(BaseModel):
    """
    Exhaustive schema for HydraNet operations. 
    Any missing field here will trigger a loud validation error at startup.
    """
    # 1. High-Level Partitioning
    run_type: str = Field(..., description="Partition: calibration, validation, or forecasting")
    steps: List[int] = Field(..., description="List of forecast steps (e.g. range(1,37))")
    time_steps: int = Field(default=0, description="Calculated automatically from steps")
    
    # 2. Data Slicing & Scaling
    input_channels: int = Field(default=3, ge=1)
    target_variable: TargetVariable = Field(..., description="The primary target (sb, ns, os)")
    targets: List[str] = Field(default_factory=list)
    transform: str = Field(default="log1p")
    
    # 3. Training Architecture & Hyperparameters
    model: str = Field(default="HydraBNUNet06_LSTM4")
    window_dim: int = Field(default=32)
    total_hidden_channels: int = Field(default=32)
    dropout_rate: float = Field(default=0.125)
    
    # 4. Optimization
    learning_rate: float = Field(default=0.001)
    weight_decay: float = Field(default=0.1)
    batch_size: int = Field(default=3)
    scheduler: str = Field(default="WarmupDecay")
    warmup_steps: int = Field(default=100)
    
    # 5. Loss Functions
    loss_reg: str = Field(default="b")
    loss_class: str = Field(default="b")
    loss_reg_a: float = Field(default=258)
    loss_reg_c: float = Field(default=0.001)
    loss_class_gamma: float = Field(default=1.5)
    loss_class_alpha: float = Field(default=0.75)
    
    # 6. Sampling & Reproducibility
    samples: int = Field(default=300)
    test_samples: int = Field(..., ge=1)
    np_seed: int = Field(default=4)
    torch_seed: int = Field(default=4)
    
    # 7. Spatial Windowing Logic
    min_events: int = Field(default=5)
    slope_ratio: float = Field(default=0.75)
    roof_ratio: float = Field(default=0.7)
    freeze_h: str = Field(default="hl")

    # Metadata
    model_time_stamp: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def align_steps_and_time(cls, data: Any) -> Any:
        if isinstance(data, dict) and "steps" in data:
            data["time_steps"] = len(data["steps"])
        return data

    @field_validator("transform")
    @classmethod
    def validate_transform(cls, v: str) -> str:
        if v not in TRANSFORMS:
            raise ValueError(f"Transform '{v}' not supported. Available: {list(TRANSFORMS.keys())}")
        return v

    @field_validator("run_type")
    @classmethod
    def validate_run_type(cls, v: str) -> str:
        valid = ["calibration", "validation", "forecasting"]
        if v not in valid:
            raise ValueError(f"run_type must be one of {valid}")
        return v


    class Config:
        extra = "allow" # Allow extra fields from the broader pipeline for now
