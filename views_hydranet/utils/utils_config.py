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
    Strictly-typed configuration for HydraNet.
    Ensures that all required parameters are present and valid before execution.
    """
    run_type: str = Field(..., description="Partition: calibration, validation, or forecasting")
    steps: List[int] = Field(default_factory=lambda: list(range(1, 37)), description="The list of forecast steps")
    time_steps: int = Field(default=36, description="Derived: number of months to predict (len(steps))")
    
    test_samples: int = Field(..., ge=1, description="Number of posterior samples to draw")
    input_channels: int = Field(default=3, ge=1, description="Number of input feature channels")
    target_variable: TargetVariable = Field(default=TargetVariable.SB_BEST, description="The primary target head")
    targets: List[str] = Field(default_factory=list, description="List of target column names for evaluation")
    
    transform: str = Field(default="log1p", description="The numerical transformation to use (log1p, asinh, identity)")
    
    freeze_h: str = Field(default="none", description="Memory freezing strategy")
    
    # Optional metadata
    model_time_stamp: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def align_steps_and_time(cls, data: Any) -> Any:
        """Ensures time_steps is exactly the length of the steps list."""
        if isinstance(data, dict):
            if "steps" in data:
                data["time_steps"] = len(data["steps"])
        return data

    @field_validator("run_type")
    @classmethod
    def validate_run_type(cls, v: str) -> str:
        valid = ["calibration", "validation", "forecasting"]
        if v not in valid:
            raise ValueError(f"run_type must be one of {valid}")
        return v
    
    @field_validator("transform")
    @classmethod
    def validate_transform(cls, v: str) -> str:
        if v not in TRANSFORMS:
            raise ValueError(f"transform must be one of {list(TRANSFORMS.keys())}")
        return v

    class Config:
        extra = "allow" # Allow extra fields from the broader pipeline for now
